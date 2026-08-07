"""Inventory v2.1 的身份模型（纯函数，不依赖 torch / mujoco）。

契约冻结于 docs/experiments/checkpoint_inventory_v21_prereg_20260807.md。

每个函数对应 v2 犯过的一个真实错误：

  parse_filename        v2 的 env 段正则 ``[^_]+`` 吃不下 ``h1hand-balance_hard-v0``，
                        6/20 sentinel ``fname_parsed=False`` 却全部 ``ELIGIBLE``（fail-open）
  effective_endpoint    v2 用 total_timesteps 判 completion，忽略 run_stop_step
  canonical_steps       v2 让 total=13 的 run 列出 10k/20k/50k/100k
  digest 三分           v2 要求配对时 sha256(ptf_cfg) 相同 —— 那会拒绝所有真实对照
  resolve_execution     v2 的 exp_name#seed 表达不了"同一 CLI 的两次独立执行"
"""

from __future__ import annotations

import hashlib
import json
import re

# ── eligibility / 状态取值（预注册 §3 / §8）────────────────────────
ELIGIBLE = "ELIGIBLE"
EXCLUDED_UNPARSEABLE_NAME = "EXCLUDED_UNPARSEABLE_NAME"
EXCLUDED_IDENTITY_MISMATCH = "EXCLUDED_IDENTITY_MISMATCH"
EXCLUDED_UNREADABLE = "EXCLUDED_UNREADABLE"
AMBIGUOUS_EXECUTION = "AMBIGUOUS_EXECUTION"
FORMAL_ALIAS_INTEGRITY_FAILURE = "FORMAL_ALIAS_INTEGRITY_FAILURE"
INVALID_ENDPOINT_CONFIG = "INVALID_ENDPOINT_CONFIG"
EXACT_ALIAS = "EXACT_ALIAS"

UNKNOWN_ROLE = "UNKNOWN_ROLE"
UNKNOWN_EXECUTION_ROLE = "UNKNOWN_EXECUTION_ROLE"
UNKNOWN_COMPLETION = "UNKNOWN_COMPLETION"
NO_PROTOCOL = "NO_PROTOCOL"

EXEC_FORMAL = "FORMAL"
EXEC_DUPLICATE = "REPEATABILITY_DUPLICATE"

# ── 文件名解析（预注册 §3）───────────────────────────────────────
#
# 分隔符是**双下划线**；env 内只含单下划线（h1hand-balance_hard-v0）。
# 故 env 段用非贪婪 `.+?` 匹配到第一个 `__`，而不是 v2 的 `[^_]+`——
# 后者在遇到 balance_hard / bookshelf_simple 时直接失配，
# 而失配又被 v2 当成"无冲突"，是 fail-open。
FNAME_RE = re.compile(
    r"^(?P<env>.+?)__(?P<run>.+)__(?P<seed>\d+)_(?P<step>\d+|final)\.pt$")


def parse_filename(name: str) -> dict:
    """解析 ``{env}__{run}__{seed}_{step}.pt``。

    **解析失败不是"无冲突"**：调用方必须据 ``fname_parsed=False`` 判
    ``EXCLUDED_UNPARSEABLE_NAME``，不得放行。无法核对 ≠ 核对通过。
    """
    m = FNAME_RE.match(name)
    if not m:
        return {"fname_parsed": False, "fname_env": None, "fname_run": None,
                "fname_seed": None, "fname_step": None, "fname_is_final": False}
    step = m.group("step")
    return {
        "fname_parsed": True,
        "fname_env": m.group("env"),
        "fname_run": m.group("run"),
        "fname_seed": int(m.group("seed")),
        "fname_step": None if step == "final" else int(step),
        "fname_is_final": step == "final",
    }


# ── endpoint 与 canonical（预注册 §4）────────────────────────────

FIXED_CANONICAL_STEPS = (10000, 20000, 50000, 100000)
BOOTSTRAP_END_KEY = "mcg_warmup_steps"


def effective_endpoint(args: dict, ptf_cfg: dict) -> dict:
    """实际训练退出点。

    已核实 ``train_ptf.py`` 的原文语义::

        :339   run_stop_step 独立控制训练退出——total_timesteps 保持不变
               以维持 LR 余弦日程
        :2280  run_stop_step = int(ptf_cfg.get("run_stop_step") or args.total_timesteps)
        :2286  if run_stop_step <= 0 or run_stop_step > args.total_timesteps: → 报错

    **禁止 ``min(total, run_stop)``**：代码本身把 ``run_stop > total`` 当错误，
    用 min 会把非法配置静默"修好"。inventory 如实记录，不替训练脚本纠错。
    """
    args = args or {}
    ptf_cfg = ptf_cfg or {}
    total = args.get("total_timesteps")
    run_stop = ptf_cfg.get("run_stop_step")

    if run_stop is not None:
        if not isinstance(run_stop, int) or run_stop <= 0 or (
                isinstance(total, int) and run_stop > total):
            return {"endpoint": None, "source": INVALID_ENDPOINT_CONFIG,
                    "run_stop_step": run_stop, "total_timesteps": total,
                    "reason": "run_stop_step 必须在 (0, total_timesteps]"}
        return {"endpoint": run_stop, "source": "run_stop_step",
                "run_stop_step": run_stop, "total_timesteps": total}

    if isinstance(total, int) and total > 0:
        return {"endpoint": total, "source": "total_timesteps",
                "run_stop_step": None, "total_timesteps": total}
    return {"endpoint": None, "source": UNKNOWN_COMPLETION,
            "run_stop_step": None, "total_timesteps": total}


def canonical_steps(args: dict, ptf_cfg: dict, registry_points=()) -> dict:
    """协议感知 canonical，**只保留 <= effective_endpoint 的点**。

    v2 的 bug：``total_timesteps=13`` 的 diagnostic run 列出
    ``[13, 30, 10000, 20000, 50000, 100000]``。超出 endpoint 的点该 run
    根本没跑到，列出来会让下游以为"该评估却没评"。
    """
    ep = effective_endpoint(args, ptf_cfg)
    endpoint = ep["endpoint"]

    candidates: dict[int, str] = {}
    for s in FIXED_CANONICAL_STEPS:
        candidates[s] = "fixed"
    boot = (ptf_cfg or {}).get(BOOTSTRAP_END_KEY)
    if isinstance(boot, int) and boot > 0:
        candidates.setdefault(boot, f"bootstrap_end<-ptf_cfg[{BOOTSTRAP_END_KEY}]")
    for s in registry_points or ():
        if isinstance(s, int) and s > 0:
            candidates.setdefault(s, "run_card_registry")
    if isinstance(endpoint, int):
        candidates.setdefault(endpoint, f"effective_endpoint<-{ep['source']}")

    if endpoint is None:
        # endpoint 不可知 → 不裁剪也不声称 canonical，全部记为待定
        return {"steps": [], "out_of_scope": sorted(candidates),
                "provenance": candidates, "endpoint": None,
                "endpoint_source": ep["source"]}

    in_scope = sorted(s for s in candidates if s <= endpoint)
    out = sorted(s for s in candidates if s > endpoint)
    return {"steps": in_scope, "out_of_scope": out,
            "provenance": {str(k): v for k, v in sorted(candidates.items())},
            "endpoint": endpoint, "endpoint_source": ep["source"]}


def completion_status(observed_end, endpoint) -> str:
    """run 层的完成状态（预注册 §4.2）。"""
    if endpoint is None or observed_end is None:
        return UNKNOWN_COMPLETION
    return "COMPLETED" if observed_end >= endpoint else "TRUNCATED_RUN"


# ── digest 三分（预注册 §5）──────────────────────────────────────
#
# v2 把 sha256(ptf_cfg) 当 training_protocol_digest 并要求配对时相同——
# 那会拒绝所有真实的 matched comparison，因为 scratch / continuous /
# hard-exit 的 ptf_cfg **本来就必须不同**，那正是 treatment。

#: 两臂之间本就应当一致的 nuisance 配置（预注册 §5.1 冻结）。
PAIRING_INVARIANT_ARGS = (
    "total_timesteps", "num_envs", "batch_size", "buffer_size", "gamma", "tau",
    "actor_learning_rate", "critic_learning_rate", "policy_noise", "noise_clip",
    "num_updates",
)
PAIRING_INVARIANT_PTF = ("anchor_dir", "anchor_step", "anchor_provenance_groups")

MISSING = "MISSING"


def _digest(obj) -> str:
    try:
        blob = json.dumps(obj, sort_keys=True, default=str, ensure_ascii=False)
    except (TypeError, ValueError):
        blob = repr(obj)
    return hashlib.sha256(blob.encode()).hexdigest()


def compute_digests(args: dict, ptf_cfg: dict, source_names=None,
                    invariant_args=None, invariant_ptf=None) -> dict:
    """三个 digest 各司其职。

    ``ptf_cfg_digest``          整个 ptf_cfg —— 身份，**不要求**配对时相同
    ``treatment_digest``        不在白名单里的 ptf_cfg 字段 + source_names —— 允许不同
    ``pairing_invariant_digest``白名单字段 —— 配对时**必须**相同

    缺失字段记 ``MISSING`` 并参与 digest，使"缺"与"有"可区分。
    """
    args = args or {}
    ptf_cfg = ptf_cfg or {}
    inv_args = tuple(invariant_args or PAIRING_INVARIANT_ARGS)
    inv_ptf = tuple(invariant_ptf or PAIRING_INVARIANT_PTF)

    invariant = {f"args.{k}": args.get(k, MISSING) for k in inv_args}
    invariant.update({f"ptf.{k}": ptf_cfg.get(k, MISSING) for k in inv_ptf})

    treatment = {k: v for k, v in ptf_cfg.items() if k not in inv_ptf}
    treatment["__source_names__"] = list(source_names) if source_names else source_names

    return {
        "ptf_cfg_digest": _digest(ptf_cfg) if ptf_cfg else NO_PROTOCOL,
        "treatment_digest": _digest(treatment),
        "pairing_invariant_digest": _digest(invariant),
        "pairing_invariant_fields": sorted(invariant),
    }


# ── run card registry 匹配（预注册 §6）───────────────────────────

def match_run_card(entries: list, exp_name: str, path: str) -> dict | None:
    """按 registry 匹配角色。带 ``path_prefix`` 的规则优先。

    **不做任何词义猜测**：exp_name 里有 `scr` 不代表 scratch。
    匹配不上就返回 None，由调用方记 UNKNOWN_ROLE。
    """
    if not exp_name:
        return None
    norm = str(path).replace("\\", "/")

    def _fill(tpl: str, groups: dict) -> str:
        out = tpl
        for k, v in groups.items():
            if v is not None:
                out = out.replace("{" + k + "}", str(v))
        return out

    hits = []
    for e in entries or []:
        m = (e.get("match") or {})
        rx = m.get("exp_name_regex")
        if not rx:
            continue
        mm = re.match(rx, exp_name)
        if not mm:
            continue
        groups = mm.groupdict()
        prefix = m.get("path_prefix")
        if prefix:
            want = _fill(prefix, groups)
            if want not in norm:
                continue
            hits.append((1, e, groups))          # 带路径限定，优先级高
        else:
            hits.append((0, e, groups))

    if not hits:
        return None
    hits.sort(key=lambda t: -t[0])
    _prio, entry, groups = hits[0]
    return {
        "run_card_id": entry["id"],
        "experiment_role": _fill(entry["experiment_role"], groups),
        "execution_role": entry.get("execution_role", UNKNOWN_EXECUTION_ROLE),
        "match_group": _fill(entry["match_group"], groups),
        "alias_of_formal_path": bool(entry.get("alias_of_formal_path")),
        "counts_as_new_learner_replication":
            entry.get("counts_as_new_learner_replication", True),
        "registry_canonical_points": entry.get("canonical_points") or [],
        "expected_endpoint": entry.get("expected_endpoint"),
    }


# ── 三层身份（预注册 §2）─────────────────────────────────────────

def build_identity(env_name, exp_name, learner_seed, execution_role) -> dict:
    """run_family / execution_instance / learner_replication。

    关键区分：同一 ``learner_replication_id`` 可以有多个 ``execution_instance_id``
    （重复性 duplicate），但**不计作多个独立 learner seed**。
    v2 把二者混同，才把合法的 A/B 判成映射不唯一。
    """
    if env_name is None or exp_name is None or learner_seed is None:
        return {"run_family_id": None, "execution_instance_id": None,
                "learner_replication_id": None}
    family = f"{env_name}|{exp_name}|{learner_seed}"
    role = execution_role or UNKNOWN_EXECUTION_ROLE
    return {
        "run_family_id": family,
        "execution_instance_id": f"{family}|{role}",
        # 与 execution 分离：duplicate 不产生新的统计单位
        "learner_replication_id": family,
    }
