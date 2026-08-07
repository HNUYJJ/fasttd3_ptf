"""P0 独立冻结 evaluator —— schema v2.1。

契约冻结于 docs/experiments/evaluator_schema_v2_prereg_20260806.md
与 docs/experiments/evaluator_v21_hardening_prereg_20260806.md。

与 v1（scripts/p0_evaluator.py）的差别有两层：

**(a) 任务语义层**（v2 引入）：

    v1   if terminated: success = True     → terminated_success
    v2   环境事实（terminated/truncated）与任务语义（task_success）分离，
         语义一律由 fasttd3_ptf.evaluation.task_metrics 的 registry 判定。

**(b) 安全能力层**（v2.1 恢复）：v2 初版丢掉了 v1 的 checkpoint 身份校验
与拒绝覆盖，导致喂错 checkpoint/env 仍会产出外观合法的 JSON。
v2.1 恢复二者并加了 `panel_digest` —— 不同面板产出的数字不可比，
而此前无法从输出中判断两份结果是否用了同一面板。

**数值路径必须与 v1 逐位相同**：相同的面板常量、相同的双播种、相同的步进循环、
相同的 break 条件。T4 集成测试对同一 checkpoint 跑两条路径并逐 episode 比对
return / progress_max_dx / episode_length / reset seed —— 这条测试防的是
"改口径导致数字变化"被误当成"修好了 bug"。

结构性 source-free：只加载 actor + 冻结 obs normalizer，从不构建
bank/option/MCG/admission 组件。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np  # noqa: E402
import torch  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from fasttd3_ptf.evaluation import schema_v2, task_metrics  # noqa: E402
from probe_lib import load_student  # noqa: E402

# ── 面板常量：必须与 v1 逐位相同，禁止在此处"顺手改进" ──────────────
EVAL_SEEDS = (11, 23, 37, 53)
EVAL_SEEDS_128 = (11, 23, 37, 53, 71, 89, 103, 113,
                  131, 149, 163, 179, 193, 211, 227, 241)
RANKS = tuple(range(8))
EPISODE_STEPS = 1000


def _make_env(env_name: str):
    from fasttd3_ptf.official_fasttd3_ptf.paths import ensure_humanoidbench_import_path

    ensure_humanoidbench_import_path()
    import gymnasium as gym
    from gymnasium.wrappers import TimeLimit

    import humanoid_bench  # noqa: F401

    env = gym.make(env_name)
    return TimeLimit(env, max_episode_steps=EPISODE_STEPS)


def _basketball_state(env) -> dict | None:
    """basketball 判定成败需要球心与 hoop 的距离（info 里没有该字段）。

    与 `basketball.py:143` 的判定式对齐：
    ``norm(named.data.xpos["basketball"] - named.data.site_xpos["hoop_center"]) < 0.05``。

    **访问路径**：`env.unwrapped` 是 `HumanoidEnv`，它**直接**持有 `named`；
    初版误写为 `env.unwrapped._env.named`（`HumanoidEnv` 没有 `_env` 属性），
    AttributeError 被 `except` 静默吞掉 → 恒返回 None → basketball 的
    `task_success` 永远是 `INSUFFICIENT_STATE`。S5 smoke 抓到了这个缺陷。

    异常不再静默：返回 None 的同时把原因写入 `_last_error`，
    使"提取失败"与"该任务不需要 state"可区分。
    """
    try:
        named = env.unwrapped.named.data
        dist = float(np.linalg.norm(
            named.xpos["basketball"] - named.site_xpos["hoop_center"]))
        if not np.isfinite(dist):
            _basketball_state._last_error = f"dist 非有限值: {dist}"
            return None
        return {"ball_to_hoop_dist": dist}
    except (AttributeError, KeyError, IndexError, TypeError) as exc:
        # 不静默：记录具体原因，供 smoke 与调用方诊断
        _basketball_state._last_error = f"{type(exc).__name__}: {exc}"
        return None


_basketball_state._last_error = None


@torch.no_grad()
def _run_episode_v2(env, actor, obs_norm, device, seed: int, env_name: str) -> dict:
    """单 episode。**数值部分与 v1 的 _run_episode 逐行对齐。**"""
    # 双播种(E11 语义):全局 NumPy + Gymnasium np_random。与 v1 相同。
    np.random.seed(seed)
    obs, _ = env.reset(seed=seed)
    x0 = float(env.unwrapped.data.qpos[0])
    total_return = 0.0
    max_progress = 0.0
    steps = 0
    terminated = truncated = False
    info: dict = {}
    info_history: list[dict] = []

    for _ in range(EPISODE_STEPS):
        obs_tensor = torch.as_tensor(obs, device=device, dtype=torch.float32).unsqueeze(0)
        action = actor(obs_norm(obs_tensor)).squeeze(0).cpu().numpy()
        obs, reward, terminated, truncated, info = env.step(action)
        total_return += float(reward)
        max_progress = max(max_progress, float(env.unwrapped.data.qpos[0]) - x0)
        steps += 1
        info_history.append(dict(info))
        if terminated:
            break
        if truncated:
            break

    spec = task_metrics.TASK_METRIC_REGISTRY.get(env_name)
    mj_state = None
    mj_error = None
    if spec and spec.needs_mujoco_state:
        _basketball_state._last_error = None
        mj_state = _basketball_state(env)
        if mj_state is None:
            mj_error = _basketball_state._last_error or "未知原因"
    required = spec.required_info_keys if spec else ()

    # 传整条 info_history：milestone 沿 trajectory 聚合（预注册 v21b §3）。
    # 只读最后一步会丢掉中途装上车又掉下的 package、中途穿筐的球。
    task_success, semantics, status, milestones = task_metrics.resolve_task_outcome(
        env_name=env_name, terminated=terminated, truncated=truncated,
        info_history=info_history, mj_state=mj_state,
    )
    diagnostics, unsupported = schema_v2.summarize_info(info_history, required_keys=required)

    return schema_v2.build_episode_record(
        seed=seed,
        total_return=total_return,
        progress_max_dx=max_progress,
        episode_length=steps,
        terminated=terminated,
        truncated=truncated,
        task_success=task_success,
        termination_semantics=semantics,
        metric_status=status,
        milestones=milestones,
        mujoco_state=mj_state,
        mujoco_state_error=mj_error,
        info_diagnostics=diagnostics,
        info_diagnostics_unsupported=unsupported,
    )


def run_panel_v2(checkpoint: str, env_name: str, eval_seeds=EVAL_SEEDS,
                 device: str | None = None, n_episodes: int | None = None) -> list[dict]:
    """跑冻结面板，返回 v2 episode 记录列表。供 T4 与主流程共用。"""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    # load_student → (actor, critic, obs_norm, critic_norm, global_step)；
    # 结构性 source-free：critic 与 critic_norm 在 eval 路径上一律不使用。
    actor, _critic, obs_norm, _critic_norm, _global_step = load_student(str(checkpoint), device)
    actor.eval()
    env = _make_env(env_name)
    episodes: list[dict] = []
    try:
        for eval_seed in eval_seeds:
            for rank in RANKS:
                episodes.append(
                    _run_episode_v2(env, actor, obs_norm, device, eval_seed * 1000 + rank, env_name)
                )
                if n_episodes is not None and len(episodes) >= n_episodes:
                    return episodes
    finally:
        env.close()
    return episodes


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _digest_obj(obj) -> str:
    """对任意可 JSON 化对象取稳定摘要（键排序）；不可 JSON 化时降级为 repr。"""
    try:
        blob = json.dumps(obj, sort_keys=True, default=str, ensure_ascii=False)
    except (TypeError, ValueError):
        blob = repr(obj)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def panel_digest(eval_seeds, ranks, episode_steps: int, deterministic: bool) -> str:
    """面板指纹。不同面板产出的数字不可比，而此前无法从输出中判断这一点。"""
    return _digest_obj({
        "eval_seeds": list(eval_seeds),
        "ranks": list(ranks),
        "episode_steps": episode_steps,
        "deterministic": deterministic,
    })


def atomic_write_json(out_path, payload, *, allow_overwrite: bool = False) -> None:
    """原子写 JSON：tempfile → flush → fsync → os.replace（protocol v21c §2）。

    不允许覆盖时，先用 ``os.link`` 做**原子抢占**再 ``os.replace`` 完成提交。
    为什么需要 link 这一步：单用 ``os.replace`` 无法表达 fail-if-exists——
    它会静默覆盖。而"先 ``exists()`` 再 replace"是 TOCTOU，
    两个进程可同时通过检查。``os.link`` 的 fail-if-exists 由内核保证，无竞态窗口。

    跨文件系统会 ``EXDEV``，但 tmp 与目标同目录，不会发生。
    无论成败都清理 tmp——半截的 JSON 比没有文件更危险，它看上去是合法产物。
    """
    out_path = Path(out_path)
    tmp = out_path.with_name(f"{out_path.name}.tmp.{os.getpid()}")
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
            fh.flush()
            os.fsync(fh.fileno())
        if not allow_overwrite:
            # 原子抢占：目标已存在即 FileExistsError（内核保证），无 TOCTOU 窗口。
            # 成功后 out_path 与 tmp 是同一 inode，内容已经正确。
            os.link(tmp, out_path)
        # 最终提交统一是 replace。抢占过时二者同 inode，此步等价于清理 tmp；
        # 允许覆盖时它就是那个原子替换动作。
        os.replace(tmp, out_path)
    finally:
        if tmp.exists():
            try:
                os.unlink(tmp)
            except OSError:
                pass


IDENTITY_MODES = ("formal", "debug")

#: identity manifest 的必需字段（protocol v21c §1）。缺任一 → 非零退出。
MANIFEST_REQUIRED_FIELDS = (
    "checkpoint_sha256", "env_name", "learner_seed", "global_step")

#: checkpoint 内**存在** ptf_cfg / protocol 声明时，manifest 额外必需的字段。
#: scratch checkpoint 无此类声明，故不强制——否则整条 scratch 基线进不了门。
MANIFEST_PROTOCOL_FIELD = "training_protocol_digest"

#: 参与 evaluation_semantics_digest 的文件（protocol v21c §3）。
#: 这三个文件任何一个改变都会改变 task_success / milestones 的含义，
#: 而这类改变**不会**体现在 panel_digest 上。
SEMANTICS_FILES = (
    "scripts/p0_evaluator_v2.py",
    "fasttd3_ptf/evaluation/task_metrics.py",
    "fasttd3_ptf/evaluation/schema_v2.py",
)

SOURCE_FREE_MODE = "structural: actor + frozen obs normalizer only"


def semantics_file_digests() -> dict:
    """逐文件 sha256。读**磁盘内容**而非 import 后的模块对象，
    以便同一进程内也能检测到文件被替换。"""
    out = {}
    for rel in SEMANTICS_FILES:
        p = REPO_ROOT / rel
        out[rel] = _sha256(p) if p.exists() else "MISSING"
    return out


def evaluation_semantics_digest(schema_version) -> str:
    """覆盖 schema、source-free 模式与三个语义文件内容的摘要（protocol v21c §3）。

    **正式可比性要求本 digest 相同**：``panel_digest`` 只覆盖 seeds/ranks/steps，
    两份用不同版本 ``task_metrics.py`` 产出的结果会被它判为可比，
    而语义映射一变 ``task_success`` 的含义就变了——数字长得一样也不可比。
    """
    h = hashlib.sha256()
    h.update(f"schema_version={schema_version}\n".encode())
    h.update(f"source_free_mode={SOURCE_FREE_MODE}\n".encode())
    for rel, dig in sorted(semantics_file_digests().items()):
        h.update(f"{rel}={dig}\n".encode())
    return h.hexdigest()


def load_identity_manifest(path) -> dict:
    """读 identity manifest（JSON）。格式错误即抛，不做任何默认填充。"""
    p = Path(path)
    if not p.exists():
        raise ValueError(f"identity manifest 不存在：{p}")
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"identity manifest 不是合法 JSON：{p}（{exc}）") from exc
    if not isinstance(data, dict):
        raise ValueError(f"identity manifest 顶层必须是对象：{p}")
    return data


def verify_checkpoint_identity(
    checkpoint: str,
    env_name: str,
    expect_global_step=None,
    expect_seed=None,
    expect_admission_mode=None,
    identity_mode: str = "formal",
    manifest: dict | None = None,
) -> dict:
    """核对 checkpoint 内部身份，不符即抛错。恢复自 v1（p0_evaluator.py:152-175）。

    强制项（两种模式下都执行、不可关闭）：checkpoint 内 ``args["env_name"]``
    必须等于命令行 ``--env-name``。喂错 env 会产出完全合法但语义错误的结果，
    且无法从输出中看出来。

    ``identity_mode``：

    ``formal``
        正式评估，**唯一可用于科学裁决的模式**。必须显式声明
        ``--expect-global-step`` 与 ``--expect-seed`` 并全部匹配；
        checkpoint 内缺这些字段时**同样硬失败**——无法核对 ≠ 核对通过。
    ``debug``
        冒烟 / 探查。允许不声明，但产物带 ``scientific_use_permitted: false``
        毒性标记。旧实现的问题正是没有这个区分：传任意一个 ``--expect-*``
        就算 ``identity_checked=true``，而 seed、global_step 可以全都没声明。

    返回身份摘要 dict，供写入评估产物。
    """
    if identity_mode not in IDENTITY_MODES:
        raise ValueError(f"identity_mode 必须是 {IDENTITY_MODES} 之一，收到 {identity_mode!r}")

    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    ckpt_args = state.get("args") or {}
    ptf_cfg = state.get("ptf_cfg") or {}

    errors = []
    if ckpt_args.get("env_name") != env_name:
        errors.append(
            f"checkpoint env_name={ckpt_args.get('env_name')!r} != --env-name {env_name!r}")
    if expect_global_step is not None and state.get("global_step") != expect_global_step:
        errors.append(
            f"checkpoint global_step={state.get('global_step')} != expected {expect_global_step}")
    if expect_seed is not None and ckpt_args.get("seed") != expect_seed:
        errors.append(f"checkpoint seed={ckpt_args.get('seed')} != expected {expect_seed}")
    if expect_admission_mode is not None:
        actual = ptf_cfg.get("admission_mode")
        if actual != expect_admission_mode:
            errors.append(
                f"checkpoint admission_mode={actual!r} != expected {expect_admission_mode!r}")

    explicit = [
        name for name, val in (
            ("expect_global_step", expect_global_step),
            ("expect_seed", expect_seed),
            ("expect_admission_mode", expect_admission_mode),
        ) if val is not None
    ]

    manifest_checked: list = []
    if identity_mode == "formal":
        # formal 身份**只能**来自 manifest：两套来源并存会导致
        # "看起来声明了、实际走的是弱路径"。
        if explicit:
            errors.append(
                f"formal 模式禁止使用 --expect-*（收到 {explicit}）；"
                f"身份必须由 --identity-manifest 提供。"
                f"--expect-* 仅 debug 模式可用")
        if manifest is None:
            errors.append(
                "formal 模式必须传 --identity-manifest。"
                "若只是冒烟/探查请显式传 --identity-mode debug"
                "（其产物将标记 scientific_use_permitted=false 且文件名须带 .debug.）")
        else:
            missing = [k for k in MANIFEST_REQUIRED_FIELDS if manifest.get(k) is None]
            if missing:
                errors.append(f"identity manifest 缺必需字段 {missing}")

            # checkpoint 有 ptf_cfg / protocol 声明时，protocol digest 必需。
            has_protocol = bool(ptf_cfg) or state.get("protocol") is not None
            if has_protocol and manifest.get(MANIFEST_PROTOCOL_FIELD) is None:
                errors.append(
                    f"checkpoint 含 ptf_cfg/protocol 声明，manifest 必须给 "
                    f"{MANIFEST_PROTOCOL_FIELD}")

            actual_sha = _sha256(Path(checkpoint))
            actual_protocol = _digest_obj(ptf_cfg) if has_protocol else None
            for field_name, want, got in (
                ("checkpoint_sha256", manifest.get("checkpoint_sha256"), actual_sha),
                ("env_name", manifest.get("env_name"), ckpt_args.get("env_name")),
                ("learner_seed", manifest.get("learner_seed"), ckpt_args.get("seed")),
                ("global_step", manifest.get("global_step"), state.get("global_step")),
                (MANIFEST_PROTOCOL_FIELD,
                 manifest.get(MANIFEST_PROTOCOL_FIELD), actual_protocol),
            ):
                if want is None:
                    continue
                if want != got:
                    errors.append(
                        f"manifest {field_name}={want!r} != checkpoint 实际 {got!r}")
                else:
                    manifest_checked.append(field_name)

        # 无法核对 != 核对通过：checkpoint 内缺字段时不得放行。
        if state.get("global_step") is None:
            errors.append("checkpoint 内缺 global_step，formal 模式无法核对身份")
        if ckpt_args.get("seed") is None:
            errors.append("checkpoint 内 args 缺 seed，formal 模式无法核对身份")

    if errors:
        raise ValueError("checkpoint identity mismatch: " + "; ".join(errors))

    info = {
        "global_step": state.get("global_step"),
        "learner_seed": ckpt_args.get("seed"),
        "env_name_in_checkpoint": ckpt_args.get("env_name"),
        "training_commit": ckpt_args.get("git_commit") or ckpt_args.get("commit") or "UNKNOWN",
        "args_digest": _digest_obj(ckpt_args),
        "ptf_cfg_digest": _digest_obj(ptf_cfg),
        "ptf_cfg_summary": {
            "admission_mode": ptf_cfg.get("admission_mode", "UNKNOWN"),
            "admission_replay_mode": ptf_cfg.get("admission_replay_mode", "UNKNOWN"),
            "mcg_warmup_steps": ptf_cfg.get("mcg_warmup_steps", "UNKNOWN"),
            "source_names": state.get("source_names", "UNKNOWN"),
        },
        "identity_mode": identity_mode,
        "manifest_checked_fields": sorted(manifest_checked),
        # formal 模式能走到这里说明 manifest 全部必需项已声明且匹配（不匹配已抛错）；
        # debug 模式恒 False。旧实现的 `len(explicit) >= 1` 已移除——
        # 只声明一个 admission_mode 也算"已校验"是不成立的。
        "identity_checked": identity_mode == "formal",
        # 显式毒性标记，给下游读：false 的产物不得进入任何科学裁决。
        "scientific_use_permitted": identity_mode == "formal",
        "identity_expectations": explicit,
    }
    del state
    return info


def _aggregate(episodes: list[dict]) -> dict:
    """面板聚合。

    **不再输出 success_count**——v1 的该字段读自 terminated，在 Walk 系上是
    摔倒计数（CLAUDE.md §6）。v2 改为分列三态计数，缺 adapter 的任务
    task_success_unknown 会等于 episode 数，一眼可见"没测"而不是"没成功"。
    """
    returns = [e["return"] for e in episodes]
    progress = [e["progress_max_dx"] for e in episodes]
    successes = [e["task_success"] for e in episodes]
    return {
        "return_mean": float(np.mean(returns)),
        "return_std": float(np.std(returns, ddof=1)) if len(returns) > 1 else 0.0,
        "progress_max_dx_mean": float(np.mean(progress)),
        "episode_length_mean": float(np.mean([e["episode_length"] for e in episodes])),
        "terminated_count": int(sum(e["terminated"] for e in episodes)),
        "truncated_count": int(sum(e["truncated"] for e in episodes)),
        "task_success_true": int(sum(1 for s in successes if s is True)),
        "task_success_false": int(sum(1 for s in successes if s is False)),
        "task_success_unknown": int(sum(1 for s in successes if s is None)),
        "metric_status_counts": {
            st: int(sum(1 for e in episodes if e["metric_status"] == st))
            for st in sorted({e["metric_status"] for e in episodes})
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--env-name", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--eval-seeds", default=None,
                        help="逗号分隔，或 'panel128'；前 4 个必须为 (11,23,37,53)")
    # 身份声明（恢复自 v1）：正式评估必须显式声明预期身份，
    # checkpoint 与声明不符即拒绝——防止喂错 checkpoint 静默产出合法外观的 JSON。
    parser.add_argument("--expect-global-step", type=int, default=None)
    parser.add_argument("--expect-seed", type=int, default=None)
    parser.add_argument("--expect-admission-mode", default=None,
                        help="lease 分支=all，abstain 分支=none")
    parser.add_argument("--allow-overwrite", action="store_true",
                        help="显式允许覆盖已存在的评估产物（默认硬拒绝）")
    parser.add_argument("--identity-manifest", default=None,
                        help="JSON identity manifest；formal 模式必需。须含 "
                             "checkpoint_sha256/env_name/learner_seed/global_step，"
                             "checkpoint 有 ptf_cfg 时还需 training_protocol_digest")
    parser.add_argument("--identity-mode", choices=IDENTITY_MODES, default="formal",
                        help="formal（默认）要求 --identity-manifest 且禁用 "
                             "--expect-* 与 --allow-overwrite；debug 允许省略身份声明，"
                             "但产物标记 scientific_use_permitted=false 且文件名须带 .debug.")
    args = parser.parse_args()

    if args.eval_seeds is None:
        eval_seeds = EVAL_SEEDS
    elif args.eval_seeds == "panel128":
        eval_seeds = EVAL_SEEDS_128
    else:
        eval_seeds = tuple(int(x) for x in args.eval_seeds.split(","))
    if eval_seeds[:len(EVAL_SEEDS)] != EVAL_SEEDS:
        raise SystemExit(
            f"面板前 4 个 eval seed 必须为 {EVAL_SEEDS} 以保持向后兼容，"
            f"收到 {eval_seeds[:len(EVAL_SEEDS)]}"
        )

    # ── 防覆盖 + 文件名护栏：都在跑任何 episode 之前，避免白跑几十分钟 ──
    out_path = Path(args.out)

    # formal 模式**无法**放行覆盖（protocol v21c §2）：
    # 不是"默认拒绝可放行"，是"没有放行这个选项"。
    formal = args.identity_mode == "formal"
    effective_overwrite = args.allow_overwrite and not formal
    if formal and args.allow_overwrite:
        raise SystemExit(
            "formal 模式禁止 --allow-overwrite。正式评估产物一旦存在即不可覆盖；"
            "若确需重算请改路径，或用 --identity-mode debug 走探查通道")
    if out_path.exists() and not effective_overwrite:
        raise SystemExit(
            f"拒绝覆盖已存在的评估产物：{out_path}\n"
            f"若确需覆盖请显式传 --allow-overwrite（仅 debug 模式有效）")

    # debug 产物的文件名必须带 .debug. 段（protocol v21c §1.1）。
    # scientific_use_permitted 字段要打开文件才看得到，而批量分析脚本按 glob
    # 收文件——文件名标记是唯一在**收集阶段**就能生效的护栏。
    if not formal and ".debug." not in out_path.name:
        raise SystemExit(
            f"debug 模式的输出文件名必须含 '.debug.' 段：{out_path.name}\n"
            f"例如 {out_path.stem}.debug{out_path.suffix}")

    # ── 身份校验：不符即拒绝，不产出任何字节 ─────────────────────────
    manifest = load_identity_manifest(args.identity_manifest) if args.identity_manifest else None
    identity = verify_checkpoint_identity(
        args.checkpoint, args.env_name,
        args.expect_global_step, args.expect_seed, args.expect_admission_mode,
        identity_mode=args.identity_mode, manifest=manifest)
    if not identity["scientific_use_permitted"]:
        print(
            "[WARN] identity-mode=debug：产物标记 scientific_use_permitted=false，"
            "不得用于任何科学裁决。正式评估请用默认的 formal 模式。",
            file=sys.stderr)

    if args.env_name not in task_metrics.TASK_METRIC_REGISTRY:
        print(
            f"[WARN] {args.env_name} 未在 TASK_METRIC_REGISTRY 中注册："
            f"task_success 将全部为 null（UNREGISTERED）。这是 fail-closed 行为，"
            f"不是错误——但该 checkpoint 无法用于任何 milestone 判据。",
            file=sys.stderr,
        )

    episodes = run_panel_v2(args.checkpoint, args.env_name, eval_seeds, args.device)

    ckpt_path = Path(args.checkpoint)
    payload = {
        # **必须**取自 schema_v2.SCHEMA_VERSION，不得硬编码：
        # require_comparable 用的是这个顶层字段，硬编码会导致 schema 升级后
        # 顶层不跟着变 → 两份不同 schema 的结果被判为可比。
        # formal pipeline smoke 抓到过顶层 "2.1" 而 episode 为 2.2（见 d577e6e）。
        "schema_version": schema_v2.SCHEMA_VERSION,
        "protocol": {
            "eval_seeds": list(eval_seeds),
            "ranks": list(RANKS),
            "episode_steps": EPISODE_STEPS,
            "deterministic": True,
            "source_free": "structural: actor + frozen obs normalizer only",
        },
        "checkpoint": {
            "path": str(ckpt_path),
            "sha256": _sha256(ckpt_path),
            **{k: v for k, v in identity.items()
               if k not in ("identity_checked", "identity_expectations",
                            "identity_mode", "scientific_use_permitted")},
        },
        "identity_mode": identity["identity_mode"],
        "identity_checked": identity["identity_checked"],
        "scientific_use_permitted": identity["scientific_use_permitted"],
        "identity_expectations": identity["identity_expectations"],
        # panel_digest 的语义**收窄且固定**为仅 seeds/ranks/episode_steps/deterministic
        # （protocol v21c §3），不再承担任何其他含义。
        "panel_digest": panel_digest(eval_seeds, RANKS, EPISODE_STEPS, True),
        # 语义指纹：schema + source-free 模式 + 三个语义文件的内容摘要。
        # 正式可比性要求它相同——语义映射一变 task_success 的含义就变了。
        "evaluation_semantics_digest": evaluation_semantics_digest(schema_v2.SCHEMA_VERSION),
        "evaluation_semantics_files": semantics_file_digests(),
        "source_free_mode": SOURCE_FREE_MODE,
        "overwrote_existing": bool(out_path.exists() and effective_overwrite),
        "env_name": args.env_name,
        "task_metrics_registered": args.env_name in task_metrics.TASK_METRIC_REGISTRY,
        "task_metrics_source": (
            task_metrics.TASK_METRIC_REGISTRY[args.env_name].source
            if args.env_name in task_metrics.TASK_METRIC_REGISTRY else None
        ),
        "git_head": subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
            cwd=str(REPO_ROOT),
        ).stdout.strip(),
        "utc": datetime.now(timezone.utc).isoformat(),
        "episodes": episodes,
        "aggregate": _aggregate(episodes),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(out_path, payload, allow_overwrite=effective_overwrite)
    print(f"wrote {out_path}")
    print(json.dumps(payload["aggregate"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
