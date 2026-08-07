#!/usr/bin/env python3
"""Checkpoint inventory v2 —— 身份六元组 + 协议感知 canonical（P2）。

判据冻结于 docs/experiments/checkpoint_inventory_v2_prereg_20260807.md。

**默认只跑 sentinel**（约 20 个文件）。全量 deep scan 需要显式批准：
预注册 §6 规定 sentinel 通过后停下等 review，不得自行全量扫描。

用法::

    python scripts/analysis/build_checkpoint_inventory_v2.py            # sentinel
    python scripts/analysis/build_checkpoint_inventory_v2.py --full --approved-by "<who>"

数据不全 / 身份冲突未识别 / 静默猜值 / 映射不唯一 → 非零退出（预注册 §6）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

V1_MANIFEST = REPO / "docs/data/checkpoint_inventory_v1/manifest.json"
OUT_DIR = REPO / "docs/data/checkpoint_inventory_v2"

# ── UNKNOWN 取值（预注册 §1）。绝不用具体值代替。 ────────────────────
UNKNOWN_RUN_INSTANCE = "UNKNOWN_RUN_INSTANCE"
UNKNOWN_SEED = "UNKNOWN_SEED"
UNKNOWN_MECHANISM = "UNKNOWN_MECHANISM"
UNKNOWN_ROLE = "UNKNOWN_ROLE"
UNKNOWN_COMPLETION = "UNKNOWN_COMPLETION"
NO_PROTOCOL = "NO_PROTOCOL"

EXCLUDED_IDENTITY_MISMATCH = "EXCLUDED_IDENTITY_MISMATCH"
EXCLUDED_UNREADABLE = "EXCLUDED_UNREADABLE"
AMBIGUOUS_RUN_INSTANCE = "AMBIGUOUS_RUN_INSTANCE"

#: mechanism_family 真值表（预注册 §1.3，运行前冻结）。
#: 它回答"文件里存了什么"，**不回答**"这次实验想干什么"。
MECH_NO_PTF = "NO_PTF"
MECH_PTF_NULL_BANK = "PTF_NULL_BANK"
MECH_PTF_WITH_SOURCES = "PTF_WITH_SOURCES"

#: 协议感知 canonical 的固定点（预注册 §3）。
FIXED_CANONICAL_STEPS = (10000, 20000, 50000, 100000)

#: `bootstrap_end` 的来源键。已核实 `train_ptf.py:188/412/560` —— 该键存在且被消费。
BOOTSTRAP_END_KEY = "mcg_warmup_steps"

#: `hard_exit_step`：**已核实没有专用配置键**。
#: `grep -rn "hard_exit" fasttd3_ptf/` 只在 site_rules 的注释与一个编排脚本的
#: 实验名里出现，`ptf_cfg` 内无对应项。hard exit 是**实验设计概念**
#: （源在 warmup 结束后退出），其步值在语义上等于 `mcg_warmup_steps`，
#: 但那是设计层解释、不是配置层事实——判定它需要 experiment_role，
#: 而 role 只能来自 run card（当前没有）。故按预注册 §3 规则 2 记 UNKNOWN，不猜。
HARD_EXIT_KEY = None

FNAME_RE = re.compile(r"^(?P<env>[^_]+(?:-[^_]+)*)__(?P<run>.+)__(?P<seed>\d+)_(?P<step>\d+|final)\.pt$")


def parse_filename(path: str) -> dict:
    """从文件名解析 (env, run_name, seed, step_token)。解析失败返回 None 字段。"""
    m = FNAME_RE.match(Path(path).name)
    if not m:
        return {"fname_env": None, "fname_run": None,
                "fname_seed": None, "fname_step": None, "fname_parsed": False}
    step = m.group("step")
    return {
        "fname_env": m.group("env"),
        "fname_run": m.group("run"),
        "fname_seed": int(m.group("seed")),
        "fname_step": None if step == "final" else int(step),
        "fname_is_final": step == "final",
        "fname_parsed": True,
    }


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def digest_obj(obj) -> str:
    try:
        blob = json.dumps(obj, sort_keys=True, default=str, ensure_ascii=False)
    except (TypeError, ValueError):
        blob = repr(obj)
    return hashlib.sha256(blob.encode()).hexdigest()


def classify_mechanism(ptf_cfg, source_names, agent_type) -> str:
    """预注册 §1.3 冻结的真值表。"""
    if not ptf_cfg:
        return MECH_NO_PTF
    names = list(source_names) if source_names else []
    if not names or names == ["null"]:
        return MECH_PTF_NULL_BANK
    return MECH_PTF_WITH_SOURCES


def canonical_steps_for_run(args: dict, ptf_cfg: dict) -> dict:
    """逐 run 计算 canonical 步集（预注册 §3）。缺失项**跳过，不填默认值**。"""
    steps, provenance = set(FIXED_CANONICAL_STEPS), {}
    for s in FIXED_CANONICAL_STEPS:
        provenance[str(s)] = "fixed"

    boot = (ptf_cfg or {}).get(BOOTSTRAP_END_KEY)
    if isinstance(boot, int) and boot > 0:
        steps.add(boot)
        provenance[str(boot)] = f"bootstrap_end<-ptf_cfg[{BOOTSTRAP_END_KEY}]"

    total = (args or {}).get("total_timesteps")
    if isinstance(total, int) and total > 0:
        steps.add(total)
        provenance.setdefault(str(total), "configured_total_timesteps")

    return {
        "steps": sorted(steps),
        "provenance": provenance,
        # 预注册 §3 规则 2：找不到对应键即 UNKNOWN，不猜
        "hard_exit_step": "UNKNOWN_NO_DEDICATED_KEY",
    }


def read_identity(path: Path) -> dict:
    """读 checkpoint 内部身份。**只读元数据**，不构建任何模型。"""
    import torch

    try:
        state = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:  # noqa: BLE001
        return {"readable": False, "error": f"{type(exc).__name__}: {exc}"}

    args = state.get("args") or {}
    ptf_cfg = state.get("ptf_cfg") or {}
    source_names = state.get("source_names")
    exp_name = args.get("exp_name")
    seed = args.get("seed")

    out = {
        "readable": True,
        "inner_env_name": args.get("env_name"),
        "inner_seed": seed,
        "inner_global_step": state.get("global_step"),
        "exp_name": exp_name,
        "source_names": list(source_names) if source_names else source_names,
        "agent_type": state.get("agent_type"),
        "mechanism_family": classify_mechanism(ptf_cfg, source_names, state.get("agent_type")),
        "training_protocol_digest": digest_obj(ptf_cfg) if ptf_cfg else NO_PROTOCOL,
        "learner_replication_id": seed if seed is not None else UNKNOWN_SEED,
        "run_instance_id": (f"{exp_name}#{seed}"
                            if exp_name is not None and seed is not None
                            else UNKNOWN_RUN_INSTANCE),
        # experiment_role 只能来自 run card；当前仓库没有 → 一律 UNKNOWN（§1.2）
        "experiment_role": UNKNOWN_ROLE,
        "configured_total_timesteps": args.get("total_timesteps"),
        # 诊断量，不参与任何判定：run_stop_step 是训练退出（train_ptf.py:344/486），
        # 与 total_timesteps 可能不同。见结果文档对预注册 §4 的缺陷说明。
        "configured_run_stop_step": ptf_cfg.get("run_stop_step"),
        "canonical": canonical_steps_for_run(args, ptf_cfg),
    }
    del state
    return out


def check_identity_conflict(fname: dict, inner: dict) -> list:
    """文件名解析值与 checkpoint 内部值**三项逐一**比对（预注册 §2）。

    任一不符即冲突。不做修正猜测——文件名由人/脚本写，内部由训练进程写，
    不一致意味着至少一方错了，此时选哪一方都是猜。
    """
    conflicts = []
    if not fname.get("fname_parsed"):
        return conflicts
    pairs = [
        ("env_name", fname["fname_env"], inner.get("inner_env_name")),
        ("learner_seed", fname["fname_seed"], inner.get("inner_seed")),
    ]
    if not fname.get("fname_is_final"):
        pairs.append(("global_step", fname["fname_step"], inner.get("inner_global_step")))
    for field, from_name, from_ckpt in pairs:
        if from_name is None or from_ckpt is None:
            continue
        # 文件名的 env 段不带 -v0 后缀之外的差异；直接比较完整字符串
        if str(from_name) != str(from_ckpt):
            conflicts.append({"field": field, "from_filename": from_name,
                              "from_checkpoint": from_ckpt})
    return conflicts


# ══════════════════════════════════════════════════════════════════════
# Sentinel 选取（预注册 §5，规则先于运行冻结）
# ══════════════════════════════════════════════════════════════════════

def load_v1_rows() -> list:
    if not V1_MANIFEST.exists():
        raise SystemExit(f"INCOMPLETE: v1 manifest 不存在：{V1_MANIFEST}")
    return json.loads(V1_MANIFEST.read_text(encoding="utf-8"))["rows"]


def select_sentinels(rows: list) -> tuple[list, list]:
    """按预注册 §5 的规则选取。返回 (选中列表, 不可用项说明)。

    命中不足时如实记 SENTINEL_UNAVAILABLE，**不替换成别的文件凑数**。
    """
    eligible = [r for r in rows if r.get("eligibility") != "EXCLUDED"]
    by_path = sorted(eligible, key=lambda r: r["path"])
    picked, seen, unavailable = [], set(), []

    def take(pred, n, tag):
        got = 0
        for r in by_path:
            if got >= n:
                break
            if r["path"] in seen or not pred(r):
                continue
            seen.add(r["path"])
            picked.append({**r, "sentinel_reason": tag})
            got += 1
        if got < n:
            unavailable.append({"rule": tag, "wanted": n, "got": got,
                                "status": "SENTINEL_UNAVAILABLE"})

    name = lambda r: (r.get("run_name") or "")            # noqa: E731
    take(lambda r: re.search(r"scr|scratch", name(r), re.I), 3, "scratch")
    take(lambda r: re.search(r"bac|bootstrap|admission", name(r), re.I), 3, "ptf_candidate")
    take(lambda r: re.search(r"none|null", name(r), re.I), 2, "null_bank_candidate")
    for env in ("h1hand-crawl-v0", "h1hand-slide-v0", "h1hand-truck-v0"):
        take(lambda r, e=env: r.get("env_name") == e, 1, f"target:{env}")
    take(lambda r: r["path"].endswith("_final.pt"), 3, "final_file")
    take(lambda r: re.search(r"racing", name(r), re.I), 2, "racing")
    take(lambda r: re.search(r"exit", name(r), re.I), 2, "exit_named")

    # 重复 step：同一 (run_name, seed) 下同 global_step 有多个文件
    groups = defaultdict(list)
    for r in eligible:
        groups[(r.get("run_name"), r.get("learner_seed"), r.get("global_step"))].append(r)
    dup = [g for g in groups.values() if len(g) > 1]
    got = 0
    for g in sorted(dup, key=lambda g: g[0]["path"])[:2]:
        for r in g[:2]:
            if r["path"] not in seen:
                seen.add(r["path"])
                picked.append({**r, "sentinel_reason": "duplicate_step"})
        got += 1
    if got < 2:
        unavailable.append({"rule": "duplicate_step", "wanted": 2, "got": got,
                            "status": "SENTINEL_UNAVAILABLE"})
    return picked, unavailable


def make_injected_conflicts(tmpdir: Path, donor: Path) -> list:
    """构造两个**人为身份冲突**（预注册 §5 第 19/20 项）。

    放临时目录，不污染 models/。解析器不崩不算通过——必须**报错**才算。
    """
    injected = []
    m = FNAME_RE.match(donor.name)
    if not m:
        return injected
    env, run, seed, step = m.group("env"), m.group("run"), m.group("seed"), m.group("step")

    bad_seed = tmpdir / f"{env}__{run}__{int(seed) + 77}_{step}.pt"
    shutil.copy2(donor, bad_seed)
    injected.append({"path": str(bad_seed), "expect": EXCLUDED_IDENTITY_MISMATCH,
                     "sentinel_reason": "injected_seed_conflict"})

    if step != "final":
        bad_step = tmpdir / f"{env}__{run}__{seed}_{int(step) + 12345}.pt"
        shutil.copy2(donor, bad_step)
        injected.append({"path": str(bad_step), "expect": EXCLUDED_IDENTITY_MISMATCH,
                         "sentinel_reason": "injected_step_conflict"})
    return injected


def process(path_str: str, meta: dict) -> dict:
    p = Path(path_str)
    row = {"path": path_str, "sentinel_reason": meta.get("sentinel_reason")}
    if not p.exists():
        row.update({"eligibility": EXCLUDED_UNREADABLE, "error": "文件不存在"})
        return row
    row.update(parse_filename(path_str))
    row["checkpoint_id"] = sha256_file(p)
    inner = read_identity(p)
    if not inner.get("readable"):
        row.update({"eligibility": EXCLUDED_UNREADABLE, "error": inner.get("error")})
        return row
    row.update({k: v for k, v in inner.items() if k != "readable"})

    conflicts = check_identity_conflict(row, inner)
    row["identity_conflicts"] = conflicts
    row["eligibility"] = EXCLUDED_IDENTITY_MISMATCH if conflicts else "ELIGIBLE"
    return row


def resolve_final_and_duplicates(rows: list) -> tuple[dict, list]:
    """FINAL 解析成内部 global_step 并去重（预注册 §3.1）。"""
    by_run = defaultdict(list)
    for r in rows:
        if r.get("eligibility") != "ELIGIBLE":
            continue
        by_run[r.get("run_instance_id", UNKNOWN_RUN_INSTANCE)].append(r)

    notes, ambiguous = {}, []
    for run_id, group in by_run.items():
        by_step = defaultdict(list)
        for r in group:
            gs = r.get("inner_global_step")
            if gs is not None:
                by_step[gs].append(r)
        for gs, items in by_step.items():
            shas = {i["checkpoint_id"] for i in items}
            if len(items) > 1 and len(shas) == 1:
                finals = [i for i in items if i.get("fname_is_final")]
                for f in finals:
                    f["final_resolution"] = f"FINAL_DUPLICATE_OF_{gs}"
                    f["is_canonical"] = False
                notes[f"{run_id}@{gs}"] = f"{len(items)} 个文件同 sha，FINAL 已去重"
            elif len(shas) > 1:
                # 同 run 同 step 不同 sha → 映射不唯一（§1.1）
                ambiguous.append({"run_instance_id": run_id, "global_step": gs,
                                  "n_distinct_sha": len(shas),
                                  "paths": [i["path"] for i in items]})
                for i in items:
                    i["eligibility"] = AMBIGUOUS_RUN_INSTANCE
    return notes, ambiguous


def compute_run_completion(rows: list) -> dict:
    """completion_status 在 **run 层**判断（预注册 §4）。

    v1 把正常中间 checkpoint 判成 interrupted——中间点"未完成"是定义使然，
    不是异常。故该字段挂在 run 上，单个 checkpoint 只记 is_run_endpoint。
    """
    by_run = defaultdict(list)
    for r in rows:
        if r.get("eligibility") == "ELIGIBLE":
            by_run[r.get("run_instance_id", UNKNOWN_RUN_INSTANCE)].append(r)

    out = {}
    for run_id, group in by_run.items():
        steps = [r["inner_global_step"] for r in group if r.get("inner_global_step") is not None]
        if not steps:
            out[run_id] = {"completion_status": UNKNOWN_COMPLETION,
                           "reason": "无可用 global_step"}
            continue
        observed_end = max(steps)
        totals = {r.get("configured_total_timesteps") for r in group
                  if r.get("configured_total_timesteps") is not None}
        for r in group:
            r["is_run_endpoint"] = (r.get("inner_global_step") == observed_end)
        if len(totals) != 1:
            out[run_id] = {"completion_status": UNKNOWN_COMPLETION,
                           "observed_end": observed_end,
                           "reason": f"configured_total_timesteps 不唯一：{sorted(totals)}"}
            continue
        total = totals.pop()
        out[run_id] = {
            "completion_status": "COMPLETED" if observed_end >= total else "TRUNCATED_RUN",
            "observed_end": observed_end,
            "configured_total_timesteps": total,
            # 诊断量：不参与判定，见结果文档对预注册 §4 的缺陷说明
            "configured_run_stop_step": next(
                (r.get("configured_run_stop_step") for r in group
                 if r.get("configured_run_stop_step") is not None), None),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--full", action="store_true",
                    help="全量 deep scan；预注册 §6 要求单独批准")
    ap.add_argument("--approved-by", default=None)
    ap.add_argument("--out", default=str(OUT_DIR / "sentinel.json"))
    args = ap.parse_args()

    if args.full and not args.approved_by:
        raise SystemExit(
            "全量 deep scan 需要 --approved-by。预注册 §6：sentinel 通过后停下等 review，"
            "不得自行全量扫描")

    rows_v1 = load_v1_rows()
    if args.full:
        targets = [{"path": r["path"], "sentinel_reason": "FULL_SCAN"}
                   for r in rows_v1 if r.get("eligibility") != "EXCLUDED"]
        injected = []
        tmpdir = None
    else:
        picked, unavailable = select_sentinels(rows_v1)
        targets = picked
        tmpdir = Path(tempfile.mkdtemp(prefix="inv2_injected_"))
        donor = next((Path(REPO / t["path"]) for t in targets
                      if (REPO / t["path"]).exists() and not t["path"].endswith("_final.pt")),
                     None)
        injected = make_injected_conflicts(tmpdir, donor) if donor else []

    print(f"处理 {len(targets)} 个 sentinel + {len(injected)} 个注入冲突", flush=True)

    processed = []
    for t in targets:
        p = t["path"] if Path(t["path"]).is_absolute() else str(REPO / t["path"])
        processed.append(process(p, t))
    injected_rows = [process(t["path"], t) for t in injected]

    # ── 注入的冲突必须被识别（预注册 §5 第 19/20 项）─────────────────
    injection_failures = []
    for t, r in zip(injected, injected_rows):
        if r.get("eligibility") != t["expect"]:
            injection_failures.append({
                "path": r["path"], "expected": t["expect"],
                "actual": r.get("eligibility"),
                "note": "注入的身份冲突未被识别 —— 解析器不崩不算通过"})

    dedup_notes, ambiguous = resolve_final_and_duplicates(processed)
    run_completion = compute_run_completion(processed)

    # ── 静默猜值检查：experiment_role 必须全是 UNKNOWN（§1.2 已核实无 run card）
    guessed = [r["path"] for r in processed
               if r.get("experiment_role") not in (None, UNKNOWN_ROLE)]

    n_mismatch = sum(1 for r in processed if r.get("eligibility") == EXCLUDED_IDENTITY_MISMATCH)
    n_eligible = sum(1 for r in processed if r.get("eligibility") == "ELIGIBLE")

    failures = []
    if injection_failures:
        failures.append(f"{len(injection_failures)} 个注入冲突未被识别")
    if ambiguous:
        failures.append(f"{len(ambiguous)} 组 AMBIGUOUS_RUN_INSTANCE（映射不唯一）")
    if guessed:
        failures.append(f"{len(guessed)} 个 experiment_role 被静默猜值")
    if not processed:
        failures.append("INCOMPLETE：无任何 sentinel 被处理")

    payload = {
        "prereg": "docs/experiments/checkpoint_inventory_v2_prereg_20260807.md",
        "mode": "full" if args.full else "sentinel",
        "verdict": "SENTINEL_PASS" if not failures else "SENTINEL_FAILED",
        "failures": failures,
        "totals": {
            "n_processed": len(processed),
            "n_eligible": n_eligible,
            "n_identity_mismatch": n_mismatch,
            "n_injected": len(injected_rows),
        },
        "sentinel_unavailable": unavailable if not args.full else [],
        "injection_checks": injection_failures or "全部注入冲突均被正确识别",
        "ambiguous_run_instances": ambiguous,
        "final_dedup_notes": dedup_notes,
        "run_completion": run_completion,
        "canonical_policy": {
            "fixed_steps": list(FIXED_CANONICAL_STEPS),
            "bootstrap_end_key": BOOTSTRAP_END_KEY,
            "hard_exit_step": "UNKNOWN_NO_DEDICATED_KEY",
            "hard_exit_note": (
                "已核实 ptf_cfg 内无 hard_exit 专用键。hard exit 是实验设计概念"
                "（源在 warmup 结束后退出），判定它需要 experiment_role，"
                "而 role 只能来自 run card（当前没有）。按预注册 §3 规则 2 记 UNKNOWN。"),
        },
        "rows": processed,
        "injected_rows": injected_rows,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, out_path)

    if tmpdir and tmpdir.exists():
        shutil.rmtree(tmpdir, ignore_errors=True)

    print(f"\nVERDICT: {payload['verdict']}")
    print(f"  处理 {len(processed)} | ELIGIBLE {n_eligible} | 身份冲突 {n_mismatch}")
    print(f"  注入冲突 {len(injected_rows)} 个，识别情况："
          f"{'全部正确' if not injection_failures else injection_failures}")
    if unavailable if not args.full else False:
        print(f"  SENTINEL_UNAVAILABLE: {unavailable}")
    for f in failures:
        print(f"  ! {f}")
    print(f"wrote {out_path.relative_to(REPO)}")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
