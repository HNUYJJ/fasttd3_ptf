#!/usr/bin/env python3
"""S1–S6 + S10 真实 runtime smoke（evaluator v2.1c）。

判据冻结于：
  S1–S6   docs/experiments/evaluator_v21b_prereg_20260807.md §5
  S10     docs/experiments/evaluator_v21c_reverification_protocol_20260807.md §6

S1–S6 的检查函数直接复用 v21b 脚本，**不重写**——重写会给"顺手改判据"留口子。
本脚本只新增 S10 的**正常分支**（registry 冻结 reducer 在真实 runtime 上的输出）；
S10 的 fail-closed 分支、S7–S9、S11 由单元测试覆盖（需要构造损坏的 manifest
与被篡改的文件，在真实 MuJoCo 里做既慢又不可控）。

**smoke 数值不得用于任何科学判断。**
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "scripts/analysis"))

import p0_evaluator_v2 as ev  # noqa: E402
import smoke_evaluator_v21b as v21b  # noqa: E402
from fasttd3_ptf.evaluation import schema_v2, task_metrics  # noqa: E402

N_EPISODES = v21b.N_EPISODES
DEVICE = os.environ.get("SMOKE_DEVICE", "cpu")
OUT = REPO / "docs/data/evaluator_v21c_smoke"


def check_S10(eps, env_name):
    """registry 冻结的 reducer 在真实 runtime 上必须真的输出。

    通过条件（protocol v21c §6）：
      - 无 episode 报 MISSING_MILESTONE_FIELD；
      - 每个声明了 reducer 的 milestone 都存在聚合结构（无 MISSING 标记）；
      - `max` / `final` 声明处非 null；`ever_true` 声明处非 null。
        **`first_hit_step` 允许为 null**——truck 的策略可能一个 package 都没装上，
        "从未达成"是合法观测，不是缺陷。
    """
    fails = []
    spec = task_metrics.TASK_METRIC_REGISTRY.get(env_name)
    if spec is None or not spec.milestone_reducers:
        return [f"{env_name} 未声明 milestone_reducers，S10 无对象可验"], []

    bad_status = [e["seed"] for e in eps
                  if e["metric_status"] == task_metrics.STATUS_MISSING_MILESTONE_FIELD]
    if bad_status:
        fails.append(f"{len(bad_status)} 个 episode 报 MISSING_MILESTONE_FIELD："
                     f"声明的 milestone 在真实 runtime 上缺失")

    for name, reducers in spec.milestone_reducers.items():
        for e in eps:
            slot = (e.get("milestones") or {}).get(name)
            if slot is None:
                fails.append(f"seed={e['seed']} 缺 milestone {name!r} 的聚合结构")
                break
            if slot.get("status") == task_metrics.MILESTONE_MISSING:
                fails.append(f"seed={e['seed']} milestone {name!r} 标记为 "
                             f"{task_metrics.MILESTONE_MISSING}")
                break
            for r in reducers:
                if r == task_metrics.REDUCER_FIRST_HIT_STEP:
                    continue          # 允许 None：从未达成是合法观测
                if slot.get(r) is None:
                    fails.append(f"seed={e['seed']} {name}.{r} 为 null（声明了该 reducer）")
                    break
    return fails, ["milestone_reducers"]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    results, n_fail = {}, 0
    episodes_by_tag, ident_by_tag = {}, {}

    for tag, env_name, ckpt_rel in v21b.CASES:
        ckpt = REPO / ckpt_rel
        print(f"\n{'='*70}\n{tag}  {env_name}\n  {ckpt_rel}", flush=True)
        if not ckpt.exists():
            print("  FAILED: checkpoint 不存在")
            results[tag] = {"env": env_name, "status": "FAILED",
                            "fails": ["checkpoint 不存在"], "checkpoint": ckpt_rel}
            n_fail += 1
            continue
        try:
            ident = ev.verify_checkpoint_identity(
                str(ckpt), env_name, identity_mode="debug")
            eps = ev.run_panel_v2(str(ckpt), env_name, device=DEVICE, n_episodes=N_EPISODES)
        except Exception as exc:  # noqa: BLE001
            print(f"  FAILED: {type(exc).__name__}: {exc}")
            results[tag] = {"env": env_name, "status": "FAILED",
                            "fails": [f"{type(exc).__name__}: {exc}"], "checkpoint": ckpt_rel}
            n_fail += 1
            continue

        episodes_by_tag[tag], ident_by_tag[tag] = eps, ident
        fails, verified, unverified = v21b.CHECKS[tag](eps)
        n_fail += bool(fails)
        summary = v21b._summarize(tag, env_name, ckpt_rel, ident, eps,
                                  fails, verified, unverified)
        results[tag] = summary
        print(f"  {summary['status']}  terminated={summary['n_terminated']}/{len(eps)} "
              f"semantics={summary['semantics']} status={summary['metric_status']}")
        for f in fails:
            print(f"    ! {f}")
        for u in unverified:
            print(f"    ~ 未验证路径 {u['path']}：{u['reason']}")

    # ── S6：复用 S3 的 episodes ──────────────────────────────────────
    for tag, reuse, checker in (("S6", v21b.S6_REUSES, v21b.CHECKS["S6"]),):
        print(f"\n{'='*70}\n{tag}  （复用 {reuse} 的 episodes）")
        if reuse not in episodes_by_tag:
            results[tag] = {"status": "FAILED", "fails": [f"{reuse} 未产出 episodes"]}
            n_fail += 1
            continue
        eps = episodes_by_tag[reuse]
        env_name = dict((t, e) for t, e, _ in v21b.CASES)[reuse]
        ckpt_rel = dict((t, c) for t, _, c in v21b.CASES)[reuse]
        fails, verified, unverified = checker(eps)
        n_fail += bool(fails)
        summary = v21b._summarize(tag, env_name, ckpt_rel, ident_by_tag[reuse],
                                  eps, fails, verified, unverified)
        summary["reuses_episodes_from"] = reuse
        results[tag] = summary
        print(f"  {summary['status']}")
        for f in fails:
            print(f"    ! {f}")

    # ── S10：registry reducer 的正常分支，逐个有声明的任务验 ──────────
    print(f"\n{'='*70}\nS10  registry 冻结 reducer 的真实 runtime 输出")
    s10_fails, s10_detail = [], {}
    for tag in ("S3", "S4", "S5"):
        if tag not in episodes_by_tag:
            continue
        env_name = dict((t, e) for t, e, _ in v21b.CASES)[tag]
        fails, _verified = check_S10(episodes_by_tag[tag], env_name)
        s10_fails.extend(f"[{tag} {env_name}] {f}" for f in fails)
        sample = (episodes_by_tag[tag][0].get("milestones") or {})
        s10_detail[env_name] = {
            "declared": {k: list(v) for k, v in
                         task_metrics.TASK_METRIC_REGISTRY[env_name].milestone_reducers.items()},
            "sample_episode_0": sample,
        }
        print(f"  {env_name:34s} {'FAILED' if fails else 'PASS'}")
        for f in fails:
            print(f"    ! {f}")
    n_fail += bool(s10_fails)
    results["S10"] = {
        "status": "FAILED" if s10_fails else "PASS",
        "fails": s10_fails,
        "verified_paths": [] if s10_fails else ["milestone_reducers"],
        "unverified_paths": [],
        "detail": s10_detail,
    }

    vacuous = [t for t, r in results.items() if r.get("status") == "VACUOUS"]
    verdict = (f"FAILED_{n_fail}" if n_fail
               else "CORE_PATHS_VERIFIED_WITH_GAPS" if vacuous
               else "ALL_PATHS_EXERCISED")

    blocked, verified_map = {}, {"termination_semantics": [], "milestone": []}
    for r in results.values():
        env = r.get("env")
        if not env:
            continue
        for u in r.get("unverified_paths") or []:
            blocked.setdefault(env, []).append({"path": u["path"], "reason": u["reason"]})
        for p in r.get("verified_paths") or []:
            if p in verified_map and env not in verified_map[p]:
                verified_map[p].append(env)

    payload = {
        "protocol": [
            "docs/experiments/evaluator_v21b_prereg_20260807.md",
            "docs/experiments/evaluator_v21c_reverification_protocol_20260807.md",
        ],
        "n_episodes_per_case": N_EPISODES,
        "device": DEVICE,
        "warning": "smoke 数值仅验证通路，不得用于任何科学判断",
        "evaluation_semantics_digest": ev.evaluation_semantics_digest(
            schema_v2.SCHEMA_VERSION),
        "evaluation_semantics_files": ev.semantics_file_digests(),
        "verdict": verdict,
        "n_failed_cases": n_fail,
        "vacuous_cases": vacuous,
        "scientific_use_blocked": blocked,
        "runtime_verified": {k: sorted(v) for k, v in verified_map.items()},
        "cases": results,
    }
    out = OUT / "smoke.json"
    ev.atomic_write_json(out, payload, allow_overwrite=True)
    print(f"\n{'='*70}\nVERDICT: {verdict}")
    if vacuous:
        print(f"VACUOUS（判据真空成立，路径未被执行）: {vacuous}")
    if blocked:
        print("科学裁决阻断：")
        for env, items in blocked.items():
            print(f"  {env}: {[i['path'] for i in items]}")
    print(f"wrote {out.relative_to(REPO)}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
