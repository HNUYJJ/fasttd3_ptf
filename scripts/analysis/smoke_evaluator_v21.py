#!/usr/bin/env python3
"""S1–S5 真实 runtime smoke（evaluator v2.1）。

判据冻结于 docs/experiments/evaluator_v21_hardening_prereg_20260806.md §5。
**必须用真实 checkpoint + 真实 MuJoCo 环境**，不接受 mock。

每项 8 episodes，CPU 即可。目的是验证通路而非获得科学结论——
**其数值不得用于任何科学判断**。

任一项失败即非零退出；按预注册 §7，S1–S5 未全通过不得进入 P2。
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

import p0_evaluator_v2 as ev  # noqa: E402

N_EPISODES = 8
DEVICE = os.environ.get("SMOKE_DEVICE", "cpu")
OUT = REPO / "docs/data/evaluator_v21_smoke"

CASES = [
    ("S1", "h1hand-crawl-v0",
     "models/h1hand-crawl-v0__h1hand_crawl_tp_scr_s1_20260615T044012Z__1_final.pt"),
    ("S2", "h1hand-slide-v0",
     "models/h1hand-slide-v0__h1hand_slide_tp_scr_s1_20260615T044012Z__1_final.pt"),
    ("S3", "h1hand-truck-v0",
     "models/h1hand-truck-v0__h1hand_truck_scratch50k_s1_20260612T161202Z__1_final.pt"),
    ("S4", "h1hand-bookshelf_simple-v0",
     "models/h1hand-bookshelf_simple-v0__h1hand_bookshelf_simple_b2_scr_s1_20260705T153732Z__1_final.pt"),
    ("S5", "h1hand-basketball-v0",
     "models/h1hand-basketball-v0__h1hand_basketball_b2_scr_s1_20260705T153732Z__1_30000.pt"),
]


def check_S1(eps):
    """crawl：neutral 语义。Crawl.get_terminated 恒 False。"""
    fails = []
    if any(e["terminated"] for e in eps):
        fails.append("crawl 出现了 terminated=true，与 get_terminated 恒 False 矛盾")
    if not all(e["termination_semantics"] == "neutral" for e in eps):
        got = sorted({e["termination_semantics"] for e in eps})
        fails.append(f"语义应恒为 neutral，实得 {got}")
    if not all(e["task_success"] is False for e in eps):
        fails.append("task_success 应恒为 False")
    if not all(e["metric_status"] == "OK" for e in eps):
        fails.append(f"metric_status 应恒为 OK，实得 {sorted({e['metric_status'] for e in eps})}")
    return fails


def check_S2(eps):
    """slide：failure 语义。需存在摔倒终止的 episode。"""
    fails = []
    term = [e for e in eps if e["terminated"]]
    if not term:
        fails.append("8 个 episode 中无一 terminated，无法验证 failure 语义")
    for e in term:
        if e["task_success"] is not False:
            fails.append(f"seed={e['seed']} 摔倒终止但 task_success={e['task_success']}")
        if e["termination_semantics"] != "failure":
            fails.append(f"seed={e['seed']} 语义={e['termination_semantics']}，应为 failure")
    return fails


def check_S3(eps):
    """truck：milestone 通路必须可用；若有终止则须判为成功。

    降级条款（预注册 §5）：truck 极难，8 episodes 内可能无成功，
    故硬性要求是 milestone 提取通路可用。
    """
    fails = []
    if not all(e["metric_status"] == "OK" for e in eps):
        fails.append(f"metric_status 实得 {sorted({e['metric_status'] for e in eps})}")
    if not any("success_subtasks" in (e.get("milestones") or {}) for e in eps):
        fails.append("无任何 episode 提取到 success_subtasks —— milestone 通路不可用")
    for e in eps:
        if e["terminated"] and e["task_success"] is not True:
            fails.append(f"seed={e['seed']} terminated 但 task_success={e['task_success']}")
    return fails


def check_S4(eps):
    """bookshelf：条件终止。终止时 terminated_reason 必须存在且被正确映射。"""
    fails = []
    if not all(e["metric_status"] == "OK" for e in eps):
        fails.append(f"metric_status 实得 {sorted({e['metric_status'] for e in eps})}")
    for e in eps:
        if not e["terminated"]:
            continue
        sem = e["termination_semantics"]
        if sem not in ("success", "failure"):
            fails.append(f"seed={e['seed']} 终止但语义={sem}（应为 success/failure，"
                         f"若为 unknown 说明 terminated_reason 缺失）")
    return fails


def check_S5(eps):
    """basketball：MuJoCo state 提取。ball_to_hoop_dist 不得恒为 None。

    无降级条款：恒 None 意味 basketball 的 task_success 永远是
    INSUFFICIENT_STATE，该任务实质不可评估。
    """
    fails = []
    n_insuf = sum(1 for e in eps if e["metric_status"] == "INSUFFICIENT_STATE")
    n_term = sum(1 for e in eps if e["terminated"])
    if n_term and n_insuf == n_term:
        fails.append(
            f"{n_term} 个终止 episode 全部 INSUFFICIENT_STATE —— "
            f"ball_to_hoop_dist 提取路径是坏的")
    return fails


CHECKS = {"S1": check_S1, "S2": check_S2, "S3": check_S3, "S4": check_S4, "S5": check_S5}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    results, n_fail = {}, 0

    for tag, env_name, ckpt_rel in CASES:
        ckpt = REPO / ckpt_rel
        print(f"\n{'='*70}\n{tag}  {env_name}\n  {ckpt_rel}", flush=True)
        if not ckpt.exists():
            print(f"  FAIL: checkpoint 不存在")
            results[tag] = {"env": env_name, "status": "FAIL",
                            "fails": ["checkpoint 不存在"], "checkpoint": ckpt_rel}
            n_fail += 1
            continue
        try:
            ident = ev.verify_checkpoint_identity(str(ckpt), env_name)
            eps = ev.run_panel_v2(str(ckpt), env_name, device=DEVICE, n_episodes=N_EPISODES)
        except Exception as exc:  # noqa: BLE001
            print(f"  FAIL: {type(exc).__name__}: {exc}")
            results[tag] = {"env": env_name, "status": "FAIL",
                            "fails": [f"{type(exc).__name__}: {exc}"], "checkpoint": ckpt_rel}
            n_fail += 1
            continue

        fails = CHECKS[tag](eps)
        status = "PASS" if not fails else "FAIL"
        n_fail += bool(fails)

        summary = {
            "env": env_name, "status": status, "fails": fails,
            "checkpoint": ckpt_rel,
            "checkpoint_global_step": ident["global_step"],
            "learner_seed": ident["learner_seed"],
            "n_episodes": len(eps),
            "n_terminated": sum(1 for e in eps if e["terminated"]),
            "n_truncated": sum(1 for e in eps if e["truncated"]),
            "semantics": {s: sum(1 for e in eps if e["termination_semantics"] == s)
                          for s in sorted({e["termination_semantics"] for e in eps})},
            "metric_status": {s: sum(1 for e in eps if e["metric_status"] == s)
                              for s in sorted({e["metric_status"] for e in eps})},
            "task_success": {
                "true": sum(1 for e in eps if e["task_success"] is True),
                "false": sum(1 for e in eps if e["task_success"] is False),
                "null": sum(1 for e in eps if e["task_success"] is None)},
            "milestone_keys": sorted({k for e in eps for k in (e.get("milestones") or {})}),
            "return_mean": sum(e["return"] for e in eps) / len(eps),
        }
        results[tag] = summary
        print(f"  {status}  terminated={summary['n_terminated']}/{len(eps)} "
              f"semantics={summary['semantics']} status={summary['metric_status']}")
        print(f"  task_success={summary['task_success']} milestones={summary['milestone_keys']}")
        for f in fails:
            print(f"    ! {f}")

    payload = {
        "prereg": "docs/experiments/evaluator_v21_hardening_prereg_20260806.md",
        "n_episodes_per_case": N_EPISODES,
        "device": DEVICE,
        "warning": "smoke 数值仅验证通路，不得用于任何科学判断",
        "verdict": "ALL_PASS" if n_fail == 0 else f"{n_fail}_CASES_FAILED",
        "cases": results,
    }
    out = OUT / "smoke.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n{'='*70}\nVERDICT: {payload['verdict']}\nwrote {out.relative_to(REPO)}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
