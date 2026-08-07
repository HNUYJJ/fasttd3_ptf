#!/usr/bin/env python3
"""S1–S6 真实 runtime smoke（evaluator v2.1b）。

判据冻结于 docs/experiments/evaluator_v21b_prereg_20260807.md §5。
**必须用真实 checkpoint + 真实 MuJoCo 环境**，不接受 mock。

每项 8 episodes，CPU 即可。目的是验证通路而非获得科学结论——
**其数值不得用于任何科学判断**。

与 v2.1 首轮（已降级为 DIAGNOSTIC）的区别：

  S5  判据改回预注册原文 isfinite(ball_to_hoop_dist)，不再用
      metric_status != INSUFFICIENT_STATE 这个弱代理（后者 0 终止时真空通过）
  S6  新增，无条件验证 milestone 的 trajectory 聚合结构真的在跑
  verdict 禁用 ALL_PASS 及任何全称词

身份校验用 debug 模式：smoke 就是冒烟，产物本就不得用于科学裁决。
formal 模式的强制性由单元测试覆盖（那里能构造缺字段的 checkpoint）。
"""

from __future__ import annotations

import json
import math
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
OUT = REPO / "docs/data/evaluator_v21b_smoke"

# checkpoint 选择在预注册 §8.2 中冻结，本轮不得更换。
# S2 用 20k 而非 final：验证 failure 语义**必须有终止 episode**，
# 而 final scratch 已学会不摔倒（首轮实测 0/8）。若这次跑出 0/8，
# S2 记 VACUOUS，**不许再换第三个**——那会变成挑数据让判据通过。
CASES = [
    ("S1", "h1hand-crawl-v0",
     "models/h1hand-crawl-v0__h1hand_crawl_tp_scr_s1_20260615T044012Z__1_final.pt"),
    ("S2", "h1hand-slide-v0",
     "models/h1hand-slide-v0__slide_bac_walk_s1__1_20000.pt"),
    ("S3", "h1hand-truck-v0",
     "models/h1hand-truck-v0__h1hand_truck_scratch50k_s1_20260612T161202Z__1_final.pt"),
    ("S4", "h1hand-bookshelf_simple-v0",
     "models/h1hand-bookshelf_simple-v0__h1hand_bookshelf_simple_b2_scr_s1_20260705T153732Z__1_final.pt"),
    ("S5", "h1hand-basketball-v0",
     "models/h1hand-basketball-v0__h1hand_basketball_b2_scr_s1_20260705T153732Z__1_30000.pt"),
]

# S6 复用 S3 的 episodes（同一 checkpoint、同一批 episode），不重跑。
S6_REUSES = "S3"


def _ms(e, key):
    """取某 milestone 的聚合结构；不存在返回 None。"""
    return (e.get("milestones") or {}).get(key)


def check_S1(eps):
    """crawl：neutral 语义。Crawl.get_terminated 恒 return False。

    这一项无条件——8/8 都要满足具体值，不存在真空通过的可能。
    """
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
    # crawl 的"未终止 → neutral"路径被 8/8 执行，故 termination 语义已验证。
    return fails, ["termination_semantics"], []


def check_S2(eps):
    """slide：failure 语义。需存在摔倒终止的 episode。"""
    fails = []
    term = [e for e in eps if e["terminated"]]
    for e in term:
        if e["task_success"] is not False:
            fails.append(f"seed={e['seed']} 摔倒终止但 task_success={e['task_success']}")
        if e["termination_semantics"] != "failure":
            fails.append(f"seed={e['seed']} 语义={e['termination_semantics']}，应为 failure")
    if not term:
        return fails, [], [{"path": "termination_semantics",
                            "reason": "0/8 终止，failure 语义路径未被执行"}]
    return fails, ["termination_semantics"], []


def check_S3(eps):
    """truck：milestone 通路（无条件）+ 若终止则须成功（条件式）。

    预注册 §5 的降级条款：truck 极难，8 episodes 内可能无成功，
    故硬性要求是 milestone 提取通路可用。
    """
    fails, verified, unverified = [], [], []
    if not all(e["metric_status"] == "OK" for e in eps):
        fails.append(f"metric_status 实得 {sorted({e['metric_status'] for e in eps})}")

    with_ms = [e for e in eps if _ms(e, "success_subtasks")]
    if not with_ms:
        fails.append("无任何 episode 提取到 success_subtasks —— milestone 通路不可用")
    else:
        if any((_ms(e, "success_subtasks") or {}).get("n_steps_present", 0) <= 0
               for e in with_ms):
            fails.append("success_subtasks 的 n_steps_present <= 0")
        else:
            verified.append("milestone")

    term = [e for e in eps if e["terminated"]]
    for e in term:
        if e["task_success"] is not True:
            fails.append(f"seed={e['seed']} terminated 但 task_success={e['task_success']}")
    if not term:
        unverified.append({"path": "termination_semantics",
                           "reason": "0/8 终止，success 语义路径未被执行"})
    else:
        verified.append("termination_semantics")
    return fails, verified, unverified


def check_S4(eps):
    """bookshelf：条件终止。终止时 terminated_reason 必须存在且被正确映射。

    首轮的教训：0 终止时该条件式判据**真空成立**，首版报成 PASS 会误导。
    真空成立不是验证——此时记 VACUOUS，并把该任务挡在科学裁决之外。
    """
    fails, verified, unverified = [], [], []
    if not all(e["metric_status"] == "OK" for e in eps):
        fails.append(f"metric_status 实得 {sorted({e['metric_status'] for e in eps})}")

    term = [e for e in eps if e["terminated"]]
    for e in term:
        sem = e["termination_semantics"]
        if sem not in ("success", "failure"):
            fails.append(f"seed={e['seed']} 终止但语义={sem}（应为 success/failure，"
                         f"若为 unknown 说明 terminated_reason 缺失）")
    if not term:
        unverified.append({"path": "termination_semantics",
                           "reason": "0/8 终止，terminated_reason 0/1/2 映射未被执行"})
    else:
        verified.append("termination_semantics")

    if any(_ms(e, "success_subtasks") for e in eps):
        verified.append("milestone")
    else:
        unverified.append({"path": "milestone",
                           "reason": "无 episode 提取到 success_subtasks"})
    return fails, verified, unverified


def check_S5(eps):
    """basketball：MuJoCo state 提取。

    **判据是预注册原文**：至少一个 episode 的 ball_to_hoop_dist 是有限数值。
    首轮实现退化成了 "若有终止 episode 则不全为 INSUFFICIENT_STATE"，
    那在 0 终止时真空通过；而球与筐的距离**不论是否终止都可测**，
    故原文判据是无条件的。
    """
    fails, verified, unverified = [], [], []
    dists = []
    for e in eps:
        st = e.get("mujoco_state") or {}
        d = st.get("ball_to_hoop_dist")
        if d is not None and isinstance(d, (int, float)) and math.isfinite(float(d)):
            dists.append(float(d))
    if not dists:
        errs = sorted({str(e.get("mujoco_state_error")) for e in eps})
        fails.append(f"无任何 episode 提取到有限的 ball_to_hoop_dist —— "
                     f"提取路径是坏的。错误：{errs}")
    else:
        verified.append("mujoco_state")

    term = [e for e in eps if e["terminated"]]
    for e in term:
        if e["termination_semantics"] not in ("success", "failure"):
            fails.append(f"seed={e['seed']} 终止但语义={e['termination_semantics']}")
    if not term:
        unverified.append({"path": "termination_semantics",
                           "reason": "0/8 终止，进筐/未进筐判定未被执行"})
    else:
        verified.append("termination_semantics")
    return fails, verified, unverified


def check_S6(eps):
    """truck：trajectory 聚合结构（无条件）。

    验证 §3 的聚合真的在跑，而不是把最后一步的值包了一层壳：
    只包壳的话 max_step / n_steps_present 无从产生。
    """
    fails = []
    slots = [_ms(e, "success_subtasks") for e in eps]
    slots = [s for s in slots if s]
    if not slots:
        fails.append("无任何 episode 有 success_subtasks 的聚合结构")
        return fails, [], []
    for field in ("max", "final", "max_step", "n_steps_present"):
        missing = [i for i, s in enumerate(slots) if s.get(field) is None]
        if missing:
            fails.append(f"{len(missing)}/{len(slots)} 个 episode 的 "
                         f"success_subtasks.{field} 为 null（要求全部非 null）")
    # 聚合必须覆盖整条 trajectory，而不是只看最后一步
    for i, (s, e) in enumerate(zip(slots, eps)):
        n = s.get("n_steps_present")
        if isinstance(n, int) and n <= 1 and e["episode_length"] > 1:
            fails.append(f"seed={e['seed']} episode_length={e['episode_length']} 但 "
                         f"n_steps_present={n} —— 聚合只覆盖了单步")
    return fails, ["milestone_trajectory_aggregation"], []


CHECKS = {"S1": check_S1, "S2": check_S2, "S3": check_S3,
          "S4": check_S4, "S5": check_S5, "S6": check_S6}


def _summarize(tag, env_name, ckpt_rel, ident, eps, fails, verified, unverified):
    status = "FAILED" if fails else ("VACUOUS" if unverified else "PASS")
    summary = {
        "env": env_name, "status": status, "fails": fails,
        "verified_paths": verified,
        "unverified_paths": unverified,
        "checkpoint": ckpt_rel,
        "checkpoint_global_step": ident["global_step"],
        "learner_seed": ident["learner_seed"],
        "identity_mode": ident["identity_mode"],
        "scientific_use_permitted": ident["scientific_use_permitted"],
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
    if tag in ("S5",):
        summary["mujoco_state_samples"] = [e.get("mujoco_state") for e in eps[:3]]
        summary["mujoco_state_errors"] = sorted(
            {str(e.get("mujoco_state_error")) for e in eps})
    if tag in ("S3", "S6"):
        summary["success_subtasks_agg_samples"] = [
            _ms(e, "success_subtasks") for e in eps[:3]]
    return summary


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    results, n_fail = {}, 0
    episodes_by_tag, ident_by_tag = {}, {}

    for tag, env_name, ckpt_rel in CASES:
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
        fails, verified, unverified = CHECKS[tag](eps)
        n_fail += bool(fails)
        summary = _summarize(tag, env_name, ckpt_rel, ident, eps, fails, verified, unverified)
        results[tag] = summary
        print(f"  {summary['status']}  terminated={summary['n_terminated']}/{len(eps)} "
              f"semantics={summary['semantics']} status={summary['metric_status']}")
        print(f"  task_success={summary['task_success']} milestones={summary['milestone_keys']}")
        for f in fails:
            print(f"    ! {f}")
        for u in unverified:
            print(f"    ~ 未验证路径 {u['path']}：{u['reason']}")

    # ── S6：复用 S3 的 episodes，不重跑 ─────────────────────────────
    tag = "S6"
    print(f"\n{'='*70}\n{tag}  milestone trajectory 聚合（复用 {S6_REUSES} 的 episodes）")
    if S6_REUSES not in episodes_by_tag:
        results[tag] = {"status": "FAILED",
                        "fails": [f"{S6_REUSES} 未产出 episodes，无法复用"]}
        n_fail += 1
        print(f"  FAILED: {S6_REUSES} 未产出 episodes")
    else:
        eps = episodes_by_tag[S6_REUSES]
        env_name = dict((t, e) for t, e, _ in CASES)[S6_REUSES]
        ckpt_rel = dict((t, c) for t, _, c in CASES)[S6_REUSES]
        fails, verified, unverified = CHECKS[tag](eps)
        n_fail += bool(fails)
        summary = _summarize(tag, env_name, ckpt_rel, ident_by_tag[S6_REUSES],
                             eps, fails, verified, unverified)
        summary["reuses_episodes_from"] = S6_REUSES
        results[tag] = summary
        print(f"  {summary['status']}")
        for f in fails:
            print(f"    ! {f}")

    # ── verdict：禁用 ALL_PASS 及任何全称词（预注册 §6）──────────────
    vacuous = [t for t, r in results.items() if r.get("status") == "VACUOUS"]
    if n_fail:
        verdict = f"FAILED_{n_fail}"
    elif vacuous:
        verdict = "CORE_PATHS_VERIFIED_WITH_GAPS"
    else:
        verdict = "ALL_PATHS_EXERCISED"

    # 未验证路径 → 该任务的该用途不得进入科学裁决
    blocked, verified_map = {}, {"termination_semantics": [], "milestone": []}
    for tag_, r in results.items():
        env = r.get("env")
        if not env:
            continue
        for u in r.get("unverified_paths") or []:
            blocked.setdefault(env, []).append({"path": u["path"], "reason": u["reason"]})
        for p in r.get("verified_paths") or []:
            if p in verified_map and env not in verified_map[p]:
                verified_map[p].append(env)

    payload = {
        "prereg": "docs/experiments/evaluator_v21b_prereg_20260807.md",
        "n_episodes_per_case": N_EPISODES,
        "device": DEVICE,
        "warning": "smoke 数值仅验证通路，不得用于任何科学判断",
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
