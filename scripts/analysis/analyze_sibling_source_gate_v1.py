"""Slide ↔ Stair 双向 sibling-source gate 裁决（预注册，揭盲前定稿，只读）。

检验的假设（预注册 §1）：
    在地形仍不相同的条件下，与 target 共用同一 reward 实现（ClimbingUpwards）的
    sibling source，其 RBO 学习效用应稳定高于通用 walk source。

主比较（同 learner seed 配对）：
    D_sib(dir) = J(sibling source arm) − J(walk source arm)

    dir = slide→stair ：stair target，源为在 slide 上训练的冻结策略
    dir = stair→slide ：slide target，源为在 stair 上训练的冻结策略

裁决（冻结，逐字取自预注册 §4）：
    SIBLING_PRIOR_SUPPORTED       两个方向的配对均值都 > 0，
                                  且每个方向至少 2/3 seed 同向为正
    SIBLING_DIRECTION_DEPENDENT   仅一个方向成立
    SIBLING_PRIOR_REFUTED         两个方向均不成立
                                  → 停止 taxonomy 预测路线

不加 seeds、不调剂量、不换 horizon、不改评估面板来抢救。
episode-level SE 仅作评价可靠性诊断，不得代替 learner-seed 不确定性（教训 M16）。
"""
from __future__ import annotations

import glob
import json
import math
import os
import statistics as st
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SEEDS = [1, 2, 3]
T90_DF2 = 2.919986

# (方向名, target, sibling 臂名, 对照臂名)
DIRECTIONS = [
    ("slide_to_stair", "stair", "slidesrc", "walk"),
    ("stair_to_slide", "slide", "stairsrc", "walk"),
]


def panel(target: str, arm: str, seed: int) -> dict:
    p = REPO / f"docs/data/{target}_bac_gate_v1/source_free_eval/{arm}_s{seed}_step20000.json"
    d = json.loads(p.read_text())
    rets = [float(e["return"]) for e in d["episodes"]]
    n = len(rets)
    return {"mean": sum(rets) / n, "n": n,
            "se": st.pstdev(rets) / math.sqrt(n) if n > 1 else None}


def interval(values: list[float]) -> dict:
    n = len(values)
    mean = sum(values) / n
    sd = math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1))
    half = T90_DF2 * sd / math.sqrt(n)
    return dict(mean=mean, sd=sd, lo=mean - half, hi=mean + half, n=n)


def audit(target: str, arm: str, seed: int) -> dict:
    import torch
    hits = sorted(glob.glob(str(REPO / f"models/*{target}_bac_{arm}_s{seed}__*_20000.pt")))
    if not hits:
        return {"checkpoint": None}
    s = torch.load(hits[0], map_location="cpu", weights_only=False)
    a = s.get("admission_audit") or {}
    ex, cr = a.get("execution_counts"), a.get("critic_sample_counts")
    out = {"checkpoint": os.path.basename(hits[0]), "global_step": s.get("global_step")}
    if ex and sum(ex) > 0:
        out["behavior_source_share"] = sum(ex[:-1]) / sum(ex)
    if cr and sum(cr) > 0:
        out["critic_source_share"] = sum(cr[:-1]) / sum(cr)
    return out


def main() -> None:
    report = dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        experiment="sibling_source_gate_v1",
        hypothesis="sibling source sharing the exact reward implementation "
                   "(ClimbingUpwards) outperforms the generic walk source",
        primary="D_sib = J(sibling arm) - J(walk arm), paired by learner seed",
        evaluator="frozen source-free panel, 16 eval seeds x 8 ranks = 128 episodes",
        interval_rule="90% t interval over 3 paired LEARNER seeds (df=2)",
        directions={},
    )

    ok = {}
    for name, target, sib, ctrl in DIRECTIONS:
        per, d = {}, []
        for s in SEEDS:
            js, jc = panel(target, sib, s), panel(target, ctrl, s)
            jstu = panel(target, "student", s)
            per[str(s)] = {"J_sibling": js["mean"], "J_walk": jc["mean"],
                           "J_student": jstu["mean"],
                           "D_sib": js["mean"] - jc["mean"],
                           "U_sibling_vs_student": js["mean"] - jstu["mean"],
                           "episode_se": {"sibling": js["se"], "walk": jc["se"]}}
            d.append(js["mean"] - jc["mean"])
        iv = interval(d)
        n_pos = sum(1 for x in d if x > 0)
        passed = iv["mean"] > 0 and n_pos >= 2
        ok[name] = passed
        report["directions"][name] = dict(
            target=target, sibling_arm=sib, control_arm=ctrl,
            per_seed=per, D_sib=iv, n_positive_seeds=n_pos, direction_passed=passed,
            U_sibling_vs_student=interval(
                [per[str(s)]["U_sibling_vs_student"] for s in SEEDS]),
            treatment_audit={str(s): audit(target, sib, s) for s in SEEDS},
        )

    n_ok = sum(ok.values())
    verdict = ("SIBLING_PRIOR_SUPPORTED" if n_ok == 2 else
               "SIBLING_DIRECTION_DEPENDENT" if n_ok == 1 else
               "SIBLING_PRIOR_REFUTED")
    report["verdict"] = verdict

    out = REPO / "docs/data/sibling_source_gate_v1/sibling_source_gate_v1_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, ensure_ascii=False))

    print("Slide ↔ Stair sibling-source gate（128-ep 面板, 3 learner seeds, 90% t 区间）\n")
    for name, target, sib, ctrl in DIRECTIONS:
        r = report["directions"][name]
        print(f"### {name}   target={target}  sibling={sib}  control={ctrl}")
        print(f"{'seed':5s} {'student':>9s} {'walk':>9s} {'sibling':>9s} {'D_sib':>9s}")
        for s in SEEDS:
            p = r["per_seed"][str(s)]
            print(f"{s:<5d} {p['J_student']:9.2f} {p['J_walk']:9.2f} "
                  f"{p['J_sibling']:9.2f} {p['D_sib']:+9.2f}")
        iv = r["D_sib"]
        print(f"  D_sib mean={iv['mean']:+8.2f}  90%CI=[{iv['lo']:+8.2f}, {iv['hi']:+8.2f}]"
              f"  正 seed {r['n_positive_seeds']}/3  方向通过={r['direction_passed']}")
        u = r["U_sibling_vs_student"]
        print(f"  次级 sibling−student mean={u['mean']:+8.2f} "
              f"90%CI=[{u['lo']:+8.2f}, {u['hi']:+8.2f}]\n")
    print(f"VERDICT: {verdict}\nsaved: {out}")


if __name__ == "__main__":
    main()
