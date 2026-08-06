"""Door fixed-horizon option handoff 裁决（预注册，揭盲前定稿，只读）。

四条件（前三组复用既有结果）：
    Student          B=0, R=0    docs/data/door_at10k_gate_v1/
    Joint RBO        segment, R=1
    Segment B-only   segment, R=0  docs/data/door_channel_decomposition_v1/
    Prefix B-only    prefix,  R=0  docs/data/door_prefix_handoff_v1/     ← 本轮新跑

主比较（唯一决定裁决）：
    Delta_placement = J(prefix B-only) - J(segment B-only)      同 seed 配对

它隔离的唯一因子是 source 的**时间放置方式**：source 身份、训练阶段、
总行为剂量、replay eligibility 全部相同。

裁决（PI 冻结）：
    PREFIX_SUPERIOR       LCB_90(Delta_placement) > 0
    PREFIX_PROMISING      3/3 seed Delta > 0 且 mean(Delta) >= +30
                          （仅报告，等 PI 决定；不自动扩展，不作确认性结论）
    PREFIX_NOT_SUPPORTED  其余；停止该路线，不调 H、不加 bandit、不复活 termination

次级（单独报告，不参与裁决）：prefix vs student、prefix vs joint、跨 seed 方差。
episode-level SE 只作评价可靠性诊断，不得代替 learner-seed 不确定性（教训 M16）。
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
DIR_GATE = REPO / "docs/data/door_at10k_gate_v1/source_free_eval"
DIR_SEG = REPO / "docs/data/door_channel_decomposition_v1/source_free_eval"
DIR_PRE = REPO / "docs/data/door_prefix_handoff_v1/source_free_eval"
OUT = REPO / "docs/data/door_prefix_handoff_v1/door_prefix_handoff_v1_results.json"
SOURCE = "run"
SEEDS = [1, 2, 3]
T90_DF2 = 2.919986
PROMISING_MIN_MEAN = 30.0


def panel(path: Path) -> dict:
    d = json.loads(path.read_text())
    rets = [float(e["return"]) for e in d["episodes"]]
    n = len(rets)
    return {"mean": sum(rets) / n, "n": n,
            "se": st.pstdev(rets) / math.sqrt(n) if n > 1 else None,
            "global_step": d["checkpoint"]["global_step"]}


def interval(values: list[float]) -> dict:
    n = len(values)
    mean = sum(values) / n
    sd = math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1))
    half = T90_DF2 * sd / math.sqrt(n)
    return dict(mean=mean, sd=sd, lo=mean - half, hi=mean + half, n=n)


def sign_of(iv: dict) -> str:
    return "neg" if iv["hi"] < 0 else ("pos" if iv["lo"] > 0 else "unc")


def audit(seed: int) -> dict:
    import torch

    hits = sorted(glob.glob(str(REPO / f"models/*door_prefix_{SOURCE}_s{seed}__*_20000.pt")))
    if not hits:
        return {"checkpoint": None}
    s = torch.load(hits[0], map_location="cpu", weights_only=False)
    a = s.get("admission_audit") or {}
    cfg = s.get("ptf_cfg") or {}
    ex, cr = a.get("execution_counts"), a.get("critic_sample_counts")
    out = {"checkpoint": os.path.basename(hits[0]), "global_step": s.get("global_step"),
           "episode_prefix_steps": cfg.get("mcg_episode_prefix_steps"),
           "replay_mode": cfg.get("admission_replay_mode"),
           "active_buffer_counts": a.get("active_buffer_counts"),
           "sampling_phase": a.get("sampling_phase")}
    if ex and sum(ex) > 0:
        out["behavior_source_share"] = sum(ex[:-1]) / sum(ex)
    if cr is not None:
        out["critic_source_sample_count"] = int(sum(cr[:-1]))
    return out


def main() -> None:
    J0 = {s: panel(DIR_GATE / f"student_s{s}_step20000.json") for s in SEEDS}
    JBR = {s: panel(DIR_GATE / f"{SOURCE}_s{s}_step20000.json") for s in SEEDS}
    JSEG = {s: panel(DIR_SEG / f"{SOURCE}_s{s}_step20000.json") for s in SEEDS}
    JPRE = {s: panel(DIR_PRE / f"{SOURCE}_s{s}_step20000.json") for s in SEEDS}

    d_place = [JPRE[s]["mean"] - JSEG[s]["mean"] for s in SEEDS]
    d_stu = [JPRE[s]["mean"] - J0[s]["mean"] for s in SEEDS]
    d_joint = [JPRE[s]["mean"] - JBR[s]["mean"] for s in SEEDS]

    iv_place, iv_stu, iv_joint = interval(d_place), interval(d_stu), interval(d_joint)

    if iv_place["lo"] > 0:
        verdict = "PREFIX_SUPERIOR"
    elif all(v > 0 for v in d_place) and iv_place["mean"] >= PROMISING_MIN_MEAN:
        verdict = "PREFIX_PROMISING"
    else:
        verdict = "PREFIX_NOT_SUPPORTED"

    s_stu = sign_of(iv_stu)
    vs_student = {"pos": "positive_transfer", "neg": "still_negative_transfer",
                  "unc": "mitigation_or_uncertain"}[s_stu]

    report = dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        experiment="door_prefix_handoff_v1",
        positioning="PTF-derived fixed-horizon option handoff feasibility ablation "
                    "(no curriculum, no G_i, no bandit, no learned termination)",
        primary="Delta_placement = J(prefix B-only) - J(segment B-only), paired by seed",
        evaluator="frozen source-free panel, 16 eval seeds x 8 ranks = 128 episodes",
        interval_rule="90% t interval over 3 paired LEARNER seeds (df=2); "
                      "episode-level SE diagnostic only",
        per_seed={str(s): {
            "J_student": J0[s]["mean"], "J_joint": JBR[s]["mean"],
            "J_segment_Bonly": JSEG[s]["mean"], "J_prefix_Bonly": JPRE[s]["mean"],
            "Delta_placement": JPRE[s]["mean"] - JSEG[s]["mean"],
            "prefix_minus_student": JPRE[s]["mean"] - J0[s]["mean"],
            "prefix_minus_joint": JPRE[s]["mean"] - JBR[s]["mean"],
            "episode_se": {"student": J0[s]["se"], "joint": JBR[s]["se"],
                           "segment": JSEG[s]["se"], "prefix": JPRE[s]["se"]},
        } for s in SEEDS},
        Delta_placement=dict(iv_place, sign=sign_of(iv_place)),
        prefix_vs_student=dict(iv_stu, sign=s_stu, interpretation=vs_student),
        prefix_vs_joint=dict(iv_joint, sign=sign_of(iv_joint)),
        cross_seed_sd={
            "student": st.stdev([J0[s]["mean"] for s in SEEDS]),
            "joint": st.stdev([JBR[s]["mean"] for s in SEEDS]),
            "segment_Bonly": st.stdev([JSEG[s]["mean"] for s in SEEDS]),
            "prefix_Bonly": st.stdev([JPRE[s]["mean"] for s in SEEDS]),
            "note": "descriptive only; 3 seeds cannot upgrade a variance drop "
                    "into a mechanism contribution",
        },
        treatment_audit={str(s): audit(s) for s in SEEDS},
        verdict=verdict,
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=1, ensure_ascii=False))

    print("Door fixed-horizon option handoff（128-ep 面板, 3 learner seeds, 90% t 区间）\n")
    print(f"{'seed':5s} {'student':>9s} {'joint':>9s} {'segment':>9s} {'prefix':>9s} "
          f"{'Δ_place':>9s} {'vs stu':>9s}")
    for s in SEEDS:
        r = report["per_seed"][str(s)]
        print(f"{s:<5d} {r['J_student']:9.2f} {r['J_joint']:9.2f} {r['J_segment_Bonly']:9.2f} "
              f"{r['J_prefix_Bonly']:9.2f} {r['Delta_placement']:+9.2f} "
              f"{r['prefix_minus_student']:+9.2f}")
    print()
    print(f"主比较 Δ_placement   mean={iv_place['mean']:+8.2f}  "
          f"90%CI=[{iv_place['lo']:+8.2f}, {iv_place['hi']:+8.2f}]  {sign_of(iv_place)}")
    print(f"次级 prefix−student  mean={iv_stu['mean']:+8.2f}  "
          f"90%CI=[{iv_stu['lo']:+8.2f}, {iv_stu['hi']:+8.2f}]  → {vs_student}")
    print(f"次级 prefix−joint    mean={iv_joint['mean']:+8.2f}  "
          f"90%CI=[{iv_joint['lo']:+8.2f}, {iv_joint['hi']:+8.2f}]")
    csd = report["cross_seed_sd"]
    print(f"\n跨 seed sd（描述性）: student {csd['student']:.1f} | joint {csd['joint']:.1f} | "
          f"segment {csd['segment_Bonly']:.1f} | prefix {csd['prefix_Bonly']:.1f}")
    print(f"\nVERDICT: {verdict}\nsaved: {OUT}")


if __name__ == "__main__":
    main()
