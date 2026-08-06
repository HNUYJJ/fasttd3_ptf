"""Door 顺序因果分解裁决（预注册，揭盲前定稿，只读）。

分解（PI 冻结）
--------------
    J_0  = J_sf@20k(student)      B=0, R=0
    J_B  = J_sf@20k(B-only)       B=1, R=0   source 有 behavior authority，
                                             但 critic 只采 student provenance
    J_BR = J_sf@20k(joint RBO)    B=1, R=1   已在 door_at10k_gate_v1 测得

    U^B        = J_B  - J_0       固定交互预算下授予 behavior authority 的总效果。
                                  它**包含**source 占用了原属 student 的交互机会
                                  ——这是行为通道的真实机会成本，不是 confound。
    Δ^{R|B}    = J_BR - J_B       在同一 behavior authority 条件下，进一步允许
                                  source transitions 参与 critic replay 的条件增量。

精确恒等式：U^BR = U^B + Δ^{R|B}（脚本会数值校验）。
这不是纯 U^R，也不能单独识别交互项。

统计（PI 冻结）
--------------
对 3 个 **learner seeds** 分别配对，取 90% t 区间（df=2）。
episode-level paired SE 只作评价可靠性诊断，**不得**用来代替 learner-seed 不确定性
——Door 已经证明这两种方差量级不同。

裁决（PI 冻结）
--------------
以区间符号定义显著性：区间完全 <0 记 neg，完全 >0 记 pos，跨 0 记 unc。

    U^B=neg 且 Δ=pos                    → CHANNEL_CONFLICT
    U^B=pos 且 Δ=neg                    → CHANNEL_CONFLICT
    U^B=neg 且 Δ=neg                    → BOTH_CHANNELS_HARM
    U^B=neg 且 Δ∈{unc}                  → BEHAVIOR_COST_DOMINANT
    Δ=neg   且 U^B∈{unc,pos}            → REPLAY_ELIGIBILITY_DOMINANT
    其余                                 → UNRESOLVED

特别地：若 joint 的 U^BR 显著而两个分量各自都不显著，**只能**裁 UNRESOLVED，
不得称"纯交互"——那可能只是两个分量分别功效不足。
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
JOINT_DIR = REPO / "docs/data/door_at10k_gate_v1/source_free_eval"
BONLY_DIR = REPO / "docs/data/door_channel_decomposition_v1/source_free_eval"
OUT = REPO / "docs/data/door_channel_decomposition_v1/door_channel_decomposition_v1_results.json"
SOURCE = "run"          # PI 冻结：只分解 door-run（Door gate 中 harmful 且行为先验为正）
SEEDS = [1, 2, 3]
T90_DF2 = 2.919986


def _panel(path: Path) -> dict:
    d = json.loads(path.read_text())
    rets = [float(e["return"]) for e in d["episodes"]]
    n = len(rets)
    return {
        "mean": sum(rets) / n,
        "n": n,
        "se": st.pstdev(rets) / math.sqrt(n) if n > 1 else None,
        "eval_seeds": d["protocol"]["eval_seeds"],
        "global_step": d["checkpoint"]["global_step"],
    }


def interval(values: list[float]) -> dict:
    n = len(values)
    mean = sum(values) / n
    sd = math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1))
    half = T90_DF2 * sd / math.sqrt(n)
    return dict(mean=mean, sd=sd, lo=mean - half, hi=mean + half, n=n)


def sign_of(iv: dict) -> str:
    if iv["hi"] < 0:
        return "neg"
    if iv["lo"] > 0:
        return "pos"
    return "unc"


def treatment_audit(seed: int) -> dict:
    import torch

    hits = sorted(glob.glob(str(REPO / f"models/*door_Bonly_{SOURCE}_s{seed}*_20000.pt")))
    if not hits:
        return {"checkpoint": None}
    state = torch.load(hits[0], map_location="cpu", weights_only=False)
    audit = state.get("admission_audit") or {}
    ex, cr = audit.get("execution_counts"), audit.get("critic_sample_counts")
    out = {
        "checkpoint": os.path.basename(hits[0]),
        "global_step": state.get("global_step"),
        "replay_mode": (state.get("ptf_cfg") or {}).get("admission_replay_mode"),
        "active_buffer_counts": audit.get("active_buffer_counts"),
        "sampling_phase": audit.get("sampling_phase"),
    }
    if ex and sum(ex) > 0:
        out["behavior_source_share"] = sum(ex[:-1]) / sum(ex)
    if cr is not None:
        out["critic_source_sample_count"] = int(sum(cr[:-1]))
        out["critic_student_sample_count"] = int(cr[-1])
    return out


def main() -> None:
    J0 = {s: _panel(JOINT_DIR / f"student_s{s}_step20000.json") for s in SEEDS}
    JBR = {s: _panel(JOINT_DIR / f"{SOURCE}_s{s}_step20000.json") for s in SEEDS}
    JB = {s: _panel(BONLY_DIR / f"{SOURCE}_s{s}_step20000.json") for s in SEEDS}

    UB = [JB[s]["mean"] - J0[s]["mean"] for s in SEEDS]
    DR = [JBR[s]["mean"] - JB[s]["mean"] for s in SEEDS]
    UBR = [JBR[s]["mean"] - J0[s]["mean"] for s in SEEDS]

    iv_UB, iv_DR, iv_UBR = interval(UB), interval(DR), interval(UBR)
    s_UB, s_DR, s_UBR = sign_of(iv_UB), sign_of(iv_DR), sign_of(iv_UBR)

    if (s_UB == "neg" and s_DR == "pos") or (s_UB == "pos" and s_DR == "neg"):
        verdict = "CHANNEL_CONFLICT"
    elif s_UB == "neg" and s_DR == "neg":
        verdict = "BOTH_CHANNELS_HARM"
    elif s_UB == "neg" and s_DR == "unc":
        verdict = "BEHAVIOR_COST_DOMINANT"
    elif s_DR == "neg" and s_UB in ("unc", "pos"):
        verdict = "REPLAY_ELIGIBILITY_DOMINANT"
    else:
        verdict = "UNRESOLVED"

    joint_sig_components_not = s_UBR != "unc" and s_UB == "unc" and s_DR == "unc"

    report = dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        experiment="door_channel_decomposition_v1",
        source=SOURCE,
        decomposition="U^BR = U^B + Delta^{R|B}  (exact identity)",
        definitions={
            "U^B": "J(B-only) - J(student); behavior authority under a fixed interaction budget, "
                   "including the opportunity cost of source consuming student interactions",
            "Delta^{R|B}": "J(joint) - J(B-only); conditional increment of allowing source "
                           "transitions into critic replay, given the same behavior authority",
        },
        evaluator="frozen source-free panel, 16 eval seeds x 8 ranks = 128 episodes",
        interval_rule="90% t interval over 3 paired LEARNER seeds (df=2); "
                      "episode-level SE is diagnostic only",
        per_seed={
            str(s): {
                "J_student": J0[s]["mean"], "J_Bonly": JB[s]["mean"], "J_joint": JBR[s]["mean"],
                "U_B": JB[s]["mean"] - J0[s]["mean"],
                "Delta_R_given_B": JBR[s]["mean"] - JB[s]["mean"],
                "U_BR": JBR[s]["mean"] - J0[s]["mean"],
                "episode_se": {"student": J0[s]["se"], "Bonly": JB[s]["se"], "joint": JBR[s]["se"]},
            }
            for s in SEEDS
        },
        U_B=dict(iv_UB, sign=s_UB),
        Delta_R_given_B=dict(iv_DR, sign=s_DR),
        U_BR=dict(iv_UBR, sign=s_UBR),
        identity_max_abs_error=max(
            abs(UBR[i] - (UB[i] + DR[i])) for i in range(len(SEEDS))
        ),
        joint_significant_but_components_not=joint_sig_components_not,
        treatment_audit={str(s): treatment_audit(s) for s in SEEDS},
        verdict=verdict,
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=1, ensure_ascii=False))

    print(f"Door-{SOURCE} 顺序因果分解  (128-episode 面板, 3 learner seeds, 90% t 区间)\n")
    print(f"{'seed':5s} {'J_student':>10s} {'J_Bonly':>10s} {'J_joint':>10s} "
          f"{'U^B':>9s} {'Δ^R|B':>9s} {'U^BR':>9s}")
    for s in SEEDS:
        r = report["per_seed"][str(s)]
        print(f"{s:<5d} {r['J_student']:10.2f} {r['J_Bonly']:10.2f} {r['J_joint']:10.2f} "
              f"{r['U_B']:+9.2f} {r['Delta_R_given_B']:+9.2f} {r['U_BR']:+9.2f}")
    print()
    for name, iv, sg in (("U^B", iv_UB, s_UB), ("Δ^{R|B}", iv_DR, s_DR), ("U^BR", iv_UBR, s_UBR)):
        print(f"{name:9s} mean={iv['mean']:+8.2f}  90%CI=[{iv['lo']:+8.2f}, {iv['hi']:+8.2f}]  {sg}")
    print(f"\n恒等式校验 |U^BR - (U^B + Δ^R|B)| 最大误差 = {report['identity_max_abs_error']:.2e}")
    if joint_sig_components_not:
        print("注意：joint 显著而两分量各自不显著 → 只能裁 UNRESOLVED（不得称纯交互）")
    print(f"\nVERDICT: {verdict}\nsaved: {OUT}")


if __name__ == "__main__":
    main()
