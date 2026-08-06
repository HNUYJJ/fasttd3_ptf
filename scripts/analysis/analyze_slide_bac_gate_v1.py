"""Slide BAC 判决场裁决（预注册，揭盲前定稿，只读）。

本实验检验的是 Bottleneck-Aligned Coverage 指标的**前瞻预测**，
预测已冻结于 docs/experiments/bottleneck_aligned_coverage_v1_prereg_20260728.md：

    zero-shot return 预测   stand(88.5) > walk(45.7) > run(27.8)
    NET(BAC) 预测           walk(0.5153) ≈ run(0.5086) ≫ stand(0.0129)   （差 40 倍）

两者顺序完全相反，故一次三臂实验即可裁决。

被估量（与 door/cabinet 系列一致）：
    U_i(10k, 10k) = J_sf@20k(source_i) - J_sf@20k(student)      同 seed 配对

主判据是**源之间的配对差**（student 项抵消，故等价于 J 的配对差）：
    D_walk = J(walk) - J(stand)
    D_run  = J(run)  - J(stand)

评估面板（训练前冻结）：
    primary   = 128 deterministic source-free episodes（16 eval seeds × 8 ranks）
    secondary = 前 32 个 episode，与既往所有 32-episode 面板逐位兼容，
                仅作可比性说明，**不参与裁决**。

三级裁决（PI 冻结，逐字取自预注册文档 §5）：
    BAC_SUPPORTED   LCB_90(D_walk) > 0 且 LCB_90(D_run) > 0
                    → NET 在 return 反向的场合胜出，指标进入论文主线
    BAC_PARTIAL     stand 为三者最差但区间跨零，或 walk/run 仅一者显著高于 stand
                    → 报告，不外推；须在 stair 上重复一次才决定去留
    BAC_REFUTED     U(stand) 不低于 walk/run 中的任何一个
                    → 指标失败。不调 BOTTLENECK_MASS / SIGN_EPS / SEPARATION_MIN、
                      不换瓶颈定义来抢救。gate 失败后不得调参重跑。

次级（单独报告，不参与裁决）：各臂相对 student 的 U（判断是正迁移还是仅相对更优）、
episode-level SE（仅作评价可靠性诊断，**不得**代替 learner-seed 不确定性，教训 M16）。
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
EVAL_DIR = REPO / "docs/data/slide_bac_gate_v1/source_free_eval"
OUT = REPO / "docs/data/slide_bac_gate_v1/slide_bac_gate_v1_results.json"
SOURCES = ["stand", "walk", "run"]
SEEDS = [1, 2, 3]
T90_DF2 = 2.919986        # two-sided 90% t quantile, df=2
COMPAT_N = 32

# 事前冻结的预测（写死在此，防止事后重述）
PREREG = {
    "NET": {"stand": 0.0129, "walk": 0.5153, "run": 0.5086},
    "zero_shot_return": {"stand": 88.5, "walk": 45.7, "run": 27.8},
    "NET_rank": ["walk", "run", "stand"],
    "return_rank": ["stand", "walk", "run"],
}


def load(arm: str, seed: int) -> dict:
    d = json.loads((EVAL_DIR / f"{arm}_s{seed}_step20000.json").read_text())
    rets = [float(e["return"]) for e in d["episodes"]]
    n = len(rets)
    sub = rets[:COMPAT_N]
    return {
        "primary_mean": sum(rets) / n,
        "primary_n": n,
        "primary_se": (st.pstdev(rets) / math.sqrt(n)) if n > 1 else None,
        "primary_median": st.median(rets),
        "secondary_mean": sum(sub) / len(sub),
        "eval_seeds": d["protocol"]["eval_seeds"],
    }


def interval(values: list[float]) -> dict:
    n = len(values)
    mean = sum(values) / n
    sd = math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1))
    half = T90_DF2 * sd / math.sqrt(n)
    return dict(mean=mean, sd=sd, lo=mean - half, hi=mean + half, n=n)


def classify(iv: dict) -> str:
    if iv["lo"] > 0:
        return "helpful"
    if iv["hi"] < 0:
        return "harmful"
    return "uncertain"


def treatment_audit(arm: str, seed: int) -> dict:
    import torch

    hits = sorted(glob.glob(str(REPO / f"models/*slide_bac_{arm}_s{seed}*_20000.pt")))
    if not hits:
        return {"checkpoint": None}
    state = torch.load(hits[0], map_location="cpu", weights_only=False)
    audit = state.get("admission_audit") or {}
    ex, cr = audit.get("execution_counts"), audit.get("critic_sample_counts")
    out = {"checkpoint": os.path.basename(hits[0]), "global_step": state.get("global_step")}
    if ex and sum(ex) > 0:
        out["behavior_source_share"] = sum(ex[:-1]) / sum(ex)
    if cr and sum(cr) > 0:
        out["critic_source_share"] = sum(cr[:-1]) / sum(cr)
    out["active_buffer_counts"] = audit.get("active_buffer_counts")
    out["sampling_phase"] = audit.get("sampling_phase")
    return out


def main() -> None:
    student = {s: load("student", s) for s in SEEDS}
    per = {src: {s: load(src, s) for s in SEEDS} for src in SOURCES}

    # 主判据：相对 stand 的配对差（student 项抵消）
    d_walk = [per["walk"][s]["primary_mean"] - per["stand"][s]["primary_mean"] for s in SEEDS]
    d_run = [per["run"][s]["primary_mean"] - per["stand"][s]["primary_mean"] for s in SEEDS]
    iv_walk, iv_run = interval(d_walk), interval(d_run)

    # 次级：相对 student 的 U
    U = {src: interval([per[src][s]["primary_mean"] - student[s]["primary_mean"]
                        for s in SEEDS]) for src in SOURCES}

    means = {src: st.mean([per[src][s]["primary_mean"] for s in SEEDS]) for src in SOURCES}
    observed_rank = [k for k, _ in sorted(means.items(), key=lambda kv: -kv[1])]
    stand_is_worst = means["stand"] == min(means.values())

    # 与预注册 §5 逐条对应：
    #   SUPPORTED = walk 与 run 都显著高于 stand
    #   PARTIAL   = stand 为三者最差但区间跨零，或 walk/run 仅一者显著高于 stand
    #   REFUTED   = 其余（即 stand 不低于 walk/run 中任何一个，且无一显著）
    sig_walk, sig_run = iv_walk["lo"] > 0, iv_run["lo"] > 0
    if sig_walk and sig_run:
        verdict = "BAC_SUPPORTED"
    elif stand_is_worst or sig_walk or sig_run:
        verdict = "BAC_PARTIAL"
    else:
        verdict = "BAC_REFUTED"

    report = dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        experiment="slide_bac_gate_v1",
        positioning="forward test of Bottleneck-Aligned Coverage; predictions frozen "
                    "in docs/experiments/bottleneck_aligned_coverage_v1_prereg_20260728.md "
                    "before any slide arm was trained",
        prereg=PREREG,
        label="U_i(t=10k,K=10k) = J_sf@20k(source) - J_sf@20k(student), paired by seed",
        primary="D_x = J(x) - J(stand), paired by learner seed, for x in {walk, run}",
        evaluator="frozen source-free panel, 16 eval seeds x 8 ranks = 128 episodes",
        interval_rule="90% t interval over 3 paired LEARNER seeds (df=2); "
                      "episode-level SE diagnostic only",
        per_seed={str(s): {
            "J_student": student[s]["primary_mean"],
            **{f"J_{src}": per[src][s]["primary_mean"] for src in SOURCES},
            "D_walk": per["walk"][s]["primary_mean"] - per["stand"][s]["primary_mean"],
            "D_run": per["run"][s]["primary_mean"] - per["stand"][s]["primary_mean"],
            "episode_se": {"student": student[s]["primary_se"],
                           **{src: per[src][s]["primary_se"] for src in SOURCES}},
        } for s in SEEDS},
        D_walk=dict(iv_walk, significant=sig_walk),
        D_run=dict(iv_run, significant=sig_run),
        U_vs_student={src: dict(U[src], classification=classify(U[src])) for src in SOURCES},
        arm_means=means,
        observed_rank=observed_rank,
        rank_matches_NET=observed_rank == PREREG["NET_rank"],
        rank_matches_return=observed_rank == PREREG["return_rank"],
        stand_is_worst=stand_is_worst,
        treatment_audit={src: {str(s): treatment_audit(src, s) for s in SEEDS}
                         for src in SOURCES},
        verdict=verdict,
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=1, ensure_ascii=False))

    print("Slide BAC 判决场（128-ep 面板, 3 learner seeds, 90% t 区间）\n")
    print(f"{'seed':5s} {'student':>9s} {'stand':>9s} {'walk':>9s} {'run':>9s} "
          f"{'D_walk':>9s} {'D_run':>9s}")
    for s in SEEDS:
        r = report["per_seed"][str(s)]
        print(f"{s:<5d} {r['J_student']:9.2f} {r['J_stand']:9.2f} {r['J_walk']:9.2f} "
              f"{r['J_run']:9.2f} {r['D_walk']:+9.2f} {r['D_run']:+9.2f}")
    print()
    print(f"主判据 D_walk = J(walk)-J(stand)  mean={iv_walk['mean']:+8.2f}  "
          f"90%CI=[{iv_walk['lo']:+8.2f}, {iv_walk['hi']:+8.2f}]  显著={sig_walk}")
    print(f"主判据 D_run  = J(run)-J(stand)   mean={iv_run['mean']:+8.2f}  "
          f"90%CI=[{iv_run['lo']:+8.2f}, {iv_run['hi']:+8.2f}]  显著={sig_run}")
    print()
    print("次级 U（相对 student）:")
    for src in SOURCES:
        iv = U[src]
        print(f"   {src:6s} mean={iv['mean']:+8.2f}  90%CI=[{iv['lo']:+8.2f}, "
              f"{iv['hi']:+8.2f}]  {classify(iv)}")
    print()
    print(f"实测排序   {'>'.join(observed_rank)}")
    print(f"NET 预测   {'>'.join(PREREG['NET_rank'])}   命中={report['rank_matches_NET']}")
    print(f"return 预测 {'>'.join(PREREG['return_rank'])}   命中={report['rank_matches_return']}")
    print(f"\nVERDICT: {verdict}\nsaved: {OUT}")


if __name__ == "__main__":
    main()
