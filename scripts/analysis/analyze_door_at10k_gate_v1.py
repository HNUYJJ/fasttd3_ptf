"""Door@10k gate 裁决分析(预注册,揭盲前定稿,只读)。

标签(PI 冻结):
    U_i(10k, 10k) = J_sf@20k(source_i arm) - J_sf@20k(student arm)   同 seed 配对

评估面板(PI 冻结,训练前决定):
    primary   = 128 deterministic source-free episodes(16 eval seeds × 8 ranks)
    secondary = 前 32 个 episode(4 eval seeds × 8 ranks),与 Cabinet gate 及既往
                所有 32-episode 面板逐位兼容,仅用于可比性说明,**不参与裁决**。

分类(3 个配对 seed 的 90% 区间,t 分布 df=2,与 Cabinet gate 相同):
    区间完全 > 0 → helpful;完全 < 0 → harmful;跨 0 → uncertain。

三级裁决:
    至少一个 helpful 且至少一个 harmful → DOOR_HETEROGENEITY_PASS
    所有可判定 source 同号                → DOOR_ALL_SAME_SIGN
    否则                                  → DOOR_UNCERTAIN

后两者一律停止本轮指标建模:不加任务、不加 seed、不延长 K、不换指标、不改阈值。

历史定位(不影响裁决,仅供报告):Transfer Map v1 已记录 door 的**行为层**异质性
(run 101 vs zero 64 = +56%;walk 25 且 62% 摔;stand 59)。本实验测的是
**学习效用**层,两者不可混同——行为即时效果不等于后续学习价值。
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
EVAL_DIR = REPO / "docs/data/door_at10k_gate_v1/source_free_eval"
OUT = REPO / "docs/data/door_at10k_gate_v1/door_at10k_gate_v1_results.json"
SOURCES = ["stand", "walk", "run"]
SEEDS = [1, 2, 3]
T90_DF2 = 2.919986        # two-sided 90% t quantile, df=2
COMPAT_N = 32             # secondary 子面板大小


def load(arm: str, seed: int) -> dict:
    d = json.loads((EVAL_DIR / f"{arm}_s{seed}_step20000.json").read_text())
    eps = d["episodes"]
    rets = [float(e["return"]) for e in eps]
    n = len(rets)
    sub = rets[:COMPAT_N]
    return {
        "primary_mean": sum(rets) / n,
        "primary_n": n,
        "primary_se": (st.pstdev(rets) / math.sqrt(n)) if n > 1 else None,
        "primary_median": st.median(rets),
        "secondary_mean": sum(sub) / len(sub),
        "secondary_n": len(sub),
        "eval_seeds": d["protocol"]["eval_seeds"],
    }


def interval(values: list[float]) -> dict:
    n = len(values)
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    sd = math.sqrt(var)
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

    hits = sorted(glob.glob(str(REPO / f"models/*door_at10k_{arm}_s{seed}*_20000.pt")))
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
    report = dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        experiment="door_at10k_gate_v1",
        positioning=("directed RBO learning-utility calibration with a known behavioural "
                     "prior (Transfer Map v1: run +56%, walk negative, stand ~flat); "
                     "NOT a blind or external validation"),
        label="U_i(t=10k,K=10k) = J_sf@20k(source) - J_sf@20k(student), paired by seed",
        evaluator="frozen source-free panel, 16 eval seeds x 8 ranks = 128 episodes (primary)",
        secondary_panel=f"first {COMPAT_N} episodes = legacy 4x8 panel, reported for "
                        f"comparability only, not used for adjudication",
        interval_rule="90% t interval over 3 paired seeds (df=2)",
        arms={}, sources={},
    )
    student = {s: load("student", s) for s in SEEDS}
    report["arms"]["student"] = {str(s): v for s, v in student.items()}
    labels = {}
    for src in SOURCES:
        per = {s: load(src, s) for s in SEEDS}
        d_pri = [per[s]["primary_mean"] - student[s]["primary_mean"] for s in SEEDS]
        d_sec = [per[s]["secondary_mean"] - student[s]["secondary_mean"] for s in SEEDS]
        iv, iv_s = interval(d_pri), interval(d_sec)
        lab = classify(iv)
        labels[src] = lab
        report["arms"][src] = {str(s): v for s, v in per.items()}
        report["sources"][src] = dict(
            per_seed_U=dict(zip(map(str, SEEDS), d_pri)),
            U_mean=iv["mean"], U_sd=iv["sd"], U_lo90=iv["lo"], U_hi90=iv["hi"],
            classification=lab,
            secondary_per_seed_U=dict(zip(map(str, SEEDS), d_sec)),
            secondary_U_mean=iv_s["mean"], secondary_U_lo90=iv_s["lo"],
            secondary_U_hi90=iv_s["hi"], secondary_classification=classify(iv_s),
            treatment_audit={str(s): treatment_audit(src, s) for s in SEEDS},
        )
    vals = list(labels.values())
    decided = [x for x in vals if x != "uncertain"]
    if "helpful" in vals and "harmful" in vals:
        verdict = "DOOR_HETEROGENEITY_PASS"
    elif decided and len(set(decided)) == 1:
        verdict = "DOOR_ALL_SAME_SIGN"
    else:
        verdict = "DOOR_UNCERTAIN"
    report["classifications"] = labels
    report["verdict"] = verdict
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=1, ensure_ascii=False))

    print("PRIMARY panel (128 episodes) — 裁决依据")
    print(f"{'source':7s} {'U_mean':>10s} {'lo90':>10s} {'hi90':>10s}  {'class':10s}  per-seed U")
    for src in SOURCES:
        r = report["sources"][src]
        pers = " ".join(f"{r['per_seed_U'][str(s)]:+.2f}" for s in SEEDS)
        print(f"{src:7s} {r['U_mean']:+10.3f} {r['U_lo90']:+10.3f} {r['U_hi90']:+10.3f}  "
              f"{r['classification']:10s}  {pers}")
    print(f"\nSECONDARY 子面板 (前 {COMPAT_N} episodes,仅可比性,不裁决)")
    for src in SOURCES:
        r = report["sources"][src]
        print(f"{src:7s} {r['secondary_U_mean']:+10.3f} {r['secondary_U_lo90']:+10.3f} "
              f"{r['secondary_U_hi90']:+10.3f}  {r['secondary_classification']}")
    print(f"\nstudent per-seed J_sf@20k (primary): "
          f"{ {s: round(student[s]['primary_mean'], 2) for s in SEEDS} }")
    print(f"VERDICT: {verdict}\nsaved: {OUT}")


if __name__ == "__main__":
    main()
