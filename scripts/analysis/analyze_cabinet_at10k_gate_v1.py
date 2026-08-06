"""Cabinet@10k gate 裁决分析(预注册,揭盲前定稿,只读)。

标签(PI 冻结):
    U_i(10k, 10k) = J_sf@20k(source_i arm) - J_sf@20k(student arm)   同 seed 配对
J_sf = 冻结 source-free evaluator(4 eval seeds × 8 ranks = 32 episodes)的 return 均值。

分类(3 个配对 seed 的 90% 区间,t 分布 df=2):
    区间完全 > 0 → helpful;完全 < 0 → harmful;跨 0 → uncertain。

三级裁决:
    至少一个 helpful 且至少一个 harmful → CABINET_HETEROGENEITY_PASS
    所有可判定 source 同号                → CABINET_ALL_SAME_SIGN
    否则                                  → CABINET_UNCERTAIN

后两者一律停止本轮指标建模,不加任务/seed、不改阈值。
"""
from __future__ import annotations

import glob
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
EVAL_DIR = REPO / "docs/data/cabinet_at10k_gate_v1/source_free_eval"
OUT = REPO / "docs/data/cabinet_at10k_gate_v1/cabinet_at10k_gate_v1_results.json"
SOURCES = ["stand", "walk", "run"]
SEEDS = [1, 2, 3]
T90_DF2 = 2.919986  # two-sided 90% t quantile, df=2


def load_return(arm: str, seed: int) -> float:
    path = EVAL_DIR / f"{arm}_s{seed}_step20000.json"
    return float(json.loads(path.read_text())["aggregate"]["return_mean"])


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

    hits = sorted(glob.glob(str(REPO / f"models/*cabinet_at10k_{arm}_s{seed}*_20000.pt")))
    if not hits:
        return {"checkpoint": None}
    state = torch.load(hits[0], map_location="cpu", weights_only=False)
    audit = state.get("admission_audit") or {}
    exec_counts = audit.get("execution_counts")
    critic_counts = audit.get("critic_sample_counts")
    out = {"checkpoint": os.path.basename(hits[0]), "global_step": state.get("global_step")}
    if exec_counts and sum(exec_counts) > 0:
        out["behavior_source_share"] = sum(exec_counts[:-1]) / sum(exec_counts)
    if critic_counts and sum(critic_counts) > 0:
        out["critic_source_share"] = sum(critic_counts[:-1]) / sum(critic_counts)
    out["active_buffer_counts"] = audit.get("active_buffer_counts")
    out["sampling_phase"] = audit.get("sampling_phase")
    return out


def main() -> None:
    report = dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        experiment="cabinet_at10k_gate_v1",
        label="U_i(t=10k,K=10k) = J_sf@20k(source) - J_sf@20k(student), paired by seed",
        evaluator="frozen source-free panel, 4 eval seeds x 8 ranks = 32 episodes",
        interval_rule="90% t interval over 3 paired seeds (df=2)",
        arms={},
        sources={},
    )
    student = {s: load_return("student", s) for s in SEEDS}
    report["arms"]["student"] = student
    verdict_labels = {}
    for src in SOURCES:
        per_seed = {s: load_return(src, s) for s in SEEDS}
        deltas = [per_seed[s] - student[s] for s in SEEDS]
        iv = interval(deltas)
        label = classify(iv)
        verdict_labels[src] = label
        report["arms"][src] = per_seed
        report["sources"][src] = dict(
            per_seed_return=per_seed,
            per_seed_U=dict(zip(map(str, SEEDS), deltas)),
            U_mean=iv["mean"], U_sd=iv["sd"], U_lo90=iv["lo"], U_hi90=iv["hi"],
            classification=label,
            treatment_audit={str(s): treatment_audit(src, s) for s in SEEDS},
        )
    labels = list(verdict_labels.values())
    decided = [l for l in labels if l != "uncertain"]
    if "helpful" in labels and "harmful" in labels:
        verdict = "CABINET_HETEROGENEITY_PASS"
    elif decided and len(set(decided)) == 1:
        verdict = "CABINET_ALL_SAME_SIGN"
    else:
        verdict = "CABINET_UNCERTAIN"
    report["classifications"] = verdict_labels
    report["verdict"] = verdict
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=1, ensure_ascii=False))

    print(f"{'source':7s} {'U_mean':>10s} {'lo90':>10s} {'hi90':>10s}  {'class':10s}  per-seed U")
    for src in SOURCES:
        r = report["sources"][src]
        pers = " ".join(f"{r['per_seed_U'][str(s)]:+.2f}" for s in SEEDS)
        print(f"{src:7s} {r['U_mean']:+10.3f} {r['U_lo90']:+10.3f} {r['U_hi90']:+10.3f}  "
              f"{r['classification']:10s}  {pers}")
    print(f"\nstudent per-seed J_sf@20k: {student}")
    print(f"VERDICT: {verdict}\nsaved: {OUT}")


if __name__ == "__main__":
    main()
