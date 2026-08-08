#!/usr/bin/env python3
"""PARE Experiment A 的判决（判据冻结于 docs/experiments/pare_gate_a_prereg_20260808.md）。

本脚本在**看到任何评估结果之前**写就并提交。之后只允许改路径参数。

判据（预注册 §2，逐条对齐，不换更易过的代理）：
    G1  early scaffold gain    3/3 同号正 且 t = mean/(sd/√3) > 2.92（df=2 单侧 .05）
    G2  residual headroom      mean J_exit(100k) < 500（理论上限 1000 的 50%）
    G3  非"训练量不够"          r = ΔJ(50k→100k)/ΔJ(20k→50k) < 0.5

不确定度一律用 **learner 间方差**，绝不用 episode 面板 SE（M16）。
只用 return，不用 success_count——后者在 locomotion 上读的是摔倒早停（CLAUDE.md §6）。
任一 (task, arm, seed, step) 缺失 → INCOMPLETE 且非零退出，且**独立扫描全部组合**，
不因前置缺失而 continue（CLAUDE.md §4）。

用法：python scripts/analysis/adjudicate_pare_gate_a_v1.py [--eval-root DIR] [--out FILE]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

TASKS = ("stair", "truck")
#: (臂, 评估步)。exit 的 20k 点由 scaffold run 提供——release 点二者同一状态。
NEEDED = (
    ("scaf", 20000),
    ("exit", 50000),
    ("exit", 100000),
    ("scratch", 20000),
    ("scratch", 50000),
    ("scratch", 100000),
)
SEEDS = (1, 2, 3)

T_CRIT_DF2_ONESIDED_05 = 2.92
HEADROOM_MAX = 500.0          # 理论上限 1000 的 50%
PLATEAU_RATIO_MAX = 0.5
EPS = 1e-9


def theory_max() -> float:
    """HumanoidBench：per-step reward 是若干 [0,1] 项相乘，episode 1000 步。"""
    return 1000.0


def load_return(eval_root: Path, task: str, arm: str, seed: int, step: int):
    """读一个评估点的 128-episode 平均 return。缺失返回 (None, 原因)。"""
    path = eval_root / f"{task}_{arm}_s{seed}_step{step}.json"
    if not path.exists():
        return None, f"缺文件 {path.relative_to(REPO) if path.is_relative_to(REPO) else path}"
    try:
        blob = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        return None, f"{path.name} 不是合法 JSON: {exc}"
    agg = blob.get("aggregate") or {}
    for key in ("return_mean", "mean_return", "avg_return"):
        if key in agg:
            return float(agg[key]), None
        if key in blob:
            return float(blob[key]), None
    return None, f"{path.name} 内无 return_mean/mean_return/avg_return 字段"


def mean_sd(xs):
    n = len(xs)
    m = sum(xs) / n
    if n < 2:
        return m, float("nan")
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    return m, math.sqrt(var)


def adjudicate_task(eval_root: Path, task: str) -> dict:
    # ── 独立扫描全部组合，不因前置缺失而跳过后续 ──────────────────
    values: dict[tuple[str, int, int], float] = {}
    missing: list[str] = []
    for arm, step in NEEDED:
        for seed in SEEDS:
            v, why = load_return(eval_root, task, arm, seed, step)
            if v is None:
                missing.append(why)
            else:
                values[(arm, step, seed)] = v

    if missing:
        return {"verdict": "INCOMPLETE", "n_missing": len(missing),
                "missing": missing[:20]}

    def col(arm, step):
        return [values[(arm, step, s)] for s in SEEDS]

    # ── G1 ────────────────────────────────────────────────────────
    # exit 臂的 20k 点 = scaffold run 的 20k checkpoint（release 点）
    deltas = [a - b for a, b in zip(col("scaf", 20000), col("scratch", 20000))]
    d_mean, d_sd = mean_sd(deltas)
    se_learner = d_sd / math.sqrt(len(SEEDS))
    t_stat = d_mean / se_learner if se_learner > EPS else float("inf")
    all_pos = all(d > 0 for d in deltas)
    if all_pos and t_stat > T_CRIT_DF2_ONESIDED_05:
        g1 = "PASS"
    elif all_pos:
        g1 = "WEAK_GAIN"
    else:
        g1 = "FAIL"

    # ── G2 ────────────────────────────────────────────────────────
    j100 = col("exit", 100000)
    j100_mean, j100_sd = mean_sd(j100)
    g2 = "PASS" if j100_mean < HEADROOM_MAX else "FAIL"

    # ── G3 ────────────────────────────────────────────────────────
    j20_mean, _ = mean_sd(col("scaf", 20000))
    j50_mean, _ = mean_sd(col("exit", 50000))
    early = j50_mean - j20_mean
    late = j100_mean - j50_mean
    if early <= EPS:
        g3, ratio = "NON_MONOTONE", None
    else:
        ratio = late / early
        g3 = "PASS" if ratio < PLATEAU_RATIO_MAX else "STILL_IMPROVING"

    # ── 任务裁决（预注册 §3 表，顺序即优先级）──────────────────────
    if g1 == "FAIL":
        verdict = "NO_SCAFFOLD_EFFECT"
    elif g2 == "FAIL":
        verdict = "SATURATED"
    elif g3 in ("STILL_IMPROVING", "NON_MONOTONE"):
        verdict = "CONFOUNDED_BY_BUDGET"
    elif g1 == "WEAK_GAIN":
        verdict = "WEAK_CANDIDATE"
    else:
        verdict = "PARE_CANDIDATE"

    return {
        "verdict": verdict,
        "G1": {"result": g1, "deltas_per_seed": [round(d, 3) for d in deltas],
               "mean": round(d_mean, 3), "sd_learner": round(d_sd, 3),
               "se_learner": round(se_learner, 3), "t": round(t_stat, 3),
               "t_crit": T_CRIT_DF2_ONESIDED_05,
               "note": "learner 间方差，非 episode 面板 SE"},
        "G2": {"result": g2, "J_exit_100k_mean": round(j100_mean, 3),
               "J_exit_100k_sd": round(j100_sd, 3),
               "per_seed": [round(x, 3) for x in j100],
               "threshold": HEADROOM_MAX,
               "theory_max": theory_max(),
               "frac_of_theory_max": round(j100_mean / theory_max(), 4)},
        "G3": {"result": g3,
               "J_20k": round(j20_mean, 3), "J_50k": round(j50_mean, 3),
               "J_100k": round(j100_mean, 3),
               "delta_early": round(early, 3), "delta_late": round(late, 3),
               "ratio": None if ratio is None else round(ratio, 4),
               "threshold": PLATEAU_RATIO_MAX},
        "reference": {
            "J_scratch_100k_mean": round(mean_sd(col("scratch", 100000))[0], 3),
            "note": "仅供参照，不参与 G1–G3 判定",
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-root", default="docs/data/pare_gate_a_v1/source_free_eval")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    eval_root = (REPO / args.eval_root) if not Path(args.eval_root).is_absolute() \
        else Path(args.eval_root)

    per_task = {t: adjudicate_task(eval_root, t) for t in TASKS}

    candidates = [t for t, r in per_task.items()
                  if r["verdict"] in ("PARE_CANDIDATE", "WEAK_CANDIDATE")]
    incomplete = [t for t, r in per_task.items() if r["verdict"] == "INCOMPLETE"]

    if incomplete:
        # 缺失绝不落进实质裁决分支——"没跑完"不是"方向被否定"。
        overall = "INCOMPLETE"
        development, holdout = None, None
    elif not candidates:
        overall = "F1_TRIGGERED_CLOSE_PARE"
        development, holdout = None, None
    else:
        overall = "PROCEED_TO_PARE"
        if len(candidates) == 1:
            development, holdout = candidates[0], None
        else:
            development = max(candidates, key=lambda t: per_task[t]["G1"]["t"])
            holdout = [t for t in candidates if t != development][0]

    report = {
        "verdict": overall,
        "prereg": "docs/experiments/pare_gate_a_prereg_20260808.md",
        "per_task": per_task,
        "development_task": development,
        "holdout_task": holdout,
        "scope": "只回答『有没有值得 PARE 解决的 residual problem』，不是论文性能结果",
    }
    text = json.dumps(report, indent=2, ensure_ascii=False)
    print(text)
    if args.out:
        out = (REPO / args.out) if not Path(args.out).is_absolute() else Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n")
    return 0 if overall in ("PROCEED_TO_PARE", "F1_TRIGGERED_CLOSE_PARE") else 1


if __name__ == "__main__":
    sys.exit(main())
