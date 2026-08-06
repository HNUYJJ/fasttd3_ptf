"""hurdle 样本效率加速倍数裁决。

判据冻结于 docs/experiments/hurdle_speedup_v1_prereg_20260730.md §2/§4 (ba7a7de),
**先于任何长程臂被评估**。本脚本只实现,不得事后调整。

    speedup(θ) = steps_scratch(θ) / steps_source(θ)
    steps_X(θ) = 臂 X 的 source-free 回报首次 ≥ θ 的步数(相邻评估点线性插值)
    θ ∈ {200, 300, 400}   ← 由已公开的 r@end=597.5 冻结,与本实验结果无关

    CONFIRMED: ≥2/3 阈值上 speedup ≥ 2.0,且这些阈值上 3/3 seed 的 per-seed
               speedup ≥ 1.5
    REFUTED  : 全部阈值 speedup < 1.5,或 source 臂在 100k 被 scratch 反超
    否则      : PARTIAL
"""
from __future__ import annotations

import json
import statistics as st
from pathlib import Path

EVAL = Path("docs/data/hurdle_speedup_v1/source_free_eval")
OUT = Path("docs/data/hurdle_speedup_v1/hurdle_speedup_v1_results.json")
SEEDS = (1, 2, 3)
STEPS = (10000, 20000, 30000, 50000, 75000, 100000)
THRESHOLDS = (200.0, 300.0, 400.0)
CENSOR = float(STEPS[-1])


def curve(arm: str, seed: int) -> list[tuple[int, float]]:
    pts = []
    for s in STEPS:
        f = EVAL / f"{arm}_s{seed}_step{s}.json"
        if f.exists():
            pts.append((s, float(json.loads(f.read_text())["aggregate"]["return_mean"])))
    return pts


def steps_to(pts: list[tuple[int, float]], theta: float) -> tuple[float, bool]:
    """首次 ≥ θ 的步数;线性插值。返回 (steps, censored)。"""
    prev = None
    for s, r in pts:
        if r >= theta:
            if prev is None:
                return float(s), False
            (s0, r0) = prev
            if r == r0:
                return float(s), False
            frac = (theta - r0) / (r - r0)
            return s0 + frac * (s - s0), False
        prev = (s, r)
    return CENSOR, True  # 右删失:全程未达到


def main() -> None:
    curves = {a: {sd: curve(a, sd) for sd in SEEDS} for a in ("scratch", "source")}
    res: dict = {
        "prereg": "docs/experiments/hurdle_speedup_v1_prereg_20260730.md",
        "prereg_commit": "ba7a7de",
        "curves": {a: {str(sd): curves[a][sd] for sd in SEEDS} for a in curves},
        "thresholds": {},
    }

    per_threshold_pass = []
    for th in THRESHOLDS:
        rows, per_seed_sp = [], []
        for sd in SEEDS:
            sc, sc_c = steps_to(curves["scratch"][sd], th)
            so, so_c = steps_to(curves["source"][sd], th)
            sp = sc / so if so > 0 else float("nan")
            per_seed_sp.append(sp)
            rows.append({"seed": sd, "steps_scratch": round(sc), "steps_source": round(so),
                         "speedup": round(sp, 3),
                         "censored": {"scratch": sc_c, "source": so_c}})
        # 汇总 speedup 用中位数(对右删失更稳健),同时报均值
        med = st.median(per_seed_sp)
        ok = med >= 2.0 and all(x >= 1.5 for x in per_seed_sp)
        any_censored = any(r["censored"]["scratch"] or r["censored"]["source"] for r in rows)
        res["thresholds"][str(int(th))] = {
            "per_seed": rows,
            "speedup_median": round(med, 3),
            "speedup_mean": round(st.mean(per_seed_sp), 3),
            "all_seeds_ge_1.5": all(x >= 1.5 for x in per_seed_sp),
            "pass": ok,
            "has_censored": any_censored,
        }
        per_threshold_pass.append(ok)

    # 反超检查:100k 终点
    end = {}
    for a in ("scratch", "source"):
        vals = [r for sd in SEEDS for (s, r) in curves[a][sd] if s == STEPS[-1]]
        end[a] = round(st.mean(vals), 2) if vals else None
    overtaken = (end["scratch"] is not None and end["source"] is not None
                 and end["scratch"] > end["source"])
    res["endpoint_100k"] = end
    res["scratch_overtook_source"] = overtaken

    n_pass = sum(per_threshold_pass)
    all_low = all(res["thresholds"][str(int(t))]["speedup_median"] < 1.5 for t in THRESHOLDS)
    if n_pass >= 2 and not overtaken:
        verdict = "SPEEDUP_CONFIRMED"
    elif all_low or overtaken:
        verdict = "SPEEDUP_REFUTED"
    else:
        verdict = "SPEEDUP_PARTIAL"
    res["verdict"] = verdict

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2, ensure_ascii=False))

    print("=" * 88)
    print(f"VERDICT: {verdict}")
    print("=" * 88)
    for a in ("scratch", "source"):
        print(f"\n[{a}] source-free student 回报曲线")
        for sd in SEEDS:
            pts = curves[a][sd]
            print(f"   s{sd}: " + "  ".join(f"{s//1000}k={r:.1f}" for s, r in pts))
    print()
    for th in THRESHOLDS:
        t = res["thresholds"][str(int(th))]
        flag = "PASS" if t["pass"] else "----"
        cen = " (含右删失)" if t["has_censored"] else ""
        print(f"θ={int(th):4d}  speedup 中位数={t['speedup_median']:.2f}  "
              f"均值={t['speedup_mean']:.2f}  per-seed={[r['speedup'] for r in t['per_seed']]}  "
              f"{flag}{cen}")
    print(f"\n100k 终点: scratch={end['scratch']}  source={end['source']}  "
          f"反超={overtaken}")
    print(f"\nwritten {OUT}")


if __name__ == "__main__":
    main()
