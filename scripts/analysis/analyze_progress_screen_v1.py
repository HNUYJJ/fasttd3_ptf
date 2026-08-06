"""progress 粗筛裁决（预注册 docs/experiments/progress_screen_v1_prereg_20260804.md §5）。

判据在见到任何探针数据之前冻结：

    design data   crawl（期望全拒）+ hurdle（期望不拒 run）
    holdout       slide（不参与阈值设定）

    lo = max_i P(i, crawl)        hi = P(run, hurdle)        θ = sqrt(lo × hi)

    lo <  hi 且 P(walk, slide) >  θ   → PROGRESS_SCREEN_VIABLE
    lo >= hi                          → SEPARATION_FAILED
    lo <  hi 且 P(walk, slide) <= θ   → HOLDOUT_FAILED
    任一组合缺失                       → INCOMPLETE（非零退出）

数值保护（见数据前写定）：位移测量恒 >= 0（max_dx 初值 0），lo=0 时几何平均
退化，故取 lo_eff = max(lo, 1e-6)。这只影响 θ 的数值，不改变任何分支条件。
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

DATA = Path("docs/data/progress_screen_v1")
PROBE = DATA / "probe.json"
OUT = DATA / "results.json"

CRAWL = "h1hand-crawl-v0"
HURDLE = "h1hand-hurdle-v0"
SLIDE = "h1hand-slide-v0"
WALKENV = "h1hand-walk-v0"
SRCS = ("stand", "walk", "run")
REQUIRED = [f"{e}|{s}" for e in (CRAWL, HURDLE, SLIDE, WALKENV) for s in SRCS]


def main() -> int:
    if not PROBE.exists():
        print(f"VERDICT: INCOMPLETE  (missing probe file {PROBE})")
        return 2
    blob = json.loads(PROBE.read_text())
    res = blob.get("results", {})

    # §4: 独立扫描全部组合,不在前置项缺失时提前返回
    missing = [k for k in REQUIRED if k not in res
               or res[k].get("progress_max_dx_mean") is None]
    if missing:
        print(f"VERDICT: INCOMPLETE  (missing {len(missing)}/{len(REQUIRED)} combos)")
        for k in missing:
            print(f"  missing: {k}")
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(
            {"verdict": "INCOMPLETE", "missing": missing}, indent=2, ensure_ascii=False))
        return 2

    def P(env: str, src: str) -> float:
        return float(res[f"{env}|{src}"]["progress_max_dx_mean"])

    def SE(env: str, src: str) -> float:
        return float(res[f"{env}|{src}"]["progress_max_dx_se"])

    crawl_p = {s: P(CRAWL, s) for s in SRCS}
    lo_src = max(crawl_p, key=crawl_p.get)
    lo = crawl_p[lo_src]
    hi = P(HURDLE, "run")
    theta = math.sqrt(max(lo, 1e-6) * hi) if hi > 0 else 0.0
    slide_walk = P(SLIDE, "walk")

    if lo >= hi:
        verdict = "SEPARATION_FAILED"
    elif slide_walk > theta:
        verdict = "PROGRESS_SCREEN_VIABLE"
    else:
        verdict = "HOLDOUT_FAILED"

    report = {
        "prereg": "docs/experiments/progress_screen_v1_prereg_20260804.md",
        "design": {
            "crawl_progress": {s: {"mean": P(CRAWL, s), "se": SE(CRAWL, s)} for s in SRCS},
            "lo_source": lo_src, "lo": lo,
            "hurdle_run": {"mean": hi, "se": SE(HURDLE, "run")}, "hi": hi,
            "theta": theta,
        },
        "holdout": {
            "slide_walk": {"mean": slide_walk, "se": SE(SLIDE, "walk")},
            "passes_theta": bool(slide_walk > theta),
        },
        "reference_only_not_in_criterion": {
            "slide_all": {s: P(SLIDE, s) for s in SRCS},
            "hurdle_all": {s: P(HURDLE, s) for s in SRCS},
            "flat_walk_env": {s: P(WALKENV, s) for s in SRCS},
            "return_mean": {k: res[k]["return_mean"] for k in REQUIRED},
        },
        "verdict": verdict,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    print("=" * 78)
    print(f"VERDICT: {verdict}")
    print("=" * 78)
    print("design data")
    for s in SRCS:
        print(f"  crawl  {s:5s}  dx={P(CRAWL,s):7.3f} ± {SE(CRAWL,s):.3f}")
    print(f"  lo = max_i P(i,crawl) = {lo:.3f}   (source={lo_src})")
    print(f"  hi = P(run,hurdle)    = {hi:.3f} ± {SE(HURDLE,'run'):.3f}")
    print(f"  theta = sqrt(lo*hi)   = {theta:.3f}")
    print("holdout")
    print(f"  P(walk,slide) = {slide_walk:.3f} ± {SE(SLIDE,'walk'):.3f}   "
          f"{'>' if slide_walk > theta else '<='} theta")
    print("reference (not in criterion)")
    for e in (HURDLE, SLIDE, WALKENV):
        row = "  ".join(f"{s}={P(e,s):7.3f}" for s in SRCS)
        print(f"  {e:20s} {row}")
    print(f"\nwritten {OUT}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:                      # §4: 绝不让异常被读成实质裁决
        print(f"VERDICT: INCOMPLETE  (unhandled error: {type(exc).__name__}: {exc})")
        raise
