"""RACING_K v1 裁决：自动选源的最小测量代价 K*。

判据冻结于 docs/experiments/racing_min_horizon_v1_prereg_20260730.md §5 (6776c03)，
**先于任何臂被评估**。本脚本只实现，不得事后调整。

    U_i(K) = J_sf(源臂 i at K) − J_sf(student-only 臂 at K)      per-seed 配对
    K*     = 最小的 K ∈ {2000, 5000, 10000} 使 3/3 seed 满足 argmax_i U_i(K) = run

    K* ≤ 5000   -> RACING_CHEAP
    K* = 10000  -> RACING_VIABLE
    都做不到    -> RACING_REFUTED（自动选源关闭）

次判据（不参与主裁决）：U_walk(K) > U_stand(K) 的 seed 数 ≥ 2/3，
即 racing 排出了 zero-shot 排反的那一对（zero-shot: run>stand>walk；真实: run>walk>stand）。
"""
from __future__ import annotations

import json
import statistics as st
from pathlib import Path

# 仅路径可通过命令行覆盖；判据逻辑不受影响（冻结于预注册 §5）。
import sys as _sys
_ROOT = Path(_sys.argv[1]) if len(_sys.argv) > 1 else Path("docs/data/racing_min_horizon_v1")
EVAL = _ROOT / "source_free_eval"
OUT = _ROOT / "results.json"
SEEDS = (1, 2, 3)
KS = (2000, 5000, 10000)
SOURCES = ("run", "walk", "stand")
TRUE_TOP1 = "run"

# ground truth（EQD30K，K=30000，endpoint）。stand 仅 single_seed，见预注册 §3。
GT_U_K30K = {"run": 379.659, "walk": 104.885, "stand": 51.278}
# 今日 zero-shot 探针（32 ep，确定性），用于辨别 racing 是否只是行为排序
ZERO_SHOT = {"run": 169.21, "stand": 146.94, "walk": 96.35}


def ret(arm: str, seed: int, k: int) -> float | None:
    f = EVAL / f"{arm}_s{seed}_step{k}.json"
    if not f.exists():
        return None
    return float(json.loads(f.read_text())["aggregate"]["return_mean"])


def main() -> None:
    res: dict = {
        "prereg": "docs/experiments/racing_min_horizon_v1_prereg_20260730.md",
        "prereg_commit": "6776c03",
        "ground_truth_U_K30000": GT_U_K30K,
        "zero_shot_reference": ZERO_SHOT,
        "per_K": {},
    }

    # 先独立扫描全部 (arm, seed, K) 组合，保证缺失报告完整
    # （不能在 student 缺失时 continue，否则源臂的缺失不会被统计到）
    missing = [f"{arm}_s{sd}_step{k}"
               for k in KS for sd in SEEDS for arm in ("student",) + SOURCES
               if ret(arm, sd, k) is None]

    U: dict[int, dict[str, dict[int, float]]] = {}
    for k in KS:
        U[k] = {s: {} for s in SOURCES}
        for sd in SEEDS:
            base = ret("student", sd, k)
            if base is None:
                continue
            for s in SOURCES:
                r = ret(s, sd, k)
                if r is not None:
                    U[k][s][sd] = r - base
    # 数据不全时必须报 INCOMPLETE 而非 REFUTED——
    # 否则"评估没跑完"会被误判成"方向被否定"。
    if missing:
        print(f"[INCOMPLETE] 缺失 {len(missing)}/{len(KS)*len(SEEDS)*(len(SOURCES)+1)} 个评估点")
        print(f"  {missing[:10]}{' ...' if len(missing) > 10 else ''}")
        res["verdict"] = "INCOMPLETE"
        res["missing"] = missing
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(res, indent=2, ensure_ascii=False))
        print(f"\n评估未完成，不予裁决。written {OUT}")
        raise SystemExit(2)

    k_star = None
    for k in KS:
        rows, top1_hits, walk_gt_stand = [], 0, 0
        for sd in SEEDS:
            if not all(sd in U[k][s] for s in SOURCES):
                continue
            per = {s: U[k][s][sd] for s in SOURCES}
            pick = max(per, key=per.get)
            hit = pick == TRUE_TOP1
            top1_hits += int(hit)
            walk_gt_stand += int(per["walk"] > per["stand"])
            rows.append({"seed": sd, "U": {s: round(per[s], 3) for s in SOURCES},
                         "argmax": pick, "top1_hit": hit,
                         "walk_gt_stand": per["walk"] > per["stand"]})
        n = len(rows)
        mean_U = {s: round(st.mean([U[k][s][sd] for sd in SEEDS if sd in U[k][s]]), 3)
                  if any(sd in U[k][s] for sd in SEEDS) else None for s in SOURCES}
        passed = n == len(SEEDS) and top1_hits == len(SEEDS)
        res["per_K"][str(k)] = {
            "per_seed": rows, "mean_U": mean_U,
            "top1_hits": f"{top1_hits}/{n}", "top1_pass": passed,
            "walk_gt_stand": f"{walk_gt_stand}/{n}",
            "secondary_pass": n > 0 and walk_gt_stand >= 2,
        }
        if passed and k_star is None:
            k_star = k

    if k_star is None:
        verdict = "RACING_REFUTED"
    elif k_star <= 5000:
        verdict = "RACING_CHEAP"
    else:
        verdict = "RACING_VIABLE"
    res["K_star"] = k_star
    res["verdict"] = verdict

    # 成本—收益（公式冻结于预注册 §5），仅在非 REFUTED 时有意义
    if k_star is not None:
        cost = len(SOURCES) * k_star
        sp = Path("docs/data/hurdle_speedup_v1/hurdle_speedup_v1_results.json")
        benefit = None
        if sp.exists():
            th = json.loads(sp.read_text())["thresholds"]["300"]["per_seed"]
            benefit = st.median([r["steps_scratch"] - r["steps_source"] for r in th])
        res["cost_benefit"] = {
            "racing_cost_steps": cost,
            "source_selection_benefit_steps_theta300": benefit,
            "net_steps": (benefit - cost) if benefit is not None else None,
            "note": "收益取自 hurdle_speedup_v1 θ=300 的 per-seed 中位数差",
        }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2, ensure_ascii=False))

    print("=" * 84)
    print(f"VERDICT: {verdict}    K* = {k_star}")
    print("=" * 84)
    print(f"\nground truth (EQD30K, K=30000): "
          + "  ".join(f"{s}={GT_U_K30K[s]:+.1f}" for s in SOURCES))
    print(f"zero-shot 行为参照            : "
          + "  ".join(f"{s}={ZERO_SHOT[s]:.1f}" for s in SOURCES))
    for k in KS:
        d = res["per_K"][str(k)]
        print(f"\n--- K={k} ---   top-1 命中 {d['top1_hits']}   "
              f"{'PASS' if d['top1_pass'] else '----'}    "
              f"walk>stand {d['walk_gt_stand']}")
        print("    mean U: " + "  ".join(
            f"{s}={d['mean_U'][s]:+.1f}" if d['mean_U'][s] is not None else f"{s}=NA"
            for s in SOURCES))
        for r in d["per_seed"]:
            print(f"    s{r['seed']}: " + "  ".join(f"{s}={r['U'][s]:+8.2f}" for s in SOURCES)
                  + f"   argmax={r['argmax']:5s} {'HIT' if r['top1_hit'] else 'MISS'}")
    if res.get("cost_benefit"):
        c = res["cost_benefit"]
        print(f"\n成本-收益: racing 成本={c['racing_cost_steps']} 步  "
              f"选源收益={c['source_selection_benefit_steps_theta300']} 步  "
              f"净={c['net_steps']} 步")
    print(f"\nwritten {OUT}")


if __name__ == "__main__":
    main()
