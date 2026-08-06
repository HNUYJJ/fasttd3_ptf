"""RACING_REJECT v1 裁决：racing 能否在全负场地正确拒绝。

判据冻结于 docs/experiments/racing_reject_door_v1_prereg_20260730.md §4/§5 (18a20cb)，
**先于任何臂被评估**。本脚本只实现，不得事后调整。

    U_i(K) = J_sf(源臂 i at 10000+K) − J_sf(student 臂 at 10000+K)    per-seed 配对
    K*_reject = 最小的 K ∈ {2000,5000,10000} 使 3/3 seed 满足 max_i U_i(K) < 0

    K*_reject ≤ 5000  -> REJECT_CHEAP
    K*_reject = 10000 -> REJECT_VIABLE
    都做不到          -> REJECT_REFUTED（racing 不提供负迁移保护）

内建 sanity check（必须先过，否则整体作废）：K=10000 的结果须复现
door_at10k_gate_v1 的已发表 ground truth——9/9 per-seed 同号（全负），
且三源 mean U 与已发表值之差在配对面板 SE 的 3 倍以内。

次判据（不参与主裁决）：argmax_i U_i(K) = walk 的 seed 数（door 上 walk 最不负）。
"""
from __future__ import annotations

import json
import math
import statistics as st
import sys
from pathlib import Path

_ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("docs/data/racing_reject_door_v1")
EVAL = _ROOT / "source_free_eval"
OUT = _ROOT / "results.json"

SEEDS = (1, 2, 3)
ANCHOR = 10000
KS = (2000, 5000, 10000)
SOURCES = ("stand", "walk", "run")
LEAST_HARMFUL = "walk"          # ground truth：door 上 walk 最不负（3/3 seed）

# 已发表 ground truth（door_at10k_gate_v1_results_20260727.md，K=10000，128 ep panel）
GT_PER_SEED = {
    "stand": (-23.43, -42.83, -31.66),
    "walk": (-7.06, -38.58, -20.96),
    "run": (-17.78, -41.04, -33.08),
}
GT_MEAN = {k: st.mean(v) for k, v in GT_PER_SEED.items()}
# 行为层排序（Transfer Map v1）：run 101 ≫ stand 59 > walk 25 —— 与学习效用完全相反
BEHAVIOR = {"run": 101.0, "stand": 59.0, "walk": 25.0}
SANITY_SE_MULT = 3.0


def _agg(arm: str, seed: int, gstep: int):
    f = EVAL / f"{arm}_s{seed}_step{gstep}.json"
    if not f.exists():
        return None
    d = json.loads(f.read_text())
    eps = [e["return"] for e in d["episodes"]]
    n = len(eps)
    return d["aggregate"]["return_mean"], st.pstdev(eps) / math.sqrt(n) if n else float("nan")


def main() -> None:
    res: dict = {
        "prereg": "docs/experiments/racing_reject_door_v1_prereg_20260730.md",
        "prereg_commit": "18a20cb",
        "ground_truth_K10000": GT_PER_SEED,
        "behavior_reference": BEHAVIOR,
        "per_K": {},
    }

    missing = [f"{a}_s{sd}_step{ANCHOR+k}"
               for k in KS for sd in SEEDS for a in ("student",) + SOURCES
               if _agg(a, sd, ANCHOR + k) is None]
    if missing:
        print(f"[INCOMPLETE] 缺失 {len(missing)}/{len(KS)*len(SEEDS)*(len(SOURCES)+1)} 个评估点")
        print(f"  {missing[:10]}{' ...' if len(missing) > 10 else ''}")
        res["verdict"] = "INCOMPLETE"; res["missing"] = missing
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(res, indent=2, ensure_ascii=False))
        print(f"\n评估未完成，不予裁决。written {OUT}")
        raise SystemExit(2)

    U, USE = {}, {}
    for k in KS:
        U[k], USE[k] = {}, {}
        for sd in SEEDS:
            bm, bse = _agg("student", sd, ANCHOR + k)
            for a in SOURCES:
                m, se = _agg(a, sd, ANCHOR + k)
                U[k][(a, sd)] = m - bm
                USE[k][(a, sd)] = math.sqrt(se ** 2 + bse ** 2)

    # ---- 内建 sanity check（K=10000 复现 ground truth）----
    k = 10000
    sanity, sane = {}, True
    for a in SOURCES:
        vals = [U[k][(a, sd)] for sd in SEEDS]
        allneg = all(v < 0 for v in vals)
        pooled_se = math.sqrt(sum(USE[k][(a, sd)] ** 2 for sd in SEEDS)) / len(SEEDS)
        dev = st.mean(vals) - GT_MEAN[a]
        ok = allneg and abs(dev) <= SANITY_SE_MULT * pooled_se
        sane = sane and ok
        sanity[a] = {"per_seed": [round(v, 2) for v in vals], "mean": round(st.mean(vals), 2),
                     "gt_mean": round(GT_MEAN[a], 2), "deviation": round(dev, 2),
                     "pooled_se": round(pooled_se, 2),
                     "tolerance": round(SANITY_SE_MULT * pooled_se, 2),
                     "all_negative": allneg, "pass": ok}
    res["sanity_check"] = {"per_source": sanity, "pass": sane,
                           "rule": "9/9 per-seed 全负 且 |mean 偏差| <= 3×pooled SE"}

    # ---- 主判据 ----
    k_star = None
    for k in KS:
        rows, reject_hits, least_hits = [], 0, 0
        for sd in SEEDS:
            per = {a: U[k][(a, sd)] for a in SOURCES}
            mx = max(per, key=per.get)
            rejected = per[mx] < 0
            reject_hits += int(rejected)
            least_hits += int(mx == LEAST_HARMFUL)
            rows.append({"seed": sd, "U": {a: round(per[a], 2) for a in SOURCES},
                         "argmax": mx, "max_U": round(per[mx], 2),
                         "reject_all": rejected, "argmax_is_least_harmful": mx == LEAST_HARMFUL})
        passed = reject_hits == len(SEEDS)
        res["per_K"][str(k)] = {
            "per_seed": rows,
            "mean_U": {a: round(st.mean([U[k][(a, sd)] for sd in SEEDS]), 2) for a in SOURCES},
            "reject_all_hits": f"{reject_hits}/{len(SEEDS)}", "reject_pass": passed,
            "argmax_is_walk": f"{least_hits}/{len(SEEDS)}",
            "secondary_pass": least_hits >= 2,
        }
        if passed and k_star is None:
            k_star = k

    if not sane:
        verdict = "VOID_SANITY_FAILED"
    elif k_star is None:
        verdict = "REJECT_REFUTED"
    elif k_star <= 5000:
        verdict = "REJECT_CHEAP"
    else:
        verdict = "REJECT_VIABLE"
    res["K_star_reject"] = k_star
    res["verdict"] = verdict
    if k_star is not None and sane:
        res["avoided_loss"] = {
            "reject_cost_steps": len(SOURCES) * k_star,
            "avoided_loss_return": round(abs(st.mean(list(GT_MEAN.values()))), 2),
            "note": "避损口径，与 hurdle 的加速收益不得合并比较（预注册 §5）",
        }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2, ensure_ascii=False))

    print("=" * 86)
    print(f"VERDICT: {verdict}    K*_reject = {k_star}")
    print("=" * 86)
    print("\n--- 内建 sanity check：K=10000 是否复现已发表 ground truth ---")
    for a in SOURCES:
        s = sanity[a]
        print(f"  {a:5s} 本次={s['per_seed']} mean={s['mean']:+7.2f}  "
              f"已发表={s['gt_mean']:+7.2f}  偏差={s['deviation']:+6.2f} "
              f"(容差±{s['tolerance']:.2f})  全负={s['all_negative']}  "
              f"{'PASS' if s['pass'] else 'FAIL'}")
    print(f"  → sanity {'PASS' if sane else 'FAIL —— 预注册要求整体作废'}")
    print(f"\n行为层排序(对照): " + "  ".join(f"{a}={BEHAVIOR[a]:.0f}" for a in
                                              sorted(BEHAVIOR, key=BEHAVIOR.get, reverse=True)))
    for k in KS:
        d = res["per_K"][str(k)]
        print(f"\n--- K={k} ---  全部拒绝 {d['reject_all_hits']} "
              f"{'PASS' if d['reject_pass'] else '----'}   argmax=walk {d['argmax_is_walk']}")
        print("    mean U: " + "  ".join(f"{a}={d['mean_U'][a]:+7.2f}" for a in SOURCES))
        for r in d["per_seed"]:
            print(f"    s{r['seed']}: " + "  ".join(f"{a}={r['U'][a]:+7.2f}" for a in SOURCES)
                  + f"   max={r['argmax']:5s}({r['max_U']:+7.2f})  "
                  + ("拒绝" if r["reject_all"] else "**未拒绝**"))
    if res.get("avoided_loss"):
        v = res["avoided_loss"]
        print(f"\n避损核算: 成本={v['reject_cost_steps']} 步  避免损失≈{v['avoided_loss_return']} return")
    print(f"\nwritten {OUT}")


if __name__ == "__main__":
    main()
