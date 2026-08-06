"""端到端系统裁决（预注册 docs/experiments/endtoend_v1_prereg_20260804.md §6）。

判据在任何新训练启动之前冻结：

    逐 (target, seed) 配对比较 100k 的 source-free return（panel128）
    (a) hurdle 与 slide：C > A    每 target 3/3
    (b) 三个 target 全部：C > B    每 target 3/3

    (a) ∧ (b)            → ENDTOEND_SUPPORTED
    (b) 且 ¬(a)          → ENDTOEND_PARTIAL
    ¬(b)                 → ENDTOEND_REFUTED
    缺失                  → INCOMPLETE（非零退出）

crawl 的 C ≡ A（决策为 REJECT），故 crawl 不参与 (a)——否则自动成立（§8.4）。
跨 learner 汇总用 3 seeds 的 sd，不用 episode 面板 SE（M16）。
"""
from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path

OUT = Path("docs/data/endtoend_v1/results.json")
SEEDS = (1, 2, 3)
STEP = 100000

# 臂 → 评估 JSON 路径模板。C 臂在 crawl 上复用 A 臂数据（REJECT 后即纯 student）。
PATHS = {
    "hurdle": {
        "A": "docs/data/hurdle_speedup_v1/source_free_eval/scratch_s{s}_step{k}.json",
        "B": "docs/data/hurdle_speedup_v1/source_free_eval/source_s{s}_step{k}.json",
        "C": "docs/data/endtoend_v1/hurdle/source_free_eval/exit_s{s}_step{k}.json",
    },
    "slide": {
        "A": "docs/data/slide_speedup_v1/source_free_eval/scratch_s{s}_step{k}.json",
        "B": "docs/data/slide_speedup_v1/source_free_eval/walk_s{s}_step{k}.json",
        "C": "docs/data/slide_hard_exit_v1/source_free_eval/exit_s{s}_step{k}.json",
    },
    "crawl": {
        "A": "docs/data/endtoend_v1/crawl/source_free_eval/scratch_s{s}_step{k}.json",
        "B": "docs/data/endtoend_v1/crawl/source_free_eval/blind_s{s}_step{k}.json",
        "C": "docs/data/endtoend_v1/crawl/source_free_eval/scratch_s{s}_step{k}.json",
    },
}
CHECK_A = ("hurdle", "slide")          # 判据 (a) 的适用 target
CHECK_B = ("hurdle", "slide", "crawl")  # 判据 (b) 的适用 target


def ret(path: str) -> float:
    return float(json.loads(Path(path).read_text())["aggregate"]["return_mean"])


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)

    # 层1：完整性——独立扫描全部组合，不因前置缺失提前返回（§4）
    missing = []
    for t, arms in PATHS.items():
        for a, tpl in arms.items():
            for s in SEEDS:
                p = Path(tpl.format(s=s, k=STEP))
                if not p.exists():
                    missing.append(f"{t}/{a}/s{s} -> {p}")
    if missing:
        print(f"VERDICT: INCOMPLETE  ({len(missing)} 项缺失)")
        for m in missing[:15]:
            print(f"  missing: {m}")
        if len(missing) > 15:
            print(f"  ... 共 {len(missing)} 项")
        OUT.write_text(json.dumps({"verdict": "INCOMPLETE", "missing": missing},
                                  indent=2, ensure_ascii=False))
        return 2

    per_target = {}
    for t, arms in PATHS.items():
        rows = []
        for s in SEEDS:
            v = {a: ret(arms[a].format(s=s, k=STEP)) for a in ("A", "B", "C")}
            rows.append({"seed": s, **{f"{a}_return": round(v[a], 3) for a in v},
                         "C_minus_A": round(v["C"] - v["A"], 3),
                         "C_minus_B": round(v["C"] - v["B"], 3)})
        per_target[t] = {
            "per_seed": rows,
            "mean": {a: round(st.mean([r[f"{a}_return"] for r in rows]), 3) for a in "ABC"},
            "sd_across_learners": {a: round(st.stdev([r[f"{a}_return"] for r in rows]), 3)
                                   for a in "ABC"},
            "C_gt_A": f"{sum(r['C_minus_A'] > 0 for r in rows)}/3",
            "C_gt_B": f"{sum(r['C_minus_B'] > 0 for r in rows)}/3",
        }

    a_pass = all(per_target[t]["C_gt_A"] == "3/3" for t in CHECK_A)
    b_pass = all(per_target[t]["C_gt_B"] == "3/3" for t in CHECK_B)
    verdict = ("ENDTOEND_SUPPORTED" if (a_pass and b_pass)
               else "ENDTOEND_PARTIAL" if b_pass else "ENDTOEND_REFUTED")

    report = {
        "prereg": "docs/experiments/endtoend_v1_prereg_20260804.md",
        "decisions": "docs/data/endtoend_v1/decisions.json",
        "step": STEP,
        "criterion_a_targets": list(CHECK_A), "criterion_a_pass": a_pass,
        "criterion_b_targets": list(CHECK_B), "criterion_b_pass": b_pass,
        "racing_cost_steps": 40000,
        "per_target": per_target,
        "verdict": verdict,
    }
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    print("=" * 82)
    print(f"VERDICT: {verdict}     (a) C>A on {CHECK_A}: {a_pass}     (b) C>B on all: {b_pass}")
    print("=" * 82)
    for t, d in per_target.items():
        note = "  [C≡A, 不参与判据(a)]" if t == "crawl" else ""
        print(f"\n{t}{note}")
        for r in d["per_seed"]:
            print(f"  s{r['seed']}  A={r['A_return']:8.1f}  B={r['B_return']:8.1f}  "
                  f"C={r['C_return']:8.1f}   C−A={r['C_minus_A']:+8.1f}  C−B={r['C_minus_B']:+8.1f}")
        m, sd = d["mean"], d["sd_across_learners"]
        print(f"  mean A={m['A']:8.1f}±{sd['A']:6.1f}  B={m['B']:8.1f}±{sd['B']:6.1f}  "
              f"C={m['C']:8.1f}±{sd['C']:6.1f}   (sd = learner 间)")
        print(f"  C>A {d['C_gt_A']}    C>B {d['C_gt_B']}")
    print(f"\n端到端臂另需 racing 成本 40000 步（口径见预注册 §4，两个口径都须报）")
    print(f"\nwritten {OUT}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"VERDICT: INCOMPLETE  (unhandled error: {type(exc).__name__}: {exc})")
        raise
