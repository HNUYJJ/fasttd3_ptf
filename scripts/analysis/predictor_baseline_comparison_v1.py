"""BAC vs 更简单基线的正面比较（零训练成本，回应外部审核的阻塞项）。

外部审核（ChatGPT，2026-07-29）指出一个此前未被检验的问题：

    slide 上 mean per-timestep reward 也预测 walk>run>stand；
    stair 上同样如此。因此尚未证明"reward 分量分解"比"去掉 episode 长度"
    这一简单修正有任何增量。

若 BAC 不优于简单基线，则它的全部复杂度（17 任务结构核准、瓶颈集、
Coverage/Damage 正负不对称、乘性边际敏感度）都是无增量的，
该线应停止作为迁移性指标，只保留"生存时长校正 + 任务分量诊断"这两个结果。

四个被比较的预测器（全部零额外交互，只用 zero-shot probe）：
    P1 episodic return          既有八信号族之一，已知被生存时长污染
    P2 per-step reward          最简修正：去掉 episode 长度
    P3 main progress component  取非通用分量中边际敏感度最大的**单个**分量
    P4 BAC / NET                本文提出的指标

评分对象只用有配对学习效用真值、且该真值可判定的 target。
stair 被排除：其三源 U 全部跨零（无判决力），不能用于区分预测器。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts/analysis"))

import bottleneck_aligned_coverage_v1 as B  # noqa: E402
from configs.reward_structure.humanoidbench_v1 import (  # noqa: E402
    GENERIC_TERMS, SPEC, UNBOUNDED,
)

# 实测配对学习效用 U（相对 student/scratch），来自各 gate 的 128-ep 冻结面板。
# stair 不列入评分：三源 U 全部跨零。
TRUTH = {
    "h1hand-hurdle-v0": {"walk": 104.89, "run": 379.66},           # stand 未测
    "h1hand-crawl-v0":  {"stand": -448.48, "walk": -216.60, "run": -208.07},
    "h1hand-door-v0":   {"stand": -32.64, "walk": -22.20, "run": -30.64},
    "h1hand-slide-v0":  {"stand": -1.21, "walk": 56.95, "run": 16.90},
}
SRC = ("stand", "walk", "run")


def rank(d: dict) -> list:
    return [k for k, _ in sorted(d.items(), key=lambda kv: -kv[1])]


def main_progress_component(target: str, zero: dict) -> str | None:
    """非通用分量中边际敏感度最大的单个分量。"""
    spec = SPEC[target]
    if spec["kind"] == UNBOUNDED:
        return None
    cand = {}
    if spec["kind"] == "multiplicative":
        facs = spec["factors"]
        vals = {c: B._resolve(zero, c) for c in facs}
        for c in facs:
            if c in GENERIC_TERMS:
                continue
            p = 1.0
            for c2 in facs:
                if c2 != c:
                    p *= vals[c2]
            cand[c] = p
    else:
        gate = 1.0
        for g in spec.get("gates", []):
            gate *= B._resolve(zero, g)
        for c, w in spec.get("terms", {}).items():
            if c in GENERIC_TERMS or "*" in c:
                continue
            cand[c] = w * gate
        for w, grp in spec.get("min_groups", []):
            cand["min(" + ",".join(grp) + ")"] = w * gate
    return max(cand, key=cand.get) if cand else None


def value_of(info: dict, key: str) -> float:
    if key.startswith("min("):
        return min(B._resolve(info, g) for g in key[4:-1].split(","))
    return B._resolve(info, key)


def main() -> None:
    by = B.load_probe()
    rows, score = [], {p: {"full": 0, "worst": 0, "n": 0} for p in
                       ("P1_return", "P2_per_step", "P3_progress", "P4_BAC")}

    for tg, truth in TRUTH.items():
        arms = by[tg]
        zero = arms["zero"]["info_means"]
        srcs = [s for s in SRC if s in truth]

        p1 = {s: arms[s]["return_mean"] for s in srcs}
        p2 = {s: arms[s]["info_means"]["per_timestep_reward"] for s in srcs}
        comp = main_progress_component(tg, zero)
        p3 = {s: value_of(arms[s]["info_means"], comp) for s in srcs} if comp else None
        r = B.analyze(tg, arms)
        p4 = {s: r["sources"][s]["NET"] for s in srcs}

        t_rank = rank(truth)
        t_worst = t_rank[-1]
        preds = {"P1_return": p1, "P2_per_step": p2, "P3_progress": p3, "P4_BAC": p4}
        row = {"target": tg, "component": comp, "truth_rank": t_rank}
        for name, p in preds.items():
            if p is None:
                row[name] = None
                continue
            pr = rank(p)
            full = pr == t_rank
            worst = pr[-1] == t_worst
            row[name] = {"rank": pr, "full": full, "worst": worst}
            score[name]["full"] += int(full)
            score[name]["worst"] += int(worst)
            score[name]["n"] += 1
        rows.append(row)

    print("预测器正面比较（评分只用有可判定学习效用真值的 target；stair 因三源 U 全跨零被排除）\n")
    for row in rows:
        print(f"### {row['target']}   主任务分量 = {row['component']}")
        print(f"    实测 U 排序      {'>'.join(row['truth_rank'])}")
        for name in ("P1_return", "P2_per_step", "P3_progress", "P4_BAC"):
            v = row[name]
            if v is None:
                print(f"    {name:14s} —"); continue
            marks = ("全序✓" if v["full"] else "全序✗") + " " + ("最差✓" if v["worst"] else "最差✗")
            print(f"    {name:14s} {'>'.join(v['rank']):22s} {marks}")
        print()

    print(f"{'预测器':16s} {'全序命中':>10s} {'最差源命中':>12s}")
    print("-" * 42)
    for name, s in score.items():
        print(f"{name:16s} {s['full']}/{s['n']:<9d} {s['worst']}/{s['n']:<11d}")

    out = REPO / "docs/data/predictor_baseline_comparison_v1.json"
    out.write_text(json.dumps({"rows": rows, "score": score}, indent=1, ensure_ascii=False))
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
