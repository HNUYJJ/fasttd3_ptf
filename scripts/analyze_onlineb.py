"""Step A 方向验证分析: online_bootstrap(student-as-arm) vs wfix/safe/rand/scr。

判读标准(docs/advisor_feedback_analysis_20260702.md §5):
  crawl(负迁移 case): onlineb >= scr → 负迁移关闭 ✓
  pole (正迁移 case): onlineb ≈ wfix → 机制不误伤 ✓
先只有 s1(与 wfix s1 同 option_seed 配对), 共同窗口 AUC。
另拉 mcg/arm_value_* 与 mcg/online_wp 曲线,解读 arm 竞争过程。

用法: python scripts/analyze_onlineb.py
"""
from __future__ import annotations

import re
from collections import defaultdict

import numpy as np
import wandb

api = wandb.Api()
runs = api.runs("yujiajie-nju/fasttd3_ptf")

PAT = re.compile(
    r"h1hand_(stair|slide|pole|crawl)_tp_"
    r"(safe|rand|scr|wfix|onlineb|obrw|tgated|aonly|conly|split)_s(\d)_")
TASKS = ["crawl", "pole"]
METHODS = ["scr", "rand", "wfix", "safe", "onlineb", "obrw", "tgated",
           "aonly", "conly", "split"]


def auc_window(s, v, hi):
    mask = s <= hi
    s2, v2 = s[mask], v[mask]
    if len(s2) < 2:
        return float("nan")
    return float(np.trapz(v2, s2) / (s2[-1] - s2[0]))


data: dict = defaultdict(lambda: defaultdict(dict))
arm_runs = {}
for run in runs:
    name = run.config.get("exp_name", "") or run.name
    m = PAT.search(name)
    if not m:
        continue
    task, method, seed = m.group(1), m.group(2), int(m.group(3))
    if task not in TASKS:
        continue
    h = run.history(samples=4000, keys=["_step", "eval_avg_return"])
    if "eval_avg_return" not in h:
        continue
    ev = h.dropna(subset=["eval_avg_return"])
    if len(ev) < 2:
        continue
    s = ev["_step"].values
    v = ev["eval_avg_return"].values
    max_step = float(s[-1])
    prev = data[task][method].get(seed)
    if prev is None or max_step > prev[0]:
        data[task][method][seed] = (max_step, s, v, name)
        if method == "onlineb":
            arm_runs[(task, seed)] = run

print("=" * 96)
print("Step A 验证: onlineb(student-as-arm) vs 四方 (per-task 共同窗口 AUC)")
print("=" * 96)

for task in TASKS:
    steps = [data[task][m][sd][0] for m in METHODS if m in data[task]
             for sd in data[task][m]]
    if not steps:
        continue
    hi = min(steps)
    print(f"\n[{task}] 共同窗口 {hi/1000:.0f}k")
    print(f"{'method':8s} {'seed':>4s} {'AUC':>8s}")
    aucs = {}
    for m in METHODS:
        for sd in sorted(data[task].get(m, {})):
            _, s, v, _ = data[task][m][sd]
            a = auc_window(s, v, hi)
            aucs.setdefault(m, {})[sd] = a
            print(f"{m:8s} {sd:4d} {a:8.1f}")
    # 判读(seed1 配对); Step B 判读: obrw vs onlineb(增量) + vs scr/wfix(绝对)
    scr1 = aucs.get("scr", {}).get(1, float("nan"))
    wfx1 = aucs.get("wfix", {}).get(1, float("nan"))
    for meth in ("onlineb", "obrw", "tgated", "aonly", "conly", "split"):
        ob = aucs.get(meth, {}).get(1)
        if ob is None:
            continue
        if task == "crawl":
            verdict = "PASS(负迁移关闭)" if ob >= scr1 else f"未达标(差 {scr1-ob:+.1f})"
            print(f"  → crawl {meth}: {ob:.1f} vs scr {scr1:.1f} → {verdict}")
        if task == "pole":
            rel = (ob - wfx1) / abs(wfx1) if wfx1 == wfx1 and wfx1 != 0 else float("nan")
            verdict = "PASS(不误伤)" if rel > -0.10 else f"损害 {rel:.0%}"
            print(f"  → pole {meth}: {ob:.1f} vs wfix {wfx1:.1f} ({rel:+.0%}) → {verdict}")
    ob_a = aucs.get("onlineb", {}).get(1)
    ob_b = aucs.get("obrw", {}).get(1)
    if ob_a is not None and ob_b is not None:
        print(f"  → Step B 增量(obrw−onlineb): {ob_b - ob_a:+.1f}")
    # split 归因(裁定 2026-07-02): 路径分解——both(0.1/0.1) vs aonly(0.1/-) vs
    # conly(-/0.1) 只差作用路径; split(0.05/0.4)=设计版
    parts = {m: aucs.get(m, {}).get(1) for m in ("aonly", "conly", "split")}
    if ob_a is not None and any(v is not None for v in parts.values()):
        row = "  → 归因(相对 onlineb): "
        for m, v in parts.items():
            if v is not None:
                row += f"{m} {v - ob_a:+.1f}  "
        if ob_b is not None:
            row += f"| both {ob_b - ob_a:+.1f}"
        print(row)

print("\n" + "=" * 96)
print("arm 竞争过程 (mcg/arm_value_* 与 student exec share)")
print("=" * 96)
for (task, sd), run in sorted(arm_runs.items()):
    keys = ["_step", "mcg/online_wp", "mcg/arm_value_src0", "mcg/arm_value_src1",
            "mcg/arm_value_src2", "mcg/arm_value_student", "mcg/exec_env_frac"]
    h = run.history(samples=2000, keys=keys)
    if "mcg/arm_value_student" not in h:
        print(f"[{task} s{sd}] 无 arm 指标(检查 log 键名)")
        continue
    ev = h.dropna(subset=["mcg/arm_value_student"])
    print(f"\n[{task} s{sd}] (src0=stand src1=walk src2=run; student_share=1-exec_env_frac)")
    print(f"{'step':>7s} {'wp':>5s} {'stand':>8s} {'walk':>8s} {'run':>8s} "
          f"{'student':>8s} {'stu_share':>9s}")
    idx = np.linspace(0, len(ev) - 1, min(8, len(ev))).astype(int)
    for i in idx:
        r = ev.iloc[i]
        print(f"{r['_step']:7.0f} {r.get('mcg/online_wp', float('nan')):5.2f} "
              f"{r['mcg/arm_value_src0']:8.3f} {r['mcg/arm_value_src1']:8.3f} "
              f"{r['mcg/arm_value_src2']:8.3f} {r['mcg/arm_value_student']:8.3f} "
              f"{1-r['mcg/exec_env_frac']:9.2f}")
