"""多任务三方对比聚合(RBO-PTF Day5-6)。

balance/hurdle/cabinet × {safe,rand,scr} × 3 seed + window(safe=d56,rand/scr=mt)。
核心论据:safe-horizon bootstrap 相对 random warmup,在多任务上(a)方差更小、
(b)符号一致(无负迁移翻转)、(c)regret 更低。从 wandb eval_avg_return 算 AUC。

用法: python scripts/aggregate_multitask.py
"""
from __future__ import annotations

import re
import numpy as np
import wandb

api = wandb.Api()
runs = api.runs("yujiajie-nju/fasttd3_ptf")

def auc(s, v):
    return float(np.trapz(v, s) / (s[-1] - s[0])) if len(s) > 1 else float("nan")

# h1hand_<task>_mt_<safe|rand|scr>_s<seed>  或  h1hand_window_d56_safe_s<seed>
MT = re.compile(r"h1hand_(balance_hard|hurdle|cabinet|window)_(?:mt_(safe|rand|scr)|(d56_safe))_s(\d+)_")

data: dict = {}
for run in runs:
    name = run.config.get("exp_name", "") or run.name
    m = MT.search(name)
    if not m:
        continue
    task = m.group(1)
    method = m.group(2) or "safe"
    seed = int(m.group(4))
    h = run.history(samples=4000, keys=["_step", "eval_avg_return"])
    if "eval_avg_return" not in h:
        continue
    ev = h.dropna(subset=["eval_avg_return"])
    if len(ev) < 2:
        continue
    data.setdefault(task, {}).setdefault(method, {})[seed] = auc(
        ev["_step"].values, ev["eval_avg_return"].values)

print("=" * 86)
print("多任务三方对比(AUC, ROI vs scratch; safe/rand 均 bootstrap_only 口径, 只差 warmup 方式)")
print("=" * 86)
print(f"{'task':12s} {'scr_AUC':>8s} | {'method':5s} {'ROI/seed':22s} {'mean±std':12s} {'符号':5s} {'regret':>7s}")
roi_std = {"safe": [], "rand": []}
for task in ["balance_hard", "hurdle", "cabinet", "window"]:
    d = data.get(task, {})
    if "scr" not in d:
        print(f"{task:12s}  (无 scratch)")
        continue
    base = float(np.mean(list(d["scr"].values())))
    print(f"{task:12s} {base:8.0f} |")
    for method in ["rand", "safe"]:
        if method not in d:
            print(f"{'':12s} {'':8s} | {method:5s} (缺)")
            continue
        seeds = sorted(d[method])
        rois = [(d[method][s] - base) / abs(base) for s in seeds]
        mean_auc = float(np.mean([d[method][s] for s in seeds]))
        regret = max(0.0, base - mean_auc)
        sign = "一致" if all(np.sign(r) == np.sign(rois[0]) for r in rois) else "翻转!"
        roi_std[method].append(float(np.std(rois)))
        roistr = " ".join(f"{r:+.0%}" for r in rois)
        print(f"{'':12s} {'':8s} | {method:5s} {roistr:22s} "
              f"{np.mean(rois):+5.0%}±{np.std(rois):4.0%}  {sign:5s} {regret:7.0f}")

print("=" * 86)
print("绝对 AUC(避开 ROI 分母 confound; safe>rand>scr 才是真优)")
print(f"{'task':12s} | {'scr (per seed)':28s} | {'rand':26s} | {'safe':26s}")
for task in ["balance_hard", "hurdle", "cabinet", "window"]:
    d = data.get(task, {})
    cells = []
    for method in ["scr", "rand", "safe"]:
        if method in d:
            seeds = sorted(d[method])
            vals = [d[method][s] for s in seeds]
            cells.append(f"{np.mean(vals):4.0f}±{np.std(vals):3.0f} {[round(v) for v in vals]}")
        else:
            cells.append("(缺)")
    print(f"{task:12s} | {cells[0]:28s} | {cells[1]:26s} | {cells[2]:26s}")

print("=" * 86)
print("核心论据: safe vs rand 的 ROI 方差(std,跨 seed)逐任务 + 平均")
for method in ["rand", "safe"]:
    if roi_std[method]:
        print(f"  {method:5s} ROI std 各任务: {[f'{x:.0%}' for x in roi_std[method]]}  "
              f"平均 {np.mean(roi_std[method]):.0%}")
if roi_std["safe"] and roi_std["rand"]:
    print(f"  → safe 平均方差 {np.mean(roi_std['safe']):.0%} vs rand {np.mean(roi_std['rand']):.0%}"
          f"  (safe 更稳={'是' if np.mean(roi_std['safe']) < np.mean(roi_std['rand']) else '否'})")
