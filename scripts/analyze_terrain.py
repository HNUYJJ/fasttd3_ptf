"""terrain 三方核心论点分析(RBO-PTF 第①项, 多 seed 版)。

stair/slide/pole/crawl × {safe(reward-weighted 源选择)、rand(uniform)、scr(scratch)}
× seed 1/2/3。核心问:reward-weighted 是否正迁移、且优于 uniform random。

wandb 坐标(核实自 aggregate_multitask.py):
  api.runs("yujiajie-nju/fasttd3_ptf"); name=run.config["exp_name"] or run.name;
  history(keys=["_step","eval_avg_return"]); AUC=trapz(v,s)/step_span。

seed1: rand/scr=OLD stamp 20260615T044012Z, safe=REC 20260615T094159Z;
seed2/3: 20260616T000532Z。正则不限 stamp, 同 (task,method,seed) 取 max_step 最大者。
公平对比: 每 task 三方所有 seed 对齐到共同窗口(全体 max_step 最小值)。

用法: python scripts/analyze_terrain.py
"""
from __future__ import annotations

import re
import numpy as np
import wandb

api = wandb.Api()
runs = api.runs("yujiajie-nju/fasttd3_ptf")

PAT = re.compile(r"h1hand_(stair|slide|pole|crawl)_tp_(safe|rand|scr)_s(\d)_")
TASKS = ["stair", "slide", "pole", "crawl"]
METHODS = ["scr", "rand", "safe"]


def auc_window(s, v, hi):
    mask = s <= hi
    s2, v2 = s[mask], v[mask]
    if len(s2) < 2:
        return float("nan")
    return float(np.trapz(v2, s2) / (s2[-1] - s2[0]))


# data[task][method][seed] = (max_step, s, v, name)  取 max_step 最大者(过滤 OOM 残留)
data: dict = {}
for run in runs:
    name = run.config.get("exp_name", "") or run.name
    m = PAT.search(name)
    if not m:
        continue
    task, method, seed = m.group(1), m.group(2), int(m.group(3))
    h = run.history(samples=4000, keys=["_step", "eval_avg_return"])
    if "eval_avg_return" not in h:
        continue
    ev = h.dropna(subset=["eval_avg_return"])
    if len(ev) < 2:
        continue
    s = ev["_step"].values
    v = ev["eval_avg_return"].values
    max_step = float(s[-1])
    slot = data.setdefault(task, {}).setdefault(method, {})
    prev = slot.get(seed)
    if prev is None or max_step > prev[0]:
        slot[seed] = (max_step, s, v, name)

print("=" * 96)
print("terrain 三方核心论点(3-seed): reward-weighted(safe) vs uniform(rand) vs scratch(scr)")
print("=" * 96)

print("\n[0] 选中 run 及 max_step (诊断双批 stamp/OOM 残留; 每格应有 seed 1/2/3)")
for task in TASKS:
    for method in METHODS:
        seeds = data.get(task, {}).get(method, {})
        got = ",".join(f"s{sd}:{int(seeds[sd][0]/1000)}k" for sd in sorted(seeds))
        print(f"  {task:6s} {method:4s}  [{got}]")

# 每 task 共同窗口 = 三方全 seed max_step 最小值
print("\n[1] 共同窗口 AUC: 每 task × method 的 per-seed + mean±std")
print(f"{'task':6s} | {'hi':>6s} | {'method':4s} | {'per-seed AUC':>26s} | {'mean±std':>14s}")
win = {t: {m: {} for m in METHODS} for t in TASKS}  # win[task][method][seed]=auc
hi_of = {}
for task in TASKS:
    d = data.get(task, {})
    all_steps = [d[m][sd][0] for m in METHODS if m in d for sd in d[m]]
    if not all_steps:
        continue
    hi = min(all_steps)
    hi_of[task] = hi
    for mi, method in enumerate(METHODS):
        seeds = sorted(d.get(method, {}))
        vals = []
        for sd in seeds:
            _, s, v, _ = d[method][sd]
            a = auc_window(s, v, hi)
            win[task][method][sd] = a
            vals.append(a)
        perseed = " ".join(f"s{sd}={win[task][method][sd]:6.1f}" for sd in seeds)
        mu = np.mean(vals) if vals else float("nan")
        sd_ = np.std(vals) if len(vals) > 1 else 0.0
        tag = f"{task:6s} | {hi/1000:5.0f}k" if mi == 0 else f"{'':6s} | {'':6s}"
        print(f"{tag} | {method:4s} | {perseed:>26s} | {mu:7.1f}±{sd_:5.1f}")

print("\n[2] 增益 (共同窗口, per-seed; 看符号一致性)")
print(f"{'task':6s} | {'metric':9s} | {'s1':>8s} {'s2':>8s} {'s3':>8s} | {'mean±std':>14s}")
pair_safe_rand, pair_safe_scr, pair_rand_scr = [], [], []
for task in TASKS:
    seeds = sorted(set(win[task]["safe"]) & set(win[task]["rand"]) & set(win[task]["scr"]))
    if not seeds:
        continue
    for label, a, b, acc in [
        ("safe-scr", "safe", "scr", pair_safe_scr),
        ("rand-scr", "rand", "scr", pair_rand_scr),
        ("safe-rand", "safe", "rand", pair_safe_rand),
    ]:
        diffs = [win[task][a][sd] - win[task][b][sd] for sd in seeds]
        acc.extend(diffs)
        cells = " ".join(f"{diffs[i]:+8.1f}" if i < len(diffs) else f"{'':>8s}" for i in range(3))
        print(f"{task:6s} | {label:9s} | {cells} | {np.mean(diffs):+7.1f}±{np.std(diffs):5.1f}")
    print(f"{'':6s} |")

print("\n[3] 判读 (全 4 task × 3 seed = 12 个 (task,seed) 组合)")
n = len(pair_safe_rand)
print(f"  safe > rand : {sum(1 for x in pair_safe_rand if x>0)}/{n}  "
      f"(mean {np.mean(pair_safe_rand):+.1f}±{np.std(pair_safe_rand):.1f})  ← 核心卖点(源选择价值)")
print(f"  safe > scr  : {sum(1 for x in pair_safe_scr if x>0)}/{n}  "
      f"(mean {np.mean(pair_safe_scr):+.1f}±{np.std(pair_safe_scr):.1f})  ← 正迁移")
print(f"  rand > scr  : {sum(1 for x in pair_rand_scr if x>0)}/{n}  "
      f"(mean {np.mean(pair_rand_scr):+.1f}±{np.std(pair_rand_scr):.1f})  ← uniform 不稳/无价值")
# 简单 paired t 近似(safe-rand)
mu, sd = np.mean(pair_safe_rand), np.std(pair_safe_rand, ddof=1)
t = mu / (sd / np.sqrt(n)) if sd > 0 else float("inf")
print(f"  safe-rand paired: mean={mu:+.1f}, t≈{t:.2f} (n={n})")
print(f"  → 核心论点(safe>rand>scr) {'成立' if np.mean(pair_safe_rand)>0 and np.mean(pair_safe_scr)>0 else '需诊断'}")
