"""wfix 解耦消融分析(RBO-PTF 第②项, ChatGPT Step 1) —— 3-seed 版。

回应 reviewer: safe 比 rand 好, 是"源选择"好还是"horizon 更长"好?
四方(每个 task × seed 共同窗口 AUC):
  scr  = scratch
  rand = uniform 源选择 + horizon=25
  wfix = weighted 源选择 + horizon=25   (与 rand 只差源选择)
  safe = weighted 源选择 + horizon=50   (与 wfix 只差 horizon)
解耦(按 (task,seed) 配对):
  wfix - rand  = 纯源选择增益(horizon 都=25)
  safe - wfix  = 纯执行时长(horizon)增益(源都 weighted)
  safe - rand  = 总增益 = 源选择 + horizon

seed1 早跑(wfix STAMP=20260620T041436Z; safe=094159Z; rand/scr=044012Z);
seed2/3 wfix=20260620T145205Z, 其余沿用 terrain 的 s2/s3。正则不限 stamp。
用法: python scripts/analyze_wfix.py
"""
from __future__ import annotations

import re
from collections import defaultdict

import numpy as np
import wandb

api = wandb.Api()
runs = api.runs("yujiajie-nju/fasttd3_ptf")

PAT = re.compile(r"h1hand_(stair|slide|pole|crawl)_tp_(safe|rand|scr|wfix)_s(\d)_")
TASKS = ["stair", "slide", "pole", "crawl"]
METHODS = ["scr", "rand", "wfix", "safe"]


def auc_window(s, v, hi):
    mask = s <= hi
    s2, v2 = s[mask], v[mask]
    if len(s2) < 2:
        return float("nan")
    return float(np.trapz(v2, s2) / (s2[-1] - s2[0]))


def paired_t(diffs):
    """单样本配对 t(对每个配对差做单样本 t 检验, H0: mean=0)。返回 (t, n)。"""
    d = np.asarray([x for x in diffs if np.isfinite(x)], dtype=float)
    n = len(d)
    if n < 2:
        return float("nan"), n
    sd = d.std(ddof=1)
    if sd == 0:
        return float("inf") * np.sign(d.mean()), n
    return float(d.mean() / (sd / np.sqrt(n))), n


# data[task][method][seed] = (max_step, s, v, name)
data: dict = defaultdict(lambda: defaultdict(dict))
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
    prev = data[task][method].get(seed)
    if prev is None or max_step > prev[0]:  # 过滤 OOM 残留(取跑得最远的)
        data[task][method][seed] = (max_step, s, v, name)

print("=" * 96)
print("wfix 解耦消融(3-seed, 每 task 共同窗口 AUC): 源选择 vs 执行时长 哪个是主因")
print("=" * 96)

# 每个 task 的共同窗口 = 该 task 下所有 (method,seed) 的 max_step 的最小值
common_hi = {}
for task in TASKS:
    steps = [data[task][m][sd][0] for m in METHODS for sd in data[task][m]]
    common_hi[task] = min(steps) if steps else float("nan")

# auc[task][method][seed] = AUC@common_hi
auc = defaultdict(lambda: defaultdict(dict))
for task in TASKS:
    hi = common_hi[task]
    for m in METHODS:
        for sd, (_, s, v, _) in data[task][m].items():
            auc[task][m][sd] = auc_window(s, v, hi)

print(f"\n[1] 四方 AUC: 每 task per-method mean±std (跨 seed), 括号内 = seed 数")
print(f"{'task':6s} | {'hi':>6s} | " + " | ".join(f"{m:>13s}" for m in METHODS))
for task in TASKS:
    hi = common_hi[task]
    cells = []
    for m in METHODS:
        vals = [auc[task][m][sd] for sd in sorted(auc[task][m])
                if np.isfinite(auc[task][m][sd])]
        if vals:
            cells.append(f"{np.mean(vals):7.1f}±{np.std(vals):4.0f}({len(vals)})")
        else:
            cells.append(f"{'--':>13s}")
    print(f"{task:6s} | {hi/1000:5.0f}k | " + " | ".join(cells))

# 按 (task,seed) 配对解耦
print(f"\n[2] 解耦(按 (task,seed) 配对): "
      f"src=wfix-rand  hor=safe-wfix  tot=safe-rand")
print(f"{'task':6s} {'seed':>4s} | {'wfix-rand':>10s} {'safe-wfix':>10s} | "
      f"{'safe-rand':>10s}")
pair = {"src": [], "hor": [], "tot": []}      # 跨所有 (task,seed)
pair_by_task = defaultdict(lambda: {"src": [], "hor": [], "tot": []})
for task in TASKS:
    for sd in sorted(set().union(*[set(auc[task][m]) for m in METHODS])):
        if not all(sd in auc[task][m] and np.isfinite(auc[task][m][sd])
                   for m in METHODS):
            continue
        w = {m: auc[task][m][sd] for m in METHODS}
        src = w["wfix"] - w["rand"]
        hor = w["safe"] - w["wfix"]
        tot = w["safe"] - w["rand"]
        pair["src"].append(src); pair["hor"].append(hor); pair["tot"].append(tot)
        pt = pair_by_task[task]
        pt["src"].append(src); pt["hor"].append(hor); pt["tot"].append(tot)
        print(f"{task:6s} {sd:4d} | {src:+10.1f} {hor:+10.1f} | {tot:+10.1f}")

print(f"\n[2b] per-task 跨 seed 均值")
print(f"{'task':6s} | {'src(mean)':>10s} {'hor(mean)':>10s} {'tot(mean)':>10s}  n")
for task in TASKS:
    pt = pair_by_task[task]
    if not pt["src"]:
        continue
    print(f"{task:6s} | {np.mean(pt['src']):+10.1f} {np.mean(pt['hor']):+10.1f} "
          f"{np.mean(pt['tot']):+10.1f}  {len(pt['src'])}")

print("\n[3] 跨所有 (task,seed) 组合的总判读")
if pair["src"]:
    n = len(pair["src"])
    t_src, _ = paired_t(pair["src"])
    t_hor, _ = paired_t(pair["hor"])
    pos_src = sum(1 for x in pair["src"] if x > 0)
    pos_hor = sum(1 for x in pair["hor"] if x > 0)
    print(f"  组合总数 N = {n} (task×seed)")
    print(f"  纯源选择(wfix-rand): 平均 {np.mean(pair['src']):+.1f}, "
          f">0 {pos_src}/{n}, paired t = {t_src:+.2f}")
    print(f"  纯horizon(safe-wfix): 平均 {np.mean(pair['hor']):+.1f}, "
          f">0 {pos_hor}/{n}, paired t = {t_hor:+.2f}")
    s_mean, h_mean = np.mean(pair["src"]), np.mean(pair["hor"])
    if s_mean > 0 and abs(s_mean) > abs(h_mean):
        verd = "主因=源选择(reward-weighted), horizon 次要/有害"
    elif h_mean > 0 and abs(h_mean) > abs(s_mean):
        verd = "主因=horizon, 源选择次要"
    else:
        verd = "两者都有贡献/需细看"
    print(f"  → {verd}")
    print("  注: terrain safe horizon=50(fall 低未截断), 解耦看 stair/slide/pole;")
    print("      crawl 是负迁移任务, 解耦增益预期负或混乱(分层呈现)。")

    # 非 crawl 子集(去掉负迁移任务)
    sub_src, sub_hor = [], []
    for task in ["stair", "slide", "pole"]:
        pt = pair_by_task[task]
        sub_src += pt["src"]; sub_hor += pt["hor"]
    if sub_src:
        ts, _ = paired_t(sub_src); th, _ = paired_t(sub_hor)
        print(f"\n  [非crawl子集 stair/slide/pole, N={len(sub_src)}]")
        print(f"    源选择: 平均 {np.mean(sub_src):+.1f}, "
              f">0 {sum(1 for x in sub_src if x>0)}/{len(sub_src)}, t={ts:+.2f}")
        print(f"    horizon: 平均 {np.mean(sub_hor):+.1f}, "
              f">0 {sum(1 for x in sub_hor if x>0)}/{len(sub_hor)}, t={th:+.2f}")
