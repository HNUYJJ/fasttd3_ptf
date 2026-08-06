"""标签可识别性审计 v1(只读、零训练、零 GPU)。

问题
----
Cabinet@10k gate 返回 CABINET_UNCERTAIN:在 3 训练 seed × 每臂 32 评估 episode 下
无法可靠判定 source-specific transfer effect。因此在投入下一次 source 标定之前,
必须先回答:**已有的 student/scratch 数据里,是否存在一个比 Cabinet 更可测的
task × stage?**

盲化
----
筛选只允许使用**无源臂**数据。判据不依赖实验名猜测,而用训练代码自己打印的事实:

    train_ptf.py:989   print(f"Loaded source bank options: {source_bank.names()}")

    bank == ['null']        -> 无源臂,回报曲线可读
    bank 含任何非 null 条目  -> 有源臂,**立即停止读取该文件**,只计数
    该行缺失(旧代码路径)    -> unknown,排除

这样"哪个 source 有效"不可能泄漏进筛选,避免结果导向的任务挑选。

指标链
------
训练中的 `eval_avg_return` 是 num_eval_envs=128 的**面板均值**,故
    sigma_episode = sigma_panel(128) * sqrt(128)
Cabinet gate 的评估面板是 32 episodes,故
    SE_32 = sigma_episode / sqrt(32) = 2 * sigma_panel(128)

gate 判据是 3 个配对 seed 的 90% t 区间(df=2)不跨 0,故**最小可判定效应**
    U_min = t90 * sqrt(2) * SE_32 / sqrt(3) = 2.384 * SE_32
这是乐观下界(真实 sd(U) 还含 learner-seed 异质性,只会更大)。

sigma_panel 用两个互补估计量,取较差者作保守值:
  (a) 一阶差分 MAD —— 对平滑上升曲线会把学习增量误当噪声(高估)
  (b) 10k–30k 去趋势残差 MAD —— 对曲率大的曲线会把曲率误当噪声(高估)
两者都只会**高估**噪声,故 max(a,b) 是保守方向。

第二个必需维度:**stage 内相对学习速率** = trend_per_10k / r@20k。
Cabinet 失败的一个核心机制是该窗口学习停滞(11.6%);噪声再低,若该 stage 本身
没有多少学习可被干预放大,效应绝对值也会很小。

定标
----
两个已实测的锚点给出可用的判据边界:
    crawl   保守 U/|r| = 0.29,相对学习率 34.5%  -> 干净标签(实测成功)
    cabinet 保守 U/|r| = 1.19,相对学习率 11.6%  -> CABINET_UNCERTAIN(实测失败)

用法
----
    python scripts/analysis/label_identifiability_audit_v1.py <out.json>
"""
from __future__ import annotations

import glob
import json
import math
import os
import re
import statistics as st
import sys
from collections import defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EVAL_RE = re.compile(r"\[eval\]\s+step=(\d+)\s+return=(-?[\d.]+)\s+length=([\d.]+)")
BANK_RE = re.compile(r"Loaded source bank options:\s*(\[.*\])")
T90_DF2 = 2.919986
K = T90_DF2 * math.sqrt(2) / math.sqrt(3)  # 2.384
PANEL_RATIO = 2.0                          # sqrt(128/32)


# ---------------------------------------------------------------- 提取

def read_cfg(run_dir: str):
    p = os.path.join(run_dir, "files", "config.yaml")
    if not os.path.exists(p):
        return None, None, None
    env = exp = seed = None
    lines = open(p, errors="replace").read().splitlines()
    for i, ln in enumerate(lines):
        s = ln.strip()
        for key in ("env_name:", "exp_name:", "seed:"):
            if not s.startswith(key):
                continue
            for j in range(i + 1, min(i + 4, len(lines))):
                m = re.search(r"value:\s*(\S+)", lines[j])
                if not m:
                    continue
                v = m.group(1)
                if key == "env_name:" and env is None:
                    env = v
                elif key == "exp_name:" and exp is None:
                    exp = v
                elif key == "seed:" and seed is None:
                    try:
                        seed = int(v)
                    except ValueError:
                        pass
                break
    return env, exp, seed


def scan(run_dir: str):
    """返回 (kind, curve)。kind in {'nosrc','source','unknown'}。
    有源臂一经识别立即返回,其回报数值从不被解析(盲化保证)。"""
    p = os.path.join(run_dir, "files", "output.log")
    if not os.path.exists(p):
        return "unknown", []
    kind, curve = "unknown", []
    with open(p, errors="replace") as f:
        for ln in f:
            if kind == "unknown" and "Loaded source bank options" in ln:
                m = BANK_RE.search(ln)
                if m:
                    names = re.findall(r"'([^']*)'", m.group(1))
                    if not set(names) <= {"null"}:
                        return "source", []
                    kind = "nosrc"
                continue
            if kind == "nosrc" and "[eval]" in ln:
                m = EVAL_RE.search(ln)
                if m:
                    curve.append((int(m.group(1)), float(m.group(2)), float(m.group(3))))
    return kind, curve


# ---------------------------------------------------------------- 统计

def interp(curve, step):
    pts = [(s, r) for s, r, _ in curve]
    if not pts or step < pts[0][0] or step > pts[-1][0]:
        return None
    for i in range(len(pts) - 1):
        s0, r0 = pts[i]
        s1, r1 = pts[i + 1]
        if s0 <= step <= s1:
            return r0 if s1 == s0 else r0 + (r1 - r0) * (step - s0) / (s1 - s0)
    return pts[-1][1]


def mad(xs):
    m = st.median(xs)
    return 1.4826 * st.median([abs(x - m) for x in xs])


def sigma_firstdiff(curve, wmax=40000):
    pts = [(s, r) for s, r, _ in curve if s <= wmax]
    if len(pts) < 5:
        return None
    diffs = [abs(pts[i + 1][1] - pts[i][1]) for i in range(len(pts) - 1)]
    med = st.median(diffs)
    return med / 1.349 / math.sqrt(2) if med > 0 else None


def sigma_detrended(curve, lo=10000, hi=30000):
    pts = [(s, r) for s, r, _ in curve if lo <= s <= hi]
    if len(pts) < 5:
        return None, None
    n = len(pts)
    mx = sum(p[0] for p in pts) / n
    my = sum(p[1] for p in pts) / n
    sxx = sum((p[0] - mx) ** 2 for p in pts)
    if sxx == 0:
        return None, None
    b = sum((p[0] - mx) * (p[1] - my) for p in pts) / sxx
    a = my - b * mx
    res = [p[1] - (a + b * p[0]) for p in pts]
    s = mad(res)
    return (s if s > 0 else None), b * 10000


# ---------------------------------------------------------------- 主流程

def main():
    runs = sorted(glob.glob(os.path.join(REPO, "wandb/run-*"))) + \
           sorted(glob.glob(os.path.join(REPO, "wandb/offline-run-*")))
    per_env = defaultdict(lambda: {"curves": [], "n_source": 0, "n_unknown": 0})
    stats = defaultdict(int)
    for rd in runs:
        env, exp, seed = read_cfg(rd)
        if not env:
            stats["no_env"] += 1
            continue
        kind, curve = scan(rd)
        stats[kind] += 1
        if kind == "source":
            per_env[env]["n_source"] += 1
        elif kind == "unknown":
            per_env[env]["n_unknown"] += 1
        elif curve:
            per_env[env]["curves"].append({"exp": exp, "seed": seed, "curve": curve})

    tasks = {}
    for env, d in per_env.items():
        sd_f, sd_d, trends, r10s, r20s, ends = [], [], [], [], [], []
        for rec in d["curves"]:
            c = rec["curve"]
            sf = sigma_firstdiff(c)
            sdt, tr = sigma_detrended(c)
            if sf:
                sd_f.append(sf)
            if sdt:
                sd_d.append(sdt)
                trends.append(tr)
            for step, box in ((10000, r10s), (20000, r20s)):
                v = interp(c, step)
                if v is not None:
                    box.append(v)
            if c[-1][0] >= 90000:
                ends.append(c[-1][1])
        if len(sd_f) < 2 or len(sd_d) < 2 or not r20s:
            continue
        s_f, s_d = st.median(sd_f), st.median(sd_d)
        r20 = st.median(r20s)
        trend = st.median(trends) if trends else None
        u_f = K * PANEL_RATIO * s_f
        u_d = K * PANEL_RATIO * s_d
        r_end = st.median(ends) if ends else None
        cv = (st.pstdev(r20s) / abs(st.mean(r20s))) if len(r20s) >= 3 and st.mean(r20s) else None
        tasks[env] = {
            "n_source_free_curves": len(d["curves"]),
            "n_source_arm_runs_skipped": d["n_source"],
            "sigma_panel128_firstdiff": round(s_f, 2),
            "sigma_panel128_detrended": round(s_d, 2),
            "SE32_conservative": round(PANEL_RATIO * max(s_f, s_d), 2),
            "U_over_r_firstdiff": round(abs(u_f / r20), 3),
            "U_over_r_detrended": round(abs(u_d / r20), 3),
            "U_over_r_conservative": round(max(abs(u_f / r20), abs(u_d / r20)), 3),
            "rel_learning_rate_10k_20k": round(trend / r20, 3) if trend and r20 else None,
            "cross_seed_cv_at20k": round(cv, 3) if cv else None,
            "r_at_10k": round(st.median(r10s), 1) if r10s else None,
            "r_at_20k": round(r20, 1),
            "r_at_end": round(r_end, 1) if r_end else None,
        }

    report = {
        "audit": "label_identifiability_audit_v1",
        "scope": "zero-training, read-only; only bank==['null'] arms parsed",
        "runs_scanned": dict(stats),
        "total_run_dirs": len(runs),
        "measured_anchors": {
            "crawl": {"U_over_r_conservative": 0.29, "rel_learning_rate": 0.345,
                      "outcome": "clean labels (measured)"},
            "cabinet": {"U_over_r_conservative": 1.19, "rel_learning_rate": 0.116,
                        "outcome": "CABINET_UNCERTAIN (measured)"},
        },
        "tasks": tasks,
    }
    if len(sys.argv) > 1:
        json.dump(report, open(sys.argv[1], "w"), indent=1, ensure_ascii=False)

    print(f"扫描 {len(runs)} 个 run: {dict(stats)}\n")
    hdr = (f"{'env':28s} {'n':>2s} {'保守U/|r|':>10s} {'相对学习率':>11s} "
           f"{'seedCV':>7s} {'r@20k':>9s} {'r@end':>9s}")
    print(hdr); print("-" * len(hdr))
    for env, t in sorted(tasks.items(), key=lambda kv: kv[1]["U_over_r_conservative"]):
        def f(x, w=9, p=3):
            return f"{x:{w}.{p}f}" if isinstance(x, (int, float)) else " " * (w - 1) + "-"
        print(f"{env:28s} {t['n_source_free_curves']:2d} {f(t['U_over_r_conservative'],10)} "
              f"{f(t['rel_learning_rate_10k_20k'],11)} {f(t['cross_seed_cv_at20k'],7)} "
              f"{f(t['r_at_20k'],9,1)} {f(t['r_at_end'],9,1)}")
    print("\n定标: crawl 0.29/34.5% -> 实测成功;cabinet 1.19/11.6% -> 实测 UNCERTAIN")


if __name__ == "__main__":
    main()
