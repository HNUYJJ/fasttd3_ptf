"""active perturbation recovery 探针(RBO-PTF Open Q2, ChatGPT 高价值版; 不训练)。

passive probe 局限: terrain 一摔即 done, 测不到"跌倒后恢复"。active probe 控制"被多大扰动",
直接测恢复能力: policy 先 settle 到站立 → 施加 root 速度+角速度冲击(推+倾倒) → 看能否站回来。

v2(加大幅度+角速度): v1 纯 1-3 m/s 线冲击太弱(min_upright 仍 0.86), 改为 linear+angular 混合,
按强度档(med/large/xl)分组, 重点看能把 min_upright 真正压到 <0.5(踉跄/半倒)的档上的 method 差异。

做法(单 env, 直接访问 MjData):
  stair/slide/pole × {safe,rand,scr} seed1。每 trial: reset+settle 到站立 → 冲击 → RECOVER 步,
  检测末段连续 K 步 upright>U_STAND 且未摔 done = recovered。
  4 方向(±x 前后推+pitch, ±y 侧推+roll) × 3 强度 × REPS。

用法: CUDA_VISIBLE_DEVICES=5 python scripts/probe_active_recovery.py
"""
from __future__ import annotations

import glob
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from fasttd3_ptf.official_fasttd3_ptf.paths import (
    ensure_fasttd3_import_path,
    ensure_humanoidbench_import_path,
)
from fasttd3_ptf.ptf.source_policy import SourcePolicy

ACTION_DIM = 61
OBS_DIM = 151
TASKS = ["stair", "slide", "pole"]
METHODS = ["safe", "rand", "scr"]

U_STAND = 0.75
H_STAND = 0.6
K = 15
SETTLE = 120
RECOVER = 250
REPS = 2

# 强度档: (linear m/s, angular rad/s)
STRENGTHS = {"med": (6.0, 5.0), "large": (9.0, 8.0), "xl": (13.0, 11.0)}
# 方向: (lin_axis, lin_sign, ang_axis, ang_sign) —— 推 + 同向倾倒
DIRS = {
    "+x": (0, +1, 4, +1), "-x": (0, -1, 4, -1),   # 前后推 + pitch(绕y)
    "+y": (1, +1, 3, -1), "-y": (1, -1, 3, +1),   # 侧推 + roll(绕x)
}


def find_ckpt(task, method):
    pats = glob.glob(
        f"models/h1hand-{task}-v0__h1hand_{task}_tp_{method}_s1_*__1_final.pt")
    return sorted(pats)[-1] if pats else None


def main():
    ensure_fasttd3_import_path()
    ensure_humanoidbench_import_path()
    import gymnasium as gym
    import humanoid_bench  # noqa: F401
    import mujoco

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # agg[(task,method,strength)] = dict(settled, rec, survive, min_ups[])
    agg = defaultdict(lambda: dict(settled=0, rec=0, survive=0, min_ups=[]))
    for task in TASKS:
        env = gym.make(f"h1hand-{task}-v0")
        u = env.unwrapped
        for method in METHODS:
            ck = find_ckpt(task, method)
            if ck is None:
                print(f"[{task} {method}] MISSING", flush=True)
                continue
            sp = SourcePolicy(
                f"{task}_{method}", ck, device, target_action_dim=ACTION_DIM,
                source_obs_dim=OBS_DIM, source_action_dim=ACTION_DIM,
                obs_adapter_spec={"type": "identity", "output_dim": OBS_DIM})

            def act(o):
                a = sp.act(torch.as_tensor(o, device=device, dtype=torch.float32))
                return np.asarray(a.cpu().numpy()).reshape(-1)

            t0 = time.time()
            seed = task.__hash__() % 1000 + 1000
            for sname, (lin, ang) in STRENGTHS.items():
                for dname, (la, lsgn, aa, asgn) in DIRS.items():
                    for _ in range(REPS):
                        seed += 1
                        out = env.reset(seed=seed)
                        obs = out[0] if isinstance(out, tuple) else out
                        ok = True
                        last_up = 0.0
                        for _ in range(SETTLE):
                            obs, rew, term, trunc, info = env.step(act(obs))
                            last_up = float(info.get("upright", 0.0))
                            if term or trunc:
                                ok = False
                                break
                        if not ok or last_up < U_STAND or float(u.data.qpos[2]) < H_STAND:
                            continue
                        key = (task, method, sname)
                        agg[key]["settled"] += 1
                        u.data.qvel[la] += lsgn * lin
                        u.data.qvel[aa] += asgn * ang
                        mujoco.mj_forward(u.model, u.data)
                        ups, done_fall = [], False
                        for _ in range(RECOVER):
                            obs, rew, term, trunc, info = env.step(act(obs))
                            ups.append(float(info.get("upright", np.nan)))
                            if term and not trunc:
                                done_fall = True
                                break
                            if trunc:
                                break
                        agg[key]["min_ups"].append(float(np.nanmin(ups)) if ups else np.nan)
                        if not done_fall:
                            agg[key]["survive"] += 1
                        if (not done_fall and len(ups) >= K
                                and all(x > U_STAND for x in ups[-K:])):
                            agg[key]["rec"] += 1
            print(f"[{task:5s} {method:4s}] done ({time.time()-t0:.0f}s)", flush=True)
        env.close()

    print("\n" + "=" * 92)
    print("active perturbation recovery v2: 按强度档分组")
    print("min_upright 远<0.75 = 冲击有效(真把它推到踉跄/倒); 看该档上 method 的 P(recover)/survive")
    print("=" * 92)
    print(f"{'task':6s} {'method':6s} {'strength':8s} {'settled':>7s} "
          f"{'min_up':>7s} {'P(recover)':>10s} {'survive':>8s}")
    for task in TASKS:
        for sname in STRENGTHS:
            for method in METHODS:
                d = agg.get((task, method, sname))
                if not d or d["settled"] == 0:
                    continue
                mu = float(np.nanmean(d["min_ups"])) if d["min_ups"] else float("nan")
                pr = d["rec"] / d["settled"]
                sv = d["survive"] / d["settled"]
                print(f"{task:6s} {method:6s} {sname:8s} {d['settled']:7d} "
                      f"{mu:7.2f} {pr:10.0%} {sv:8.0%}")
        print()


if __name__ == "__main__":
    main()
