#!/usr/bin/env python3
"""N1 机制描述性测量：直接读取 truck 的 upright 与 package 分量。

**DESCRIPTIVE ONLY** —— 无判据、无裁决、不改动 N1 的任何 verdict。
存在的理由：`return_mean` 与 `progress_max_dx_mean` 的负相关只是聚合观察，
而 `upright` 此前只能由 `return/1000/(1+package)` 反推（该反推隐含
"upright 与 package_reward 在 episode 内不相关"，未经验证）。
`tasks.py:68` 把 `reward_info` 整个并进 info，故可直接测。

同时记录每步位移增量，用于分辨"移动多"与"直立差"的先后关系。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from probe_lib import load_student  # noqa: E402
from p0_evaluator import _make_env, EPISODE_STEPS  # noqa: E402

PANEL = [s * 1000 + r for s in (11, 23) for r in range(8)]


@torch.no_grad()
def run_episode(env, actor, obs_norm, device, seed: int) -> dict:
    np.random.seed(seed)
    obs, _ = env.reset(seed=seed)
    x0 = float(env.unwrapped.data.qpos[0])
    upr, pkg, rew = [], [], 0.0
    max_dx = 0.0
    speeds = []
    prev_x = x0
    for _ in range(EPISODE_STEPS):
        obs_t = torch.as_tensor(obs, device=device, dtype=torch.float32).unsqueeze(0)
        action = actor(obs_norm(obs_t)).squeeze(0).cpu().numpy()
        obs, reward, terminated, truncated, info = env.step(action)
        rew += float(reward)
        if "upright" in info:
            upr.append(float(info["upright"]))
        if "reward_robot_package_truck" in info:
            pkg.append(float(info["reward_robot_package_truck"]))
        x = float(env.unwrapped.data.qpos[0])
        speeds.append(abs(x - prev_x))
        prev_x = x
        max_dx = max(max_dx, x - x0)
        if terminated or truncated:
            break
    return {
        "seed": seed, "return": rew, "steps": len(speeds),
        "upright_mean": float(np.mean(upr)) if upr else None,
        "package_mean": float(np.mean(pkg)) if pkg else None,
        "progress_max_dx": max_dx,
        "path_length": float(np.sum(speeds)),   # 总走过的路程（非净位移）
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    actor, _, obs_norm, _ = load_student(args.checkpoint, device)[:4]
    env = _make_env("h1hand-truck-v0")
    eps = [run_episode(env, actor, obs_norm, device, s) for s in PANEL]

    def m(k):
        vals = [e[k] for e in eps if e[k] is not None]
        return float(np.mean(vals)) if vals else None

    out = {
        "status": "DESCRIPTIVE_ONLY",
        "note": "无判据、不裁决；仅用于机制解释",
        "checkpoint": args.checkpoint, "arm": args.arm, "learner_seed": args.seed,
        "aggregate": {k: m(k) for k in
                      ("return", "upright_mean", "package_mean",
                       "progress_max_dx", "path_length", "steps")},
        "episodes": eps,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1))
    a = out["aggregate"]
    print(f"[{args.arm} s{args.seed}] return={a['return']:.1f} "
          f"upright={a['upright_mean']:.4f} pkg={a['package_mean']:.4f} "
          f"dx={a['progress_max_dx']:.3f} path={a['path_length']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
