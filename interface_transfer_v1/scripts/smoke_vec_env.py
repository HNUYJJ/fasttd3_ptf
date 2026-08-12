#!/usr/bin/env python3
"""并行环境 smoke：flat 与 interface 两种模式下张量契约一致。

判据（先于运行写定）：
- 两臂的 `num_obs`、`max_episode_steps`、各返回张量形状必须一致；
- `num_actions` 必须为 flat=61、interface=3；
- 任一不符 → 非零退出，禁止继续。

必须走真实脚本文件：`SubprocVecEnv` 用 spawn 启动子进程，会重新导入主模块。
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from interface_vec_env import InterfaceHumanoidBenchEnv  # noqa: E402

EXPECTED = {"flat": 61, "reach_single": 3}


def main() -> int:
    rows = {}
    for pt in (None, "reach_single"):
        tag = pt or "flat"
        envs = InterfaceHumanoidBenchEnv("h1hand-push-v0", num_envs=4, policy_type=pt)
        obs = envs.reset()
        a = torch.zeros(4, envs.num_actions, device=envs.sim_device)
        o2, r, d, info = envs.step(a)
        rows[tag] = {
            "num_obs": envs.num_obs, "num_actions": envs.num_actions,
            "max_steps": envs.max_episode_steps,
            "obs_shape": tuple(o2.shape), "rew_shape": tuple(r.shape),
            "done_shape": tuple(d.shape), "timeouts": tuple(info["time_outs"].shape),
            "raw_obs": tuple(info["observations"]["raw"]["obs"].shape),
        }
        envs.envs.close()
        print(f"[{tag:<13}] {rows[tag]}")

    f, i = rows["flat"], rows["reach_single"]
    ok = True
    if f["num_actions"] != EXPECTED["flat"] or i["num_actions"] != EXPECTED["reach_single"]:
        print(f"FAIL: num_actions 不符预期 {EXPECTED}"); ok = False
    for k in ("num_obs", "max_steps", "obs_shape", "rew_shape", "done_shape",
              "timeouts", "raw_obs"):
        if f[k] != i[k]:
            print(f"FAIL: 两臂 {k} 不一致 flat={f[k]} interface={i[k]}"); ok = False
    print("\n判定:", "CONTRACT_OK —— 两臂除动作维度外张量契约一致" if ok else "CONTRACT_MISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
