#!/usr/bin/env python3
"""接口正确性 smoke：低层 reaching controller 是否真的响应高层 setpoint。

**这是工程验证，不是科学裁决。** 判据写在此处并先于运行：

1. `RESPONSIVE`：+x 与 −x 两个恒定 setpoint 下，手的净位移在 x 上符号相反，
   且两者差值 > 0.02 m。这证明高层指令确实经低层被执行。
2. `UNRESPONSIVE`：符号相同或差值 ≤ 0.02 m —— 说明接口没接通
   （obs 映射错、权重没加载、setpoint 被吞），属 silent corruption，
   必须先修好再谈任何实验。

之所以必须做：`SingleReachWrapper` 的低层是 `TorchModel(55, 19)`，
而 h1hand 的 obs 是 163 维、action 61 维。中间的索引映射
（`body_idxs` / `act_idxs`）若错位，环境仍能 step 且不报错。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hb_interface_env import make_env  # noqa: E402

STEPS = 200
DELTA_MIN = 0.02


def run(env, setpoint: np.ndarray, seed: int = 0) -> dict:
    np.random.seed(seed)
    obs, _ = env.reset(seed=seed)
    u = env.unwrapped
    # 左手末端位置（SingleReachWrapper 控制的就是左手）
    def hand_xyz():
        return np.array(u.named.data.xpos["left_hand"]) if hasattr(u, "named") \
            else np.array(u.data.body("left_hand").xpos)
    p0 = hand_xyz().copy()
    dists = []
    for _ in range(STEPS):
        obs, r, term, trunc, info = env.step(setpoint.astype(np.float32))
        if "hand_dist" in info:
            dists.append(float(info["hand_dist"]))
        if term or trunc:
            break
    p1 = hand_xyz().copy()
    return {"disp": (p1 - p0).tolist(), "dx": float(p1[0] - p0[0]),
            "hand_dist_first": dists[0] if dists else None,
            "hand_dist_last": dists[-1] if dists else None}


def main() -> int:
    env = make_env("h1hand-push-v0", policy_type="reach_single")
    print(f"action_space={env.action_space.shape}（应为 (3,)）\n")

    pos = run(env, np.array([+1.0, 0.0, 0.0]))
    neg = run(env, np.array([-1.0, 0.0, 0.0]))
    print(f"setpoint +x : 手位移 dx={pos['dx']:+.4f}  全位移={np.round(pos['disp'],4).tolist()}")
    print(f"setpoint −x : 手位移 dx={neg['dx']:+.4f}  全位移={np.round(neg['disp'],4).tolist()}")

    gap = pos["dx"] - neg["dx"]
    opposite = (pos["dx"] > 0) != (neg["dx"] > 0)
    print(f"\n两者 dx 之差 = {gap:+.4f}（阈值 {DELTA_MIN}）；符号相反 = {opposite}")

    if opposite and abs(gap) > DELTA_MIN:
        print("判定: RESPONSIVE —— 接口接通，高层 setpoint 确实驱动低层")
        return 0
    print("判定: UNRESPONSIVE —— 接口未接通，禁止在此基础上做任何实验")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
