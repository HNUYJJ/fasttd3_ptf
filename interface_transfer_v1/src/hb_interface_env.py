"""带 goal-conditioned 低层接口的 HumanoidBench 环境构造。

隔离说明：本目录不 import 主线的 `fasttd3_ptf.*`，只用 `vendor/` 下自带的
HumanoidBench 与 FastTD3 副本，因此对 bootstrap 主线零影响。

官方 HumanoidBench 已实现 hierarchical 控制（`env.py` 按 `policy_type` 包上
`SingleReachWrapper` / `DoubleReach*Wrapper`），但主线的 `humanoid_bench_env.py`
只调 `gym.make(env_name)`、不透传这些 kwargs，所以接口一直没被启用。
本模块补上这层透传，不改动 vendor 里的任何官方代码。

接口语义（读自 `wrappers.py:82-130`）：
- 高层 action 变为 3 维（单手）/ 6 维（双手），是**末端目标的增量**，
  乘 `max_delta` 后累加到 `last_target`，再 clip 到任务自带的
  `htarget_low/high`；
- 低层 `TorchModel(55, 19)` 只输出 **19 维 body action**，手部不受其控制；
- 因此 h1hand（61 维）上，hands 维度由 wrapper 自己补零/保持。
"""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

VENDOR = Path(__file__).resolve().parents[1] / "vendor"
HB_PKG = VENDOR / "humanoid_bench_pkg"
REACH_DATA = HB_PKG / "data"

#: 官方预训练低层 reaching controller 的三件套（权重 + obs 归一化统计）
INTERFACE_ASSETS = {
    "reach_single": REACH_DATA / "reach_one_hand",
    "reach_double_absolute": REACH_DATA / "reach_two_hands",
    "reach_double_relative": REACH_DATA / "reach_two_hands",
}


def _ensure_vendor_on_path() -> None:
    import sys
    p = str(HB_PKG)
    if p not in sys.path:
        sys.path.insert(0, p)


def interface_kwargs(policy_type: str) -> dict:
    """返回官方 wrapper 需要的 policy_path / mean_path / var_path。"""
    if policy_type in (None, "", "flat"):
        return {}
    if policy_type not in INTERFACE_ASSETS:
        raise ValueError(f"未知 policy_type: {policy_type}；可选 {list(INTERFACE_ASSETS)}")
    d = INTERFACE_ASSETS[policy_type]
    paths = {
        "policy_path": d / "torch_model.pt",
        "mean_path": d / "mean.npy",
        "var_path": d / "var.npy",
    }
    for k, v in paths.items():
        if not v.exists():
            raise FileNotFoundError(f"接口资产缺失 {k}: {v}")
    return {"policy_type": policy_type, **{k: str(v) for k, v in paths.items()}}


def make_env(env_name: str, policy_type: str | None = None,
             max_episode_steps: int = 1000, **extra):
    """构造环境；`policy_type=None/flat` 即原始 61 维 flat 控制。"""
    _ensure_vendor_on_path()
    import gymnasium as gym
    import humanoid_bench  # noqa: F401  注册 h1hand-*-v0
    from gymnasium.wrappers import TimeLimit

    kwargs = dict(interface_kwargs(policy_type or "flat"))
    kwargs.update(extra)
    env = gym.make(env_name, **kwargs)
    return TimeLimit(env, max_episode_steps=max_episode_steps)
