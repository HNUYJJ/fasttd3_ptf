"""并行 HumanoidBench 环境，支持透传 goal-conditioned 低层接口。

与 vendor 版 `HumanoidBenchEnv` 的**唯一差别**是 `make_env` 会把
`policy_type` / `policy_path` / `mean_path` / `var_path` 传给 `gym.make`，
从而启用官方的 `SingleReachWrapper`。其余（SubprocVecEnv、张量搬运、
`time_outs` 语义、`max_episode_steps` 分支）逐行沿用 vendor 实现，
以保证 flat 与 interface 两臂除接口外完全同构。

vendor 文件本身**未被修改**。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor" / "fast_td3"))
sys.path.insert(0, str(ROOT / "vendor" / "humanoid_bench_pkg"))
sys.path.insert(0, str(ROOT / "src"))

import gymnasium as gym  # noqa: E402
from gymnasium.wrappers import TimeLimit  # noqa: E402
from stable_baselines3.common.vec_env import SubprocVecEnv  # noqa: E402

from hb_interface_env import interface_kwargs  # noqa: E402

#: 与 vendor `humanoid_bench_env.py:25-34` 完全一致的短 episode 任务表
SHORT_EPISODE_ENVS = {
    "h1hand-push-v0", "h1-push-v0", "h1hand-cube-v0", "h1cube-v0",
    "h1hand-basketball-v0", "h1-basketball-v0", "h1hand-kitchen-v0", "h1-kitchen-v0",
}


def make_env(env_name: str, rank: int, policy_type: str | None = None,
             render_mode=None, seed: int = 0):
    max_episode_steps = 500 if env_name in SHORT_EPISODE_ENVS else 1000
    kwargs = dict(interface_kwargs(policy_type or "flat"))

    def _init():
        import humanoid_bench  # noqa: F401
        env = gym.make(env_name, render_mode=render_mode, **kwargs)
        env = TimeLimit(env, max_episode_steps=max_episode_steps)
        env.unwrapped.seed(seed + rank)
        return env

    return _init


class InterfaceHumanoidBenchEnv:
    """并行封装；`policy_type=None` 时行为与 vendor 版逐位一致。"""

    def __init__(self, env_name: str, num_envs: int = 1,
                 policy_type: str | None = None, render_mode=None, device=None):
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sim_device = device
        self.num_envs = num_envs
        self.policy_type = policy_type or "flat"

        self.envs = SubprocVecEnv(
            [make_env(env_name, i, policy_type=policy_type, render_mode=render_mode)
             for i in range(num_envs)]
        )
        self.max_episode_steps = 500 if env_name in SHORT_EPISODE_ENVS else 1000
        self.asymmetric_obs = False
        self.num_obs = self.envs.observation_space.shape[-1]
        self.num_actions = self.envs.action_space.shape[-1]

    def reset(self):
        obs = self.envs.reset()
        return torch.from_numpy(obs).to(device=self.sim_device, dtype=torch.float)

    def render(self):
        assert self.num_envs == 1, "Currently only supports single environment rendering"
        return self.envs.render()

    def step(self, actions):
        assert isinstance(actions, torch.Tensor)
        actions = actions.cpu().numpy()
        observations, rewards, dones, raw_infos = self.envs.step(actions)

        infos = dict()
        infos["observations"] = {"raw": {"obs": observations.copy()}}
        truncateds = np.zeros_like(dones)
        for i in range(self.num_envs):
            if raw_infos[i].get("TimeLimit.truncated", False):
                truncateds[i] = True
                infos["observations"]["raw"]["obs"][i] = raw_infos[i]["terminal_observation"]

        observations = torch.from_numpy(observations).to(device=self.sim_device, dtype=torch.float)
        rewards = torch.from_numpy(rewards).to(device=self.sim_device, dtype=torch.float)
        dones = torch.from_numpy(dones).to(device=self.sim_device)
        truncateds = torch.from_numpy(truncateds).to(device=self.sim_device)
        infos["observations"]["raw"]["obs"] = torch.from_numpy(
            infos["observations"]["raw"]["obs"]
        ).to(device=self.sim_device, dtype=torch.float)
        infos["time_outs"] = truncateds
        return observations, rewards, dones, infos

    # ── 供诊断使用：取回底层 info（vendor 版没有暴露）──────────────────
    def step_with_raw(self, actions):
        """与 step 相同，但额外返回各 env 的原始 info dict 列表。"""
        assert isinstance(actions, torch.Tensor)
        a = actions.cpu().numpy()
        observations, rewards, dones, raw_infos = self.envs.step(a)
        return observations, rewards, dones, raw_infos
