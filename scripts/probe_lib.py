"""探针脚本公共工具库。

历史上这些函数分散在 package 专项探针(probe_zero_shot_package.py)与 MCG 探针
(probe_modular_critic_gating.py)中；两条线均已停止并删除(2026-07-16 整理,
原件见 git 快照 a5cec9d),把仍被活跃分析脚本(analyze_tcritic_offline.py 等)
依赖的通用函数抽取到这里。
"""
from __future__ import annotations

import numpy as np
import torch

from fasttd3_ptf.official_fasttd3_ptf.paths import (
    ensure_fasttd3_import_path,
    ensure_humanoidbench_import_path,
)

# h1hand 全任务统一的 61 维动作空间(legs 0:10 / torso 10 / arms 11:21 / hands 21:61)
ACTION_DIM = 61


def make_env_fn(env_name: str, rank: int, max_episode_steps: int, seed: int = 0):
    """构造单个 HumanoidBench 环境的工厂函数(用于 AsyncVectorEnv 等)。"""

    def _init():
        ensure_humanoidbench_import_path()
        import gymnasium as gym
        import humanoid_bench  # noqa: F401  (注册 h1hand-* 环境)
        from gymnasium.wrappers import TimeLimit

        env = gym.make(env_name)
        env = TimeLimit(env, max_episode_steps=max_episode_steps)
        env.unwrapped.seed(seed + rank)
        return env

    return _init


def _strip_prefix(sd: dict) -> dict:
    """去掉 torch.compile 保存时引入的 _orig_mod. 前缀。"""
    if not any(k.startswith("_orig_mod.") for k in sd):
        return sd
    return {k.replace("_orig_mod.", "", 1): v for k, v in sd.items()}


class _FrozenNorm:
    """从 checkpoint 状态复原的冻结 obs 归一化(推断期只读,不再更新统计量)。"""

    def __init__(self, state: dict, device: torch.device, eps: float = 1e-2):
        self.eps = eps
        self.mean = torch.as_tensor(state["_mean"], device=device, dtype=torch.float32)
        self.std = torch.as_tensor(state["_std"], device=device, dtype=torch.float32)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / (self.std + self.eps)


def load_student(ckpt_path: str, device: torch.device):
    """从官方格式 checkpoint 重建 actor + distributional critic + 两套 normalizer。"""
    ensure_fasttd3_import_path()
    from fast_td3 import Actor, Critic

    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    args = state["args"]
    n_obs = int(torch.as_tensor(state["obs_normalizer_state"]["_mean"]).shape[-1])

    actor = Actor(
        n_obs=n_obs, n_act=ACTION_DIM, num_envs=1, device=device,
        init_scale=float(args.get("init_scale", 0.01)),
        hidden_dim=int(args.get("actor_hidden_dim", 512)),
        std_min=float(args.get("std_min", 0.001)),
        std_max=float(args.get("std_max", 0.4)),
    ).to(device)
    # noise_scales 是 per-env 探索噪声 buffer(shape 随训练 num_envs),与
    # deterministic 前向无关,剔除后非严格加载。
    actor_sd = {k: v for k, v in _strip_prefix(state["actor_state_dict"]).items() if k != "noise_scales"}
    actor.load_state_dict(actor_sd, strict=False)
    actor.eval()

    critic = Critic(
        n_obs=n_obs, n_act=ACTION_DIM,
        num_atoms=int(args["num_atoms"]),
        v_min=float(args["v_min"]), v_max=float(args["v_max"]),
        hidden_dim=int(args.get("critic_hidden_dim", 1024)),
        device=device,
    ).to(device)
    critic.load_state_dict(_strip_prefix(state["qnet_state_dict"]))
    critic.eval()

    obs_norm = _FrozenNorm(state["obs_normalizer_state"], device)
    critic_norm = _FrozenNorm(state["critic_obs_normalizer_state"], device)
    return actor, critic, obs_norm, critic_norm, int(state.get("global_step", -1))


@torch.no_grad()
def collect_states(envs, actor, obs_norm, device, steps: int, noise: float, seed: int) -> np.ndarray:
    """用 student actor(+探索噪声)rollout,收集原始 obs,近似该阶段 buffer 分布。"""
    g = torch.Generator(device="cpu").manual_seed(seed)
    obs = envs.reset()
    out = []
    for _ in range(steps):
        out.append(obs.copy())
        obs_t = torch.as_tensor(obs, device=device, dtype=torch.float32)
        act = actor(obs_norm(obs_t))
        if noise > 0:
            act = act + noise * torch.randn(act.shape, generator=g).to(device)
        act = act.clamp(-1.0, 1.0)
        obs, _, _, _ = envs.step(act.cpu().numpy())
    return np.concatenate(out, axis=0)
