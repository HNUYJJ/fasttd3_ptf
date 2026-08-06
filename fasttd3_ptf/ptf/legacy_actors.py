"""加载旧格式源策略 checkpoint 所需的 actor 架构(兼容层,只在加载时使用)。

这些类原位于 my_fasttd3_ptf/models/(legacy 模块化实现,2026-06-10 已移除)。
保留原因:`checkpoints/sources/*/final.pt` 等仍在使用的源策略 checkpoint 是
`FastTD3Agent.state_dict.v1` 格式,SourcePolicy 需要按 `actor_kwargs.model_class`
("Actor" / "UpstreamFastTD3Actor")重建同构网络来加载权重。新训练一律走
官方 FastTD3 路径("OfficialFastTD3Actor" 分支),不再产生这两种格式。
"""
from __future__ import annotations

from typing import Any

import torch
from torch import nn

from fasttd3_ptf.ptf.option_module import build_mlp


def init_last_layer_small(module: nn.Module, scale: float = 1e-3) -> None:
    for m in reversed(list(module.modules())):
        if isinstance(m, nn.Linear):
            nn.init.uniform_(m.weight, -scale, scale)
            nn.init.uniform_(m.bias, -scale, scale)
            break


def _action_bounds(action_dim: int, action_low=None, action_high=None) -> tuple[torch.Tensor, torch.Tensor]:
    if action_low is None or action_high is None:
        low = -torch.ones(action_dim, dtype=torch.float32)
        high = torch.ones(action_dim, dtype=torch.float32)
    else:
        low = torch.as_tensor(action_low, dtype=torch.float32).view(-1)
        high = torch.as_tensor(action_high, dtype=torch.float32).view(-1)
    if low.numel() != action_dim or high.numel() != action_dim:
        raise ValueError(f"Action bounds must have {action_dim} values, got {low.numel()} and {high.numel()}")
    return low, high


class Actor(nn.Module):
    """legacy 模块化实现的 tanh-bounded MLP actor(checkpoint_format=FastTD3Agent.state_dict.v1)。"""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims=(512, 256, 128),
        action_low=None,
        action_high=None,
    ):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.hidden_dims = tuple(int(x) for x in hidden_dims)
        self.net = build_mlp(self.obs_dim, self.hidden_dims, self.action_dim)
        init_last_layer_small(self.net, scale=1e-3)
        if action_low is None or action_high is None:
            low = -torch.ones(self.action_dim)
            high = torch.ones(self.action_dim)
        else:
            low = torch.as_tensor(action_low, dtype=torch.float32).view(-1)
            high = torch.as_tensor(action_high, dtype=torch.float32).view(-1)
        self.register_buffer("action_low", low)
        self.register_buffer("action_high", high)
        self.register_buffer("action_scale", (high - low) / 2.0)
        self.register_buffer("action_bias", (high + low) / 2.0)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        raw = torch.tanh(self.net(obs))
        return raw * self.action_scale + self.action_bias

    def clamp_action(self, action: torch.Tensor) -> torch.Tensor:
        return torch.max(torch.min(action, self.action_high), self.action_low)

    def export_kwargs(self) -> dict:
        return {
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "hidden_dims": list(self.hidden_dims),
            "action_low": self.action_low.detach().cpu().tolist(),
            "action_high": self.action_high.detach().cpu().tolist(),
        }


class UpstreamFastTD3Actor(nn.Module):
    """复刻上游 FastTD3 actor 架构的 legacy 副本(model_class=UpstreamFastTD3Actor)。

    与官方 fast_td3.Actor 的差别:动作有 scale/bias 缓冲、噪声缓冲非持久化
    (源策略 checkpoint 不应依赖训练时的 num_envs)。
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        *,
        num_envs: int = 1,
        init_scale: float = 0.01,
        hidden_dim: int = 512,
        std_min: float = 0.001,
        std_max: float = 0.4,
        action_low=None,
        action_high=None,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_envs = int(num_envs)
        self.init_scale = float(init_scale)
        self.std_min_value = float(std_min)
        self.std_max_value = float(std_max)

        self.net = nn.Sequential(
            nn.Linear(self.obs_dim, self.hidden_dim, device=device),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2, device=device),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, self.hidden_dim // 4, device=device),
            nn.ReLU(),
        )
        self.fc_mu = nn.Sequential(
            nn.Linear(self.hidden_dim // 4, self.action_dim, device=device),
            nn.Tanh(),
        )
        nn.init.normal_(self.fc_mu[0].weight, 0.0, self.init_scale)
        nn.init.constant_(self.fc_mu[0].bias, 0.0)

        low, high = _action_bounds(self.action_dim, action_low, action_high)
        self.register_buffer("action_low", low.to(device=device))
        self.register_buffer("action_high", high.to(device=device))
        self.register_buffer("action_scale", (self.action_high - self.action_low) / 2.0)
        self.register_buffer("action_bias", (self.action_high + self.action_low) / 2.0)
        self.register_buffer("std_min", torch.as_tensor(self.std_min_value, device=device))
        self.register_buffer("std_max", torch.as_tensor(self.std_max_value, device=device))
        noise_scales = torch.rand(self.num_envs, 1, device=device) * (self.std_max - self.std_min) + self.std_min
        self.register_buffer("noise_scales", noise_scales, persistent=False)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        raw = self.fc_mu(self.net(obs))
        return raw * self.action_scale + self.action_bias

    def _ensure_noise_shape(self, batch_size: int, device: torch.device) -> None:
        if self.noise_scales.shape[0] == batch_size and self.noise_scales.device == device:
            return
        noise_scales = torch.rand(batch_size, 1, device=device) * (self.std_max - self.std_min) + self.std_min
        self.noise_scales = noise_scales
        self.num_envs = int(batch_size)

    def explore(self, obs: torch.Tensor, dones: torch.Tensor | None = None, deterministic: bool = False) -> torch.Tensor:
        self._ensure_noise_shape(obs.shape[0], obs.device)
        if dones is not None and dones.numel() > 0 and bool(dones.any()):
            new_scales = torch.rand_like(self.noise_scales) * (self.std_max - self.std_min) + self.std_min
            dones_view = dones.to(device=obs.device, dtype=torch.bool).view(-1, 1)
            self.noise_scales.copy_(torch.where(dones_view, new_scales, self.noise_scales))

        action = self(obs)
        if deterministic:
            return action
        return action + torch.randn_like(action) * self.noise_scales

    def clamp_action(self, action: torch.Tensor) -> torch.Tensor:
        return torch.max(torch.min(action, self.action_high), self.action_low)

    def export_kwargs(self) -> dict[str, Any]:
        return {
            "model_class": "UpstreamFastTD3Actor",
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "num_envs": self.num_envs,
            "init_scale": self.init_scale,
            "hidden_dim": self.hidden_dim,
            "std_min": self.std_min_value,
            "std_max": self.std_max_value,
            "action_low": self.action_low.detach().cpu().tolist(),
            "action_high": self.action_high.detach().cpu().tolist(),
        }
