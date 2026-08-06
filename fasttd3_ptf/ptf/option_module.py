"""PTF 的 option-value + termination 模块(主路径活跃代码)。

原位于 my_fasttd3_ptf/models/option.py;official 主路径(train_ptf.py)依赖它,
2026-06-10 结构整理时迁移至此,使 PTF 主线不再依赖已移除的 legacy 包。
"""
from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


def straight_through_clamp(value: torch.Tensor, bound: float) -> torch.Tensor:
    """Clamp in the forward pass while preserving identity gradients.

    A regular ``torch.clamp`` has zero gradient outside the interval and would
    merely replace sigmoid saturation with a hard-clamp dead zone.  The
    straight-through form keeps the forward sigmoid derivative bounded away
    from zero while allowing later non-greedy evidence to move the raw logit
    back toward the interior.
    """
    limit = float(bound)
    if limit <= 0.0:
        raise ValueError(f"straight-through clamp bound must be positive, got {limit}")
    clipped = value.clamp(min=-limit, max=limit)
    return value + (clipped - value).detach()


def build_mlp(
    input_dim: int,
    hidden_dims: Sequence[int],
    output_dim: int,
    activation: type[nn.Module] = nn.ReLU,
    output_activation: nn.Module | None = None,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    last = input_dim
    for h in hidden_dims:
        layers.append(nn.Linear(last, int(h)))
        layers.append(activation())
        last = int(h)
    layers.append(nn.Linear(last, output_dim))
    if output_activation is not None:
        layers.append(output_activation)
    return nn.Sequential(*layers)


class OptionModule(nn.Module):
    """Option-value and termination module for PTF.

    The last option is typically reserved as the null/self option.

    The default path preserves this project's historical architecture and
    bounded-beta adaptation. ``released_code_fidelity=True`` instead restores
    the author-released PTF network: one ReLU6 hidden layer, tanh option
    values, a bare sigmoid termination head, and zero termination bias.
    """

    def __init__(
        self,
        obs_dim: int,
        num_options: int,
        hidden_dims: Sequence[int] = (256, 256),
        beta_min: float = 0.05,
        beta_max: float = 0.95,
        beta_logit_clip: float | None = None,
        released_code_fidelity: bool = False,
    ):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.num_options = int(num_options)
        self.hidden_dims = tuple(int(x) for x in hidden_dims)
        self.released_code_fidelity = bool(released_code_fidelity)
        if len(self.hidden_dims) == 0:
            raise ValueError("OptionModule requires at least one hidden layer")
        if self.released_code_fidelity and len(self.hidden_dims) != 1:
            raise ValueError(
                "released-code fidelity requires exactly one hidden layer, "
                f"got {self.hidden_dims}"
            )
        if not (0.0 <= beta_min < beta_max <= 1.0):
            raise ValueError(f"need 0 <= beta_min < beta_max <= 1, got {beta_min}, {beta_max}")
        self.beta_min = float(beta_min)
        self.beta_max = float(beta_max)
        if beta_logit_clip is not None and float(beta_logit_clip) <= 0.0:
            raise ValueError(
                f"beta_logit_clip must be positive when enabled, got {beta_logit_clip}"
            )
        self.beta_logit_clip = (
            None if beta_logit_clip is None else float(beta_logit_clip)
        )
        if self.released_code_fidelity:
            if self.beta_logit_clip is not None:
                raise ValueError(
                    "released-code fidelity uses the author's bare sigmoid "
                    "termination head; beta_logit_clip must be disabled"
                )
            self.trunk = nn.Sequential(
                nn.Linear(self.obs_dim, self.hidden_dims[0]),
                nn.ReLU6(),
            )
        else:
            self.trunk = build_mlp(
                self.obs_dim,
                self.hidden_dims[:-1],
                self.hidden_dims[-1],
            )
        self.q_head = nn.Linear(self.hidden_dims[-1], self.num_options)
        self.beta_head = nn.Linear(self.hidden_dims[-1], self.num_options)
        if self.released_code_fidelity:
            # tf.random_normal_initializer(0., .01) in the released code.
            for layer in (self.trunk[0], self.q_head, self.beta_head):
                nn.init.normal_(layer.weight, mean=0.0, std=0.01)
                nn.init.zeros_(layer.bias)
        else:
            nn.init.constant_(self.beta_head.bias, -2.0)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(obs)
        q_logits = self.q_head(h)
        q = torch.tanh(q_logits) if self.released_code_fidelity else q_logits
        if self.released_code_fidelity:
            return q, torch.sigmoid(self.beta_head(h))
        # β is rescaled to [beta_min, beta_max] instead of the PTF paper's bare
        # sigmoid (Eq. 5). This keeps the exposed (1-β) transfer weight away from
        # exactly zero, but it is not an anti-saturation mechanism for the
        # internal beta-head logit.
        beta_logits = self.beta_head(h)
        if self.beta_logit_clip is not None:
            beta_logits = straight_through_clamp(beta_logits, self.beta_logit_clip)
        beta_unbounded = torch.sigmoid(beta_logits)
        beta = self.beta_min + (self.beta_max - self.beta_min) * beta_unbounded
        return q, beta

    def export_kwargs(self) -> dict:
        return {
            "obs_dim": self.obs_dim,
            "num_options": self.num_options,
            "hidden_dims": list(self.hidden_dims),
            "beta_min": self.beta_min,
            "beta_max": self.beta_max,
            "beta_logit_clip": self.beta_logit_clip,
            "released_code_fidelity": self.released_code_fidelity,
        }
