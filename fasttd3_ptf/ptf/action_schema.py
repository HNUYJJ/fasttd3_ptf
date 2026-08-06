"""h1hand 动作分组 schema(legs/torso/arms/hands),供 per-source action mask 使用。

原位于 my_fasttd3_ptf/envs/action_schema.py;ptf/adapters.py 依赖它,
2026-06-10 结构整理时迁移至此,使 PTF 主线不再依赖已移除的 legacy 包。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch


@dataclass(frozen=True)
class ActionSlice:
    name: str
    start: int
    end: int

    def mask(self, action_dim: int, device: torch.device | None = None) -> torch.Tensor:
        m = torch.zeros(int(action_dim), device=device, dtype=torch.float32)
        m[self.start:self.end] = 1.0
        return m


@dataclass
class ActionSchema:
    dim: int
    slices: tuple[ActionSlice, ...]

    def get(self, name: str) -> ActionSlice:
        for sl in self.slices:
            if sl.name == name:
                return sl
        raise KeyError(name)

    def mask(self, names: Iterable[str], device: torch.device | None = None) -> torch.Tensor:
        out = torch.zeros(self.dim, device=device, dtype=torch.float32)
        for name in names:
            out = torch.maximum(out, self.get(name).mask(self.dim, device=device))
        return out


def h1hand_default_action_schema() -> ActionSchema:
    # HumanoidBench h1hand_pos.xml actuator order:
    # 0-9 legs, 10 torso, 11-20 arms/wrists, 21-40 left Shadow hand,
    # 41-60 right Shadow hand. The Gym action space is normalized to [-1, 1]
    # but keeps this actuator ordering.
    return ActionSchema(
        dim=61,
        slices=(
            ActionSlice("legs", 0, 10),
            ActionSlice("torso", 10, 11),
            ActionSlice("legs_torso", 0, 11),
            ActionSlice("left_arm", 11, 16),
            ActionSlice("right_arm", 16, 21),
            ActionSlice("arms", 11, 21),
            ActionSlice("left_hand", 21, 41),
            ActionSlice("right_hand", 41, 61),
            ActionSlice("hands", 21, 61),
            # Backward-compatible aliases used by older configs.
            ActionSlice("left_arm_proximal", 11, 16),
            ActionSlice("right_arm_proximal", 16, 21),
            ActionSlice("left_fingers", 23, 41),
            ActionSlice("right_fingers", 43, 61),
            ActionSlice("upper_body_left", 10, 41),
            ActionSlice("upper_body_both", 10, 61),
        ),
    )
