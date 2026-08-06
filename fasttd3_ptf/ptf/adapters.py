from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from fasttd3_ptf.ptf.action_schema import h1hand_default_action_schema


class ObsAdapter:
    def __call__(self, target_obs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class ActionAdapter:
    def __call__(self, source_action: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


@dataclass
class IdentityObsAdapter(ObsAdapter):
    output_dim: int | None = None
    allow_truncate: bool = False
    allow_pad: bool = False

    def __call__(self, target_obs: torch.Tensor) -> torch.Tensor:
        x = target_obs
        if self.output_dim is None or x.shape[-1] == self.output_dim:
            return x
        if x.shape[-1] > self.output_dim:
            if not self.allow_truncate:
                raise ValueError(
                    f"IdentityObsAdapter got obs_dim={x.shape[-1]} but expected {self.output_dim}; "
                    "use an explicit slice/robot_only adapter for cross-task transfer."
                )
            return x[..., : self.output_dim]
        if not self.allow_pad:
            raise ValueError(
                f"IdentityObsAdapter got obs_dim={x.shape[-1]} but expected {self.output_dim}; "
                "use an explicit adapter instead of implicit zero padding."
            )
        pad = torch.zeros(*x.shape[:-1], self.output_dim - x.shape[-1], device=x.device, dtype=x.dtype)
        return torch.cat([x, pad], dim=-1)


@dataclass
class SliceObsAdapter(ObsAdapter):
    indices: list[int] | None = None
    start: int | None = None
    end: int | None = None
    output_dim: int | None = None
    allow_truncate: bool = False
    allow_pad: bool = False

    def __call__(self, target_obs: torch.Tensor) -> torch.Tensor:
        if self.indices is not None:
            idx = torch.as_tensor(self.indices, device=target_obs.device, dtype=torch.long)
            x = target_obs.index_select(-1, idx)
        else:
            start = 0 if self.start is None else int(self.start)
            end = target_obs.shape[-1] if self.end is None else int(self.end)
            x = target_obs[..., start:end]
        if self.output_dim is not None and x.shape[-1] != self.output_dim:
            if x.shape[-1] > self.output_dim:
                if not self.allow_truncate:
                    raise ValueError(
                        f"SliceObsAdapter produced dim={x.shape[-1]} but expected {self.output_dim}; "
                        "fix indices/start/end or set allow_truncate explicitly."
                    )
                x = x[..., : self.output_dim]
            else:
                if not self.allow_pad:
                    raise ValueError(
                        f"SliceObsAdapter produced dim={x.shape[-1]} but expected {self.output_dim}; "
                        "fix the observation adapter instead of implicit zero padding."
                    )
                pad = torch.zeros(*x.shape[:-1], self.output_dim - x.shape[-1], device=x.device, dtype=x.dtype)
                x = torch.cat([x, pad], dim=-1)
        return x


@dataclass
class RobotOnlyObsAdapter(SliceObsAdapter):
    """Default HumanoidBench adapter: use the first source_obs_dim entries.

    HumanoidBench keeps robot proprioception first in state observations. For
    h1hand this is commonly 151 robot-only dimensions before task-specific
    object state.
    """


@dataclass
class ReachObsAdapter(SliceObsAdapter):
    """Generic reach adapter.

    Real projects should configure explicit indices for hand target/object pose.
    If no indices are provided, this falls back to the first source_obs_dim
    dimensions, which is safe for same-task smoke tests but should be inspected
    for target manipulation tasks.
    """


@dataclass
class HumanoidBenchRobotQposQvelAdapter(ObsAdapter):
    """Extract robot qpos/qvel from HumanoidBench full qpos+qvel observations.

    Some HumanoidBench manipulation tasks inherit Task.get_obs(), which returns
    full qpos followed by full qvel. For h1hand targets with task DoFs this is
    not the same as the first 151 entries. This adapter explicitly selects
    qpos[:robot_dof] and qvel[:robot_dof - 1].
    """

    qpos_dim: int
    robot_dof: int = 76
    output_dim: int | None = None

    def __call__(self, target_obs: torch.Tensor) -> torch.Tensor:
        qpos_dim = int(self.qpos_dim)
        robot_dof = int(self.robot_dof)
        qvel_start = qpos_dim
        qvel_end = qvel_start + robot_dof - 1
        if target_obs.shape[-1] < qvel_end:
            raise ValueError(
                f"HumanoidBenchRobotQposQvelAdapter needs obs_dim>={qvel_end}, got {target_obs.shape[-1]}"
            )
        x = torch.cat([target_obs[..., :robot_dof], target_obs[..., qvel_start:qvel_end]], dim=-1)
        expected = self.output_dim if self.output_dim is not None else (robot_dof * 2 - 1)
        if x.shape[-1] != expected:
            raise ValueError(f"HumanoidBenchRobotQposQvelAdapter produced dim={x.shape[-1]}, expected {expected}")
        return x


@dataclass
class ActionPassthroughAdapter(ActionAdapter):
    output_dim: int | None = None
    allow_truncate: bool = False
    allow_pad: bool = False

    def __call__(self, source_action: torch.Tensor) -> torch.Tensor:
        x = source_action
        if self.output_dim is None or x.shape[-1] == self.output_dim:
            return x
        if x.shape[-1] > self.output_dim:
            if not self.allow_truncate:
                raise ValueError(
                    f"ActionPassthroughAdapter got action_dim={x.shape[-1]} but expected {self.output_dim}; "
                    "use an explicit action_pad adapter with source_indices/target_indices."
                )
            return x[..., : self.output_dim]
        if not self.allow_pad:
            raise ValueError(
                f"ActionPassthroughAdapter got action_dim={x.shape[-1]} but expected {self.output_dim}; "
                "use an explicit action_pad adapter instead of implicit zero padding."
            )
        pad = torch.zeros(*x.shape[:-1], self.output_dim - x.shape[-1], device=x.device, dtype=x.dtype)
        return torch.cat([x, pad], dim=-1)


@dataclass
class ActionPadAdapter(ActionAdapter):
    output_dim: int
    fill_value: float = 0.0
    source_indices: list[int] | None = None
    target_indices: list[int] | None = None

    def __call__(self, source_action: torch.Tensor) -> torch.Tensor:
        out = torch.full(
            (*source_action.shape[:-1], self.output_dim),
            float(self.fill_value),
            device=source_action.device,
            dtype=source_action.dtype,
        )
        if self.source_indices is None and self.target_indices is None:
            n = min(source_action.shape[-1], self.output_dim)
            out[..., :n] = source_action[..., :n]
        else:
            if self.source_indices is None or self.target_indices is None:
                raise ValueError("source_indices and target_indices must be provided together")
            sidx = torch.as_tensor(self.source_indices, device=source_action.device, dtype=torch.long)
            tidx = torch.as_tensor(self.target_indices, device=source_action.device, dtype=torch.long)
            out[..., tidx] = source_action.index_select(-1, sidx)
        return out


def build_obs_adapter(spec: dict[str, Any] | str | None, source_obs_dim: int | None = None) -> ObsAdapter:
    if spec is None:
        return IdentityObsAdapter(output_dim=source_obs_dim)
    if isinstance(spec, str):
        spec = {"type": spec}
    typ = str(spec.get("type", "identity")).lower()
    output_dim = spec.get("output_dim", source_obs_dim)
    if typ in {"identity", "same"}:
        return IdentityObsAdapter(
            output_dim=output_dim,
            allow_truncate=bool(spec.get("allow_truncate", False)),
            allow_pad=bool(spec.get("allow_pad", False)),
        )
    if typ in {"slice", "indices"}:
        return SliceObsAdapter(
            indices=spec.get("indices"),
            start=spec.get("start"),
            end=spec.get("end"),
            output_dim=output_dim,
            allow_truncate=bool(spec.get("allow_truncate", False)),
            allow_pad=bool(spec.get("allow_pad", False)),
        )
    if typ in {"robot_only", "robotonly", "humanoid_robot_only"}:
        return RobotOnlyObsAdapter(
            indices=spec.get("indices"),
            start=spec.get("start", 0),
            end=spec.get("end", output_dim),
            output_dim=output_dim,
            allow_truncate=bool(spec.get("allow_truncate", False)),
            allow_pad=bool(spec.get("allow_pad", False)),
        )
    if typ in {"reach", "reach_obs"}:
        return ReachObsAdapter(
            indices=spec.get("indices"),
            start=spec.get("start", 0),
            end=spec.get("end", output_dim),
            output_dim=output_dim,
            allow_truncate=bool(spec.get("allow_truncate", False)),
            allow_pad=bool(spec.get("allow_pad", False)),
        )
    if typ in {"humanoidbench_robot_qpos_qvel", "hb_robot_qpos_qvel", "h1hand_full_state_robot"}:
        if "qpos_dim" not in spec:
            raise ValueError("humanoidbench_robot_qpos_qvel adapter requires qpos_dim")
        return HumanoidBenchRobotQposQvelAdapter(
            qpos_dim=int(spec["qpos_dim"]),
            robot_dof=int(spec.get("robot_dof", 76)),
            output_dim=output_dim,
        )
    raise ValueError(f"Unknown obs adapter type: {typ}")


def build_action_adapter(spec: dict[str, Any] | str | None, target_action_dim: int | None = None) -> ActionAdapter:
    if spec is None:
        return ActionPassthroughAdapter(output_dim=target_action_dim)
    if isinstance(spec, str):
        spec = {"type": spec}
    typ = str(spec.get("type", "passthrough")).lower()
    if typ in {"passthrough", "identity", "same"}:
        return ActionPassthroughAdapter(
            output_dim=spec.get("output_dim", target_action_dim),
            allow_truncate=bool(spec.get("allow_truncate", False)),
            allow_pad=bool(spec.get("allow_pad", False)),
        )
    if typ in {"pad", "action_pad"}:
        if target_action_dim is None and "output_dim" not in spec:
            raise ValueError("ActionPadAdapter needs target_action_dim or output_dim")
        return ActionPadAdapter(
            output_dim=int(spec.get("output_dim", target_action_dim)),
            fill_value=float(spec.get("fill_value", 0.0)),
            source_indices=spec.get("source_indices"),
            target_indices=spec.get("target_indices"),
        )
    raise ValueError(f"Unknown action adapter type: {typ}")


def build_action_mask(spec: dict[str, Any] | str | list[int] | None, action_dim: int, device: torch.device) -> torch.Tensor:
    mask = torch.zeros(action_dim, device=device, dtype=torch.float32)
    if spec is None:
        mask.fill_(1.0)
        return mask
    if isinstance(spec, list):
        idx = torch.as_tensor(spec, device=device, dtype=torch.long)
        mask[idx] = 1.0
        return mask
    if isinstance(spec, str):
        if spec in {"full", "all"}:
            mask.fill_(1.0)
            return mask
        if spec.startswith("first_"):
            n = int(spec.split("_", 1)[1])
            mask[:n] = 1.0
            return mask
        raise ValueError(f"Unknown action mask string: {spec}")
    typ = str(spec.get("type", "full")).lower()
    if typ in {"full", "all"}:
        mask.fill_(1.0)
    elif typ in {"indices", "index"}:
        idx = torch.as_tensor(spec.get("indices", []), device=device, dtype=torch.long)
        mask[idx] = 1.0
    elif typ in {"first_n", "first"}:
        mask[: int(spec["n"])] = 1.0
    elif typ == "ranges":
        for start, end in spec.get("ranges", []):
            mask[int(start) : int(end)] = 1.0
    elif typ in {"groups", "named_groups"}:
        schema_name = str(spec.get("schema", "h1hand_default")).lower()
        if schema_name not in {"h1hand", "h1hand_default"}:
            raise ValueError(f"Unknown action mask schema: {schema_name}")
        schema = h1hand_default_action_schema()
        if int(action_dim) != schema.dim:
            raise ValueError(f"Action mask schema {schema_name} expects action_dim={schema.dim}, got {action_dim}")
        group_mask = schema.mask(spec.get("groups", []), device=device)
        mask = torch.maximum(mask, group_mask)
        if spec.get("indices"):
            idx = torch.as_tensor(spec["indices"], device=device, dtype=torch.long)
            mask[idx] = 1.0
    else:
        raise ValueError(f"Unknown action mask type: {typ}")
    return mask
