from __future__ import annotations

from typing import Any

import torch
from torch import nn

from fasttd3_ptf.config import load_yaml
from fasttd3_ptf.ptf.source_policy import SourcePolicy


class SourcePolicyBank(nn.Module):
    def __init__(self, sources: list[SourcePolicy], target_action_dim: int, device: torch.device, null_option: bool = True):
        super().__init__()
        self.sources = nn.ModuleList(sources)
        self.target_action_dim = int(target_action_dim)
        self.device = device
        self.null_option = bool(null_option)
        self.num_sources = len(self.sources)
        self.null_option_idx = self.num_sources if self.null_option else -1
        self.num_options = self.num_sources + (1 if self.null_option else 0)
        if self.num_sources > 0:
            masks = torch.stack([s.action_mask for s in self.sources], dim=0)
            sigmas = torch.tensor([s.compatibility_sigma for s in self.sources], device=device, dtype=torch.float32)
        else:
            masks = torch.empty(0, self.target_action_dim, device=device)
            sigmas = torch.empty(0, device=device)
        self.register_buffer("source_masks", masks)
        self.register_buffer("source_sigmas", sigmas)

    @classmethod
    def from_config(
        cls,
        cfg_or_path: str | dict[str, Any],
        device: torch.device,
        target_action_dim: int,
    ) -> "SourcePolicyBank":
        cfg = load_yaml(cfg_or_path) if isinstance(cfg_or_path, str) else cfg_or_path
        null_cfg = cfg.get("null_option", True)
        null_enabled = bool(null_cfg.get("enabled", True)) if isinstance(null_cfg, dict) else bool(null_cfg)
        sources = [
            SourcePolicy.from_spec(spec, device=device, target_action_dim=target_action_dim)
            for spec in cfg.get("sources", [])
        ]
        return cls(sources, target_action_dim=target_action_dim, device=device, null_option=null_enabled)

    @torch.no_grad()
    def act_all(self, target_obs_raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.num_sources == 0:
            return (
                torch.empty(target_obs_raw.shape[0], 0, self.target_action_dim, device=self.device),
                torch.empty(0, self.target_action_dim, device=self.device),
            )
        actions = [src.act(target_obs_raw).to(dtype=torch.float32) for src in self.sources]
        return torch.stack(actions, dim=1), self.source_masks

    @torch.no_grad()
    def act_selected(self, target_obs_raw: torch.Tensor, option_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = target_obs_raw.shape[0]
        out_action = torch.zeros(batch, self.target_action_dim, device=self.device, dtype=torch.float32)
        out_mask = torch.zeros(batch, self.target_action_dim, device=self.device, dtype=torch.float32)
        active = torch.zeros(batch, device=self.device, dtype=torch.bool)
        option_ids = option_ids.to(self.device, dtype=torch.long).view(-1)
        for i, src in enumerate(self.sources):
            idx = (option_ids == i).nonzero(as_tuple=False).view(-1)
            if idx.numel() == 0:
                continue
            act = src.act(target_obs_raw.index_select(0, idx)).to(dtype=out_action.dtype)
            out_action.index_copy_(0, idx, act)
            out_mask.index_copy_(0, idx, src.action_mask.view(1, -1).expand(idx.numel(), -1))
            active[idx] = True
        return out_action, out_mask, active

    def names(self) -> list[str]:
        names = [s.name for s in self.sources]
        if self.null_option:
            names.append("null")
        return names
