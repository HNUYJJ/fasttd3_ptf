from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_action_distillation_loss(
    target_action: torch.Tensor,
    source_action: torch.Tensor,
    action_mask: torch.Tensor,
    loss_type: str = "huber",
    delta: float = 1.0,
) -> torch.Tensor:
    """Return per-sample masked action imitation loss."""
    if action_mask.ndim == 1:
        action_mask = action_mask.view(1, -1).expand_as(target_action)
    denom = action_mask.sum(dim=-1).clamp_min(1.0)
    if loss_type.lower() in {"huber", "smooth_l1", "smoothl1"}:
        loss = F.smooth_l1_loss(target_action * action_mask, source_action * action_mask, reduction="none", beta=delta)
    elif loss_type.lower() in {"mse", "l2"}:
        loss = (target_action - source_action).pow(2) * action_mask
    elif loss_type.lower() in {"l1", "mae"}:
        loss = (target_action - source_action).abs() * action_mask
    else:
        raise ValueError(f"Unknown distillation loss type: {loss_type}")
    return loss.sum(dim=-1) / denom
