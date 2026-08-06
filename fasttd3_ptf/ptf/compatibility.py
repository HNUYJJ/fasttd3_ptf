from __future__ import annotations

import torch


def gaussian_action_compatibility_all(
    action: torch.Tensor,
    source_actions: torch.Tensor,
    masks: torch.Tensor,
    sigma: float | torch.Tensor = 0.25,
) -> torch.Tensor:
    """Soft compatibility between sampled actions and all source policies.

    action: [B, A]
    source_actions: [B, K, A]
    masks: [K, A] or [B, K, A]
    returns: [B, K]

    The compatibility is the Gaussian kernel of the action distance restricted
    to the source's action mask: ``exp(-dist^2 / (2 sigma^2))`` where ``dist`` is
    the mean squared distance over the masked dimensions. It replaces the
    released PTF code's binary, all-dimensions ``mu +/- 1 sigma`` support test
    under the TD3 deterministic-actor adaptation (the frozen TD3 actor exposes
    an action mean but no learned state-dependent action density).

    Edge case: a source whose action mask is all-zero is degenerate (it applies
    no teacher on any dimension). The masked diff is 0, so without the guard
    below ``dist=0 -> compat=exp(0)=1`` would silently treat it as fully
    compatible with every transition and let it absorb Q_o updates from noise.
    We force its compatibility to 0 so it does not participate in any Q_o
    update. Real source-bank configs never have an all-zero mask (each source
    controls at least one body group), so this guard is defensive and does not
    change any existing experiment's behaviour.
    """
    if source_actions.numel() == 0:
        return torch.empty(action.shape[0], 0, device=action.device)
    if masks.ndim == 2:
        masks = masks.unsqueeze(0).expand(action.shape[0], -1, -1)
    diff2 = ((action.unsqueeze(1) - source_actions) * masks).pow(2)
    mask_sum = masks.sum(dim=-1)
    denom = mask_sum.clamp_min(1.0)
    dist = diff2.sum(dim=-1) / denom
    if not isinstance(sigma, torch.Tensor):
        sigma_t = torch.as_tensor(float(sigma), device=action.device, dtype=action.dtype)
    else:
        sigma_t = sigma.to(device=action.device, dtype=action.dtype)
    if sigma_t.ndim == 1:
        sigma_t = sigma_t.view(1, -1)
    compat = torch.exp(-dist / (2.0 * sigma_t.pow(2).clamp_min(1e-6)))
    # Force zero compatibility for all-zero-mask sources (see docstring edge case).
    compat = torch.where(mask_sum > 0, compat, torch.zeros_like(compat))
    return compat
