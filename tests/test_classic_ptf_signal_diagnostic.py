from __future__ import annotations

import torch

from scripts.analyze_classic_ptf_signal_offline import compatibility_summary


def test_compatibility_summary_matches_training_kernel() -> None:
    dist = torch.tensor([0.0, 0.125, 0.5])
    result = compatibility_summary(dist, sigma=0.25)
    expected = torch.exp(-dist / (2.0 * 0.25**2))

    assert result["mean"] == torch.mean(expected).item()
    assert result["fraction_gt_0p9"] == torch.mean((expected > 0.9).float()).item()
    assert result["fraction_lt_0p05"] == torch.mean((expected < 0.05).float()).item()
