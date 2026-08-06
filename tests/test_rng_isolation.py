from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fasttd3_ptf.official_fasttd3_ptf.rng_isolation import (
    GlobalRngState,
    capture_rng_after_reference_construction,
)
from fasttd3_ptf.ptf.option_module import OptionModule


def test_transfer_construction_does_not_shift_learner_rng_streams() -> None:
    random.seed(17)
    np.random.seed(17)
    torch.manual_seed(17)
    state = GlobalRngState.capture("cpu")
    expected = (
        random.random(),
        float(np.random.rand()),
        torch.rand(8),
    )
    state.restore()
    _ = torch.nn.Sequential(torch.nn.Linear(16, 32), torch.nn.Linear(32, 4))
    _ = [random.random() for _ in range(5)]
    _ = np.random.rand(5)
    state.restore()
    actual = (
        random.random(),
        float(np.random.rand()),
        torch.rand(8),
    )
    assert actual[0] == expected[0]
    assert actual[1] == expected[1]
    assert torch.equal(actual[2], expected[2])


def test_exact_abstention_matches_empty_bank_option_scaffold_rng() -> None:
    """A 10-option abstaining build must leave the 1-option scratch stream."""

    torch.manual_seed(41)
    _ = OptionModule(12, 1, (32, 32))
    expected = torch.rand(16)

    torch.manual_seed(41)
    scratch_after = capture_rng_after_reference_construction(
        lambda: OptionModule(12, 1, (32, 32)), "cpu"
    )
    _ = OptionModule(12, 10, (32, 32))
    scratch_after.restore()
    actual = torch.rand(16)

    assert torch.equal(actual, expected)
