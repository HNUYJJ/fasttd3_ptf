"""Capture/restore learner RNG streams around transfer-only construction."""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import torch


@dataclass
class GlobalRngState:
    python_state: object
    numpy_state: tuple
    torch_cpu_state: torch.Tensor
    torch_device_state: torch.Tensor | None
    device: torch.device

    @classmethod
    def capture(cls, device: torch.device | str) -> "GlobalRngState":
        device = torch.device(device)
        device_state = (
            torch.cuda.get_rng_state(device).clone()
            if device.type == "cuda"
            else None
        )
        return cls(
            python_state=random.getstate(),
            numpy_state=np.random.get_state(),
            torch_cpu_state=torch.get_rng_state().clone(),
            torch_device_state=device_state,
            device=device,
        )

    def restore(self) -> None:
        random.setstate(self.python_state)
        np.random.set_state(self.numpy_state)
        torch.set_rng_state(self.torch_cpu_state)
        if self.torch_device_state is not None:
            torch.cuda.set_rng_state(self.torch_device_state, self.device)


def capture_rng_after_reference_construction(
    factory: Callable[[], Any], device: torch.device | str
) -> GlobalRngState:
    """Measure a reference constructor's RNG effect without retaining it.

    The caller's RNG streams are restored before return.  The returned state is
    the state *after* the reference constructor and can later be restored after
    a larger optional subsystem has been built.  This makes an abstaining path
    match the actual target-only scaffold, rather than an imaginary path that
    constructed no common modules at all.
    """

    before = GlobalRngState.capture(device)
    reference = factory()
    after = GlobalRngState.capture(device)
    del reference
    before.restore()
    return after
