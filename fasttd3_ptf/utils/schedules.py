from __future__ import annotations

import math


def linear_schedule(step: int, start: float, end: float, duration: int) -> float:
    if duration <= 0:
        return end
    mix = min(max(step / float(duration), 0.0), 1.0)
    return start + mix * (end - start)


class LinearScheduler:
    def __init__(self, start: float, end: float, duration: int):
        self.start = float(start)
        self.end = float(end)
        self.duration = int(duration)

    def __call__(self, step: int) -> float:
        return linear_schedule(step, self.start, self.end, self.duration)


class ReleasedPTFTanhScheduler:
    """Normalized author-released PTF transfer schedule.

    Released PTF uses ``0.5 + tanh(3 - c3 * progress) / 2``.  HumanoidBench
    training is step-budgeted rather than episode-budgeted, so ``duration``
    maps the released schedule's argument from +3 to -3 over the requested
    budget while ``scale`` retains an explicit loss-scale adapter.
    """

    def __init__(self, scale: float, duration: int):
        self.scale = float(scale)
        self.duration = int(duration)

    def __call__(self, step: int) -> float:
        if self.duration <= 0:
            return 0.0
        progress = min(max(float(step) / float(self.duration), 0.0), 1.0)
        return self.scale * (0.5 + 0.5 * math.tanh(3.0 - 6.0 * progress))
