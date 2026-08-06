from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class NormalizerConfig:
    enabled: bool = True
    clip: float = 5.0
    eps: float = 1e-5


class RunningMeanStd:
    def __init__(self, shape, device: torch.device, eps: float = 1e-4):
        self.mean = torch.zeros(shape, device=device, dtype=torch.float32)
        self.var = torch.ones(shape, device=device, dtype=torch.float32)
        self.count = torch.tensor(eps, device=device, dtype=torch.float32)

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        if x.numel() == 0:
            return
        x = x.detach().to(dtype=torch.float32)
        if x.ndim == self.mean.ndim:
            x = x.unsqueeze(0)
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        batch_count = torch.tensor(x.shape[0], device=x.device, dtype=torch.float32)
        self.update_from_moments(batch_mean, batch_var, batch_count)

    @torch.no_grad()
    def update_from_moments(self, batch_mean, batch_var, batch_count) -> None:
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta.pow(2) * self.count * batch_count / total_count
        new_var = m2 / total_count
        self.mean.copy_(new_mean)
        self.var.copy_(new_var.clamp_min(1e-8))
        self.count.copy_(total_count)

    def state_dict(self) -> dict:
        return {
            "mean": self.mean.detach().cpu(),
            "var": self.var.detach().cpu(),
            "count": self.count.detach().cpu(),
        }

    def load_state_dict(self, state: dict) -> None:
        self.mean.copy_(torch.as_tensor(state["mean"], device=self.mean.device, dtype=torch.float32))
        self.var.copy_(torch.as_tensor(state["var"], device=self.var.device, dtype=torch.float32))
        self.count.copy_(torch.as_tensor(state["count"], device=self.count.device, dtype=torch.float32))


class TensorNormalizer:
    def __init__(self, shape, device: torch.device, enabled: bool = True, clip: float = 5.0, eps: float = 1e-5):
        self.enabled = enabled
        self.clip = clip
        self.eps = eps
        self.rms = RunningMeanStd(shape, device=device)

    def normalize(self, x: torch.Tensor, update: bool = False) -> torch.Tensor:
        if not self.enabled:
            return x
        if update:
            self.rms.update(x)
        y = (x - self.rms.mean) / torch.sqrt(self.rms.var + self.eps)
        if self.clip is not None and self.clip > 0:
            y = torch.clamp(y, -self.clip, self.clip)
        return y

    def state_dict(self) -> dict:
        return {"enabled": self.enabled, "clip": self.clip, "eps": self.eps, "rms": self.rms.state_dict()}

    def load_state_dict(self, state: dict) -> None:
        self.enabled = bool(state.get("enabled", self.enabled))
        self.clip = float(state.get("clip", self.clip))
        self.eps = float(state.get("eps", self.eps))
        if "rms" in state:
            self.rms.load_state_dict(state["rms"])
