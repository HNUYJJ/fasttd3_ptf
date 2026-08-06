from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from fasttd3_ptf.utils.checkpoint import load_json, save_json


@dataclass
class SourceManifest:
    name: str
    env_id: str
    checkpoint: str
    obs_dim: int
    action_dim: int
    actor_hidden_dims: list[int] = field(default_factory=lambda: [512, 256, 128])
    action_low: list[float] | None = None
    action_high: list[float] | None = None
    checkpoint_format: str = "FastTD3Agent.state_dict.v1"
    normalizer: dict[str, Any] = field(
        default_factory=lambda: {
            "obs": "checkpoint.obs_normalizer",
            "critic_obs": "checkpoint.critic_obs_normalizer",
        }
    )
    obs_adapter: dict[str, Any] = field(default_factory=lambda: {"type": "identity"})
    action_adapter: dict[str, Any] = field(default_factory=lambda: {"type": "passthrough"})
    action_mask: dict[str, Any] = field(default_factory=lambda: {"type": "full"})
    compatibility_sigma: float = 0.25
    eval_return: float | None = None
    obs_metadata: dict[str, Any] = field(default_factory=dict)
    action_metadata: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        save_json(self.to_dict(), path)

    @classmethod
    def load(cls, path: str | Path) -> "SourceManifest":
        return cls(**load_json(path))
