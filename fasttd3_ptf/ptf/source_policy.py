from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from fasttd3_ptf.official_fasttd3_ptf import ensure_fasttd3_import_path
from fasttd3_ptf.ptf.legacy_actors import Actor, UpstreamFastTD3Actor
from fasttd3_ptf.ptf.adapters import build_action_adapter, build_action_mask, build_obs_adapter
from fasttd3_ptf.utils.checkpoint import load_json, load_torch
from fasttd3_ptf.utils.normalization import TensorNormalizer


def _strip_compile_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if not any(k.startswith("_orig_mod.") for k in state_dict):
        return state_dict
    return {k.replace("_orig_mod.", "", 1): v for k, v in state_dict.items()}


def _load_matching_state(module: nn.Module, state_dict: dict[str, torch.Tensor]) -> tuple[int, int, list[str]]:
    own = module.state_dict()
    filtered = {}
    skipped = []
    for key, value in state_dict.items():
        if key in own and tuple(own[key].shape) == tuple(value.shape):
            filtered[key] = value
        else:
            skipped.append(key)
    module.load_state_dict(filtered, strict=False)
    return len(filtered), len(skipped), skipped[:10]


class _IdentitySourceNormalizer:
    def normalize(self, x: torch.Tensor, update: bool = False) -> torch.Tensor:
        return x


class _FrozenOfficialEmpiricalNormalizer:
    """Inference-only adapter for upstream FastTD3 EmpiricalNormalization."""

    def __init__(self, state: dict[str, torch.Tensor], device: torch.device, eps: float = 1e-2):
        self.eps = float(eps)
        self.mean = torch.as_tensor(state["_mean"], device=device, dtype=torch.float32)
        self.std = torch.as_tensor(state["_std"], device=device, dtype=torch.float32)

    def normalize(self, x: torch.Tensor, update: bool = False) -> torch.Tensor:
        return (x - self.mean) / (self.std + self.eps)


class SourcePolicy(nn.Module):
    """Frozen FastTD3 actor used as a PTF source option."""

    def __init__(
        self,
        name: str,
        checkpoint: str | Path,
        device: torch.device,
        target_action_dim: int,
        obs_adapter_spec: dict[str, Any] | str | None = None,
        action_adapter_spec: dict[str, Any] | str | None = None,
        action_mask_spec: dict[str, Any] | str | list[int] | None = None,
        source_obs_dim: int | None = None,
        source_action_dim: int | None = None,
        actor_hidden_dims: list[int] | tuple[int, ...] | None = None,
        compatibility_sigma: float = 0.25,
    ):
        super().__init__()
        self.name = name
        self.checkpoint = str(checkpoint)
        self.device = device
        state = load_torch(checkpoint, map_location=device)

        actor_kwargs = state.get("actor_kwargs") or {}
        official_args = state.get("args") if isinstance(state.get("args"), dict) else None
        if official_args is not None and not actor_kwargs:
            if str(official_args.get("agent", "fasttd3")) != "fasttd3":
                raise ValueError(
                    f"Official source checkpoint {checkpoint} uses agent="
                    f"{official_args.get('agent')}; only official fasttd3 actors are supported for PTF sources."
                )
            actor_kwargs = {
                "model_class": "OfficialFastTD3Actor",
                "init_scale": official_args.get("init_scale", 0.01),
                "hidden_dim": official_args.get("actor_hidden_dim", 512),
                "std_min": official_args.get("std_min", 0.001),
                "std_max": official_args.get("std_max", 0.4),
            }
        if source_obs_dim is None:
            source_obs_dim = actor_kwargs.get("obs_dim") or actor_kwargs.get("n_obs")
        if source_action_dim is None:
            source_action_dim = actor_kwargs.get("action_dim") or actor_kwargs.get("n_act")
        if actor_hidden_dims is None:
            actor_hidden_dims = actor_kwargs.get("hidden_dims", [512, 256, 128])
        if source_obs_dim is None or source_action_dim is None:
            raise ValueError(f"Source {name} needs source_obs_dim/source_action_dim in manifest or checkpoint actor_kwargs")

        self.source_obs_dim = int(source_obs_dim)
        self.source_action_dim = int(source_action_dim)
        self.target_action_dim = int(target_action_dim)
        self.compatibility_sigma = float(compatibility_sigma)

        action_low = actor_kwargs.get("action_low", None)
        action_high = actor_kwargs.get("action_high", None)
        model_class = str(actor_kwargs.get("model_class", "Actor"))
        if model_class == "OfficialFastTD3Actor":
            ensure_fasttd3_import_path()
            from fast_td3 import Actor as OfficialActor  # type: ignore

            self.actor = OfficialActor(
                n_obs=self.source_obs_dim,
                n_act=self.source_action_dim,
                num_envs=1,
                device=device,
                init_scale=float(actor_kwargs.get("init_scale", 0.01)),
                hidden_dim=int(actor_kwargs.get("hidden_dim", actor_hidden_dims[0] if actor_hidden_dims else 512)),
                std_min=float(actor_kwargs.get("std_min", 0.001)),
                std_max=float(actor_kwargs.get("std_max", 0.4)),
            ).to(device)
        elif model_class == "UpstreamFastTD3Actor":
            self.actor = UpstreamFastTD3Actor(
                self.source_obs_dim,
                self.source_action_dim,
                num_envs=1,
                init_scale=float(actor_kwargs.get("init_scale", 0.01)),
                hidden_dim=int(actor_kwargs.get("hidden_dim", actor_hidden_dims[0] if actor_hidden_dims else 512)),
                std_min=float(actor_kwargs.get("std_min", 0.001)),
                std_max=float(actor_kwargs.get("std_max", 0.4)),
                action_low=action_low,
                action_high=action_high,
                device=device,
            ).to(device)
        else:
            self.actor = Actor(
                self.source_obs_dim,
                self.source_action_dim,
                hidden_dims=actor_hidden_dims,
                action_low=action_low,
                action_high=action_high,
            ).to(device)
        actor_state = state.get("actor_state_dict") or state.get("actor") or state.get("model")
        if actor_state is None:
            raise KeyError(f"Could not find actor_state_dict in source checkpoint: {checkpoint}")
        loaded_keys, skipped_keys, skipped_preview = _load_matching_state(self.actor, _strip_compile_prefix(actor_state))
        if loaded_keys == 0:
            raise ValueError(
                f"Source {name} loaded zero actor tensors from {checkpoint}; "
                "check source_obs_dim/source_action_dim/actor hidden size and checkpoint type."
            )
        if skipped_keys > 0:
            preview = ", ".join(skipped_preview)
            print(
                f"[SourcePolicy:{name}] loaded {loaded_keys} actor tensors from {checkpoint}; "
                f"skipped {skipped_keys} non-matching tensors"
                + (f" ({preview})" if preview else "")
            )
        self.actor.eval()
        for p in self.actor.parameters():
            p.requires_grad_(False)

        if official_args is not None and "obs_normalizer_state" in state:
            official_norm_state = state["obs_normalizer_state"] or {}
            if "_mean" in official_norm_state and "_std" in official_norm_state:
                self.obs_normalizer = _FrozenOfficialEmpiricalNormalizer(official_norm_state, device)
            else:
                self.obs_normalizer = _IdentitySourceNormalizer()
        else:
            self.obs_normalizer = TensorNormalizer(self.source_obs_dim, device, enabled=True)
        if "obs_normalizer" in state and hasattr(self.obs_normalizer, "load_state_dict"):
            self.obs_normalizer.load_state_dict(state["obs_normalizer"])
        elif "obs_rms" in state and hasattr(self.obs_normalizer, "rms"):
            self.obs_normalizer.rms.load_state_dict(state["obs_rms"])
        self.obs_adapter = build_obs_adapter(obs_adapter_spec, source_obs_dim=self.source_obs_dim)
        self.action_adapter = build_action_adapter(action_adapter_spec, target_action_dim=self.target_action_dim)
        self.register_buffer("action_mask", build_action_mask(action_mask_spec, self.target_action_dim, device))

    @torch.no_grad()
    def act(self, target_obs_raw: torch.Tensor) -> torch.Tensor:
        source_obs = self.obs_adapter(target_obs_raw.to(self.device, dtype=torch.float32))
        source_obs = self.obs_normalizer.normalize(source_obs, update=False)
        source_action = self.actor(source_obs)
        return self.action_adapter(source_action)

    @classmethod
    def from_spec(cls, spec: dict[str, Any], device: torch.device, target_action_dim: int) -> "SourcePolicy":
        # A spec can point to a manifest created by train_source.py. Values in the
        # explicit source-bank YAML override manifest values.
        manifest_data: dict[str, Any] = {}
        if "manifest" in spec:
            manifest_data = load_json(spec["manifest"])
        merged = {**manifest_data, **spec}
        name = str(merged.get("name", Path(merged["checkpoint"]).stem))
        return cls(
            name=name,
            checkpoint=merged["checkpoint"],
            device=device,
            target_action_dim=target_action_dim,
            obs_adapter_spec=merged.get("obs_adapter"),
            action_adapter_spec=merged.get("action_adapter"),
            action_mask_spec=merged.get("action_mask"),
            source_obs_dim=merged.get("obs_dim") or merged.get("source_obs_dim"),
            source_action_dim=merged.get("action_dim") or merged.get("source_action_dim"),
            actor_hidden_dims=merged.get("actor_hidden_dims"),
            compatibility_sigma=float(merged.get("compatibility_sigma", 0.25)),
        )

