from __future__ import annotations

import argparse
from pathlib import Path

from fasttd3_ptf.source_bank.manifest import SourceManifest
from fasttd3_ptf.utils.checkpoint import load_torch


def _infer_dims_from_actor_state(state: dict) -> tuple[int, int]:
    actor_state = state.get("actor_state_dict") or state.get("actor") or state.get("model")
    if not isinstance(actor_state, dict):
        return 0, 0
    weights = [
        tensor
        for key, tensor in actor_state.items()
        if key.endswith(".weight") and getattr(tensor, "ndim", 0) == 2
    ]
    if not weights:
        return 0, 0
    obs_dim = int(weights[0].shape[1])
    action_dim = int(weights[-1].shape[0])
    return obs_dim, action_dim


def export_source_manifest(checkpoint: str, env_id: str, output: str, name: str | None = None, compatibility_sigma: float = 0.25) -> SourceManifest:
    state = load_torch(checkpoint, map_location="cpu")
    actor_kwargs = state.get("actor_kwargs", {})
    official_args = state.get("args") if isinstance(state.get("args"), dict) else {}
    inferred_obs_dim, inferred_action_dim = _infer_dims_from_actor_state(state)
    obs_dim = int(actor_kwargs.get("obs_dim", state.get("obs_dim", inferred_obs_dim)))
    action_dim = int(actor_kwargs.get("action_dim", state.get("action_dim", inferred_action_dim)))
    actor_hidden_dim = int(official_args.get("actor_hidden_dim", 512))
    hidden = list(actor_kwargs.get("hidden_dims", state.get("actor_hidden_dims", [actor_hidden_dim, actor_hidden_dim // 2, actor_hidden_dim // 4])))
    if obs_dim <= 0 or action_dim <= 0:
        raise ValueError("Checkpoint does not expose actor dimensions and they could not be inferred from actor_state_dict.")
    if official_args:
        checkpoint_format = f"OfficialFastTD3.{official_args.get('agent', 'fasttd3')}.state_dict.v1"
    else:
        checkpoint_format = str(state.get("agent_type", "FastTD3Agent")) + ".state_dict.v1"
    manifest = SourceManifest(
        name=name or Path(checkpoint).parent.name or Path(checkpoint).stem,
        env_id=env_id,
        checkpoint=checkpoint,
        obs_dim=obs_dim,
        action_dim=action_dim,
        actor_hidden_dims=hidden,
        action_low=actor_kwargs.get("action_low"),
        action_high=actor_kwargs.get("action_high"),
        checkpoint_format=checkpoint_format,
        normalizer={
            "obs": (
                "checkpoint.obs_normalizer_state"
                if "obs_normalizer_state" in state
                else "checkpoint.obs_normalizer" if "obs_normalizer" in state else "missing"
            ),
            "critic_obs": (
                "checkpoint.critic_obs_normalizer_state"
                if "critic_obs_normalizer_state" in state
                else "checkpoint.critic_obs_normalizer" if "critic_obs_normalizer" in state else "missing"
            ),
            "reward": "checkpoint.reward_normalizer" if "reward_normalizer" in state else "missing",
        },
        obs_adapter={"type": "identity", "output_dim": obs_dim},
        action_adapter={"type": "passthrough", "output_dim": action_dim},
        action_mask={"type": "full"},
        compatibility_sigma=compatibility_sigma,
        obs_metadata={
            "schema": "h1hand_robot_first_plus_task" if env_id.startswith("h1hand-") else "flat_box",
            "robot": "h1hand" if env_id.startswith("h1hand-") else None,
        },
        action_metadata={
            "schema": "h1hand_default" if env_id.startswith("h1hand-") and action_dim == 61 else "flat_box",
            "robot": "h1hand" if env_id.startswith("h1hand-") else None,
        },
    )
    manifest.save(output)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--env-id", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--name", default=None)
    parser.add_argument("--compatibility-sigma", type=float, default=0.25)
    args = parser.parse_args()
    manifest = export_source_manifest(args.checkpoint, args.env_id, args.output, args.name, args.compatibility_sigma)
    print(f"Wrote source manifest for {manifest.name} to {args.output}")


if __name__ == "__main__":
    main()
