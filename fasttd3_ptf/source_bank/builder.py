from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from fasttd3_ptf.config import save_yaml
from fasttd3_ptf.utils.checkpoint import load_json


def _is_h1hand_manifest(manifest: dict[str, Any]) -> bool:
    env_id = str(manifest.get("env_id", ""))
    robot = str(manifest.get("action_metadata", {}).get("robot", ""))
    return env_id.startswith("h1hand-") or robot == "h1hand" or int(manifest.get("action_dim", 0)) == 61


def _task_name(manifest: dict[str, Any], path: str) -> str:
    return f"{manifest.get('name', '')} {manifest.get('env_id', '')} {Path(path).stem}".lower()


def _h1hand_obs_adapter(manifest: dict[str, Any], path: str) -> dict[str, Any]:
    task = _task_name(manifest, path)
    if "reach" in task:
        return {"type": "reach"}
    return {"type": "robot_only"}


def _h1hand_action_mask(manifest: dict[str, Any], path: str) -> dict[str, Any]:
    task = _task_name(manifest, path)
    if "reach" in task:
        return {"type": "groups", "schema": "h1hand_default", "groups": ["arms", "hands"]}
    if "walk" in task or "run" in task:
        return {"type": "groups", "schema": "h1hand_default", "groups": ["legs_torso"]}
    if "stand" in task or "balance" in task:
        return {
            "type": "groups",
            "schema": "h1hand_default",
            "groups": ["legs_torso", "arms"],
        }
    return {"type": "groups", "schema": "h1hand_default", "groups": ["legs_torso"]}


def _is_identity_adapter(spec: Any) -> bool:
    return isinstance(spec, dict) and str(spec.get("type", "identity")).lower() in {"identity", "same"}


def _is_full_mask(spec: Any) -> bool:
    if spec is None:
        return True
    if isinstance(spec, str):
        return spec in {"full", "all"}
    return isinstance(spec, dict) and str(spec.get("type", "full")).lower() in {"full", "all"}


def build_source_bank(sources: list[str], output: str, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    specs = []
    for path in sources:
        manifest = load_json(path)
        obs_adapter = manifest.get("obs_adapter", {"type": "identity", "output_dim": manifest["obs_dim"]})
        action_mask = manifest.get("action_mask", {"type": "full"})
        if _is_h1hand_manifest(manifest):
            if _is_identity_adapter(obs_adapter):
                obs_adapter = _h1hand_obs_adapter(manifest, path)
            if _is_full_mask(action_mask):
                action_mask = _h1hand_action_mask(manifest, path)
        spec = {
            "name": manifest.get("name", Path(path).stem),
            "checkpoint": manifest["checkpoint"],
            "env_id": manifest.get("env_id"),
            "checkpoint_format": manifest.get("checkpoint_format", "FastTD3Agent.state_dict.v1"),
            "obs_dim": manifest["obs_dim"],
            "action_dim": manifest["action_dim"],
            "actor_hidden_dims": manifest.get("actor_hidden_dims", [512, 256, 128]),
            "action_low": manifest.get("action_low"),
            "action_high": manifest.get("action_high"),
            "normalizer": manifest.get("normalizer", {"obs": "checkpoint.obs_normalizer"}),
            "obs_adapter": obs_adapter,
            "action_adapter": manifest.get("action_adapter", {"type": "passthrough", "output_dim": manifest["action_dim"]}),
            "action_mask": action_mask,
            "compatibility_sigma": manifest.get("compatibility_sigma", 0.25),
            "obs_metadata": manifest.get("obs_metadata", {}),
            "action_metadata": manifest.get("action_metadata", {}),
            "metadata": manifest.get("metadata", {}),
        }
        specs.append(spec)
    cfg = {"sources": specs, "null_option": {"enabled": True, "name": "no_transfer"}}
    if overrides:
        cfg.update(overrides)
    save_yaml(cfg, output)
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", nargs="+", required=True, help="Source manifest JSON files")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    cfg = build_source_bank(args.sources, args.output)
    print(f"Wrote {args.output} with {len(cfg['sources'])} source policies")


if __name__ == "__main__":
    main()
