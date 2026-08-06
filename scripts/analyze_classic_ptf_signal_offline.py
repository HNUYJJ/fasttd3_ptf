"""Frozen-rollout signal diagnostic for classic PTF on walk -> hurdle.

This script performs no learning and never mutates a checkpoint.  For each
learned-PTF checkpoint it rolls out the frozen deterministic student on a
fixed seed panel, then evaluates:

* the student/source masked action-distance distribution;
* counterfactual compatibility weights for predeclared sigma values;
* the Q_walk - Q_null ordering learned by the actual sigma=1.5 run;
* beta conditioned on whether each option is greedy or non-greedy.

Important estimand boundary: checkpoints do not contain replay observations,
the historical current option, or selector RNG state.  These are frozen-policy
cross-sections on freshly collected student occupancy, not reconstructions of
the training trajectory.  Likewise, changing sigma below only recomputes the
compatibility weights; it does not create a Q network trained with that sigma.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from probe_lib import ACTION_DIM, load_student  # noqa: E402
from fasttd3_ptf.official_fasttd3_ptf.paths import (  # noqa: E402
    ensure_humanoidbench_import_path,
)
from fasttd3_ptf.ptf.option_module import OptionModule  # noqa: E402
from fasttd3_ptf.ptf.source_bank import SourcePolicyBank  # noqa: E402


DEFAULT_CHECKPOINT_GLOB = (
    "models/h1hand-hurdle-v0__classic_ptf_hurdle_walk_v1_"
    "ptf_s[123]_formal_20260721T1710Z__*_*.pt"
)
DEFAULT_SIGMAS = (0.25, 0.5, 1.0, 1.5)
EVAL_BASE_SEEDS = (11, 23, 37, 53)
EVAL_RANKS = (0, 1)
MAX_EPISODE_STEPS = 1000
EXPECTED_STEPS = (25_000, 50_000, 75_000, 100_000)
EXPECTED_SEEDS = (1, 2, 3)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    ).stdout.strip()


def _make_env(env_name: str):
    ensure_humanoidbench_import_path()
    import gymnasium as gym
    import humanoid_bench  # noqa: F401
    from gymnasium.wrappers import TimeLimit

    return TimeLimit(gym.make(env_name), max_episode_steps=MAX_EPISODE_STEPS)


@torch.no_grad()
def _collect_frozen_rollout_states(
    env_name: str,
    actor,
    obs_norm,
    device: torch.device,
) -> tuple[torch.Tensor, list[dict]]:
    """Collect raw observations under a deterministic frozen student actor."""

    env = _make_env(env_name)
    states: list[np.ndarray] = []
    episodes: list[dict] = []
    try:
        for base_seed in EVAL_BASE_SEEDS:
            for rank in EVAL_RANKS:
                reset_seed = base_seed * 1000 + rank
                np.random.seed(reset_seed)
                obs, _ = env.reset(seed=reset_seed)
                episode_return = 0.0
                episode_steps = 0
                for _ in range(MAX_EPISODE_STEPS):
                    states.append(np.asarray(obs, dtype=np.float32).copy())
                    obs_t = torch.as_tensor(
                        obs, device=device, dtype=torch.float32
                    ).unsqueeze(0)
                    action = actor(obs_norm(obs_t)).clamp(-1.0, 1.0)
                    obs, reward, terminated, truncated, _info = env.step(
                        action.squeeze(0).cpu().numpy()
                    )
                    episode_return += float(reward)
                    episode_steps += 1
                    if terminated or truncated:
                        break
                episodes.append(
                    {
                        "reset_seed": reset_seed,
                        "return": episode_return,
                        "steps": episode_steps,
                    }
                )
    finally:
        env.close()
    return torch.as_tensor(np.stack(states), device=device), episodes


def _quantiles(values: torch.Tensor) -> dict[str, float]:
    values = values.detach().float().reshape(-1).cpu()
    probs = torch.tensor([0.01, 0.10, 0.25, 0.50, 0.75, 0.90, 0.99])
    result = torch.quantile(values, probs)
    return {
        name: float(value)
        for name, value in zip(
            ("p01", "p10", "p25", "p50", "p75", "p90", "p99"),
            result,
            strict=True,
        )
    }


def _basic_summary(values: torch.Tensor) -> dict[str, float | dict[str, float]]:
    flat = values.detach().float().reshape(-1)
    return {
        "mean": float(flat.mean().cpu()),
        "std": float(flat.std(unbiased=False).cpu()),
        "quantiles": _quantiles(flat),
    }


def compatibility_summary(dist: torch.Tensor, sigma: float) -> dict:
    """Summarize the exact scalar Gaussian compatibility used by training."""

    weights = torch.exp(-dist / (2.0 * float(sigma) ** 2))
    total = weights.sum()
    ess = total.square() / weights.square().sum().clamp_min(1e-12)
    n = max(int(weights.numel()), 1)
    return {
        **_basic_summary(weights),
        "sigma": float(sigma),
        "effective_sample_fraction": float((ess / n).cpu()),
        "fraction_lt_0p05": float((weights < 0.05).float().mean().cpu()),
        "fraction_0p1_to_0p9": float(
            ((weights >= 0.1) & (weights <= 0.9)).float().mean().cpu()
        ),
        "fraction_gt_0p9": float((weights > 0.9).float().mean().cpu()),
    }


def _conditional_beta_summary(
    beta: torch.Tensor,
    q: torch.Tensor,
    option_idx: int,
) -> dict:
    greedy = q.argmax(dim=1) == int(option_idx)
    beta_o = beta[:, option_idx]

    def summarize(mask: torch.Tensor) -> dict:
        if not bool(mask.any()):
            return {"count": 0, "mean": None, "quantiles": None}
        selected = beta_o[mask]
        return {
            "count": int(mask.sum().cpu()),
            "mean": float(selected.mean().cpu()),
            "quantiles": _quantiles(selected),
        }

    return {
        "when_greedy": summarize(greedy),
        "when_non_greedy": summarize(~greedy),
    }


@torch.no_grad()
def _analyze_checkpoint(
    checkpoint: Path,
    device: torch.device,
    sigmas: Iterable[float],
    source_bank: SourcePolicyBank | None,
) -> tuple[dict, SourcePolicyBank]:
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if "option_state_dict" not in state:
        raise ValueError(f"learned-PTF option state missing: {checkpoint}")
    args = state.get("args") or {}
    ptf_cfg = state.get("ptf_cfg") or {}
    env_name = str(args.get("env_name"))
    seed = int(args.get("seed"))
    step = int(state.get("global_step"))
    if env_name != "h1hand-hurdle-v0":
        raise ValueError(f"unexpected env_name={env_name}: {checkpoint}")
    if state.get("source_names") != ["walk", "null"]:
        raise ValueError(
            f"unexpected source_names={state.get('source_names')}: {checkpoint}"
        )

    actor, _critic, obs_norm, _critic_norm, loaded_step = load_student(
        str(checkpoint), device
    )
    if loaded_step != step:
        raise ValueError(f"checkpoint step mismatch {loaded_step} != {step}")

    if source_bank is None:
        source_bank = SourcePolicyBank.from_config(
            ptf_cfg["source_bank"], device=device, target_action_dim=ACTION_DIM
        )
        source_bank.eval()
    if source_bank.names() != ["walk", "null"]:
        raise ValueError(f"unexpected bank options: {source_bank.names()}")

    option_module = OptionModule(**state["option_kwargs"]).to(device)
    option_module.load_state_dict(state["option_state_dict"], strict=True)
    option_module.eval()

    raw_obs, episodes = _collect_frozen_rollout_states(
        env_name, actor, obs_norm, device
    )
    batch_size = 2048
    dist_parts: list[torch.Tensor] = []
    q_parts: list[torch.Tensor] = []
    beta_parts: list[torch.Tensor] = []
    for begin in range(0, raw_obs.shape[0], batch_size):
        raw_batch = raw_obs[begin : begin + batch_size]
        norm_batch = obs_norm(raw_batch)
        student_action = actor(norm_batch).clamp(-1.0, 1.0)
        source_actions, masks = source_bank.act_all(raw_batch)
        if source_actions.shape[1] != 1:
            raise ValueError("diagnostic expects exactly one non-null source")
        mask = masks[0].view(1, -1)
        dist = ((student_action - source_actions[:, 0]) * mask).square().sum(dim=1)
        dist = dist / mask.sum().clamp_min(1.0)
        q, beta = option_module(norm_batch)
        dist_parts.append(dist.cpu())
        q_parts.append(q.cpu())
        beta_parts.append(beta.cpu())

    dist = torch.cat(dist_parts)
    q = torch.cat(q_parts)
    beta = torch.cat(beta_parts)
    q_diff = q[:, 0] - q[:, 1]
    greedy = q.argmax(dim=1)
    q_gap = q.max(dim=1).values - q.min(dim=1).values
    adaptive_margin = 0.8 * q_gap
    max_q = q.max(dim=1).values
    advantage = q - max_q[:, None] + adaptive_margin[:, None]
    advantage = advantage.clamp(
        min=-adaptive_margin[:, None].clamp_min(1e-6),
        max=adaptive_margin[:, None].clamp_min(1e-6),
    )

    return (
        {
            "checkpoint": {
                "path": str(checkpoint.resolve()),
                "sha256": _sha256(checkpoint),
                "seed": seed,
                "global_step": step,
                "trained_compatibility_sigma": float(source_bank.source_sigmas[0].cpu()),
            },
            "occupancy": {
                "kind": "fresh deterministic frozen-student rollouts",
                "eval_base_seeds": list(EVAL_BASE_SEEDS),
                "eval_ranks": list(EVAL_RANKS),
                "state_count": int(raw_obs.shape[0]),
                "episodes": episodes,
            },
            "action_mean_squared_distance": _basic_summary(dist),
            "counterfactual_compatibility": {
                str(float(sigma)): compatibility_summary(dist, float(sigma))
                for sigma in sigmas
            },
            "trained_q_beta": {
                "q_walk_minus_null": _basic_summary(q_diff),
                "q_gap": _basic_summary(q_gap),
                "walk_argmax_fraction": float((greedy == 0).float().mean()),
                "null_argmax_fraction": float((greedy == 1).float().mean()),
                "beta_walk": _basic_summary(beta[:, 0]),
                "beta_null": _basic_summary(beta[:, 1]),
                "beta_walk_conditioned": _conditional_beta_summary(beta, q, 0),
                "beta_null_conditioned": _conditional_beta_summary(beta, q, 1),
                "advantage_walk_positive_fraction": float(
                    (advantage[:, 0] > 0).float().mean()
                ),
                "advantage_walk_negative_fraction": float(
                    (advantage[:, 0] < 0).float().mean()
                ),
                "advantage_null_positive_fraction": float(
                    (advantage[:, 1] > 0).float().mean()
                ),
                "advantage_null_negative_fraction": float(
                    (advantage[:, 1] < 0).float().mean()
                ),
            },
        },
        source_bank,
    )


def _discover_checkpoints(pattern: str) -> list[Path]:
    checkpoints = [Path(path) for path in sorted(glob.glob(pattern))]
    if not checkpoints:
        raise FileNotFoundError(f"no checkpoints match: {pattern}")

    identities: set[tuple[int, int]] = set()
    for checkpoint in checkpoints:
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        identity = (int((state.get("args") or {})["seed"]), int(state["global_step"]))
        if identity in identities:
            raise ValueError(f"duplicate seed/step checkpoint {identity}: {checkpoint}")
        identities.add(identity)
    expected = {(seed, step) for seed in EXPECTED_SEEDS for step in EXPECTED_STEPS}
    if identities != expected:
        raise ValueError(
            f"checkpoint matrix mismatch: missing={sorted(expected - identities)}, "
            f"extra={sorted(identities - expected)}"
        )
    return checkpoints


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-glob", default=DEFAULT_CHECKPOINT_GLOB)
    parser.add_argument(
        "--checkpoint",
        action="append",
        default=None,
        help="Explicit checkpoint(s) for a smoke diagnostic; skips the formal 3x4 matrix check.",
    )
    parser.add_argument("--sigmas", default=",".join(str(x) for x in DEFAULT_SIGMAS))
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    out = Path(args.out)
    if out.exists():
        raise FileExistsError(f"refusing to overwrite existing result: {out}")
    sigmas = tuple(float(value) for value in args.sigmas.split(","))
    if any(value <= 0 for value in sigmas):
        raise ValueError("all sigma values must be positive")

    if args.checkpoint:
        checkpoints = [Path(value) for value in args.checkpoint]
        missing = [str(path) for path in checkpoints if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"explicit checkpoints missing: {missing}")
    else:
        checkpoints = _discover_checkpoints(args.checkpoint_glob)
    device = torch.device(args.device)
    runs = []
    for index, checkpoint in enumerate(checkpoints, start=1):
        print(f"[signal-diagnostic] {index}/{len(checkpoints)} {checkpoint.name}")
        # Explicit smoke comparisons may mix source-bank configs (for example,
        # trained sigma 1.5 vs 0.5), so construct the frozen bank from each
        # checkpoint's own config instead of reusing the first one.
        result, _source_bank = _analyze_checkpoint(
            checkpoint, device, sigmas, None
        )
        runs.append(result)

    report = {
        "protocol": {
            "scientific_scope": (
                "frozen-policy cross-sections; no replay/current-option reconstruction; "
                "counterfactual sigma scan changes weights only, not trained Q"
            ),
            "checkpoint_glob": args.checkpoint_glob,
            "sigmas": list(sigmas),
            "expected_seeds": list(EXPECTED_SEEDS),
            "expected_steps": list(EXPECTED_STEPS),
            "max_episode_steps": MAX_EPISODE_STEPS,
            "gradient_updates": 0,
        },
        "git_head": _git_head(),
        "git_dirty": bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=REPO_ROOT,
                text=True,
                stdout=subprocess.PIPE,
                check=False,
            ).stdout.strip()
        ),
        "utc": datetime.now(timezone.utc).isoformat(),
        "runs": sorted(
            runs,
            key=lambda row: (
                row["checkpoint"]["global_step"],
                row["checkpoint"]["seed"],
            ),
        ),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"[signal-diagnostic] wrote {len(runs)} frozen cross-sections -> {out}")


if __name__ == "__main__":
    main()
