#!/usr/bin/env python3
"""Matched-state, student-relative component probe for source admission.

This is a minimum-cost falsification gate.  It never updates the student or
constructs replay; it compares a source and the current student from the same
MuJoCo FULLPHYSICS state for one RBO-sized segment (default h=25).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from fasttd3_ptf.official_fasttd3_ptf.paths import ensure_humanoidbench_import_path
from fasttd3_ptf.official_fasttd3_ptf.target_evidence import TargetEvidenceContract
from fasttd3_ptf.official_fasttd3_ptf.target_evidence_probe import (
    TargetEvidenceProbeProtocol,
    run_target_evidence_probe,
)
from fasttd3_ptf.ptf.source_policy import SourcePolicy
from probe_lib import ACTION_DIM, load_student

RESET_SEEDS = (11_001, 23_001, 37_001, 53_001)
OCCUPANCY_AGES = (0, 5, 10, 25, 50, 100, 150, 200)
BOOTSTRAP_SEED = 20_260_726
BOOTSTRAP_SAMPLES = 5_000
CONFIDENCE = 0.90


@dataclass(frozen=True)
class PhysicsSnapshot:
    state: np.ndarray
    elapsed_steps: int
    stream_seed: int
    occupancy_age: int
    reset_count: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _make_env(env_name: str):
    ensure_humanoidbench_import_path()
    import gymnasium as gym
    import humanoid_bench  # noqa: F401

    return gym.make(env_name)


def _capture(env, stream_seed: int, occupancy_age: int, reset_count: int) -> PhysicsSnapshot:
    base = env.unwrapped
    spec = mujoco.mjtState.mjSTATE_FULLPHYSICS
    state = np.empty(mujoco.mj_stateSize(base.model, spec), dtype=np.float64)
    mujoco.mj_getState(base.model, base.data, state, spec)
    return PhysicsSnapshot(
        state=state,
        elapsed_steps=int(getattr(env, "_elapsed_steps", 0)),
        stream_seed=stream_seed,
        occupancy_age=occupancy_age,
        reset_count=reset_count,
    )


def _restore(env, snapshot: PhysicsSnapshot) -> np.ndarray:
    base = env.unwrapped
    spec = mujoco.mjtState.mjSTATE_FULLPHYSICS
    mujoco.mj_setState(base.model, base.data, snapshot.state, spec)
    mujoco.mj_forward(base.model, base.data)
    if hasattr(env, "_elapsed_steps"):
        env._elapsed_steps = snapshot.elapsed_steps
    return np.asarray(base.task.get_obs(), dtype=np.float32)


def _roll_branch(
    env,
    snapshot: PhysicsSnapshot,
    act_fn: Callable[[np.ndarray], np.ndarray],
    evidence_contract: TargetEvidenceContract,
    horizon: int,
) -> dict:
    obs = _restore(env, snapshot)
    evidence = evidence_contract.new_accumulator(env)
    total_reward = 0.0
    terminated = truncated = False
    step_count = 0
    for _ in range(horizon):
        action = np.asarray(act_fn(obs), dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        evidence.observe(info)
        step_count += 1
        if terminated or truncated:
            break
    evidence_result = evidence.finish()
    return {
        "return": total_reward,
        "progress": evidence_result["progress"],
        "feasibility": evidence_result["feasibility"],
        "steps": step_count,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
    }


def _bootstrap_interval(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("bootstrap values must be a finite non-empty vector")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(0, values.size, size=(BOOTSTRAP_SAMPLES, values.size))
    means = values[indices].mean(axis=1)
    alpha = 1.0 - CONFIDENCE
    return {
        "mean": float(values.mean()),
        "lcb90": float(np.quantile(means, alpha / 2.0)),
        "ucb90": float(np.quantile(means, 1.0 - alpha / 2.0)),
        "n": int(values.size),
    }


def _classify_source(
    differences: dict[str, np.ndarray],
    feasibility_differences: dict[str, np.ndarray] | None = None,
    hard_constraints: tuple[str, ...] = (),
) -> dict:
    intervals = {name: _bootstrap_interval(values) for name, values in differences.items()}
    feasibility_intervals = {
        name: _bootstrap_interval(values)
        for name, values in (feasibility_differences or {}).items()
    }
    admitted = (
        intervals["return"]["lcb90"] > 0.0
        and intervals["progress"]["lcb90"] > 0.0
        and all(
            feasibility_intervals[name]["lcb90"] >= 0.0
            for name in hard_constraints
        )
    )
    return {
        "admitted": admitted,
        "rank_key": intervals["progress"]["lcb90"],
        "intervals": intervals,
        "feasibility_intervals": feasibility_intervals,
        "hard_constraints": list(hard_constraints),
    }


def _parse_named_paths(values: list[str], flag: str) -> dict[str, Path]:
    parsed: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"{flag} entries must use NAME=PATH, got {value!r}")
        name, raw_path = value.split("=", 1)
        path = Path(raw_path).resolve()
        if not name or name in parsed or not path.is_file():
            raise ValueError(f"invalid {flag} entry {value!r}")
        parsed[name] = path
    return parsed


def _load_sources(bank_paths: dict[str, Path], device: torch.device) -> dict[str, SourcePolicy]:
    sources = {}
    for expected_name, bank_path in bank_paths.items():
        bank = yaml.safe_load(bank_path.read_text())
        specs = bank.get("sources") or []
        if len(specs) != 1:
            raise ValueError(f"{bank_path} must contain exactly one source")
        spec = dict(specs[0])
        if spec.get("name") != expected_name:
            raise ValueError(
                f"{bank_path} source name={spec.get('name')!r}, expected {expected_name!r}"
            )
        sources[expected_name] = SourcePolicy.from_spec(
            spec, device=device, target_action_dim=ACTION_DIM
        )
    return sources


def _collect_snapshots(env, student_act, stream_seed: int) -> list[PhysicsSnapshot]:
    obs, _ = env.reset(seed=stream_seed)
    reset_count = 0
    snapshots = []
    age_set = set(OCCUPANCY_AGES)
    for age in range(max(OCCUPANCY_AGES) + 1):
        if age in age_set:
            snapshots.append(_capture(env, stream_seed, age, reset_count))
        if age == max(OCCUPANCY_AGES):
            break
        obs, _, terminated, truncated, _ = env.step(student_act(obs))
        if terminated or truncated:
            reset_count += 1
            obs, _ = env.reset(seed=stream_seed + 100_000 * reset_count)
    return snapshots


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-label", required=True)
    parser.add_argument("--target-evidence", required=True)
    parser.add_argument("--student-checkpoint", required=True)
    parser.add_argument(
        "--source-bank", action="append", default=[], metavar="NAME=PATH", required=True
    )
    parser.add_argument("--horizon", type=int, default=25)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.horizon != 25:
        raise ValueError("v1 preregistration freezes horizon=25")

    device = torch.device(args.device)
    checkpoint = Path(args.student_checkpoint).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    bank_paths = _parse_named_paths(args.source_bank, "--source-bank")
    if set(bank_paths) != {"stand", "walk", "run"}:
        raise ValueError("v1 requires exactly stand, walk, and run source banks")

    actor, _critic, obs_norm, _critic_norm, global_step = load_student(str(checkpoint), device)
    evidence_path = Path(args.target_evidence).resolve()
    evidence_contract = TargetEvidenceContract.from_yaml(evidence_path)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    ckpt_env = (state.get("args") or {}).get("env_name")
    expected_env = evidence_contract.env_name
    if ckpt_env != expected_env:
        raise ValueError(f"checkpoint env={ckpt_env!r}, expected {expected_env!r}")
    del state
    sources = _load_sources(bank_paths, device)

    @torch.no_grad()
    def student_act(obs: np.ndarray) -> np.ndarray:
        tensor = torch.as_tensor(obs, device=device, dtype=torch.float32).unsqueeze(0)
        return actor(obs_norm(tensor)).squeeze(0).cpu().numpy()

    source_act = {
        name: (
            lambda obs, policy=policy: policy.act(
                torch.as_tensor(obs, device=device, dtype=torch.float32).unsqueeze(0)
            )
            .squeeze(0)
            .cpu()
            .numpy()
        )
        for name, policy in sources.items()
    }

    probe = run_target_evidence_probe(
        contract=evidence_contract,
        student_act=student_act,
        source_actions=source_act,
        protocol=TargetEvidenceProbeProtocol(horizon=args.horizon),
    )
    records = probe["records"]
    classifications = probe["classifications"]
    admitted_order = probe["admitted_order"]
    report = {
        "experiment": "stage_conditioned_component_probe_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip(),
        "task": args.task_label,
        "env_name": expected_env,
        "student": {
            "checkpoint": str(checkpoint),
            "sha256": _sha256(checkpoint),
            "global_step": global_step,
        },
        "protocol": probe["protocol"],
        "target_evidence": {
            "path": str(evidence_path),
            "sha256": _sha256(evidence_path),
            "name": evidence_contract.name,
            "progress": evidence_contract.progress.__dict__,
            "feasibility": [spec.__dict__ for spec in evidence_contract.feasibility],
            "hard_constraints": list(evidence_contract.hard_constraints),
        },
        "source_banks": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in sorted(bank_paths.items())
        },
        "classifications": classifications,
        "admitted_order": admitted_order,
        "exact_abstention": not admitted_order,
        "records": records,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        raise FileExistsError(f"refusing to overwrite {out_path}")
    out_path.write_text(json.dumps(report, indent=2) + "\n")

    print(
        f"[{args.task_label}@{global_step}] admitted={admitted_order or 'NONE'} "
        f"panel={len(records)} -> {out_path}"
    )
    for name in ("stand", "walk", "run"):
        result = classifications[name]
        intervals = result["intervals"]
        feasibility_text = ",".join(
            f"{key}:{item['lcb90']:+.3f}"
            for key, item in result["feasibility_intervals"].items()
        ) or "none"
        print(
            f"  {name:5s} admit={str(result['admitted']):5s} "
            f"dR={intervals['return']['mean']:+.3f}"
            f"[{intervals['return']['lcb90']:+.3f},{intervals['return']['ucb90']:+.3f}] "
            f"dP={intervals['progress']['mean']:+.3f}"
            f"[{intervals['progress']['lcb90']:+.3f},{intervals['progress']['ucb90']:+.3f}] "
            f"dF_lcb=({feasibility_text})"
        )


if __name__ == "__main__":
    main()
