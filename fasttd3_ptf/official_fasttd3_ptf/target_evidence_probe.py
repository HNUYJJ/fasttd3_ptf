"""Matched-state target-evidence probe shared by offline and online admission.

The controller is task agnostic. It receives a declarative
``TargetEvidenceContract`` and compares each source with the current student
from identical MuJoCo states. Probe transitions are returned as an audit
artifact only; this module has no replay-buffer dependency.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import mujoco
import numpy as np

from fasttd3_ptf.official_fasttd3_ptf.paths import ensure_humanoidbench_import_path
from fasttd3_ptf.official_fasttd3_ptf.admission_control import AdmissionSnapshot
from fasttd3_ptf.official_fasttd3_ptf.target_evidence import TargetEvidenceContract


DEFAULT_RESET_SEEDS = (11_001, 23_001, 37_001, 53_001)
DEFAULT_OCCUPANCY_AGES = (0, 5, 10, 25, 50, 100, 150, 200)
DEFAULT_BOOTSTRAP_SEED = 20_260_726
DEFAULT_BOOTSTRAP_SAMPLES = 5_000
DEFAULT_CONFIDENCE = 0.90


@dataclass(frozen=True)
class TargetEvidenceProbeProtocol:
    horizon: int = 25
    reset_seeds: tuple[int, ...] = DEFAULT_RESET_SEEDS
    occupancy_ages: tuple[int, ...] = DEFAULT_OCCUPANCY_AGES
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES
    confidence: float = DEFAULT_CONFIDENCE

    def __post_init__(self) -> None:
        if self.horizon <= 0:
            raise ValueError("probe horizon must be positive")
        if not self.reset_seeds or len(set(self.reset_seeds)) != len(self.reset_seeds):
            raise ValueError("probe reset seeds must be non-empty and unique")
        if (
            not self.occupancy_ages
            or tuple(sorted(set(self.occupancy_ages))) != self.occupancy_ages
            or self.occupancy_ages[0] < 0
        ):
            raise ValueError(
                "probe occupancy ages must be non-empty, unique, sorted, and non-negative"
            )
        if self.bootstrap_samples <= 0:
            raise ValueError("bootstrap_samples must be positive")
        if not 0.0 < self.confidence < 1.0:
            raise ValueError("probe confidence must lie in (0, 1)")


@dataclass(frozen=True)
class PhysicsSnapshot:
    state: np.ndarray
    elapsed_steps: int
    stream_seed: int
    occupancy_age: int
    reset_count: int


def make_target_env(env_name: str):
    ensure_humanoidbench_import_path()
    import gymnasium as gym
    import humanoid_bench  # noqa: F401

    return gym.make(env_name)


def _capture(
    env,
    *,
    stream_seed: int,
    occupancy_age: int,
    reset_count: int,
) -> PhysicsSnapshot:
    base = env.unwrapped
    spec = mujoco.mjtState.mjSTATE_FULLPHYSICS
    state = np.empty(mujoco.mj_stateSize(base.model, spec), dtype=np.float64)
    mujoco.mj_getState(base.model, base.data, state, spec)
    return PhysicsSnapshot(
        state=state,
        elapsed_steps=int(getattr(env, "_elapsed_steps", 0)),
        stream_seed=int(stream_seed),
        occupancy_age=int(occupancy_age),
        reset_count=int(reset_count),
    )


def _restore(env, snapshot: PhysicsSnapshot) -> np.ndarray:
    base = env.unwrapped
    spec = mujoco.mjtState.mjSTATE_FULLPHYSICS
    mujoco.mj_setState(base.model, base.data, snapshot.state, spec)
    mujoco.mj_forward(base.model, base.data)
    if hasattr(env, "_elapsed_steps"):
        env._elapsed_steps = snapshot.elapsed_steps
    return np.asarray(base.task.get_obs(), dtype=np.float32)


def _collect_snapshots(
    env,
    student_act: Callable[[np.ndarray], np.ndarray],
    protocol: TargetEvidenceProbeProtocol,
) -> list[PhysicsSnapshot]:
    snapshots: list[PhysicsSnapshot] = []
    ages = set(protocol.occupancy_ages)
    max_age = protocol.occupancy_ages[-1]
    for stream_seed in protocol.reset_seeds:
        obs, _ = env.reset(seed=stream_seed)
        reset_count = 0
        for age in range(max_age + 1):
            if age in ages:
                snapshots.append(
                    _capture(
                        env,
                        stream_seed=stream_seed,
                        occupancy_age=age,
                        reset_count=reset_count,
                    )
                )
            if age == max_age:
                break
            obs, _, terminated, truncated, _ = env.step(student_act(obs))
            if terminated or truncated:
                reset_count += 1
                obs, _ = env.reset(seed=stream_seed + 100_000 * reset_count)
    expected = len(protocol.reset_seeds) * len(protocol.occupancy_ages)
    if len(snapshots) != expected:
        raise AssertionError(f"unexpected occupancy panel size {len(snapshots)} != {expected}")
    return snapshots


def _roll_branch(
    env,
    snapshot: PhysicsSnapshot,
    act_fn: Callable[[np.ndarray], np.ndarray],
    contract: TargetEvidenceContract,
    horizon: int,
) -> dict[str, Any]:
    obs = _restore(env, snapshot)
    evidence = contract.new_accumulator(env)
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
    result = evidence.finish()
    return {
        "return": total_reward,
        "progress": result["progress"],
        "feasibility": result["feasibility"],
        "steps": step_count,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
    }


def bootstrap_interval(
    values: np.ndarray,
    *,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("bootstrap values must be a finite non-empty vector")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(samples, values.size))
    means = values[indices].mean(axis=1)
    alpha = 1.0 - confidence
    return {
        "mean": float(values.mean()),
        "lcb90": float(np.quantile(means, alpha / 2.0)),
        "ucb90": float(np.quantile(means, 1.0 - alpha / 2.0)),
        "n": int(values.size),
    }


def classify_source(
    differences: Mapping[str, np.ndarray],
    feasibility_differences: Mapping[str, np.ndarray] | None = None,
    *,
    hard_constraints: Sequence[str] = (),
    protocol: TargetEvidenceProbeProtocol | None = None,
) -> dict[str, Any]:
    protocol = protocol or TargetEvidenceProbeProtocol()
    interval_kwargs = {
        "seed": protocol.bootstrap_seed,
        "samples": protocol.bootstrap_samples,
        "confidence": protocol.confidence,
    }
    intervals = {
        name: bootstrap_interval(values, **interval_kwargs)
        for name, values in differences.items()
    }
    feasibility_intervals = {
        name: bootstrap_interval(values, **interval_kwargs)
        for name, values in (feasibility_differences or {}).items()
    }
    admitted = (
        float(intervals["return"]["lcb90"]) > 0.0
        and float(intervals["progress"]["lcb90"]) > 0.0
        and all(
            float(feasibility_intervals[name]["lcb90"]) >= 0.0
            for name in hard_constraints
        )
    )
    return {
        "admitted": admitted,
        "rank_key": float(intervals["progress"]["lcb90"]),
        "intervals": intervals,
        "feasibility_intervals": feasibility_intervals,
        "hard_constraints": list(hard_constraints),
    }


def run_target_evidence_probe(
    *,
    contract: TargetEvidenceContract,
    student_act: Callable[[np.ndarray], np.ndarray],
    source_actions: Mapping[str, Callable[[np.ndarray], np.ndarray]],
    protocol: TargetEvidenceProbeProtocol | None = None,
    env=None,
) -> dict[str, Any]:
    """Run the matched source-vs-student probe and return a JSON-ready report."""

    protocol = protocol or TargetEvidenceProbeProtocol()
    if not source_actions:
        raise ValueError("target-evidence probe requires at least one source")

    owns_env = env is None
    env = make_target_env(contract.env_name) if env is None else env
    records: list[dict[str, Any]] = []
    try:
        snapshots = _collect_snapshots(env, student_act, protocol)
        for panel_index, snapshot in enumerate(snapshots):
            student = _roll_branch(
                env, snapshot, student_act, contract, protocol.horizon
            )
            source_outcomes = {
                name: _roll_branch(
                    env, snapshot, act_fn, contract, protocol.horizon
                )
                for name, act_fn in source_actions.items()
            }
            records.append(
                {
                    "panel_index": panel_index,
                    "stream_seed": snapshot.stream_seed,
                    "occupancy_age": snapshot.occupancy_age,
                    "reset_count": snapshot.reset_count,
                    "student": student,
                    "sources": source_outcomes,
                }
            )
    finally:
        if owns_env:
            env.close()

    classifications: dict[str, Any] = {}
    for source_name in sorted(source_actions):
        differences = {
            metric: np.asarray(
                [
                    row["sources"][source_name][metric] - row["student"][metric]
                    for row in records
                ],
                dtype=np.float64,
            )
            for metric in ("return", "progress")
        }
        feasibility_differences = {
            spec.name: np.asarray(
                [
                    row["sources"][source_name]["feasibility"][spec.name]
                    - row["student"]["feasibility"][spec.name]
                    for row in records
                ],
                dtype=np.float64,
            )
            for spec in contract.feasibility
        }
        classifications[source_name] = classify_source(
            differences,
            feasibility_differences,
            hard_constraints=contract.hard_constraints,
            protocol=protocol,
        )

    admitted_order = [
        name
        for name, result in sorted(
            classifications.items(),
            key=lambda item: item[1]["rank_key"],
            reverse=True,
        )
        if result["admitted"]
    ]
    return {
        "protocol": {
            "horizon": protocol.horizon,
            "reset_seeds": list(protocol.reset_seeds),
            "occupancy_ages": list(protocol.occupancy_ages),
            "panel_size": len(records),
            "bootstrap_samples": protocol.bootstrap_samples,
            "bootstrap_seed": protocol.bootstrap_seed,
            "confidence": protocol.confidence,
            "admission_rule": (
                "LCB(dTargetReturn)>0 and "
                "LCB(dTargetAchievementProgress)>0; only explicit hard "
                "constraints have veto authority"
            ),
            "ranking": "descending progress LCB among admitted sources",
        },
        "classifications": classifications,
        "admitted_order": admitted_order,
        "exact_abstention": not admitted_order,
        "records": records,
    }


def build_top1_admission_snapshot(
    *,
    source_names: Sequence[str],
    probe_result: Mapping[str, Any],
    decision_step: int,
    quarantine_artifact: str,
    quarantine_digest: str,
) -> AdmissionSnapshot:
    """Convert a probe result into top-1 admission or exact abstention.

    Equal zero logits make the selected source and student equal first-class
    candidates (0.5/0.5 at any positive softmax temperature). The function has
    no task-specific branch and never admits a source that failed the probe.
    """

    names = tuple(str(name) for name in source_names)
    if len(names) != len(set(names)):
        raise ValueError("target-evidence source names must be unique")
    admitted_order = tuple(str(name) for name in probe_result["admitted_order"])
    unknown = sorted(set(admitted_order).difference(names))
    if unknown:
        raise ValueError(f"probe admitted unknown sources: {unknown}")
    selected = admitted_order[:1]
    selected_set = set(selected)
    return AdmissionSnapshot(
        source_names=names,
        admitted=tuple(name in selected_set for name in names),
        source_logits=tuple(0.0 for _ in names),
        student_logit=0.0,
        decision_id=f"target-evidence-step-{int(decision_step)}",
        mode="target_evidence",
        quarantine_artifact=str(quarantine_artifact),
        quarantine_digest=str(quarantine_digest),
    )
