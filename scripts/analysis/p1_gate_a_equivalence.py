#!/usr/bin/env python3
"""Finite CPU equivalence gate for Phase-1 basketball reuse.

The scientific question is narrow: before the scheduled decision at step 30k,
does ``admission_mode=all`` have the same behavior/replay semantics as a
schedule whose step-0 decision admits the whole bank?  This harness checks the
immutable decision, a 100-step MCG trace (four 25-step segments and resets),
and actor/critic replay indices.  It intentionally does not reproduce a full
30k training run; equal inputs to the same downstream learner imply equal
updates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import torch
from tensordict import TensorDict

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from fasttd3_ptf.config import load_yaml  # noqa: E402
from fasttd3_ptf.official_fasttd3_ptf.admission_control import (  # noqa: E402
    AdmissionSnapshot,
    build_admission_schedule,
    build_admission_snapshot,
)
from fasttd3_ptf.official_fasttd3_ptf.paths import (  # noqa: E402
    ensure_fasttd3_import_path,
)
from fasttd3_ptf.official_fasttd3_ptf.ptf_replay import PTFReplayWrapper  # noqa: E402
from fasttd3_ptf.ptf.action_schema import h1hand_default_action_schema  # noqa: E402
from fasttd3_ptf.ptf.mcg import McgBehaviorController  # noqa: E402

ensure_fasttd3_import_path()
from fast_td3_utils import SimpleReplayBuffer  # type: ignore  # noqa: E402


BANK = ROOT / "configs/source_banks/h1hand_std9_wfix_basketball.yaml"
STUDENT_LOGIT = 3.5892126423877646
TRACE_STEPS = 100
NUM_ENVS = 8
OBS_DIM = 8


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def _semantic(snapshot: AdmissionSnapshot) -> dict[str, Any]:
    return {
        "source_names": snapshot.source_names,
        "admitted": snapshot.admitted,
        "source_logits": snapshot.source_logits,
        "student_logit": snapshot.student_logit,
        "candidate_probabilities": tuple(
            float(v) for v in snapshot.candidate_probabilities(tau=1.0)
        ),
    }


def _bank_inputs() -> tuple[list[str], torch.Tensor, torch.Tensor]:
    raw = load_yaml(BANK)
    sources = raw["sources"]
    names = [str(source["name"]) for source in sources]
    weights = torch.tensor(
        [float(source["bootstrap"]["weight"]) for source in sources],
        dtype=torch.float32,
    )
    horizons = torch.tensor(
        [int(source["bootstrap"]["horizon"]) for source in sources],
        dtype=torch.long,
    )
    return names, weights, horizons


def _controller(
    *, weights: torch.Tensor, horizons: torch.Tensor, admitted: torch.Tensor
) -> McgBehaviorController:
    schema = h1hand_default_action_schema()
    masks = torch.stack(
        [schema.get("legs_torso").mask(schema.dim), schema.get("arms").mask(schema.dim)]
    )
    return McgBehaviorController(
        num_envs=NUM_ENVS,
        num_groups=2,
        device="cpu",
        group_masks=masks,
        warmup_min_steps=25,
        seed=1719,
        warmup_mode="admission_bootstrap",
        bootstrap_weights=weights.clone(),
        bootstrap_horizons=horizons.clone(),
        admitted_sources=admitted.clone(),
        admission_student_logit=STUDENT_LOGIT,
    )


def _replay(num_sources: int, candidate_masses: torch.Tensor) -> PTFReplayWrapper:
    replay = PTFReplayWrapper(
        SimpleReplayBuffer(
            n_env=NUM_ENVS,
            buffer_size=128,
            n_obs=OBS_DIM,
            n_act=h1hand_default_action_schema().dim,
            n_critic_obs=OBS_DIM,
            n_steps=1,
            device=torch.device("cpu"),
        )
    )
    replay.enable_provenance(group_count=2)
    replay.set_admission_policy(
        admitted_sources=torch.ones(num_sources, dtype=torch.bool),
        candidate_masses=candidate_masses,
        recency_half_life=0.0,
        uniform_mix=1.0,
        priority_alpha=0.0,
    )
    return replay


def _transition(
    obs: torch.Tensor,
    actions: torch.Tensor,
    next_obs: torch.Tensor,
    rewards: torch.Tensor,
    dones: torch.Tensor,
) -> TensorDict:
    return TensorDict(
        {
            "observations": obs,
            "actions": actions,
            "next": {
                "rewards": rewards,
                "dones": dones.long(),
                "truncations": torch.zeros_like(dones, dtype=torch.long),
                "observations": next_obs,
            },
        },
        batch_size=NUM_ENVS,
    )


def _assert_tree_equal(left: Any, right: Any, prefix: str = "root") -> None:
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor) and torch.equal(left, right), prefix
    elif isinstance(left, dict):
        assert isinstance(right, dict) and left.keys() == right.keys(), prefix
        for key in left:
            _assert_tree_equal(left[key], right[key], f"{prefix}.{key}")
    elif isinstance(left, (list, tuple)):
        assert isinstance(right, type(left)) and len(left) == len(right), prefix
        for index, (a, b) in enumerate(zip(left, right)):
            _assert_tree_equal(a, b, f"{prefix}[{index}]")
    else:
        assert left == right, prefix


def run_gate() -> dict[str, Any]:
    torch.set_num_threads(1)
    names, weights, horizons = _bank_inputs()
    static = build_admission_snapshot(
        mode="all",
        source_names=names,
        source_logits=weights.tolist(),
        student_logit=STUDENT_LOGIT,
    )
    with tempfile.TemporaryDirectory(prefix="p1_gate_a_") as tmp:
        schedule_path = Path(tmp) / "schedule.yaml"
        schedule_path.write_text(
            "\n".join(
                [
                    "schema_version: 1",
                    f"source_names: [{', '.join(names)}]",
                    "decisions:",
                    "  - step: 0",
                    "    decision_id: admit-all",
                    f"    admitted_sources: [{', '.join(names)}]",
                    "  - step: 30000",
                    "    decision_id: exact-abstention",
                    "    admitted_sources: []",
                ]
            )
            + "\n"
        )
        schedule = build_admission_schedule(
            schedule_path=schedule_path,
            source_names=names,
            source_logits=weights.tolist(),
            student_logit=STUDENT_LOGIT,
        )
        for step in (0, 25, 29_999):
            assert _semantic(static) == _semantic(schedule.snapshot_at(step))
        assert schedule.snapshot_at(30_000).exact_abstain

        scheduled = schedule.snapshot_at(0)
        admitted = static.admitted_tensor("cpu")
        behavior_a = _controller(weights=weights, horizons=horizons, admitted=admitted)
        behavior_b = _controller(
            weights=weights,
            horizons=horizons,
            admitted=scheduled.admitted_tensor("cpu"),
        )
        masses = static.candidate_probabilities(tau=1.0)
        replay_a = _replay(len(names), masses)
        replay_b = _replay(len(names), masses)

        generator = torch.Generator().manual_seed(20260719)
        obs = torch.randn(NUM_ENVS, OBS_DIM, generator=generator)
        trace = hashlib.sha256()
        segment_ids = torch.arange(NUM_ENVS, dtype=torch.int64)
        previous = torch.full((NUM_ENVS,), -99, dtype=torch.long)
        segment_steps = torch.zeros(NUM_ENVS, dtype=torch.int16)
        actor_index_draws = 0
        critic_index_draws = 0

        for step in range(TRACE_STEPS):
            student = torch.tanh(
                torch.arange(NUM_ENVS * 61, dtype=torch.float32).reshape(NUM_ENVS, 61)
                / 101.0
                + step / 100.0
            )
            source_actions = torch.tanh(
                torch.arange(NUM_ENVS * len(names) * 61, dtype=torch.float32).reshape(
                    NUM_ENVS, len(names), 61
                )
                / 503.0
                - step / 200.0
            )
            dones = torch.zeros(NUM_ENVS, dtype=torch.bool)
            if step in (37, 74):
                dones[(step // 37) % NUM_ENVS] = True

            actions_a, info_a = behavior_a.step(
                student, source_actions, best=None, gate=None, dones=dones
            )
            actions_b, info_b = behavior_b.step(
                student, source_actions, best=None, gate=None, dones=dones
            )
            assert torch.equal(actions_a, actions_b)
            assert info_a == info_b
            assert torch.equal(behavior_a.current, behavior_b.current)
            assert torch.equal(behavior_a.current_arm, behavior_b.current_arm)
            assert torch.equal(behavior_a.steps_left, behavior_b.steps_left)
            assert torch.equal(
                behavior_a.admission_probabilities(),
                behavior_b.admission_probabilities(),
            )

            canonical = behavior_a.current[:, 0].clone()
            changed = canonical != previous
            segment_ids[changed] += NUM_ENVS
            segment_steps = torch.where(
                changed, torch.zeros_like(segment_steps), segment_steps + 1
            )
            previous = canonical.clone()
            next_obs = 0.97 * obs + 0.03 * actions_a[:, :OBS_DIM]
            next_obs[dones] = 0.0
            rewards = 1.0 - actions_a.square().mean(dim=1)
            provenance = {
                "behavior_source": canonical.to(torch.int16),
                "source_by_group": behavior_a.current.to(torch.int16),
                "executed_group_mask": behavior_a.current >= 0,
                "segment_id": segment_ids,
                "segment_step": segment_steps,
                "anchor_id": torch.full((NUM_ENVS,), -1, dtype=torch.int32),
                "env_rank": torch.arange(NUM_ENVS, dtype=torch.int16),
                "learner_step": torch.full((NUM_ENVS,), step, dtype=torch.int64),
            }
            transition = _transition(obs, actions_a, next_obs, rewards, dones)
            replay_a.extend(transition.clone(), canonical, provenance=provenance)
            replay_b.extend(transition.clone(), canonical, provenance=provenance)

            for role in ("critic", "actor"):
                rng = torch.random.get_rng_state()
                index_a = replay_a.draw_indices(batch_size=16, role=role)
                torch.random.set_rng_state(rng)
                index_b = replay_b.draw_indices(batch_size=16, role=role)
                assert torch.equal(index_a, index_b), f"{role} indices at step {step}"
                batch_a = replay_a.gather(index_a)
                batch_b = replay_b.gather(index_b)
                assert set(batch_a.keys(True, True)) == set(batch_b.keys(True, True))
                for key in batch_a.keys(True, True):
                    assert torch.equal(batch_a[key], batch_b[key]), (step, role, key)
                if role == "critic":
                    critic_index_draws += int(index_a.numel())
                else:
                    actor_index_draws += int(index_a.numel())

            for tensor in (actions_a, rewards, dones, next_obs):
                trace.update(tensor.contiguous().numpy().tobytes())
            obs = next_obs

        assert torch.equal(behavior_a.generator.get_state(), behavior_b.generator.get_state())
        _assert_tree_equal(replay_a.export_valid(), replay_b.export_valid())
        _assert_tree_equal(replay_a.admission_audit(), replay_b.admission_audit())

    return {
        "schema_version": 1,
        "gate": "phase1_gate_a_basketball_all_vs_schedule_step0_all",
        "status": "PASS",
        "git_head": _git_head(),
        "bank": str(BANK.relative_to(ROOT)),
        "bank_sha256": _sha256(BANK),
        "student_logit": STUDENT_LOGIT,
        "source_names": names,
        "candidate_masses": [float(value) for value in masses],
        "static_steps_checked": [0, 25, 29_999, 30_000],
        "dynamic_trace_steps": TRACE_STEPS,
        "num_envs": NUM_ENVS,
        "segment_boundaries_crossed": 4,
        "forced_done_steps": [37, 74],
        "critic_index_draws": critic_index_draws,
        "actor_index_draws": actor_index_draws,
        "transition_trace_sha256": trace.hexdigest(),
        "claim": (
            "Before step 30000, static all and schedule step-0 admit-all produce "
            "the same admission, MCG behavior, provenance, and replay sampling trace."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = run_gate()
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        if args.out.exists():
            raise FileExistsError(args.out)
        args.out.write_text(payload)
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
