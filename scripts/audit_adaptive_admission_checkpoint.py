#!/usr/bin/env python3
"""Audit one adaptive_admission_v1 checkpoint and its source lifecycle.

The older admission checkpoint audit assumes that every admitted source has
non-zero probability.  Adaptive admission intentionally permits effectively
zero-mass candidates and can revoke only a subset of sources.  This verifier
therefore checks the stronger lifecycle invariants directly from the immutable
decision history and independent cumulative execution/critic counters.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if hasattr(value, "__dict__"):
        return vars(value)
    return {}


def _parse_bool(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in {"true", "1", "yes"}:
        return True
    if lowered in {"false", "0", "no"}:
        return False
    raise argparse.ArgumentTypeError("expected true or false")


def _csv_names(value: str) -> set[str]:
    return {item.strip() for item in value.split(",") if item.strip()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-step", type=int, required=True)
    parser.add_argument("--expected-adaptive", type=_parse_bool, required=True)
    parser.add_argument("--require-revocation", action="store_true")
    parser.add_argument("--forbid-revocations", type=_csv_names, default=set())
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    ptf_cfg = _as_dict(payload.get("ptf_cfg"))
    audit = _as_dict(payload.get("admission_audit"))
    history = list(audit.get("decision_history") or [])
    initial = next(
        (event for event in history if event.get("decision_id") == "explicit-all"),
        None,
    )
    source_names = list(initial.get("source_names") or []) if initial else []
    source_count = len(source_names)
    candidate_count = source_count + 1

    admitted = [bool(value) for value in audit.get("admitted_sources") or []]
    masses = [float(value) for value in audit.get("candidate_masses") or []]
    active = [int(value) for value in audit.get("active_buffer_counts") or []]
    main_counts = [int(value) for value in audit.get("main_buffer_counts") or []]
    effective = [float(value) for value in audit.get("effective_replay_masses") or []]
    critic = [int(value) for value in audit.get("critic_sample_counts") or []]
    actor = [int(value) for value in audit.get("actor_independent_sample_counts") or []]
    execution = [int(value) for value in audit.get("execution_counts") or []]
    arrays = (masses, active, main_counts, effective, critic, actor, execution)
    train_args = _as_dict(payload.get("args"))

    adaptive_windows = [
        event for event in history if event.get("event") == "adaptive_admission_window"
    ]
    revocation_windows = [event for event in adaptive_windows if event.get("revoked_sources")]
    revoked_names = {
        str(name)
        for event in revocation_windows
        for name in event.get("revoked_sources") or []
    }

    checks: dict[str, bool] = {
        "checkpoint_exists": args.checkpoint.is_file(),
        "global_step": int(payload.get("global_step", -1)) == args.expected_step,
        "admission_all_initial": ptf_cfg.get("admission_mode") == "all",
        "adaptive_flag": bool(ptf_cfg.get("admission_adaptive"))
        == args.expected_adaptive,
        "physical_after_authority": ptf_cfg.get("admission_replay_handoff")
        == "physical_after_authority",
        "initial_decision_present": initial is not None,
        "source_names_unique": bool(source_names)
        and len(source_names) == len(set(source_names)),
        "array_shapes": len(admitted) == source_count
        and all(len(values) == candidate_count for values in arrays),
        "candidate_normalized": len(masses) == candidate_count
        and math.isclose(sum(masses), 1.0, rel_tol=0.0, abs_tol=2e-6),
        "effective_normalized": len(effective) == candidate_count
        and math.isclose(sum(effective), 1.0, rel_tol=0.0, abs_tol=2e-6),
        "actor_shared_critic_batch": audit.get("actor_sampling")
        == "shared_critic_batch",
        "no_independent_actor_samples": len(actor) == candidate_count
        and sum(actor) == 0,
        "counts_nonnegative": all(
            value >= 0 for values in (active, main_counts, critic, actor, execution)
            for value in values
        ),
        "physical_not_above_execution": len(main_counts) == len(execution)
        and all(
            physical <= cumulative
            for physical, cumulative in zip(main_counts, execution)
        ),
        "active_not_above_physical": len(active) == len(main_counts)
        and all(
            allowed <= physical for allowed, physical in zip(active, main_counts)
        ),
        "adaptive_history_present_iff_enabled": bool(adaptive_windows)
        == args.expected_adaptive,
        "revocation_requirement": (not args.require_revocation)
        or bool(revocation_windows),
        "forbidden_revocations_absent": not (
            revoked_names & set(args.forbid_revocations)
        ),
    }

    warmup_steps = int(ptf_cfg.get("mcg_warmup_steps", -1))
    buffer_size = int(train_args.get("buffer_size", -1))
    num_envs = int(train_args.get("num_envs", -1))
    if 0 <= args.expected_step < buffer_size:
        checks["main_matches_execution_before_wrap"] = main_counts == execution
    elif buffer_size > 0 and num_envs > 0:
        checks["main_buffer_full_after_wrap"] = (
            sum(main_counts) == buffer_size * num_envs
        )
    if args.expected_step >= warmup_steps >= 0:
        checks["authority_released_after_warmup"] = (
            audit.get("source_authority_active") is False
        )
        checks["physical_sampling_after_warmup"] = (
            audit.get("sampling_phase") == "physical_allowed"
        )

    per_revocation: list[dict[str, Any]] = []
    policy_events = list(audit.get("policy_events") or [])
    for event in revocation_windows:
        step = int(event.get("completed_step", -1))
        replay_at_apply = _as_dict(event.get("replay_at_apply"))
        execution_at_apply = [
            int(value) for value in event.get("execution_counts_at_apply") or []
        ]
        critic_at_apply = [
            int(value) for value in replay_at_apply.get("critic_sample_counts") or []
        ]
        active_at_apply = [
            int(value) for value in replay_at_apply.get("active_buffer_counts") or []
        ]
        mass_at_apply = [
            float(value) for value in replay_at_apply.get("effective_replay_masses") or []
        ]
        candidate_at_apply = [
            float(value) for value in replay_at_apply.get("candidate_masses") or []
        ]
        admitted_at_apply = [
            bool(value) for value in replay_at_apply.get("admitted_sources") or []
        ]
        persistence = [int(value) for value in event.get("persistence_counts") or []]
        positive_votes = [bool(value) for value in event.get("positive_votes") or []]
        configured_persistence = int(ptf_cfg.get("admission_persistence", -1))
        policy_event = next(
            (
                candidate
                for candidate in policy_events
                if candidate.get("event") == "admission_policy"
                and int(candidate.get("replay_ptr", -1)) == step
            ),
            None,
        )

        source_reports: list[dict[str, Any]] = []
        for name in event.get("revoked_sources") or []:
            index = source_names.index(str(name))
            later_counts = []
            for later in adaptive_windows:
                if int(later.get("completed_step", -1)) <= step:
                    continue
                stats_by_name = {
                    stat.get("candidate"): stat
                    for stat in later.get("statistics") or []
                }
                if name in stats_by_name:
                    later_counts.append(int(stats_by_name[name].get("count", -1)))
            source_checks = {
                "positive_vote": len(positive_votes) > index
                and positive_votes[index],
                "persistence_reached": len(persistence) > index
                and persistence[index] >= configured_persistence,
                "atomic_admitted_false": len(admitted_at_apply) > index
                and not admitted_at_apply[index],
                "atomic_candidate_mass_zero": len(candidate_at_apply) > index
                and candidate_at_apply[index] == 0.0,
                "atomic_active_replay_zero": len(active_at_apply) > index
                and active_at_apply[index] == 0,
                "atomic_effective_mass_zero": len(mass_at_apply) > index
                and mass_at_apply[index] == 0.0,
                "execution_frozen": len(execution_at_apply) > index
                and len(execution) > index
                and execution[index] == execution_at_apply[index],
                "critic_frozen": len(critic_at_apply) > index
                and len(critic) > index
                and critic[index] == critic_at_apply[index],
                "checkpoint_admitted_false": len(admitted) > index
                and not admitted[index],
                "checkpoint_active_replay_zero": len(active) > index
                and active[index] == 0,
                "checkpoint_effective_mass_zero": len(effective) > index
                and effective[index] == 0.0,
                "later_behavior_counts_zero": not later_counts
                or all(value == 0 for value in later_counts),
            }
            source_reports.append(
                {
                    "source": name,
                    "index": index,
                    "checks": source_checks,
                    "execution_at_apply": execution_at_apply[index],
                    "execution_at_checkpoint": execution[index],
                    "critic_at_apply": critic_at_apply[index],
                    "critic_at_checkpoint": critic[index],
                    "later_window_counts": later_counts,
                }
            )
            for key, value in source_checks.items():
                checks[f"revocation_{step}_{name}_{key}"] = value

        event_checks = {
            "decision_applied": event.get("decision_applied") is True,
            "window_boundary": step > 0
            and step % int(ptf_cfg.get("admission_stage_window_steps", 1)) == 0,
            "partial_discard_nonnegative": int(
                event.get("discarded_partial_segments", -1)
            )
            >= 0,
            "policy_event_present": policy_event is not None,
            "policy_critic_snapshot_matches": policy_event is not None
            and list(
                _as_dict(policy_event.get("sample_counts_at_apply")).get("critic")
                or []
            )
            == critic_at_apply,
        }
        for key, value in event_checks.items():
            checks[f"revocation_{step}_{key}"] = value
        per_revocation.append(
            {
                "step": step,
                "revoked_sources": list(event.get("revoked_sources") or []),
                "discarded_partial_segments": int(
                    event.get("discarded_partial_segments", 0)
                ),
                "event_checks": event_checks,
                "sources": source_reports,
            }
        )

    report = {
        "schema_version": 1,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "expected_step": args.expected_step,
        "expected_adaptive": args.expected_adaptive,
        "source_names": source_names,
        "revoked_sources": sorted(revoked_names),
        "checks": checks,
        "revocations": per_revocation,
        "summary": {
            "admitted_sources": admitted,
            "candidate_masses": masses,
            "main_buffer_counts": main_counts,
            "active_buffer_counts": active,
            "effective_replay_masses": effective,
            "execution_counts": execution,
            "critic_sample_counts": critic,
            "actor_independent_sample_counts": actor,
            "source_authority_active": audit.get("source_authority_active"),
            "sampling_phase": audit.get("sampling_phase"),
        },
        "pass": all(checks.values()),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "checkpoint": str(args.checkpoint),
                "pass": report["pass"],
                "revoked_sources": sorted(revoked_names),
                "failed_checks": [key for key, value in checks.items() if not value],
            },
            indent=2,
        )
    )
    if not report["pass"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
