#!/usr/bin/env python3
"""Audit one admission checkpoint before accepting its performance curve."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch


def _source_sum(values: list[int]) -> int:
    return sum(int(value) for value in values[:-1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--mode", choices=("all", "none"), required=True)
    parser.add_argument("--expected-step", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    audit = payload["admission_audit"]
    ptf_cfg = payload["ptf_cfg"]
    candidate = [float(value) for value in audit["candidate_masses"]]
    execution = [int(value) for value in audit["execution_counts"]]
    physical = [int(value) for value in audit["main_buffer_counts"]]
    critic = [int(value) for value in audit["critic_sample_counts"]]
    actor_independent = [
        int(value) for value in audit["actor_independent_sample_counts"]
    ]

    execution_total = sum(execution)
    critic_total = sum(critic)
    execution_share = [value / execution_total for value in execution]
    critic_share = [value / critic_total for value in critic]
    checks = {
        "global_step": int(payload["global_step"]) == args.expected_step,
        "admission_mode": ptf_cfg["admission_mode"] == args.mode,
        "array_shapes": len(candidate)
        == len(execution)
        == len(physical)
        == len(critic)
        == len(actor_independent),
        "candidate_normalized": math.isclose(sum(candidate), 1.0, abs_tol=1e-6),
        "actor_shared_critic_batch": audit.get("actor_sampling")
        == "shared_critic_batch",
        "no_independent_actor_samples": sum(actor_independent) == 0,
    }
    if args.mode == "none":
        checks.update(
            {
                "student_mass_one": math.isclose(candidate[-1], 1.0, abs_tol=1e-7),
                "source_execution_zero": _source_sum(execution) == 0,
                "source_physical_replay_zero": _source_sum(physical) == 0,
                "source_critic_zero": _source_sum(critic) == 0,
            }
        )
    else:
        checks.update(
            {
                "student_mass_half": math.isclose(candidate[-1], 0.5, abs_tol=1e-7),
                "every_source_executed": all(value > 0 for value in execution[:-1]),
                "every_source_sampled": all(value > 0 for value in critic[:-1]),
                "physical_matches_execution": physical == execution,
                "execution_share_tracks_quota": max(
                    abs(actual - expected)
                    for actual, expected in zip(execution_share, candidate)
                )
                <= 0.015,
                "critic_share_tracks_quota": max(
                    abs(actual - expected)
                    for actual, expected in zip(critic_share, candidate)
                )
                <= 0.015,
            }
        )

    report = {
        "schema_version": 1,
        "checkpoint": str(args.checkpoint),
        "mode": args.mode,
        "expected_step": args.expected_step,
        "pass": all(checks.values()),
        "checks": checks,
        "candidate_masses": candidate,
        "execution_counts": execution,
        "execution_shares": execution_share,
        "main_buffer_counts": physical,
        "critic_sample_counts": critic,
        "critic_sample_shares": critic_share,
        "actor_independent_sample_counts": actor_independent,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"checkpoint": str(args.checkpoint), "pass": report["pass"], "checks": checks}, indent=2))
    if not report["pass"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
