#!/usr/bin/env python3
"""Verify that an admission-core training cell produced a valid final checkpoint.

This is intentionally independent of the launcher exit status.  It is used to
recover a completed training run when a post-training shell error happened
after the final checkpoint had already been written.  The original launcher
metadata is never modified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch


CELLS = {
    "basketball_exact_none": {"task": "basketball", "mode": "none"},
    "powerlift_retain_all": {"task": "powerlift", "mode": "all"},
}


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell", choices=sorted(CELLS), required=True)
    parser.add_argument("--seed", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--stamp", required=True)
    parser.add_argument("--models", type=Path, default=Path("models"))
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expected-step", type=int, default=100000)
    args = parser.parse_args()

    cell = CELLS[args.cell]
    task = cell["task"]
    mode = cell["mode"]
    exp_name = f"h1hand_{task}_admission_v1_{mode}_s{args.seed}_{args.stamp}"
    checkpoint = args.models / (
        f"h1hand-{task}-v0__{exp_name}__{args.seed}_final.pt"
    )

    checks: dict[str, bool] = {
        "checkpoint_exists": checkpoint.is_file(),
        "log_exists": args.log.is_file(),
    }
    details: dict[str, Any] = {}
    if checkpoint.is_file():
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        train_args = _as_dict(payload.get("args"))
        ptf_cfg = _as_dict(payload.get("ptf_cfg"))
        audit = _as_dict(payload.get("admission_audit"))
        source_names = list(payload.get("source_names") or [])
        candidate_masses = list(audit.get("candidate_masses") or [])
        admitted_sources = list(audit.get("admitted_sources") or [])
        execution_counts = list(audit.get("execution_counts") or [])
        critic_counts = list(audit.get("critic_sample_counts") or [])

        checks.update(
            {
                "global_step": int(payload.get("global_step", -1)) == args.expected_step,
                "seed": int(train_args.get("seed", -1)) == args.seed,
                "exp_name": train_args.get("exp_name") == exp_name,
                "admission_mode": ptf_cfg.get("admission_mode") == mode,
                "warmup_mode": ptf_cfg.get("mcg_warmup_mode") == "admission_bootstrap",
                "bootstrap_only": ptf_cfg.get("mcg_ablation") == "bootstrap_only",
                "candidate_shape": len(candidate_masses) == len(source_names),
                "execution_shape": len(execution_counts) == len(source_names),
                "critic_shape": len(critic_counts) == len(source_names),
            }
        )
        if candidate_masses:
            checks["student_candidate_mass"] = math.isclose(
                float(candidate_masses[-1]), 1.0 if mode == "none" else 0.5,
                abs_tol=1e-7,
            )
        else:
            checks["student_candidate_mass"] = False
        if mode == "none":
            checks["no_admitted_source"] = not any(bool(x) for x in admitted_sources)
            checks["zero_source_execution"] = sum(int(x) for x in execution_counts[:-1]) == 0
            checks["zero_source_critic"] = sum(int(x) for x in critic_counts[:-1]) == 0
        else:
            checks["all_sources_admitted"] = bool(admitted_sources) and all(
                bool(x) for x in admitted_sources
            )
            checks["source_executed"] = sum(int(x) for x in execution_counts[:-1]) > 0
            checks["source_sampled"] = sum(int(x) for x in critic_counts[:-1]) > 0
        details = {
            "global_step": int(payload.get("global_step", -1)),
            "source_names": source_names,
            "candidate_masses": candidate_masses,
            "execution_counts": execution_counts,
            "critic_sample_counts": critic_counts,
            "checkpoint_sha256": _sha256(checkpoint),
        }

    if args.log.is_file():
        log_text = args.log.read_text(errors="replace")
        checks["final_save_logged"] = str(checkpoint) in log_text
    else:
        checks["final_save_logged"] = False

    verified = bool(checks) and all(checks.values())
    report = {
        "schema_version": 1,
        "verified": verified,
        "cell": args.cell,
        "seed": args.seed,
        "stamp": args.stamp,
        "checkpoint": str(checkpoint),
        "log": str(args.log),
        "checks": checks,
        "details": details,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not verified:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
