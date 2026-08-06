#!/usr/bin/env python3
"""Verify one completed admission_handoff_v1 training cell."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch


CELLS = {
    "powerlift_admission_all_fix": {"task": "powerlift", "sources": 9},
    "truck_admission_h4_fix": {"task": "truck", "sources": 4},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if hasattr(value, "__dict__"):
        return vars(value)
    return {}


def checkpoint_path(root: Path, task: str, seed: int, stamp: str, suffix: str) -> Path:
    exp = f"h1hand_{task}_admission_handoff_v1_all_s{seed}_{stamp}"
    return root / "models" / f"h1hand-{task}-v0__{exp}__{seed}_{suffix}.pt"


def audit_checkpoint(path: Path, expected_step: int, expected_sources: int) -> tuple[dict[str, bool], dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    ptf = as_dict(payload.get("ptf_cfg"))
    audit = as_dict(payload.get("admission_audit"))
    admitted = list(audit.get("admitted_sources") or [])
    counts = [int(value) for value in audit.get("active_buffer_counts") or []]
    masses = [float(value) for value in audit.get("effective_replay_masses") or []]
    release_events = [
        event
        for event in audit.get("policy_events") or []
        if event.get("event") == "source_authority"
        and not event.get("source_authority_active", True)
    ]
    physical_source = sum(counts[:-1]) / sum(counts) if counts and sum(counts) else math.nan
    effective_source = sum(masses[:-1]) if masses else math.nan
    checks = {
        "global_step": int(payload.get("global_step", -1)) == expected_step,
        "handoff_config": ptf.get("admission_replay_handoff") == "physical_after_authority",
        "bootstrap_only": ptf.get("mcg_ablation") == "bootstrap_only",
        "admission_all": ptf.get("admission_mode") == "all",
        "all_sources_admitted": len(admitted) == expected_sources and all(admitted),
        "authority_released": audit.get("source_authority_active") is False,
        "physical_phase": audit.get("sampling_phase") == "physical_allowed",
        "release_at_30k": any(int(event.get("replay_ptr", -1)) == 30000 for event in release_events),
        "effective_mass_is_physical": math.isclose(
            physical_source, effective_source, rel_tol=0.0, abs_tol=1e-6
        ),
    }
    details = {
        "checkpoint": str(path),
        "physical_allowed_source_share": physical_source,
        "effective_source_replay_mass": effective_source,
        "release_events": release_events,
        "sha256": sha256(path),
    }
    return checks, details


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell", choices=sorted(CELLS), required=True)
    parser.add_argument("--seed", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--stamp", required=True)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    cell = CELLS[args.cell]
    checks: dict[str, bool] = {
        "log_exists": args.log.is_file(),
    }
    details: dict[str, Any] = {}
    for step in (60000, 90000):
        path = checkpoint_path(root, cell["task"], args.seed, args.stamp, str(step))
        checks[f"checkpoint_{step}_exists"] = path.is_file()
        if path.is_file():
            step_checks, step_details = audit_checkpoint(
                path, step, int(cell["sources"])
            )
            checks.update({f"step_{step}_{name}": value for name, value in step_checks.items()})
            details[f"step_{step}"] = step_details
    final_path = checkpoint_path(root, cell["task"], args.seed, args.stamp, "final")
    checks["final_checkpoint_exists"] = final_path.is_file()
    if final_path.is_file():
        final_checks, final_details = audit_checkpoint(
            final_path, 100000, int(cell["sources"])
        )
        checks.update({f"final_{name}": value for name, value in final_checks.items()})
        details["final"] = final_details
    if args.log.is_file():
        text = args.log.read_text(errors="replace")
        checks["authority_release_logged"] = "source authority released at step 30000" in text
        checks["final_save_logged"] = str(final_path.relative_to(root)) in text
        checks["wandb_initialized"] = "wandb:" in text and (
            "Syncing run" in text or "View run" in text
        )
    result = {
        "schema_version": 1,
        "cell": args.cell,
        "seed": args.seed,
        "stamp": args.stamp,
        "checks": checks,
        "details": details,
        "pass": all(checks.values()),
    }
    out = args.out if args.out.is_absolute() else root / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not result["pass"]:
        raise SystemExit(5)


if __name__ == "__main__":
    main()
