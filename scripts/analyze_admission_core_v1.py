#!/usr/bin/env python3
"""Audit admission-core checkpoints and local training curves."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fasttd3_ptf.official_fasttd3_ptf.source_admission import validate_quarantine_bank


EVAL_RE = re.compile(r"\[eval\] step=(\d+) return=([-+0-9.eE]+) length=([-+0-9.eE]+)")


def _checkpoint(root: Path, task: str, mode: str, seed: int, stamp: str) -> Path:
    return root / (
        f"h1hand-{task}-v0__h1hand_{task}_admission_v1_{mode}_s{seed}_{stamp}"
        f"__{seed}_final.pt"
    )


def _curve(path: Path) -> list[dict[str, float | int]]:
    if not path.is_file():
        return []
    return [
        {"step": int(step), "return": float(ret), "length": float(length)}
        for step, ret, length in EVAL_RE.findall(path.read_text(errors="replace"))
    ]


def _sum_sources(values: list[int]) -> int:
    return int(sum(int(value) for value in values[:-1]))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _implementation_integrity(path: Path) -> dict[str, object]:
    rows = []
    for raw in path.read_text().splitlines():
        expected, filename = raw.split(maxsplit=1)
        target = Path(filename)
        actual = _sha256(target) if target.is_file() else None
        rows.append(
            {
                "path": filename,
                "expected_sha256": expected,
                "actual_sha256": actual,
                "matches": actual == expected,
                "mtime_ns": target.stat().st_mtime_ns if target.is_file() else None,
            }
        )
    return {
        "manifest": str(path),
        "all_match": bool(rows) and all(bool(row["matches"]) for row in rows),
        "files": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=Path, default=Path("models"))
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("logs/train/admission_core_v1_20260712TFINALV2Z"),
    )
    parser.add_argument(
        "--basketball-log-dir",
        type=Path,
        default=Path("logs/train/admission_core_v1_20260712TFINALV2Z"),
    )
    parser.add_argument("--stamp", default="20260712TFINALV2Z")
    parser.add_argument("--basketball-stamp", default="20260712TFINALV2Z")
    parser.add_argument("--powerlift-stamp", default="20260712TFINALV2Z")
    parser.add_argument(
        "--revocation-checkpoint",
        type=Path,
        default=Path(
            "models/h1hand-powerlift-v0__admission_schedule_revoke_smoke_final_v2__1_final.pt"
        ),
    )
    parser.add_argument(
        "--out", type=Path, default=Path("artifacts/admission_core_v1/engineering_audit.json")
    )
    parser.add_argument(
        "--quarantine-artifact",
        type=Path,
        default=Path(
            "artifacts/source_admission_gate_v1/cabinet_s1_step10000/quarantine.pt"
        ),
    )
    parser.add_argument(
        "--implementation-hashes",
        type=Path,
        default=Path("artifacts/admission_core_v1/final_v2_implementation_sha256.txt"),
    )
    args = parser.parse_args()

    rows = []
    for task, cell, mode in (
        ("basketball", "basketball_exact_none", "none"),
        ("powerlift", "powerlift_retain_all", "all"),
    ):
        for seed in (1, 2, 3):
            checkpoint = _checkpoint(
                args.models,
                task,
                mode,
                seed,
                args.basketball_stamp if task == "basketball" else args.powerlift_stamp,
            )
            log_path = (
                args.basketball_log_dir if task == "basketball" else args.log_dir
            ) / f"{cell}_s{seed}.log"
            row = {
                "task": task,
                "cell": cell,
                "mode": mode,
                "seed": seed,
                "checkpoint": str(checkpoint),
                "checkpoint_exists": checkpoint.is_file(),
                "eval_curve": _curve(log_path),
            }
            if checkpoint.is_file():
                payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
                audit = payload.get("admission_audit") or {}
                row["global_step"] = int(payload.get("global_step", -1))
                row["audit"] = audit
                row["source_execution_count"] = _sum_sources(audit["execution_counts"])
                row["source_main_buffer_count"] = _sum_sources(audit["main_buffer_counts"])
                row["source_critic_sample_count"] = _sum_sources(
                    audit["critic_sample_counts"]
                )
                row["student_candidate_mass"] = float(audit["candidate_masses"][-1])
                row["actor_sampling"] = audit.get("actor_sampling")
                row["admission_mode"] = (payload.get("ptf_cfg") or {}).get(
                    "admission_mode"
                )
                row["exact_zero_source_channels"] = all(
                    row[key] == 0
                    for key in (
                        "source_execution_count",
                        "source_main_buffer_count",
                        "source_critic_sample_count",
                    )
                )
            rows.append(row)

    basketball = [row for row in rows if row["task"] == "basketball"]
    powerlift = [row for row in rows if row["task"] == "powerlift"]
    complete = all(row["checkpoint_exists"] for row in rows)
    revocation = {"checkpoint": str(args.revocation_checkpoint), "checkpoint_exists": False}
    if args.revocation_checkpoint.is_file():
        payload = torch.load(
            args.revocation_checkpoint, map_location="cpu", weights_only=False
        )
        audit = payload["admission_audit"]
        change = audit["decision_history"][-1]
        event = audit["policy_events"][-1]
        final_execution = audit["execution_counts"]
        execution_at_change = change["execution_counts_at_apply"]
        final_samples = audit["critic_sample_counts"]
        samples_at_change = event["sample_counts_at_apply"]["critic"]
        revocation = {
            "checkpoint": str(args.revocation_checkpoint),
            "checkpoint_exists": True,
            "change_step": int(change["step"]),
            "physical_source_count": _sum_sources(audit["main_buffer_counts"]),
            "active_source_count": _sum_sources(audit["active_buffer_counts"]),
            "source_execution_after_change": _sum_sources(final_execution)
            - _sum_sources(execution_at_change),
            "source_critic_samples_after_change": _sum_sources(final_samples)
            - _sum_sources(samples_at_change),
        }
        revocation["exact_revocation"] = (
            revocation["physical_source_count"] > 0
            and revocation["active_source_count"] == 0
            and revocation["source_execution_after_change"] == 0
            and revocation["source_critic_samples_after_change"] == 0
        )
    quarantine = {
        "artifact": str(args.quarantine_artifact),
        "artifact_exists": args.quarantine_artifact.is_file(),
        "validated": False,
    }
    if args.quarantine_artifact.is_file():
        bank = torch.load(
            args.quarantine_artifact, map_location="cpu", weights_only=False
        )
        validate_quarantine_bank(bank)
        metadata = bank["metadata"]
        quarantine.update(
            {
                "validated": True,
                "file_sha256": _sha256(args.quarantine_artifact),
                "content_digest": bank.get("content_digest"),
                "quarantine_only": bool(metadata["quarantine_only"]),
                "learner_updates": int(metadata["learner_updates"]),
                "main_replay_writes": int(metadata["main_replay_writes"]),
                "valid_anchors": int(metadata["valid_anchors"]),
                "source_horizon": int(metadata["source_horizon"]),
                "followup_horizon": int(metadata["followup_horizon"]),
                "paths": sorted(bank["paths"]),
                "sources": sorted(bank["sources"]),
                "provenance_groups": list(metadata["provenance_groups"]),
            }
        )
    implementation = {
        "manifest": str(args.implementation_hashes),
        "all_match": False,
        "files": [],
    }
    if args.implementation_hashes.is_file():
        implementation = _implementation_integrity(args.implementation_hashes)
    report = {
        "schema_version": 1,
        "experiment": "admission_core_v1",
        "complete": complete,
        "engineering_verdict": {
            "exact_abstention_all_seeds": complete
            and all(row.get("exact_zero_source_channels", False) for row in basketball),
            "powerlift_student_mass_half_all_seeds": complete
            and all(
                math.isclose(row.get("student_candidate_mass", math.nan), 0.5, abs_tol=1e-7)
                for row in powerlift
            ),
            "runtime_revocation_exact": bool(revocation.get("exact_revocation", False)),
            "actor_critic_shared_batch_all_seeds": complete
            and all(row.get("actor_sampling") == "shared_critic_batch" for row in rows),
            "quarantine_main_replay_isolated": bool(quarantine.get("validated"))
            and bool(quarantine.get("quarantine_only"))
            and quarantine.get("learner_updates") == 0
            and quarantine.get("main_replay_writes") == 0,
            "implementation_hashes_match": bool(implementation.get("all_match")),
            "scientific_performance_verdict": "pending_paired_evaluation",
        },
        "runtime_revocation": revocation,
        "quarantine": quarantine,
        "implementation_integrity": implementation,
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report["engineering_verdict"], indent=2, sort_keys=True))
    if not complete:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
