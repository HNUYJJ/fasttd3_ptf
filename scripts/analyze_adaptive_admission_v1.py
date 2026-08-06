#!/usr/bin/env python3
"""Analyze and literally adjudicate adaptive_admission_v1.

The evaluation grid and thresholds are read from the frozen protocol.  Truck
and powerlift use the preregistered admission_handoff_v1 comparators; crawl and
basketball use the matched static runs from the same launch stamp.  The script
also reads immutable checkpoint decision histories so performance and source
lifecycle correctness remain separate evidence channels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any

import torch
import yaml


EVAL_RE = re.compile(rb"\[eval\] step=(\d+) return=([-+0-9.eE]+)")
WANDB_RE = re.compile(rb"View run at (https://wandb\.ai/[^\s]+)")
SEEDS = (1, 2, 3)
TASKS = ("crawl", "truck", "powerlift", "basketball")
HANDOFF_STAMP = "20260713THANDOFFV1Z"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def eval_curve(path: Path) -> dict[int, float]:
    matches = EVAL_RE.findall(path.read_bytes())
    curve: dict[int, float] = {}
    for raw_step, raw_value in matches:
        step = int(raw_step)
        if step in curve:
            raise ValueError(f"duplicate eval step {step} in {path}")
        curve[step] = float(raw_value)
    return curve


def wandb_url(path: Path) -> str | None:
    matches = WANDB_RE.findall(path.read_bytes())
    return matches[-1].decode() if matches else None


def current_log(root: Path, stamp: str, task: str, mode: str, seed: int) -> Path:
    return (
        root
        / f"logs/train/adaptive_admission_v1_{stamp}"
        / f"{task}_{mode}_s{seed}.log"
    )


def comparator_log(root: Path, stamp: str, task: str, seed: int) -> Path:
    if task in {"crawl", "basketball"}:
        return current_log(root, stamp, task, "static", seed)
    cell = (
        "truck_admission_h4_fix"
        if task == "truck"
        else "powerlift_admission_all_fix"
    )
    return (
        root
        / f"logs/train/admission_handoff_v1_{HANDOFF_STAMP}"
        / f"{cell}_s{seed}.log"
    )


def experiment_checkpoint(
    root: Path, stamp: str, task: str, mode: str, seed: int, suffix: str
) -> Path:
    exp = f"h1hand_{task}_adaptive_admission_v1_{mode}_s{seed}_{stamp}"
    return root / "models" / f"h1hand-{task}-v0__{exp}__{seed}_{suffix}.pt"


def latest_checkpoint(
    root: Path, stamp: str, task: str, mode: str, seed: int
) -> Path | None:
    for suffix in ("final", "90000", "60000", "30000"):
        path = experiment_checkpoint(root, stamp, task, mode, seed, suffix)
        if path.is_file():
            return path
    return None


def checkpoint_lifecycle(path: Path, *, expected_adaptive: bool) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    audit = payload["admission_audit"]
    history = list(audit.get("decision_history") or [])
    initial = next(
        event for event in history if event.get("decision_id") == "explicit-all"
    )
    source_names = list(initial["source_names"])
    execution = [int(value) for value in audit["execution_counts"]]
    critic = [int(value) for value in audit["critic_sample_counts"]]
    active = [int(value) for value in audit["active_buffer_counts"]]
    effective = [float(value) for value in audit["effective_replay_masses"]]
    admitted = [bool(value) for value in audit["admitted_sources"]]
    windows = [
        event for event in history if event.get("event") == "adaptive_admission_window"
    ]
    revocations = []
    checks: dict[str, bool] = {
        "adaptive_flag": bool(payload["ptf_cfg"].get("admission_adaptive"))
        == expected_adaptive,
        "adaptive_history_consistent": bool(windows) == expected_adaptive,
        "physical_after_authority": payload["ptf_cfg"].get(
            "admission_replay_handoff"
        )
        == "physical_after_authority",
        "actor_shared_critic_batch": audit.get("actor_sampling")
        == "shared_critic_batch",
        "no_independent_actor_samples": sum(
            int(value) for value in audit["actor_independent_sample_counts"]
        )
        == 0,
    }
    warmup = int(payload["ptf_cfg"]["mcg_warmup_steps"])
    if int(payload["global_step"]) >= warmup:
        checks["authority_released"] = audit.get("source_authority_active") is False
        checks["physical_sampling_phase"] = (
            audit.get("sampling_phase") == "physical_allowed"
        )

    for event in windows:
        revoked = list(event.get("revoked_sources") or [])
        if not revoked:
            continue
        step = int(event["completed_step"])
        replay = event["replay_at_apply"]
        execution_at_apply = [int(value) for value in event["execution_counts_at_apply"]]
        critic_at_apply = [int(value) for value in replay["critic_sample_counts"]]
        active_at_apply = [int(value) for value in replay["active_buffer_counts"]]
        effective_at_apply = [float(value) for value in replay["effective_replay_masses"]]
        candidate_at_apply = [float(value) for value in replay["candidate_masses"]]
        admitted_at_apply = [bool(value) for value in replay["admitted_sources"]]
        source_rows = []
        for name in revoked:
            index = source_names.index(name)
            later_counts = []
            for later in windows:
                if int(later["completed_step"]) <= step:
                    continue
                stat = next(
                    row for row in later["statistics"] if row["candidate"] == name
                )
                later_counts.append(int(stat["count"]))
            source_checks = {
                "admitted_zero_at_apply": not admitted_at_apply[index],
                "candidate_mass_zero_at_apply": candidate_at_apply[index] == 0.0,
                "active_replay_zero_at_apply": active_at_apply[index] == 0,
                "effective_mass_zero_at_apply": effective_at_apply[index] == 0.0,
                "execution_frozen": execution[index] == execution_at_apply[index],
                "critic_frozen": critic[index] == critic_at_apply[index],
                "admitted_zero_at_checkpoint": not admitted[index],
                "active_replay_zero_at_checkpoint": active[index] == 0,
                "effective_mass_zero_at_checkpoint": effective[index] == 0.0,
                "later_behavior_zero": not later_counts
                or all(value == 0 for value in later_counts),
            }
            for check, passed in source_checks.items():
                checks[f"step_{step}_{name}_{check}"] = passed
            source_rows.append(
                {
                    "source": name,
                    "statistics": next(
                        row for row in event["statistics"] if row["candidate"] == name
                    ),
                    "execution_at_apply": execution_at_apply[index],
                    "execution_at_checkpoint": execution[index],
                    "critic_at_apply": critic_at_apply[index],
                    "critic_at_checkpoint": critic[index],
                    "later_window_counts": later_counts,
                    "checks": source_checks,
                }
            )
        revocations.append(
            {
                "step": step,
                "window_index": int(event["window_index"]),
                "sources": source_rows,
                "student_statistics": next(
                    row
                    for row in event["statistics"]
                    if row["candidate"] == "student"
                ),
                "persistence_counts": event["persistence_counts"],
                "discarded_partial_segments": int(
                    event.get("discarded_partial_segments", 0)
                ),
            }
        )
    return {
        "checkpoint": str(path),
        "checkpoint_sha256": sha256(path),
        "global_step": int(payload["global_step"]),
        "expected_adaptive": expected_adaptive,
        "source_names": source_names,
        "revocations": revocations,
        "admitted_sources": admitted,
        "source_authority_active": bool(audit["source_authority_active"]),
        "sampling_phase": audit["sampling_phase"],
        "checks": checks,
        "pass": all(checks.values()),
    }


def formal_no_trigger_match(
    adaptive_path: Path, static_path: Path, *, expected_step: int
) -> dict[str, Any]:
    adaptive_payload = torch.load(
        adaptive_path, map_location="cpu", weights_only=False
    )
    static_payload = torch.load(static_path, map_location="cpu", weights_only=False)
    adaptive_audit = adaptive_payload["admission_audit"]
    static_audit = static_payload["admission_audit"]
    revocations = [
        event
        for event in adaptive_audit.get("decision_history") or []
        if event.get("event") == "adaptive_admission_window"
        and event.get("revoked_sources")
    ]
    if revocations:
        return {
            "applicable": False,
            "reason": "adaptive run triggered before or at the 30k checkpoint",
            "revocation_steps": [int(event["completed_step"]) for event in revocations],
        }
    keys = (
        "execution_counts",
        "main_buffer_counts",
        "active_buffer_counts",
        "critic_sample_counts",
        "actor_independent_sample_counts",
        "candidate_masses",
    )
    checks = {
        key: list(adaptive_audit[key]) == list(static_audit[key]) for key in keys
    }
    checks["adaptive_expected_step"] = (
        int(adaptive_payload["global_step"]) == expected_step
    )
    checks["static_expected_step"] = (
        int(static_payload["global_step"]) == expected_step
    )
    return {
        "applicable": True,
        "expected_step": expected_step,
        "adaptive_checkpoint": str(adaptive_path),
        "static_checkpoint": str(static_path),
        "checks": checks,
        "pass": all(checks.values()),
        "adaptive": {key: list(adaptive_audit[key]) for key in keys},
        "static": {key: list(static_audit[key]) for key in keys},
    }


def paired_seed(
    adaptive: dict[int, float], comparator: dict[int, float], grid: list[int]
) -> dict[str, Any]:
    available = [step for step in grid if step in adaptive and step in comparator]
    missing = [step for step in grid if step not in adaptive or step not in comparator]
    deltas = {step: adaptive[step] - comparator[step] for step in available}
    return {
        "available_steps": available,
        "missing_steps": missing,
        "adaptive": {step: adaptive[step] for step in available},
        "comparator": {step: comparator[step] for step in available},
        "delta": deltas,
        "mean_delta": statistics.mean(deltas.values()) if deltas else None,
        "complete": not missing,
    }


def no_trigger_regression(root: Path) -> dict[str, Any]:
    path = root / "artifacts/adaptive_admission_v1/no_trigger_smoke_20260714.json"
    payload = json.loads(path.read_text())
    checks = {
        "status_pass": str(payload.get("status", "")).startswith("pass"),
        "no_decisions": int(payload.get("adaptive_decisions", -1)) == 0,
        "unit_trace_present": bool(payload.get("unit_trace_evidence")),
        "execution_strata_present": bool(
            payload.get("adaptive_and_static_execution_counts")
        ),
        "critic_strata_present": bool(
            payload.get("adaptive_and_static_critic_sample_counts")
        ),
    }
    return {
        "artifact": str(path),
        "artifact_sha256": sha256(path),
        "checks": checks,
        "pass": all(checks.values()),
        "cuda_weight_boundary": payload.get("cuda_weight_boundary"),
    }


def historical_comparator_verification(root: Path) -> dict[str, Any]:
    directory = (
        root
        / "artifacts/admission_handoff_v1"
        / HANDOFF_STAMP
        / "training_verification"
    )
    rows: dict[str, Any] = {}
    for task, cell in (
        ("truck", "truck_admission_h4_fix"),
        ("powerlift", "powerlift_admission_all_fix"),
    ):
        for seed in SEEDS:
            path = directory / f"{cell}_s{seed}.json"
            payload = json.loads(path.read_text())
            rows[f"{task}_s{seed}"] = {
                "artifact": str(path),
                "artifact_sha256": sha256(path),
                "pass": payload.get("pass") is True,
            }
    return {"rows": rows, "pass": all(row["pass"] for row in rows.values())}


def analyze(root: Path, stamp: str, *, allow_incomplete: bool) -> dict[str, Any]:
    protocol_path = root / "configs/experiments/adaptive_admission_v1.yaml"
    protocol = yaml.safe_load(protocol_path.read_text())
    grid = [int(value) for value in protocol["preregistered_gates"]["evaluation_grid"]]
    task_reports: dict[str, Any] = {}
    for task in TASKS:
        seeds: dict[str, Any] = {}
        for seed in SEEDS:
            adaptive_log = current_log(root, stamp, task, "adaptive", seed)
            baseline_log = comparator_log(root, stamp, task, seed)
            if not adaptive_log.is_file() or not baseline_log.is_file():
                if not allow_incomplete:
                    raise FileNotFoundError(adaptive_log if not adaptive_log.is_file() else baseline_log)
                seeds[f"s{seed}"] = {
                    "available": False,
                    "adaptive_log": str(adaptive_log),
                    "comparator_log": str(baseline_log),
                }
                continue
            paired = paired_seed(eval_curve(adaptive_log), eval_curve(baseline_log), grid)
            checkpoint = latest_checkpoint(root, stamp, task, "adaptive", seed)
            static_checkpoint = (
                latest_checkpoint(root, stamp, task, "static", seed)
                if task in {"crawl", "basketball"}
                else None
            )
            formal_matches = []
            if task in {"crawl", "basketball"}:
                for suffix, expected_step in (
                    ("30000", 30000),
                    ("60000", 60000),
                    ("90000", 90000),
                    ("final", 100000),
                ):
                    adaptive_match_path = experiment_checkpoint(
                        root, stamp, task, "adaptive", seed, suffix
                    )
                    static_match_path = experiment_checkpoint(
                        root, stamp, task, "static", seed, suffix
                    )
                    if adaptive_match_path.is_file() and static_match_path.is_file():
                        formal_matches.append(
                            formal_no_trigger_match(
                                adaptive_match_path,
                                static_match_path,
                                expected_step=expected_step,
                            )
                        )
            seeds[f"s{seed}"] = {
                "available": True,
                "adaptive_log": str(adaptive_log),
                "comparator_log": str(baseline_log),
                "wandb_url": wandb_url(adaptive_log),
                "paired_grid": paired,
                "checkpoint_lifecycle": (
                    checkpoint_lifecycle(checkpoint, expected_adaptive=True)
                    if checkpoint
                    else None
                ),
                "matched_static_checkpoint_lifecycle": (
                    checkpoint_lifecycle(static_checkpoint, expected_adaptive=False)
                    if static_checkpoint
                    else None
                ),
                "formal_no_trigger_match": (
                    formal_matches[0] if formal_matches else None
                ),
                "formal_no_trigger_checkpoint_matches": formal_matches,
            }
        complete_seed_rows = [
            row
            for row in seeds.values()
            if row.get("available") and row["paired_grid"]["complete"]
        ]
        available_seed_rows = [
            row for row in seeds.values() if row.get("available") and row["paired_grid"]["available_steps"]
        ]
        per_seed_means = [
            row["paired_grid"]["mean_delta"] for row in complete_seed_rows
        ]
        task_reports[task] = {
            "seeds": seeds,
            "complete_seed_count": len(complete_seed_rows),
            "available_seed_count": len(available_seed_rows),
            "per_seed_complete_mean_deltas": per_seed_means,
            "complete_mean_delta": (
                statistics.mean(per_seed_means) if len(per_seed_means) == 3 else None
            ),
        }

    gates: dict[str, Any] = {}
    crawl = task_reports["crawl"]
    if crawl["complete_seed_count"] == 3:
        revocation_by_seed = {
            seed: bool(row["checkpoint_lifecycle"]["revocations"])
            for seed, row in crawl["seeds"].items()
        }
        mean_delta_by_seed = {
            seed: float(row["paired_grid"]["mean_delta"])
            for seed, row in crawl["seeds"].items()
        }
        triggered_seed_means = {
            seed: mean_delta_by_seed[seed]
            for seed, triggered in revocation_by_seed.items()
            if triggered
        }
        no_trigger_placebo_means = {
            seed: mean_delta_by_seed[seed]
            for seed, triggered in revocation_by_seed.items()
            if not triggered
        }
        event_aligned_triggered: dict[str, Any] = {}
        for seed, triggered in revocation_by_seed.items():
            if not triggered:
                continue
            row = crawl["seeds"][seed]
            first_revocation_step = min(
                int(event["step"])
                for event in row["checkpoint_lifecycle"]["revocations"]
            )
            deltas = {
                int(step): float(value)
                for step, value in row["paired_grid"]["delta"].items()
            }
            pre_steps = sorted(
                step for step in deltas if step < first_revocation_step
            )
            post_steps = sorted(
                step for step in deltas if step >= first_revocation_step
            )
            pre_mean = (
                statistics.mean(deltas[step] for step in pre_steps)
                if pre_steps
                else None
            )
            post_mean = (
                statistics.mean(deltas[step] for step in post_steps)
                if post_steps
                else None
            )
            event_aligned_triggered[seed] = {
                "first_revocation_step": first_revocation_step,
                "pre_steps": pre_steps,
                "post_steps": post_steps,
                "pre_mean_delta": pre_mean,
                "post_mean_delta": post_mean,
                "post_minus_pre_delta": (
                    post_mean - pre_mean
                    if pre_mean is not None and post_mean is not None
                    else None
                ),
            }
        checks = {
            "mean_delta_at_least_30": crawl["complete_mean_delta"]
            >= float(protocol["preregistered_gates"]["crawl"]["adaptive_minus_static_mean_return_min"]),
            "positive_3_of_3": sum(
                value > 0 for value in crawl["per_seed_complete_mean_deltas"]
            )
            == int(protocol["preregistered_gates"]["crawl"]["positive_seed_count"]),
            "revocation_event_exists": any(revocation_by_seed.values()),
        }
        gates["crawl"] = {
            "status": "PASS" if all(checks.values()) else "FAIL",
            "checks": checks,
            "revocation_by_seed": revocation_by_seed,
            "attribution_diagnostic": {
                "is_preregistered_gate": False,
                "triggered_seed_count": sum(revocation_by_seed.values()),
                "all_three_seeds_triggered": all(revocation_by_seed.values()),
                "mean_delta_by_seed": mean_delta_by_seed,
                "triggered_seed_mean_deltas": triggered_seed_means,
                "no_trigger_placebo_mean_deltas": no_trigger_placebo_means,
                "event_aligned_triggered_seeds": event_aligned_triggered,
                "mechanism_attribution_supported": all(checks.values())
                and all(revocation_by_seed.values()),
                "interpretation": (
                    "Literal gate and mechanism attribution are distinct. "
                    "A no-trigger adaptive/static pair is algorithmically the same "
                    "method; its return difference is not revocation evidence."
                ),
            },
        }
    else:
        completed_means = crawl["per_seed_complete_mean_deltas"]
        positive_completed = sum(value > 0 for value in completed_means)
        required_positive = int(
            protocol["preregistered_gates"]["crawl"]["positive_seed_count"]
        )
        remaining = 3 - len(completed_means)
        gates["crawl"] = {
            "status": "PENDING",
            "feasibility": {
                "completed_seed_count": len(completed_means),
                "positive_completed_seed_count": positive_completed,
                "remaining_seed_count": remaining,
                "positive_seed_gate_still_reachable": (
                    positive_completed + remaining >= required_positive
                ),
            },
        }

    truck = task_reports["truck"]
    if truck["complete_seed_count"] == 3:
        forbidden = set(protocol["preregistered_gates"]["truck"]["forbidden_revocations"])
        observed = {
            source["source"]
            for row in truck["seeds"].values()
            for event in row["checkpoint_lifecycle"]["revocations"]
            for source in event["sources"]
        }
        checks = {
            "absolute_mean_delta_at_most_60": abs(truck["complete_mean_delta"])
            <= float(protocol["preregistered_gates"]["truck"]["abs_adaptive_minus_fix_mean_return_max"]),
            "forbidden_revocations_absent": not (observed & forbidden),
        }
        gates["truck"] = {
            "status": "PASS" if all(checks.values()) else "FAIL",
            "checks": checks,
            "observed_revocations": sorted(observed),
        }
    else:
        gates["truck"] = {"status": "PENDING"}

    power = task_reports["powerlift"]
    if power["complete_seed_count"] == 3:
        checks = {
            "mean_delta_at_least_minus_20": power["complete_mean_delta"]
            >= float(protocol["preregistered_gates"]["powerlift"]["adaptive_minus_fix_mean_return_min"])
        }
        gates["powerlift"] = {
            "status": "PASS" if all(checks.values()) else "FAIL",
            "checks": checks,
        }
    else:
        gates["powerlift"] = {"status": "PENDING"}
    gates["basketball"] = {"status": "DESCRIPTIVE"}

    adaptive_mechanism_rows = [
        row["checkpoint_lifecycle"]
        for task in task_reports.values()
        for row in task["seeds"].values()
        if row.get("checkpoint_lifecycle") is not None
    ]
    static_mechanism_rows = [
        row["matched_static_checkpoint_lifecycle"]
        for task in task_reports.values()
        for row in task["seeds"].values()
        if row.get("matched_static_checkpoint_lifecycle") is not None
    ]
    mechanism_rows = adaptive_mechanism_rows + static_mechanism_rows
    grid_complete = all(
        task["complete_seed_count"] == 3 for task in task_reports.values()
    )
    adaptive_finals_complete = (
        len(adaptive_mechanism_rows) == 12
        and all(row["global_step"] == 100000 for row in adaptive_mechanism_rows)
    )
    static_finals_complete = (
        len(static_mechanism_rows) == 6
        and all(row["global_step"] == 100000 for row in static_mechanism_rows)
    )
    mechanism = {
        "audited_checkpoint_count": len(mechanism_rows),
        "all_available_pass": bool(mechanism_rows)
        and all(row["pass"] for row in mechanism_rows),
        "all_12_final_adaptive_checkpoints_audited": adaptive_finals_complete,
        "all_6_final_matched_static_checkpoints_audited": static_finals_complete,
    }
    complete = grid_complete and adaptive_finals_complete and static_finals_complete
    regression = no_trigger_regression(root)
    formal_no_trigger_rows = [
        match
        for task in task_reports.values()
        for row in task["seeds"].values()
        for match in row.get("formal_no_trigger_checkpoint_matches") or []
        if match.get("applicable") is True
    ]
    formal_no_trigger = {
        "applicable_pair_count": len(formal_no_trigger_rows),
        "all_applicable_pass": all(
            row.get("pass") is True for row in formal_no_trigger_rows
        ),
        "rows": formal_no_trigger_rows,
    }
    comparator_verification = historical_comparator_verification(root)
    report = {
        "schema_version": 1,
        "experiment": "adaptive_admission_v1",
        "stamp": stamp,
        "protocol": str(protocol_path),
        "protocol_sha256": sha256(protocol_path),
        "evaluation_grid": grid,
        "comparator_policy": {
            "crawl": "same-launch matched static",
            "basketball": "same-launch matched static",
            "truck": f"admission_handoff_v1/{HANDOFF_STAMP}/truck_admission_h4_fix",
            "powerlift": f"admission_handoff_v1/{HANDOFF_STAMP}/powerlift_admission_all_fix",
        },
        "tasks": task_reports,
        "preregistered_gates": gates,
        "mechanism_audit": mechanism,
        "no_trigger_regression": regression,
        "formal_no_trigger_regression": formal_no_trigger,
        "historical_comparator_verification": comparator_verification,
        "evaluation_grid_complete": grid_complete,
        "complete": complete,
    }
    if complete:
        report["overall_status"] = (
            "PASS"
            if all(gates[name]["status"] == "PASS" for name in ("crawl", "truck", "powerlift"))
            and mechanism["all_12_final_adaptive_checkpoints_audited"]
            and mechanism["all_6_final_matched_static_checkpoints_audited"]
            and mechanism["all_available_pass"]
            and regression["pass"]
            and formal_no_trigger["all_applicable_pass"]
            and comparator_verification["pass"]
            else "FAIL"
        )
    else:
        report["overall_status"] = "PENDING"
    return report


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# adaptive_admission_v1 adjudication",
        "",
        f"- Stamp: `{report['stamp']}`",
        f"- Status: **{report['overall_status']}**",
        f"- Complete: `{report['complete']}`",
        "",
        "## Preregistered gates",
        "",
        "| Task | Complete seeds | Mean adaptive-comparator | Gate |",
        "|---|---:|---:|---|",
    ]
    for task in TASKS:
        row = report["tasks"][task]
        mean = row["complete_mean_delta"]
        mean_text = "pending" if mean is None else f"{mean:+.3f}"
        lines.append(
            f"| {task} | {row['complete_seed_count']}/3 | {mean_text} | "
            f"{report['preregistered_gates'][task]['status']} |"
        )
    lines.extend(
        [
            "",
            "## Mechanism evidence",
            "",
            f"- Available lifecycle audits pass: `{report['mechanism_audit']['all_available_pass']}`",
            f"- All 12 final adaptive checkpoints audited: `{report['mechanism_audit']['all_12_final_adaptive_checkpoints_audited']}`",
            f"- All 6 final matched-static checkpoints audited: `{report['mechanism_audit']['all_6_final_matched_static_checkpoints_audited']}`",
            f"- No-trigger regression: `{report['no_trigger_regression']['pass']}`",
            f"- Formal no-trigger matched pairs pass: `{report['formal_no_trigger_regression']['all_applicable_pass']}`",
            f"- Historical comparator verification: `{report['historical_comparator_verification']['pass']}`",
            "",
            "Revocation times and per-source execution/replay/critic freeze evidence are stored in the JSON companion.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--stamp", required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    report = analyze(root, args.stamp, allow_incomplete=args.allow_incomplete)
    out_dir = args.out_dir or Path("artifacts/adaptive_admission_v1") / args.stamp
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"analysis_{args.stamp}.json"
    md_path = out_dir / f"analysis_{args.stamp}.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    md_path.write_text(markdown(report))
    print(
        json.dumps(
            {
                "status": report["overall_status"],
                "complete": report["complete"],
                "json": str(json_path),
                "markdown": str(md_path),
            },
            indent=2,
        )
    )
    if not args.allow_incomplete and report["overall_status"] != "PASS":
        raise SystemExit(3)


if __name__ == "__main__":
    main()
