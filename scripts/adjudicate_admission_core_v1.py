#!/usr/bin/env python3
"""Apply the preregistered admission-core performance gates."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _aggregate(summary: dict, task: str, condition: str, metric: str) -> dict:
    return summary["tasks"][task]["comparisons"][condition]["by_checkpoint_step"][
        "100000"
    ]["aggregate_over_train_seeds"][metric]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--md-out", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text())
    if not summary["coverage"]["complete"]:
        raise ValueError("paired evaluation coverage is incomplete")
    if not summary["exact_reset_pairing_validated"]:
        raise ValueError("reset pairing was not validated")

    bb_adm = _aggregate(
        summary, "basketball", "admission_none", "common_prefix_progress_delta"
    )
    bb_wfix = _aggregate(summary, "basketball", "wfix", "common_prefix_progress_delta")
    bb_return = _aggregate(summary, "basketball", "admission_none", "return_delta")
    bb_recovery = float(bb_adm["mean"] - bb_wfix["mean"])
    bb_checks = {
        "progress_delta_ge_minus_0_05": float(bb_adm["mean"]) >= -0.05,
        "recovery_over_wfix_ge_0_10": bb_recovery >= 0.10,
        "t_vs_zero_gt_minus_2_92": float(bb_adm["t_vs_zero"]) > -2.92,
    }

    pl_adm = _aggregate(
        summary, "powerlift", "admission_all", "common_prefix_progress_delta"
    )
    pl_wfix = _aggregate(summary, "powerlift", "wfix", "common_prefix_progress_delta")
    pl_return = _aggregate(summary, "powerlift", "admission_all", "return_delta")
    retention = (
        float(pl_adm["mean"]) / float(pl_wfix["mean"])
        if float(pl_wfix["mean"]) > 0
        else float("nan")
    )
    pl_checks = {
        "progress_delta_positive": float(pl_adm["mean"]) > 0.0,
        "t_vs_zero_ge_2_92": float(pl_adm["t_vs_zero"]) >= 2.92,
        "retention_fraction_ge_0_50": retention >= 0.50,
    }
    result = {
        "schema_version": 1,
        "experiment": "admission_core_v1",
        "evaluation_step": 100000,
        "coverage_complete": True,
        "exact_reset_pairing_validated": True,
        "basketball_negative_safety": {
            "pass": all(bb_checks.values()),
            "checks": bb_checks,
            "admission_progress_delta": bb_adm,
            "legacy_wfix_progress_delta": bb_wfix,
            "recovery_over_legacy_wfix": bb_recovery,
            "admission_return_delta": bb_return,
        },
        "powerlift_positive_retention": {
            "pass": all(pl_checks.values()),
            "checks": pl_checks,
            "admission_progress_delta": pl_adm,
            "legacy_wfix_progress_delta": pl_wfix,
            "retention_fraction": retention,
            "admission_return_delta": pl_return,
        },
    }
    result["both_performance_gates_pass"] = bool(
        result["basketball_negative_safety"]["pass"]
        and result["powerlift_positive_retention"]["pass"]
    )
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(result, indent=2, allow_nan=True) + "\n")
    lines = [
        "# Admission Core v1 performance adjudication",
        "",
        f"- Basketball negative-safety gate: **{'PASS' if result['basketball_negative_safety']['pass'] else 'FAIL'}**",
        f"- Powerlift positive-retention gate: **{'PASS' if result['powerlift_positive_retention']['pass'] else 'FAIL'}**",
        f"- Both gates: **{'PASS' if result['both_performance_gates_pass'] else 'FAIL'}**",
        "",
        "The outcome validates or falsifies the fixed admission configuration; it does not validate an automatic transferability estimator.",
        "",
        "```json",
        json.dumps(result, indent=2, allow_nan=True),
        "```",
    ]
    args.md_out.write_text("\n".join(lines) + "\n")
    print(json.dumps({
        "basketball": result["basketball_negative_safety"]["pass"],
        "powerlift": result["powerlift_positive_retention"]["pass"],
        "both": result["both_performance_gates_pass"],
    }, indent=2))


if __name__ == "__main__":
    main()
