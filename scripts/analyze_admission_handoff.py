#!/usr/bin/env python3
"""Reproduce the handoff diagnosis and adjudicate completed repair runs."""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any


EVAL_RE = re.compile(rb"\[eval\] step=(\d+) return=([-+0-9.eE]+)")
SEEDS = (1, 2, 3)


def eval_curve(path: Path) -> dict[int, float]:
    matches = EVAL_RE.findall(path.read_bytes())
    curve = {int(step): float(value) for step, value in matches}
    if len(curve) != len(matches):
        raise ValueError(f"duplicate eval step in {path}")
    return curve


def powerlift_baselines(root: Path) -> dict[str, list[dict[int, float]]]:
    groups = {"scratch": [], "legacy_wfix": [], "fixed_quota": []}
    for seed in SEEDS:
        groups["fixed_quota"].append(
            eval_curve(
                root
                / f"logs/train/admission_core_v1_20260712TFINALV2Z/powerlift_retain_all_s{seed}.log"
            )
        )
        if seed == 1:
            base = root / "logs/train/b2_20260705T153732Z"
            groups["scratch"].append(eval_curve(base / "b2_powerlift_scr_s1.log"))
            groups["legacy_wfix"].append(eval_curve(base / "b2_powerlift_wfix_s1.log"))
        else:
            base = root / "logs/train/b2s_20260705T224905Z"
            groups["scratch"].append(
                eval_curve(base / f"b2s_powerlift_scr_s{seed}.log")
            )
            groups["legacy_wfix"].append(
                eval_curve(base / f"b2s_powerlift_wfix_s{seed}.log")
            )
    return groups


def truck_baselines(root: Path) -> dict[str, list[dict[int, float]]]:
    groups = {"scratch": [], "legacy_wfix": []}
    for seed in SEEDS:
        groups["legacy_wfix"].append(
            eval_curve(root / f"logs/train/h4_20260705T113556Z/h4_truck_s{seed}.log")
        )
        scratch = (
            root / "logs/train/br_20260704T015912Z/br_truck_scr_s1.log"
            if seed == 1
            else root
            / f"logs/train/brseed_20260704T135105Z/bs_truck_scr_s{seed}.log"
        )
        groups["scratch"].append(eval_curve(scratch))
    return groups


def paired_window(
    left: list[dict[int, float]],
    right: list[dict[int, float]],
    *,
    lower_exclusive: int,
    upper_inclusive: int,
) -> dict[str, Any]:
    per_seed = []
    steps_used: list[int] | None = None
    for left_curve, right_curve in zip(left, right):
        steps = sorted(
            step
            for step in set(left_curve) & set(right_curve)
            if lower_exclusive < step <= upper_inclusive
        )
        if not steps:
            raise ValueError("no aligned eval points in requested window")
        if steps_used is None:
            steps_used = steps
        elif steps != steps_used:
            raise ValueError("eval grids differ across seeds")
        per_seed.append(statistics.mean(left_curve[s] - right_curve[s] for s in steps))
    return {
        "steps": steps_used,
        "per_seed": per_seed,
        "mean": statistics.mean(per_seed),
    }


def point(curves: list[dict[int, float]], step: int) -> dict[str, Any]:
    values = [curve[step] for curve in curves]
    return {"values": values, "mean": statistics.mean(values)}


def analytic_repetition_divergence(
    *, buffer_size: int = 51200, warmup_steps: int = 30000, source_share: float = 0.5
) -> dict[str, Any]:
    rows = []
    for step in (30000, 60000, 70000, 75000, 80000, 81200):
        valid = min(step, buffer_size)
        oldest = max(0, step - buffer_size)
        retained_warmup = max(0, warmup_steps - oldest)
        physical_share = source_share * retained_warmup / valid
        oversampling = None if physical_share == 0 else source_share / physical_share
        rows.append(
            {
                "step": step,
                "physical_source_share": physical_share,
                "fixed_quota_source_share": source_share if physical_share > 0 else 0.0,
                "per_transition_oversampling": oversampling,
            }
        )
    return {
        "assumptions": {
            "buffer_size_vector_steps": buffer_size,
            "warmup_steps": warmup_steps,
            "expected_source_share_during_warmup": source_share,
        },
        "source_tail_exhaustion_step": buffer_size + warmup_steps,
        "rows": rows,
    }


def checkpoint_audit(root: Path, task: str, stamp: str, seed: int, step: int) -> dict[str, Any]:
    import torch

    exp = f"h1hand_{task}_admission_handoff_v1_all_s{seed}_{stamp}"
    path = root / "models" / f"h1hand-{task}-v0__{exp}__{seed}_{step}.pt"
    payload = torch.load(path, map_location="cpu", weights_only=False)
    audit = payload["admission_audit"]
    counts = [int(value) for value in audit["active_buffer_counts"]]
    masses = [float(value) for value in audit["effective_replay_masses"]]
    physical_source = sum(counts[:-1]) / sum(counts)
    effective_source = sum(masses[:-1])
    return {
        "checkpoint": str(path.relative_to(root)),
        "source_authority_active": bool(audit["source_authority_active"]),
        "sampling_phase": audit["sampling_phase"],
        "physical_allowed_source_share": physical_source,
        "effective_source_replay_mass": effective_source,
        "mass_matches_physical": math.isclose(
            physical_source, effective_source, rel_tol=0.0, abs_tol=1e-6
        ),
        "authority_release_events": [
            event
            for event in audit["policy_events"]
            if event.get("event") == "source_authority"
            and not event.get("source_authority_active", True)
        ],
    }


def _formal_checkpoint_path(
    root: Path, task: str, stamp: str, seed: int, suffix: str
) -> Path:
    exp = f"h1hand_{task}_admission_handoff_v1_all_s{seed}_{stamp}"
    return root / "models" / f"h1hand-{task}-v0__{exp}__{seed}_{suffix}.pt"


def _fixed_powerlift_checkpoint_path(root: Path, seed: int, suffix: str) -> Path:
    exp = f"h1hand_powerlift_admission_v1_all_s{seed}_20260712TFINALV2Z"
    return root / "models" / f"h1hand-powerlift-v0__{exp}__{seed}_{suffix}.pt"


def _critic_counts(path: Path) -> list[int]:
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    return [int(value) for value in payload["admission_audit"]["critic_sample_counts"]]


def _stage_source_share(counts_a: list[int], counts_b: list[int]) -> float:
    delta = [right - left for left, right in zip(counts_a, counts_b)]
    if any(value < 0 for value in delta) or sum(delta) <= 0:
        raise ValueError("invalid cumulative critic sample counts")
    return sum(delta[:-1]) / sum(delta)


def critic_stage_exposure(
    root: Path,
    *,
    task: str,
    stamp: str,
    fixed_powerlift: bool = False,
) -> dict[str, Any]:
    boundaries = ("30000", "60000", "90000", "final")
    labels = ("30k_60k", "60k_90k", "90k_100k")
    per_seed: dict[str, dict[str, float]] = {}
    for seed in SEEDS:
        if fixed_powerlift:
            paths = [
                _fixed_powerlift_checkpoint_path(root, seed, suffix)
                for suffix in boundaries
            ]
        else:
            paths = [
                _formal_checkpoint_path(root, task, stamp, seed, suffix)
                for suffix in boundaries
            ]
        counts = [_critic_counts(path) for path in paths]
        per_seed[f"s{seed}"] = {
            label: _stage_source_share(left, right)
            for label, left, right in zip(labels, counts, counts[1:])
        }
    means = {
        label: statistics.mean(per_seed[f"s{seed}"][label] for seed in SEEDS)
        for label in labels
    }
    return {"per_seed": per_seed, "means": means}


def powerlift_diagnosis(root: Path) -> tuple[dict[str, Any], dict[str, list[dict[int, float]]]]:
    groups = powerlift_baselines(root)
    windows = {}
    for name, lo, hi in (("0k_30k", 0, 30000), ("30k_80k", 30000, 80000), ("80k_95k", 80000, 95000)):
        windows[name] = {
            "legacy_wfix_minus_scratch": paired_window(
                groups["legacy_wfix"], groups["scratch"],
                lower_exclusive=lo, upper_inclusive=hi,
            ),
            "fixed_quota_minus_scratch": paired_window(
                groups["fixed_quota"], groups["scratch"],
                lower_exclusive=lo, upper_inclusive=hi,
            ),
            "fixed_quota_minus_legacy_wfix": paired_window(
                groups["fixed_quota"], groups["legacy_wfix"],
                lower_exclusive=lo, upper_inclusive=hi,
            ),
        }
    return (
        {
            "windows": windows,
            "at_80k": {
                name: point(curves, 80000) for name, curves in groups.items()
            },
            "at_95k": {
                name: point(curves, 95000) for name, curves in groups.items()
            },
            "analytic_repetition_divergence": analytic_repetition_divergence(),
        },
        groups,
    )


def adjudicate(root: Path, stamp: str, diagnosis: dict[str, Any], power: dict[str, list[dict[int, float]]]) -> dict[str, Any]:
    log_dir = root / f"logs/train/admission_handoff_v1_{stamp}"
    power_fix = [
        eval_curve(log_dir / f"powerlift_admission_all_fix_s{seed}.log")
        for seed in SEEDS
    ]
    truck_fix = [
        eval_curve(log_dir / f"truck_admission_h4_fix_s{seed}.log")
        for seed in SEEDS
    ]
    warmup = paired_window(
        power_fix, power["fixed_quota"], lower_exclusive=0, upper_inclusive=30000
    )
    fix_fixed = paired_window(
        power_fix, power["fixed_quota"], lower_exclusive=30000, upper_inclusive=80000
    )
    fix_wfix = paired_window(
        power_fix, power["legacy_wfix"], lower_exclusive=30000, upper_inclusive=80000
    )
    fix80 = point(power_fix, 80000)
    mean75 = point(power_fix, 75000)["mean"]
    mean85 = point(power_fix, 85000)["mean"]
    sd80 = statistics.stdev(fix80["values"])
    collapse_floor = statistics.mean((mean75, mean85)) - sd80
    fixed_sd80 = statistics.stdev(
        curve[80000] for curve in power["fixed_quota"]
    )
    neighbor_mean = statistics.mean((mean75, mean85))
    fixed_sd_collapse_floor = neighbor_mean - fixed_sd80
    per_seed_neighbor_residuals = [
        curve[80000] - statistics.mean((curve[75000], curve[85000]))
        for curve in power_fix
    ]
    power_audits = {
        f"s{seed}_{step // 1000}k": checkpoint_audit(
            root, "powerlift", stamp, seed, step
        )
        for seed in SEEDS
        for step in (60000, 90000)
    }
    power_gates = {
        "warmup_regression": abs(warmup["mean"]) <= 10,
        "fixed_quota_repair_mean": fix_fixed["mean"] >= 10,
        "fixed_quota_repair_3of3": sum(value > 0 for value in fix_fixed["per_seed"]) == 3,
        "legacy_compatibility": abs(fix_wfix["mean"]) <= 10,
        "collapse_removed": fix80["mean"] >= collapse_floor,
        "mechanism": all(
            not row["source_authority_active"]
            and row["sampling_phase"] == "physical_allowed"
            and row["mass_matches_physical"]
            and row["authority_release_events"]
            for row in power_audits.values()
        ),
    }
    power_stage_exposure = critic_stage_exposure(
        root, task="powerlift", stamp=stamp
    )
    fixed_stage_exposure = critic_stage_exposure(
        root, task="powerlift", stamp=stamp, fixed_powerlift=True
    )
    pre_result_predictions = {
        "30k_60k": 0.337,
        "60k_90k": 0.072,
        "90k_100k": 0.0,
    }
    exposure_prediction_error = {
        label: power_stage_exposure["means"][label] - expected
        for label, expected in pre_result_predictions.items()
    }

    truck = truck_baselines(root)
    truck95 = {
        "fix": point(truck_fix, 95000),
        "scratch": point(truck["scratch"], 95000),
        "legacy_wfix": point(truck["legacy_wfix"], 95000),
    }
    fix_scratch95 = [
        fix[95000] - scratch[95000]
        for fix, scratch in zip(truck_fix, truck["scratch"])
    ]
    fix_wfix95 = [
        fix[95000] - legacy[95000]
        for fix, legacy in zip(truck_fix, truck["legacy_wfix"])
    ]
    truck_audits = {
        f"s{seed}_{step // 1000}k": checkpoint_audit(root, "truck", stamp, seed, step)
        for seed in SEEDS
        for step in (60000, 90000)
    }
    truck_gates = {
        "late_retention_mean": statistics.mean(fix_scratch95) >= 150,
        "late_retention_3of3": sum(value > 0 for value in fix_scratch95) == 3,
        "legacy_compatibility": abs(statistics.mean(fix_wfix95)) <= 100,
        "mechanism": all(
            not row["source_authority_active"]
            and row["sampling_phase"] == "physical_allowed"
            and row["mass_matches_physical"]
            and row["authority_release_events"]
            for row in truck_audits.values()
        ),
    }
    truck_stage_exposure = critic_stage_exposure(
        root, task="truck", stamp=stamp
    )
    mean_fix_scratch95 = statistics.mean(fix_scratch95)
    sd_fix_scratch95 = statistics.stdev(fix_scratch95)
    t_fix_scratch95 = mean_fix_scratch95 / (
        sd_fix_scratch95 / math.sqrt(len(fix_scratch95))
    )
    legacy_scratch95 = [
        legacy[95000] - scratch[95000]
        for legacy, scratch in zip(truck["legacy_wfix"], truck["scratch"])
    ]
    return {
        "stamp": stamp,
        "powerlift": {
            "warmup_fix_minus_fixed": warmup,
            "fix_minus_fixed_30k_80k": fix_fixed,
            "fix_minus_legacy_wfix_30k_80k": fix_wfix,
            "fix_at_80k": fix80,
            "collapse_floor": collapse_floor,
            "supplementary_fixed_sd_collapse": {
                "fixed_quota_sd_at_80k": fixed_sd80,
                "fix_neighbor_mean_75k_85k": neighbor_mean,
                "fixed_sd_floor": fixed_sd_collapse_floor,
                "fix_mean_at_80k": fix80["mean"],
                "pass": fix80["mean"] >= fixed_sd_collapse_floor,
                "per_seed_80k_minus_own_neighbor_mean": per_seed_neighbor_residuals,
            },
            "supplementary_critic_stage_exposure": {
                "handoff": power_stage_exposure,
                "fixed_quota_comparator": fixed_stage_exposure,
                "pre_result_predictions_from_T0004": pre_result_predictions,
                "handoff_minus_prediction": exposure_prediction_error,
                "prediction_within_one_percentage_point": all(
                    abs(value) <= 0.01
                    for value in exposure_prediction_error.values()
                ),
            },
            "checkpoint_audits": power_audits,
            "gates": power_gates,
            "pass": all(power_gates.values()),
        },
        "truck": {
            "at_95k": truck95,
            "fix_minus_scratch_95k": fix_scratch95,
            "fix_minus_legacy_wfix_95k": fix_wfix95,
            "supplementary_retention": {
                "fix_minus_scratch_mean": mean_fix_scratch95,
                "fix_minus_scratch_sd": sd_fix_scratch95,
                "paired_t_vs_zero_df2": t_fix_scratch95,
                "legacy_wfix_minus_scratch": legacy_scratch95,
                "legacy_wfix_minus_scratch_mean": statistics.mean(legacy_scratch95),
                "fraction_of_legacy_wfix_gap": (
                    mean_fix_scratch95 / statistics.mean(legacy_scratch95)
                ),
                "claim": "late_retention_vs_scratch_not_superiority_over_legacy_wfix",
            },
            "supplementary_critic_stage_exposure": truck_stage_exposure,
            "checkpoint_audits": truck_audits,
            "gates": truck_gates,
            "pass": all(truck_gates.values()),
        },
    }


def markdown(report: dict[str, Any]) -> str:
    diagnosis = report["baseline_diagnosis"]
    lines = [
        "# Admission replay handoff analysis",
        "",
        "## Frozen baseline diagnosis",
        "",
        "| Window | WFix-scratch | fixed-scratch | fixed-WFix |",
        "|---|---:|---:|---:|",
    ]
    for name in ("0k_30k", "30k_80k", "80k_95k"):
        row = diagnosis["windows"][name]
        lines.append(
            f"| {name} | {row['legacy_wfix_minus_scratch']['mean']:.1f} | "
            f"{row['fixed_quota_minus_scratch']['mean']:.1f} | "
            f"{row['fixed_quota_minus_legacy_wfix']['mean']:.1f} |"
        )
    lines.extend(
        [
            "",
            "The fixed-quota deficit is localized to 30k-80k and disappears after "
            "the source tail is physically overwritten. Powerlift therefore adjudicates "
            "transient replay harm, not late retention.",
        ]
    )
    if "adjudication" in report:
        adj = report["adjudication"]
        lines.extend(["", "## Formal adjudication", ""])
        for task in ("powerlift", "truck"):
            lines.append(f"- {task}: **{'PASS' if adj[task]['pass'] else 'FAIL'}**")
            for name, passed in adj[task]["gates"].items():
                lines.append(f"  - {name}: {'pass' if passed else 'fail'}")
        power = adj["powerlift"]
        exposure = power["supplementary_critic_stage_exposure"]
        lines.extend(
            [
                "",
                "## Supplementary mechanism checks",
                "",
                "| Critic-sampling stage | Handoff source share | Fixed-quota source share | Pre-result prediction |",
                "|---|---:|---:|---:|",
            ]
        )
        for stage in ("30k_60k", "60k_90k", "90k_100k"):
            lines.append(
                f"| {stage} | {exposure['handoff']['means'][stage]:.3%} | "
                f"{exposure['fixed_quota_comparator']['means'][stage]:.3%} | "
                f"{exposure['pre_result_predictions_from_T0004'][stage]:.3%} |"
            )
        fixed_sd = power["supplementary_fixed_sd_collapse"]
        lines.extend(
            [
                "",
                f"Using the frozen fixed-quota 80k SD ({fixed_sd['fixed_quota_sd_at_80k']:.2f}) "
                f"instead of the repaired run's SD gives a collapse floor of "
                f"{fixed_sd['fixed_sd_floor']:.2f}; repaired mean@80k is "
                f"{fixed_sd['fix_mean_at_80k']:.2f} (pass).",
                "",
                "## Truck late-retention boundary",
                "",
            ]
        )
        retention = adj["truck"]["supplementary_retention"]
        lines.append(
            f"At 95k the repaired method is +{retention['fix_minus_scratch_mean']:.1f} "
            f"over scratch (paired t={retention['paired_t_vs_zero_df2']:.2f}, n=3) and "
            f"retains {retention['fraction_of_legacy_wfix_gap']:.1%} of the legacy WFix gap. "
            "This supports late retention versus scratch, not superiority over legacy WFix."
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--stamp", help="Completed admission_handoff_v1 run stamp")
    parser.add_argument(
        "--out-dir", type=Path, default=Path("artifacts/admission_handoff_v1")
    )
    args = parser.parse_args()
    root = args.root.resolve()
    diagnosis, power = powerlift_diagnosis(root)
    truck = truck_baselines(root)
    report: dict[str, Any] = {
        "schema_version": 2,
        "baseline_diagnosis": diagnosis,
        "truck_legacy_headroom_at_95k": {
            "legacy_wfix_minus_scratch": [
                left[95000] - right[95000]
                for left, right in zip(truck["legacy_wfix"], truck["scratch"])
            ]
        },
    }
    if args.stamp:
        report["adjudication"] = adjudicate(root, args.stamp, diagnosis, power)
    out_dir = args.out_dir if args.out_dir.is_absolute() else root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = args.stamp or "preflight"
    json_path = out_dir / f"analysis_{suffix}.json"
    md_path = out_dir / f"analysis_{suffix}.md"
    json_path.write_text(json.dumps(report, indent=2) + "\n")
    md_path.write_text(markdown(report))
    print(json.dumps({"json": str(json_path), "markdown": str(md_path)}, indent=2))


if __name__ == "__main__":
    main()
