"""Audit and aggregate breadth batch-2 runs from local W&B streams.

This analysis intentionally avoids the W&B network API.  It discovers run
streams through the training logs, validates the expected 5k--95k evaluation
grid, and computes the same normalized trapezoidal AUC used by the earlier
project analyses.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any

try:
    from analyze_warmup_source_dose import read_wandb_history
except ModuleNotFoundError:  # Allows import as scripts.analyze_breadth_batch2_local.
    from scripts.analyze_warmup_source_dose import read_wandb_history


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG_DIRS = (
    ROOT / "logs/train/b2_20260705T153732Z",
    ROOT / "logs/train/b2s_20260705T224905Z",
    ROOT / "logs/train/b2_recovery_20260712T103302Z",
)
TASKS = ("bookshelf_simple", "basketball", "window", "powerlift", "balance_hard")
METHODS = ("scr", "rand", "wfix", "obrw")
EXPECTED_STEPS = tuple(range(5_000, 95_001, 5_000))
NAME_RE = re.compile(
    r"exp_name='h1hand_(bookshelf_simple|basketball|window|powerlift|balance_hard)"
    r"_b2_(scr|rand|wfix|obrw)_s([123])_20260705T153732Z'"
)
RUN_DIR_RE = re.compile(
    r"Run data is saved locally in (/home/yjj/fasttd3_ptf/wandb/run-[^\s]+)"
)


def _auc(points: list[tuple[int, float]]) -> float:
    area = sum(
        (step_1 - step_0) * (value_0 + value_1) / 2.0
        for (step_0, value_0), (step_1, value_1) in zip(points, points[1:])
    )
    return area / (points[-1][0] - points[0][0])


def _stats(values: list[float]) -> dict[str, Any]:
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "sample_sd": statistics.stdev(values) if len(values) > 1 else None,
        "values": values,
    }


def _paired_t(deltas: list[float]) -> float | None:
    if len(deltas) < 2:
        return None
    sd = statistics.stdev(deltas)
    if sd == 0:
        return math.copysign(math.inf, statistics.fmean(deltas))
    return statistics.fmean(deltas) / (sd / math.sqrt(len(deltas)))


def analyze(log_dirs: tuple[Path, ...]) -> dict[str, Any]:
    runs: list[dict[str, Any]] = []
    auc_by_key: dict[tuple[str, str, int], float] = {}
    attempts_by_key: dict[tuple[str, str, int], list[dict[str, Any]]] = {}

    for log_dir in log_dirs:
        for log_path in sorted(log_dir.glob("*.log")):
            text = log_path.read_text(errors="replace")
            name_match = NAME_RE.search(text)
            if name_match is None:
                continue
            task, method, seed_text = name_match.groups()
            seed = int(seed_text)
            run_match = RUN_DIR_RE.search(text)
            status = (
                "success"
                if "Saved official PTF parameters and configuration" in text
                else "failed"
            )
            record: dict[str, Any] = {
                "task": task,
                "method": method,
                "seed": seed,
                "log_path": str(log_path.relative_to(ROOT)),
                "status": status,
            }
            key = (task, method, seed)
            if run_match is None:
                record["quality_errors"] = ["missing local W&B run directory in log"]
                runs.append(record)
                attempts_by_key.setdefault(key, []).append(record)
                continue

            run_dir = Path(run_match.group(1))
            run_id = run_dir.name.rsplit("-", 1)[-1]
            wandb_path = run_dir / f"run-{run_id}.wandb"
            record.update(
                {
                    "wandb_run_id": run_id,
                    "wandb_path": str(wandb_path.relative_to(ROOT)),
                }
            )
            if status != "success":
                record["quality_errors"] = ["training did not reach final checkpoint"]
                runs.append(record)
                attempts_by_key.setdefault(key, []).append(record)
                continue

            rows = read_wandb_history(wandb_path)
            points = sorted(
                (int(row["_step"]), float(row["eval_avg_return"]))
                for row in rows
                if "eval_avg_return" in row and int(row["_step"]) <= 95_000
            )
            steps = tuple(step for step, _ in points)
            quality_errors = []
            if steps != EXPECTED_STEPS:
                quality_errors.append(
                    f"evaluation grid mismatch: expected {list(EXPECTED_STEPS)}, got {list(steps)}"
                )
            if len({step for step, _ in points}) != len(points):
                quality_errors.append("duplicate evaluation step")
            record.update(
                {
                    "history_rows": len(rows),
                    "evaluation_points": len(points),
                    "evaluation_steps": list(steps),
                    "quality_errors": quality_errors,
                }
            )
            if not quality_errors:
                record["auc_5k_95k"] = _auc(points)
                if key in auc_by_key:
                    raise ValueError(f"multiple valid attempts for logical run slot {key}")
                auc_by_key[key] = record["auc_5k_95k"]
            runs.append(record)
            attempts_by_key.setdefault(key, []).append(record)

    aggregates: dict[str, Any] = {}
    comparisons: dict[str, Any] = {}
    for task in TASKS:
        aggregates[task] = {}
        for method in METHODS:
            values = [
                auc_by_key[(task, method, seed)]
                for seed in (1, 2, 3)
                if (task, method, seed) in auc_by_key
            ]
            if values:
                aggregates[task][method] = _stats(values)

        comparisons[task] = {}
        for left, right in (
            ("wfix", "scr"),
            ("rand", "scr"),
            ("wfix", "rand"),
            ("obrw", "scr"),
            ("obrw", "wfix"),
        ):
            seeds = [
                seed
                for seed in (1, 2, 3)
                if (task, left, seed) in auc_by_key and (task, right, seed) in auc_by_key
            ]
            if not seeds:
                continue
            deltas = [
                auc_by_key[(task, left, seed)] - auc_by_key[(task, right, seed)]
                for seed in seeds
            ]
            comparisons[task][f"{left}_minus_{right}"] = {
                "paired_seeds": seeds,
                "n": len(seeds),
                "mean_delta": statistics.fmean(deltas),
                "paired_t": _paired_t(deltas),
                "deltas": deltas,
            }

    successful_attempts = sum(run["status"] == "success" for run in runs)
    valid_attempts = sum("auc_5k_95k" in run for run in runs)
    recovered_slots = [
        {
            "task": key[0],
            "method": key[1],
            "seed": key[2],
            "attempts": len(attempts),
        }
        for key, attempts in sorted(attempts_by_key.items())
        if any(attempt["status"] == "failed" for attempt in attempts)
        and any("auc_5k_95k" in attempt for attempt in attempts)
    ]
    return {
        "schema_version": 1,
        "analysis": "breadth_batch2_local_5k_95k_auc",
        "metric": {
            "key": "eval_avg_return",
            "window": [5_000, 95_000],
            "evaluation_interval": 5_000,
            "estimator": "trapezoidal_integral_divided_by_step_span",
        },
        "quality_summary": {
            "expected_run_slots": 48,
            "discovered_run_slots": len(attempts_by_key),
            "discovered_attempts": len(runs),
            "successful_attempts": successful_attempts,
            "failed_attempts": len(runs) - successful_attempts,
            "valid_auc_attempts": valid_attempts,
            "valid_auc_run_slots": len(auc_by_key),
            "recovered_run_slots": recovered_slots,
            "complete_three_seed_tasks_for_scr_rand_wfix": [
                task
                for task in TASKS
                if all(
                    (task, method, seed) in auc_by_key
                    for method in ("scr", "rand", "wfix")
                    for seed in (1, 2, 3)
                )
            ],
        },
        "runs": runs,
        "aggregates": aggregates,
        "paired_comparisons": comparisons,
    }


def render_markdown(result: dict[str, Any]) -> str:
    quality = result["quality_summary"]
    lines = [
        "# Breadth batch-2 local audit",
        "",
        "Metric: normalized trapezoidal AUC of `eval_avg_return` on the exact 5k--95k grid.",
        "All curves are read from local W&B binary streams; the network API is not used.",
        "",
        "## Data quality",
        "",
        f"- Run slots: {quality['discovered_run_slots']}/{quality['expected_run_slots']} discovered.",
        f"- Attempts: {quality['discovered_attempts']} total; "
        f"{quality['successful_attempts']} successful and {quality['failed_attempts']} failed.",
        f"- Valid logical run slots: {quality['valid_auc_run_slots']}/{quality['expected_run_slots']}.",
        "- Complete 3-seed tasks for scratch/rand/WFix: "
        + ", ".join(quality["complete_three_seed_tasks_for_scr_rand_wfix"])
        + ".",
        "- `balance_hard` rand-s3 and WFix-s2 first failed before training due CUDA OOM; "
        "both logical slots were recovered on 2026-07-12 with unchanged configurations.",
        "",
        "## Aggregate AUC",
        "",
        "| task | scratch | rand | WFix | OBRW | WFix-scratch | OBRW-scratch |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    def cell(task: str, method: str) -> str:
        value = result["aggregates"][task].get(method)
        if value is None:
            return "--"
        sd = value["sample_sd"]
        return f"{value['mean']:.1f}±{sd:.1f} (n={value['n']})" if sd is not None else f"{value['mean']:.1f} (n=1)"

    def delta(task: str, comparison: str) -> str:
        value = result["paired_comparisons"][task].get(comparison)
        if value is None:
            return "--"
        t_value = value["paired_t"]
        t_text = "NA" if t_value is None else f"{t_value:+.2f}"
        return f"{value['mean_delta']:+.1f} (t={t_text}, n={value['n']})"

    for task in TASKS:
        lines.append(
            f"| {task} | {cell(task, 'scr')} | {cell(task, 'rand')} | "
            f"{cell(task, 'wfix')} | {cell(task, 'obrw')} | "
            f"{delta(task, 'wfix_minus_scr')} | {delta(task, 'obrw_minus_scr')} |"
        )
    lines.extend(
        [
            "",
            "Sample SD is reported across available seeds. Paired comparisons use matching seed IDs;",
            "historical failed attempts remain in the audit trail but do not replace recovered valid slots.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "artifacts/breadth_batch2_local_audit",
    )
    args = parser.parse_args()
    result = analyze(DEFAULT_LOG_DIRS)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "analysis.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    )
    (args.output_dir / "report.md").write_text(render_markdown(result))
    print(render_markdown(result), end="")


if __name__ == "__main__":
    main()
