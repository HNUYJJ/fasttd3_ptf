"""Audit the realized MCG source dose from local W&B history streams.

The logged ``mcg/exec_*`` values are cross-sectional fractions of vectorized
environments at a logging step.  Consequently this tool reports an unweighted
mean over retained W&B history cross-sections.  It does *not* reconstruct exact
behavior-segment starts, segment counts, or replay-transition counts.

Only history records in the local ``.wandb`` stream are used for dose values.
The preregistered warmup filter is strict: ``0 <= _step < 30000``.

Example:
  python scripts/analyze_warmup_source_dose.py \
    --spec configs/experiments/cabinet_p2_warmup_source_dose.json \
    --json-out logs/probe/cabinet_p2_warmup_source_dose.json \
    --md-out docs/cabinet_p2_warmup_source_dose.md
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any, Callable, Iterable


ROOT = Path(__file__).resolve().parents[1]
WARMUP_STEP_MIN = 0
WARMUP_STEP_MAX_EXCLUSIVE = 30_000
TEACHER_KEY = "mcg/exec_env_frac"
TEACHER_PART_KEY = "mcg/exec_part_frac"
SOURCE_KEY_PREFIX = "mcg/exec_share_src"


def load_spec(path: str | Path) -> dict[str, Any]:
    spec = json.loads(Path(path).read_text())
    if int(spec.get("schema_version", 0)) != 1:
        raise ValueError("expected schema_version=1")

    step_filter = spec.get("warmup_step_filter", {})
    if (
        int(step_filter.get("min_inclusive", -1)) != WARMUP_STEP_MIN
        or int(step_filter.get("max_exclusive", -1)) != WARMUP_STEP_MAX_EXCLUSIVE
    ):
        raise ValueError("warmup_step_filter must be exactly 0 <= _step < 30000")

    expected_seeds = [int(seed) for seed in spec.get("expected_seeds", [])]
    if not expected_seeds or len(expected_seeds) != len(set(expected_seeds)):
        raise ValueError("expected_seeds must be a non-empty unique list")

    condition_names: set[str] = set()
    for condition in spec.get("conditions", []):
        name = str(condition.get("name", ""))
        if not name or name in condition_names:
            raise ValueError(f"invalid or duplicate condition name: {name!r}")
        condition_names.add(name)

        source_names = [str(source) for source in condition.get("source_names", [])]
        if not source_names or len(source_names) != len(set(source_names)):
            raise ValueError(f"{name}: source_names must be a non-empty unique list")

        runs = condition.get("runs", [])
        run_seeds = [int(run["seed"]) for run in runs]
        if sorted(run_seeds) != sorted(expected_seeds):
            raise ValueError(
                f"{name}: run seeds {sorted(run_seeds)} do not match "
                f"expected_seeds {sorted(expected_seeds)}"
            )
        for run in runs:
            wandb_path = Path(str(run.get("wandb_path", "")))
            if wandb_path.suffix != ".wandb":
                raise ValueError(f"{name} seed {run['seed']}: wandb_path must end in .wandb")

    if not condition_names:
        raise ValueError("spec contains no conditions")
    return spec


def _history_item_value(item: Any) -> Any:
    try:
        return json.loads(item.value_json)
    except (json.JSONDecodeError, TypeError):
        return item.value_json


def read_wandb_history(path: str | Path) -> list[dict[str, Any]]:
    """Read history records from one local W&B binary stream.

    Run config, summary, output logs, and the W&B network/API are intentionally
    not consulted.  ``DataStore`` and the protobuf type are the readers shipped
    with the installed W&B SDK that produced these local streams.
    """
    path = Path(path)
    if path.suffix != ".wandb":
        raise ValueError(f"expected a .wandb stream, got: {path}")
    if not path.is_file():
        raise FileNotFoundError(path)

    try:
        from wandb.proto import wandb_internal_pb2
        from wandb.sdk.internal.datastore import DataStore
    except ImportError as exc:  # pragma: no cover - depends on runtime packaging
        raise RuntimeError("the installed wandb SDK is required to read .wandb history") from exc

    store = DataStore()
    rows: list[dict[str, Any]] = []
    store.open_for_scan(str(path))
    try:
        while True:
            data = store.scan_data()
            if data is None:
                break
            record = wandb_internal_pb2.Record()
            record.ParseFromString(data)
            if not record.HasField("history"):
                continue
            row: dict[str, Any] = {}
            for item in record.history.item:
                key = item.key or ".".join(item.nested_key)
                row[key] = _history_item_value(item)
            rows.append(row)
    finally:
        store.close()
    return rows


def _finite_fraction(row: dict[str, Any], key: str, step: int, tolerance: float) -> float:
    if key not in row:
        raise ValueError(f"history step {step}: missing {key!r}")
    value = row[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"history step {step}: {key!r} is not numeric")
    value = float(value)
    if not math.isfinite(value) or value < -tolerance or value > 1.0 + tolerance:
        raise ValueError(f"history step {step}: {key!r}={value!r} is not a fraction")
    return value


def analyze_history_rows(
    rows: Iterable[dict[str, Any]],
    source_names: list[str],
    *,
    share_tolerance: float = 1e-8,
) -> dict[str, Any]:
    """Summarize one run after the strict warmup history filter."""
    if share_tolerance < 0 or not math.isfinite(share_tolerance):
        raise ValueError("share_tolerance must be finite and non-negative")
    if not source_names or len(source_names) != len(set(source_names)):
        raise ValueError("source_names must be a non-empty unique list")

    selected: list[tuple[int, dict[str, Any]]] = []
    for row in rows:
        if "_step" not in row:
            continue
        raw_step = row["_step"]
        if isinstance(raw_step, bool) or not isinstance(raw_step, (int, float)):
            raise ValueError(f"history _step is not numeric: {raw_step!r}")
        step_float = float(raw_step)
        if not math.isfinite(step_float) or not step_float.is_integer():
            raise ValueError(f"history _step is not a finite integer: {raw_step!r}")
        step = int(step_float)
        if WARMUP_STEP_MIN <= step < WARMUP_STEP_MAX_EXCLUSIVE:
            selected.append((step, row))

    if not selected:
        raise ValueError("no W&B history cross-sections satisfy 0 <= _step < 30000")
    steps = [step for step, _ in selected]
    if len(steps) != len(set(steps)):
        raise ValueError("duplicate retained W&B history _step values would overweight a cross-section")

    source_values = {source: [] for source in source_names}
    teacher_values: list[float] = []
    student_values: list[float] = []
    source_teacher_errors: list[float] = []
    env_part_errors: list[float] = []

    for step, row in selected:
        teacher = _finite_fraction(row, TEACHER_KEY, step, share_tolerance)
        teacher_part = _finite_fraction(row, TEACHER_PART_KEY, step, share_tolerance)
        shares = []
        for index, source in enumerate(source_names):
            share = _finite_fraction(
                row, f"{SOURCE_KEY_PREFIX}{index}", step, share_tolerance
            )
            source_values[source].append(share)
            shares.append(share)

        source_teacher_error = abs(sum(shares) - teacher)
        env_part_error = abs(teacher_part - teacher)
        if source_teacher_error > share_tolerance:
            raise ValueError(
                f"history step {step}: source shares sum to {sum(shares):.12g}, "
                f"teacher share is {teacher:.12g}, error {source_teacher_error:.3g} "
                f"> tolerance {share_tolerance:.3g}"
            )
        if env_part_error > share_tolerance:
            raise ValueError(
                f"history step {step}: exec_env_frac and exec_part_frac differ by "
                f"{env_part_error:.3g} > tolerance {share_tolerance:.3g}; "
                "the cabinet audit assumes group-synchronous source execution"
            )

        teacher_values.append(teacher)
        student_values.append(1.0 - teacher)
        source_teacher_errors.append(source_teacher_error)
        env_part_errors.append(env_part_error)

    return {
        "n_history_cross_sections": len(selected),
        "selected_step_min": min(steps),
        "selected_step_max": max(steps),
        "source_absolute_shares": {
            source: statistics.fmean(values) for source, values in source_values.items()
        },
        "teacher_share": statistics.fmean(teacher_values),
        "student_share": statistics.fmean(student_values),
        "validation": {
            "share_tolerance": share_tolerance,
            "max_abs_source_sum_minus_teacher": max(source_teacher_errors),
            "max_abs_exec_env_minus_exec_part": max(env_part_errors),
            "passed": True,
        },
    }


def _seed_stats(values: Iterable[float]) -> dict[str, Any]:
    clean = [float(value) for value in values]
    if not clean:
        raise ValueError("cannot aggregate an empty seed set")
    return {
        "n_seeds": len(clean),
        "mean": statistics.fmean(clean),
        "sd": statistics.stdev(clean) if len(clean) > 1 else None,
    }


def aggregate_runs(runs: list[dict[str, Any]], source_names: list[str]) -> dict[str, Any]:
    if not runs:
        raise ValueError("cannot aggregate an empty run list")
    return {
        "source_absolute_shares": {
            source: _seed_stats(run["source_absolute_shares"][source] for run in runs)
            for source in source_names
        },
        "teacher_share": _seed_stats(run["teacher_share"] for run in runs),
        "student_share": _seed_stats(run["student_share"] for run in runs),
    }


def analyze_spec(
    spec: dict[str, Any],
    *,
    root: str | Path = ROOT,
    history_reader: Callable[[str | Path], list[dict[str, Any]]] = read_wandb_history,
) -> dict[str, Any]:
    root = Path(root)
    tolerance = float(spec.get("share_validation_atol", 1e-8))
    conditions: dict[str, Any] = {}

    for condition in spec["conditions"]:
        name = str(condition["name"])
        source_names = [str(source) for source in condition["source_names"]]
        run_results = []
        for run in sorted(condition["runs"], key=lambda item: int(item["seed"])):
            configured_path = Path(str(run["wandb_path"]))
            resolved_path = configured_path if configured_path.is_absolute() else root / configured_path
            result = analyze_history_rows(
                history_reader(resolved_path),
                source_names,
                share_tolerance=tolerance,
            )
            run_results.append(
                {
                    "seed": int(run["seed"]),
                    "wandb_path": str(configured_path),
                    **result,
                }
            )
        conditions[name] = {
            "source_names": source_names,
            "runs": run_results,
            "across_seeds": aggregate_runs(run_results, source_names),
        }

    return {
        "schema_version": 1,
        "analysis": "warmup_source_dose_cross_sectional",
        "experiment": spec.get("experiment"),
        "task": spec.get("task"),
        "step_filter": {
            "min_inclusive": WARMUP_STEP_MIN,
            "max_exclusive": WARMUP_STEP_MAX_EXCLUSIVE,
            "expression": "0 <= _step < 30000",
        },
        "estimator": {
            "type": "unweighted_mean_of_wandb_history_cross_sections",
            "observation_unit": "logged cross-section of vectorized environment assignments",
            "limitations": (
                "periodic cross-sectional sampling; not exact behavior-segment, "
                "environment-step, transition, or replay-buffer counts"
            ),
        },
        "conditions": conditions,
    }


def _pct(value: float) -> str:
    return f"{100.0 * value:.3f}%"


def _mean_sd(stats: dict[str, Any]) -> str:
    sd = stats["sd"]
    return _pct(stats["mean"]) if sd is None else f"{_pct(stats['mean'])} ± {_pct(sd)}"


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Cabinet P2 warmup source-dose audit",
        "",
        "> Dose values are the unweighted mean of periodic W&B history cross-sections ",
        "> of vectorized environment assignments. They are not exact behavior-segment, ",
        "> environment-step, replay-transition, or replay-buffer counts.",
        "",
        "Strict history filter: `0 <= _step < 30000`. Source shares are absolute shares ",
        "over all sampled environment assignments, not source-conditional teacher weights.",
        "",
        "## Per-run cross-sectional estimates",
        "",
        "| condition | seed | retained samples | retained steps | source absolute shares | teacher | student | max source-sum error |",
        "|---|---:|---:|---:|---|---:|---:|---:|",
    ]
    for condition_name, condition in summary["conditions"].items():
        for run in condition["runs"]:
            sources = ", ".join(
                f"{source}={_pct(run['source_absolute_shares'][source])}"
                for source in condition["source_names"]
            )
            lines.append(
                f"| {condition_name} | {run['seed']} | {run['n_history_cross_sections']} "
                f"| {run['selected_step_min']}–{run['selected_step_max']} | {sources} "
                f"| {_pct(run['teacher_share'])} | {_pct(run['student_share'])} "
                f"| {run['validation']['max_abs_source_sum_minus_teacher']:.3g} |"
            )

    lines.extend(
        [
            "",
            "## Across-seed estimates",
            "",
            "Sample SD is computed across the per-run estimates, with one value per training seed.",
            "",
            "| condition | share | mean ± SD across seeds | n seeds |",
            "|---|---|---:|---:|",
        ]
    )
    for condition_name, condition in summary["conditions"].items():
        aggregate = condition["across_seeds"]
        for source in condition["source_names"]:
            stats = aggregate["source_absolute_shares"][source]
            lines.append(
                f"| {condition_name} | source `{source}` (absolute) | {_mean_sd(stats)} | {stats['n_seeds']} |"
            )
        for key, label in (("teacher_share", "teacher"), ("student_share", "student")):
            stats = aggregate[key]
            lines.append(
                f"| {condition_name} | {label} | {_mean_sd(stats)} | {stats['n_seeds']} |"
            )
    lines.extend(
        [
            "",
            "Every retained cross-section passed the configured check that the sum of ",
            "source absolute shares matches teacher share within tolerance.",
            "",
        ]
    )
    return "\n".join(lines)


def _write(path: str | Path, content: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--json-out", required=True)
    parser.add_argument("--md-out", required=True)
    parser.add_argument(
        "--root",
        default=str(ROOT),
        help="base directory for relative .wandb paths (default: repository root)",
    )
    args = parser.parse_args(argv)

    summary = analyze_spec(load_spec(args.spec), root=args.root)
    _write(args.json_out, json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    _write(args.md_out, render_markdown(summary))
    print(f"wrote {args.json_out}")
    print(f"wrote {args.md_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
