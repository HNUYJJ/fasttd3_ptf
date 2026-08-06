#!/usr/bin/env python3
"""Backfill locally recorded admission-core-v1 metrics into Weights & Biases.

The formal runs disabled W&B, so this script only uploads values that can be
recovered exactly from stdout or frozen checkpoints.  It deliberately does not
invent optimizer losses/rewards that were never persisted.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EVAL_RE = re.compile(
    r"\[eval\]\s*step=(\d+)\s+return=([-+0-9.eE]+)\s+length=([-+0-9.eE]+)"
)
SPEED_RE = re.compile(
    r"([0-9]+(?:\.[0-9]+)?) sps, :[^\r\n]*?\|\s*(\d+)/(\d+)"
)
ARGS_RE = re.compile(r"^[A-Za-z_]\w*Args\((.*)\)$", re.MULTILINE)
PTF_RE = re.compile(r"^(\{'ptf': .*\})$", re.MULTILINE)

RECOVERED_METRICS = [
    "speed",
    "frame",
    "critic_lr",
    "actor_lr",
    "eval_avg_return",
    "eval_avg_length",
    "admission checkpoint counters/shares at 30k, 60k, 90k, and 100k",
]
UNRECOVERABLE_METRICS = [
    "actor_loss",
    "qf_loss",
    "qf_max",
    "qf_min",
    "actor_grad_norm",
    "critic_grad_norm",
    "env_rewards",
    "buffer_rewards",
    "per-update MCG tensors not printed to stdout",
]


@dataclass(frozen=True)
class LocalRun:
    log_path: Path
    meta_path: Path
    metadata: dict[str, str]
    args: dict[str, Any]
    ptf: dict[str, Any]
    history: dict[int, dict[str, float]]
    checkpoint_paths: list[Path]

    @property
    def run_name(self) -> str:
        return f"{self.args['env_name']}__{self.args['exp_name']}__{self.args['seed']}"

    @property
    def run_id(self) -> str:
        digest = hashlib.sha256(self.run_name.encode()).hexdigest()[:12]
        return f"acv1bf-{digest}"


def parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("logs/train/admission_core_v1_20260712TFINALV2Z"),
    )
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/admission_core_v1/wandb_backfill_manifest.json"),
    )
    parser.add_argument("--entity", default="yujiajie-nju")
    parser.add_argument("--project", default="fasttd3_ptf")
    parser.add_argument("--group", default="admission_core_v1_20260712TFINALV2Z_backfill")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip stable run IDs that already exist instead of failing.",
    )
    parser.add_argument(
        "--only-run-id",
        help="Process only this deterministic local run ID (useful for repair).",
    )
    parser.add_argument(
        "--run-id-suffix",
        default="",
        help="Append a suffix to the uploaded run ID (useful after deleting a defective run).",
    )
    return parser.parse_args()


def parse_metadata(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            result[key] = value
    required = {"started_at", "finished_at", "cell", "seed", "exp_name", "exit_code"}
    missing = required - result.keys()
    if missing:
        raise ValueError(f"{path}: missing metadata fields {sorted(missing)}")
    if result["exit_code"] != "0":
        raise ValueError(f"{path}: refusing to backfill failed run exit_code={result['exit_code']}")
    return result


def literal_call_kwargs(call_text: str) -> dict[str, Any]:
    node = ast.parse(f"f({call_text})", mode="eval").body
    if not isinstance(node, ast.Call):
        raise ValueError("training arguments are not a call expression")
    values: dict[str, Any] = {}
    for keyword in node.keywords:
        if keyword.arg is None:
            raise ValueError("unexpected **kwargs in training arguments")
        values[keyword.arg] = ast.literal_eval(keyword.value)
    return values


def add_metric(history: dict[int, dict[str, float]], step: int, key: str, value: Any) -> None:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"non-finite metric at step={step}: {key}={value}")
    history.setdefault(step, {})[key] = number


def parse_stdout(log_path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[int, dict[str, float]]]:
    text = log_path.read_text(errors="replace")
    args_match = ARGS_RE.search(text)
    ptf_match = PTF_RE.search(text)
    if not args_match or not ptf_match:
        raise ValueError(f"{log_path}: could not recover training configuration")
    args = literal_call_kwargs(args_match.group(1))
    ptf_wrapper = ast.literal_eval(ptf_match.group(1))
    ptf = ptf_wrapper["ptf"]
    total_timesteps = int(args["total_timesteps"])
    num_envs = int(args["num_envs"])

    history: dict[int, dict[str, float]] = {}
    # tqdm first renders the newly assigned description at the current global
    # step.  Restricting to exact 100-step displays avoids its repeated redraws.
    speed_by_step: dict[int, float] = {}
    for speed, displayed_step, displayed_total in SPEED_RE.findall(text):
        step = int(displayed_step)
        if int(displayed_total) != total_timesteps:
            continue
        if step % 100 == 0 and 100 <= step < total_timesteps:
            speed_by_step[step] = float(speed)
    expected_speed_steps = set(range(100, total_timesteps, 100))
    if speed_by_step.keys() != expected_speed_steps:
        missing = sorted(expected_speed_steps - speed_by_step.keys())
        extra = sorted(speed_by_step.keys() - expected_speed_steps)
        raise ValueError(
            f"{log_path}: incomplete speed trace, missing={missing[:10]} extra={extra[:10]}"
        )
    for step, speed in speed_by_step.items():
        add_metric(history, step, "speed", speed)
        add_metric(history, step, "frame", step * num_envs)
        # Both schedules were configured with identical start/end values in
        # these frozen runs, so the LR trace is exactly constant.
        if args["critic_learning_rate"] != args["critic_learning_rate_end"]:
            raise ValueError(f"{log_path}: non-constant critic LR cannot be reconstructed safely")
        if args["actor_learning_rate"] != args["actor_learning_rate_end"]:
            raise ValueError(f"{log_path}: non-constant actor LR cannot be reconstructed safely")
        add_metric(history, step, "critic_lr", args["critic_learning_rate"])
        add_metric(history, step, "actor_lr", args["actor_learning_rate"])

    eval_points = EVAL_RE.findall(text)
    expected_eval_steps = set(range(int(args["eval_interval"]), total_timesteps, int(args["eval_interval"])))
    observed_eval_steps = {int(step) for step, _, _ in eval_points}
    if observed_eval_steps != expected_eval_steps:
        raise ValueError(
            f"{log_path}: incomplete eval trace, observed={sorted(observed_eval_steps)} "
            f"expected={sorted(expected_eval_steps)}"
        )
    for step, avg_return, avg_length in eval_points:
        add_metric(history, int(step), "eval_avg_return", avg_return)
        add_metric(history, int(step), "eval_avg_length", avg_length)
    return args, ptf, history


def load_checkpoint_metrics(
    checkpoint_path: Path, history: dict[int, dict[str, float]]
) -> None:
    import torch

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    step = int(checkpoint["global_step"])
    audit = checkpoint.get("admission_audit")
    if not audit:
        raise ValueError(f"{checkpoint_path}: missing admission_audit")
    # Saved source_names includes the policy module's terminal "null" option;
    # admission_audit represents that same slot explicitly as the student.
    saved_source_names = list(checkpoint["source_names"])
    teacher_count = len(audit["admitted_sources"])
    source_names = saved_source_names[:teacher_count]
    candidate_names = source_names + ["student"]
    if len(candidate_names) != len(audit["candidate_masses"]):
        raise ValueError(f"{checkpoint_path}: source/admission array length mismatch")

    vector_metrics = {
        "candidate_mass": audit["candidate_masses"],
        "main_buffer_count": audit["main_buffer_counts"],
        "active_buffer_count": audit["active_buffer_counts"],
        "critic_sample_count": audit["critic_sample_counts"],
        "actor_independent_sample_count": audit["actor_independent_sample_counts"],
        "execution_count": audit["execution_counts"],
    }
    for metric, values in vector_metrics.items():
        if len(values) != len(candidate_names):
            raise ValueError(f"{checkpoint_path}: {metric} length mismatch")
        for name, value in zip(candidate_names, values):
            add_metric(history, step, f"admission/{metric}/{name}", value)

    share_bases = {
        "main_buffer_share": audit["main_buffer_counts"],
        "active_buffer_share": audit["active_buffer_counts"],
        "critic_sample_share": audit["critic_sample_counts"],
        "execution_share": audit["execution_counts"],
    }
    for metric, values in share_bases.items():
        total = float(sum(values))
        for name, value in zip(candidate_names, values):
            add_metric(history, step, f"admission/{metric}/{name}", value / total if total else 0.0)
    for name, admitted in zip(source_names, audit["admitted_sources"]):
        add_metric(history, step, f"admission/admitted/{name}", int(bool(admitted)))
    add_metric(history, step, "admission/exact_abstain", int(not any(audit["admitted_sources"])))


def discover_runs(log_dir: Path, model_dir: Path) -> list[LocalRun]:
    runs: list[LocalRun] = []
    for log_path in sorted(log_dir.glob("*_s[123].log")):
        meta_path = log_path.with_suffix(".meta.txt")
        if not meta_path.exists():
            raise FileNotFoundError(meta_path)
        metadata = parse_metadata(meta_path)
        args, ptf, history = parse_stdout(log_path)
        if metadata["seed"] != str(args["seed"]) or metadata["exp_name"] != args["exp_name"]:
            raise ValueError(f"{log_path}: stdout/metadata identity mismatch")
        prefix = f"{args['env_name']}__{args['exp_name']}__{args['seed']}"
        checkpoint_paths = [
            model_dir / f"{prefix}_{step}.pt" for step in (30000, 60000, 90000)
        ] + [model_dir / f"{prefix}_final.pt"]
        missing = [str(path) for path in checkpoint_paths if not path.exists()]
        if missing:
            raise FileNotFoundError(f"{log_path}: missing checkpoints {missing}")
        for checkpoint_path in checkpoint_paths:
            load_checkpoint_metrics(checkpoint_path, history)
        runs.append(
            LocalRun(
                log_path=log_path,
                meta_path=meta_path,
                metadata=metadata,
                args=args,
                ptf=ptf,
                history=history,
                checkpoint_paths=checkpoint_paths,
            )
        )
    if len(runs) != 6:
        raise ValueError(f"expected six formal runs, found {len(runs)} in {log_dir}")
    return runs


def iso_timestamp(value: str) -> float:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def event_timestamp(local_run: LocalRun, step: int) -> float:
    start = iso_timestamp(local_run.metadata["started_at"])
    finish = iso_timestamp(local_run.metadata["finished_at"])
    total = int(local_run.args["total_timesteps"])
    # Preserve original wall-clock ordering. This timestamp is interpolated,
    # while all scientific metric values and steps remain exact.
    return start + (finish - start) * min(max(step / total, 0.0), 1.0)


def manifest_entry(local_run: LocalRun) -> dict[str, Any]:
    eval_steps = [step for step, values in local_run.history.items() if "eval_avg_return" in values]
    speed_steps = [step for step, values in local_run.history.items() if "speed" in values]
    return {
        "run_id": local_run.run_id,
        "run_name": local_run.run_name,
        "cell": local_run.metadata["cell"],
        "seed": int(local_run.metadata["seed"]),
        "log_path": str(local_run.log_path),
        "meta_path": str(local_run.meta_path),
        "checkpoint_paths": [str(path) for path in local_run.checkpoint_paths],
        "history_rows": len(local_run.history),
        "speed_points": len(speed_steps),
        "eval_points": len(eval_steps),
        "eval_steps": sorted(eval_steps),
        "started_at_original": local_run.metadata["started_at"],
        "finished_at_original": local_run.metadata["finished_at"],
        "recovered_metrics": RECOVERED_METRICS,
        "unrecoverable_metrics": UNRECOVERABLE_METRICS,
    }


def upload_run(
    local_run: LocalRun,
    *,
    entity: str,
    project: str,
    group: str,
    skip_existing: bool,
    run_id: str | None = None,
) -> tuple[str, str]:
    import wandb

    effective_run_id = run_id or local_run.run_id
    api = wandb.Api(timeout=60)
    try:
        existing = api.run(f"{entity}/{project}/{effective_run_id}")
    except Exception as exc:
        # W&B raises CommError for a missing run; avoid binding to a private
        # exception class that has moved between client versions.
        if "Could not find run" not in str(exc) and "not found" not in str(exc).lower():
            raise
        existing = None
    if existing is not None:
        if skip_existing:
            return existing.url, "skipped_existing"
        raise RuntimeError(
            f"stable W&B run already exists: {existing.url}; pass --skip-existing after auditing it"
        )

    config = {
        **local_run.args,
        "ptf": local_run.ptf,
        "backfill": {
            "is_backfilled": True,
            "source": "local stdout plus frozen admission checkpoints",
            "scientific_values_inferred": False,
            "event_timestamp_interpolated": True,
            "original_use_wandb": False,
            "original_started_at": local_run.metadata["started_at"],
            "original_finished_at": local_run.metadata["finished_at"],
            "recovered_metrics": RECOVERED_METRICS,
            "unrecoverable_metrics": UNRECOVERABLE_METRICS,
            "metadata": local_run.metadata,
        },
    }
    run = wandb.init(
        entity=entity,
        project=project,
        id=effective_run_id,
        name=local_run.run_name,
        group=group,
        job_type="metrics-backfill",
        tags=["backfilled", "admission-core-v1", local_run.metadata["cell"]],
        notes=(
            "Backfilled from the frozen local stdout/checkpoints after the original run "
            "was launched with W&B disabled. Missing optimizer/reward metrics were not fabricated."
        ),
        config=config,
        resume="never",
        settings=wandb.Settings(init_timeout=120),
    )
    run.define_metric("global_step")
    run.define_metric("*", step_metric="global_step")
    for step in sorted(local_run.history):
        payload: dict[str, Any] = {
            "global_step": step,
            "_timestamp": event_timestamp(local_run, step),
            **local_run.history[step],
        }
        run.log(payload, step=step)

    eval_rows = [
        (step, values["eval_avg_return"], values["eval_avg_length"])
        for step, values in sorted(local_run.history.items())
        if "eval_avg_return" in values
    ]
    run.summary["backfill/is_backfilled"] = True
    run.summary["backfill/recovered_speed_points"] = sum(
        "speed" in values for values in local_run.history.values()
    )
    run.summary["backfill/recovered_eval_points"] = len(eval_rows)
    run.summary["backfill/unrecoverable_metric_count"] = len(UNRECOVERABLE_METRICS)
    run.summary["eval/best_return"] = max(row[1] for row in eval_rows)
    run.summary["eval/last_observed_return"] = eval_rows[-1][1]
    run.summary["eval/last_observed_step"] = eval_rows[-1][0]

    artifact = wandb.Artifact(
        name=f"{effective_run_id}-local-evidence",
        type="training-log-evidence",
        metadata={
            "run_name": local_run.run_name,
            "original_git_head": local_run.metadata.get("git_head"),
            "log_sha256": hashlib.sha256(local_run.log_path.read_bytes()).hexdigest(),
            "meta_sha256": hashlib.sha256(local_run.meta_path.read_bytes()).hexdigest(),
        },
    )
    artifact.add_file(str(local_run.log_path), name=local_run.log_path.name)
    artifact.add_file(str(local_run.meta_path), name=local_run.meta_path.name)
    run.log_artifact(artifact)
    url = run.url
    run.finish()
    return url, "uploaded"


def main() -> None:
    cli = parse_cli()
    runs = discover_runs(cli.log_dir, cli.model_dir)
    if cli.only_run_id:
        runs = [run for run in runs if run.run_id == cli.only_run_id]
        if not runs:
            raise ValueError(f"unknown --only-run-id={cli.only_run_id}")
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "entity": cli.entity,
        "project": cli.project,
        "group": cli.group,
        "dry_run": cli.dry_run,
        "policy": {
            "scientific_values_inferred": False,
            "event_timestamp_interpolated": True,
            "recovered_metrics": RECOVERED_METRICS,
            "unrecoverable_metrics": UNRECOVERABLE_METRICS,
        },
        "runs": [],
    }
    for local_run in runs:
        entry = manifest_entry(local_run)
        effective_run_id = local_run.run_id + cli.run_id_suffix
        if effective_run_id != local_run.run_id:
            entry["source_run_id"] = local_run.run_id
            entry["run_id"] = effective_run_id
        if cli.dry_run:
            entry["upload_status"] = "dry_run_validated"
        else:
            url, status = upload_run(
                local_run,
                entity=cli.entity,
                project=cli.project,
                group=cli.group,
                skip_existing=cli.skip_existing,
                run_id=effective_run_id,
            )
            entry["wandb_url"] = url
            entry["upload_status"] = status
        manifest["runs"].append(entry)
        print(json.dumps(entry, ensure_ascii=False))
    cli.output.parent.mkdir(parents=True, exist_ok=True)
    cli.output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {cli.output}")


if __name__ == "__main__":
    main()
