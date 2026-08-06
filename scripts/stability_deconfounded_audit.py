"""Stability-deconfounded transfer audit.

This tool evaluates target-policy checkpoints on identical environment seeds and
stores episode-level metric traces.  Its summary compares every condition with
scratch at the shorter episode length of each paired rollout.  The common-prefix
comparison asks whether task progress remains after removing extra exposure from
surviving longer.

Collection is intentionally separate from training.  Use ``--dry-run`` first to
validate checkpoint coverage without importing MuJoCo or torch.

Examples:
  python scripts/stability_deconfounded_audit.py collect \
    --spec configs/experiments/stability_deconfounded_audit_v1.json --dry-run
  python scripts/stability_deconfounded_audit.py collect \
    --spec configs/experiments/stability_deconfounded_audit_v1.json \
    --out logs/probe/stability_deconfounded_audit_v1.jsonl
  python scripts/stability_deconfounded_audit.py summarize \
    --spec configs/experiments/stability_deconfounded_audit_v1.json \
    --input logs/probe/stability_deconfounded_audit_v1.jsonl \
    --json-out logs/probe/stability_deconfounded_audit_v1_summary.json \
    --md-out docs/stability_deconfounded_audit_v1_results.md
"""
from __future__ import annotations

import argparse
import copy
import glob
import json
import math
import os
import statistics
import sys
import time
from collections import defaultdict
from itertools import product
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ACTION_DIM = 61
REDUCERS = {"max", "min", "mean", "sum", "last"}


def load_spec(path: str | Path) -> dict[str, Any]:
    spec = json.loads(Path(path).read_text())
    if int(spec.get("schema_version", 0)) != 1:
        raise ValueError("expected schema_version=1")
    for task in spec.get("tasks", []):
        metric = task.get("primary_metric", {})
        if metric.get("reducer") not in REDUCERS:
            raise ValueError(f"{task.get('name')}: unsupported reducer {metric.get('reducer')!r}")
    return spec


def reduce_trace(trace: Iterable[float], reducer: str, horizon: int | None = None) -> float:
    values = [float(v) for v in trace]
    if horizon is not None:
        values = values[: max(0, int(horizon))]
    if not values:
        return float("nan")
    if reducer == "max":
        return max(values)
    if reducer == "min":
        return min(values)
    if reducer == "mean":
        return statistics.fmean(values)
    if reducer == "sum":
        return sum(values)
    if reducer == "last":
        return values[-1]
    raise ValueError(f"unsupported reducer: {reducer}")


def normalize_metric_spec(metric: str | dict[str, Any], default_reducer: str = "max") -> dict[str, str]:
    if isinstance(metric, str):
        return {"key": metric, "reducer": default_reducer}
    return {"key": str(metric["key"]), "reducer": str(metric.get("reducer", default_reducer))}


def _mean(values: Iterable[float]) -> float:
    clean = [float(v) for v in values if math.isfinite(float(v))]
    return statistics.fmean(clean) if clean else float("nan")


def _mean_sd_t(values: Iterable[float]) -> dict[str, float | int]:
    clean = [float(v) for v in values if math.isfinite(float(v))]
    if not clean:
        return {"n_train_seeds": 0, "mean": float("nan"), "sd": float("nan"), "t_vs_zero": float("nan")}
    mean = statistics.fmean(clean)
    if len(clean) == 1:
        return {"n_train_seeds": 1, "mean": mean, "sd": float("nan"), "t_vs_zero": float("nan")}
    sd = statistics.stdev(clean)
    if sd > 0:
        t_value = mean / (sd / math.sqrt(len(clean)))
    elif mean > 0:
        t_value = math.inf
    elif mean < 0:
        t_value = -math.inf
    else:
        t_value = 0.0
    return {"n_train_seeds": len(clean), "mean": mean, "sd": sd, "t_vs_zero": t_value}


def task_episode_steps(spec: dict[str, Any], task: dict[str, Any]) -> int:
    """Return the evaluation horizon for the task's training MDP."""

    steps = int(task.get("episode_steps", spec["episode_steps"]))
    if steps <= 0:
        raise ValueError(f"{task.get('name')}: episode_steps must be positive")
    return steps


def _step_token(step: int) -> str:
    return "final" if int(step) == 100000 else str(int(step))


def _checkpoint_for(pattern: str, seed: int, step: int) -> str | None:
    matches = sorted(glob.glob(pattern.format(seed=seed, step=_step_token(step))))
    return matches[-1] if matches else None


def checkpoint_inventory(spec: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task in spec["tasks"]:
        for condition in task["conditions"]:
            for seed in spec["train_seeds"]:
                for step in spec.get("checkpoint_steps", [100000]):
                    path = _checkpoint_for(condition["checkpoint_glob"], int(seed), int(step))
                    rows.append({
                        "task": task["name"],
                        "condition": condition["name"],
                        "train_seed": int(seed),
                        "checkpoint_step": int(step),
                        "required": bool(condition.get("required", True)),
                        "checkpoint": path,
                    })
    return rows


def filtered_collect_spec(spec: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """Apply collection-only CLI filters without mutating the preregistered spec."""
    out = copy.deepcopy(spec)
    if args.tasks:
        wanted = set(args.tasks)
        known = {task["name"] for task in out["tasks"]}
        unknown = wanted - known
        if unknown:
            raise ValueError(f"unknown tasks: {sorted(unknown)}")
        out["tasks"] = [task for task in out["tasks"] if task["name"] in wanted]
    if args.conditions:
        wanted = set(args.conditions)
        for task in out["tasks"]:
            task["conditions"] = [
                condition for condition in task["conditions"] if condition["name"] in wanted
            ]
            if not task["conditions"]:
                raise ValueError(f"{task['name']}: no conditions remain after filtering")
    if args.train_seeds:
        out["train_seeds"] = [int(seed) for seed in args.train_seeds]
    if args.checkpoint_steps:
        out["checkpoint_steps"] = [int(step) for step in args.checkpoint_steps]
    if args.eval_seeds:
        out["eval_seeds"] = [int(seed) for seed in args.eval_seeds]
    if args.num_envs is not None:
        out["num_envs"] = int(args.num_envs)
    if args.episode_steps is not None:
        out["episode_steps"] = int(args.episode_steps)
        for task in out["tasks"]:
            task["episode_steps"] = int(args.episode_steps)
    reference_condition = getattr(args, "reference_condition", None)
    if reference_condition:
        out["reference_condition"] = str(reference_condition)
        for task in out["tasks"]:
            condition_names = {condition["name"] for condition in task["conditions"]}
            if reference_condition not in condition_names:
                raise ValueError(
                    f"{task['name']}: reference condition {reference_condition!r} "
                    "is not present after filtering"
                )
    return out


def _numeric(value: Any) -> float | None:
    if isinstance(value, (bool, int, float)):
        return float(value)
    try:
        import numpy as np

        if isinstance(value, (np.bool_, np.integer, np.floating)):
            return float(value)
    except ImportError:
        pass
    return None


def _make_env_fn(env_id: str, rank: int, max_steps: int, seed: int):
    del rank, seed  # VecEnv.seed below supplies reset seeds per rank.

    def _init():
        from fasttd3_ptf.official_fasttd3_ptf.paths import ensure_humanoidbench_import_path

        ensure_humanoidbench_import_path()
        import gymnasium as gym
        import humanoid_bench  # noqa: F401
        from gymnasium.wrappers import TimeLimit
        from fasttd3_ptf.official_fasttd3_ptf.humanoid_bench_env import (
            GlobalNumpySeedOnReset,
        )

        env = gym.make(env_id)
        env = TimeLimit(env, max_episode_steps=max_steps)
        # HumanoidBench base reset uses env.np_random, while basketball and a
        # few task-specific resets use worker-global np.random. Seed both.
        env = GlobalNumpySeedOnReset(env)
        return env

    return _init


def _rollout(
    envs,
    act_fn,
    *,
    task: dict[str, Any],
    condition: str,
    checkpoint: str,
    checkpoint_step: int,
    train_seed: int,
    eval_seed: int,
    episode_steps: int,
) -> list[dict[str, Any]]:
    import numpy as np

    secondary_specs = [normalize_metric_spec(metric) for metric in task.get("secondary_metrics", [])]
    stability_specs = [normalize_metric_spec(metric, "mean") for metric in task.get("stability_metrics", [])]
    metric_keys = {
        task["primary_metric"]["key"],
        *(metric["key"] for metric in secondary_specs),
        *(metric["key"] for metric in stability_specs),
    }
    num_envs = int(envs.num_envs)
    obs = envs.reset()
    active = np.ones(num_envs, dtype=bool)
    returns = np.zeros(num_envs, dtype=np.float64)
    lengths = np.zeros(num_envs, dtype=np.int64)
    traces = [{key: [] for key in metric_keys} for _ in range(num_envs)]
    terminal = [None for _ in range(num_envs)]

    for _ in range(episode_steps + 5):
        actions = act_fn(obs)
        obs, rewards, dones, infos = envs.step(actions)
        for i in np.nonzero(active)[0]:
            returns[i] += float(rewards[i])
            lengths[i] += 1
            info = infos[i]
            for key in metric_keys:
                value = _numeric(info.get(key, 0.0))
                traces[i][key].append(0.0 if value is None else value)
            if bool(dones[i]):
                truncated = bool(info.get("TimeLimit.truncated", False))
                success = bool(info.get("success", False)) or max(traces[i].get("success", [0.0])) > 0
                reason = _numeric(info.get("terminated_reason"))
                early_failure = not truncated and not success
                mode = task.get("fall_mode", "unclassified_early_failure")
                if mode == "non_success_termination":
                    fallen: bool | None = early_failure
                elif mode == "reason_zero":
                    fallen = reason == 0.0
                else:
                    fallen = None
                terminal[i] = {
                    "truncated": truncated,
                    "success": success,
                    "early_failure": early_failure,
                    "fallen": fallen,
                    "termination_reason": reason,
                }
                active[i] = False
        if not active.any():
            break

    rows = []
    for i in range(num_envs):
        term = terminal[i] or {
            "truncated": True,
            "success": False,
            "early_failure": False,
            "fallen": False if task.get("fall_mode") == "non_success_termination" else None,
            "termination_reason": None,
        }
        rows.append({
            "schema_version": 1,
            "seed_protocol": "gymnasium_plus_global_numpy_vec_reset_v2",
            "task": task["name"],
            "env_id": task["env_id"],
            "condition": condition,
            "train_seed": int(train_seed),
            "eval_seed": int(eval_seed),
            "env_rank": i,
            "checkpoint": checkpoint,
            "checkpoint_step": int(checkpoint_step),
            "return": float(returns[i]),
            "ep_len": int(lengths[i]),
            **term,
            "metrics": traces[i],
            "evaluated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })
    return rows


def collect(args: argparse.Namespace) -> int:
    spec = filtered_collect_spec(load_spec(args.spec), args)
    inventory = checkpoint_inventory(spec)
    missing_required = [r for r in inventory if r["required"] and not r["checkpoint"]]
    for row in inventory:
        status = row["checkpoint"] or ("MISSING(required)" if row["required"] else "missing(optional)")
        print(
            f"{row['task']:12s} {row['condition']:8s} s{row['train_seed']} "
            f"@{row['checkpoint_step']:6d}  {status}"
        )
    if missing_required:
        print(f"\nerror: {len(missing_required)} required checkpoints are missing", file=sys.stderr)
        return 2
    if args.dry_run:
        present = sum(bool(r["checkpoint"]) for r in inventory)
        print(f"\ndry-run complete: {present}/{len(inventory)} checkpoint cells present")
        return 0

    out_path = Path(args.out)
    if out_path.exists() and not args.overwrite:
        raise FileExistsError(f"{out_path} exists; pass --overwrite to replace it")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("MUJOCO_GL", "egl")
    import torch
    from stable_baselines3.common.vec_env import SubprocVecEnv

    from fasttd3_ptf.official_fasttd3_ptf.paths import (
        ensure_fasttd3_import_path,
        ensure_humanoidbench_import_path,
    )
    from fasttd3_ptf.ptf.source_policy import SourcePolicy

    ensure_fasttd3_import_path()
    ensure_humanoidbench_import_path()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    by_key = {
        (r["task"], r["condition"], r["train_seed"], r["checkpoint_step"]): r
        for r in inventory
    }

    with out_path.open("w") as fout:
        for task in spec["tasks"]:
            rollout_steps = task_episode_steps(spec, task)
            for condition in task["conditions"]:
                for train_seed in spec["train_seeds"]:
                    for checkpoint_step in spec.get("checkpoint_steps", [100000]):
                        checkpoint = by_key[
                            (task["name"], condition["name"], int(train_seed), int(checkpoint_step))
                        ]["checkpoint"]
                        if not checkpoint:
                            continue
                        for eval_seed in spec["eval_seeds"]:
                        # Recreate envs for every condition so paired policies receive the
                        # same initial RNG state rather than an advanced shared stream.
                            envs = SubprocVecEnv([
                                _make_env_fn(task["env_id"], rank, rollout_steps, int(eval_seed))
                                for rank in range(int(spec["num_envs"]))
                            ])
                            # SB3 passes eval_seed + rank on the next reset;
                            # GlobalNumpySeedOnReset applies it to both the
                            # Gymnasium RNG and task-level global NumPy RNG.
                            envs.seed(int(eval_seed))
                            try:
                                obs_dim = int(envs.observation_space.shape[0])
                                policy = SourcePolicy(
                                    Path(checkpoint).stem,
                                    checkpoint,
                                    device=device,
                                    target_action_dim=ACTION_DIM,
                                    source_obs_dim=obs_dim,
                                    source_action_dim=ACTION_DIM,
                                )
                                act_fn = lambda obs, _p=policy: _p.act(
                                    torch.as_tensor(obs, device=device, dtype=torch.float32)
                                ).cpu().numpy()
                                rows = _rollout(
                                    envs,
                                    act_fn,
                                    task=task,
                                    condition=condition["name"],
                                    checkpoint=checkpoint,
                                    checkpoint_step=int(checkpoint_step),
                                    train_seed=int(train_seed),
                                    eval_seed=int(eval_seed),
                                    episode_steps=rollout_steps,
                                )
                            finally:
                                envs.close()
                            for row in rows:
                                fout.write(json.dumps(row, allow_nan=True) + "\n")
                            fout.flush()
                            print(
                                f"collected {task['name']} {condition['name']} s{train_seed} "
                                f"@{checkpoint_step} eval_seed={eval_seed}: {len(rows)} episodes",
                                flush=True,
                            )
    print(f"saved -> {out_path}")
    return 0


def _pair_key(row: dict[str, Any]) -> tuple[int, int, int, int]:
    return (
        int(row["train_seed"]),
        int(row.get("checkpoint_step", 100000)),
        int(row["eval_seed"]),
        int(row["env_rank"]),
    )


def validate_record_coverage(spec: dict[str, Any], records: list[dict[str, Any]]) -> dict[str, Any]:
    """Validate duplicate-free full factorial coverage for every observed condition."""
    errors: list[str] = []
    seen: set[tuple[Any, ...]] = set()
    duplicate_count = 0
    for row in records:
        key = (
            row["task"], row["condition"], int(row["train_seed"]),
            int(row.get("checkpoint_step", 100000)), int(row["eval_seed"]), int(row["env_rank"]),
        )
        if key in seen:
            duplicate_count += 1
        seen.add(key)
    if duplicate_count:
        errors.append(f"duplicate episode keys: {duplicate_count}")

    train_seeds = spec.get("train_seeds") or sorted({int(row["train_seed"]) for row in records})
    checkpoint_steps = spec.get("checkpoint_steps") or sorted({
        int(row.get("checkpoint_step", 100000)) for row in records
    })
    eval_seeds = spec.get("eval_seeds") or sorted({int(row["eval_seed"]) for row in records})
    num_envs = int(spec.get("num_envs", max((int(row["env_rank"]) for row in records), default=-1) + 1))
    expected_tail = set(product(
        [int(seed) for seed in train_seeds],
        [int(step) for step in checkpoint_steps],
        [int(seed) for seed in eval_seeds],
        range(num_envs),
    ))
    task_coverage: dict[str, Any] = {}
    for task in spec["tasks"]:
        task_name = task["name"]
        conditions = sorted({row["condition"] for row in records if row["task"] == task_name})
        condition_coverage = {}
        if not conditions:
            errors.append(f"{task_name}: no records")
        for condition in conditions:
            actual = {
                (int(row["train_seed"]), int(row.get("checkpoint_step", 100000)),
                 int(row["eval_seed"]), int(row["env_rank"]))
                for row in records
                if row["task"] == task_name and row["condition"] == condition
            }
            missing = expected_tail - actual
            extra = actual - expected_tail
            condition_coverage[condition] = {
                "expected": len(expected_tail),
                "actual": len(actual),
                "missing": len(missing),
                "extra": len(extra),
            }
            if missing or extra:
                errors.append(
                    f"{task_name}/{condition}: missing={len(missing)} extra={len(extra)}"
                )
        task_coverage[task_name] = condition_coverage
    return {
        "complete": not errors,
        "records": len(records),
        "duplicate_count": duplicate_count,
        "tasks": task_coverage,
        "errors": errors,
    }


def records_for_spec(spec: dict[str, Any], records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Restrict combined input files to the factorial cells selected in ``spec``."""
    conditions = {
        task["name"]: {condition["name"] for condition in task["conditions"]}
        for task in spec["tasks"]
    }
    train_seeds = {int(seed) for seed in spec["train_seeds"]}
    checkpoint_steps = {int(step) for step in spec.get("checkpoint_steps", [100000])}
    eval_seeds = {int(seed) for seed in spec["eval_seeds"]}
    num_envs = int(spec["num_envs"])
    return [
        row for row in records
        if row["task"] in conditions
        and row["condition"] in conditions[row["task"]]
        and int(row["train_seed"]) in train_seeds
        and int(row.get("checkpoint_step", 100000)) in checkpoint_steps
        and int(row["eval_seed"]) in eval_seeds
        and 0 <= int(row["env_rank"]) < num_envs
    ]


def summarize_records(spec: dict[str, Any], records: list[dict[str, Any]]) -> dict[str, Any]:
    reference = spec["reference_condition"]
    seed_protocols = sorted({str(row.get("seed_protocol", "legacy_unknown")) for row in records})
    pairing_validated = seed_protocols == [
        "gymnasium_plus_global_numpy_vec_reset_v2"
    ]
    output: dict[str, Any] = {
        "schema_version": 1,
        "reference_condition": reference,
        "estimand": "paired common-survival-prefix task progress",
        "coverage": validate_record_coverage(spec, records),
        "seed_protocols": seed_protocols,
        "exact_reset_pairing_validated": pairing_validated,
        "tasks": {},
    }
    for task in spec["tasks"]:
        task_rows = [r for r in records if r["task"] == task["name"]]
        by_condition: dict[str, dict[tuple[int, int, int, int], dict[str, Any]]] = defaultdict(dict)
        for row in task_rows:
            by_condition[row["condition"]][_pair_key(row)] = row
        ref = by_condition.get(reference, {})
        reducer = task["primary_metric"]["reducer"]
        metric_key = task["primary_metric"]["key"]
        secondary_specs = [normalize_metric_spec(metric) for metric in task.get("secondary_metrics", [])]
        task_out = {
            "primary_metric": task["primary_metric"],
            "secondary_metrics": secondary_specs,
            "comparisons": {},
        }
        for condition, index in sorted(by_condition.items()):
            if condition == reference:
                continue
            common = sorted(set(ref) & set(index))
            per_step_seed: dict[int, dict[int, dict[str, list[float]]]] = defaultdict(
                lambda: defaultdict(lambda: defaultdict(list))
            )
            for key in common:
                r0, r1 = ref[key], index[key]
                horizon = min(int(r0["ep_len"]), int(r1["ep_len"]))
                trace0 = r0["metrics"].get(metric_key, [])
                trace1 = r1["metrics"].get(metric_key, [])
                prefix_delta = reduce_trace(trace1, reducer, horizon) - reduce_trace(trace0, reducer, horizon)
                raw_delta = reduce_trace(trace1, reducer) - reduce_trace(trace0, reducer)
                seed, checkpoint_step = int(key[0]), int(key[1])
                metrics = per_step_seed[checkpoint_step][seed]
                metrics["common_prefix_progress_reference"].append(
                    reduce_trace(trace0, reducer, horizon)
                )
                metrics["common_prefix_progress_condition"].append(
                    reduce_trace(trace1, reducer, horizon)
                )
                metrics["common_prefix_progress_delta"].append(prefix_delta)
                metrics["raw_progress_reference"].append(reduce_trace(trace0, reducer))
                metrics["raw_progress_condition"].append(reduce_trace(trace1, reducer))
                metrics["raw_progress_delta"].append(raw_delta)
                metrics["episode_length_reference"].append(float(r0["ep_len"]))
                metrics["episode_length_condition"].append(float(r1["ep_len"]))
                metrics["episode_length_delta"].append(float(r1["ep_len"] - r0["ep_len"]))
                metrics["return_reference"].append(float(r0["return"]))
                metrics["return_condition"].append(float(r1["return"]))
                metrics["return_delta"].append(float(r1["return"] - r0["return"]))
                metrics["early_failure_reference"].append(float(bool(r0["early_failure"])))
                metrics["early_failure_condition"].append(float(bool(r1["early_failure"])))
                metrics["early_failure_delta"].append(
                    float(bool(r1["early_failure"])) - float(bool(r0["early_failure"]))
                )
                for secondary in secondary_specs:
                    secondary_key = secondary["key"]
                    secondary_reducer = secondary["reducer"]
                    trace0_secondary = r0["metrics"].get(secondary_key, [])
                    trace1_secondary = r1["metrics"].get(secondary_key, [])
                    prefix = f"secondary::{secondary_key}::"
                    metrics[prefix + "common_prefix_reference"].append(
                        reduce_trace(trace0_secondary, secondary_reducer, horizon)
                    )
                    metrics[prefix + "common_prefix_condition"].append(
                        reduce_trace(trace1_secondary, secondary_reducer, horizon)
                    )
                    metrics[prefix + "common_prefix_delta"].append(
                        reduce_trace(trace1_secondary, secondary_reducer, horizon)
                        - reduce_trace(trace0_secondary, secondary_reducer, horizon)
                    )
                    metrics[prefix + "raw_reference"].append(
                        reduce_trace(trace0_secondary, secondary_reducer)
                    )
                    metrics[prefix + "raw_condition"].append(
                        reduce_trace(trace1_secondary, secondary_reducer)
                    )
                    metrics[prefix + "raw_delta"].append(
                        reduce_trace(trace1_secondary, secondary_reducer)
                        - reduce_trace(trace0_secondary, secondary_reducer)
                    )
                if r0.get("fallen") is not None and r1.get("fallen") is not None:
                    metrics["fall_delta"].append(float(bool(r1["fallen"])) - float(bool(r0["fallen"])))
                stability0, stability1 = [], []
                for stability in task.get("stability_metrics", []):
                    stability_key = normalize_metric_spec(stability, "mean")["key"]
                    if stability_key in r0["metrics"] and stability_key in r1["metrics"]:
                        stability0.append(reduce_trace(r0["metrics"][stability_key], "mean", horizon))
                        stability1.append(reduce_trace(r1["metrics"][stability_key], "mean", horizon))
                if stability0 and stability1:
                    metrics["common_prefix_stability_delta"].append(_mean(stability1) - _mean(stability0))

            by_step = {}
            for checkpoint_step, per_seed in sorted(per_step_seed.items()):
                seed_means: dict[str, dict[str, float]] = {}
                metric_names = sorted({name for metrics in per_seed.values() for name in metrics})
                for seed, metrics in sorted(per_seed.items()):
                    seed_means[str(seed)] = {
                        name: _mean(metrics.get(name, [])) for name in metric_names
                    }
                    seed_means[str(seed)]["n_episode_pairs"] = len(
                        metrics.get("common_prefix_progress_delta", [])
                    )
                aggregate = {
                    name: _mean_sd_t(seed_means[str(seed)][name] for seed in sorted(per_seed))
                    for name in metric_names
                }
                by_step[str(checkpoint_step)] = {
                    "n_episode_pairs": sum(
                        len(metrics.get("common_prefix_progress_delta", []))
                        for metrics in per_seed.values()
                    ),
                    "seed_means": seed_means,
                    "aggregate_over_train_seeds": aggregate,
                }
            task_out["comparisons"][condition] = {
                "n_episode_pairs": len(common),
                "by_checkpoint_step": by_step,
            }
        output["tasks"][task["name"]] = task_out
    return output


def render_markdown(summary: dict[str, Any]) -> str:
    reference = summary["reference_condition"]
    pairing_validated = bool(summary.get("exact_reset_pairing_validated", False))
    pairing_note = (
        "> Pairing validity: Gymnasium reset seeds were explicitly applied through VecEnv.seed; "
        "same seed/rank denotes the same reset state."
        if pairing_validated
        else "> **Validity warning:** these records predate verified Gymnasium reset seeding. "
        "Condition means remain descriptive, but episode rows are not exact same-state counterfactual pairs."
    )
    lines = [
        "# Stability-deconfounded transfer audit v1 results",
        "",
        pairing_note,
        "",
        "> Generated from episode-paired evaluations. Statistical units are training seeds;",
        "> environment episodes are averaged within each seed before descriptive t statistics.",
        "",
        "The primary estimand is task-progress difference at the shorter survival prefix of",
        f"each condition/{reference} episode pair. Positive values therefore cannot be explained",
        "only by the condition remaining alive for more steps.",
        "",
        f"| task | step | condition vs {reference} | pairs | common-prefix progress {reference}→condition (Δ) | raw progress {reference}→condition (Δ) | stability Δ | return {reference}→condition (Δ) | episode length Δ | early failure {reference}→condition (Δ) |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    def fmt(metric: dict[str, Any] | None) -> str:
        if not metric or not math.isfinite(float(metric.get("mean", float("nan")))):
            return "—"
        mean = float(metric["mean"])
        sd = float(metric.get("sd", float("nan")))
        return f"{mean:+.4g}±{sd:.3g}" if math.isfinite(sd) else f"{mean:+.4g}"

    def triplet(agg: dict[str, Any], reference: str, condition: str, delta: str) -> str:
        return f"{fmt(agg.get(reference))}→{fmt(agg.get(condition))} ({fmt(agg.get(delta))})"

    for task, task_data in summary["tasks"].items():
        for condition, comparison in task_data["comparisons"].items():
            for checkpoint_step, step_data in comparison["by_checkpoint_step"].items():
                agg = step_data["aggregate_over_train_seeds"]
                lines.append(
                    f"| {task} | {checkpoint_step} | {condition} | {step_data['n_episode_pairs']} | "
                    f"{triplet(agg, 'common_prefix_progress_reference', 'common_prefix_progress_condition', 'common_prefix_progress_delta')} | "
                    f"{triplet(agg, 'raw_progress_reference', 'raw_progress_condition', 'raw_progress_delta')} | "
                    f"{fmt(agg.get('common_prefix_stability_delta'))} | "
                    f"{triplet(agg, 'return_reference', 'return_condition', 'return_delta')} | "
                    f"{fmt(agg.get('episode_length_delta'))} | "
                    f"{triplet(agg, 'early_failure_reference', 'early_failure_condition', 'early_failure_delta')} |"
                )
    lines.extend([
        "",
        "## Secondary task metrics at the common survival prefix",
        "",
        f"| task | step | condition | metric | {reference}→condition (Δ) |",
        "|---|---:|---|---|---:|",
    ])
    for task, task_data in summary["tasks"].items():
        for condition, comparison in task_data["comparisons"].items():
            for checkpoint_step, step_data in comparison["by_checkpoint_step"].items():
                agg = step_data["aggregate_over_train_seeds"]
                for secondary in task_data.get("secondary_metrics", []):
                    prefix = f"secondary::{secondary['key']}::common_prefix_"
                    lines.append(
                        f"| {task} | {checkpoint_step} | {condition} | {secondary['key']} | "
                        f"{triplet(agg, prefix + 'reference', prefix + 'condition', prefix + 'delta')} |"
                    )
    lines.extend([
        "",
        "Interpretation rule: a positive raw-progress delta with a near-zero common-prefix",
        "delta is consistent with stability/exposure mediation. A positive common-prefix delta",
        "shows faster progress in the evaluated condition distribution; it is a same-state",
        "counterfactual contrast only when exact_reset_pairing_validated=true. Cabinet episode",
        "length cannot by itself remove posture mediation because falling does not terminate it.",
        "",
    ])
    return "\n".join(lines)


def summarize(args: argparse.Namespace) -> int:
    spec = filtered_collect_spec(load_spec(args.spec), args)
    records = []
    for input_path in args.input:
        records.extend(
            json.loads(line)
            for line in Path(input_path).read_text().splitlines()
            if line.strip()
        )
    records = records_for_spec(spec, records)
    coverage = validate_record_coverage(spec, records)
    if args.require_complete and not coverage["complete"]:
        raise ValueError("incomplete audit records: " + "; ".join(coverage["errors"]))
    summary = summarize_records(spec, records)
    json_path = Path(args.json_out)
    md_path = Path(args.md_out)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(summary, indent=2, allow_nan=True) + "\n")
    md_path.write_text(render_markdown(summary))
    print(render_markdown(summary))
    print(f"saved JSON -> {json_path}")
    print(f"saved Markdown -> {md_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    collect_parser = sub.add_parser("collect")
    collect_parser.add_argument("--spec", required=True)
    collect_parser.add_argument("--out", default="logs/probe/stability_deconfounded_audit_v1.jsonl")
    collect_parser.add_argument("--dry-run", action="store_true")
    collect_parser.add_argument("--overwrite", action="store_true")
    collect_parser.add_argument("--tasks", nargs="+")
    collect_parser.add_argument("--conditions", nargs="+")
    collect_parser.add_argument("--train-seeds", nargs="+", type=int)
    collect_parser.add_argument("--checkpoint-steps", nargs="+", type=int)
    collect_parser.add_argument("--eval-seeds", nargs="+", type=int)
    collect_parser.add_argument("--num-envs", type=int)
    collect_parser.add_argument("--episode-steps", type=int)
    collect_parser.set_defaults(func=collect)
    summary_parser = sub.add_parser("summarize")
    summary_parser.add_argument("--spec", required=True)
    summary_parser.add_argument("--input", nargs="+", required=True)
    summary_parser.add_argument("--json-out", required=True)
    summary_parser.add_argument("--md-out", required=True)
    summary_parser.add_argument("--require-complete", action="store_true")
    summary_parser.add_argument("--tasks", nargs="+")
    summary_parser.add_argument("--conditions", nargs="+")
    summary_parser.add_argument("--train-seeds", nargs="+", type=int)
    summary_parser.add_argument("--checkpoint-steps", nargs="+", type=int)
    summary_parser.add_argument("--eval-seeds", nargs="+", type=int)
    summary_parser.add_argument("--num-envs", type=int)
    summary_parser.add_argument("--episode-steps", type=int)
    summary_parser.add_argument("--reference-condition")
    summary_parser.set_defaults(func=summarize)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
