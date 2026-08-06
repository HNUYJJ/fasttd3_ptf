import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.stability_deconfounded_audit import (
    _mean_sd_t,
    filtered_collect_spec,
    records_for_spec,
    reduce_trace,
    summarize_records,
    task_episode_steps,
)


def test_reduce_trace_prefix_semantics():
    trace = [1.0, 3.0, 2.0, 5.0]
    assert reduce_trace(trace, "max", 3) == 3.0
    assert reduce_trace(trace, "min", 3) == 1.0
    assert reduce_trace(trace, "mean", 2) == 2.0
    assert reduce_trace(trace, "sum", 2) == 4.0
    assert reduce_trace(trace, "last", 3) == 2.0


def test_zero_variance_t_keeps_the_sign():
    assert _mean_sd_t([1.0, 1.0, 1.0])["t_vs_zero"] == float("inf")
    assert _mean_sd_t([-1.0, -1.0, -1.0])["t_vs_zero"] == float("-inf")
    assert _mean_sd_t([0.0, 0.0, 0.0])["t_vs_zero"] == 0.0


def test_task_specific_episode_steps_and_cli_override():
    spec = {
        "episode_steps": 1000,
        "tasks": [
            {"name": "powerlift", "episode_steps": 1000, "conditions": [{"name": "scratch"}]},
            {"name": "basketball", "episode_steps": 500, "conditions": [{"name": "scratch"}]},
        ],
    }
    assert task_episode_steps(spec, spec["tasks"][0]) == 1000
    assert task_episode_steps(spec, spec["tasks"][1]) == 500
    args = argparse.Namespace(
        tasks=None, conditions=None, train_seeds=None, checkpoint_steps=None,
        eval_seeds=None, num_envs=None, episode_steps=25,
    )
    filtered = filtered_collect_spec(spec, args)
    assert all(task_episode_steps(filtered, task) == 25 for task in filtered["tasks"])


def _row(condition, train_seed, env_rank, length, progress, stability, ret, failure=False):
    return {
        "task": "toy",
        "condition": condition,
        "train_seed": train_seed,
        "checkpoint_step": 100000,
        "eval_seed": 7,
        "env_rank": env_rank,
        "ep_len": length,
        "return": ret,
        "early_failure": failure,
        "fallen": failure,
        "metrics": {"progress": progress, "upright": stability},
    }


def test_common_prefix_removes_extra_survival_exposure():
    spec = {
        "reference_condition": "scratch",
        "tasks": [{
            "name": "toy",
            "primary_metric": {"key": "progress", "reducer": "sum", "direction": "max"},
            "stability_metrics": ["upright"],
        }],
    }
    records = []
    for seed in (1, 2):
        # Same progress rate during the first two steps.  Transfer only survives
        # two extra steps, so raw progress improves but deconfounded progress is 0.
        records.append(_row("scratch", seed, 0, 2, [1, 1], [1, 1], 2, failure=True))
        records.append(_row("wfix", seed, 0, 4, [1, 1, 1, 1], [1, 1, 1, 1], 4))
    summary = summarize_records(spec, records)
    agg = summary["tasks"]["toy"]["comparisons"]["wfix"]["by_checkpoint_step"]["100000"]["aggregate_over_train_seeds"]
    assert agg["common_prefix_progress_delta"]["mean"] == 0.0
    assert agg["raw_progress_delta"]["mean"] == 2.0
    assert agg["episode_length_delta"]["mean"] == 2.0


def test_common_prefix_detects_true_progress_acceleration():
    spec = {
        "reference_condition": "scratch",
        "tasks": [{
            "name": "toy",
            "primary_metric": {"key": "progress", "reducer": "max", "direction": "max"},
            "stability_metrics": ["upright"],
        }],
    }
    records = [
        _row("scratch", 1, 0, 3, [0, 1, 1], [1, 1, 1], 2),
        _row("source", 1, 0, 3, [0, 2, 3], [1, 1, 1], 4),
    ]
    summary = summarize_records(spec, records)
    agg = summary["tasks"]["toy"]["comparisons"]["source"]["by_checkpoint_step"]["100000"]["aggregate_over_train_seeds"]
    assert agg["common_prefix_progress_delta"]["mean"] == 2.0


def test_exact_pairing_requires_dual_rng_seed_protocol_v2():
    spec = {
        "reference_condition": "scratch",
        "tasks": [{
            "name": "toy",
            "primary_metric": {"key": "progress", "reducer": "max", "direction": "max"},
            "stability_metrics": ["upright"],
        }],
    }
    records = [
        _row("scratch", 1, 0, 2, [0, 1], [1, 1], 1),
        _row("source", 1, 0, 2, [0, 1], [1, 1], 1),
    ]
    for row in records:
        row["seed_protocol"] = "gymnasium_plus_global_numpy_vec_reset_v2"
    assert summarize_records(spec, records)["exact_reset_pairing_validated"] is True

    records[0]["seed_protocol"] = "gymnasium_vec_reset_v1"
    assert summarize_records(spec, records)["exact_reset_pairing_validated"] is False


def test_collect_filters_do_not_mutate_preregistered_spec():
    spec = {
        "tasks": [
            {"name": "a", "conditions": [{"name": "scratch"}, {"name": "wfix"}]},
            {"name": "b", "conditions": [{"name": "scratch"}]},
        ],
        "train_seeds": [1, 2, 3],
        "checkpoint_steps": [10000, 30000, 100000],
        "eval_seeds": [7, 1007],
        "num_envs": 16,
        "episode_steps": 1000,
    }
    args = argparse.Namespace(
        tasks=["a"], conditions=["scratch"], train_seeds=[2], checkpoint_steps=[30000],
        eval_seeds=[9], num_envs=2, episode_steps=20,
    )
    filtered = filtered_collect_spec(spec, args)
    assert [task["name"] for task in filtered["tasks"]] == ["a"]
    assert [condition["name"] for condition in filtered["tasks"][0]["conditions"]] == ["scratch"]
    assert filtered["train_seeds"] == [2]
    assert filtered["checkpoint_steps"] == [30000]
    assert filtered["eval_seeds"] == [9]
    assert filtered["num_envs"] == 2
    assert filtered["episode_steps"] == 20
    assert len(spec["tasks"]) == 2


def test_summary_reference_condition_override_is_validated():
    spec = {
        "reference_condition": "scratch",
        "tasks": [{
            "name": "toy",
            "conditions": [{"name": "scratch"}, {"name": "stand"}, {"name": "run"}],
        }],
        "train_seeds": [1],
        "checkpoint_steps": [100000],
        "eval_seeds": [7],
        "num_envs": 1,
        "episode_steps": 10,
    }
    args = argparse.Namespace(
        tasks=["toy"], conditions=["stand", "run"], train_seeds=None,
        checkpoint_steps=None, eval_seeds=None, num_envs=None, episode_steps=None,
        reference_condition="stand",
    )
    filtered = filtered_collect_spec(spec, args)
    assert filtered["reference_condition"] == "stand"
    assert spec["reference_condition"] == "scratch"

    args.reference_condition = "scratch"
    try:
        filtered_collect_spec(spec, args)
    except ValueError as exc:
        assert "not present after filtering" in str(exc)
    else:
        raise AssertionError("missing filtered reference condition should fail")


def test_combined_inputs_are_filtered_to_selected_factorial_cells():
    spec = {
        "tasks": [{"name": "toy", "conditions": [{"name": "scratch"}, {"name": "stand"}]}],
        "train_seeds": [1], "checkpoint_steps": [10000], "eval_seeds": [7], "num_envs": 1,
    }
    base = {
        "task": "toy", "condition": "scratch", "train_seed": 1,
        "checkpoint_step": 10000, "eval_seed": 7, "env_rank": 0,
    }
    records = [
        base,
        {**base, "condition": "stand"},
        {**base, "condition": "wfix"},
        {**base, "train_seed": 2},
        {**base, "task": "other"},
    ]
    selected = records_for_spec(spec, records)
    assert [(row["condition"], row["train_seed"]) for row in selected] == [
        ("scratch", 1), ("stand", 1)
    ]
