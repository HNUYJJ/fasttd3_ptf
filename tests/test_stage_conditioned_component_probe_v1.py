import importlib.util
from pathlib import Path
import sys

import numpy as np


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "probe_stage_conditioned_components_v1.py"
SPEC = importlib.util.spec_from_file_location("stage_component_probe", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_source_requires_reward_progress_and_feasibility_lower_bounds():
    positive = np.linspace(0.2, 1.0, 32)
    result = MODULE._classify_source(
        {"return": positive, "progress": positive},
        {"constraint": np.zeros(32)},
        hard_constraints=("constraint",),
    )
    assert result["admitted"] is True

    harmful_feasibility = MODULE._classify_source(
        {"return": positive, "progress": positive},
        {"constraint": -positive},
        hard_constraints=("constraint",),
    )
    assert harmful_feasibility["admitted"] is False


def test_exact_zero_progress_is_not_admitted():
    positive = np.ones(32)
    result = MODULE._classify_source(
        {"return": positive, "progress": np.zeros(32)},
        {"constraint": np.zeros(32)},
        hard_constraints=("constraint",),
    )
    assert result["admitted"] is False


def test_target_evidence_is_loaded_without_task_name_branches():
    from fasttd3_ptf.official_fasttd3_ptf.target_evidence import TargetEvidenceContract

    root = Path(__file__).resolve().parents[1]
    hurdle = TargetEvidenceContract.from_yaml(
        root / "configs/target_evidence/humanoidbench_hurdle_v1.yaml"
    )
    crawl = TargetEvidenceContract.from_yaml(
        root / "configs/target_evidence/humanoidbench_crawl_v1.yaml"
    )
    assert [item.name for item in hurdle.feasibility] == [
        "upright_support",
        "obstacle_clearance",
    ]
    assert [item.name for item in crawl.feasibility] == [
        "crawl_posture",
        "tunnel_occupancy",
    ]
    assert hurdle.progress.gate_components == (
        "upright_support",
        "obstacle_clearance",
    )
    assert crawl.progress.gate_components == ("crawl_posture", "tunnel_occupancy")
    assert hurdle.hard_constraints == ()
    assert crawl.hard_constraints == ()


def test_target_achievement_progress_is_gated_by_declared_components():
    from fasttd3_ptf.official_fasttd3_ptf.target_evidence import TargetEvidenceContract

    class Data:
        qpos = np.array([1.0])

    class Env:
        data = Data()
        unwrapped = None

    env = Env()
    env.unwrapped = env
    contract = TargetEvidenceContract.from_mapping(
        {
            "schema_version": 1,
            "name": "generic",
            "env_name": "generic-v0",
            "progress": {
                "kind": "sim_state_delta",
                "array": "qpos",
                "index": 0,
                "direction": "maximize",
                "gate_components": ["posture", "workspace"],
                "gate_reducer": "product",
            },
            "feasibility": [
                {
                    "name": "posture",
                    "info_keys": ["body", "head"],
                    "step_reducer": "min",
                    "temporal_reducer": "mean",
                },
                {
                    "name": "workspace",
                    "info_keys": ["inside"],
                    "step_reducer": "min",
                    "temporal_reducer": "mean",
                },
            ],
            "hard_constraints": [],
        }
    )
    evidence = contract.new_accumulator(env)
    evidence.observe({"body": 0.5, "head": 0.8, "inside": 1.0})
    env.data.qpos[0] = 3.0
    result = evidence.finish()
    assert result["progress"] == 1.0  # raw delta 2 × posture 0.5 × inside 1


def test_top1_snapshot_keeps_student_and_exactly_abstains_when_empty():
    import torch

    from fasttd3_ptf.official_fasttd3_ptf.target_evidence_probe import (
        build_top1_admission_snapshot,
    )

    admitted = build_top1_admission_snapshot(
        source_names=("stand", "walk", "run"),
        probe_result={"admitted_order": ["run", "walk"]},
        decision_step=10_000,
        quarantine_artifact="/tmp/probe.json",
        quarantine_digest="abc",
    )
    assert admitted.admitted_names == ("run",)
    torch.testing.assert_close(
        admitted.candidate_probabilities(tau=1.0),
        torch.tensor([0.0, 0.0, 0.5, 0.5]),
    )

    abstain = build_top1_admission_snapshot(
        source_names=("stand", "walk", "run"),
        probe_result={"admitted_order": []},
        decision_step=20_000,
        quarantine_artifact="/tmp/probe.json",
        quarantine_digest="def",
    )
    assert abstain.exact_abstain
    torch.testing.assert_close(
        abstain.candidate_probabilities(tau=1.0),
        torch.tensor([0.0, 0.0, 0.0, 1.0]),
    )
