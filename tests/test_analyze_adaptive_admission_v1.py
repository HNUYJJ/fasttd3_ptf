from __future__ import annotations

from pathlib import Path

import torch

from scripts.analyze_adaptive_admission_v1 import (
    checkpoint_lifecycle,
    paired_seed,
)


def test_paired_seed_uses_only_frozen_grid_and_reports_missing() -> None:
    row = paired_seed(
        {5: 100.0, 10: 4.0, 15: 9.0, 999: 1000.0},
        {10: 1.0, 15: 5.0, 999: -1000.0},
        [10, 15, 20],
    )
    assert row["available_steps"] == [10, 15]
    assert row["missing_steps"] == [20]
    assert row["delta"] == {10: 3.0, 15: 4.0}
    assert row["mean_delta"] == 3.5
    assert row["complete"] is False


def _adaptive_payload() -> dict:
    initial = {
        "step": 0,
        "decision_id": "explicit-all",
        "source_names": ["bad"],
    }
    revocation = {
        "event": "adaptive_admission_window",
        "completed_step": 10,
        "window_index": 1,
        "revoked_sources": ["bad"],
        "statistics": [
            {"candidate": "bad", "count": 8, "mean": 0.0},
            {"candidate": "student", "count": 8, "mean": 1.0},
        ],
        "persistence_counts": [3],
        "discarded_partial_segments": 2,
        "execution_counts_at_apply": [5, 5],
        "replay_at_apply": {
            "critic_sample_counts": [100, 100],
            "active_buffer_counts": [0, 5],
            "effective_replay_masses": [0.0, 1.0],
            "candidate_masses": [0.0, 1.0],
            "admitted_sources": [False],
        },
    }
    later = {
        "event": "adaptive_admission_window",
        "completed_step": 20,
        "window_index": 2,
        "revoked_sources": [],
        "statistics": [
            {"candidate": "bad", "count": 0, "mean": None},
            {"candidate": "student", "count": 10, "mean": 1.2},
        ],
    }
    return {
        "global_step": 20,
        "ptf_cfg": {
            "admission_adaptive": True,
            "admission_replay_handoff": "physical_after_authority",
            "mcg_warmup_steps": 20,
        },
        "admission_audit": {
            "decision_history": [initial, revocation, later],
            "execution_counts": [5, 15],
            "critic_sample_counts": [100, 200],
            "active_buffer_counts": [0, 15],
            "effective_replay_masses": [0.0, 1.0],
            "admitted_sources": [False],
            "actor_sampling": "shared_critic_batch",
            "actor_independent_sample_counts": [0, 0],
            "source_authority_active": False,
            "sampling_phase": "physical_allowed",
        },
    }


def test_checkpoint_lifecycle_proves_revoked_source_freeze(tmp_path: Path) -> None:
    path = tmp_path / "adaptive.pt"
    torch.save(_adaptive_payload(), path)
    report = checkpoint_lifecycle(path, expected_adaptive=True)
    assert report["pass"] is True
    assert report["revocations"][0]["step"] == 10
    source = report["revocations"][0]["sources"][0]
    assert source["execution_at_apply"] == source["execution_at_checkpoint"] == 5
    assert source["critic_at_apply"] == source["critic_at_checkpoint"] == 100
    assert source["later_window_counts"] == [0]


def test_checkpoint_lifecycle_rejects_post_revocation_critic_growth(
    tmp_path: Path,
) -> None:
    payload = _adaptive_payload()
    payload["admission_audit"]["critic_sample_counts"][0] = 101
    path = tmp_path / "broken.pt"
    torch.save(payload, path)
    report = checkpoint_lifecycle(path, expected_adaptive=True)
    assert report["pass"] is False
    assert report["checks"]["step_10_bad_critic_frozen"] is False
