from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fasttd3_ptf.official_fasttd3_ptf.admission_control import (
    AdaptiveAdmissionController,
    build_admission_schedule,
    build_admission_snapshot,
    desired_admission_source_authority,
)


def test_none_is_exact_student_one_hot() -> None:
    snapshot = build_admission_snapshot(
        mode="none",
        source_names=["stand", "walk", "run"],
        source_logits=[100.0, 100.0, 100.0],
    )
    assert snapshot.exact_abstain
    torch.testing.assert_close(
        snapshot.candidate_probabilities(tau=1.0),
        torch.tensor([0.0, 0.0, 0.0, 1.0]),
    )


def test_target_evidence_starts_as_dynamic_exact_abstention() -> None:
    snapshot = build_admission_snapshot(
        mode="target_evidence",
        source_names=["stand", "walk", "run"],
        source_logits=[0.0, 0.0, 0.0],
    )
    assert snapshot.mode == "target_evidence"
    assert snapshot.exact_abstain
    torch.testing.assert_close(
        snapshot.candidate_probabilities(tau=1.0),
        torch.tensor([0.0, 0.0, 0.0, 1.0]),
    )


def test_static_masks_rejected_sources_and_keeps_student() -> None:
    snapshot = build_admission_snapshot(
        mode="static",
        source_names=["stand", "walk", "run"],
        source_logits=[5.0, 1.0, 2.0],
        student_logit=0.5,
        admitted_sources="run",
    )
    probs = snapshot.candidate_probabilities(tau=1.0)
    assert probs[0] == 0 and probs[1] == 0
    assert probs[2] > 0 and probs[3] > 0
    torch.testing.assert_close(probs.sum(), torch.tensor(1.0))


def test_manifest_binds_quarantine_digest(tmp_path: Path) -> None:
    artifact = tmp_path / "quarantine.pt"
    artifact.write_bytes(b"isolated probe data")
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    manifest = tmp_path / "decision.yaml"
    manifest.write_text(
        "\n".join(
            [
                "schema_version: 1",
                "decision_id: test-decision",
                "source_names: [stand, run]",
                "admitted_sources: [run]",
                "quarantine:",
                "  artifact: quarantine.pt",
                f"  sha256: {digest}",
            ]
        )
    )
    snapshot = build_admission_snapshot(
        mode="manifest",
        source_names=["stand", "run"],
        source_logits=[0.0, 1.0],
        manifest_path=manifest,
    )
    assert snapshot.admitted_names == ("run",)
    assert snapshot.quarantine_digest == digest
    assert snapshot.quarantine_artifact == str(artifact.resolve())


def test_unknown_static_source_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown sources"):
        build_admission_snapshot(
            mode="static",
            source_names=["stand"],
            source_logits=[0.0],
            admitted_sources="crawl",
        )


def test_explicit_schedule_changes_from_source_to_exact_abstention(tmp_path: Path) -> None:
    quarantine = tmp_path / "probe.pt"
    quarantine.write_bytes(b"quarantine-only")
    digest = hashlib.sha256(quarantine.read_bytes()).hexdigest()
    schedule_path = tmp_path / "schedule.yaml"
    schedule_path.write_text(
        "\n".join(
            [
                "schema_version: 1",
                "source_names: [stand, run]",
                "decisions:",
                "  - step: 0",
                "    decision_id: admit-run",
                "    admitted_sources: [run]",
                "    quarantine:",
                "      artifact: probe.pt",
                f"      sha256: {digest}",
                "  - step: 20",
                "    decision_id: revoke-all",
                "    admitted_sources: []",
            ]
        )
    )
    schedule = build_admission_schedule(
        schedule_path=schedule_path,
        source_names=["stand", "run"],
        source_logits=[0.0, 1.0],
    )
    assert schedule.snapshot_at(0).admitted_names == ("run",)
    assert schedule.snapshot_at(19).decision_id == "admit-run"
    assert schedule.snapshot_at(20).exact_abstain
    assert schedule.snapshot_at(20).quarantine_artifact is None


def _adaptive_controller(*, persistence: int = 2) -> AdaptiveAdmissionController:
    snapshot = build_admission_snapshot(
        mode="all",
        source_names=["bad", "good"],
        source_logits=[0.0, 0.0],
        student_logit=0.0,
    )
    return AdaptiveAdmissionController(
        initial_snapshot=snapshot,
        stage_window_steps=10,
        confidence_z=1.645,
        min_segments=2,
        persistence=persistence,
    )


def _record_clear_bad_source(controller: AdaptiveAdmissionController) -> None:
    controller.record_segments(
        candidate_ids=[0, 0, 1, 1, 2, 2],
        segment_mean_rewards=[-10.0, -10.0, 10.0, 10.0, 0.0, 0.0],
    )


def test_adaptive_revocation_requires_consecutive_stage_windows() -> None:
    controller = _adaptive_controller(persistence=2)
    _record_clear_bad_source(controller)
    first = controller.maybe_close_window(10)
    assert first is not None
    assert first.snapshot is None
    assert first.positive_votes == (True, False)
    assert first.persistence_counts == (1, 0)

    _record_clear_bad_source(controller)
    second = controller.maybe_close_window(20)
    assert second is not None and second.snapshot is not None
    assert second.revoked_sources == ("bad",)
    assert second.snapshot.admitted == (False, True)
    assert not second.snapshot.exact_abstain
    assert second.snapshot.decision_id == "adaptive-window-2-step-20"


def test_adaptive_insufficient_window_resets_persistence_and_statistics() -> None:
    controller = _adaptive_controller(persistence=2)
    _record_clear_bad_source(controller)
    first = controller.maybe_close_window(10)
    assert first is not None and first.persistence_counts[0] == 1

    controller.record_segments(
        candidate_ids=[0, 2, 2],
        segment_mean_rewards=[-10.0, 0.0, 0.0],
    )
    insufficient = controller.maybe_close_window(20)
    assert insufficient is not None
    assert insufficient.statistics[0].count == 1
    assert insufficient.persistence_counts[0] == 0

    empty = controller.maybe_close_window(30)
    assert empty is not None
    assert all(value.count == 0 for value in empty.statistics)
    assert empty.snapshot is None


def test_adaptive_multi_source_revocation_emits_one_exact_snapshot() -> None:
    controller = _adaptive_controller(persistence=1)
    controller.record_segments(
        candidate_ids=[0, 0, 1, 1, 2, 2],
        segment_mean_rewards=[-2.0, -2.0, -1.0, -1.0, 1.0, 1.0],
    )
    result = controller.maybe_close_window(10)
    assert result is not None and result.snapshot is not None
    assert result.revoked_sources == ("bad", "good")
    assert result.snapshot.exact_abstain
    assert result.snapshot.admitted_names == ()
    assert result.as_dict()["decision"]["mode"] == "adaptive"


def test_adaptive_window_clock_closes_once_and_rejects_skips() -> None:
    controller = _adaptive_controller()
    assert controller.maybe_close_window(9) is None
    assert controller.maybe_close_window(10) is not None
    assert controller.maybe_close_window(10) is None
    with pytest.raises(ValueError, match="boundary skipped"):
        controller.maybe_close_window(21)


def test_exact_abstention_cannot_be_revived_by_warmup_authority() -> None:
    admitted = build_admission_snapshot(
        mode="all", source_names=["source"], source_logits=[0.0]
    )
    exact = build_admission_snapshot(
        mode="none", source_names=["source"], source_logits=[0.0]
    )
    assert desired_admission_source_authority(
        admitted,
        global_step=10,
        warmup_steps=30,
        warmup_authority=True,
        post_warmup_authority=False,
    )
    assert not desired_admission_source_authority(
        exact,
        global_step=10,
        warmup_steps=30,
        warmup_authority=True,
        post_warmup_authority=False,
    )
    assert not desired_admission_source_authority(
        admitted,
        global_step=30,
        warmup_steps=30,
        warmup_authority=True,
        post_warmup_authority=False,
    )
