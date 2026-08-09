from __future__ import annotations

import sys
from pathlib import Path

import torch
from tensordict import TensorDict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fasttd3_ptf.official_fasttd3_ptf.paths import ensure_fasttd3_import_path

ensure_fasttd3_import_path()
from fast_td3_utils import SimpleReplayBuffer  # type: ignore  # noqa: E402

from fasttd3_ptf.official_fasttd3_ptf.ptf_replay import PTFReplayWrapper  # noqa: E402
from fasttd3_ptf.official_fasttd3_ptf.admission_control import (  # noqa: E402
    AdaptiveAdmissionController,
    build_admission_snapshot,
)
from fasttd3_ptf.ptf.action_schema import h1hand_default_action_schema  # noqa: E402
from fasttd3_ptf.ptf.mcg import McgBehaviorController  # noqa: E402


def _make_replay(capacity: int = 3, n_env: int = 2) -> PTFReplayWrapper:
    return PTFReplayWrapper(
        SimpleReplayBuffer(
            n_env=n_env,
            buffer_size=capacity,
            n_obs=3,
            n_act=2,
            n_critic_obs=3,
            n_steps=1,
            device=torch.device("cpu"),
        )
    )


def _transition(value: int, n_env: int = 2) -> TensorDict:
    env_offset = torch.arange(n_env, dtype=torch.float32).view(n_env, 1)
    obs = torch.full((n_env, 3), float(value)) + env_offset
    actions = torch.full((n_env, 2), float(value) / 10.0) + env_offset
    return TensorDict(
        {
            "observations": obs,
            "actions": actions,
            "next": {
                "rewards": torch.full((n_env,), float(value)),
                "dones": torch.zeros(n_env, dtype=torch.long),
                "truncations": torch.zeros(n_env, dtype=torch.long),
                "observations": obs + 0.5,
            },
        },
        batch_size=n_env,
    )


def _provenance(value: int, n_env: int = 2) -> dict[str, torch.Tensor]:
    ranks = torch.arange(n_env)
    return {
        "behavior_source": torch.where(ranks == 0, -1, value).to(torch.int16),
        "source_by_group": torch.stack(
            [torch.full((4,), -1), torch.full((4,), value)]
        ).to(torch.int16),
        "executed_group_mask": torch.stack(
            [torch.zeros(4, dtype=torch.bool), torch.ones(4, dtype=torch.bool)]
        ),
        "segment_id": torch.full((n_env,), 100 + value),
        "segment_step": torch.full((n_env,), value),
        "anchor_id": torch.full((n_env,), 200 + value),
        "env_rank": ranks,
        "learner_step": torch.full((n_env,), 10_000),
    }


def test_valid_snapshot_roundtrip_preserves_ring_order_and_fixed_gather():
    replay = _make_replay()
    replay.enable_provenance(group_count=4)
    for value in range(5):
        replay.extend(
            _transition(value),
            torch.tensor([-1, value]),
            provenance=_provenance(value),
        )

    assert replay.ptr == 5
    snapshot = replay.export_valid(require_complete_provenance=True)
    assert snapshot["metadata"]["valid_size"] == 3
    torch.testing.assert_close(
        snapshot["tensors"]["observations"][0, :, 0],
        torch.tensor([2.0, 3.0, 4.0]),
    )

    restored = _make_replay()
    restored.import_valid(snapshot)
    assert restored.ptr == replay.ptr
    restored.assert_complete_provenance()

    restored_snapshot = restored.export_valid(require_complete_provenance=True)
    for name, tensor in snapshot["tensors"].items():
        assert torch.equal(tensor, restored_snapshot["tensors"][name]), name
    assert snapshot["provenance"] is not None
    assert restored_snapshot["provenance"] is not None
    for name, tensor in snapshot["provenance"].items():
        assert torch.equal(tensor, restored_snapshot["provenance"][name]), name

    indices = torch.tensor([[0, 2], [1, 0]])
    original_batch = replay.gather(indices)
    restored_batch = restored.gather(indices)
    assert set(original_batch.keys(True, True)) == set(restored_batch.keys(True, True))
    for key in original_batch.keys(True, True):
        assert torch.equal(original_batch[key], restored_batch[key]), key


def test_old_extend_and_sample_api_remain_available_without_provenance():
    replay = _make_replay(capacity=4)
    replay.extend(_transition(1), torch.tensor([-1, -1]))
    replay.extend(_transition(2), torch.tensor([-1, 0]))
    indices = replay.draw_indices(batch_size=3)
    assert indices.shape == (2, 3)
    batch = replay.gather(indices)
    assert batch.batch_size == torch.Size([6])
    assert "provenance_written" not in batch.keys()
    sampled = replay.sample(2, role="actor")
    assert sampled.batch_size == torch.Size([4])


def test_provenance_export_rejects_uninstrumented_valid_transition():
    replay = _make_replay(capacity=4)
    replay.enable_provenance(group_count=4)
    replay.extend(_transition(1), torch.tensor([-1, -1]))
    try:
        replay.export_valid(require_complete_provenance=True)
    except AssertionError as exc:
        assert "lack behavior provenance" in str(exc)
    else:
        raise AssertionError("missing behavior provenance must fail the paper-data gate")


def _filled_admission_replay() -> PTFReplayWrapper:
    replay = _make_replay(capacity=8, n_env=2)
    option_rows = [
        torch.tensor([-1, -1]),
        torch.tensor([0, 0]),
        torch.tensor([1, 1]),
        torch.tensor([-1, -1]),
        torch.tensor([0, 0]),
        torch.tensor([1, 1]),
    ]
    for value, options in enumerate(option_rows):
        replay.extend(_transition(value), options)
    return replay


def test_exact_replay_revocation_samples_only_student() -> None:
    replay = _filled_admission_replay()
    replay.set_admission_policy(
        admitted_sources=torch.tensor([False, False]),
        candidate_masses=torch.tensor([0.0, 0.0, 1.0]),
        uniform_mix=1.0,
    )
    batch = replay.sample(256)
    assert torch.all(batch["options"] == -1)


def test_startup_exact_abstention_uses_scratch_randint_stream() -> None:
    replay = _make_replay(capacity=8, n_env=2)
    for value in range(6):
        replay.extend(_transition(value), torch.tensor([-1, -1]))
    replay.set_admission_policy(
        admitted_sources=torch.tensor([False, False]),
        candidate_masses=torch.tensor([0.0, 0.0, 1.0]),
        uniform_mix=1.0,
    )
    torch.manual_seed(1234)
    actual = replay.draw_indices(batch_size=32)
    torch.manual_seed(1234)
    expected = torch.randint(0, 6, (2, 32))
    assert torch.equal(actual, expected)


def test_runtime_revocation_preserves_audit_data_but_removes_active_exposure() -> None:
    replay = _filled_admission_replay()
    replay.set_admission_policy(
        admitted_sources=torch.tensor([True, False]),
        candidate_masses=torch.tensor([0.8, 0.0, 0.2]),
        uniform_mix=1.0,
    )
    replay.sample(512)
    before = replay.admission_audit()
    assert before is not None and before["critic_sample_counts"][0] > 0
    replay.set_admission_policy(
        admitted_sources=torch.tensor([False, False]),
        candidate_masses=torch.tensor([0.0, 0.0, 1.0]),
        uniform_mix=1.0,
    )
    after_batch = replay.sample(1024)
    assert torch.all(after_batch["options"] == -1)
    after = replay.admission_audit()
    assert after is not None
    assert after["main_buffer_counts"][0] > 0  # retained only as immutable audit history
    assert after["active_buffer_counts"][0] == 0
    assert len(after["policy_events"]) == 2
    assert after["critic_sample_counts"][0] == before["critic_sample_counts"][0]
    assert after["policy_events"][1]["sample_counts_at_apply"]["critic"][0] == before[
        "critic_sample_counts"
    ][0]


def test_adaptive_exact_decision_updates_behavior_replay_and_authority_together() -> None:
    initial = build_admission_snapshot(
        mode="all",
        source_names=["source0", "source1"],
        source_logits=[0.0, 0.0],
        student_logit=0.0,
    )
    controller = AdaptiveAdmissionController(
        initial_snapshot=initial,
        stage_window_steps=10,
        min_segments=2,
        persistence=1,
    )
    controller.record_segments(
        candidate_ids=[0, 0, 1, 1, 2, 2],
        segment_mean_rewards=[-2.0, -2.0, -1.0, -1.0, 1.0, 1.0],
    )
    result = controller.maybe_close_window(10)
    assert result is not None and result.snapshot is not None
    assert result.snapshot.exact_abstain

    schema = h1hand_default_action_schema()
    masks = torch.stack(
        [schema.get("legs_torso").mask(schema.dim), schema.get("arms").mask(schema.dim)]
    )
    behavior = McgBehaviorController(
        num_envs=2,
        num_groups=2,
        device="cpu",
        group_masks=masks,
        warmup_mode="admission_bootstrap",
        bootstrap_weights=torch.tensor([0.0, 0.0]),
        bootstrap_horizons=torch.tensor([25, 25]),
        admitted_sources=torch.tensor([True, True]),
    )
    behavior.current.copy_(torch.tensor([[0, 0], [1, 1]]))
    behavior.current_arm.copy_(behavior.current)
    behavior.steps_left.fill_(12)

    replay = _filled_admission_replay()
    replay.set_admission_policy(
        admitted_sources=torch.tensor([True, True]),
        candidate_masses=torch.tensor([0.25, 0.25, 0.5]),
        uniform_mix=1.0,
    )
    replay.sample(512)
    before = replay.admission_audit()
    assert before is not None

    exact = result.snapshot
    behavior.set_admission_policy(
        admitted_sources=exact.admitted_tensor("cpu"),
        source_logits=torch.tensor(exact.source_logits),
        student_logit=exact.student_logit,
    )
    replay.set_admission_policy(
        admitted_sources=exact.admitted_tensor("cpu"),
        candidate_masses=exact.candidate_probabilities(tau=1.0),
        uniform_mix=1.0,
    )
    replay.set_admission_source_authority(False, reason="adaptive_exact_abstention")

    assert torch.all(behavior.current == -1)
    assert torch.all(behavior.steps_left == 0)
    assert torch.all(replay.sample(1024)["options"] == -1)
    after = replay.admission_audit()
    assert after is not None
    assert after["admitted_sources"] == [False, False]
    assert after["active_buffer_counts"][:2] == [0, 0]
    assert after["source_authority_active"] is False
    assert after["critic_sample_counts"][:2] == before["critic_sample_counts"][:2]


def test_mixed_mcg_transition_exits_when_any_contributing_source_is_revoked() -> None:
    replay = _make_replay(capacity=4, n_env=2)
    replay.enable_provenance(group_count=2)
    ranks = torch.arange(2, dtype=torch.int16)

    def provenance(source_by_group: torch.Tensor, segment: int) -> dict[str, torch.Tensor]:
        canonical = torch.where(
            source_by_group >= 0,
            source_by_group,
            torch.full_like(source_by_group, 2),
        ).min(dim=1).values
        return {
            "behavior_source": torch.where(
                canonical < 2, canonical, torch.full_like(canonical, -1)
            ).to(torch.int16),
            "source_by_group": source_by_group.to(torch.int16),
            "executed_group_mask": source_by_group >= 0,
            "segment_id": torch.full((2,), segment, dtype=torch.int64),
            "segment_step": torch.zeros(2, dtype=torch.int16),
            "anchor_id": torch.full((2,), -1, dtype=torch.int32),
            "env_rank": ranks,
            "learner_step": torch.full((2,), segment, dtype=torch.int64),
        }

    mixed = torch.tensor([[0, 1], [0, 1]])
    student = torch.full((2, 2), -1)
    replay.extend(_transition(0), torch.tensor([0, 0]), provenance=provenance(mixed, 0))
    replay.extend(
        _transition(1), torch.tensor([-1, -1]), provenance=provenance(student, 1)
    )
    replay.set_admission_policy(
        admitted_sources=torch.tensor([True, True]),
        candidate_masses=torch.tensor([0.4, 0.4, 0.2]),
        uniform_mix=1.0,
    )
    assert torch.any(replay.sample(512)["options"] == 0)
    replay.set_admission_policy(
        admitted_sources=torch.tensor([True, False]),
        candidate_masses=torch.tensor([0.8, 0.0, 0.2]),
        uniform_mix=1.0,
    )
    batch = replay.sample(1024)
    assert torch.all(batch["options"] == -1)
    audit = replay.admission_audit()
    assert audit is not None
    assert audit["main_buffer_counts"][0] == 2
    assert audit["active_buffer_counts"][0] == 0


def test_revoked_mixed_slot_stays_zero_with_allowed_same_canonical_stratum() -> None:
    """Uniform coverage must not resurrect a provenance-forbidden transition."""

    replay = _make_replay(capacity=4, n_env=1)
    replay.enable_provenance(group_count=2)

    def provenance(source_by_group: list[int], segment: int) -> dict[str, torch.Tensor]:
        groups = torch.tensor([source_by_group], dtype=torch.long)
        sentinel = torch.full_like(groups, 2)
        canonical = torch.where(groups >= 0, groups, sentinel).min(dim=1).values
        return {
            "behavior_source": torch.where(
                canonical < 2, canonical, torch.full_like(canonical, -1)
            ).to(torch.int16),
            "source_by_group": groups.to(torch.int16),
            "executed_group_mask": groups >= 0,
            "segment_id": torch.tensor([segment], dtype=torch.int64),
            "segment_step": torch.zeros(1, dtype=torch.int16),
            "anchor_id": torch.full((1,), -1, dtype=torch.int32),
            "env_rank": torch.zeros(1, dtype=torch.int16),
            "learner_step": torch.tensor([segment], dtype=torch.int64),
        }

    # Slots 0 and 1 share canonical source 0. Slot 0 also contains source 1,
    # so revoking source 1 must exclude only slot 0 while leaving slot 1 active.
    replay.extend(
        _transition(0, n_env=1),
        torch.tensor([0]),
        provenance=provenance([0, 1], 0),
    )
    replay.extend(
        _transition(1, n_env=1),
        torch.tensor([0]),
        provenance=provenance([0, 0], 1),
    )
    replay.extend(
        _transition(2, n_env=1),
        torch.tensor([-1]),
        provenance=provenance([-1, -1], 2),
    )
    replay.set_admission_policy(
        admitted_sources=torch.tensor([True, False]),
        candidate_masses=torch.tensor([0.8, 0.0, 0.2]),
        uniform_mix=1.0,
    )

    allowed = replay._admission_allowed_slots(replay.options[:, :3], 3)
    assert allowed.tolist() == [[False, True, True]]
    weights = replay._admission_slot_weights(3)
    assert weights[0, 0].item() == 0.0
    torch.testing.assert_close(weights.sum(dim=1), torch.ones(1))

    torch.manual_seed(1234)
    indices = replay.draw_indices(batch_size=5000)
    assert not torch.any(indices == 0)
    source_share = float((indices == 1).float().mean())
    assert abs(source_share - 0.8) < 0.03


def test_admission_replay_quota_is_independent_of_historical_counts() -> None:
    replay = _filled_admission_replay()
    replay.set_admission_policy(
        admitted_sources=torch.tensor([True, False]),
        candidate_masses=torch.tensor([0.7, 0.0, 0.3]),
        uniform_mix=1.0,
    )
    batch = replay.sample(5000)
    options = batch["options"]
    assert not torch.any(options == 1)
    source_share = float((options == 0).float().mean())
    assert abs(source_share - 0.7) < 0.03


def test_authority_handoff_matches_legacy_randint_when_all_slots_allowed() -> None:
    replay = _filled_admission_replay()
    replay.set_admission_policy(
        admitted_sources=torch.tensor([True, True]),
        candidate_masses=torch.tensor([0.8, 0.1, 0.1]),
        uniform_mix=1.0,
    )
    replay.set_admission_source_authority(
        False, reason="post_warmup_behavior_phase"
    )
    torch.manual_seed(2468)
    actual = replay.draw_indices(batch_size=128)
    torch.manual_seed(2468)
    expected = torch.randint(0, 6, (2, 128))
    assert torch.equal(actual, expected)
    audit = replay.admission_audit()
    assert audit is not None
    assert audit["sampling_phase"] == "physical_allowed"
    assert audit["policy_events"][-1]["event"] == "source_authority"


def test_authority_handoff_is_physical_uniform_but_keeps_exact_revoke() -> None:
    replay = _filled_admission_replay()
    replay.set_admission_policy(
        admitted_sources=torch.tensor([True, False]),
        candidate_masses=torch.tensor([0.9, 0.0, 0.1]),
        uniform_mix=1.0,
    )
    replay.set_admission_source_authority(False)
    batch = replay.sample(5000)
    options = batch["options"]
    assert not torch.any(options == 1)
    # Two source-0 and two student slots are physically active per env.
    source_share = float((options == 0).float().mean())
    assert abs(source_share - 0.5) < 0.03
    audit = replay.admission_audit()
    assert audit is not None
    assert audit["effective_replay_masses"] == [0.5, 0.0, 0.5]


def test_exact_none_randint_is_unchanged_by_authority_handoff() -> None:
    replay = _make_replay(capacity=8, n_env=2)
    for value in range(6):
        replay.extend(_transition(value), torch.tensor([-1, -1]))
    replay.set_admission_policy(
        admitted_sources=torch.tensor([False, False]),
        candidate_masses=torch.tensor([0.0, 0.0, 1.0]),
        uniform_mix=1.0,
    )
    replay.set_admission_source_authority(False)
    torch.manual_seed(97531)
    actual = replay.draw_indices(batch_size=64)
    torch.manual_seed(97531)
    expected = torch.randint(0, 6, (2, 64))
    assert torch.equal(actual, expected)


def test_recency_and_priority_only_change_within_active_stratum() -> None:
    replay = _filled_admission_replay()
    replay.set_admission_policy(
        admitted_sources=torch.tensor([True, False]),
        candidate_masses=torch.tensor([0.5, 0.0, 0.5]),
        recency_half_life=1.0,
        uniform_mix=0.0,
        priority_alpha=1.0,
    )
    # Source-0 physical slots are 1 and 4. Make old slot 1 very high priority;
    # priority should compete with recency only inside source-0, not alter its
    # total 0.5 quota.
    indices = torch.tensor([[1], [1]])
    replay.update_priorities(indices, torch.tensor([[100.0], [100.0]]))
    batch = replay.sample(6000)
    source_share = float((batch["options"] == 0).float().mean())
    assert abs(source_share - 0.5) < 0.03
    assert "replay_indices" in batch.keys()


def test_admission_sampling_snapshot_roundtrip() -> None:
    replay = _filled_admission_replay()
    replay.set_admission_policy(
        admitted_sources=torch.tensor([True, False]),
        candidate_masses=torch.tensor([0.6, 0.0, 0.4]),
        recency_half_life=3.0,
        uniform_mix=0.1,
        priority_alpha=0.5,
    )
    replay.set_admission_source_authority(False, reason="snapshot_test")
    replay.set_admission_replay_physical(True)
    snapshot = replay.export_valid()
    restored = _make_replay(capacity=8, n_env=2)
    restored.import_valid(snapshot)
    restored_snapshot = restored.export_valid()
    left = snapshot["admission_sampling"]
    right = restored_snapshot["admission_sampling"]
    assert left is not None and right is not None
    for key in ("source_mask", "candidate_masses", "priorities", "slot_write_step"):
        assert torch.equal(left[key], right[key]), key
    assert left["source_authority_active"] is False
    assert right["source_authority_active"] is False
    assert left["replay_physical"] is True
    assert right["replay_physical"] is True
    assert restored.admission_replay_physical is True
    assert replay.admission_audit()["sampling_phase"] == "physical_allowed"
    assert restored.admission_audit()["sampling_phase"] == "physical_allowed"


def test_replay_physical_audit_reports_physical_sampling_while_authority_active() -> None:
    replay = _filled_admission_replay()
    replay.set_admission_policy(
        admitted_sources=torch.tensor([True, False]),
        candidate_masses=torch.tensor([0.9, 0.0, 0.1]),
        uniform_mix=1.0,
    )
    replay.set_admission_replay_physical(True)

    audit = replay.admission_audit()
    assert audit is not None
    assert audit["source_authority_active"] is True
    assert audit["replay_physical"] is True
    assert audit["sampling_phase"] == "physical_allowed"
    # Source-0 and student each occupy two active slots per environment;
    # source-1 is rejected even though it remains in physical history.
    assert audit["effective_replay_masses"] == [0.5, 0.0, 0.5]


def test_displacement_summary_keeps_endpoint_and_cumulative_timebases_separate() -> None:
    replay = _make_replay(capacity=8, n_env=2)
    replay.enable_provenance(group_count=4)
    for value in range(4):
        replay.extend(
            _transition(value),
            torch.tensor([-1, 0]),
            provenance=_provenance(0),
        )
    replay.set_admission_policy(
        admitted_sources=torch.tensor([True]),
        candidate_masses=torch.tensor([0.5, 0.5]),
        uniform_mix=1.0,
    )
    replay.set_admission_replay_physical(True)
    replay.sample(4096)

    summary = replay.replay_displacement_summary()
    assert summary["rho_endpoint"] == 0.5
    assert summary["rho_endpoint_by_group"] == [0.5, 0.5, 0.5, 0.5]
    assert abs(summary["q_cumulative"] - 0.5) < 0.03
    assert abs(summary["cohort_exposure_ratio"] - 1.0) < 0.06
    assert "rho_S" not in summary
    assert "q_S" not in summary
    assert "A" not in summary
