from __future__ import annotations

import sys
from pathlib import Path

import torch
from tensordict import TensorDict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fasttd3_ptf.official_fasttd3_ptf.admission_control import (
    build_admission_snapshot,
)
from fasttd3_ptf.official_fasttd3_ptf.paths import ensure_fasttd3_import_path
from fasttd3_ptf.official_fasttd3_ptf.ptf_replay import PTFReplayWrapper
from fasttd3_ptf.official_fasttd3_ptf.train_ptf import (
    apply_runtime_admission_policy_after_resume,
)
from fasttd3_ptf.ptf.mcg import McgBehaviorController

ensure_fasttd3_import_path()
from fast_td3_utils import SimpleReplayBuffer  # type: ignore  # noqa: E402


def _replay() -> PTFReplayWrapper:
    return PTFReplayWrapper(
        SimpleReplayBuffer(
            n_env=2,
            buffer_size=12,
            n_obs=3,
            n_act=2,
            n_critic_obs=3,
            n_steps=1,
            device=torch.device("cpu"),
        )
    )


def _transition(value: int) -> TensorDict:
    obs = torch.full((2, 3), float(value))
    return TensorDict(
        {
            "observations": obs,
            "actions": torch.full((2, 2), value / 10.0),
            "next": {
                "rewards": torch.full((2,), float(value)),
                "dones": torch.zeros(2, dtype=torch.long),
                "truncations": torch.zeros(2, dtype=torch.long),
                "observations": obs + 1,
            },
        },
        batch_size=2,
    )


def _behavior() -> McgBehaviorController:
    return McgBehaviorController(
        num_envs=2,
        num_groups=1,
        device="cpu",
        group_masks=torch.ones(1, 2, dtype=torch.bool),
        warmup_mode="admission_bootstrap",
        bootstrap_weights=torch.tensor([0.0]),
        bootstrap_horizons=torch.tensor([25]),
        admitted_sources=torch.tensor([True]),
        admission_student_logit=0.0,
    )


def test_runtime_policy_overrides_imported_branch_policy_and_releases_latch() -> None:
    branch = _replay()
    for step in range(6):
        # Both strata are physically present for every vector-env row, matching
        # a mature 0.5/0.5 intervention branch rather than a tiny degenerate toy.
        options = torch.tensor([0, -1]) if step % 2 == 0 else torch.tensor([-1, 0])
        branch.extend(_transition(step), options)
    branch.set_admission_policy(
        admitted_sources=torch.tensor([True]),
        candidate_masses=torch.tensor([0.9, 0.1]),
        uniform_mix=1.0,
    )
    torch.manual_seed(7)
    branch.sample(512, role="critic")
    before = branch.admission_audit()
    assert before is not None
    assert before["critic_sample_counts"][0] > 0

    restored = _replay()
    restored.import_valid(branch.export_valid())
    behavior = _behavior()
    behavior.current.fill_(0)
    behavior.current_arm.fill_(0)
    behavior.steps_left.fill_(17)
    exact_none = build_admission_snapshot(
        mode="none",
        source_names=["walk"],
        source_logits=[0.0],
        student_logit=0.0,
    )
    apply_runtime_admission_policy_after_resume(
        replay=restored,
        behavior=behavior,
        snapshot=exact_none,
        device="cpu",
        bootstrap_tau=1.0,
        replay_mode="shared",
        recency_half_life=0.0,
        uniform_mix=1.0,
        priority_alpha=0.0,
    )

    # Runtime exact abstention, not the imported source policy, is authoritative.
    assert torch.all(behavior.current == -1)
    assert torch.all(behavior.current_arm == -1)
    assert torch.all(behavior.steps_left == 0)
    audit = restored.admission_audit()
    assert audit is not None
    assert audit["main_buffer_counts"][0] > 0
    assert audit["active_buffer_counts"][0] == 0
    assert audit["candidate_masses"] == [0.0, 1.0]
    source_samples_before = audit["critic_sample_counts"][0]
    batch = restored.sample(1024, role="critic")
    assert torch.all(batch["options"] == -1)
    after = restored.admission_audit()
    assert after is not None
    assert after["critic_sample_counts"][0] == source_samples_before


def test_runtime_shared_policy_preserves_source_authority() -> None:
    replay = _replay()
    for step in range(4):
        replay.extend(_transition(step), torch.tensor([0, -1]))
    behavior = _behavior()
    admit_all = build_admission_snapshot(
        mode="all",
        source_names=["walk"],
        source_logits=[0.0],
        student_logit=0.0,
    )
    masses = apply_runtime_admission_policy_after_resume(
        replay=replay,
        behavior=behavior,
        snapshot=admit_all,
        device="cpu",
        bootstrap_tau=1.0,
        replay_mode="shared",
        recency_half_life=0.0,
        uniform_mix=1.0,
        priority_alpha=0.0,
    )
    torch.testing.assert_close(masses, torch.tensor([0.5, 0.5]))
    audit = replay.admission_audit()
    assert audit is not None
    assert audit["admitted_sources"] == [True]
    assert audit["active_buffer_counts"][0] > 0
