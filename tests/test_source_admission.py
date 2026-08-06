from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fasttd3_ptf.official_fasttd3_ptf.source_admission import validate_quarantine_bank

def _path(reward, done=None, *, head_height=None, torso_upright=None):
    reward = torch.as_tensor(reward, dtype=torch.float32)
    n, t = reward.shape
    if done is None:
        done = torch.zeros(n, t, dtype=torch.bool)
    path = {
        "observations": torch.zeros(n, t, 3),
        "actions": torch.zeros(n, t, 2),
        "rewards": reward,
        "dones": torch.as_tensor(done, dtype=torch.bool),
        "truncations": torch.zeros(n, t, dtype=torch.bool),
        "active": torch.ones(n, t, dtype=torch.bool),
        "next_observations": torch.zeros(n, t, 3),
        "provenance": {
            "behavior_source": torch.full((n, t), -1, dtype=torch.int16),
            "source_by_group": torch.full((n, t, 4), -1, dtype=torch.int16),
            "executed_group_mask": torch.zeros(n, t, 4, dtype=torch.bool),
            "segment_id": torch.arange(n).view(-1, 1).expand(n, t),
            "segment_step": torch.arange(t).view(1, -1).expand(n, t),
            "anchor_id": torch.arange(n).view(-1, 1).expand(n, t),
            "learner_step": torch.zeros(n, t, dtype=torch.int64),
        },
    }
    if head_height is not None or torso_upright is not None:
        path["diagnostics"] = {
            "head_height": torch.full((n, t), float(head_height or 1.5)),
            "torso_upright": torch.full((n, t), float(torso_upright or 0.9)),
        }
    return path


def _bank(source_reward, *, source_done=None):
    n, t = 64, 4
    baseline = torch.zeros(n, t)
    null = torch.zeros(n, t)
    null[:, 0] = torch.linspace(-0.05, 0.05, n)
    null[:, 2] = torch.linspace(0.04, -0.04, n)
    return {
        "schema_version": 1,
        "metadata": {
            "quarantine_only": True,
            "learner_updates": 0,
            "main_replay_writes": 0,
            "valid_anchors": n,
            "source_horizon": 2,
            "followup_horizon": 2,
            "provenance_groups": ["g0", "g1", "g2", "g3"],
        },
        "paths": {
            "student": _path(baseline),
            "student_duplicate": _path(baseline.clone()),
            "student_null": _path(null),
        },
        "sources": {"run": _path(source_reward, done=source_done)},
    }



def test_duplicate_mismatch_is_rejected():
    bank = _bank(torch.ones(64, 4))
    bank["paths"]["student_duplicate"]["actions"][0, 0, 0] = 1.0
    with pytest.raises(ValueError, match="duplicate mismatch"):
        validate_quarantine_bank(bank)

def test_valid_quarantine_bank_passes():
    validate_quarantine_bank(_bank(torch.ones(64, 4)))
