import torch
import pytest

from fasttd3_ptf.ptf.action_schema import h1hand_default_action_schema
from fasttd3_ptf.ptf.adapters import build_action_adapter, build_action_mask, build_obs_adapter


def test_identity_obs_adapter():
    obs = torch.randn(4, 6)
    adapter = build_obs_adapter({"type": "identity"}, source_obs_dim=6)
    assert torch.allclose(adapter(obs), obs)


def test_indices_obs_adapter():
    obs = torch.arange(24, dtype=torch.float32).view(4, 6)
    adapter = build_obs_adapter({"type": "indices", "indices": [0, 2, 5]}, source_obs_dim=3)
    out = adapter(obs)
    assert out.shape == (4, 3)
    assert torch.allclose(out[:, 1], obs[:, 2])


def test_humanoidbench_full_state_robot_obs_adapter():
    obs = torch.arange(2 * 155, dtype=torch.float32).view(2, 155)
    adapter = build_obs_adapter(
        {"type": "humanoidbench_robot_qpos_qvel", "qpos_dim": 78, "robot_dof": 76},
        source_obs_dim=151,
    )
    out = adapter(obs)
    assert out.shape == (2, 151)
    assert torch.allclose(out[:, :76], obs[:, :76])
    assert torch.allclose(out[:, 76:], obs[:, 78:153])


def test_action_pad_adapter_and_mask():
    adapter = build_action_adapter({"type": "pad", "source_indices": [0, 1], "target_indices": [0, 2]}, target_action_dim=4)
    src = torch.tensor([[1.0, -1.0]])
    out = adapter(src)
    assert out.shape == (1, 4)
    assert out[0, 0] == 1.0 and out[0, 2] == -1.0
    mask = build_action_mask({"type": "indices", "indices": [0, 2]}, 4, torch.device("cpu"))
    assert mask.tolist() == [1.0, 0.0, 1.0, 0.0]


def test_passthrough_adapters_reject_implicit_shape_changes():
    obs = torch.randn(2, 6)
    obs_adapter = build_obs_adapter({"type": "identity", "output_dim": 4}, source_obs_dim=4)
    with pytest.raises(ValueError, match="explicit"):
        obs_adapter(obs)

    action_adapter = build_action_adapter({"type": "passthrough", "output_dim": 4}, target_action_dim=4)
    with pytest.raises(ValueError, match="action_pad"):
        action_adapter(torch.randn(2, 2))


def test_h1hand_group_action_mask():
    mask = build_action_mask(
        {"type": "groups", "schema": "h1hand_default", "groups": ["legs_torso", "right_arm"]},
        61,
        torch.device("cpu"),
    )
    assert mask[:11].sum().item() == 11
    assert mask[16:21].sum().item() == 5
    assert mask[21:].sum().item() == 0


def test_h1hand_schema_matches_humanoidbench_h1hand_pos_xml_order():
    schema = h1hand_default_action_schema()
    expected = {
        "legs": (0, 10),
        "torso": (10, 11),
        "legs_torso": (0, 11),
        "left_arm": (11, 16),
        "right_arm": (16, 21),
        "left_hand": (21, 41),
        "right_hand": (41, 61),
    }
    for name, bounds in expected.items():
        sl = schema.get(name)
        assert (sl.start, sl.end) == bounds
