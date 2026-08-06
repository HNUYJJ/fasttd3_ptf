"""SourcePolicy 旧格式 checkpoint 加载与 source-bank builder 推断的契约测试。

旧格式 = `FastTD3Agent.state_dict.v1`(已移除的 my_fasttd3_ptf 路径产出,
`checkpoints/sources/*/final.pt` 等仍在使用)。这里手工构造与该格式同构的
checkpoint dict,验证 SourcePolicy 的两条加载分支(默认 "Actor" 与
"UpstreamFastTD3Actor")依然工作。
"""
from pathlib import Path

import torch

from fasttd3_ptf.ptf.legacy_actors import Actor, UpstreamFastTD3Actor
from fasttd3_ptf.ptf.source_policy import SourcePolicy
from fasttd3_ptf.source_bank.builder import build_source_bank
from fasttd3_ptf.utils.checkpoint import save_json, save_torch


def _save_legacy_ckpt(path: Path, actor) -> None:
    # 与 FastTD3Agent.state_dict.v1 中 SourcePolicy 实际读取的键同构
    save_torch(
        {
            "agent_type": "FastTD3Agent",
            "actor_kwargs": actor.export_kwargs(),
            "actor_state_dict": actor.state_dict(),
        },
        path,
    )


def test_source_policy_loads_legacy_actor_checkpoint(tmp_path: Path):
    device = torch.device("cpu")
    actor = Actor(4, 2, hidden_dims=[8])
    ckpt = tmp_path / "source.pt"
    _save_legacy_ckpt(ckpt, actor)
    src = SourcePolicy("src", ckpt, device=device, target_action_dim=2,
                       source_obs_dim=4, source_action_dim=2, actor_hidden_dims=[8])
    action = src.act(torch.zeros(3, 4))
    assert action.shape == (3, 2)
    assert not action.requires_grad


def test_source_policy_loads_upstream_fasttd3_checkpoint(tmp_path: Path):
    device = torch.device("cpu")
    actor = UpstreamFastTD3Actor(4, 2, num_envs=4, hidden_dim=16, init_scale=0.01)
    ckpt = tmp_path / "upstream_source.pt"
    _save_legacy_ckpt(ckpt, actor)
    src = SourcePolicy("src", ckpt, device=device, target_action_dim=2,
                       source_obs_dim=4, source_action_dim=2)
    action = src.act(torch.zeros(3, 4))
    assert action.shape == (3, 2)
    assert not action.requires_grad


def test_source_bank_builder_infers_h1hand_masks(tmp_path: Path):
    manifest = {
        "name": "h1hand_walk",
        "env_id": "h1hand-walk-v0",
        "checkpoint": "checkpoints/sources/h1hand_walk/final.pt",
        "obs_dim": 151,
        "action_dim": 61,
        "actor_hidden_dims": [64, 64],
        "obs_adapter": {"type": "identity", "output_dim": 151},
        "action_adapter": {"type": "passthrough", "output_dim": 61},
        "action_mask": {"type": "full"},
    }
    manifest_path = tmp_path / "manifest.json"
    output_path = tmp_path / "bank.yaml"
    save_json(manifest, manifest_path)
    cfg = build_source_bank([str(manifest_path)], str(output_path))
    source = cfg["sources"][0]
    assert source["obs_adapter"]["type"] == "robot_only"
    assert source["action_mask"]["type"] == "groups"
    assert source["action_mask"]["groups"] == ["legs_torso"]


def test_source_bank_builder_infers_h1hand_reach_mask(tmp_path: Path):
    manifest = {
        "name": "h1hand_reach",
        "env_id": "h1hand-reach-v0",
        "checkpoint": "checkpoints/sources/h1hand_reach/final.pt",
        "obs_dim": 157,
        "action_dim": 61,
        "actor_hidden_dims": [64, 64],
        "obs_adapter": {"type": "identity", "output_dim": 157},
        "action_adapter": {"type": "passthrough", "output_dim": 61},
        "action_mask": {"type": "full"},
    }
    manifest_path = tmp_path / "manifest.json"
    output_path = tmp_path / "bank.yaml"
    save_json(manifest, manifest_path)
    cfg = build_source_bank([str(manifest_path)], str(output_path))
    source = cfg["sources"][0]
    assert source["obs_adapter"]["type"] == "reach"
    assert source["action_mask"]["groups"] == ["arms", "hands"]
