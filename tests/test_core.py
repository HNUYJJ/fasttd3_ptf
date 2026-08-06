"""legacy_actors(旧格式 checkpoint 加载所需的 actor 架构)与蒸馏损失的基础测试。

原 test_core.py 测试 my_fasttd3_ptf 的模块化实现;该路径于 2026-06-10 移除后,
本文件只保留仍被主路径使用的部分:legacy_actors(SourcePolicy 加载旧 checkpoint
时重建网络)与 masked_action_distillation_loss。
"""
import torch

from fasttd3_ptf.ptf.legacy_actors import Actor, UpstreamFastTD3Actor
from fasttd3_ptf.ptf.distillation import masked_action_distillation_loss


def test_legacy_actor_shapes_and_export_kwargs():
    actor = Actor(4, 2, hidden_dims=[16, 16])
    obs = torch.zeros(5, 4)
    act = actor(obs)
    assert act.shape == (5, 2)
    kw = actor.export_kwargs()
    assert kw["obs_dim"] == 4 and kw["action_dim"] == 2
    # export_kwargs 必须能重建同构网络(SourcePolicy 加载路径的前提)
    actor2 = Actor(kw["obs_dim"], kw["action_dim"], hidden_dims=kw["hidden_dims"],
                   action_low=kw["action_low"], action_high=kw["action_high"])
    actor2.load_state_dict(actor.state_dict())
    assert torch.allclose(actor(obs), actor2(obs))


def test_upstream_fasttd3_actor_shapes_and_noise_reset():
    actor = UpstreamFastTD3Actor(4, 2, num_envs=5, hidden_dim=16, init_scale=0.01)
    obs = torch.zeros(5, 4)
    dones = torch.tensor([True, False, False, True, False])
    old_scales = actor.noise_scales.clone()
    act = actor.explore(obs, dones=dones, deterministic=False)
    assert act.shape == (5, 2)
    assert not torch.equal(old_scales[dones], actor.noise_scales[dones])
    assert actor.clamp_action(act).abs().max() <= 1.0
    assert actor.export_kwargs()["model_class"] == "UpstreamFastTD3Actor"


def test_distillation_loss():
    a = torch.zeros(4, 2)
    b = torch.ones(4, 2)
    mask = torch.tensor([1.0, 0.0])
    loss = masked_action_distillation_loss(a, b, mask)
    assert loss.shape == (4,)
    assert torch.all(loss > 0)
