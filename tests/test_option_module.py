import torch
from torch import nn

from fasttd3_ptf.ptf.option_module import OptionModule, straight_through_clamp
from fasttd3_ptf.ptf.option_update import (
    compatible_option_q_loss,
    option_td_target,
    option_u_value,
    released_code_option_u_value,
    select_termination_batch,
    termination_loss,
    termination_loss_at_next_state,
)
from fasttd3_ptf.utils.schedules import ReleasedPTFTanhScheduler


def test_select_termination_batch_restores_current_ptf_transition() -> None:
    replay = (
        torch.full((3, 2), 1.0),
        torch.tensor([0, 0, 0]),
        torch.tensor([True, True, True]),
    )
    current = (
        torch.full((2, 2), 7.0),
        torch.tensor([1, 0]),
        torch.tensor([True, False]),
    )

    selected = select_termination_batch(
        "current_transition",
        replay_next_obs=replay[0],
        replay_option_ids=replay[1],
        replay_valid=replay[2],
        current_transition=current,
    )

    assert selected is current
    assert torch.equal(selected[0], current[0])
    assert torch.equal(selected[1], current[1])
    assert torch.equal(selected[2], current[2])


def test_select_termination_batch_replay_is_backward_compatible() -> None:
    replay_obs = torch.randn(4, 3)
    replay_ids = torch.tensor([0, 1, 0, 1])
    replay_valid = torch.tensor([True, True, False, True])

    selected = select_termination_batch(
        "replay",
        replay_next_obs=replay_obs,
        replay_option_ids=replay_ids,
        replay_valid=replay_valid,
    )

    assert selected[0] is replay_obs
    assert selected[1] is replay_ids
    assert selected[2] is replay_valid
from fasttd3_ptf.ptf.option_selector import OptionSelector


def test_option_module_shapes():
    module = OptionModule(obs_dim=5, num_options=3, hidden_dims=[16])
    q, beta = module(torch.randn(7, 5))
    assert q.shape == (7, 3)
    assert beta.shape == (7, 3)
    assert torch.all(beta >= 0) and torch.all(beta <= 1)


def test_straight_through_logit_clamp_bounds_forward_but_not_gradient() -> None:
    logits = torch.tensor([-20.0, 0.0, 20.0], requires_grad=True)

    bounded = straight_through_clamp(logits, 4.0)
    bounded.sum().backward()

    assert torch.equal(bounded.detach(), torch.tensor([-4.0, 0.0, 4.0]))
    assert torch.equal(logits.grad, torch.ones_like(logits))


def test_option_module_exports_beta_logit_clip() -> None:
    module = OptionModule(
        obs_dim=5,
        num_options=2,
        hidden_dims=[8],
        beta_logit_clip=4.0,
    )
    kwargs = module.export_kwargs()

    assert kwargs["beta_logit_clip"] == 4.0
    restored = OptionModule(**kwargs)
    restored.load_state_dict(module.state_dict(), strict=True)


def test_option_update_helpers():
    q = torch.tensor([[1.0, 2.0]])
    beta = torch.tensor([[0.0, 1.0]])
    u = option_u_value(q, beta)
    assert torch.allclose(u, torch.tensor([[1.0, 2.0]]))
    loss = termination_loss(q.repeat(3, 1), beta.repeat(3, 1), torch.zeros(3, dtype=torch.long))
    assert loss.ndim == 0


def test_termination_loss_xi_zero_uses_adaptive_margin():
    q = torch.tensor([[2.0, 1.0, 0.0], [2.0, 1.0, 0.0]])
    beta = torch.ones_like(q, requires_grad=True)

    loss = termination_loss(q, beta, torch.tensor([0, 1]), xi=0.0)
    loss.backward()

    assert beta.grad is not None
    assert beta.grad[0, 0] > 0.0
    assert beta.grad[1, 1] < 0.0


def test_option_selector_does_not_consume_global_torch_rng():
    class StaticOption(torch.nn.Module):
        def forward(self, obs):
            q = torch.tensor([[1.0, 0.0]], device=obs.device).expand(obs.shape[0], 2)
            beta = torch.full_like(q, 0.5)
            return q, beta

    torch.manual_seed(123)
    expected_next = torch.rand(1)
    torch.manual_seed(123)
    selector = OptionSelector(num_envs=1, num_options=2, device=torch.device("cpu"), seed=99)
    selector.step(torch.zeros(1, 3), StaticOption(), epsilon=0.5)
    actual_next = torch.rand(1)

    assert torch.allclose(actual_next, expected_next)


def test_termination_loss_at_next_state_uses_next_observation_gradient():
    class StateSensitiveOption(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.logit = torch.nn.Parameter(torch.tensor(0.0))

        def forward(self, obs):
            q_selected = obs[:, :1]
            q_other = torch.zeros_like(q_selected)
            q = torch.cat([q_selected, q_other], dim=1)
            beta = torch.sigmoid(self.logit).expand(obs.shape[0], 2)
            return q, beta

    module = StateSensitiveOption()
    next_obs = torch.tensor([[-1.0]])
    loss = termination_loss_at_next_state(
        module,
        next_obs,
        option_ids=torch.zeros(1, dtype=torch.long),
        xi=0.01,
    )
    loss.backward()
    assert module.logit.grad is not None
    assert module.logit.grad.item() < 0.0


def test_option_module_beta_is_clamped_to_min_max():
    """beta output must stay within [beta_min, beta_max] regardless of input.

    Sigmoid saturation at the (0,1) rails kills the (1-beta) transfer gate
    and zeros out option gradients via vanishing dsigmoid/dz. Clamping the
    output range keeps beta recoverable.
    """
    torch.manual_seed(0)
    module = OptionModule(obs_dim=4, num_options=3, hidden_dims=[8],
                          beta_min=0.1, beta_max=0.9)
    # Drive the beta_head logits to extreme values via large-magnitude inputs.
    for scale in [1.0, 10.0, 1000.0, -1000.0]:
        obs = scale * torch.randn(64, 4)
        _, beta = module(obs)
        assert torch.all(beta >= 0.1 - 1e-6), f"beta below clamp at scale={scale}: min={beta.min().item()}"
        assert torch.all(beta <= 0.9 + 1e-6), f"beta above clamp at scale={scale}: max={beta.max().item()}"


def test_option_module_beta_clamp_default_range():
    """Default clamp is [0.05, 0.95]; export_kwargs should include both bounds."""
    module = OptionModule(obs_dim=3, num_options=2, hidden_dims=[4])
    kw = module.export_kwargs()
    assert kw["beta_min"] == 0.05
    assert kw["beta_max"] == 0.95
    # Reconstruct from export_kwargs should produce an equivalent module.
    module2 = OptionModule(**kw)
    assert module2.beta_min == 0.05
    assert module2.beta_max == 0.95


def test_released_code_fidelity_module_restores_author_architecture() -> None:
    torch.manual_seed(9)
    module = OptionModule(
        obs_dim=5,
        num_options=3,
        hidden_dims=[20],
        released_code_fidelity=True,
    )
    obs = torch.randn(17, 5)
    q, beta = module(obs)

    assert isinstance(module.trunk[1], nn.ReLU6)
    assert torch.all(q >= -1.0) and torch.all(q <= 1.0)
    assert torch.all(beta >= 0.0) and torch.all(beta <= 1.0)
    assert torch.equal(module.trunk[0].bias, torch.zeros_like(module.trunk[0].bias))
    assert torch.equal(module.q_head.bias, torch.zeros_like(module.q_head.bias))
    assert torch.equal(module.beta_head.bias, torch.zeros_like(module.beta_head.bias))
    assert module.export_kwargs()["released_code_fidelity"] is True


def test_released_code_fidelity_rejects_hybrid_network_depth() -> None:
    try:
        OptionModule(
            obs_dim=5,
            num_options=3,
            hidden_dims=[20, 20],
            released_code_fidelity=True,
        )
    except ValueError as exc:
        assert "exactly one hidden layer" in str(exc)
    else:
        raise AssertionError("hybrid released-code network depth was accepted")


def test_released_code_option_target_uses_online_argmax_target_value() -> None:
    q_online = torch.tensor([[10.0, 0.0], [0.0, 10.0]])
    beta_online = torch.ones_like(q_online)
    # Target maxima deliberately disagree with online greedy choices.
    q_target = torch.tensor([[1.0, 9.0], [9.0, 2.0]])

    u = released_code_option_u_value(q_online, beta_online, q_target)

    assert torch.equal(u, torch.tensor([[1.0, 1.0], [2.0, 2.0]]))


def test_option_reward_scale_bounds_hurdle_tanh_q_target() -> None:
    rewards = torch.tensor([0.0, 1.0])
    bootstrap = torch.ones(2, 1)
    u_next = torch.tensor([[-1.0, 1.0], [-1.0, 1.0]])

    scaled = option_td_target(
        rewards,
        gamma=0.99,
        bootstrap=bootstrap,
        u_next=u_next,
        reward_scale=0.01,
    )
    unscaled = option_td_target(
        rewards,
        gamma=0.99,
        bootstrap=bootstrap,
        u_next=u_next,
    )

    assert scaled.min() >= -1.0
    assert scaled.max() <= 1.0
    assert unscaled.max() > 1.0


def test_option_reward_scale_rejects_nonpositive_values() -> None:
    try:
        option_td_target(
            torch.ones(1),
            gamma=0.99,
            bootstrap=torch.ones(1, 1),
            u_next=torch.ones(1, 2),
            reward_scale=0.0,
        )
    except ValueError as exc:
        assert "must be positive" in str(exc)
    else:
        raise AssertionError("nonpositive option reward scale was accepted")


def test_released_code_mixed_source_q_loss_sums_compatible_options() -> None:
    q = torch.zeros(2, 3)
    target = torch.ones(2, 3)
    compatibility = torch.tensor(
        [
            [1.0, 0.5, 0.0],
            [0.0, 1.0, 1.0],
        ]
    )

    released = compatible_option_q_loss(
        q,
        target,
        compatibility,
        released_code_reduction=True,
    )
    legacy = compatible_option_q_loss(q, target, compatibility)

    assert torch.allclose(released, torch.tensor(1.75))
    assert torch.allclose(legacy, torch.tensor(1.0))


def test_released_code_termination_keeps_deep_non_greedy_advantage() -> None:
    q = torch.tensor([[10.0, 9.0, -10.0]])
    beta = torch.ones_like(q)
    option = torch.tensor([2])

    released = termination_loss(
        q,
        beta,
        option,
        xi=0.0,
        clamp_advantage=False,
    )
    adapted = termination_loss(
        q,
        beta,
        option,
        xi=0.0,
        clamp_advantage=True,
    )

    assert torch.allclose(released, torch.tensor(-19.2))
    assert torch.allclose(adapted, torch.tensor(-0.8))


def test_released_ptf_tanh_schedule_maps_full_budget() -> None:
    schedule = ReleasedPTFTanhScheduler(scale=2.0, duration=100)

    assert schedule(0) > 1.99
    assert 0.99 < schedule(50) < 1.01
    assert schedule(100) < 0.01
    assert schedule(1000) == schedule(100)
