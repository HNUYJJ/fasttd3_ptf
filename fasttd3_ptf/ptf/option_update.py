from __future__ import annotations

import torch


def select_termination_batch(
    mode: str,
    *,
    replay_next_obs: torch.Tensor,
    replay_option_ids: torch.Tensor,
    replay_valid: torch.Tensor,
    current_transition: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select the data source used to train the PTF termination head.

    The released PTF implementation trains ``Q_omega`` from replay, but trains
    termination ``beta`` on the just-observed transition and the option that
    produced it.  ``replay`` preserves this project's historical adaptation;
    ``current_transition`` restores the released PTF sampling semantics while
    leaving the replay-based option-value update unchanged.
    """
    if mode == "replay":
        return replay_next_obs, replay_option_ids, replay_valid
    if mode == "current_transition":
        if current_transition is None:
            raise ValueError(
                "current_transition beta update requires the latest "
                "(next_obs, option_ids, valid) rollout tuple"
            )
        return current_transition
    raise ValueError(
        f"unknown beta update mode {mode!r}; expected 'replay' or 'current_transition'"
    )


def option_u_value(q_next: torch.Tensor, beta_next: torch.Tensor) -> torch.Tensor:
    """PTF 的 U 值:延续概率(1-β)保留当前 option 的 Q,β 概率切到最优 option。"""
    max_q = q_next.max(dim=1, keepdim=True).values
    return (1.0 - beta_next) * q_next + beta_next * max_q


def released_code_option_u_value(
    q_next_online: torch.Tensor,
    beta_next_online: torch.Tensor,
    q_next_target: torch.Tensor,
) -> torch.Tensor:
    """Author-released PTF target: online β/argmax with target-network values.

    The released TensorFlow implementation chooses the greedy option with the
    online network, evaluates that choice with the target network, and uses the
    online termination probability for the continuation/switch mixture.
    """
    if q_next_online.shape != beta_next_online.shape:
        raise ValueError(
            "online option values and termination probabilities must have "
            f"the same shape, got {q_next_online.shape} and {beta_next_online.shape}"
        )
    if q_next_target.shape != q_next_online.shape:
        raise ValueError(
            "online and target option values must have the same shape, got "
            f"{q_next_online.shape} and {q_next_target.shape}"
        )
    greedy_ids = q_next_online.argmax(dim=1, keepdim=True)
    target_greedy = q_next_target.gather(1, greedy_ids)
    return (
        (1.0 - beta_next_online) * q_next_target
        + beta_next_online * target_greedy
    )


def option_td_target(
    rewards: torch.Tensor,
    *,
    gamma: float,
    bootstrap: torch.Tensor,
    u_next: torch.Tensor,
    reward_scale: float = 1.0,
) -> torch.Tensor:
    """Build the option-value TD target with an option-only reward scale.

    ``reward_scale`` deliberately affects only ``Q_omega``.  The FastTD3
    critic continues to train on its existing reward path.  This is needed
    when the released bounded ``tanh`` Q head is adapted to a target domain
    whose raw discounted returns exceed ``[-1, 1]``.
    """
    scale = float(reward_scale)
    if not scale > 0.0:
        raise ValueError(f"option reward_scale must be positive, got {scale}")
    return rewards.view(-1, 1) * scale + float(gamma) * bootstrap * u_next


def compatible_option_q_loss(
    q: torch.Tensor,
    target: torch.Tensor,
    compatibility: torch.Tensor,
    *,
    released_code_reduction: bool = False,
) -> torch.Tensor:
    """Squared option TD loss under per-transition compatibility support.

    The historical project path normalizes by total compatibility mass.  The
    author-released code first sums compatible option errors per transition and
    then averages transitions, so transitions compatible with multiple sources
    deliberately contribute more total option-value supervision.
    """
    if q.shape != target.shape or q.shape != compatibility.shape:
        raise ValueError(
            "q, target, and compatibility must share shape, got "
            f"{q.shape}, {target.shape}, and {compatibility.shape}"
        )
    weighted = compatibility * (q - target.detach()).pow(2)
    if released_code_reduction:
        return weighted.sum(dim=1).mean()
    return weighted.sum() / compatibility.sum().clamp_min(1.0)


def termination_margin(q: torch.Tensor, xi: float = 0.01) -> torch.Tensor:
    """Return the continuation margin used by the PTF termination loss.

    The released PTF implementation treats ``xi == 0`` as an adaptive margin:
    ``0.8 * (max_o Q(s, o) - second_best_o Q(s, o))``. Keeping that convention
    lets configs request the released code's dynamic margin without adding a new mode
    flag. Positive margin keeps the current best option from terminating just
    because of tiny Q noise.

    Note on the paper-vs-code convention: the PTF paper's Table 4 reports a
    fixed ``xi = 0.001``, whereas the released PTF code (and our default
    ``xi = 0.0``) routes through the adaptive ``0.8 * (top1Q - top2Q)`` branch.
    These two are NOT equivalent -- the adaptive margin scales with the
    option-Q spread (order ~100 on HumanoidBench) while the fixed 0.001 was
    tuned for toy domains with Q magnitudes in [0, 5]. We follow the code
    convention because it auto-scales to HB's Q range; if reproducing the
    paper's exact setting, set ``xi`` to a positive constant explicitly.
    """
    if float(xi) != 0.0 or q.shape[1] < 2:
        return torch.full_like(q[:, 0], float(xi))
    top2 = torch.topk(q, k=2, dim=1).values
    return 0.8 * (top2[:, 0] - top2[:, 1])


def termination_loss(
    q: torch.Tensor,
    beta: torch.Tensor,
    option_ids: torch.Tensor,
    valid: torch.Tensor | None = None,
    xi: float = 0.01,
    clamp_advantage: bool = True,
) -> torch.Tensor:
    """PTF termination loss for q/beta evaluated at the termination state.

    In the PTF update this state is the transition's next observation ``s'``:
    after executing option ``o`` and arriving at ``s'``, beta decides whether
    to terminate ``o`` before the next option decision.

    The PTF paper formula is ``β · (A + ξ)`` where ``A = Q(s, o) - max_o' Q(s, o')``
    is the regret of executing ``o``. The toy domains used in the paper have
    bounded Q magnitudes (rewards in [0, 5]), so the asymmetry between
    "push-down" force on the argmax option (~+ξ) and "push-up" force on
    deeply-suboptimal options (large negative A) is small.

    This adaptation caps very large per-sample termination gradients by
    clamping the advantage to ``[-margin, +margin]``. It does *not* make the
    adaptive-margin dynamics symmetric in the two-option case: with
    ``xi=0`` the argmax option still receives ``+0.8 * gap`` while the other
    option receives ``-0.2 * gap``, both already inside the clamp. That
    distinction matters when interpreting a termination head pinned to a rail.
    """
    option_ids = option_ids.long().view(-1, 1)
    beta_o = beta.gather(1, option_ids).squeeze(1)
    q_o = q.gather(1, option_ids).squeeze(1)
    max_q = q.max(dim=1).values
    margin = termination_margin(q, xi)
    advantage_raw = q_o - max_q + margin
    if clamp_advantage:
        # Project adaptation: cap very large asymmetric gradients. The released
        # PTF code uses advantage_raw directly and selects that behavior via
        # clamp_advantage=False.
        margin_safe = (
            margin.clamp_min(1e-6)
            if margin.dim() > 0
            else torch.tensor(max(float(margin), 1e-6), device=q.device)
        )
        advantage = advantage_raw.clamp(
            min=-margin_safe,
            max=margin_safe,
        ).detach()
    else:
        advantage = advantage_raw.detach()
    if valid is None:
        return (beta_o * advantage).mean()
    valid_f = valid.float().view(-1)
    return (beta_o * advantage * valid_f).sum() / valid_f.sum().clamp_min(1.0)


def termination_loss_at_next_state(
    option_module,
    next_obs: torch.Tensor,
    option_ids: torch.Tensor,
    valid: torch.Tensor | None = None,
    xi: float = 0.01,
    clamp_advantage: bool = True,
) -> torch.Tensor:
    q_next_online, beta_next_online = option_module(next_obs)
    return termination_loss(
        q_next_online,
        beta_next_online,
        option_ids,
        valid=valid,
        xi=xi,
        clamp_advantage=clamp_advantage,
    )
