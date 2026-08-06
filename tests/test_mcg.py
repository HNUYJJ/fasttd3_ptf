"""MCG 核心逻辑单测:gating 的 Δ 符号/选择、模块化蒸馏、行为控制器锁存语义。"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fasttd3_ptf.ptf.action_schema import h1hand_default_action_schema
from fasttd3_ptf.official_fasttd3_ptf.admission_control import (
    AdaptiveAdmissionController,
    build_admission_snapshot,
)
from fasttd3_ptf.ptf.mcg import (
    AdmissionSegmentTracker,
    McgBehaviorController,
    ModularGating,
    mcg_distillation_loss,
)

A = 61
GROUPS = ("legs_torso", "arms", "hands")
SCHEMA = h1hand_default_action_schema()


def make_gating(margin=0.0):
    return ModularGating(action_dim=A, groups=GROUPS, device="cpu", margin=margin)


def test_group_masks_partition_action_space():
    g = make_gating()
    total = g.group_masks.sum(dim=0)
    assert g.group_masks.shape == (3, A)
    # legs_torso+arms+hands 恰好覆盖 61 维且不重叠
    assert torch.all(total == 1.0)


def _toy_qheads(a_star):
    def qheads_fn(obs, act):
        q = -((act - a_star) ** 2).sum(dim=-1)
        return q, q

    return qheads_fn


def test_deltas_sign_structure():
    """toy critic 偏好 a*=1:教师只有 arms 接近 a* 时,只有 arms 组 Δ>0。"""
    g = make_gating()
    batch = 4
    qheads_fn = _toy_qheads(torch.ones(batch, A))

    a_student = torch.zeros(batch, A)
    # 教师 0: arms 段=1(好),其余=-1(更差);教师 1: 全 -1(全差)
    arms = SCHEMA.get("arms")
    t0 = -torch.ones(A)
    t0[arms.start:arms.end] = 1.0
    src = torch.stack([t0.expand(batch, A), -torch.ones(batch, A)], dim=1)
    obs = torch.zeros(batch, 3)

    deltas = g.deltas(qheads_fn, obs, a_student, src)
    assert deltas.shape == (batch, 2, 3)
    arms_idx = GROUPS.index("arms")
    assert torch.all(deltas[:, 0, arms_idx] > 0)  # 教师0 的 arms 替换更优
    legs_idx = GROUPS.index("legs_torso")
    assert torch.all(deltas[:, 0, legs_idx] < 0)  # 教师0 的腿更差
    assert torch.all(deltas[:, 1, :] < 0)  # 教师1 全差

    best, sig, gate, conf = g.select(deltas)
    assert torch.all(best[:, arms_idx] == 0)
    assert torch.all(gate[:, arms_idx])
    assert torch.all(~gate[:, legs_idx])
    assert torch.all(conf[:, arms_idx] > 0.5) and torch.all(conf[:, legs_idx] < 0.5)


def test_paired_head_delta_uses_min_head():
    """两 head 分歧时取 paired min(保守):head2 认为教师差 → Δ 取负值。"""
    g = make_gating()
    batch, arms = 2, SCHEMA.get("arms")

    def qheads_fn(obs, act):
        arm_mean = act[:, arms.start:arms.end].mean(dim=-1)
        return arm_mean, -arm_mean  # head1 喜欢大 arms 值,head2 相反

    a_student = torch.zeros(batch, A)
    src = torch.ones(batch, 1, A)
    deltas = g.deltas(qheads_fn, torch.zeros(batch, 1), a_student, src)
    arms_idx = GROUPS.index("arms")
    assert torch.all(deltas[:, 0, arms_idx] < 0)  # min(+1, −1) = −1


def test_select_respects_margin():
    g = make_gating(margin=100.0)
    deltas = torch.full((2, 1, 3), 5.0)
    _, _, gate, _ = g.select(deltas)
    assert not gate.any()


def test_select_with_null_margins():
    """显式 margins:Δ 必须超过组级 margin 才放行;clamp_min(self.margin) 生效。"""
    g = make_gating(margin=0.0)
    deltas = torch.full((4, 1, 3), 2.0)
    margins = torch.tensor([1.0, 3.0, 2.5])
    best, sig, gate, conf = g.select(deltas, margins=margins)
    assert torch.all(gate[:, 0])          # 2.0 > 1.0
    assert not gate[:, 1].any()           # 2.0 < 3.0
    assert not gate[:, 2].any()           # 2.0 < 2.5
    assert torch.allclose(sig[:, 0], torch.ones(4))


def test_select_respects_admission_source_mask():
    g = make_gating()
    deltas = torch.tensor([[[10.0, 10.0, 10.0], [2.0, 2.0, 2.0]]])
    best, _, gate, _ = g.select(deltas, source_mask=torch.tensor([False, True]))
    assert torch.all(best == 1)
    assert gate.all()
    _, _, gate_none, conf_none = g.select(
        deltas, source_mask=torch.tensor([False, False])
    )
    assert not gate_none.any()
    assert torch.all(conf_none == 0)


def test_null_margins_suppress_noise_false_positives():
    """关键安全性质:教师与状态无关(纯噪声 Δ)时,null margin 应把 gate 压死。

    构造一个 critic:Q 只依赖 obs 与动作的随机交互(教师动作对每个状态的
    Δ 是零均值噪声)。sign gate 放行 ~50%,null-calibrated gate 应≈≤5%。
    """
    torch.manual_seed(0)
    g = make_gating()
    batch = 512
    w = torch.randn(8, A)

    def qheads_fn(obs, act):
        q = (obs @ w @ act.T).diagonal()  # 状态-动作随机耦合 → Δ 零均值噪声
        return q, q

    obs = torch.randn(batch, 8)
    a_student = torch.zeros(batch, A)
    src = torch.tanh(torch.randn(batch, 2, A))
    deltas = g.deltas(qheads_fn, obs, a_student, src)
    margins = g.null_margins(qheads_fn, obs, a_student, src, quantile=0.95)

    _, _, gate_sign, _ = g.select(deltas)  # sign 模式
    _, _, gate_null, _ = g.select(deltas, margins=margins)
    assert gate_sign.float().mean() > 0.3  # 噪声下 sign gate 大量假阳性
    assert gate_null.float().mean() < 0.15  # null 校准显著压制(q95 → ~5-10%)


def test_mcg_distillation_loss_gated_groups_only():
    g = make_gating()
    batch = 3
    pi = torch.zeros(batch, A)
    src = torch.zeros(batch, 2, A)
    arms = SCHEMA.get("arms")
    src[:, 0, :] = 0.0
    src[:, 0, arms.start:arms.end] = 1.0  # 教师0 arms 与学生差 1
    best = torch.zeros(batch, 3, dtype=torch.long)
    gate = torch.zeros(batch, 3, dtype=torch.bool)

    # gate 全关:loss 0,active 全 False
    per, active = mcg_distillation_loss(pi, src, best, gate, g.group_masks, loss_type="mse")
    assert torch.all(per == 0) and not active.any()

    # 只放行 arms:loss 只来自 arms 段(mse=1 每维,组内归一 => 1)
    gate[:, GROUPS.index("arms")] = True
    per, active = mcg_distillation_loss(pi, src, best, gate, g.group_masks, loss_type="mse")
    assert active.all()
    assert torch.allclose(per, torch.ones(batch))

    # 学生与教师该组一致时 loss=0
    pi2 = pi.clone()
    pi2[:, arms.start:arms.end] = 1.0
    per2, _ = mcg_distillation_loss(pi2, src, best, gate, g.group_masks, loss_type="mse")
    assert torch.allclose(per2, torch.zeros(batch))


def _controller(num_envs=4, exec_prob=1.0, warmup_exec_prob=1.0, min_steps=3, warmup_min_steps=5, seed=0):
    g = make_gating()
    return g, McgBehaviorController(
        num_envs=num_envs,
        num_groups=g.num_groups,
        device="cpu",
        group_masks=g.group_masks,
        min_steps=min_steps,
        warmup_min_steps=warmup_min_steps,
        exec_prob=exec_prob,
        warmup_exec_prob=warmup_exec_prob,
        seed=seed,
    )


def test_behavior_warmup_full_action_and_latch():
    g, ctl = _controller()
    E, S = 4, 2
    a_student = torch.zeros(E, A)
    src = torch.stack([torch.full((E, A), 1.0), torch.full((E, A), 2.0)], dim=1)

    acts, info = ctl.step(a_student, src, best=None, gate=None, dones=None)
    # warmup_exec_prob=1 => 全 env 教师整动作(动作全 1 或全 2)
    assert info["mcg/exec_env_frac"] == 1.0
    first = ctl.current.clone()
    assert torch.all(first >= 0)
    assert torch.all(first == first[:, :1])  # 整动作:各组同教师
    for i in range(E):
        tid = int(first[i, 0])
        assert torch.allclose(acts[i], torch.full((A,), float(tid + 1)))

    # 锁存期内(warmup_min_steps=5)不换教师
    for _ in range(3):
        ctl.step(a_student, src, best=None, gate=None, dones=None)
        assert torch.equal(ctl.current, first)


def test_behavior_done_resets_to_student():
    g, ctl = _controller()
    E = 4
    a_student = torch.zeros(E, A)
    src = torch.ones(E, 2, A)
    ctl.step(a_student, src, best=None, gate=None, dones=None)
    dones = torch.tensor([1, 0, 0, 0])
    ctl.step(a_student, src, best=None, gate=None, dones=dones)
    # done 的 env 锁存被清,立刻重抽(warmup_exec_prob=1 仍会选教师),
    # 但其余 env 锁存保持
    assert ctl.steps_left[0, 0] == ctl.warmup_min_steps - 1


def test_behavior_gated_replaces_only_gated_groups():
    g, ctl = _controller(exec_prob=1.0)
    E, S = 4, 2
    a_student = torch.zeros(E, A)
    src = torch.stack([torch.full((E, A), 1.0), torch.full((E, A), 2.0)], dim=1)
    arms_idx = GROUPS.index("arms")
    best = torch.zeros(E, g.num_groups, dtype=torch.long)
    best[:, arms_idx] = 1  # arms 最优教师 = 1
    gate = torch.zeros(E, g.num_groups, dtype=torch.bool)
    gate[:, arms_idx] = True

    acts, info = ctl.step(a_student, src, best=best, gate=gate, dones=None)
    arms = SCHEMA.get("arms")
    legs = SCHEMA.get("legs_torso")
    assert torch.allclose(acts[:, arms.start:arms.end], torch.full((E, arms.end - arms.start), 2.0))
    assert torch.allclose(acts[:, legs.start:legs.end], torch.zeros(E, legs.end - legs.start))
    assert info["mcg/exec_part_frac"] > 0


def test_behavior_gated_all_closed_is_pure_student():
    g, ctl = _controller(exec_prob=1.0)
    E = 4
    a_student = torch.randn(E, A)
    src = torch.ones(E, 2, A)
    best = torch.zeros(E, g.num_groups, dtype=torch.long)
    gate = torch.zeros(E, g.num_groups, dtype=torch.bool)
    acts, info = ctl.step(a_student, src, best=best, gate=gate, dones=None)
    assert torch.equal(acts, a_student)
    assert info["mcg/exec_env_frac"] == 0.0


def test_behavior_deterministic_same_seed():
    _, c1 = _controller(warmup_exec_prob=0.5, seed=42)
    _, c2 = _controller(warmup_exec_prob=0.5, seed=42)
    E = 4
    a_student = torch.zeros(E, A)
    src = torch.ones(E, 2, A)
    for _ in range(10):
        a1, _ = c1.step(a_student, src)
        a2, _ = c2.step(a_student, src)
        assert torch.equal(a1, a2)
        assert torch.equal(c1.current, c2.current)


def _admission_controller(admitted, student_logit=0.0, seed=0):
    g = make_gating()
    return g, McgBehaviorController(
        num_envs=128,
        num_groups=g.num_groups,
        device="cpu",
        group_masks=g.group_masks,
        warmup_min_steps=25,
        seed=seed,
        warmup_mode="admission_bootstrap",
        bootstrap_weights=torch.tensor([2.0, 0.0, 1.0]),
        bootstrap_horizons=torch.tensor([25, 25, 25]),
        admitted_sources=torch.tensor(admitted),
        admission_student_logit=student_logit,
    )


def test_admission_bootstrap_empty_set_is_exact_student():
    _, ctl = _admission_controller([False, False, False])
    student = torch.randn(128, A)
    sources = torch.randn(128, 3, A)
    actions, info = ctl.step(student, sources)
    assert torch.equal(actions, student)
    assert torch.all(ctl.current == -1)
    assert info["mcg/admission_exact_abstain"] == 1.0
    assert info["mcg/admission_student_prob"] == 1.0
    assert sum(info[f"mcg/admission_prob_src{i}"] for i in range(3)) == 0.0


def test_admission_bootstrap_student_and_only_admitted_source_are_sampled():
    _, ctl = _admission_controller([False, False, True], student_logit=1.0, seed=4)
    student = torch.zeros(128, A)
    sources = torch.stack(
        [torch.full((128, A), float(index + 1)) for index in range(3)], dim=1
    )
    ctl.step(student, sources)
    selected = set(int(value) for value in ctl.current[:, 0].unique())
    assert selected <= {-1, 2}
    assert -1 in selected and 2 in selected


def test_admission_revocation_immediately_releases_latched_source():
    _, ctl = _admission_controller([False, False, True], student_logit=-100.0)
    student = torch.zeros(128, A)
    sources = torch.ones(128, 3, A)
    ctl.step(student, sources)
    assert torch.all(ctl.current == 2)
    ctl.set_admitted_sources(torch.tensor([False, False, False]))
    assert torch.all(ctl.current == -1)
    assert torch.all(ctl.steps_left == 0)


def test_scheduled_admission_updates_logits_and_releases_revoked_source():
    _, ctl = _admission_controller([True, False, False], student_logit=-100.0)
    student = torch.zeros(128, A)
    sources = torch.ones(128, 3, A)
    ctl.step(student, sources)
    assert torch.all(ctl.current == 0)
    ctl.set_admission_policy(
        admitted_sources=torch.tensor([False, False, True]),
        source_logits=torch.tensor([-5.0, -5.0, 7.0]),
        student_logit=1.5,
    )
    assert torch.all(ctl.current == -1)
    assert ctl.admission_student_logit == 1.5
    torch.testing.assert_close(ctl.bootstrap_weights, torch.tensor([-5.0, -5.0, 7.0]))
    probabilities = ctl.admission_probabilities()
    assert probabilities[0] == 0 and probabilities[1] == 0
    assert probabilities[2] > probabilities[3]


def test_admission_segment_tracker_closes_horizon_and_done_once():
    tracker = AdmissionSegmentTracker(num_envs=2, num_sources=2, device="cpu")
    ids, means = tracker.observe(
        executed_candidates=torch.tensor([0, 2]),
        rewards=torch.tensor([1.0, 2.0]),
        natural_ends=torch.tensor([False, False]),
    )
    assert ids.numel() == 0 and means.numel() == 0
    ids, means = tracker.observe(
        executed_candidates=torch.tensor([0, 2]),
        rewards=torch.tensor([3.0, 4.0]),
        natural_ends=torch.tensor([True, True]),
    )
    torch.testing.assert_close(ids, torch.tensor([0, 2]))
    torch.testing.assert_close(means, torch.tensor([2.0, 3.0]))
    assert torch.all(tracker.active_candidate == -1)
    assert torch.all(tracker.length == 0)

    ids, means = tracker.observe(
        executed_candidates=torch.tensor([1, 2]),
        rewards=torch.tensor([5.0, 6.0]),
        natural_ends=torch.tensor([True, False]),
    )
    torch.testing.assert_close(ids, torch.tensor([1]))
    torch.testing.assert_close(means, torch.tensor([5.0]))
    assert tracker.active_candidate.tolist() == [-1, 2]


def test_admission_segment_tracker_discards_only_revoked_partial_sources():
    tracker = AdmissionSegmentTracker(num_envs=3, num_sources=2, device="cpu")
    tracker.observe(
        executed_candidates=torch.tensor([0, 1, 2]),
        rewards=torch.tensor([1.0, 2.0, 3.0]),
        natural_ends=torch.tensor([False, False, False]),
    )
    assert tracker.discard_sources([0]) == 1
    assert tracker.active_candidate.tolist() == [-1, 1, 2]
    assert tracker.length.tolist() == [0, 1, 1]
    assert tracker.reward_sum.tolist() == [0.0, 2.0, 3.0]


def test_adaptive_no_trigger_bookkeeping_preserves_behavior_rng_trace():
    _, adaptive_behavior = _admission_controller(
        [True, True, True], student_logit=2.4076059644443806, seed=42
    )
    _, static_behavior = _admission_controller(
        [True, True, True], student_logit=2.4076059644443806, seed=42
    )
    snapshot = build_admission_snapshot(
        mode="all",
        source_names=["s0", "s1", "s2"],
        source_logits=[2.0, 0.0, 1.0],
        student_logit=2.4076059644443806,
    )
    adaptive = AdaptiveAdmissionController(
        initial_snapshot=snapshot,
        stage_window_steps=50,
        min_segments=2,
        persistence=3,
    )
    tracker = AdmissionSegmentTracker(num_envs=128, num_sources=3, device="cpu")
    student = torch.zeros(128, A)
    sources = torch.ones(128, 3, A)
    dones = torch.zeros(128)

    for completed_step in range(1, 101):
        actions_adaptive, _ = adaptive_behavior.step(
            student, sources, dones=dones
        )
        actions_static, _ = static_behavior.step(student, sources, dones=dones)
        assert torch.equal(actions_adaptive, actions_static)
        assert torch.equal(adaptive_behavior.current, static_behavior.current)
        assert torch.equal(adaptive_behavior.current_arm, static_behavior.current_arm)
        assert torch.equal(adaptive_behavior.steps_left, static_behavior.steps_left)
        assert torch.equal(
            adaptive_behavior.generator.get_state(),
            static_behavior.generator.get_state(),
        )
        candidate = torch.where(
            adaptive_behavior.current_arm[:, 0] < 0,
            torch.full((128,), 3, dtype=torch.long),
            adaptive_behavior.current_arm[:, 0],
        )
        natural_ends = adaptive_behavior.steps_left[:, 0] <= 0
        ids, means = tracker.observe(
            executed_candidates=candidate,
            rewards=torch.zeros(128),
            natural_ends=natural_ends,
        )
        adaptive.record_segments(ids, means)
        window = adaptive.maybe_close_window(completed_step)
        if window is not None:
            assert window.snapshot is None
