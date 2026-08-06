"""QMP-fidelity 选择器与探索噪声时序的 focused tests。

Run card: docs/run_card_qmp_fidelity_v1.md

最关键的一条是 `test_explore_noise_ordering_is_rng_equivalent`:
外部审查(2026-07-29)指出 v1 伪码会拿**带噪声**的 student 与**无噪声**的 source
比 Q,从而系统性偏向 source。本文件证明训练端采用的
"deterministic=True 打分 → 选完再加一次噪声" 与原 `explore()` 逐位等价。
"""
from __future__ import annotations

import pytest
import torch

from fasttd3_ptf.official_fasttd3_ptf.paths import ensure_fasttd3_import_path
from fasttd3_ptf.ptf.qmp import QmpSelector


def _const_qheads(values):
    """qheads_fn 桩:按候选动作的第 0 维取值决定 Q(便于构造确定的排序)。

    约定 action[:, 0] = 候选编号,values[k] 即该候选的 Q。
    """

    def fn(critic_obs, actions):
        idx = actions[:, 0].long()
        q = torch.tensor(values, dtype=torch.float32, device=actions.device)[idx]
        return q, q

    return fn


def _make_actions(batch, num_sources, action_dim=4):
    """student 编号 0,源 i 编号 i+1(写在动作第 0 维)。"""
    student = torch.zeros(batch, action_dim)
    src = torch.zeros(batch, num_sources, action_dim)
    for i in range(num_sources):
        src[:, i, 0] = i + 1
    return student, src


def test_ties_go_to_student():
    """全部候选 Q 相等时必须选 student(run card 冻结的 ties 规则)。"""
    sel = QmpSelector(num_sources=3, num_envs=5)
    student, src = _make_actions(5, 3)
    qfn = _const_qheads([7.0, 7.0, 7.0, 7.0])  # 四个候选完全并列
    _, choice, _ = sel.select(qfn, torch.zeros(5, 2), student, src)
    assert torch.equal(choice, torch.zeros(5, dtype=torch.long))


def test_argmax_selects_highest_q_source():
    sel = QmpSelector(num_sources=3, num_envs=4)
    student, src = _make_actions(4, 3)
    # student=1.0, src0=0.5, src1=9.0, src2=2.0  → 应选 src1(候选索引 2)
    qfn = _const_qheads([1.0, 0.5, 9.0, 2.0])
    selected, choice, _ = sel.select(qfn, torch.zeros(4, 2), student, src)
    assert torch.equal(choice, torch.full((4,), 2, dtype=torch.long))
    # 选出的动作必须是该源的**实际输出**,不是任何拼接
    assert torch.allclose(selected, src[:, 1, :])


def test_student_wins_when_its_q_is_highest():
    sel = QmpSelector(num_sources=2, num_envs=3)
    student, src = _make_actions(3, 2)
    qfn = _const_qheads([5.0, 1.0, 2.0])
    selected, choice, _ = sel.select(qfn, torch.zeros(3, 2), student, src)
    assert torch.equal(choice, torch.zeros(3, dtype=torch.long))
    assert torch.allclose(selected, student)


def test_non_finite_q_aborts():
    """run card §4.2:非有限 Q 拒绝启动,不静默跳过。"""
    sel = QmpSelector(num_sources=2, num_envs=2)
    student, src = _make_actions(2, 2)
    qfn = _const_qheads([1.0, float("nan"), 2.0])
    with pytest.raises(FloatingPointError):
        sel.select(qfn, torch.zeros(2, 2), student, src)


def test_empty_bank_rejected():
    with pytest.raises(ValueError):
        QmpSelector(num_sources=0, num_envs=2)


def test_source_count_mismatch_rejected():
    sel = QmpSelector(num_sources=3, num_envs=2)
    student, src = _make_actions(2, 2)  # 只给了 2 个源
    with pytest.raises(ValueError):
        sel.select(_const_qheads([1.0, 2.0, 3.0]), torch.zeros(2, 2), student, src)


def test_diagnostics_share_and_switch_rate():
    sel = QmpSelector(num_sources=2, num_envs=4)
    student, src = _make_actions(4, 2)
    cobs = torch.zeros(4, 2)

    # 第一步全选 src0(候选 1)
    _, _, d1 = sel.select(_const_qheads([0.0, 5.0, 1.0]), cobs, student, src)
    assert d1["qmp/source_share"].item() == pytest.approx(1.0)
    assert d1["qmp/share_src0"].item() == pytest.approx(1.0)
    assert d1["qmp/switch_rate"].item() == pytest.approx(1.0)  # 首步计为新段

    # 第二步仍选 src0 → 无切换
    _, _, d2 = sel.select(_const_qheads([0.0, 5.0, 1.0]), cobs, student, src)
    assert d2["qmp/switch_rate"].item() == pytest.approx(0.0)

    # 第三步改选 student → 全切换
    _, _, d3 = sel.select(_const_qheads([9.0, 5.0, 1.0]), cobs, student, src)
    assert d3["qmp/switch_rate"].item() == pytest.approx(1.0)
    assert d3["qmp/source_share"].item() == pytest.approx(0.0)

    # src0 连续跑了 2 步、1 段 → 平均连续长度 2
    run_len = sel.mean_run_lengths()
    assert run_len[1].item() == pytest.approx(2.0)


def test_score_gap_is_relative_to_student():
    sel = QmpSelector(num_sources=1, num_envs=2)
    student, src = _make_actions(2, 1)
    _, _, d = sel.select(_const_qheads([2.0, 6.5]), torch.zeros(2, 2), student, src)
    assert d["qmp/score_gap"].item() == pytest.approx(4.5)


def test_scores_use_the_actions_given_verbatim():
    """打分必须只用传入的动作,不得内部再加噪声/裁剪。"""
    sel = QmpSelector(num_sources=1, num_envs=3)
    student = torch.randn(3, 4)
    src = torch.randn(3, 1, 4)
    seen = []

    def fn(critic_obs, actions):
        seen.append(actions.clone())
        q = actions.sum(dim=1)
        return q, q

    sel.select(fn, torch.zeros(3, 2), student, src)
    assert torch.allclose(seen[0], student)
    assert torch.allclose(seen[1], src[:, 0, :])


def test_explore_noise_ordering_is_rng_equivalent():
    """**核心**:训练端的噪声时序与原 explore() 逐位等价。

    路径 A(原实现) : explore(deterministic=False)
    路径 B(QMP 端) : explore(deterministic=True) → 手动 randn_like * noise_scales

    `Actor.explore` 的 noise_scales 重采样发生在 `if deterministic: return act`
    **之前**,因此两条路径的 RNG 消耗顺序完全一致:rand(重采样) → forward → randn。
    若二者不等,说明 QMP 端的打分/加噪时序与 FastTD3 原语义偏离。
    """
    ensure_fasttd3_import_path()
    from fast_td3 import Actor

    device = torch.device("cpu")
    n_obs, n_act, n_envs = 6, 4, 8
    kwargs = dict(
        n_obs=n_obs, n_act=n_act, num_envs=n_envs, device=device,
        init_scale=0.01, hidden_dim=32, std_min=0.05, std_max=0.4,
    )

    torch.manual_seed(1234)
    actor_a = Actor(**kwargs)
    torch.manual_seed(1234)
    actor_b = Actor(**kwargs)
    for pa, pb in zip(actor_a.parameters(), actor_b.parameters()):
        assert torch.allclose(pa, pb)

    obs = torch.randn(n_envs, n_obs)
    dones = torch.zeros(n_envs)
    dones[:3] = 1.0  # 触发 episode 级 noise_scales 重采样

    torch.manual_seed(7)
    a_noisy = actor_a.explore(obs, dones)

    torch.manual_seed(7)
    a_clean = actor_b.explore(obs, dones, deterministic=True)
    a_manual = a_clean + torch.randn_like(a_clean) * actor_b.noise_scales

    assert torch.allclose(a_noisy, a_manual, atol=1e-6), (
        "QMP 的 deterministic 打分 + 事后加噪 与 explore() 不等价——"
        "噪声时序或 RNG 消耗被改变了"
    )
    # noise_scales 的重采样必须在两条路径上同样发生
    assert torch.allclose(actor_a.noise_scales, actor_b.noise_scales)
    # 且确实是有噪声的(否则上面的等价性是平凡的)
    assert not torch.allclose(a_clean, a_noisy)


def test_noisy_student_would_bias_toward_sources():
    """证明"为什么必须无噪声打分":带噪声的 student 会被系统性压低 Q。

    这是外部审查指出的 v1 伪码 bug 的直接复现。
    """
    torch.manual_seed(0)
    batch, dim = 512, 6
    student_clean = torch.zeros(batch, dim)
    source = torch.full((batch, dim), 0.05)

    # Q 随动作范数下降 → 噪声必然压低 Q
    def q_of(a):
        return -(a**2).sum(dim=1)

    noise_scale = 0.3
    student_noisy = student_clean + torch.randn_like(student_clean) * noise_scale

    clean_student_wins = (q_of(student_clean) > q_of(source)).float().mean()
    noisy_student_wins = (q_of(student_noisy) > q_of(source)).float().mean()

    assert clean_student_wins.item() == pytest.approx(1.0)
    assert noisy_student_wins.item() < 0.5, (
        "若比较用带噪声的 student,source 会被系统性偏好——正是本轮要避免的 bug"
    )
