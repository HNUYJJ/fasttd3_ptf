"""Evaluator schema v2 的 fail-closed golden tests（T1–T10）。

契约冻结于 docs/experiments/evaluator_schema_v2_prereg_20260806.md §4.5 与 §5，
**本文件先于实现提交**，故在实现完成前预期全部失败（红灯）。

每条测试对应 v1 已经犯过的一个真实错误：
  T1  p0_evaluator.py:93 `if terminated: success = True` —— Walk 系摔倒被记为成功
  T5  site screen v1 在 H_ms 缺失时判 SATURATED —— 数据缺失落进实质裁决分支
  T6  v1 未实现 H_raw 却输出百分比
  T7  v1 拿 stair@20k 对 slide@75k 比较 —— 跨预算比较
  T8  v1 用单 seed 的 best_observed 代表任务表现 —— winner's curse
  T9  v1 在无 hard-exit 臂的任务上讨论 post-exit deficit

运行：PYTHONPATH=. python -m pytest tests/test_evaluator_schema_v2.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

# 实现尚未编写时，整个模块 skip 而非 error——保持 CI 可读
pytest.importorskip(
    "fasttd3_ptf.evaluation.task_metrics",
    reason="evaluator v2 尚未实现（预注册先于实现，见文件头）",
)

from fasttd3_ptf.evaluation import schema_v2, site_rules, task_metrics  # noqa: E402
from fasttd3_ptf.evaluation.site_rules import IncomparableError  # noqa: E402


# ══════════════════════════════════════════════════════════════════════
# T1  locomotion 摔倒不得被记为成功
# ══════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("env_name", [
    "h1hand-walk-v0",    # qpos[2] < 0.2            basic_locomotion_envs.py:96
    "h1hand-run-v0",
    "h1hand-hurdle-v0",  # 继承 Walk
    "h1hand-slide-v0",   # torso_upright < 0.1      basic_locomotion_envs.py:216
    "h1hand-stair-v0",
    "h1hand-powerlift-v0",  # qpos[2] < 0.2         powerlift.py:99
])
def test_T1_locomotion_fall_is_not_success(env_name):
    """这些任务的 get_terminated 都是摔倒判定（已逐条读源码核实，见预注册 §4.6）。

    v1 的 `if terminated: success = True` 把它们记成了成功。
    **注意 crawl 不在此列**——它恒不终止，见 test_T1c。
    """
    success, semantics, status, _ = task_metrics.resolve_task_outcome(
        env_name=env_name, terminated=True, truncated=False, info={},
    )
    assert success is False, f"{env_name} 摔倒终止必须是 task_success=False，得到 {success}"
    assert semantics == "failure"
    assert status == "OK", "locomotion 任务必须已注册"


def test_T1b_truncation_is_not_success():
    """跑满 1000 步被 TimeLimit 截断，也不是任务成功。"""
    success, semantics, _, _ = task_metrics.resolve_task_outcome(
        env_name="h1hand-walk-v0", terminated=False, truncated=True, info={},
    )
    assert success is False
    assert semantics != "success"


def test_T1c_crawl_never_terminates():
    """Crawl.get_terminated 恒 return False（basic_locomotion_envs.py:168）。

    故 v1 的 bug 在 crawl 上**不触发**——不得笼统说"全部 locomotion 受影响"。
    语义是 neutral 而非 failure。
    """
    spec = task_metrics.TASK_METRIC_REGISTRY["h1hand-crawl-v0"]
    assert spec.termination_is == "neutral"
    success, semantics, status, _ = task_metrics.resolve_task_outcome(
        env_name="h1hand-crawl-v0", terminated=False, truncated=True, info={},
    )
    assert success is False
    assert semantics == "neutral"
    assert status == "OK"


def test_T1d_bookshelf_termination_is_conditional():
    """同一个 terminated 有三种语义，靠 info['terminated_reason'] 区分
    （bookshelf.py:190）。静态枚举表达不了，故契约放宽为 str | Callable。
    """
    fall, sem_fall, _, _ = task_metrics.resolve_task_outcome(
        "h1hand-bookshelf_simple-v0", True, False, {"terminated_reason": 0})
    done, sem_done, _, _ = task_metrics.resolve_task_outcome(
        "h1hand-bookshelf_simple-v0", True, False, {"terminated_reason": 1})
    drop, sem_drop, _, _ = task_metrics.resolve_task_outcome(
        "h1hand-bookshelf_simple-v0", True, False, {"terminated_reason": 2})
    assert (fall, sem_fall) == (False, "failure"), "reason 0 = 摔倒"
    assert (done, sem_done) == (True, "success"), "reason 1 = 完成全部子任务"
    assert (drop, sem_drop) == (False, "failure"), "reason 2 = 物体掉落"


def test_T1f_conditional_task_not_terminated_is_neutral_not_unknown():
    """条件判定任务在**未终止**的 episode 上不应报 INSUFFICIENT_STATE。

    未终止只是"环境还没给出成败信号"，不是"数据不足以判定"。
    把它报成 INSUFFICIENT_STATE 会让绝大多数正常 episode 被标为不可判定。
    """
    for env in ("h1hand-bookshelf_simple-v0", "h1hand-basketball-v0"):
        success, semantics, status, _ = task_metrics.resolve_task_outcome(
            env, terminated=False, truncated=True, info={}, mj_state=None,
        )
        assert status == "OK", f"{env} 未终止时应为 OK，得到 {status}"
        assert semantics == "neutral", f"{env} 未终止时语义应为 neutral"
        assert success is False, "已注册任务未达成 → False"


def test_T1g_unterminated_locomotion_is_neutral_not_failure():
    """跑满 1000 步没摔倒，语义是 neutral，不是 failure。"""
    _, semantics, status, _ = task_metrics.resolve_task_outcome(
        "h1hand-walk-v0", terminated=False, truncated=True, info={},
    )
    assert semantics == "neutral"
    assert status == "OK"


def test_T1h_milestones_extracted_even_when_not_terminated():
    """未终止的 episode 也可能有中间进度，milestone 不应因此丢失。"""
    _, _, status, milestones = task_metrics.resolve_task_outcome(
        "h1hand-truck-v0", terminated=False, truncated=True,
        info={"success": 0, "success_subtasks": 2},
    )
    assert status == "OK"
    assert milestones.get("success_subtasks") == 2, "中途进度必须保留"


def test_T1e_basketball_needs_state_else_null():
    """球掉 / 人摔 / 进筐三种情况都 `return True, {}`（basketball.py:143），
    info 无区分字段。缺 MuJoCo state 时必须 None，**不得猜测**。
    """
    success, _, status, _ = task_metrics.resolve_task_outcome(
        "h1hand-basketball-v0", terminated=True, truncated=False,
        info={}, mj_state=None,
    )
    assert success is None, "无 state 时不可判定，必须 None"
    assert status == "INSUFFICIENT_STATE"


# ══════════════════════════════════════════════════════════════════════
# T2  manipulation 的真实成功必须能被识别
# ══════════════════════════════════════════════════════════════════════

def test_T2_manipulation_success_detected():
    """必须依据任务自身的成功条件，而不是 terminated。"""
    success, semantics, status, milestones = task_metrics.resolve_task_outcome(
        env_name="h1hand-truck-v0",
        terminated=True, truncated=False,
        info={"success": 1, "success_subtasks": 3},
    )
    assert success is True
    assert semantics == "success"
    assert status == "OK"
    assert milestones.get("success_subtasks") == 3


def test_T2b_manipulation_non_success_termination():
    """同一任务下未达成成功条件时必须是 False，不是 None。"""
    success, _, status, _ = task_metrics.resolve_task_outcome(
        env_name="h1hand-truck-v0",
        terminated=False, truncated=True,
        info={"success": 0, "success_subtasks": 0},
    )
    assert success is False, "已注册任务未达成 → False（不是 None）"
    assert status == "OK"


# ══════════════════════════════════════════════════════════════════════
# T3  未注册任务必须输出 null，不得猜测
# ══════════════════════════════════════════════════════════════════════

def test_T3_unregistered_task_yields_null():
    success, semantics, status, milestones = task_metrics.resolve_task_outcome(
        env_name="h1hand-not-a-real-task-v0",
        terminated=True, truncated=False, info={"success": 1},
    )
    assert success is None, "未注册任务必须 None（不可判定），即使 info 里有 success"
    assert semantics == "unknown"
    assert status == "UNREGISTERED"
    assert milestones == {}


def test_T3b_null_is_distinct_from_false():
    """null（不可判定）与 false（确定未成功）必须可区分。"""
    unreg, _, _, _ = task_metrics.resolve_task_outcome(
        "h1hand-not-a-real-task-v0", True, False, {})
    reg, _, _, _ = task_metrics.resolve_task_outcome(
        "h1hand-walk-v0", True, False, {})
    assert unreg is None and reg is False
    assert unreg is not reg


# ══════════════════════════════════════════════════════════════════════
# T5  milestone 缺失时绝不得判定 SATURATED
# ══════════════════════════════════════════════════════════════════════

def test_T5_missing_milestone_never_saturated():
    """site screen v1 的真实 bug：H_ms 为 None 时只看 H_op 就判 SATURATED。"""
    verdict = site_rules.classify_headroom(h_op=-284.9, h_ms=None)
    assert verdict != "SATURATED", "milestone 未测时不得判定饱和"
    assert "UNKNOWN" in verdict or "INSUFFICIENT" in verdict


def test_T5b_saturated_requires_both():
    """两个 headroom 都 ≤ 0 才允许 SATURATED。"""
    assert site_rules.classify_headroom(h_op=-284.9, h_ms=0) == "SATURATED"
    assert site_rules.classify_headroom(h_op=+633.0, h_ms=0) != "SATURATED"
    assert site_rules.classify_headroom(h_op=-284.9, h_ms=2) != "SATURATED"


# ══════════════════════════════════════════════════════════════════════
# T6  理论上限缺失时不得计算百分比
# ══════════════════════════════════════════════════════════════════════

def test_T6_no_ceiling_no_percentage():
    assert site_rules.pct_of_ceiling(value=100.6, ceiling=None) is None


def test_T6b_ceiling_present_computes():
    got = site_rules.pct_of_ceiling(value=984.9, ceiling=1000)
    assert got == pytest.approx(98.49, abs=0.01)


# ══════════════════════════════════════════════════════════════════════
# T7  不同 global_step 不得进入正式配对
# ══════════════════════════════════════════════════════════════════════

def test_T7_cross_budget_comparison_rejected():
    """v1 的真实错误：stair@20k 对 slide@75k，得出 6.7% vs 95.1% 的无效对照。"""
    stair = {"task": "stair", "global_step": 20000, "method": "slidesrc", "return": 67.0}
    slide = {"task": "slide", "global_step": 75000, "method": "hard_exit", "return": 951.5}
    with pytest.raises(IncomparableError):
        site_rules.require_comparable(stair, slide)


def test_T7b_same_step_allowed():
    a = {"task": "crawl", "global_step": 100000, "method": "scratch", "return": 960.2}
    b = {"task": "hurdle", "global_step": 100000, "method": "scratch", "return": 387.4}
    site_rules.require_comparable(a, b)  # 不抛异常即通过


def test_T7c_missing_step_is_incomparable():
    """身份不完整同样不可比——缺 global_step 时必须拒绝，而不是当作相同。"""
    a = {"task": "crawl", "global_step": 100000, "return": 960.2}
    b = {"task": "hurdle", "return": 387.4}
    with pytest.raises(IncomparableError):
        site_rules.require_comparable(a, b)


# ══════════════════════════════════════════════════════════════════════
# T8  单 seed 不得标记 robustly solved
# ══════════════════════════════════════════════════════════════════════

def test_T8_single_seed_not_robustly_solved():
    """v1 用跨 method/seed/step 的最大值代表任务表现 = winner's curse。"""
    assert site_rules.is_robustly_solved({1: 984.9}, bar=700) is None


def test_T8b_two_seeds_still_insufficient():
    assert site_rules.is_robustly_solved({1: 984.9, 2: 939.5}, bar=700) is None


def test_T8c_three_seeds_all_above_bar():
    got = site_rules.is_robustly_solved({1: 956.3, 2: 939.5, 3: 984.9}, bar=700)
    assert got is True


def test_T8d_three_seeds_one_below_bar():
    got = site_rules.is_robustly_solved({1: 956.3, 2: 650.0, 3: 984.9}, bar=700)
    assert got is False, "有 seed 未过门槛 → 不是稳定解决（False，非 None）"


# ══════════════════════════════════════════════════════════════════════
# T9  缺 hard-exit 分支不得判定 post-exit deficit
# ══════════════════════════════════════════════════════════════════════

def test_T9_no_hard_exit_arm_no_deficit_verdict():
    """P3A 不得要求该量存在——否则循环依赖复现（预注册 §2.1）。"""
    assert site_rules.has_post_exit_deficit(
        hard_exit_stats=None, ceiling_stats={"ceiling": 1000}) is None
    assert site_rules.has_post_exit_deficit(
        hard_exit_stats={}, ceiling_stats={"ceiling": 1000}) is None


def test_T9b_hard_exit_present_yields_verdict():
    got = site_rules.has_post_exit_deficit(
        hard_exit_stats={"mean": 929.1, "n_seeds": 3, "global_step": 100000},
        ceiling_stats={"ceiling": 1000},
    )
    assert got in (True, False), "有数据时必须给出明确裁决"


# ══════════════════════════════════════════════════════════════════════
# T10  非标量 info 的三级处理（预注册 §1.4）
# ══════════════════════════════════════════════════════════════════════

def test_T10_required_field_unparseable_fails_closed():
    """注册的必需 milestone 字段解析失败 → 硬报错，绝不静默跳过。"""
    history = [{"success_subtasks": object()}]
    with pytest.raises(schema_v2.RequiredFieldError) as exc:
        schema_v2.summarize_info(history, required_keys=("success_subtasks",))
    msg = str(exc.value)
    assert "success_subtasks" in msg, "报错必须指明 key"
    assert "object" in msg.lower() or "type" in msg.lower(), "报错必须指明 type"


def test_T10b_unregistered_nonscalar_recorded_not_dropped():
    """未注册的非标量 → 记 type+shape 进 unsupported，不得静默丢弃。"""
    history = [{"contact_forces": np.zeros((3, 4)), "reward_move": 0.5}]
    diag, unsupported = schema_v2.summarize_info(history, required_keys=())
    assert "reward_move" in diag, "可转标量的诊断字段应进 info_diagnostics"
    assert "contact_forces" in unsupported, "不可转标量的字段必须留痕，不能消失"
    assert unsupported["contact_forces"]["shape"] == (3, 4)
    assert "ndarray" in unsupported["contact_forces"]["type"]


def test_T10c_required_nonscalar_parsed_correctly():
    """必需的非标量 milestone 能正确解析时不应报错。"""
    history = [{"success_subtasks": np.int64(2)}, {"success_subtasks": np.int64(3)}]
    diag, unsupported = schema_v2.summarize_info(
        history, required_keys=("success_subtasks",))
    assert diag["success_subtasks"]["max"] == 3
    assert diag["success_subtasks"]["final"] == 3
    assert "success_subtasks" not in unsupported


def test_T10d_nan_inf_do_not_crash():
    history = [{"x": float("nan")}, {"x": float("inf")}, {"x": 1.0}]
    diag, unsupported = schema_v2.summarize_info(history, required_keys=())
    assert "x" in diag or "x" in unsupported, "NaN/Inf 必须有明确归属，不得消失"


def test_T10e_time_varying_keys_handled():
    """key 只在部分步出现（HumanoidBench 的 stage 相关字段会这样）。"""
    history = [{"a": 1.0}, {"a": 2.0, "b": 5.0}, {"b": 7.0}]
    diag, _ = schema_v2.summarize_info(history, required_keys=())
    assert diag["a"]["max"] == 2.0
    assert diag["b"]["max"] == 7.0


# ══════════════════════════════════════════════════════════════════════
# T4  v1/v2 逐位兼容（集成测试，需真实 checkpoint 与 MuJoCo）
# ══════════════════════════════════════════════════════════════════════

@pytest.mark.integration
@pytest.mark.skipif(
    not __import__("os").environ.get("EVAL_V2_INTEGRATION_CKPT"),
    reason="需设置 EVAL_V2_INTEGRATION_CKPT 指向一个真实 checkpoint",
)
def test_T4_v1_v2_bitwise_identical_on_shared_fields():
    """同 checkpoint、同 32-episode 面板：return / progress / episode_length /
    reset seed 顺序必须逐 episode 完全一致。只允许 success 字段变化。

    这条防的是"改口径导致数字变化"被误当成"修好了 bug"。
    """
    import os

    from scripts import p0_evaluator  # noqa: F401
    ckpt = os.environ["EVAL_V2_INTEGRATION_CKPT"]
    env_name = os.environ.get("EVAL_V2_INTEGRATION_ENV", "h1hand-slide-v0")

    v1 = schema_v2.run_panel_v1_compat(ckpt, env_name, n_episodes=32)
    v2 = schema_v2.run_panel_v2(ckpt, env_name, n_episodes=32)

    assert len(v1) == len(v2) == 32
    for e1, e2 in zip(v1, v2):
        assert e1["seed"] == e2["seed"], "reset seed 顺序必须一致"
        assert e1["return"] == e2["return"], "return 必须逐位一致"
        assert e1["progress_max_dx"] == e2["progress_max_dx"]
        assert e1["episode_length"] == e2["episode_length"]
