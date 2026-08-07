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

import json
import sys
from pathlib import Path

import numpy as np
import pytest

# 实现尚未编写时，整个模块 skip 而非 error——保持 CI 可读
pytest.importorskip(
    "fasttd3_ptf.evaluation.task_metrics",
    reason="evaluator v2 尚未实现（预注册先于实现，见文件头）",
)

from fasttd3_ptf.evaluation import schema_v2, site_rules, task_metrics  # noqa: E402
from fasttd3_ptf.evaluation.site_rules import IncomparableError  # noqa: E402


@pytest.fixture(scope="module")
def ev_v2():
    """加载 scripts/p0_evaluator_v2.py。它不是包模块，需要显式加 sys.path。

    torch 缺失时 skip——身份校验与原子写的测试依赖 torch.save 构造 checkpoint。
    """
    repo = Path(__file__).resolve().parents[1]
    for p in (str(repo), str(repo / "scripts")):
        if p not in sys.path:
            sys.path.insert(0, p)
    pytest.importorskip("torch", reason="T12/T13 需要 torch 构造 checkpoint")
    import p0_evaluator_v2

    return p0_evaluator_v2


def _manifest_for(ev_v2, ckpt_path, **overrides) -> dict:
    """按 checkpoint 实际内容构造一份合法 identity manifest。"""
    import torch

    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    args = state.get("args") or {}
    m = {
        "checkpoint_sha256": ev_v2._sha256(Path(ckpt_path)),
        "env_name": args.get("env_name"),
        "learner_seed": args.get("seed"),
        "global_step": state.get("global_step"),
    }
    if state.get("ptf_cfg"):
        m["training_protocol_digest"] = ev_v2._digest_obj(state["ptf_cfg"])
    m.update(overrides)
    return m


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
        env_name=env_name, terminated=True, truncated=False, info_history=[{}],
    )
    assert success is False, f"{env_name} 摔倒终止必须是 task_success=False，得到 {success}"
    assert semantics == "failure"
    assert status == "OK", "locomotion 任务必须已注册"


def test_T1b_truncation_is_not_success():
    """跑满 1000 步被 TimeLimit 截断，也不是任务成功。"""
    success, semantics, _, _ = task_metrics.resolve_task_outcome(
        env_name="h1hand-walk-v0", terminated=False, truncated=True, info_history=[{}],
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
        env_name="h1hand-crawl-v0", terminated=False, truncated=True, info_history=[{}],
    )
    assert success is False
    assert semantics == "neutral"
    assert status == "OK"


def test_T1d_bookshelf_termination_is_conditional():
    """同一个 terminated 有三种语义，靠 info['terminated_reason'] 区分
    （bookshelf.py:190）。静态枚举表达不了，故契约放宽为 str | Callable。
    """
    fall, sem_fall, _, _ = task_metrics.resolve_task_outcome(
        "h1hand-bookshelf_simple-v0", True, False, [{"terminated_reason": 0}])
    done, sem_done, _, _ = task_metrics.resolve_task_outcome(
        "h1hand-bookshelf_simple-v0", True, False, [{"terminated_reason": 1}])
    drop, sem_drop, _, _ = task_metrics.resolve_task_outcome(
        "h1hand-bookshelf_simple-v0", True, False, [{"terminated_reason": 2}])
    assert (fall, sem_fall) == (False, "failure"), "reason 0 = 摔倒"
    assert (done, sem_done) == (True, "success"), "reason 1 = 完成全部子任务"
    assert (drop, sem_drop) == (False, "failure"), "reason 2 = 物体掉落"


def test_T1f_conditional_task_not_terminated_is_neutral_not_unknown():
    """条件判定任务在**未终止**的 episode 上不应报 INSUFFICIENT_STATE。

    未终止只是"环境还没给出成败信号"，不是"数据不足以判定"。
    把它报成 INSUFFICIENT_STATE 会让绝大多数正常 episode 被标为不可判定。
    """
    # 带上各自声明的 milestone 字段——否则会触发 v21c §4 的 MISSING_MILESTONE_FIELD，
    # 那是另一条正交的 fail-closed 路径（T16 专门测），不是本测试的对象。
    payloads = {
        "h1hand-bookshelf_simple-v0": {"success": 0, "success_subtasks": 0},
        "h1hand-basketball-v0": {"success_subtasks": 0},
    }
    for env, info in payloads.items():
        success, semantics, status, _ = task_metrics.resolve_task_outcome(
            env, terminated=False, truncated=True, info_history=[info], mj_state=None,
        )
        assert status == "OK", f"{env} 未终止时应为 OK，得到 {status}"
        assert semantics == "neutral", f"{env} 未终止时语义应为 neutral"
        assert success is False, "已注册任务未达成 → False"


def test_T1g_unterminated_locomotion_is_neutral_not_failure():
    """跑满 1000 步没摔倒，语义是 neutral，不是 failure。"""
    _, semantics, status, _ = task_metrics.resolve_task_outcome(
        "h1hand-walk-v0", terminated=False, truncated=True, info_history=[{}],
    )
    assert semantics == "neutral"
    assert status == "OK"


def test_T1h_milestones_extracted_even_when_not_terminated():
    """未终止的 episode 也可能有中间进度，milestone 不应因此丢失。"""
    _, _, status, milestones = task_metrics.resolve_task_outcome(
        "h1hand-truck-v0", terminated=False, truncated=True,
        info_history=[{"success": 0, "success_subtasks": 2}],
    )
    assert status == "OK"
    assert milestones["success_subtasks"]["final"] == 2, "中途进度必须保留"
    assert milestones["success_subtasks"]["max"] == 2


def test_T1e_basketball_needs_state_else_null():
    """球掉 / 人摔 / 进筐三种情况都 `return True, {}`（basketball.py:143），
    info 无区分字段。缺 MuJoCo state 时必须 None，**不得猜测**。
    """
    success, _, status, _ = task_metrics.resolve_task_outcome(
        "h1hand-basketball-v0", terminated=True, truncated=False,
        info_history=[{}], mj_state=None,
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
        info_history=[{"success": 1, "success_subtasks": 3}],
    )
    assert success is True
    assert semantics == "success"
    assert status == "OK"
    assert milestones["success_subtasks"]["final"] == 3


def test_T2b_manipulation_non_success_termination():
    """同一任务下未达成成功条件时必须是 False，不是 None。"""
    success, _, status, _ = task_metrics.resolve_task_outcome(
        env_name="h1hand-truck-v0",
        terminated=False, truncated=True,
        info_history=[{"success": 0, "success_subtasks": 0}],
    )
    assert success is False, "已注册任务未达成 → False（不是 None）"
    assert status == "OK"


# ══════════════════════════════════════════════════════════════════════
# T3  未注册任务必须输出 null，不得猜测
# ══════════════════════════════════════════════════════════════════════

def test_T3_unregistered_task_yields_null():
    success, semantics, status, milestones = task_metrics.resolve_task_outcome(
        env_name="h1hand-not-a-real-task-v0",
        terminated=True, truncated=False, info_history=[{"success": 1}],
    )
    assert success is None, "未注册任务必须 None（不可判定），即使 info 里有 success"
    assert semantics == "unknown"
    assert status == "UNREGISTERED"
    assert milestones == {}


def test_T3b_null_is_distinct_from_false():
    """null（不可判定）与 false（确定未成功）必须可区分。"""
    unreg, _, _, _ = task_metrics.resolve_task_outcome(
        "h1hand-not-a-real-task-v0", True, False, [{}])
    reg, _, _, _ = task_metrics.resolve_task_outcome(
        "h1hand-walk-v0", True, False, [{}])
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

def _rec(**kw):
    """构造一条身份完整的评估记录，便于逐字段打破。

    含 protocol v21c §5 新增的三项：evaluation_semantics_digest（所有 purpose 必需）、
    match_group 与 training_protocol_digest（paired_by_seed 必需）。
    """
    base = {
        "env_name": "h1hand-slide-v0", "global_step": 100000,
        "panel_digest": "PANEL_A", "schema_version": "2.1",
        "learner_seed": 1, "method_family": "scratch",
        "checkpoint_sha256": "abc", "return": 900.0,
        "evaluation_semantics_digest": "SEM_A",
        "match_group": "GROUP_A", "training_protocol_digest": "PROTO_A",
    }
    base.update(kw)
    return base


def test_T7_cross_budget_comparison_rejected():
    """v1 的真实错误：stair@20k 对 slide@75k，得出 6.7% vs 95.1% 的无效对照。"""
    stair = _rec(env_name="h1hand-stair-v0", global_step=20000, return_=67.0)
    slide = _rec(global_step=75000)
    with pytest.raises(IncomparableError):
        site_rules.require_comparable(stair, slide, purpose="across_methods")


def test_T7b_cross_target_always_rejected():
    """跨 target 在**所有** purpose 下均拒绝——"同为 100k" 不等于可比。

    此前的实现只查 global_step，导致 crawl@100k 与 hurdle@100k 被判可比。
    """
    crawl = _rec(env_name="h1hand-crawl-v0")
    hurdle = _rec(env_name="h1hand-hurdle-v0")
    for purpose in site_rules.COMPARISON_REQUIREMENTS:
        with pytest.raises(IncomparableError, match="env_name|不完整"):
            site_rules.require_comparable(crawl, hurdle, purpose=purpose)


def test_T7c_missing_field_is_incomparable():
    """身份不完整同样不可比——缺字段时必须拒绝，而不是当作相同。"""
    a = _rec()
    b = _rec(); del b["panel_digest"]
    with pytest.raises(IncomparableError, match="不完整"):
        site_rules.require_comparable(a, b, purpose="across_methods")


def test_T7d_purpose_is_mandatory():
    """purpose 必须显式传入——没有"随手比一下"这个选项。"""
    with pytest.raises(TypeError):
        site_rules.require_comparable(_rec(), _rec())          # 位置参数不被接受
    with pytest.raises(IncomparableError, match="未知的比较目的"):
        site_rules.require_comparable(_rec(), _rec(), purpose="whatever")


def test_T7e_different_panel_rejected():
    """不同评估面板产出的数字不可比。"""
    a, b = _rec(), _rec(panel_digest="PANEL_B")
    with pytest.raises(IncomparableError, match="panel_digest|面板"):
        site_rules.require_comparable(a, b, purpose="across_methods")


def test_T7f_purposes_have_distinct_strictness():
    """across_methods 允许 method/seed 不同；paired_by_seed 不允许 seed 不同。"""
    a = _rec(method_family="scratch", learner_seed=1)
    b = _rec(method_family="hard_exit", learner_seed=2)
    site_rules.require_comparable(a, b, purpose="across_methods")   # 通过
    with pytest.raises(IncomparableError, match="learner_seed"):
        site_rules.require_comparable(a, b, purpose="paired_by_seed")
    with pytest.raises(IncomparableError, match="method_family"):
        site_rules.require_comparable(a, b, purpose="across_seeds")


def test_T7g_same_checkpoint_requires_sha():
    """重复评估同一 checkpoint 时，SHA 不同即不可比。"""
    a, b = _rec(), _rec(checkpoint_sha256="def")
    with pytest.raises(IncomparableError, match="checkpoint_sha256"):
        site_rules.require_comparable(a, b, purpose="same_checkpoint")
    site_rules.require_comparable(a, _rec(), purpose="same_checkpoint")  # 相同则通过


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
    import sys
    from pathlib import Path

    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(repo / "scripts"))

    import p0_evaluator as v1_mod
    import p0_evaluator_v2 as v2_mod

    ckpt = os.environ["EVAL_V2_INTEGRATION_CKPT"]
    env_name = os.environ.get("EVAL_V2_INTEGRATION_ENV", "h1hand-slide-v0")
    n = int(os.environ.get("EVAL_V2_INTEGRATION_N", "8"))
    device = os.environ.get("EVAL_V2_INTEGRATION_DEVICE", "cpu")

    # v1：复用其自身的 episode 循环（独立代码路径，不共享实现）
    from probe_lib import load_student
    actor, _critic, obs_norm, _critic_norm, _step = load_student(ckpt, device)
    actor.eval()
    env = v1_mod._make_env(env_name)
    v1 = []
    try:
        for eval_seed in v1_mod.EVAL_SEEDS:
            for rank in v1_mod.RANKS:
                v1.append(v1_mod._run_episode(
                    env, actor, obs_norm, device, eval_seed * 1000 + rank))
                if len(v1) >= n:
                    raise StopIteration
    except StopIteration:
        pass
    finally:
        env.close()

    v2 = v2_mod.run_panel_v2(ckpt, env_name, device=device, n_episodes=n)

    assert len(v1) == len(v2) == n
    for e1, e2 in zip(v1, v2):
        assert e1["seed"] == e2["seed"], "reset seed 顺序必须一致"
        assert e1["return"] == e2["return"], (
            f"return 必须逐位一致：seed={e1['seed']} v1={e1['return']} v2={e2['return']}")
        assert e1["progress_max_dx"] == e2["progress_max_dx"], "progress 必须逐位一致"

    # 只允许 success 语义变化：v1 把 terminated 当成功，v2 交给 registry
    changed = [
        (e1["seed"], e1["terminated_success"], e2["task_success"])
        for e1, e2 in zip(v1, v2)
        if e1["terminated_success"] != e2["task_success"]
    ]
    print(f"\nT4: {len(v1)} episodes 数值逐位一致；success 语义变化 {len(changed)} 例")
    for seed, old, new in changed[:5]:
        print(f"  seed={seed}  v1.terminated_success={old} → v2.task_success={new}")


# ══════════════════════════════════════════════════════════════════════
# T11  milestone 沿 trajectory 聚合（预注册 v21b §3）
#
# 只读最后一步会真实丢数据——HumanoidBench 源码实测：
#   truck.py:113-115   packages_on_table 有 .remove() 分支 → success_subtasks 回落
#   basketball.py:140  success = ball_hoop_distance < 0.05 是瞬时判定
# ══════════════════════════════════════════════════════════════════════

def test_T11_milestone_max_survives_regression():
    """中途装上车又掉下来的 package 不得从记录中消失。

    这是 D3 的回归测试：旧实现只读最后一步，此处 final=1 而 max=3，
    旧实现会把"最好装到 3 个"整个丢掉。
    """
    history = [
        {"success": 0, "success_subtasks": 0},
        {"success": 0, "success_subtasks": 2},
        {"success": 0, "success_subtasks": 3},   # 峰值在中间
        {"success": 0, "success_subtasks": 1},   # 掉下来了
    ]
    _, _, status, ms = task_metrics.resolve_task_outcome(
        "h1hand-truck-v0", terminated=False, truncated=True, info_history=history)
    assert status == "OK"
    slot = ms["success_subtasks"]
    assert slot["max"] == 3, "峰值必须保留"
    assert slot["max_step"] == 2, "峰值步索引 0-based"
    assert slot["final"] == 1, "最后一步的值同时保留"
    assert slot["first_step"] == 0
    assert slot["n_steps_present"] == 4
    assert slot["ever_true"] is True


def test_T11b_transient_success_not_lost():
    """success 是瞬时判定：球穿筐后飞走，最后一步为 False。

    只读最后一步 → "成功了但记成没成功"。
    """
    history = [{"success": False}, {"success": True}, {"success": False}]
    _, _, _, ms = task_metrics.resolve_task_outcome(
        "h1hand-package-v0", terminated=False, truncated=True, info_history=history)
    assert ms["success"]["ever_true"] is True, "中途成功过必须可见"
    assert ms["success"]["final"] is False
    assert ms["success"]["max_step"] == 1


def test_T11c_key_absent_in_last_step_yields_null_final():
    """final 是"最后一步的值"，不是"最后一次出现的值"。

    两者不同时，后者会谎称最后一步仍有该字段。
    """
    history = [{"success_subtasks": 2}, {"success_subtasks": 5}, {}]
    _, _, _, ms = task_metrics.resolve_task_outcome(
        "h1hand-truck-v0", terminated=False, truncated=True, info_history=history)
    slot = ms["success_subtasks"]
    assert slot["final"] is None, "最后一步无此 key → final 必须 None"
    assert slot["max"] == 5, "但 max 仍然保留"
    assert slot["n_steps_present"] == 2


def test_T11d_non_numeric_max_is_null_but_ever_true_works():
    """非数值类型不参与 max，但 ever_true 仍可算。"""
    spec = task_metrics.TaskMetrics(
        termination_is=task_metrics.SEM_SUCCESS, source="test",
        milestone_fn=task_metrics._milestones_from_info(("stage",)))
    ms, ok = task_metrics.aggregate_milestones(
        spec, [{"stage": "throw"}, {"stage": ""}, {"stage": "catch"}])
    assert ok
    assert ms["stage"]["max"] is None and ms["stage"]["max_step"] is None
    assert ms["stage"]["ever_true"] is True, "非空字符串为真"
    assert ms["stage"]["final"] == "catch"


def test_T11e_numpy_scalars_and_nan():
    """numpy 标量按数值处理；NaN 不参与 max（否则会污染比较）。"""
    spec = task_metrics.TaskMetrics(
        termination_is=task_metrics.SEM_SUCCESS, source="test",
        milestone_fn=task_metrics._milestones_from_info(("v",)))
    ms, ok = task_metrics.aggregate_milestones(
        spec, [{"v": np.float64(1.5)}, {"v": float("nan")}, {"v": np.int64(4)}])
    assert ok
    assert ms["v"]["max"] == 4 and ms["v"]["max_step"] == 2
    import json
    json.dumps(ms)  # 必须可序列化，不能把 np 标量原样塞进 JSON


def test_T11f_milestone_fn_raising_midway_is_adapter_error():
    """任一步抛异常 → 整个 episode ADAPTER_ERROR，milestones 清空。

    不做部分容错：半截的 milestone 比没有更危险，它会被当成完整数据引用。
    """
    def boom(info, mj_state=None):
        if info.get("bad"):
            raise RuntimeError("adapter 炸了")
        return {"success_subtasks": 1}

    spec = task_metrics.TaskMetrics(
        termination_is=task_metrics.SEM_SUCCESS, source="test", milestone_fn=boom)
    ms, ok = task_metrics.aggregate_milestones(spec, [{}, {}, {"bad": True}])
    assert ok is False and ms == {}


def test_T11g_empty_history_is_not_an_error():
    """0 步的 episode → 空 milestones，不报错。"""
    _, _, status, ms = task_metrics.resolve_task_outcome(
        "h1hand-truck-v0", terminated=False, truncated=True, info_history=[])
    assert status == "OK" and ms == {}


def test_T11h_flat_milestone_format_is_gone():
    """扁平格式必须已移除，不留兼容路径——它正是丢数据的那个。"""
    _, _, _, ms = task_metrics.resolve_task_outcome(
        "h1hand-truck-v0", terminated=False, truncated=True,
        info_history=[{"success_subtasks": 2}])
    assert isinstance(ms["success_subtasks"], dict), "必须是聚合结构而非裸值"
    assert set(ms["success_subtasks"]) == {
        "final", "max", "max_step", "first_step", "first_hit_step",
        "n_steps_present", "ever_true"}


def test_T11i_resolve_rejects_old_info_kwarg():
    """旧的 info= 关键字必须不再被接受，否则会静默只读单步。"""
    with pytest.raises(TypeError):
        task_metrics.resolve_task_outcome(
            "h1hand-truck-v0", terminated=False, truncated=True,
            info={"success_subtasks": 2},
        )


# ══════════════════════════════════════════════════════════════════════
# T12  身份校验 formal / debug 双模式（预注册 v21b §1）
# ══════════════════════════════════════════════════════════════════════

@pytest.fixture
def fake_ckpt(tmp_path):
    """构造一个最小 checkpoint，避免依赖真实权重文件。"""
    import torch

    def _make(env_name="h1hand-truck-v0", seed=1, global_step=50000, **extra):
        path = tmp_path / f"ckpt_{env_name}_{seed}_{global_step}.pt"
        state = {"args": {"env_name": env_name, "seed": seed}, "global_step": global_step}
        state["args"].update(extra.pop("args_extra", {}))
        state.update(extra)
        torch.save(state, path)
        return str(path)

    return _make


def test_T12_formal_requires_all_expectations(fake_ckpt, ev_v2):
    """formal 模式下缺任一强制声明即硬失败。

    旧实现只要传**任意一个** --expect-* 就算 identity_checked=true，
    seed 和 global_step 可以全都没声明（p0_evaluator_v2.py:262）。
    """
    ckpt = fake_ckpt()
    # 无 manifest → 拒绝
    with pytest.raises(ValueError, match="必须传 --identity-manifest"):
        ev_v2.verify_checkpoint_identity(ckpt, "h1hand-truck-v0")
    # formal 下 --expect-* 一律拒绝：两套身份来源并存会导致
    # "看起来声明了、实际走的是弱路径"
    with pytest.raises(ValueError, match="禁止使用 --expect"):
        ev_v2.verify_checkpoint_identity(ckpt, "h1hand-truck-v0", expect_seed=1)

    ok = ev_v2.verify_checkpoint_identity(
        ckpt, "h1hand-truck-v0", manifest=_manifest_for(ev_v2, ckpt))
    assert ok["identity_checked"] is True
    assert ok["scientific_use_permitted"] is True
    assert ok["identity_mode"] == "formal"
    assert set(ok["manifest_checked_fields"]) >= {
        "checkpoint_sha256", "env_name", "learner_seed", "global_step"}


def test_T12b_old_single_expectation_no_longer_counts(fake_ckpt, ev_v2):
    """只声明 admission_mode 一项，旧实现记 identity_checked=true。"""
    ckpt = fake_ckpt(ptf_cfg={"admission_mode": "all"})
    with pytest.raises(ValueError, match="禁止使用 --expect"):
        ev_v2.verify_checkpoint_identity(
            ckpt, "h1hand-truck-v0", expect_admission_mode="all")


def test_T12c_debug_mode_marks_output_as_unusable(fake_ckpt, ev_v2):
    """debug 允许省略声明，但产物带毒性标记。"""
    ckpt = fake_ckpt()
    info = ev_v2.verify_checkpoint_identity(ckpt, "h1hand-truck-v0", identity_mode="debug")
    assert info["identity_checked"] is False
    assert info["scientific_use_permitted"] is False, "debug 产物不得用于科学裁决"


def test_T12d_env_cross_check_cannot_be_disabled(fake_ckpt, ev_v2):
    """强制 env 核对在 debug 模式下同样执行。"""
    ckpt = fake_ckpt(env_name="h1hand-truck-v0")
    with pytest.raises(ValueError, match="env_name"):
        ev_v2.verify_checkpoint_identity(ckpt, "h1hand-crawl-v0", identity_mode="debug")


def test_T12e_missing_field_in_checkpoint_fails_formal(fake_ckpt, ev_v2):
    """checkpoint 内缺 seed / global_step 时 formal 必须硬失败。

    **无法核对 != 核对通过**——这类文件由 P2 deep scan 统计，不在此放行。
    """
    import torch
    p = fake_ckpt()
    state = torch.load(p, map_location="cpu", weights_only=False)
    del state["args"]["seed"]
    torch.save(state, p)
    with pytest.raises(ValueError, match="缺 seed"):
        ev_v2.verify_checkpoint_identity(
            p, "h1hand-truck-v0",
            manifest={"checkpoint_sha256": ev_v2._sha256(Path(p)),
                      "env_name": "h1hand-truck-v0",
                      "learner_seed": 1, "global_step": 50000})


def test_T12f_invalid_mode_rejected(fake_ckpt, ev_v2):
    with pytest.raises(ValueError, match="identity_mode"):
        ev_v2.verify_checkpoint_identity(
            fake_ckpt(), "h1hand-truck-v0", identity_mode="loose")


# ══════════════════════════════════════════════════════════════════════
# T13  原子写（预注册 v21b §2）
# ══════════════════════════════════════════════════════════════════════

def test_T13_refuses_overwrite_atomically(tmp_path, ev_v2):
    """os.link 的 fail-if-exists 由内核保证，不是 exists() 检查（TOCTOU）。"""
    out = tmp_path / "r.json"
    ev_v2.atomic_write_json(out, {"a": 1})
    assert json.loads(out.read_text())["a"] == 1
    with pytest.raises(FileExistsError):
        ev_v2.atomic_write_json(out, {"a": 2})
    assert json.loads(out.read_text())["a"] == 1, "拒绝时原文件必须不变"


def test_T13b_allow_overwrite_replaces(tmp_path, ev_v2):
    out = tmp_path / "r.json"
    ev_v2.atomic_write_json(out, {"a": 1})
    ev_v2.atomic_write_json(out, {"a": 2}, allow_overwrite=True)
    assert json.loads(out.read_text())["a"] == 2


def test_T13c_no_tmp_files_left_behind(tmp_path, ev_v2):
    """无论成败都不留 tmp——半截 JSON 看上去是合法产物，比没有更危险。"""
    out = tmp_path / "r.json"
    ev_v2.atomic_write_json(out, {"a": 1})
    with pytest.raises(FileExistsError):
        ev_v2.atomic_write_json(out, {"a": 2})
    leftovers = [p.name for p in tmp_path.iterdir() if ".tmp." in p.name]
    assert leftovers == [], f"残留临时文件：{leftovers}"


def test_T13d_unserializable_payload_leaves_no_partial_file(tmp_path, ev_v2):
    """序列化中途失败时不得留下截断的目标文件。"""
    out = tmp_path / "r.json"
    with pytest.raises(TypeError):
        ev_v2.atomic_write_json(out, {"bad": object()})
    assert not out.exists(), "失败时不得产生目标文件"
    assert [p.name for p in tmp_path.iterdir() if ".tmp." in p.name] == []


# ══════════════════════════════════════════════════════════════════════
# T14  runtime 验证闸门（预注册 v21b §5.1）
# ══════════════════════════════════════════════════════════════════════

def test_T14_unverified_task_blocked_from_adjudication():
    """注册过 != 被执行过。真空成立的条件式判据不算验证。"""
    with pytest.raises(site_rules.UnverifiedPathError):
        site_rules.require_runtime_verified(
            "h1hand-bookshelf_simple-v0", purpose="termination_semantics")


def test_T14b_lists_match_smoke_evidence():
    """清单必须逐条对应 P1.1b smoke 的实测，未验证的任务不得混入。

    初值是空集合（fail-closed），2026-08-07 依据
    docs/data/evaluator_v21b_smoke/smoke.json 的 runtime_verified 填入。
    """
    assert task_metrics.RUNTIME_VERIFIED_TERMINATION == frozenset({
        "h1hand-basketball-v0", "h1hand-crawl-v0", "h1hand-slide-v0"})
    assert task_metrics.RUNTIME_VERIFIED_MILESTONE == frozenset({
        "h1hand-bookshelf_simple-v0", "h1hand-truck-v0"})


def test_T14f_vacuous_tasks_stay_out_of_termination_list():
    """S3/S4 是 VACUOUS——0/8 终止，条件判定路径一次都没执行过。

    这是最容易被悄悄放宽的一条：把它们加进来，
    "真空成立"就变成了"已验证"。
    """
    for env in ("h1hand-truck-v0", "h1hand-bookshelf_simple-v0"):
        assert env not in task_metrics.RUNTIME_VERIFIED_TERMINATION, (
            f"{env} 的终止语义从未在 runtime 被执行，不得计为已验证")
        with pytest.raises(site_rules.UnverifiedPathError):
            site_rules.require_runtime_verified(env, purpose="termination_semantics")


def test_T14g_verified_lists_only_contain_registered_tasks():
    """清单里不得有未注册任务——那意味着填值时抄错了名字。"""
    for name in (task_metrics.RUNTIME_VERIFIED_TERMINATION
                 | task_metrics.RUNTIME_VERIFIED_MILESTONE):
        assert name in task_metrics.TASK_METRIC_REGISTRY, f"{name} 未注册"


def test_T14c_verified_task_passes(monkeypatch):
    monkeypatch.setattr(task_metrics, "RUNTIME_VERIFIED_TERMINATION",
                        frozenset({"h1hand-slide-v0"}))
    site_rules.require_runtime_verified("h1hand-slide-v0", purpose="termination_semantics")
    with pytest.raises(site_rules.UnverifiedPathError):
        site_rules.require_runtime_verified("h1hand-crawl-v0", purpose="termination_semantics")


def test_T14d_unknown_purpose_rejected():
    with pytest.raises(site_rules.UnverifiedPathError, match="未知的裁决用途"):
        site_rules.require_runtime_verified("h1hand-slide-v0", purpose="whatever")


def test_T14e_milestone_and_termination_are_separate(monkeypatch):
    """truck 的 milestone 通路可用、但 success 终止路径从未被观察到。

    合并成一个清单会让"未验证"搭上"已验证"的便车。
    """
    monkeypatch.setattr(task_metrics, "RUNTIME_VERIFIED_MILESTONE",
                        frozenset({"h1hand-truck-v0"}))
    site_rules.require_runtime_verified("h1hand-truck-v0", purpose="milestone")
    with pytest.raises(site_rules.UnverifiedPathError):
        site_rules.require_runtime_verified("h1hand-truck-v0",
                                            purpose="termination_semantics")


# ══════════════════════════════════════════════════════════════════════
# T15  mujoco_state 必须进入 episode 记录（预注册 v21b §4）
# ══════════════════════════════════════════════════════════════════════

def test_T15_mujoco_state_is_recorded():
    """S5 的判据依赖它可见。首轮之所以退化成弱代理，正是因为
    ball_to_hoop_dist 用完即弃，smoke 无从检查"是否提取到有限数值"。
    """
    rec = schema_v2.build_episode_record(
        seed=11, total_return=1.0, progress_max_dx=0.0, episode_length=10,
        terminated=True, truncated=False, task_success=False,
        termination_semantics="failure", metric_status="OK", milestones={},
        mujoco_state={"ball_to_hoop_dist": 5.76}, mujoco_state_error=None,
        info_diagnostics={}, info_diagnostics_unsupported={})
    assert rec["mujoco_state"]["ball_to_hoop_dist"] == 5.76
    assert rec["mujoco_state_error"] is None


def test_T15b_extraction_failure_is_distinguishable_from_not_needed():
    """"提取失败"与"该任务不需要 state"必须可区分，不能都是 None。"""
    failed = schema_v2.build_episode_record(
        seed=11, total_return=1.0, progress_max_dx=0.0, episode_length=10,
        terminated=True, truncated=False, task_success=None,
        termination_semantics="unknown", metric_status="INSUFFICIENT_STATE",
        milestones={}, mujoco_state=None,
        mujoco_state_error="AttributeError: 'HumanoidEnv' object has no attribute '_env'",
        info_diagnostics={}, info_diagnostics_unsupported={})
    not_needed = schema_v2.build_episode_record(
        seed=11, total_return=1.0, progress_max_dx=0.0, episode_length=10,
        terminated=True, truncated=False, task_success=False,
        termination_semantics="failure", metric_status="OK", milestones={},
        mujoco_state=None, mujoco_state_error=None,
        info_diagnostics={}, info_diagnostics_unsupported={})
    assert failed["mujoco_state_error"] is not None
    assert not_needed["mujoco_state_error"] is None


# ══════════════════════════════════════════════════════════════════════
# T16  identity manifest（protocol v21c §1 / smoke S7）
# ══════════════════════════════════════════════════════════════════════

def test_T16_formal_requires_manifest(fake_ckpt, ev_v2):
    """formal 模式没有 manifest 就不能跑——身份不能靠散落的 --expect-* 拼。"""
    ckpt = fake_ckpt()
    with pytest.raises(ValueError, match="必须传 --identity-manifest"):
        ev_v2.verify_checkpoint_identity(ckpt, "h1hand-truck-v0")


@pytest.mark.parametrize("drop", ["checkpoint_sha256", "env_name",
                                  "learner_seed", "global_step"])
def test_T16b_manifest_missing_any_required_field_fails(fake_ckpt, ev_v2, drop):
    """四个必需字段逐一缺失都必须非零退出，不得只查其中一两个。"""
    ckpt = fake_ckpt()
    m = _manifest_for(ev_v2, ckpt)
    del m[drop]
    with pytest.raises(ValueError, match="缺必需字段"):
        ev_v2.verify_checkpoint_identity(ckpt, "h1hand-truck-v0", manifest=m)


def test_T16c_wrong_sha_rejected(fake_ckpt, ev_v2):
    """SHA 是身份的核心——喂错文件必须在这里被挡住。"""
    ckpt = fake_ckpt()
    m = _manifest_for(ev_v2, ckpt, checkpoint_sha256="0" * 64)
    with pytest.raises(ValueError, match="checkpoint_sha256"):
        ev_v2.verify_checkpoint_identity(ckpt, "h1hand-truck-v0", manifest=m)


def test_T16d_ptf_cfg_present_requires_protocol_digest(fake_ckpt, ev_v2):
    """checkpoint 带 ptf_cfg 时 manifest 必须给 training_protocol_digest。"""
    ckpt = fake_ckpt(ptf_cfg={"admission_mode": "all", "mcg_warmup_steps": 30000})
    m = _manifest_for(ev_v2, ckpt)
    m.pop("training_protocol_digest")
    with pytest.raises(ValueError, match="training_protocol_digest"):
        ev_v2.verify_checkpoint_identity(ckpt, "h1hand-truck-v0", manifest=m)
    # 给对了就通过
    ok = ev_v2.verify_checkpoint_identity(
        ckpt, "h1hand-truck-v0", manifest=_manifest_for(ev_v2, ckpt))
    assert "training_protocol_digest" in ok["manifest_checked_fields"]


def test_T16e_scratch_without_ptf_cfg_needs_no_protocol_digest(fake_ckpt, ev_v2):
    """scratch checkpoint 没有 ptf_cfg，不得因此被挡在门外。

    这条是防过度收紧：强制所有 checkpoint 都有 protocol digest
    会把整条 scratch 基线排除，那样对照组就没了。
    """
    ckpt = fake_ckpt()   # 无 ptf_cfg
    ok = ev_v2.verify_checkpoint_identity(
        ckpt, "h1hand-truck-v0", manifest=_manifest_for(ev_v2, ckpt))
    assert ok["identity_checked"] is True


def test_T16f_wrong_protocol_digest_rejected(fake_ckpt, ev_v2):
    ckpt = fake_ckpt(ptf_cfg={"admission_mode": "all"})
    m = _manifest_for(ev_v2, ckpt, training_protocol_digest="deadbeef")
    with pytest.raises(ValueError, match="training_protocol_digest"):
        ev_v2.verify_checkpoint_identity(ckpt, "h1hand-truck-v0", manifest=m)


def test_T16g_malformed_manifest_file_rejected(tmp_path, ev_v2):
    """manifest 文件本身坏掉时要报清楚，不得静默当成空 dict。"""
    bad = tmp_path / "m.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="不是合法 JSON"):
        ev_v2.load_identity_manifest(bad)
    lst = tmp_path / "l.json"
    lst.write_text("[1,2]", encoding="utf-8")
    with pytest.raises(ValueError, match="顶层必须是对象"):
        ev_v2.load_identity_manifest(lst)
    with pytest.raises(ValueError, match="不存在"):
        ev_v2.load_identity_manifest(tmp_path / "nope.json")


# ══════════════════════════════════════════════════════════════════════
# T17  evaluation_semantics_digest（protocol v21c §3 / smoke S9）
# ══════════════════════════════════════════════════════════════════════

def test_T17_digest_changes_when_semantics_file_changes(ev_v2, tmp_path, monkeypatch):
    """篡改 task_metrics.py 一个字节，digest 必须改变。

    panel_digest 覆盖不到这类改动——而语义映射一变，
    task_success 的含义就变了，数字长得一样也不可比。
    """
    before = ev_v2.evaluation_semantics_digest(2.2)

    target = Path(ev_v2.REPO_ROOT) / "fasttd3_ptf/evaluation/task_metrics.py"
    original = target.read_bytes()
    try:
        target.write_bytes(original + b"\n# tamper\n")
        after = ev_v2.evaluation_semantics_digest(2.2)
    finally:
        target.write_bytes(original)
    assert before != after, "语义文件改变后 digest 必须改变"
    assert ev_v2.evaluation_semantics_digest(2.2) == before, "恢复后必须回到原值"


def test_T17b_digest_changes_with_schema_version(ev_v2):
    assert ev_v2.evaluation_semantics_digest(2.2) != ev_v2.evaluation_semantics_digest(3.0)


def test_T17c_all_three_semantics_files_covered(ev_v2):
    digests = ev_v2.semantics_file_digests()
    assert set(digests) == {
        "scripts/p0_evaluator_v2.py",
        "fasttd3_ptf/evaluation/task_metrics.py",
        "fasttd3_ptf/evaluation/schema_v2.py"}
    assert all(v != "MISSING" for v in digests.values()), digests


def test_T17d_comparison_rejects_different_semantics_digest():
    """digest 不同即拒绝——所有 purpose 都要拒，不只是某一个。"""
    a = _rec()
    b = _rec(evaluation_semantics_digest="SEM_B")
    for purpose in ("across_methods", "across_seeds",
                    "paired_by_seed", "same_checkpoint"):
        with pytest.raises(IncomparableError, match="evaluation_semantics_digest|语义"):
            site_rules.require_comparable(a, b, purpose=purpose)


def test_T17e_missing_semantics_digest_is_incomparable():
    """旧产物没有这个字段 → 不可比，而不是当成相同放行。"""
    a = _rec()
    b = _rec()
    del b["evaluation_semantics_digest"]
    with pytest.raises(IncomparableError, match="身份不完整"):
        site_rules.require_comparable(a, b, purpose="across_methods")


# ══════════════════════════════════════════════════════════════════════
# T18  paired comparison 必须来自 manifest（protocol v21c §5 / A9）
# ══════════════════════════════════════════════════════════════════════

def test_T18_paired_requires_match_group():
    """同一 (env, seed, step) 可能来自完全不同的实验臂。

    仅凭这三项推断出的"配对"是假配对——而配对差值统计的全部效力
    都建立在配对正确之上。
    """
    a = _rec()
    b = _rec()
    del b["match_group"]
    with pytest.raises(IncomparableError, match="身份不完整"):
        site_rules.require_comparable(a, b, purpose="paired_by_seed")


def test_T18b_different_match_group_rejected():
    a, b = _rec(), _rec(match_group="GROUP_B")
    with pytest.raises(IncomparableError, match="match_group|配对"):
        site_rules.require_comparable(a, b, purpose="paired_by_seed")


def test_T18c_paired_requires_training_protocol_digest():
    a = _rec()
    b = _rec()
    del b["training_protocol_digest"]
    with pytest.raises(IncomparableError, match="身份不完整"):
        site_rules.require_comparable(a, b, purpose="paired_by_seed")


def test_T18d_sha_not_required_across_arms():
    """不同实验臂的 SHA 本来就不同，要求相等会把所有真实比较都挡掉。

    SHA 的作用是**身份**（这文件是不是我以为的那个），不是可比性。
    """
    a = _rec(checkpoint_sha256="aaa")
    b = _rec(checkpoint_sha256="bbb")
    for purpose in ("across_methods", "across_seeds", "paired_by_seed"):
        site_rules.require_comparable(a, b, purpose=purpose)   # 不得因 SHA 被拒
    with pytest.raises(IncomparableError, match="checkpoint_sha256"):
        site_rules.require_comparable(a, b, purpose="same_checkpoint")


# ══════════════════════════════════════════════════════════════════════
# T19  registry reducer 与 fail closed（protocol v21c §4 / smoke S10）
# ══════════════════════════════════════════════════════════════════════

def test_T19_frozen_reducer_mapping():
    """protocol §4 冻结的强制映射必须逐条对上。"""
    reg = task_metrics.TASK_METRIC_REGISTRY
    for env in ("h1hand-truck-v0", "h1hand-cabinet-v0", "h1hand-bookshelf_simple-v0"):
        r = reg[env].milestone_reducers
        assert set(r["success_subtasks"]) == {"max", "final"}, env
        assert set(r["success"]) == {"ever_true", "first_hit_step"}, env
    assert set(reg["h1hand-package-v0"].milestone_reducers["success"]) == {
        "ever_true", "first_hit_step"}
    assert set(reg["h1hand-basketball-v0"].milestone_reducers["success_subtasks"]) == {
        "max", "final"}


def test_T19b_first_hit_step_is_first_true_not_first_present():
    """first_hit_step（首次为真）与 first_step（首次出现）是两回事。

    truck 的 success_subtasks 从第 0 步就存在（值 0），
    但可能到第 700 步才真正装上第一个 package。
    """
    history = [{"success_subtasks": 0, "success": 0},
               {"success_subtasks": 0, "success": 0},
               {"success_subtasks": 2, "success": 0},
               {"success_subtasks": 3, "success": 1}]
    _, _, status, ms = task_metrics.resolve_task_outcome(
        "h1hand-truck-v0", terminated=False, truncated=True, info_history=history)
    assert status == "OK"
    sub = ms["success_subtasks"]
    assert sub["first_step"] == 0, "该字段从第 0 步就存在"
    assert sub["first_hit_step"] == 2, "但第一次非零是第 2 步"
    assert sub["max"] == 3 and sub["final"] == 3
    assert ms["success"]["first_hit_step"] == 3
    assert ms["success"]["ever_true"] is True


def test_T19c_never_true_yields_null_first_hit_step():
    history = [{"success_subtasks": 0, "success": 0}] * 5
    _, _, _, ms = task_metrics.resolve_task_outcome(
        "h1hand-truck-v0", terminated=False, truncated=True, info_history=history)
    assert ms["success"]["first_hit_step"] is None
    assert ms["success"]["ever_true"] is False


def test_T19d_declared_milestone_missing_is_fail_closed():
    """声明了 reducer 但字段全程缺失 → MISSING_MILESTONE_FIELD。

    不得静默给出全 null 的聚合结构——那看上去像"测了但都是空"，
    而实际是"根本没这个字段"。
    """
    history = [{"success_subtasks": 1}] * 3          # 缺 success
    _, _, status, ms = task_metrics.resolve_task_outcome(
        "h1hand-truck-v0", terminated=False, truncated=True, info_history=history)
    assert status == "MISSING_MILESTONE_FIELD"
    assert ms["success"]["status"] == task_metrics.MILESTONE_MISSING
    assert ms["success"]["declared_reducers"] == ["ever_true", "first_hit_step"]
    assert ms["success_subtasks"].get("status") is None, "存在的字段不受影响"


def test_T19e_missing_milestone_does_not_corrupt_termination_semantics():
    """milestone 缺失只降级 metric_status，不得改动成败判定。

    终止语义与 milestone 是两条独立通路，一条坏了不该污染另一条。
    """
    success, semantics, status, _ = task_metrics.resolve_task_outcome(
        "h1hand-truck-v0", terminated=True, truncated=False,
        info_history=[{"success_subtasks": 5}])
    assert semantics == "success", "truck 终止即成功，不受 milestone 缺失影响"
    assert success is True
    assert status == "MISSING_MILESTONE_FIELD"


def test_T19f_zero_step_episode_is_not_missing_field():
    """0 步 episode 是"根本没跑"，与"跑了但没这个字段"必须可区分。"""
    _, _, status, ms = task_metrics.resolve_task_outcome(
        "h1hand-truck-v0", terminated=False, truncated=True, info_history=[])
    assert status == "OK"
    assert ms == {}


def test_T19g_unknown_reducer_rejected_at_construction():
    """registry 里写错 reducer 名字必须当场炸，不能等到跑评估才发现。"""
    with pytest.raises(ValueError, match="未知 reducer"):
        task_metrics.TaskMetrics(
            termination_is="success", source="test",
            milestone_reducers={"success": ("mean",)})


# ══════════════════════════════════════════════════════════════════════
# T20  formal 禁 overwrite + debug 文件名护栏（protocol v21c §1.1/§2）
# ══════════════════════════════════════════════════════════════════════

def test_T20_atomic_commit_uses_replace_and_leaves_no_tmp(tmp_path, ev_v2):
    out = tmp_path / "r.json"
    ev_v2.atomic_write_json(out, {"a": 1})
    assert json.loads(out.read_text())["a"] == 1
    with pytest.raises(FileExistsError):
        ev_v2.atomic_write_json(out, {"a": 2})
    assert json.loads(out.read_text())["a"] == 1, "拒绝时原文件必须逐字节不变"
    assert [p.name for p in tmp_path.iterdir() if ".tmp." in p.name] == []
    ev_v2.atomic_write_json(out, {"a": 3}, allow_overwrite=True)
    assert json.loads(out.read_text())["a"] == 3
