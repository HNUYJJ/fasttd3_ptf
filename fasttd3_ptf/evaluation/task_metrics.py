"""任务语义 registry：把"环境终止"翻译成"任务成败"。

契约冻结于 docs/experiments/evaluator_schema_v2_prereg_20260806.md §1.2 / §4.5，
逐任务核实结果见 §4.6。

**为什么需要这个模块**：`p0_evaluator.py:93` 写的是 ``if terminated: success = True``，
而 HumanoidBench 各任务的 ``get_terminated`` 语义完全不同——
Walk 系是摔倒、Truck 是全部装完、Bookshelf 三种情况混在同一个 True 里、
Crawl 则根本不终止。用一个通用规则去解释它们必然出错。

**注册纪律**：只注册**已逐条读过 `get_terminated` 源码**的任务，每条带 `source`
出处。未注册任务返回 ``task_success=None`` + ``metric_status="UNREGISTERED"``，
绝不猜测（M33：不得用推理代替查询）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

# ── metric_status 取值 ───────────────────────────────────────────────
STATUS_OK = "OK"
STATUS_UNREGISTERED = "UNREGISTERED"          # 任务未注册，语义不可判定
STATUS_INSUFFICIENT_STATE = "INSUFFICIENT_STATE"  # 需 MuJoCo state 但未提供
STATUS_ADAPTER_ERROR = "ADAPTER_ERROR"        # adapter 自身抛异常

# ── termination_semantics 取值 ───────────────────────────────────────
SEM_FAILURE = "failure"
SEM_SUCCESS = "success"
SEM_NEUTRAL = "neutral"
SEM_UNKNOWN = "unknown"


@dataclass(frozen=True)
class TaskMetrics:
    """单个任务的语义声明。

    ``termination_is``
        ``"failure"`` / ``"success"`` / ``"neutral"``，
        或 ``(info, mj_state) -> str | None`` 的条件判定函数。
        返回 None 表示"数据不足以判定"，调用方会转成 INSUFFICIENT_STATE。
    """

    termination_is: str | Callable
    source: str                                   # get_terminated 的源码出处
    milestone_names: tuple[str, ...] = ()
    required_info_keys: tuple[str, ...] = ()
    milestone_fn: Callable | None = None
    needs_mujoco_state: bool = False
    note: str = ""


def _milestones_from_info(keys: tuple[str, ...]) -> Callable:
    """默认 milestone 提取：直接从 info 取指定 key。"""

    def _fn(info: dict, mj_state=None) -> dict:
        return {k: info[k] for k in keys if k in info}

    return _fn


def _bookshelf_termination(info: dict, mj_state=None):
    """bookshelf.py:190 —— 同一个 terminated 有三种语义，靠 terminated_reason 区分。

        reason 0  qpos[2] < 0.58        摔倒        → failure
        reason 1  task_index == 5       全部完成    → success
        reason 2  目标物体 z < 0.5       物体掉落    → failure
    """
    reason = info.get("terminated_reason")
    if reason is None:
        return None                      # 缺字段 → 不可判定，不猜
    return SEM_SUCCESS if reason == 1 else SEM_FAILURE


def _basketball_termination(info: dict, mj_state=None):
    """basketball.py:143 —— 球掉 / 人摔 / 进筐三者都 ``return True, {}``。

    info 里**没有**任何区分字段，必须读 MuJoCo state 比较球心与 hoop_center 距离。
    缺 state 时返回 None（→ INSUFFICIENT_STATE），不得猜测。
    """
    if mj_state is None:
        return None
    dist = mj_state.get("ball_to_hoop_dist")
    if dist is None:
        return None
    return SEM_SUCCESS if float(dist) < 0.05 else SEM_FAILURE


# ══════════════════════════════════════════════════════════════════════
# Registry —— 每条都已逐条读过源码，`source` 为出处
# ══════════════════════════════════════════════════════════════════════

_FALL_LOCOMOTION = {
    "h1hand-walk-v0": "basic_locomotion_envs.py:96  qpos[2] < 0.2",
    "h1hand-run-v0": "basic_locomotion_envs.py:96  (继承 Walk)",
    "h1hand-stand-v0": "basic_locomotion_envs.py:96  (继承 Walk)",
    "h1hand-hurdle-v0": "basic_locomotion_envs.py:96  (继承 Walk)",
    "h1hand-slide-v0": "basic_locomotion_envs.py:216  torso_upright < 0.1",
    "h1hand-stair-v0": "basic_locomotion_envs.py:216  (继承 ClimbingUpwards)",
    "h1hand-sit_simple-v0": "basic_locomotion_envs.py:356  qpos[2] < 0.5",
    "h1hand-sit_hard-v0": "basic_locomotion_envs.py:356  (继承 Sit)",
    "h1hand-powerlift-v0": "powerlift.py:99  qpos[2] < 0.2",
}

TASK_METRIC_REGISTRY: dict[str, TaskMetrics] = {
    env: TaskMetrics(
        termination_is=SEM_FAILURE,
        source=src,
        note="终止 = 摔倒。v1 的 `if terminated: success = True` 在此把摔倒记为成功。",
    )
    for env, src in _FALL_LOCOMOTION.items()
}

TASK_METRIC_REGISTRY.update({
    # ── 恒不终止 ─────────────────────────────────────────────────────
    "h1hand-crawl-v0": TaskMetrics(
        termination_is=SEM_NEUTRAL,
        source="basic_locomotion_envs.py:168  return False, {}",
        note="Crawl 恒不终止，故 v1 的 bug 在 crawl 上**不触发**。",
    ),

    # ── 终止即成功 ───────────────────────────────────────────────────
    "h1hand-truck-v0": TaskMetrics(
        termination_is=SEM_SUCCESS,
        source="truck.py:207  len(packages_on_table) == len(package_list)",
        milestone_names=("success", "success_subtasks"),
        milestone_fn=_milestones_from_info(("success", "success_subtasks")),
        note="success_subtasks = 已装上车的 package 数。",
    ),
    "h1hand-cabinet-v0": TaskMetrics(
        termination_is=SEM_SUCCESS,
        source="cabinet.py:244  current_subtask == 5",
        milestone_names=("success", "success_subtasks"),
        milestone_fn=_milestones_from_info(("success", "success_subtasks")),
    ),
    "h1hand-package-v0": TaskMetrics(
        termination_is=SEM_SUCCESS,
        source="package.py:147  dist_package_destination < 0.1",
        milestone_names=("success",),
        milestone_fn=_milestones_from_info(("success",)),
    ),

    # ── 条件判定 ─────────────────────────────────────────────────────
    "h1hand-bookshelf_simple-v0": TaskMetrics(
        termination_is=_bookshelf_termination,
        source="bookshelf.py:190  reason 0 摔倒 / 1 完成 / 2 物体掉落",
        milestone_names=("success", "success_subtasks"),
        required_info_keys=("terminated_reason",),
        milestone_fn=_milestones_from_info(("success", "success_subtasks")),
    ),
    "h1hand-bookshelf_hard-v0": TaskMetrics(
        termination_is=_bookshelf_termination,
        source="bookshelf.py:190  (继承 BookshelfBase)",
        milestone_names=("success", "success_subtasks"),
        required_info_keys=("terminated_reason",),
        milestone_fn=_milestones_from_info(("success", "success_subtasks")),
    ),
    "h1hand-basketball-v0": TaskMetrics(
        termination_is=_basketball_termination,
        source="basketball.py:143  球掉/人摔/进筐三者都 return True, {}",
        milestone_names=("success_subtasks",),
        milestone_fn=_milestones_from_info(("success_subtasks",)),
        needs_mujoco_state=True,
        note="info 无区分字段，缺 MuJoCo state 时必须 None，不得猜测。",
    ),
})


def resolve_task_outcome(
    env_name: str,
    terminated: bool,
    truncated: bool,
    info: dict | None = None,
    mj_state: dict | None = None,
) -> tuple[bool | None, str, str, dict]:
    """把环境事实翻译成任务成败。

    返回 ``(task_success, termination_semantics, metric_status, milestones)``。

    ``task_success`` 的三态语义：

    ``True``   该任务有经审计的成功定义，且本 episode 达成
    ``False``  有定义，且确定未达成
    ``None``   **不可判定**（任务未注册，或需要的状态缺失）——绝不退化为 False

    禁止任何形式的 ``terminated -> success`` 默认推断。
    """
    info = info or {}
    spec = TASK_METRIC_REGISTRY.get(env_name)
    if spec is None:
        return None, SEM_UNKNOWN, STATUS_UNREGISTERED, {}

    # ── milestones 先提取：未终止的 episode 也可能有中间进度 ────────
    milestones: dict = {}
    if spec.milestone_fn is not None:
        try:
            milestones = spec.milestone_fn(info, mj_state) or {}
        except Exception:
            return None, SEM_UNKNOWN, STATUS_ADAPTER_ERROR, {}

    # ── 未终止：环境没有给出成败信号 ────────────────────────────────
    # 这不是"数据不足以判定"，而是"这一局还没分出胜负"。
    # 必须在调用条件判定函数**之前**返回——否则 bookshelf 会因为未终止的
    # episode 里没有 terminated_reason 而被误报成 INSUFFICIENT_STATE。
    if not terminated:
        return False, SEM_NEUTRAL, STATUS_OK, milestones

    # ── 已终止：判定该次终止是成功还是失败 ──────────────────────────
    if callable(spec.termination_is):
        try:
            semantics = spec.termination_is(info, mj_state)
        except Exception:
            return None, SEM_UNKNOWN, STATUS_ADAPTER_ERROR, milestones
        if semantics is None:
            # 真正的数据不足（如 basketball 缺 MuJoCo state / bookshelf 缺 reason）
            return None, SEM_UNKNOWN, STATUS_INSUFFICIENT_STATE, milestones
    else:
        semantics = spec.termination_is

    return semantics == SEM_SUCCESS, semantics, STATUS_OK, milestones
