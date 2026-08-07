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
STATUS_MISSING_MILESTONE_FIELD = "MISSING_MILESTONE_FIELD"  # 声明了 reducer 但字段全程缺失

# ── milestone reducer（protocol v21c §4 冻结，只此四个）─────────────
REDUCER_FINAL = "final"                  # 最后一步的值；该步缺此 key 则 None
REDUCER_MAX = "max"                      # trajectory 最大值；非数值则 None
REDUCER_EVER_TRUE = "ever_true"          # bool(v) 为真是否出现过
REDUCER_FIRST_HIT_STEP = "first_hit_step"  # **首次 bool(v) 为真**的步索引
ALLOWED_REDUCERS = frozenset({
    REDUCER_FINAL, REDUCER_MAX, REDUCER_EVER_TRUE, REDUCER_FIRST_HIT_STEP})

#: milestone 字段声明了 reducer 却在整条 trajectory 中一次都没出现时的标记。
#: 不得静默给出全 None 的聚合结构——那看上去像"测了但都是空"。
MILESTONE_MISSING = "MISSING_TRAJECTORY_FIELD"

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
    #: 逐 milestone 冻结的 reducer 声明（protocol v21c §4）。
    #: 键是 milestone 名，值是该 milestone **必须**输出的 reducer 元组。
    #: 声明了却全程缺失 → MISSING_MILESTONE_FIELD（fail closed）。
    #: 空 dict 表示该任务不做 reducer 强制（诊断量仍照常输出）。
    milestone_reducers: dict = field(default_factory=dict)

    def __post_init__(self):
        bad = {k: sorted(set(v) - ALLOWED_REDUCERS)
               for k, v in (self.milestone_reducers or {}).items()
               if set(v) - ALLOWED_REDUCERS}
        if bad:
            raise ValueError(f"未知 reducer：{bad}；合法值 {sorted(ALLOWED_REDUCERS)}")


def _milestones_from_info(keys: tuple[str, ...]) -> Callable:
    """默认 milestone 提取：直接从 info 取指定 key。**逐步调用**，聚合见下。"""

    def _fn(info: dict, mj_state=None) -> dict:
        return {k: info[k] for k in keys if k in info}

    return _fn


# ══════════════════════════════════════════════════════════════════════
# milestone 的 trajectory 聚合（预注册 evaluator_v21b §3）
#
# 为什么不能只读最后一步——HumanoidBench 源码里这些量真的会回落：
#
#   truck.py:113-115   for package in self.packages_on_table: ... .remove(package)
#   truck.py:199       reward_dict["success_subtasks"] = len(self.packages_on_table)
#   basketball.py:139  "success_subtasks": 1 if self.stage == "throw" else 0
#   basketball.py:140  "success": ball_hoop_distance < 0.05     ← 瞬时判定
#
# `success` 尤其危险：球穿过篮筐后飞走，最后一步就是 False——
# "成功了但记成没成功"。故必须同时保留 max（最好到过哪里）与
# final（最后落在哪里），两者之差本身就是信号。
# ══════════════════════════════════════════════════════════════════════

def _as_number(v) -> float | None:
    """能参与 max 比较则返回 float，否则 None。NaN / inf 一律不参与。

    不 import numpy——本模块刻意不依赖数值栈。numpy 标量用鸭子类型识别：
    有 ``.item()`` 且 ``shape == ()``。
    """
    if isinstance(v, bool):          # bool 是 int 子类，显式先接住
        return float(v)
    if isinstance(v, (int, float)):
        f = float(v)
        return f if f == f and f not in (float("inf"), float("-inf")) else None
    item = getattr(v, "item", None)
    if callable(item) and getattr(v, "shape", None) == ():
        try:
            return _as_number(item())
        except Exception:
            return None
    return None


def _jsonable(v):
    """转成可 JSON 序列化的值。无法识别的类型兜底为 str，绝不让写盘阶段才崩。"""
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    item = getattr(v, "item", None)
    if callable(item) and getattr(v, "shape", None) == ():
        try:
            return item()
        except Exception:
            pass
    return str(v)


def _truthy(v) -> bool:
    try:
        return bool(v)
    except Exception:
        return False


def aggregate_milestones(spec, info_history: list, mj_state=None) -> tuple[dict, bool]:
    """沿 trajectory 聚合 milestone。返回 ``(milestones, ok)``。

    ``ok=False`` 表示 ``milestone_fn`` 在**某一步**抛了异常 → 调用方判 ADAPTER_ERROR。
    不做部分容错：半截的 milestone 比没有更危险（会被当成完整数据引用）。

    每个 key 的输出结构（预注册 §3 冻结）::

        final            最后一步的值；**该步不含此 key 则 None**
        max              trajectory 上的最大值；非数值类型则 None
        max_step         首次达到 max 的步索引（0-based）
        first_step       该 key **首次出现**的步索引
        first_hit_step   该 key **首次 bool(v) 为真**的步索引；从未为真则 None
        n_steps_present  该 key 出现过的步数
        ever_true        bool(v) 为真是否至少出现过一次

    ``first_step`` 与 ``first_hit_step`` 是两回事，都保留：
    前者是"第一次有这个字段"，后者是"第一次达成"。truck 的 ``success_subtasks``
    从第 0 步就存在（值为 0），但可能到第 700 步才真正装上第一个 package。

    **fail closed**（protocol v21c §4）：``spec.milestone_reducers`` 声明了某
    milestone，而整条 trajectory 里它一次都没出现 → 该项记
    ``{"status": MILESTONE_MISSING, ...}``，并让调用方把 ``metric_status``
    置为 ``MISSING_MILESTONE_FIELD``。不得静默给出全 None 的聚合结构——
    那看上去像"测了但都是空"，而实际是"根本没这个字段"。
    """
    if spec is None or spec.milestone_fn is None:
        return {}, True
    # 0 步的 episode：保留 v21b §3 规则 3（空 history → {}，不报错）。
    # v21c §4 的 fail-closed 不覆盖这一情形——MISSING_TRAJECTORY_FIELD 的语义是
    # "跑了但没这个字段"，而 0 步是"根本没跑"，二者要能区分。
    if not info_history:
        return {}, True

    acc: dict[str, dict] = {}
    n_steps = len(info_history)
    for step, info in enumerate(info_history):
        try:
            per_step = spec.milestone_fn(info or {}, mj_state) or {}
        except Exception:
            return {}, False
        for k, v in per_step.items():
            slot = acc.get(k)
            if slot is None:
                slot = acc[k] = {
                    "final": None, "max": None, "max_step": None,
                    "first_step": step, "first_hit_step": None,
                    "n_steps_present": 0, "ever_true": False,
                    "_last_step": None, "_max_num": None,
                }
            slot["n_steps_present"] += 1
            slot["_last_step"] = step
            slot["final"] = _jsonable(v)
            if _truthy(v):
                slot["ever_true"] = True
                if slot["first_hit_step"] is None:
                    slot["first_hit_step"] = step
            num = _as_number(v)
            if num is not None and (slot["_max_num"] is None or num > slot["_max_num"]):
                slot["_max_num"] = num
                slot["max"] = _jsonable(v)      # 存原值而非 float，保住 int 语义
                slot["max_step"] = step

    for slot in acc.values():
        # final 是"最后一步的值"，不是"最后一次出现的值"——
        # 某 key 中途消失时这两者不同，后者会谎称最后一步仍有该字段。
        if slot.pop("_last_step") != n_steps - 1:
            slot["final"] = None
        slot.pop("_max_num")

    # ── 声明了 reducer 却全程缺失 → fail closed ───────────────────────
    for name, reducers in (spec.milestone_reducers or {}).items():
        if name not in acc:
            acc[name] = {
                "status": MILESTONE_MISSING,
                "declared_reducers": list(reducers),
                "final": None, "max": None, "max_step": None,
                "first_step": None, "first_hit_step": None,
                "n_steps_present": 0, "ever_true": None,
            }
    return acc, True


def missing_declared_milestones(spec, milestones: dict) -> list:
    """返回声明了 reducer 却在 trajectory 中缺失的 milestone 名。"""
    if spec is None or not spec.milestone_reducers:
        return []
    return sorted(
        name for name in spec.milestone_reducers
        if (milestones.get(name) or {}).get("status") == MILESTONE_MISSING
    )


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
        milestone_reducers={"success_subtasks": (REDUCER_MAX, REDUCER_FINAL),
                            "success": (REDUCER_EVER_TRUE, REDUCER_FIRST_HIT_STEP)},
        note="success_subtasks = 已装上车的 package 数。truck.py:113-115 有 remove 分支，"
             "故 max 与 final 都要（装上又掉下来时二者不同）。",
    ),
    "h1hand-cabinet-v0": TaskMetrics(
        termination_is=SEM_SUCCESS,
        source="cabinet.py:244  current_subtask == 5",
        milestone_names=("success", "success_subtasks"),
        milestone_fn=_milestones_from_info(("success", "success_subtasks")),
        milestone_reducers={"success_subtasks": (REDUCER_MAX, REDUCER_FINAL),
                            "success": (REDUCER_EVER_TRUE, REDUCER_FIRST_HIT_STEP)},
    ),
    "h1hand-package-v0": TaskMetrics(
        termination_is=SEM_SUCCESS,
        source="package.py:147  dist_package_destination < 0.1",
        milestone_names=("success",),
        milestone_fn=_milestones_from_info(("success",)),
        milestone_reducers={"success": (REDUCER_EVER_TRUE, REDUCER_FIRST_HIT_STEP)},
    ),

    # ── 条件判定 ─────────────────────────────────────────────────────
    "h1hand-bookshelf_simple-v0": TaskMetrics(
        termination_is=_bookshelf_termination,
        source="bookshelf.py:190  reason 0 摔倒 / 1 完成 / 2 物体掉落",
        milestone_names=("success", "success_subtasks"),
        required_info_keys=("terminated_reason",),
        milestone_fn=_milestones_from_info(("success", "success_subtasks")),
        milestone_reducers={"success_subtasks": (REDUCER_MAX, REDUCER_FINAL),
                            "success": (REDUCER_EVER_TRUE, REDUCER_FIRST_HIT_STEP)},
    ),
    "h1hand-bookshelf_hard-v0": TaskMetrics(
        termination_is=_bookshelf_termination,
        source="bookshelf.py:190  (继承 BookshelfBase)",
        milestone_names=("success", "success_subtasks"),
        required_info_keys=("terminated_reason",),
        milestone_fn=_milestones_from_info(("success", "success_subtasks")),
        milestone_reducers={"success_subtasks": (REDUCER_MAX, REDUCER_FINAL),
                            "success": (REDUCER_EVER_TRUE, REDUCER_FIRST_HIT_STEP)},
    ),
    "h1hand-basketball-v0": TaskMetrics(
        termination_is=_basketball_termination,
        source="basketball.py:143  球掉/人摔/进筐三者都 return True, {}",
        milestone_names=("success_subtasks",),
        milestone_fn=_milestones_from_info(("success_subtasks",)),
        milestone_reducers={"success_subtasks": (REDUCER_MAX, REDUCER_FINAL)},
        needs_mujoco_state=True,
        note="info 无区分字段，缺 MuJoCo state 时必须 None，不得猜测。",
    ),
})


def resolve_task_outcome(
    env_name: str,
    terminated: bool,
    truncated: bool,
    info_history: list | None = None,
    mj_state: dict | None = None,
) -> tuple[bool | None, str, str, dict]:
    """把环境事实翻译成任务成败。

    返回 ``(task_success, termination_semantics, metric_status, milestones)``。

    ``task_success`` 的三态语义：

    ``True``   该任务有经审计的成功定义，且本 episode 达成
    ``False``  有定义，且确定未达成
    ``None``   **不可判定**（任务未注册，或需要的状态缺失）——绝不退化为 False

    禁止任何形式的 ``terminated -> success`` 默认推断。

    **参数是整条 ``info_history``，不是单个 ``info``**（预注册 v21b §3）。
    旧签名收单个 ``info`` 且只读最后一步，会丢掉中途达到又回落的进度；
    此处不保留 ``info`` 兼容参数——留着就会有人继续传单步，
    而那正是丢数据的那条路径。终止语义仍由**最后一步**的 info 判定
    （``terminated_reason`` 只在终止那一步出现）。
    """
    info_history = list(info_history or [])
    last_info = info_history[-1] if info_history else {}
    spec = TASK_METRIC_REGISTRY.get(env_name)
    if spec is None:
        return None, SEM_UNKNOWN, STATUS_UNREGISTERED, {}

    # ── milestones 沿整条 trajectory 聚合 ───────────────────────────
    # 先于终止判定执行：未终止的 episode 也可能有中间进度。
    milestones, ok = aggregate_milestones(spec, info_history, mj_state)
    if not ok:
        return None, SEM_UNKNOWN, STATUS_ADAPTER_ERROR, {}
    # 声明了 reducer 却全程缺失 → fail closed，语义仍照常判定但状态被标记，
    # 使"根本没这个字段"不会被读成"测了但都是空"。
    missing = missing_declared_milestones(spec, milestones)
    info = last_info

    # ── 未终止：环境没有给出成败信号 ────────────────────────────────
    # 这不是"数据不足以判定"，而是"这一局还没分出胜负"。
    # 必须在调用条件判定函数**之前**返回——否则 bookshelf 会因为未终止的
    # episode 里没有 terminated_reason 而被误报成 INSUFFICIENT_STATE。
    # milestone 缺失只降级 metric_status，不改动成败判定——
    # 终止语义与 milestone 是两条独立通路，一条坏了不该污染另一条。
    ok_status = STATUS_MISSING_MILESTONE_FIELD if missing else STATUS_OK

    if not terminated:
        return False, SEM_NEUTRAL, ok_status, milestones

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

    return semantics == SEM_SUCCESS, semantics, ok_status, milestones


# ══════════════════════════════════════════════════════════════════════
# runtime 验证清单（预注册 evaluator_v21b §5.1）
# ══════════════════════════════════════════════════════════════════════

#: 终止语义**经真实 runtime 终止 episode 验证过**的任务。
#:
#: 注册进 ``TASK_METRIC_REGISTRY`` 只说明"我读过源码并写下了映射"，
#: 不说明"这条映射在真实 episode 上被执行过"。bookshelf 就是反例：
#: 它的 ``terminated_reason`` 0/1/2 映射只有 mock 测试覆盖，
#: 因为本地 checkpoint 已学会不摔倒，8/8 episode 无一终止，
#: 条件判定函数一次都没被调用过（判据真空成立 = VACUOUS）。
#:
#: **初值冻结为空集合**，只能由 P1.1b smoke 结果**之后的独立 commit**
#: 依据实测填入。不得在实现 commit 里凭印象预填——那是"用结果改代码"的变体。
#:
#: 2026-08-07 依据 P1.1b smoke（`docs/data/evaluator_v21b_smoke/smoke.json`
#: 的 ``runtime_verified.termination_semantics``）填入，结果见
#: `docs/experiments/evaluator_v21b_results_20260807.md` §6。
RUNTIME_VERIFIED_TERMINATION: frozenset[str] = frozenset({
    "h1hand-basketball-v0",   # S5: 8/8 终止，全部正确判为 failure
    "h1hand-crawl-v0",        # S1: 8/8 未终止 → neutral。Crawl 恒不终止
                              #     (basic_locomotion_envs.py:168 return False)，
                              #     这就是其语义的全部，不存在未覆盖的终止分支
    "h1hand-slide-v0",        # S2: 5/8 终止，全部正确判为 failure
})

#: milestone 提取通路经真实 runtime 验证过的任务。与上者分开：
#: truck 的 milestone 通路可用、但其 success 终止路径从未被观察到，
#: 二者的证据强度不同，合并会让"未验证"搭上"已验证"的便车。
#:
#: **本集合的语义严格限于"提取通路在真实 runtime 上被执行过"。**
#: 它**不**保证 milestone 的高值区间被观察过——P1.1b 中 truck 与 bookshelf 的
#: ``success_subtasks`` 全程恒为 0（策略一个 package 都没装上 / 书都没上架），
#: 故只有低值区间有 runtime 证据。对 bookshelf 尤其要注意：它的成功事件
#: (``task_index == 5``) 与终止 reason 1 是同一事件，终止既然从未发生，
#: 高值区间必然也未被观察到。引用这两个任务的 milestone 时须知此界。
RUNTIME_VERIFIED_MILESTONE: frozenset[str] = frozenset({
    "h1hand-bookshelf_simple-v0",  # S4: success / success_subtasks 均提取到
    "h1hand-truck-v0",             # S3/S6: success_subtasks 覆盖 1000 步
})
