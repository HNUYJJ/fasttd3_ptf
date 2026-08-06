"""场地判定的 fail-closed 规则（纯函数，不依赖 torch / mujoco / gymnasium）。

契约冻结于 docs/experiments/evaluator_schema_v2_prereg_20260806.md §4.5。
每个函数都对应 site screen v1 犯过的一个真实错误：

  classify_headroom     v1 在 H_ms 缺失时只看 H_op 就判 SATURATED
  pct_of_ceiling        v1 未实现 H_raw 却输出百分比
  require_comparable    v1 拿 stair@20k 对 slide@75k 比较
  is_robustly_solved    v1 用单 seed 的最大值代表任务表现（winner's curse）
  has_post_exit_deficit v1 在无 hard-exit 臂的任务上讨论 post-exit deficit

统一原则：**数据缺失一律返回 None 或 UNKNOWN 分类，绝不落进实质裁决分支**
（CLAUDE.md §4）。
"""

from __future__ import annotations

# 判定"是否稳定解决"所需的最小 learner seed 数（M24：单批 3/3 已不足以定论，
# 少于 3 更不可能支撑任何稳定性结论）。
MIN_SEEDS_FOR_ROBUST = 3

# 比较两条记录时必须一致的身份字段。缺失即不可比。
IDENTITY_KEYS_FOR_COMPARISON = ("global_step",)


class IncomparableError(Exception):
    """两条记录的身份不允许直接比较（不同预算 / 身份不完整）。"""


def classify_headroom(h_op, h_ms) -> str:
    """按预注册 §4：SATURATED 需 ``h_op <= 0`` **且** ``h_ms <= 0``。

    关键：``h_ms`` 为 None（milestone 未测）时**绝不**返回 SATURATED——
    milestone 缺口仍可能存在，此时任务是否饱和不可判定。
    """
    if h_op is None:
        return "UNKNOWN_NO_OPERATIONAL_HEADROOM_DATA"
    if h_op > 0:
        # 存在 operational 缺口，无论 milestone 如何都不是饱和
        return "HAS_HEADROOM"
    # 到这里 h_op <= 0，是否饱和取决于 milestone
    if h_ms is None:
        return "UNKNOWN_MILESTONE_NOT_MEASURED"
    if h_ms > 0:
        return "HAS_MILESTONE_HEADROOM"
    return "SATURATED"


def pct_of_ceiling(value, ceiling) -> float | None:
    """占理论上限的百分比。``ceiling`` 未审计 / 为 None → 返回 None，不计算。

    预注册 §3：``proof_status != audited`` 时 ``H_raw = UNKNOWN``，
    且**不得计算任何百分比**。
    """
    if value is None or ceiling is None:
        return None
    if not isinstance(ceiling, (int, float)) or ceiling <= 0:
        return None
    return 100.0 * float(value) / float(ceiling)


def require_comparable(a: dict, b: dict) -> None:
    """两条记录可直接比较时静默返回，否则 raise IncomparableError。

    身份字段缺失**也**视为不可比——不得把"没写 global_step"当成"步数相同"。
    """
    for key in IDENTITY_KEYS_FOR_COMPARISON:
        va, vb = a.get(key), b.get(key)
        if va is None or vb is None:
            raise IncomparableError(
                f"身份不完整，无法比较：{key} 缺失 "
                f"(a={a.get('task')}:{va}, b={b.get('task')}:{vb})"
            )
        if va != vb:
            raise IncomparableError(
                f"跨预算比较被拒绝：{key} 不同 "
                f"(a={a.get('task')}@{va}, b={b.get('task')}@{vb})。"
                f"v1 曾拿 stair@20k 对 slide@75k，结论已撤回。"
            )


def is_robustly_solved(returns_by_seed: dict, bar) -> bool | None:
    """全部 learner seed 都超过 ``bar`` 才算稳定解决。

    seed 数不足 ``MIN_SEEDS_FOR_ROBUST`` → 返回 None（INSUFFICIENT_SEEDS），
    **不得**返回 True。单 seed 的高分只能证明"曾经可达"，不能证明稳定解决。
    """
    if not returns_by_seed or bar is None:
        return None
    if len(returns_by_seed) < MIN_SEEDS_FOR_ROBUST:
        return None
    return all(float(v) > float(bar) for v in returns_by_seed.values())


def has_post_exit_deficit(hard_exit_stats, ceiling_stats) -> bool | None:
    """clean hard-exit 之后是否仍有可测缺口。

    ``hard_exit_stats`` 为空 → 返回 None。这是 P3A / P3B 分离的技术基础：
    P3A（site discovery）**不得**要求本量存在，否则复现 v1 的循环依赖——
    没有 hard-exit 数据就永远不能成为候选，也就永远不会去跑 hard-exit 实验。
    """
    if not hard_exit_stats or not ceiling_stats:
        return None
    mean = hard_exit_stats.get("mean")
    n_seeds = hard_exit_stats.get("n_seeds", 0)
    ceiling = ceiling_stats.get("ceiling")
    if mean is None or ceiling is None:
        return None
    if n_seeds < MIN_SEEDS_FOR_ROBUST:
        return None
    return float(mean) < float(ceiling)
