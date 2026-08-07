"""Evaluator schema v2：episode 记录构造与 info 字段的分级处理。

契约冻结于 docs/experiments/evaluator_schema_v2_prereg_20260806.md §1 与 §4.5。

与 v1 的关键差别：**环境事实与任务语义分离**。
v1 把 ``terminated`` 直接写成 ``terminated_success``，而 Walk 系的
``get_terminated`` 是摔倒判定（``qpos[2] < 0.2``），于是摔倒被记成了成功。
v2 只记录环境事实，任务语义一律交给 ``task_metrics`` 的 registry。
"""

from __future__ import annotations

import math

# 2.2：milestone 由"最后一步的扁平值"改为 trajectory 聚合结构，
# 并新增 mujoco_state / mujoco_state_error。**这是破坏性变更**——
# 旧的扁平 milestone 格式已移除，不留兼容路径（预注册 v21b §3.4）。
SCHEMA_VERSION = 2.2

# v1 已有、必须逐位兼容的字段（T4 校验这些）
V1_COMPATIBLE_FIELDS = ("seed", "return", "progress_max_dx")


class RequiredFieldError(Exception):
    """registry 声明为必需的字段无法解析成标量。

    这是 fail-closed 的核心：必需字段是判据输入，解析错了会污染结论，
    因此必须中止而不是给默认值。未注册的诊断字段则走 unsupported 分支。
    """


def _describe(value) -> dict:
    """给不可转标量的值留痕：类型 + 形状。"""
    shape = None
    if hasattr(value, "shape"):
        shape = tuple(value.shape)
    elif isinstance(value, (list, tuple)):
        shape = (len(value),)
    return {"type": type(value).__name__, "shape": shape}


def _as_scalar(value):
    """尽力转成 Python float/int；失败返回 None（由调用方决定如何处理）。

    NaN / Inf 视为**可转标量**——它们是合法的数值信号（例如发散），
    静默丢弃反而会掩盖问题。
    """
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    # NumPy 标量与 0 维数组
    if hasattr(value, "ndim") and hasattr(value, "item"):
        try:
            if value.ndim == 0:
                return float(value.item())
        except (ValueError, TypeError):
            return None
        return None
    if hasattr(value, "item") and not hasattr(value, "__len__"):
        try:
            return float(value.item())
        except (ValueError, TypeError):
            return None
    return None


def summarize_info(info_history, required_keys=()) -> tuple[dict, dict]:
    """把一个 episode 的 info 序列汇总成诊断量。

    返回 ``(info_diagnostics, info_diagnostics_unsupported)``。

    三级处理（预注册 §1.4）：

    1. ``required_keys`` 中的字段不可转标量 → raise ``RequiredFieldError``；
    2. 未注册字段可转标量 → 进 ``info_diagnostics``；
    3. 未注册字段不可转标量 → 进 ``info_diagnostics_unsupported``，
       记 ``{type, shape}``，**不得静默丢弃**。

    key 只在部分步出现（时变 key）是合法的，按实际出现的步汇总。
    """
    required = set(required_keys or ())
    series: dict[str, list] = {}
    unsupported: dict[str, dict] = {}

    for step_idx, info in enumerate(info_history or ()):
        if not isinstance(info, dict):
            continue
        for key, raw in info.items():
            scalar = _as_scalar(raw)
            if scalar is None:
                if key in required:
                    desc = _describe(raw)
                    raise RequiredFieldError(
                        f"必需字段 {key!r} 在第 {step_idx} 步无法解析为标量："
                        f"type={desc['type']}, shape={desc['shape']}。"
                        f"该字段是判据输入，不得跳过或给默认值。"
                    )
                unsupported.setdefault(key, _describe(raw))
                continue
            series.setdefault(key, []).append((step_idx, scalar))

    diagnostics: dict[str, dict] = {}
    for key, pairs in series.items():
        values = [v for _, v in pairs]
        finite = [v for v in values if math.isfinite(v)]
        first_positive = next((i for i, v in pairs if v > 0), None)
        diagnostics[key] = {
            "mean": (sum(finite) / len(finite)) if finite else None,
            "max": max(finite) if finite else None,
            "final": values[-1],
            "nonzero_fraction": sum(1 for v in values if v != 0) / len(values),
            "first_positive_step": first_positive,
            "n_steps_present": len(values),
            "has_nonfinite": len(finite) != len(values),
        }

    return diagnostics, unsupported


def build_episode_record(
    *,
    seed: int,
    total_return: float,
    progress_max_dx: float,
    episode_length: int,
    terminated: bool,
    truncated: bool,
    task_success,
    termination_semantics: str,
    metric_status: str,
    milestones: dict,
    info_diagnostics: dict,
    info_diagnostics_unsupported: dict,
    mujoco_state: dict | None = None,
    mujoco_state_error: str | None = None,
) -> dict:
    """构造 v2 的单 episode 记录。

    ``terminated_success`` **不再产生**，也不保留别名——任何下游脚本必须显式
    迁移到 ``task_success``，不得静默继承旧语义（CLAUDE.md §6 的陷阱由此而来）。

    ``mujoco_state`` / ``mujoco_state_error``（v2.1b 新增）：把 MuJoCo 侧提取的
    原始量落进记录。此前它只在函数内部用完即弃，导致 smoke 无法按预注册原文
    检查"是否提取到有限 ``ball_to_hoop_dist``"，只能退而检查 ``metric_status``
    这个更弱的代理——而后者在 0 终止时真空通过。判据要能实现，
    它依赖的量就必须可见。
    """
    return {
        "schema_version": SCHEMA_VERSION,
        "seed": seed,
        "return": total_return,
        "progress_max_dx": progress_max_dx,
        "episode_length": episode_length,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "termination_semantics": termination_semantics,
        "task_success": task_success,
        "metric_status": metric_status,
        "milestones": milestones,
        "mujoco_state": mujoco_state,
        "mujoco_state_error": mujoco_state_error,
        "info_diagnostics": info_diagnostics,
        "info_diagnostics_unsupported": info_diagnostics_unsupported,
    }
