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

# 按**比较目的**要求不同的一致性集合（预注册 evaluator_v21_hardening §4）。
# 此前只检查 global_step，导致 crawl@100k 与 hurdle@100k 被判为可比——
# "同为 100k" 远不等于可比。
#: 所有 purpose 共同必需的字段（protocol v21c §5）。
#:
#: ``evaluation_semantics_digest`` 是本轮新增的第 5 项，覆盖 schema_version、
#: source-free 模式，以及 ``p0_evaluator_v2.py`` / ``task_metrics.py`` /
#: ``schema_v2.py`` 三个文件的内容摘要。为什么必需：``panel_digest`` 只覆盖
#: seeds/ranks/steps，**两份用不同版本 task_metrics.py 产出的结果会被判为可比**，
#: 而任务语义映射一变，``task_success`` 的含义就变了。
#: 这比跨预算比较更隐蔽——数字长得完全一样。
COMPARISON_BASE: frozenset = frozenset({
    "env_name", "global_step", "panel_digest", "schema_version",
    "evaluation_semantics_digest",
})

COMPARISON_REQUIREMENTS: dict[str, frozenset] = {
    # 同一 target 上比不同方法（scratch vs continuous vs hard-exit）
    "across_methods": COMPARISON_BASE,
    # 同一方法比不同 learner seed
    "across_seeds": COMPARISON_BASE | {"method_family"},
    # 逐 seed 配对（最严，用于配对差值统计）。
    #
    # ``match_group`` 必须来自**预注册的 experiment manifest**，
    # 不得由 (env, seed, step) 现场推断：同一 (env, seed, step) 完全可能来自
    # 不同的实验臂（不同 source、不同剂量、不同退出策略），推断出来的"配对"
    # 是假配对——而配对差值统计的全部效力都建立在配对正确之上。
    "paired_by_seed": COMPARISON_BASE | {
        "learner_seed", "match_group", "training_protocol_digest"},
    # 同一 checkpoint 的重复评估（验证可复现性）。
    # ``checkpoint_sha256`` **只在此 purpose 下**要求相等——其余 purpose 下
    # 不同实验臂的 SHA 本来就不同，要求相等会把所有真实比较都挡掉。
    # SHA 的作用是**身份**（这文件是不是我以为的那个），不是可比性。
    "same_checkpoint": COMPARISON_BASE | {"learner_seed", "checkpoint_sha256"},
}


class IncomparableError(Exception):
    """两条记录的身份不允许直接比较（不同预算 / 不同面板 / 身份不完整）。"""


class UnverifiedPathError(Exception):
    """该任务的语义映射未经真实 runtime 验证，不得用于科学裁决。"""


#: 需要 runtime 验证的裁决用途。milestone 与 termination 分开：
#: 一个任务可能 milestone 通路已验证、而终止语义从未被观察到（truck 即如此）。
ADJUDICATION_REQUIREMENTS: dict[str, str] = {
    "termination_semantics": "RUNTIME_VERIFIED_TERMINATION",
    "milestone": "RUNTIME_VERIFIED_MILESTONE",
}


def require_runtime_verified(env_name: str, *, purpose: str) -> None:
    """该任务的语义通路未经真实 runtime 验证则拒绝，用于科学裁决前的闸门。

    **为什么注册过还不够**：注册进 ``TASK_METRIC_REGISTRY`` 只说明
    "读过源码并写下了映射"，不说明"这条映射被真实执行过"。
    bookshelf 的 ``terminated_reason`` 0/1/2 映射只有 mock 覆盖——
    本地 checkpoint 已学会不摔倒，8/8 episode 无一终止，
    条件判定函数一次都没被调用（判据真空成立）。真空成立不是验证。

    两个清单的初值都是空集合（fail-closed），由 smoke 实测结果填入。
    当前无科学裁决在跑（P3A 未启动），故不阻塞任何工作。
    """
    from fasttd3_ptf.evaluation import task_metrics

    attr = ADJUDICATION_REQUIREMENTS.get(purpose)
    if attr is None:
        raise UnverifiedPathError(
            f"未知的裁决用途 {purpose!r}；合法值：{sorted(ADJUDICATION_REQUIREMENTS)}")
    verified = getattr(task_metrics, attr)
    if env_name not in verified:
        raise UnverifiedPathError(
            f"{env_name} 的 {purpose} 通路未经真实 runtime 验证，不得用于科学裁决。"
            f"当前 {attr} = {sorted(verified) or '空集合（尚无任务通过 runtime 验证）'}。"
            f"注册过 != 被执行过——真空成立的条件式判据不算验证。"
        )


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


def require_comparable(a: dict, b: dict, *, purpose: str) -> None:
    """按**比较目的**校验两条记录可比；不可比则 raise IncomparableError。

    ``purpose`` 必须显式传入（关键字参数，无默认值）——没有"随手比一下"这个选项。
    合法取值见 ``COMPARISON_REQUIREMENTS``。

    三条硬规则：

    1. 要求集合中任一字段在任一侧**缺失**即不可比——身份不完整 ≠ 身份相同；
    2. 跨 ``env_name`` 在**所有** purpose 下均被拒绝。若确需跨 target 陈述，
       须先在各自 target 内算配对差值再比较差值，由调用方负责；
    3. 不提供无 purpose 的兼容签名——保留弱检查入口就会有人继续用。
    """
    if purpose not in COMPARISON_REQUIREMENTS:
        raise IncomparableError(
            f"未知的比较目的 {purpose!r}；合法值："
            f"{sorted(COMPARISON_REQUIREMENTS)}"
        )
    required = COMPARISON_REQUIREMENTS[purpose]

    missing = [k for k in sorted(required) if a.get(k) is None or b.get(k) is None]
    if missing:
        raise IncomparableError(
            f"[{purpose}] 身份不完整，无法比较：字段 {missing} 在某一侧缺失。"
            f"身份不完整不等于身份相同。"
        )

    mismatched = {k: (a[k], b[k]) for k in sorted(required) if a[k] != b[k]}
    if mismatched:
        detail = "; ".join(f"{k}: {va!r} vs {vb!r}" for k, (va, vb) in mismatched.items())
        hint = ""
        if "env_name" in mismatched:
            hint = ("。跨 target 比较一律拒绝——须先在各自 target 内算配对差值，"
                    "再比较差值")
        elif "global_step" in mismatched:
            hint = "。v1 曾拿 stair@20k 对 slide@75k，该结论已撤回"
        elif "evaluation_semantics_digest" in mismatched:
            hint = ("。两份结果由**不同版本的语义映射**产出（p0_evaluator_v2.py / "
                    "task_metrics.py / schema_v2.py 之一有差异），task_success 与 "
                    "milestones 的含义已改变——数字长得一样也不可比")
        elif "match_group" in mismatched:
            hint = ("。配对必须来自预注册 experiment manifest；同一 (env,seed,step) "
                    "可能来自不同实验臂，推断出的配对是假配对")
        elif "panel_digest" in mismatched:
            hint = "。不同评估面板产出的数字不可比"
        raise IncomparableError(f"[{purpose}] 不可比：{detail}{hint}")


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
