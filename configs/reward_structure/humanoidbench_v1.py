"""HumanoidBench 17 个 target 的 reward 组合结构规格（从源码逐个核准，2026-07-28）。

来源：`fasttd3_ptf/official_code/humanoid-bench/humanoid_bench/envs/*.py` 的 `get_reward()`。
每条 spec 记录**组合算子**而不只是权重——因为本研究的核心论点是：
标量 return 把"分量质量 / 组合算子 / 生存时长"三者混成一个数，
而组合算子决定了某个分量的下降能否被其他分量补偿。

组合算子分四类：
    ADDITIVE     R = Σ w_c · x_c                 分量可互相补偿
    MULTIPLICATIVE R = Π x_c                     任一分量→0 则总 reward→0，不可补偿
    GATED        R = (Σ w_c · x_c) · Π gate_c    门控分量具乘性否决权
    UNBOUNDED    含负距离项 / 稀疏大额事件奖励    return 尺度无界、被罕见事件主导

`min_group` 表示 reward 内用 `min(a, b)` 聚合的分量组——min 本身就是瓶颈算子，
其中一个分量被压低时，另一个分量升高完全无效。

字段说明：
    terms      加性项 {info_key 或 复合表达式: 权重}
    factors    乘性因子（列出 info 中可观测的）
    gates      门控因子（乘在整个加性和之外）
    min_groups 形如 [(w, [key_a, key_b])]，reward 中以 min() 聚合
    unobserved reward 用到但 info dict 未导出的分量（指标必须显式承认这部分盲区）
    direction  {info_key: -1} 表示该分量越小越好（距离类），默认 +1
"""

ADDITIVE = "additive"
MULTIPLICATIVE = "multiplicative"
GATED = "gated"
UNBOUNDED = "unbounded"

SPEC = {
    # ---------------- 纯乘性：任一因子归零则总 reward 归零 ----------------
    "h1hand-hurdle-v0": dict(
        kind=MULTIPLICATIVE,
        factors=["small_control", "stand_reward", "move", "wall_collision_discount"],
        source="basic_locomotion_envs.py::Hurdle",
    ),
    "h1hand-stair-v0": dict(
        kind=MULTIPLICATIVE,
        factors=["stand_reward", "small_control", "move"],
        source="basic_locomotion_envs.py::ClimbingUpwards",
    ),
    "h1hand-slide-v0": dict(
        kind=MULTIPLICATIVE,
        factors=["stand_reward", "small_control", "move"],
        source="basic_locomotion_envs.py::ClimbingUpwards",
    ),
    "h1hand-sit_hard-v0": dict(
        kind=MULTIPLICATIVE,
        factors=["small_control", "sit_reward", "dont_move"],
        source="basic_locomotion_envs.py::Sit",
    ),
    "h1hand-balance_hard-v0": dict(
        kind=MULTIPLICATIVE,
        factors=["small_control", "stand_reward", "dont_move"],
        source="balance.py::BalanceBase",
    ),
    # ---------------- 门控加性：加权和乘一个否决因子 ----------------
    "h1hand-crawl-v0": dict(
        kind=GATED,
        terms={"small_control": 0.10, "move": 0.40},
        min_groups=[(0.25, ["crawling", "crawling_head"])],
        gates=["in_tunnel"],
        unobserved={"reward_xquat": 0.25},   # 骨盆姿态项，info 未导出
        source="basic_locomotion_envs.py::Crawl",
    ),
    "h1hand-pole-v0": dict(
        kind=GATED,
        terms={"stand_reward*small_control": 0.5, "move": 0.5},
        gates=["collision_discount"],
        source="pole.py::Pole",
    ),
    "h1hand-maze-v0": dict(
        kind=GATED,
        terms={"stand_reward*small_control": 0.2, "move": 0.4,
               "checkpoint_proximity_reward": 0.4},
        gates=["wall_collision_discount"],
        bonus=["stage_convert_reward"],       # 加在门控之外的阶段奖励
        source="maze.py::MazeBase",
    ),
    # ---------------- 纯加性 ----------------
    "h1hand-door-v0": dict(
        kind=ADDITIVE,
        terms={"stand_reward*small_control": 0.10,
               "door_openness_reward": 0.45,
               "door_hatch_openness_reward": 0.05,
               "hand_hatch_proximity_reward": 0.05,
               "passage_reward": 0.35},
        source="door.py::Door",
    ),
    "h1hand-spoon-v0": dict(
        kind=ADDITIVE,
        terms={"stand_reward*small_control": 0.15,
               "hand_tool_proximity_reward": 0.25,
               "reward_spoon_in_cup": 0.25,
               "spoon_spinning_reward": 0.35},
        source="spoon.py::Spoon",
    ),
    "h1hand-powerlift-v0": dict(
        kind=ADDITIVE,
        terms={"stand_reward*small_control": 0.2, "reward_dumbbell_lifted": 0.8},
        source="powerlift.py::Powerlift",
    ),
    "h1hand-room-v0": dict(
        kind=ADDITIVE,
        terms={"stand_reward*small_control": 0.2, "room_object_organized": 0.8},
        source="room.py::Room",
    ),
    "h1hand-window-v0": dict(
        kind=ADDITIVE,
        terms={"moving_wipe_reward": 0.5 * 0.4,
               "hand_tool_proximity_reward": 0.5 * 0.4,
               "window_contact_total_reward": 0.5},
        unobserved={"stand_reward*small_control*head_window_distance_reward": 0.5 * 0.2},
        source="window.py::Window",
    ),
    # ---------------- 无界 / 事件主导：return 尺度不可比 ----------------
    "h1hand-cabinet-v0": dict(
        kind=UNBOUNDED,
        terms={"stand_reward*small_control": 0.2},
        event_terms={"subtask_complete": 100.0},   # +100×subtask，且 subtask 切换会换 reward 函数
        stage_dependent=True,
        source="cabinet.py::Cabinet",
    ),
    "h1hand-package-v0": dict(
        kind=UNBOUNDED,
        terms={"stand_reward*small_control": 1.0,
               "dist_package_destination": -3.0,
               "dist_hand_package_left": -0.1,
               "dist_hand_package_right": -0.1,
               "package_height": 1.0},
        event_terms={"success": 1000.0},
        direction={"dist_package_destination": -1,
                   "dist_hand_package_left": -1, "dist_hand_package_right": -1},
        source="package.py::Package",
    ),
    "h1hand-push-v0": dict(
        kind=UNBOUNDED,
        terms={"target_dist": -1.0, "hand_dist": -1.0},
        event_terms={"reward_success": 1.0},
        direction={"target_dist": -1, "hand_dist": -1},
        source="push.py::Push",
    ),
    "h1hand-truck-v0": dict(
        kind=UNBOUNDED,
        event_terms={"packages_picked_up": 100.0, "packages_on_table": 100.0},
        terms={"reward_robot_package_truck": 1.0},
        source="truck.py::Truck",
    ),
}

# reward 里出现、但属于"通用姿态/能耗"而非任务专属目标的分量。
# 它们几乎在所有任务里都存在，是 loco 源最容易推高的部分，
# 也正是 return 被"看起来在进步"污染的主要来源。
GENERIC_TERMS = {
    "small_control", "stand_reward", "standing", "upright",
    "per_timestep_reward", "dont_move",
}


def is_bounded(target: str) -> bool:
    """return 是否有界。无界任务的 U 标签受罕见事件主导，可测性天然更差。"""
    return SPEC[target]["kind"] != UNBOUNDED
