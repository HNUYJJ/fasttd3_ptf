# 预注册：Stand-only 可行性前置探针（零训练）

> 冻结时间：2026-08-11，**先于运行任何探针**。
> 性质：**零训练判别式**（CLAUDE.md §8.2b）。纯前向 rollout，不训练、不更新参数、
> 不改动任何已冻结 verdict。目的只有一个——**在投入 Stand-only 训练之前，
> 先判断该实验的必要前提是否成立**。

## 1. 为什么先做这个

两份外部 review 都建议下一步做 Stand-only Truck 实验（source mass 全给 `stand`，
对照 hurdle-dominant mixture）。该实验的**隐含前提**是：

> `stand` 源在 truck 场景下的闭环行为，比 student 当前已有的能力"更会站"，
> 因而注入它能提升 upright。

但 N1 审计已测得：**scratch 在 20k 时 upright = 0.8776，是四臂中最高**，
且 `corr(upright, return) = +0.938`。truck 0→20k 的 reward 为
`upright × (1 + reward_robot_package_truck)`。

因此若 `stand` 源在 truck 场景的 upright **低于** scratch 已达到的水平，
那么注入它只会把 student 从更好的策略拖走——Stand-only 在开跑前就注定失败，
而这一点用一次纯前向 rollout 就能看出来，不必付 5 seeds 的训练成本。

## 2. 探针设计

在 truck 环境 (`h1hand-truck-v0`) 上做 deterministic zero-shot rollout：

| 条件 | 动作来源 |
|---|---|
| `stand_src` | truck bank 中 `stand` 源（经其 bank 内声明的 obs adapter） |
| `hurdle_src` | 同 bank 的 `hurdle` 源（N1 中占 55.5% 执行的主导源） |
| `walk_src` / `run_src` | 同 bank 其余两源，作谱系参照 |
| `scratch_20k` | N1 的 S 臂 20k checkpoint（已有测量，作为基准线复用） |

每个条件 16 episodes × 1000 步，eval seed 与 `p0_evaluator` panel 前 16 个逐位相同。
记录 `upright`、`reward_robot_package_truck`、`return`、`progress_max_dx`、`path_length`。

**adapter 正确性自检（前置，失败则整个探针作废）**：源动作维度必须为 61 且
落在 `[-1,1]`；若任一源输出恒定向量（跨 100 步标准差 < 1e-6），记
`PROBE_INVALID_ADAPTER`——这是 obs adapter mismatch 的典型症状
（silent corruption，见 `project_obs_adapter_risk`）。

## 3. 冻结判据

令 `U_x` = 条件 x 的 episode 平均 upright，`U_scratch = 0.8776`（N1 实测，五 seed 均值）。

1. **主判据**：`U_stand ≥ U_scratch`
   → `STAND_HAS_HEADROOM`：Stand-only 的前提成立，值得投入训练。
2. `U_stand < U_scratch − 0.03`
   → `STAND_NO_HEADROOM`：`stand` 源在 truck 场景站得**不如** student 自己，
   注入它缺乏机制依据。此时 Stand-only 若仍要跑，只能作探索性，不得预期正效应。
3. 其余（落在 `[U_scratch − 0.03, U_scratch)`）
   → `STAND_MARGINAL`：无明确头寸，判断留给 PI。

阈值 0.03 的依据：N1 四臂 upright 的实测跨臂极差为 `0.8776 − 0.7525 = 0.1251`，
取其约 1/4 作为"可察觉差异"；且 0.03 远大于同臂跨 seed 的 upright 波动。
阈值在看到任何探针输出之前写定。

**附带（描述性，不参与判据）**：`U_hurdle` 与 `U_stand` 的差，用于量化
"probe 打分把静止源压到 0.004%"这一 allocation 的代价方向。

## 4. 这个探针能与不能回答什么

**能**：`stand` 源在 truck 状态分布上的**开环**姿态维持能力，是否达到
student 20k 时的水平。

**不能**：
- 不能预测 Stand-only 的**训练**结果。注入是闭环干预，会改变状态分布与 replay，
  zero-shot 好不等于注入后好（本项目已有 door gate 的反例：run 的 zero-shot
  行为最好却最有害）。
- 不能替代 Stand-only 实验本身；`STAND_HAS_HEADROOM` 是**必要非充分**条件。
- 不能推广到 truck 以外的 target 或 20k 以外的阶段。

## 5. 边界

本探针不改动 N1、T4-R、T3、T2、Gate A 的任何 verdict，不构成新方向的启动，
也不是 source selector——它只检验一个**已被两份 review 建议的实验**的前提。
