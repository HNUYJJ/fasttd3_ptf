# interface_transfer_v1 — 可控接口作为迁移载体（隔离实验目录）

> **与 bootstrap 主线完全隔离。** 本目录不 import 任何 `fasttd3_ptf.*`，
> 只使用 `vendor/` 下自带的 HumanoidBench 与 FastTD3 副本；
> 所有新代码、文档、数据、checkpoint、日志都留在本目录内。
> 主线的任何文件都**未被修改**。

## 1. 这个方向要回答什么

三方讨论（Claude / ChatGPT-5.6-sol / ChatGPT-5.6-Pro）收敛出的问题：

> 过去我们把**固定 task policy**（stand/walk/run/hurdle）当作迁移单位。
> 一个**可被目标策略连续控制的接口**，是否能比固定 policy 产生更有用的
> target-domain 经验，并且这些收益能在**完全移除接口后**被 flat student 保留？

对应三个层次：

```
Generation  →  Assimilation  →  Retention
（源能否产生   （learner 能否   （移除源后
  有用经验）     吸收成自主策略）   收益能否保留）
```

主线已较深入研究 Generation 与 Retention；**Assimilation 从未作为独立问题**。

## 2. 分阶段，且每阶段都有停止规则

### Phase 0（当前）— 接口 positive control
只回答一件事：**当前 FastTD3 栈能否利用一个已知可靠的连续控制接口？**
这是工程与方法的 positive control，**不是创新实验**。

停止规则：若连 Push 上的 first-contact / box progress / 学习曲线都不能改善，
**立即关闭整条 interface 路线**——不训练 locomotion skill、不加 residual、
不做 Lift、不做多技能组合。

### Phase 1 — source-free consolidation gate（Phase 0 通过后才启动）
四臂对照：flat scratch / 固定 Reach policy bootstrap /
接口辅助的 bounded bootstrap / persistent HRL（非 source-free 上界参照）。

关键约束：接口只在限定 warmup 参与；replay 存**真实的低层 61 维 action**
与 target reward；到期彻底移除接口；**主结果只评估 flat source-free student**。

若 persistent HRL 有效但 source-free student 不保留收益，结论是
"接口是更好的部署架构，但不是更好的迁移 bootstrap"——**此时不扩建 skill library**。

## 3. 已完成

| 项 | 结果 |
|---|---|
| 官方 hierarchy 接口透传（主线 `humanoid_bench_env.py` 从未传 `policy_type`） | `src/hb_interface_env.py` |
| push 环境构造 | flat `action_space=(61,)` → 接口 `(3,)`，obs 均 163 维 |
| **接口 controllability 验证** | **RESPONSIVE**：+x setpoint 手位移 `+0.9472`，−x 为 `−0.3540`，符号相反、差 `1.3012`（阈值 0.02） |

controllability 验证是必需的，因为低层是 `TorchModel(55, 19)` 而 h1hand 是
obs 163 / action 61，中间索引映射（`body_idxs`/`act_idxs`）若错位，
环境仍能 step 且不报错（silent corruption）。

## 4. 已知的接口语义（读自 vendor 源码，非猜测）

- 高层 action = 末端目标的**增量**，乘 `max_delta=0.1` 后累加，
  再 clip 到任务自带的 `htarget_low/high`（`wrappers.py:82-130`）；
- 低层只输出 **19 维 body action**，**手部不受其控制**——
  故 Push 这类只需接近的任务可用，而需要抓握的任务它本就没有能力覆盖；
- 官方结论也正是：Reach hierarchy 对 Push 明显有帮助，对需要 lifting 的
  Package 帮助有限（controller 没学过 lifting）。

## 5. 目录

```
interface_transfer_v1/
├── vendor/          官方 HumanoidBench + FastTD3 副本（只读，未改动）
│   └── humanoid_bench_pkg/data/reach_{one_hand,two_hands}/  预训练低层权重
├── src/             本方向的代码
├── scripts/         实验与验证脚本
├── configs/  docs/  data/  models/  logs/
```

**注意**：`vendor/.../*.pt` 与 `models/*.pt` 属权重文件，
按仓库 `.gitignore` 不进版本控制；克隆后需自行补齐官方 HumanoidBench 的
`data/reach_*` 三件套（`torch_model.pt` / `mean.npy` / `var.npy`）。

## 6. 与主线结论的关系

本目录**不推翻**主线任何已冻结的 verdict。需要同时记住的主线事实：

- 同一 `hurdle4` bank 在 truck 上 **20k 显著负、95k 显著正（+229.9, t=3.47）**
  （`docs/experiments/truck_timescale_contradiction_20260811.md`）——
  所以"固定 policy 载体不行"这个论断**尚未成立**，本方向是并行探索而非替代；
- 因此 Phase 1 的对照必须包含"固定 Reach policy bootstrap"臂，
  且评估窗口不能只到 20k。
