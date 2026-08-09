# N1 Final Mechanism Gate 结果：`NDT_NOT_SUPPORTED`

> 日期：2026-08-09  
> 预注册：`docs/experiments/n1_non_displacing_transfer_prereg_20260809.md`  
> 冻结裁决：`docs/data/n1_non_displacing_transfer_v1/n1_verdict.json`

## 1. 结论先行

工程 gate 全部通过，但 **Non-Displacing Transfer 的完整算法假设未获支持**。

- 普通 physical-uniform replay 相对 fixed source quota 有弱的方向性优势：
  `FP - FF = +41.33`，5 个 learner seed 中 4 个为正；但 90% learner-seed CI
  为 `[-16.22, +98.87]`，跨 0。因此只按预注册记为
  `DIRECTIONAL_SUPPORT`，不能写成已稳定证明的性能改进。
- 只让 source 控制 `legs_torso` 并未优于同时控制 `legs_torso+arms`：
  `LP - FP = -1.92`，仅 2/5 seed 为正，裁决为
  `DIRECTIONAL_REFUTATION`。
- 最关键的恢复性检验明确失败：`LP - S = -93.49`，0/5 seed 为正，
  90% CI `[-136.07, -50.90]`。保护 arms 后仍是稳定负迁移。

因此，本轮不进入自动 action mask、自动 group discovery 或更多 NDT 扩展；不新增
seed、stage、task、source 或 catch-up replay 臂。

## 2. 理论口径

本轮接受并落实了对旧解释的关键修正。设 source 在 continuation 第 `x` 步的
buffer 物理占比为 `rho(x)`：

- fixed quota 每一时刻强制 `q(x)=m`，会相对当前物理数据量放大 source；
- physical-uniform 每一时刻满足 `q(x)=rho(x)`，所以瞬时
  `A_inst=1`；
- continuation 全程累计的 source sample fraction 约 `0.1534`，低于终点物理占比
  `0.25`，是 source 较晚进入造成的 cohort lifetime exposure，不是 physical replay
  欠采样。

因此没有建立所谓“累计 A=1”臂；那会变成新的 age-compensated prioritization，
不再是 neutral replay。

## 3. 设计

目标为 `h1hand-truck-v0`，使用 fresh learner seeds `4,5,6,7,8`。每个 seed
先独立训练 student 到 10k，随后从同一个完整 anchor、同一个 resume noise seed
分叉到 20k：

| 臂 | behavior authority | replay |
|---|---|---|
| S | exact student | student only |
| FF | source 控制 `legs_torso+arms` | fixed/shared quota |
| FP | source 控制 `legs_torso+arms` | physical uniform |
| LP | source 只控制 `legs_torso`，arms 由 student 控制 | physical uniform |

其余保持一致：同一 hurdle4 source bank、source mass 0.5、horizon 25、
`bootstrap_only`、`NUM_UPDATES=2`、FastTD3 100k 学习率日程。终点使用独立的
128-episode deterministic source-free evaluator。

## 4. 工程验收

20/20 分支均从各自 10k anchor 恢复并生成唯一 20k checkpoint；20/20 评估均为
128 episodes。全部配置、剂量、replay 语义和 provenance 检查通过：

- S：source execution 与 critic sample 均严格为 0；
- FF：behavior source share `0.4971–0.5026`，累计 critic source share
  `0.4951–0.4957`；
- FP/LP：behavior source share 与 FF 配对一致，累计 critic source share
  `0.1516–0.1546`，符合 physical-uniform 的 cohort 期望；
- FF/FP 的两组 source provenance 相等且非零；LP 仅 `legs_torso` 非零，arms
  严格为 0。

所以科学失败不能归因于 arm 未真正执行、剂量漂移或 source-free 评估失败。

## 5. Source-free return

| seed | S | FF | FP | LP |
|---:|---:|---:|---:|---:|
| 4 | 975.74 | 853.07 | 911.34 | 925.37 |
| 5 | 1006.82 | 898.08 | 835.17 | 854.28 |
| 6 | 1034.78 | 924.64 | 981.17 | 969.28 |
| 7 | 1088.44 | 931.00 | 1025.55 | 1018.75 |
| 8 | 1073.66 | 908.17 | 968.37 | 944.32 |

### 冻结主对比

| 对比 | per-seed difference | mean | 90% CI | 正向 seed | 裁决 |
|---|---|---:|---:|---:|---|
| `H_R = FP-FF` | `+58.27, -62.92, +56.53, +94.55, +60.20` | +41.33 | [-16.22, +98.87] | 4/5 | DIRECTIONAL_SUPPORT |
| `H_A = LP-FP` | `+14.03, +19.12, -11.89, -6.79, -24.05` | -1.92 | [-19.17, +15.33] | 2/5 | DIRECTIONAL_REFUTATION |
| `H_REC = LP-S` | `-50.37, -152.54, -65.50, -69.69, -129.34` | -93.49 | [-136.07, -50.90] | 0/5 | DIRECTIONAL_REFUTATION |

描述性地，`FF-S` 平均为 `-132.89`，`FP-S` 平均为 `-91.57`。physical replay
减轻了部分平均伤害，但 FP 仍在 5/5 seed 低于 S，且 seed 5 上 FP 反而比 FF 更差。
因此只能说 fixed quota amplification **可能是负迁移的一部分**，不能说它是充分解释。

## 6. 科学解释与边界

1. **Replay 结论被收窄，而非全盘否定。** fresh seeds 的方向与
   “fixed quota 会放大晚进入 source”一致，但 learner-seed 不确定性仍大；可保留为
   机制证据/设计原则，不能单独包装成稳定性能贡献。
2. **arms authority displacement 假设被否定。** 当前手工 oracle mask 没有改善
   FP，不能据此发展自动 mask。该结果也说明 Truck 早期负迁移不是简单的
   “loco source 覆盖 arms”单因解释。
3. **NDT 恢复假设被明确否定。** 即使使用 neutral physical replay 并把 arms
   交回 student，source legs scaffold 仍在 5/5 seed 造成负迁移。
4. **不外推成一般定理。** 结论限定于 Truck、该 hurdle4 bank、10k→20k、
   约 0.5 behavior dose。它足以关闭本项目中的这条扩展路线，但不证明所有任务中的
   physical replay 或局部行为迁移都无效。

## 7. 决策

按预注册停止 N1 机制搜索：

- 不开发 catch-up/A=1 replay；
- 不开发自动 action-subspace mask；
- 不换 target/source/stage 或增加 seed 抢救；
- 后续主线回到已有正证据：reward-bearing bootstrap、source identity、exact
  abstention、有限暴露与 replay lifecycle；physical-uniform 可作为更合理的默认
  replay 设计候选，但需要在真正的正迁移场景中作为方法配置验证，不能凭 N1 单独立项。

