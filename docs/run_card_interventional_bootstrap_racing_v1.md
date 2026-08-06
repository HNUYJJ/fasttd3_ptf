# Interventional Bootstrap Racing (IBR) v1 run card

> 状态：工程实现前冻结草案，2026-08-01。本文不授权正式多任务矩阵；
> 先做工程 gate，再做一个 Slide hard-exit 科学 gate。所有判断只使用
> source-free student evaluation。

## 1. 核心问题

固定比例的 reward-bearing bootstrap（RBO）能否被改造成一个有限、可选择、
可退出的迁移干预：先用真实短分支测量候选 source 对当前 learner 的作用，
选择分支的完整 learner state，并在选择后同时关闭 source behavior 与 source
replay，继续纯学生训练？

这不是零交互迁移性指标。它把此前无法可靠解决的“注入前预测”改成一个有
明确成本的直接干预测量问题。

## 2. 直接证据与动机

Slide speedup v1 的冻结结果为 `SPEEDUP_REFUTED`，但失败形态是明确的
“早期增益 + 后期上限受损”，而不是 source 无效：

- 10k/20k/30k 的 paired return 增量均为 3/3 正；均值分别为
  `+86.04 / +146.50 / +153.52`；
- 10k--30k normalized AUC 增量为
  `[+109.84, +142.51, +147.08]`；
- 50k 后开始反转，100k paired return 为
  `[-633.12, -289.99, -574.08]`；
- continuous-walk 100k 均值 `293.35`，scratch 为 `792.41`；
- behavior share 为 `0.4766--0.4788`，critic share 为 `0.5000`，不是剂量
  执行失败。

因此当前最小科学假设是：

> H1：Slide 的后期失效主要来自 source 暴露持续过久；在同一 30k walk
> branch state 上同步终止 behavior authority 与 replay eligibility，能够保留
> 既得 learner state，同时解除 continuous RBO 的后期学习上限。

H1 不声称 source 一定优于 scratch，也不声称找到了通用迁移性指标。

## 3. 方法定义

一个分支 anchor 包含：actor/critic/target、optimizer、scheduler、AMP scaler、
replay（含 provenance/priority/admission audit）、全局 RNG 和完成步数。它不含
MuJoCo simulator state，因此 continuation 的 estimand 是 reset-start learner
continuation，而非同一 episode 的无缝续接。

在分支步 `K`：

1. 保存完整 branch anchor；
2. continuation 从该 anchor 恢复核心 learner/replay/RNG；
3. 用运行时 `admission_mode=none` 覆盖 anchor 内旧的 admission policy；
4. 立即清除 source latch；
5. `K` 之后 source behavior execution 增量严格为零；
6. `K` 之后 critic source sample 增量严格为零；
7. source transition bytes 可留作审计，但不再 active 或可采样。

## 4. 第一阶段：工程 gate

只允许 additive 接口，不改变历史默认语义：

- `branch_anchor_step/branch_anchor_dir`：允许 source 分支在指定完成步保存完整
  anchor；
- core resume 后必须重新应用运行时 admission policy；
- branch anchor 恢复累计 behavior execution counts 与 admission history；
- 旧 `anchor_step` 继续只允许 fresh empty-bank scratch；
- 旧 replay snapshot import 默认行为不改。

必须通过：

1. CPU branch-anchor round trip：核心参数、optimizer/scheduler、replay ptr、
   provenance、priority、RNG、update counts 连续；
2. runtime-policy precedence：anchor 内 source policy 不能覆盖 continuation 的
   exact abstention；
3. hard-exit delta：source behavior/sample delta 均为 0，同时 replay ptr 和
   learner update counts继续增长；
4. 200-step HumanoidBench smoke。

不要求 GPU 逐 bit 长跑等价。

## 5. 第二阶段：Slide hard-exit 科学 gate

### 5.1 矩阵

- target：`h1hand-slide-v0`；seeds `1,2,3`；
- branch：复现冻结 continuous-walk 配置，运行至 `K=30000` 并保存完整 branch
  anchor；
- treatment：从各自 30k branch anchor 继续至 100k，运行时 exact abstention，
  behavior 与 replay 同步硬退出；
- matched control：从**同一个** 30k branch anchor 继续至 100k，保持
  admission `all`；两条 continuation 使用同一 resume-noise seed；
- external controls：已完成的同 seed continuous-walk 与 scratch，仅用于复现
  一致性与绝对水平解释，不替代 matched control；
- frozen evaluator：同一 128-episode source-free panel；
- checkpoints：30k/50k/75k/100k。

30k 前 treatment 与 matched-continuous 必须来自同一个 branch anchor，不重新
训练两份“看似相同”的前缀。正式新跑为每 seed 一个 prefix 加两个 continuation，
共 9 个训练进程。

### 5.2 主指标与裁决

定义：

- `D_exit(s) = J_exit,100k(s) - J_continuous,100k(s)`；
- `A_exit(s)` = 30k--100k source-free normalized AUC 的同 seed 差；
- `D_scratch(s) = J_exit,100k(s) - J_scratch,100k(s)`。

三 seed 均值使用配对 90% t 区间（`t_0.90,2=1.8856`）。固定裁决：

- `HARD_EXIT_SUPPORTED`：`D_exit` 与 `A_exit` 的 90% LCB 均大于 0，且
  `D_scratch` 不显著小于 0；
- `HARD_EXIT_PARTIAL`：`D_exit` 或 `A_exit` 明确为正，但另一项不确定，或
  hard-exit 仍显著落后 scratch；
- `HARD_EXIT_REFUTED`：`D_exit` 与 `A_exit` 都未显示改善，或任一项明确为负；
- 工程审计失败单列 `ENGINEERING_INVALID`。

不得把 `PARTIAL` 改写成支持，也不得通过移动 K、增加 seed、改变 evaluator
或降低剂量来救结果。

## 6. IBR 的条件升级

只有 `HARD_EXIT_SUPPORTED` 才升级为完整 IBR：同一 base learner anchor 下
运行 student/stand/walk/run 的 K-step 分支，以冻结 source-free selection panel
选 winner，再 hard-exit 续训。

正式 IBR 必须包含：

1. student-only best-of-N（population/order-statistic control）；
2. 从同一 mixed branches 随机选 winner（selection control）；
3. fixed-source oracle reference；
4. 单 scratch、按总环境交互量计费的 economic control；
5. selection panel 与 final panel 的 seed 集不重合。

若不能超过 best-of-N/random control，IBR 只能作为实验协议，不作为核心算法
贡献。其可主张上限是 learner/anchor-conditioned 的跨任务 RBO 直接干预测量与
有限生命周期控制，不是首次 racing、通用迁移性指标或无成本 source selector。

## 7. 停止条件

- 工程 gate 失败：只修 conclusion-relevant wiring，不启动科学实验；
- Slide hard-exit 被否证：停止 IBR/CBR 的 Slide 驱动路线；
- hard-exit 仅改善 continuous-walk、仍显著不如 scratch：保留为 lifecycle
  诊断结果，不升级论文核心；
- 禁止在揭盲后调 K、阈值、panel、source dose 或统计口径。
