# 迁移效用预测的不可能性刻画：十二个信号族的统一失败

> 2026-07-30 整理。**本文只汇总已完成实验的既有证据，不含任何未完成实验的预判。**
> 用途：论文的动机与背景，以及下一步方向的依据。

## 1. 问题设定

给定一组**冻结的**源策略 `{π_1,…,π_N}`（在其他任务上训练完成），
一个 target 任务，以及一个正在学习的 student `θ_t`，问：

> 应当把哪个源、在什么时候、以多大剂量注入 student 的训练，才能加速学习？

本项目把"某个源值不值得用"形式化为**因果干预标签**：

```
U_i(t, d, K) = J_sf( θ_i 在 t+K )  −  J_sf( θ_student 在 t+K )
```

`J_sf` = **source-free** student 的确定性评估（源在评估时不在场），
配对到 learner seed，等剂量 `d`，干预窗口 `[t, t+K]`。
这是全项目不变的口径：**最终评估的永远是不带源的 student**。

## 2. 十二个信号族，全部失败

按时间顺序。每一族都独立预注册、独立裁决。

| # | 信号族 | 它测的量 | 失败方式 |
|---|---|---|---|
| 1 | zero-shot 行为 return / 位移 | 源在 target 上的即时表现 | door-run 位移 +58% 却 harmful；walk −61% 却最不负（M19） |
| 2 | T⁰ | t=0 的即时干预效果 | 符号与延迟效用不符 |
| 3 | SIV（source intervention value） | 执行段即时 reward | 引导型好源做"脏活"，即时 reward 低（M1） |
| 4 | SHU | 同上族 | 同上 |
| 5 | adaptive revocation | 在线即时 reward 触发撤销 | 同上（M1 三重否定） |
| 6 | P0 lease | 租约期即时收益 | 同族同因 |
| 7 | update-space influence | 梯度内积 | FAIL 且**排序反转**——最有益的 hurdle_s2_run 被判最有害；诊断为"即时分布错配，非延迟学习价值" |
| 8 | T^critic | `E_{s~B_t}[ minQ(s,π_i(s)) − minQ(s,π_stu(s)) ]` | 期望符号显式写在定义里；符号负偏 |
| 9 | BAC（bottleneck-aligned coverage） | rollout 实测的 reward 分量覆盖/损伤 | 相对 per-step reward 与主进度分量**无增量**（P2/P3/P4 三者打平） |
| 10 | per-state QMP-fidelity | `argmax_i min_h Q_h(s,π_i(s))`，**无任何聚合** | 退化为 student（source_share 0.3–5.5%）；critic 对源的即时优势 **18/18 组合为负**、**2/2 任务排序错误** |
| 11 | 规格匹配（零交互） | target reward 的数值规格 + 源的训练规格常量 | 在自身最佳案例上被证伪（见 §3.2）；另有多处源码事实错误 |
| 12 | task progress（零训练） | 源在 target 上 zero-shot 的**前进进度**（位移），只用于单向排除 | `HOLDOUT_FAILED`；与效用**反向 7.9×**，且任何单调阈值都不可能成立（见 §3.1.1）|

## 3. 三条统一的刻画

### 3.1 即时量与延迟学习效用之间没有可用的关系

族 1–8 与 10 测的都是**即时**量（行为回报、即时 reward、梯度错配、critic 优势）。
最强的直证来自族 10：它取消了所有聚合，在每个状态上直接问 target 自己的 critic，

```
score_src − score_stu  在 (task, source, seed) 的 18/18 组合上全为负
                        包括 slide 上真实效用 +56.95 的 walk 源
critic 的排序          两个任务上都排 run > walk，而 walk 才是真正最有用的
```

**这排除了"聚合方式不对"这一解释**：问题不在如何聚合，而在被聚合的量本身。

#### 3.1.1 补充（2026-08-04，族 12）：连"单向排除"这个最弱的用法也不成立

族 1 的失败可能被辩护为"用法过强"——它试图给源**排序**。族 12 因此把要求
降到最弱：**只做单向排除**（进度≈0 ⇒ 拒绝），不做排序；并且把测量量从
`return` 换成 `task progress`（前进位移），因为 return 会被姿态、门控等
无关分量污染。**这两处降级都不能挽救它。**

zero-shot、不训练、32 episodes、deterministic 的实测（`P` = `max_t(x_t − x_0)`，米）：

| 环境 | stand | walk | run | 该 target 的真实最优源 |
|---|---:|---:|---:|---|
| crawl（三源**全有害**） | 0.221 | 3.664 | **14.302** | 无（U = −448 / −217 / −208）|
| hurdle | 0.188 | 8.717 | **22.521** | run（U = +379.66）|
| slide | 0.183 | **1.814** | 1.753 | walk（U = +56.95，`GEN_OK` 6 learner）|

**核心反例**：`crawl` 上的 `run`（U = −208.07，有害）位移 **14.302**，
而 `slide` 上的 `walk`（U = +56.95，最有用）位移 **1.814**——
**有害源走得比有用源远 7.9 倍**。

**不可能性论证（纯代数，不依赖任何机制解释）**：

```
单向排除要成立，必须同时满足
    P(run,  crawl) < θ      （拒绝有害源）
    P(walk, slide) > θ      （保留有用源）
即需要
    14.302 < θ < 1.814      ← 空集
```

**不存在任何阈值**。换取法、换分位数、换归一化都不改变不等式方向。
这比族 1 的 door 反例（位移 +58% 却有害）强一个数量级，
并把"行为量与效用**反向**"从个案提升为可判定的结构性结论。

附带更正一条曾被误信的直觉：`walk` 在 slide 上被斜坡阻断了 94%
（平地 31.388 m → slide 1.814 m），却是唯一稳定正迁移的源。
**"被 target 结构阻断"不蕴含"对学习无用"。**

> 出处：`docs/experiments/progress_screen_v1_results_20260804.md`
> （预注册与判据于 `788afa0` 先于数据冻结）。

### 3.2 任务定义层面的相似性也不行

三条独立路线收敛：

- reward 代数签名同构：slide/stair 同族，U 为 +56.95 与 +0.19；
- reward 分量覆盖（BAC）：无增量；
- **完全相同的 reward 实现**（sibling gate，前瞻预注册）：方向依赖，
  slide 源→stair `+15.40 [3/3 正]`，stair 源→slide `−20.79 [0/3 正]`。

族 11 死在同一处，而且更彻底：**slide 与 stair 共用同一个
`ClimbingUpwards.get_reward`、数值常量逐字节相同**，walk 的效用却是
`+56.95` 与 `+0.19 [−5.35,+5.72]`。任何只读 (source, target) 静态规格的量，
在这两个 target 上读到的输入完全一样，因而**原理上**无法产生这个差异。

### 3.3 U 不是 (source, target) 的函数

`U` 至少还依赖 learner 状态与数据分布：

- Door 通道分解：U^BR 稳健负，但通道归因**跨 seed 反向**
  （s1/s3 行为致害、replay 补偿；s2 相反），且 episode `|U|/SE` 达 10–20，**不是噪声**；
- 同一 (source, target) 在不同 t 与不同 K 下结论不同（EQD30K vs STAGE10K 属不同 protocol family，明令不得混入同一训练集）。

因此正确的写法是条件分布

```
U  ~  p( U | source, target, θ_t, D_t, occupancy_t, channel, dose, K )
```

而全部十二族都在估计一个只含 `(source, target)`（至多加 `t`）的**点函数**。

## 3.4 文献定位：这不是本项目独有的困难

**在远比 RL 简单的监督分类设定下，transferability metrics 已被系统性基准测试证明脆弱。**

`Benchmarking Transferability: A Framework for Fair and Robust Evaluation`
（arXiv 2504.20121, 2025，**仅覆盖图像分类，不涉及 RL**）报告：

- 换源数据集即崩：ETran 的平均相关性从 **0.562**（ImageNet 源）掉到 **0.143**（CIFAR-100 源），降幅 75%；
- 模型复杂度下降时退化加剧；
- LEEP / LogME / SFDA / ETran 均依赖**有标注的 target 数据**。

而分类设定里源与目标都是**静态特征 + 固定标签分布**，不存在正在变化的 learner。
RL 里 `U` 还额外依赖 `θ_t` 与它诱导的 occupancy（§3.3），困难只会更大。
**本项目十二族的系统性失败，是这一困难在 RL 中的实证，而非实现问题。**

相关的 RL 侧工作：`An advantage based policy transfer algorithm for RL with
measures of transferability`（APT-RL，arXiv 2311.06731）——它的 transferability
度量**需要 target 环境交互**（off-policy，在 target 中学习新知识），
在三个高维连续控制任务上评估，其核心声明是"在对抗性 target 上至少不差于 scratch"。
这与本项目 door 上追求的**负迁移免疫**是同一目标，可作为对照 baseline。
值得注意的是，它同样**没有**给出零交互的静态预测器——这与 §3.1–§3.2 的结论方向一致。

## 4. 两个必须同时承认的边界

1. **这不是"迁移无效"。** 在源已知有用时，收益极大且稳健：
   `EQD30K.hurdle.run` U = **+379.66**，CI90 [+271.5,+487.9]，3 seeds，
   剂量实测 0.500–0.502；`slide + walk` U = **+56.95**，3/3 seed。
   失败的是**自动选源**，不是迁移本身。
2. **这不是穷尽性证明。** 十二族覆盖了行为、价值、梯度、reward 结构、
   任务定义、静态规格、**任务进度**七个空间，但不能排除尚未想到的信号族。
   本文的主张限于："在本项目已检验的空间内，
   任何只依赖 `(source,target)` 的**即时**量都不能预测延迟学习效用。"
   族 12 额外收紧了这条主张的**用法维度**：失败不限于"排序"，
   连最弱的"单向排除"也不成立（§3.1.1）。

## 5. 由此推出的下一步方向

既然**预测**在上述空间内不可行，剩下的合法路径是**测量**：

- 用少量真实交互直接估计 `U`，而不是从廉价代理外推；
- 成本可通过**并行**摊平（N 个源各跑 K 步，墙钟等于单臂 K 步，算力 N 倍）；
- 判据是：`选对源的收益` 是否大于 `racing 的交互成本`。

### 5.1 已完成：RACING_K v1（2026-07-30，`a744adb`）

**结论 `RACING_VIABLE`，K\* = 10000。** 详见
`docs/experiments/racing_min_horizon_v1_results_20260730.md`。

```
主判据 3/3 seed 选中 run:  K=2000 → 0/6 运行   K=5000 → 4/6   K=10000 → 6/6
成本 3×10k = 30k 步（并行墙钟 10k）  vs  选对源节省 67k 步  →  净 +37k
```

与十二族的关键区别：**estimand 未变**。族 1–12 测代理量 `X` 并假设 `X → U`
（跨量类外推）；racing 直接测 `U` 本身，只缩短 `K`，
问的是 `U(小K) → U(大K)` 这一**同量 horizon 一致性**，可直接验证。

**辨别证据（预注册时冻结）**：zero-shot 行为把 walk 排垫底
（`run 169.21 > stand 146.94 > walk 96.35`），真实 U 把 walk 排第二
（`run 379.66 > walk 104.89 > stand 51.28`）。在 `K ≥ 5000` 的
**全部 12 个独立 learner 运行**中，racing 都排出 `walk > stand`——
它做到了族 1（zero-shot 行为 return / 位移）做不到的事。

因此本文的完整主张是两半：

> **零成本预测不可行**（§1–§4，十二族）；
> **但最小成本的直接测量可行**（§5.1，hurdle 上 30k 步交互）。

前一半单独看只是一串负结果；有了后一半，"改预测为测量"才成为有动机的主张。

**边界**：单 target（hurdle）、单源集合（3 个 loco 源）、K\* 非理论下界
（只测了三个 K，K=5000 已证不可靠）。racing 需要真实交互，
是"最小测量代价"的上界，**不构成对 §1–§4 的反驳**。

## 6. 出处

```
M1/M5/M15/M16/M19            docs/ISSUES_AND_LESSONS.md
族 7  influence              docs/experiments/influence_gate_v1_results_20260727.md
族 9  BAC                    docs/experiments/predictor_baseline_comparison_v1_results_20260729.md
族 10 QMP                    docs/experiments/qmp_fidelity_v1_results_20260729.md
族 11 规格匹配                docs/experiments/spec_matching_hypothesis_refuted_20260730.md
sibling gate                 docs/experiments/sibling_source_gate_v1_results_20260729.md
通道归因跨 seed 反向          docs/experiments/door_channel_decomposition_v1_results_20260728.md
A 级 U 标签                   docs/data/transfer_effect_label_inventory_20260727.json
条件分布写法                  docs/PAPER_CONTRIBUTION_RESTRUCTURE_20260728.md:197
```
