# 迁移效用不是 (source, target) 的属性：一个重新刻画

> 2026-07-31。**本文只组织已裁决的既有证据，不含任何未完成实验的预判。**
> 用途：把散落的负结果组织成一个可检验的理论主张，作为论文的问题设定。

## 1. 文献与本项目此前共有的隐含假设

几乎所有 transferability metric 工作（LEEP / LogME / SFDA / ETran，以及本项目的
十二个信号族）都在做同一件事：

> 给定 `(source, target)`，预测一个标量——**"这个源对这个任务有多有用"**。

这个提法预设了：**迁移效用是 `(source, target)` 对的一个确定性属性**，
只是我们暂时不知道它，需要一个好的估计器。

本项目此前的全部工作也建立在这个预设上：`U_i(t,d,K)` 被当作待估的真值，
`EQD30K` / `sibling gate` / `door gate` 被当作它的"ground truth 标签"。

## 2. 三层证据，逐层削弱这个预设

| 层 | 发现 | 出处 |
|---|---|---|
| **L1 · 不可零成本预测** | 十二个信号族、跨七个信号空间（行为 / 即时 reward / 梯度 / critic / reward 结构 / 任务定义 + 静态规格 / 任务进度）系统性失败 | `impossibility_characterization_of_transfer_prediction_20260730.md` |
| **L2 · 通道归因不是函数** | 固定 `(source,target,stage,剂量,anchor,噪声种子)`，行为/replay 通道的归因仍跨 seed 翻转，且非评估噪声（episode `\|U\|/SE` 10–20） | `M17`；door channel decomposition |
| **L3 · 数值不可复现** | 同协议重跑，door `\|ΔU\|` 中位 **24.23**、最大 **43.78**，而效应量本身只有 `−7~−43`；hurdle 约 15 | `M27`；`racing_reject_door_v2_results` |
| **L4 · 符号本身反转** | door：gate `s1–3` 与 holdout `s4–6` 共 **18/18** per-seed 为负；新批 `s7–9` 出现 **2/9 为正**，其中 `s9` 的 `run = +36.32 ± 3.95` **显著正**（gate 结论 `−30.63`，跨度 **67**） | `M31`；`racing_reject_door_v4_results` |

L1 说"估计器都不行"。L2–L4 说的是另一回事：**被估计的那个量，可能不是确定性的。**

## 3. 重新刻画

```
旧提法：  U = f(source, target)                      待估的确定性标量
新提法：  U ~ p( U | source, target, θ_t, D_t, ... )   随 learner 分布的随机变量
```

L4 是决定性的一步：此前 L2/L3 只说明**大小**不稳，而 L4 说明在某些 target 上
**连方向都不稳**——即"这个源对这个任务有用还是有害"这个问题，
在 door 上**没有良定义的答案**。

## 4. 由此解释十一族的失败（一个更强的说法）

此前本项目对十一族失败的解释是"即时量与延迟学习效用之间没有可用的关系"。
L4 允许一个更强、也更简洁的解释：

> **十一族在预测一个可能不存在的确定性量。**

若 `U` 在某 target 上是符号都会翻转的随机变量，那么任何只读
`(source, target)` 的函数——无论它多聪明——都**原理上**无法预测它，
因为输入里根本不含决定符号的信息（`θ_t` 与 occupancy）。

这与 §5.1 的正面结果并不矛盾：racing 之所以可行，正是因为它**不预测**，
而是在真实的 `θ_t` 上直接测量。

## 4.1 一个反转：L4 不是 racing 的威胁，而是它的**理由**

我此前一直把 `M31` 当作对 racing 的威胁——"连 ground truth 都会翻转，还怎么验证选源"。
这个读法搞错了对象。关键区分：

```
跨 learner 推广："源 X 对任务 Y 普遍有用吗"   →  L4 说：可能没有答案
同 learner 选择："对**这个** learner，哪个源最好" →  racing 直接测，不需要推广
```

**racing 的估计与它的应用发生在同一个 learner 上**，因此它**根本不需要**跨 learner 推广。
真正被 L4 否定的，是"静态源库排名"这一整类做法——
包括本项目自己此前把 `EQD30K` / `gate` 的 per-seed 值当作可复用标签的用法。

于是论证链闭合成一个正向命题：

> **正因为迁移效用不可跨 learner 推广，才必须为每个 learner 单独测量；
> 而这恰好是 racing 在做的事，且代价可控（hurdle 上 30k 步）。**

这也把"racing 是测量不是指标"从批评变成了定位：
在一个不存在可预测量的问题上，**测量就是正确答案**，指标才是错的提法。

## 5. 但必须严格限定：这不是普遍主张

**证据只支持"在某些 target 上"，不支持"普遍如此"。**

| target | 符号稳定性 | 证据 |
|---|---|---|
| **hurdle** | **稳定** | `run` 的 `U = +379.66`，CI90 `[+271.5,+487.9]`，远离零点；`RACING_K` 两批 6 个 learner 全部选中 run |
| **door** | **不稳定** | 18/18 负 → 新批 2/9 正，且其中一个显著正 |
| slide | **未知** | 仅 3 个 learner（`s1–3` 上 argmax 3/3 为 walk）；审计已预注册（`2c1804f`），进行中 |

因此本文的主张限于：

> **迁移效用的符号稳定性因 target 而异，不能默认。**
> 在符号不稳定的 target 上，"该选哪个源"没有良定义的答案；
> 在符号稳定的 target 上（hurdle），直接测量（racing）可行且代价可控（30k 步）。

## 6. 实践含义（三条可执行）

1. **任何"源 X 对任务 Y 有用/有害"的结论，必须报告其 learner 分布上的稳定性。**
   本项目已把 `door_at10k_gate_v1` 的结论限制为"在 seeds 1–6 上"。
2. **投入选源实验之前，先做标签可推广性审计**（`M31`）——
   否则可能在一个 ground truth 会翻转的场地上白跑（door 用掉了 v1–v4 四轮）。
3. **`n=3` 的 learner 面板不足以支撑符号结论**：door 用了 18 个 per-seed 效应
   仍被第 19 个推翻。

## 7. 与既有文献的关系

`Benchmarking Transferability`（arXiv 2504.20121, 2025）在**图像分类**下报告
transferability metrics 的脆弱性（ETran 的平均相关性从 0.562 掉到 0.143）。
但分类设定里源与目标都是**静态特征 + 固定标签分布**，不存在正在变化的 learner。

本文的 L4 是 RL 特有的：**同一个 `(source, target)` 对，换一批 learner 就换符号。**
据我们所掌握的材料，这一现象此前未被报告——
但**这是文献覆盖度的限制，不构成新颖性主张**；本项目未做系统性的文献检索来支撑"首次"。

## 8. 尚未做的检验（本文不主张的部分）

- slide 的审计正在进行；若它也不可推广，则 L4 不是 door 独有，主张会显著加强；
  若它可推广，则"因 target 而异"得到第二个正面案例。
- 未刻画符号稳定性与什么有关（效应量距零点的距离？任务的随机性？）——
  仅有的观察（door 效应贴近零点、hurdle 远离零点）**只有 2 个点，不足以成模式**（见 `M29`）。
- 未在 `EQD30K` / `sibling gate` 的其他 cell 上做可推广性审计。

## 9. 出处

```
L1  docs/impossibility_characterization_of_transfer_prediction_20260730.md
L2  M17；docs/experiments/door_channel_decomposition_v1_results_20260728.md
L3  M27；docs/experiments/racing_reject_door_v2_results_20260731.md
L4  M31；docs/experiments/racing_reject_door_v4_results_20260731.md
正面结果   docs/experiments/hurdle_speedup_v1_results_20260730.md
           docs/experiments/racing_min_horizon_v1_results_20260730.md
证据总表   docs/EVIDENCE_STATE_20260731.md
```
