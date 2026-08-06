# 方向决策：从"预测哪个源有用"转向"逐状态让 critic 现场决定"

> 2026-07-29。触发：sibling-source gate 判 `SIBLING_DIRECTION_DEPENDENT` 后，
> 任务定义层面的相似性路线全部关闭；PI 指示换路线、调研顶会工作、把有效模块缝合进来。
>
> 本文记录：文献调研结论、对本项目全部失败的统一诊断、方向选择的理由、撞车检查、
> 以及对该方向的自我攻击。**本文是决策记录，不含任何新实验结论。**

## 1. 我们走到了哪里

三条独立路线收敛到同一个否定结论：

| 轮次 | 检验的相似性 | 结果 |
|---|---|---|
| 任务分类学 | reward 代数签名同构 | slide/stair 同族，U 为 +56.95 与 +0.19 |
| BAC 预测器比较 | reward 分量覆盖 | 相对 per-step / 主进度分量**无增量** |
| sibling gate（前瞻） | **完全相同的 reward 实现** | **方向依赖，不稳健优于通用源** |

> 任务定义层面的相似性不能预测冻结源的迁移效用。

再往前，八个信号族（zero-shot 行为、T⁰、SIV、SHU、adaptive revocation、P0 lease、
update-space influence、T^critic sign）也全部失败。

**问题不是"还没找到对的信号"，而是可能一直在问错的问题。**

## 2. 统一诊断：所有失败共享同一个数学形式

把八个信号族和三条相似性路线写成统一形式：

```
score_i = Aggregate_{s ~ D} [ f(source_i, s) ]   →  一个标量  →  用来选源
```

- 行为族：Aggregate = episode return（对状态与时间求和）
- T^critic：`T = E_{s~B_t}[ minQ(s,π_i(s)) − minQ(s,π_stu(s)) ]`
  —— 见 `scripts/analyze_tcritic_offline.py:3`，**期望符号是显式写在定义里的**
- influence：Aggregate = 梯度内积的均值
- BAC：Aggregate = reward 分量的掩码加权和
- taxonomy / sibling：连状态都不看，直接聚合到任务定义

BAC 线已经得出"标量聚合是共同失败根因"，但它**拆错了维度**——
它拆的是 reward 分量维度，真正被抹平的是**状态维度**。

### 2.1 已有的直接证据

`project_channel_decoupling` 记录的 Door-run 分解结果：U^BR 稳健负，但通道归因
**跨 seed 反向**（s1/s3 行为致害、replay 补偿；s2 相反），且 episode |U|/SE 达 10–20，
**不是噪声**。当时的结论是"归因不是 (source, target, stage) 的函数"。

在本诊断下这个现象有了解释：**如果源的价值本来就是状态依赖的，而不同 learner seed
的学生访问不同的状态分布，那么聚合后的符号跨 seed 反向是必然的，不是异常。**
聚合本身就在丢失信息。

同理，Door gate 的"学习效用与行为先验反向"（run 行为 +58% 却 harmful，
walk 行为 −61% 却最不负）也不再是悖论：行为量是对整条 episode 的聚合。

## 3. 文献调研：别人在同一个点上怎么做的

### 3.1 QMP（ICLR 2025）——直接命中

`QMP: Q-switch Mixture of Policies for Multi-Task Behavior Sharing`
（Definition 4.2、Theorem 5.1）：

```
π_i^mix = argmax_{π' ∈ {π_1,…,π_N}}  E_{a~π'(·|s)}[ Q^{π_i}(s,a) ] + α·H[π'(·|s)]
```

要点：

1. **粒度是 per-state per-timestep**，不做任何跨状态聚合；
2. **Theorem 5.1（mixture soft policy improvement）**：
   `Q^{π_i^mix}(s,a) ≥ Q^{π_i}(s,a) ≥ Q^{π_i^old}(s,a)`
   假设：有限动作空间、tabular 分析、SAC 框架；
3. **关键设计——只用于 off-policy 数据收集，不改变学习目标**，因此保持 unbiased；
4. 早期 Q 不可靠不做特殊处理：作者论证这与 Q-learning/SAC 自身在少见状态上的
   估计误差同源，由后续在线交互纠正。

作者自陈的局限：`"This improvement magnitude is limited by the degree of shareable
behaviors and the suboptimality gap that exists."` —— 收益上限取决于任务是否真有可共享行为。

**这正是我们做不到的那件事的正解：它根本不预测"哪个源有用"，
而是在每个状态上用 target 自己的 Q 现场问一次。**

我们的 T^critic 与它是**同一个量**，差别只在于我们对状态求了期望。

### 3.2 相邻工作

- **CDS / UDS（多任务 offline RL）**：按 **per-transition** 的保守 Q 值决定数据是否共享，
  而非按任务相似性。同样是"放弃任务级规则、下沉到样本级判据"。
- **VGD / QPILOTS（2025）**：用 value 梯度在测试时 steer **冻结的** diffusion/flow 策略。
  与我们相关的是"冻结源 + Q 引导"，但对象是单个大策略的去噪过程，不是多源选择。

## 4. 本项目已有的资产：MCG 是 QMP-inspired 的变体，**不是** QMP 本身

> **2026-07-29 外部审查更正**：本节原标题为"MCG 就是它的身体组推广版"，**该表述不准确**。
> 两处实质差异：
> 1. QMP 在**完整策略**间选择，候选动作是某策略的实际输出；MCG 拼接不同教师与学生的
>    身体组动作，该混合动作**可能不在任何策略的动作流形上**，critic 在该点可能无数据支持；
> 2. MCG 的打分是 `min_h [ Q_h(a_cand) − Q_h(a_stu) ]`，既不等于 QMP 的 soft policy value，
>    也不等于 `min_h Q_h(a_cand) − min_h Q_h(a_stu)`。
>
> 准确定位：**QMP-inspired、带身体组混合与保守双 Q 的新 heuristic**。
> **不得借用 QMP Theorem 5.1 作为它的安全保证**（§8(d) 中"冻结源不破坏该保证"的论证
> 同样作废——它只对完整策略候选成立）。因此后续验证必须**先做完整策略的 QMP-fidelity
> baseline，再考虑身体组扩展**，见 `docs/run_card_qmp_fidelity_v1.md`。

`fasttd3_ptf/ptf/mcg.py`（2026-06-11 实现）：

```
option = (教师 i, 身体组 g),  g ∈ {legs_torso(11维), arms(10), hands(40)}
Δ_{i,g}(s) = Qmin(s, ã_{i,g}) − Qmin(s, a_student),   ã = 学生动作第 g 组换成教师的
option-value 不单独学习，直接从 target critic 读出
```

这是一个 **QMP-inspired 的 per-state 选择器**，比 QMP 多一个身体组维度，
但候选动作与打分函数都与 QMP 不同（见本节开头的更正）。
`deltas()` / `null_margins()` / `select()` 均已实现并有测试。

支撑身体组粒度的离线探针（push，25k，4800 状态）：

```
reach 教师 arms 组：Δ≈0,    frac+ = 0.49
reach 教师整动作：  Δ=−6.9, frac+ = 0.05
```

**"局部有用、整体有害"有 critic 级证据。** QMP 只在 Walker2D（6 维动作）等环境验证，
身体组分解在那里无从体现；H1hand 的 61 维动作是这个推广唯一有意义的舞台。

## 5. 重新审视 MCG 当年为什么判负

MCG v1 三任务 pilot（1 seed, 100k）：

| 任务 | MCG | scratch | 诊断（当时） |
|---|---|---|---|
| push | ≈+6 | +16 | 任务无瓶颈，教师无增量 |
| door | 313 | 328 | 瓶颈存在但教师库覆盖不到 |
| window | 336（v1.2 后 ~407） | 489 | 教师无关 + 机制缺陷放大为伤害 |

当年的共同根源诊断是：
> loco 教师能教的（站稳/走近），FastTD3 scratch 在 10k 内自学完成——
> 教师增量知识 < scratch 自学速度时，迁移没有对价，而行为预算的机会成本是实打实的。

### 5.1 判决力缺陷（今天才能看清）

标签可测性审计与 Door gate 都是 **2026-07-27** 才做的，比 MCG（06-11）晚一个半月。
今天回看：**MCG 的三个判决场，事后全部被证明属于"loco 源无用或有害"的类别。**
Door gate 独立证实三 loco 源在 door 上 9/9 per-seed 负、且测量干净。

在源本身就有害的场地上，任何迁移机制的最好结果是"不伤害"，**不可能有增益**。
这与 stair 被选作 BAC 判决场时"无判决力"是同一类错误
（见 `feedback-stepwise-experiments` 的反面案例）。

**结论：MCG 被判负的实验没有判决力，它从未在"源确实有用"的 target 上测试过。**

而我们现在有了当年没有的东西——sibling gate 刚刚测出：

```
slide target:  walk 源 108.12  vs  student 51.18   (+57, 3/3 seed)
stair target:  slide 源 67.02  vs  student 44.77   (+22, 3/3 seed)
```

### 5.2 一个真实的代码缺陷（机制假设的来源）

`train_ptf.py:1414-1416`：

```python
mcg_ablation = str(ptf_cfg["mcg_ablation"])          # full | bootstrap_only | no_bootstrap
mcg_warmup_bootstrap = mcg_ablation in ("full", "bootstrap_only")
mcg_gate_active      = mcg_ablation in ("full", "no_bootstrap")
```

`mcg_gate_active` **一个布尔同时控制 gate 执行（behavior）与 gate 蒸馏（loss）**。
三个 ablation 里没有任何一个能做到"执行开、蒸馏关"。

这与 `project_channel_decoupling` 记录的 `admission_bootstrap` 缺陷
（一个 categorical 同时管 behavior + replay，已修为 `admission_replay_mode`）
**是同一个设计缺陷模式在另一处的复发**。

而 window 负迁移的当年诊断，指向的正是蒸馏通道：
> margin=0 噪声蒸馏——Δ_best 全程为负但 gate_rate 仍 0.2–0.3：
> per-sample Δ 噪声的右尾持续假阳性放行，**把 actor 不断拉向无关教师**。

## 6. 机制假设

**H1（通道归因，有理论依据）**
MCG 的负迁移来自**蒸馏通道**（改变 actor loss → 破坏 QMP 的 unbiased 前提），
而非 Q-switch 本身。纯行为层 Q-switch（behavior-only，不改 loss）应具备负迁移免疫。

- 理论侧：QMP Theorem 5.1 的保证**只在"仅改 behavior"时成立**；
- 实证侧：window 的伤害机制被当年诊断为蒸馏被噪声右尾拉偏；
- 代码侧：`mcg_gate_active` 耦合两通道，该假设**此前无法被测试**。

**H2（粒度，本项目探针支持）**
在 target critic 判定 full-action 替换有害的状态上，仍存在大量身体组使
`Δ_{i,g} > 0`。即"整体有害"不蕴含"处处有害"。

**H3（正面主张）**
既然"哪个源有用"无法预测，就**不要预测**——让 target 自己的 critic
在每个状态、每个身体组上现场决定。这不是第九个信号族，
而是对前八次失败的正面回答：**取消聚合，而不是换一种聚合。**

## 7. 撞车检查

| 工作 | 设定 | 与我们的差异 |
|---|---|---|
| QMP (ICLR 2025) | 多任务**同时训练**的策略集合；Meta-World / Walker2D / Kitchen | 我们是**冻结的跨任务源**；**身体组粒度**；humanoid 61 维动作 |
| SkillBlender | 学习 skill 混合权重，需联合训练 | 已知撞车点，非冻结源 + 非 critic 现场判定 |
| MCP (2019) | 乘性基元组合，需端到端训练基元 | 基元非冻结独立任务源 |
| CDS / UDS | offline 多任务**数据**共享 | 我们是 online **行为**共享 |
| VGD / QPILOTS (2025) | Q 引导冻结 diffusion/flow 策略 | 单策略 test-time steering，非多源选择 |
| limb-level multi-agent RL | 按肢体分 agent **联合训练** | 非冻结源复用 |

检索未发现"冻结跨任务源 + per-state × per-body-group Q-switch"在 humanoid 上的工作。
HumanoidBench 上的 transfer / source reuse 也检索不到 2025–2026 的系统性结果。

## 8. 自我攻击

**（a）这是不是又一次"变体抢救"？**
这是本方向最大的风险，必须正面回答。

- 抢救 = 同一假设失败后调 margin / warmup / exec_prob 重跑 —— **禁止**；
- 本方向 = **新假设**（H1 通道归因），有独立理论依据、独立代码证据、可证伪预测；
  且判决场更换的理由是**独立证据**（Door gate 9/9 负证明当年三场无判决力）。

**约束：本方向不得调整 MCG 的任何既有超参。** 只做通道解耦（打开一个此前不存在的
配置维度）与判决场更换。若 behavior-only 仍显著负，H1 即被否，整线停止。

**（b）slide 上 walk 源已经 +57，per-group 还能加什么？**
最强的反驳。full-action bootstrap 在 slide 上已经很好，per-group 的增量空间可能很小。
所以正确的判决场不是 slide，而是**已知 full-action 有害**的 door：
full 确定为负（9/9，干净测量）+ 探针说 part≫full → 预测 per-group 应显著优于 full。
这是强预测。

**（c）door 的真瓶颈是 P3 开门探索，教师库没有这个技能。**
成立。因此 door 上最好的结果是"不伤害"，不是"成功"。
所以需要**两个场**：door 测负迁移免疫、slide/stair 测正增益保留。
一个机制若能同时做到"有害时免疫、有用时保留"，才构成贡献。

**（d）QMP 的定理在我们的设定下还成立吗？**
Theorem 5.1 的论证只依赖"在每个状态选 argmax Q 的那个策略"，
不依赖候选策略是否在训练中，因此**冻结源不破坏该保证**。
但假设中的**有限动作空间/tabular** 在我们这里不满足（61 维连续动作、
distributional critic 取 min），所以只能作为**动机**而非保证——这一点必须在论文里说清楚。

**（e）FastTD3 是确定性策略，没有 QMP 的熵项。**
Q-switch 退化为纯 argmax，失去熵正则带来的探索。FastTD3 自带 exploration noise，
但这是与 QMP 的一处实质差异，需在实现与写作中显式处理。

## 9. 执行计划

**Step 0（零训练成本，先做）**：离线探针，在 door 已有的 student checkpoint 上
直接算 full 与 per-group 的 Δ 分布，检验 H2。
- 若 per-group 与 full 无分化 → door 上没有可用的局部信号，**不投训练**；
- 若分化成立 → 继续 Step 1。

**Step 1**：通道解耦（新增 ablation 模式，使执行与蒸馏独立可配），补测试。

**Step 2**：预注册 + door 上的 behavior-only vs full-action vs student 三臂对照。

**Step 3**（仅当 Step 2 通过）：slide/stair 上测正增益保留。

每步预注册在先、裁决规则冻结、失败即停。并行训练进程 ≤ 3
（见 `feedback-node-memory-limit`）。

## 10. 参考

- QMP: Q-switch Mixture of Policies for Multi-Task Behavior Sharing, ICLR 2025.
  https://arxiv.org/abs/2302.00671 ・ 代码 https://github.com/clvrai/qmp
- Conservative Data Sharing for Multi-Task Offline RL. https://arxiv.org/abs/2109.08128
- 本项目：`docs/archive/handoff_mcg_v1_20260611.md`、`handoff_mcg_v2_20260612.md`、
  `docs/experiments/door_at10k_gate_v1_results_20260727.md`、
  `docs/experiments/sibling_source_gate_v1_results_20260729.md`
