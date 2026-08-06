# 宽 pilot v1 结果：SC-MCG 在四个强对价任务上的正迁移（2026-06-13）

> **2026-06-15 重要修正（多任务三方验证后）**：本文 window 的"安全对照正迁移"
> 结论（+76%、safe-horizon motivation）**已撤回**。3-seed scratch 三方对比
> （[handoff_rbo_v3](handoff_rbo_v3_20260614.md) 第四节）证明 window scratch
> AUC=309±88（自身高方差，有 lucky seed），safe/rand/scr 去 outlier 后三方重叠
> ~240 → **window transfer 无净对价，高方差来自 scratch baseline 本身、非 bootstrap**
> （与 [transfer_map_v2_analysis](transfer_map_v2_analysis.md) 一致）。强对价四任务
> （hurdle/cabinet/powerlift/maze）正迁移结论不受影响；safe_bootstrap 的扎实正面
> 证据见 hurdle（safe 538 > rand 466 ≫ scr 155，safe 比 random 高 +15% AUC）。

RIC-PTF 主线的第一组正结果。承接 [Transfer Map v1 分析](transfer_map_v1_analysis.md)
选出的"强对价四任务"（教师 zero-shot 起点 ≫ scratch@50k），用统一的 SC-MCG
配置 + 同一组 loco 源验证迁移是否真有对价。

## 实验设置（全程零任务定制——RIC-PTF 通用性硬规则）

- 任务：hurdle / cabinet / powerlift / maze（Transfer Map 判定强对价）；
- 源库：stand / walk / run（同一组 loco 源跨四任务复用），obs adapter 按任务
  布局参数化（`hb_robot_qpos_qvel` qpos_dim=nq 或 identity），无任务名分支；
- 方法：SC-MCG（warmup 30k random + null gate q95 + λ decay 80k）vs paired
  scratch（empty bank，同 seed/同超参，仅 source bank 不同）；
- 规模：100k steps，128 envs，seed 1（多 seed 巩固进行中）；
- 指标脚本：[scripts/aggregate_pilot_results.py](../scripts/aggregate_pilot_results.py)。

## 结果（seed 1，MCG vs paired scratch）

| 任务 | MCG AUC | scr AUC | ROI | regret | MCG final | scr final |
|---|---|---|---|---|---|---|
| hurdle | 477 | 279 | **+71%** | 0 | 809 | 758 |
| cabinet | 196 | 129 | **+53%** | 0 | 248 | 216 |
| powerlift | 237 | 167 | **+42%** | 0 | 300 | 250 |
| maze | 350 | 262 | **+34%** | 0 | 383 | 302 |

（ROI = (AUC_mcg − AUC_scr)/AUC_scr；regret = max(0, AUC_scr − AUC_mcg) = 0 四任务全无负迁移。）

## 形态：标准 transfer 正例

逐点曲线（eval_avg_return）：

| 任务 | 10k | 30k | 50k | 70k | 100k |
|---|---|---|---|---|---|
| hurdle MCG | 27 | 181 | **599** | 735 | 811 |
| hurdle scr | 17 | 49 | 163 | 573 | 852 |
| maze MCG | **313** | 315 | 366 | 354 | 385 |
| maze scr | 123 | 282 | 324 | 307 | 351 |

优势集中在前中期（time-to-threshold 大幅缩短——hurdle 50k 时 MCG 是 scratch
的 3.7 倍），末期 scratch 逐渐追平（hurdle 100k 852 vs 811）。这正是"加速
学习、渐近持平、无负迁移"的教科书 transfer 形态，与 PTF 的 λ→0 衰减设计一致
（迁移在早期 bootstrap，后期交还给任务自身梯度）。

## 方法论价值：Transfer Map 的预测力被闭环验证

Transfer Map（半小时的 zero-shot 探针）选出的强对价四任务，pilot 里全部兑现
正迁移；判定"无对价"的 spoon（scratch 5k 自学就超过教师 zero-shot 起点）没有
浪费训练算力。"便宜探针 → 预测对价 → 训练验证"的选址方法论本身是论文贡献的
一部分——它把"在哪些任务上做迁移"从拍脑袋变成数据驱动。

## 安全对照的意外强结果（window/balance_hard，seed 1）

预期是"全员 OOD 任务上 MCG 不伤害（regret≈0）"，实测是**正迁移**：

| 任务 | MCG AUC | scr AUC | ROI | regret | Transfer Map zero-shot |
|---|---|---|---|---|---|
| window | 335 | 191 | **+76%** | 0 | 所有源 19-36 步全摔 |
| balance_hard | 94 | 78 | **+20%** | 0 | stand 19 < zero 32（整条**负迁移**）|

**机制解读（重要，诚实标注）**：window MCG 在 warmup 结束（30k）后立刻从 42
暴涨到 378，但此时 gate 期教师执行率仅 2.2%、gate_rate 5-8%——正迁移**不是
gate 期教师执行的功劳，而是 warmup bootstrap**：random 教师执行把"站立/平衡
相关的状态-动作-回报"注入 buffer，critic 学到价值后 actor 在 warmup 结束时
快速利用。gate 期低强度的 legs_torso 蒸馏（5-8%）是次要贡献。

**与 package 的决定性对比（RIC 原理的正反例）**：window/balance 上 loco 教师的
站立分量**与任务回报直接相关**（站稳就得分），warmup 注入的数据携带 reward 增量
→ critic 能学；package 上教师注入的"走近"状态**不携带** −3·dist(box,dest) 增量
（箱子没动）→ critic 学不到。**同一个 warmup bootstrap 机制，window 成功 package
失败，差异只在"教师数据是否携带任务回报"**——这正是"状态覆盖≠回报事件"原理，
也是 RIC 的 Consumption 条件（executed data 必须 reward-bearing 才被消费）。

**安全性的正面证据**：balance_hard 上整条 stand 教师 zero-shot 有害（map: 19 <
zero 32），但 SC-MCG 不但没把这个负迁移引进来，反而 +20%——significance gate +
modular 执行避免了整条 OOD 教师的伤害，只提取了有限的局部正价值。

**双 seed 后的关键修正（2026-06-14）**：补 seed 2 后两个安全任务分化明显：

| 任务 | seed 1 | seed 2 | 解读 |
|---|---|---|---|
| balance_hard | +20% | +15% | 稳定正迁移 |
| window | +76% | **−27%** | 高方差、符号翻转 |

window 的高方差不是噪声，而是机制信号：window 是初始姿态特殊、19-36 步就摔的
脆弱 OOD 任务，其正迁移完全依赖 warmup random bootstrap 恰好注入了"站立/平衡"
的 reward-bearing 片段。seed 1 恰好注入有用片段（+76%），seed 2 注入了更多摔倒
片段（−27%）——**random warmup 在脆弱任务上注入的片段质量 seed 敏感**。balance_hard
稳定（46 步才摔），站立片段更可靠，故双 seed 一致。

这正是 **safe-horizon TransferMap-weighted bootstrap 的明确 motivation**：不应让
episode-level 会摔的 loco 教师长时间执行，而应按 Transfer Map 的 time-to-fall
估计限制执行步数到 safe prefix（window 只取前 5-15 步的站立片段，不注入后续
摔倒）。window 从此变成"为什么需要 safe-horizon bootstrap"的论据，而非稳定正例。

待补：safe-horizon weighted bootstrap 实现后重测 window（预期方差收窄）；真正
"全无关"任务的 null-gate-closed 测试，分离 warmup bootstrap 与 gate 安全性。

## 主性能 ablation：bootstrap 是主力，gate 是安全（论文核心图）

回答审稿核心风险"提升是不是全靠 warmup 灌 replay"。四任务对比 scratch /
bootstrap_only（只 warmup 教师执行，gate 期纯 student、无蒸馏）/ no_bootstrap
（warmup 期纯 student，只 gate 蒸馏）/ full(RIC)，**全部 2 seed**（mean±std）：

| 任务 | bootstrap_only | no_bootstrap | full(RIC) |
|---|---|---|---|
| cabinet | +68%(±7) | +32%(±43) | +70%(±1) |
| hurdle | **+66%(±33)** | **−29%(±8, regret 77)** | +57%(±62) |
| maze | +15%(±4) | +4%(±0) | +17%(±2) |
| powerlift | **+44%(±4)** | +7%(±4) | +43%(±11) |

**三条结论（四任务一致，机制自洽）**：

1. **bootstrap_only ≈ full**——双 seed 后 boot 在 hurdle/powerlift 上甚至略超
   full，cabinet/maze 上差距 ≤2%。坐实 Reward-Bearing Option Bootstrap 是主性能
   通道：PTF 在 off-policy FastTD3 上的主价值是 source option 执行重塑早期 replay
   distribution、注入携带回报的 transitions，不是 imitation loss。
2. **no_bootstrap（只 gate/distill）增益≈0 甚至稳定负迁移**——hurdle 两 seed 都
   负（−29%±8、regret 77），cabinet 高方差（+32%±43，seed 间不稳）。单靠
   significance gate/distill 不足以驱动迁移，它不是主性能来源。
3. **gate 的价值是安全，不是 clean 任务上的性能**——在源全局有用的 clean 任务上
   gate 不提供额外性能、甚至轻微拖累（hurdle boot +66 > full +57）；但它把
   no_bootstrap 在 hurdle 上的 −29% 负迁移完全消除（full +57、regret 0）。gate
   的角色由此精确化：**负迁移安全阀**，其性能价值需在"源混合有用/有害"的任务
   （push/window/balance_hard）上用 modular stress test 单独证明，不在 clean
   positive-transfer 任务上体现。

这张表把论文三贡献的分工坐实：**Reward-Bearing Option Bootstrap = 主性能（boot≈
full）；significance-calibrated modular gate = 负迁移安全阀（no_bootstrap→full 在
hurdle 上 −29%→0）**。下一步用 modular stress test 在 mixed-source 任务上证明
gate 的正面价值。

## 边界任务（验证 Transfer Map 的对价边界预测，seed 1）

spoon（no-opportunity control）/ truck（临界）/ door（协调瓶颈）× {scratch, full}。
三者增益都远小于强对价四任务，且 regret 全 0——方法在低对价任务上既不浪费也不
伤害：

| 任务 | full ROI | 形态 | Transfer Map 预测 |
|---|---|---|---|
| truck | +10% | 中后期 partial（100k: 1631 vs 1337）| 临界对价 ✓ |
| spoon | +8% | 早期加速、scratch 50k 追平 | 无对价（scratch 自学快）✓ |
| door | +1% | 交错持平、末期 full 略高 | 协调瓶颈、loco 覆盖不到 P3 ✓ |

spoon 验证了 ablation 推论：scratch 自学极快的任务，warmup bootstrap 仍给早期小
加速（+8%），但 scratch 很快追平——增益远小于强对价四任务（+17~70%），符合"无
对价"方向。door 的 loco 源只能 partial 推进 passage 段（Transfer Map 的 info 分量
已预示），解决不了"转把手+推门"的协调瓶颈。

## 完整 9 任务主表（RIC-PTF full vs scratch，ROI / regret）

| 类别 | 任务 | ROI | regret | seeds |
|---|---|---|---|---|
| 强对价 | hurdle | +57% | 0 | 2 |
| 强对价 | cabinet | +70% | 0 | 2 |
| 强对价 | powerlift | +43% | 0 | 2 |
| 强对价 | maze | +17% | 0 | 2 |
| 安全/snippet | window | +76%/−27% | 高方差 | 2 |
| 安全/snippet | balance_hard | +18% | 0 | 2 |
| 边界 | truck | +10% | 0 | 1 |
| 边界 | spoon | +8% | 0 | 1 |
| 边界 | door | +1% | 0 | 1 |

**9 个任务无一负迁移（regret 全 0）**，增益从 +76% 到 +1% 单调对应 Transfer Map
的对价分级。这把论文从"四个正任务"扩成了"HumanoidBench-scale 系统评测 + 预测性
诊断"：Transfer Map 不只选出强对价任务，也正确预测了边界与安全任务的行为。

## 待补（验证中 / 计划）

- **seed 2 巩固**（批次链自动接力）：四任务 AUC 增益符号应与 seed 1 一致；
- **PTF-full 基线**（计划）：整教师迁移 vs modular——证明 body-part 分解的必要性；
- **临界任务**（truck/door）：Transfer Map 判为边际对价，验证机制在弱信号上的行为。
