# RIC-PTF 执行报告与求诊 v1（发 ChatGPT-5.5-Pro，2026-06-13）

承接上次方向调整（你和用户都判定：停止 package 专项，把论文升级为
HumanoidBench-scale transfer mechanism paper，核心原则 Useful Transfer =
Relevance ∩ Initiation ∩ Consumption ∩ Safety，最优先做 Cross-Task Transfer
Map）。这一轮我们把方向落地，并跑出了第一批宽任务正结果。请你分析、评判、
并对几个关键决策点给意见——尤其是一个会影响论文核心贡献能否成立的发现。

## 一、方向调整执行情况

已全部落地（硬规则写进项目记忆 + 代码）：
- 主方法代码禁任务名分支，只用 progress events / obs 元数据 / critic 信号；
- 任何源必须跨 ≥2 任务才进主方法；
- 主实验 = 宽任务矩阵；package 降级为诊断 hard case。

## 二、Cross-Task Transfer Map v1（17 任务 × 6 源 zero-shot）

对每个 (source, target) cell 整动作 zero-shot rollout（16 ep × 500 步），量化
return / fall / 任务 info 分量（HB 每个任务的 info 自带细粒度 progress，如
door_openness_reward / packages_picked_up / hand_dist——这就是通用 progress
event 源，无任务名分支）。

**四类模式**：
1. 生存型对价（stand 一个源就 ×2-10，机理=HB manipulation reward 普遍带 stand
   乘子）：spoon ×10、cabinet ×7、powerlift ×9、truck ×2、room ×10、stair/slide；
2. 推进型对价（步态技能直接推进任务）：maze（walk ×3.3）、hurdle（run 157±80
   真在跨栏）、door（run +56%）；
3. 全员 OOD（安全试金石）：window（所有源 19-36 步全摔）、balance_hard（stand 比
   zero 还差，整条负迁移）；
4. return 不可分辨：package（dense 惩罚 σ>1000 淹没信号，必须用 info 分量）。

**info 分量层（比 return 高一档分辨率，RIC Relevance 的雏形）**：
- package：reach 是唯一把手带向箱的源（hand-box 1.90→1.22），walk/run 把手距拉到
  8.9/25（径直走离）——细粒度负迁移；这是个修正：现成 reach 源就有 package 接近
  能力，我们之前手工训 approach 前没在 package 上测过 reach 手距分量；
- door：run 的 passage_reward 0→0.21（真在推进通过段）；stand 反而让 hand-hatch
  proximity 0.69→0.33（"生存源"在推进型任务上有机会成本）；
- "源越强越通用"被直接证伪——迁移是 (源, 任务, 分量) 三元相关。

## 三、scratch 对价探针（排除 confound）

zero-shot 增益 ≠ 训练时迁移有对价（教师增量 < scratch 自学速度时迁移没意义）。
六任务 scratch 50k 短跑，教师 zero-shot 起点（×2 换算 eval 口径）对比 scratch@30k：

| 任务 | scratch@5k | @30k | 教师起点 | 对价 |
|---|---|---|---|---|
| hurdle | 5 | 31 | run≈314 | 强 |
| cabinet | 26 | 53 | stand≈204 | 强 |
| powerlift | 88 | 136 | stand≈342 | 强 |
| maze | 139 | 352 | walk≈758 | 强 |
| truck | 758 | 1164 | stand≈1202 | 临界 |
| spoon | 227 | 356 | stand≈146 | **无（confound 实锤：自学站立极快）** |

主战场任务集 = map 增益 ∩ scratch 慢 = **hurdle / cabinet / powerlift / maze**。

## 四、宽 pilot seed 1：四任务全部正迁移（项目至今最强正结果）

同一组 loco 源（stand/walk/run）+ 同一 SC-MCG 配置（warmup 30k random + null
gate q95 + λ decay 80k）+ 参数化 obs adapter（无任务名分支）vs paired scratch：

| 任务 | MCG AUC | scr AUC | ROI | regret | 50k 时 MCG vs scr |
|---|---|---|---|---|---|
| hurdle | 477 | 279 | **+71%** | 0 | 599 vs 163（3.7×）|
| cabinet | 196 | 129 | **+53%** | 0 | 233 vs 83 |
| powerlift | 237 | 167 | **+42%** | 0 | 334 vs 136 |
| maze | 350 | 262 | **+34%** | 0 | 366 vs 324 |

形态=标准 transfer 正例（前中期加速、time-to-threshold 大幅缩短、末期 scratch
追平、零负迁移）。Transfer Map 的预测力被闭环验证：半小时探针选的四任务全兑现，
判无对价的 spoon 没浪费算力。seed 2 配对验证进行中（mcg 已完成，scr 跑中）。

## 五、安全对照的意外强结果 + 机制解读（**最需要你评判的部分**）

预期：window/balance_hard（全员 OOD）上 MCG 应 null-gate-closed、regret≈0（不
伤害）。实测是**正迁移**：

| 任务 | MCG AUC | scr AUC | ROI | Transfer Map zero-shot |
|---|---|---|---|---|
| window | 335 | 191 | **+76%** | 所有源 19-36 步全摔 |
| balance_hard | 94 | 78 | **+20%** | stand 19 < zero 32（整条负迁移）|

**机制诊断（关键）**：拉 wandb 后发现，window MCG 在 warmup 结束（30k）时从 42
暴涨到 378，但此时 **gate 期教师执行率仅 2.2%、gate_rate 5-8%**。所以正迁移
**不是 gate 期 modular 执行的功劳，而是 warmup bootstrap**——前 30k 的 random
教师整动作执行把"站立/平衡相关的状态-动作-回报"注入了 buffer，critic 学到价值
后 actor 在 warmup 结束时快速利用。gate 期低强度（5-8%）的 legs_torso 蒸馏是次要
贡献。

**两个推论**：

1. **window 成功 vs package 失败的决定性对比**（RIC Consumption 原理的正反例）：
   同一个 warmup bootstrap 机制，window 成功是因为 loco 教师站立分量与任务回报
   直接相关（站稳得分），warmup 注入的数据携带 reward 增量；package 失败是因为
   "走近"状态不携带 −3·dist(box,dest) 增量（箱子没动）。"状态覆盖≠回报事件"从
   package 孤立教训升级为有正反例的通用原理。

2. **安全性正面证据**：balance_hard 上整条 stand 教师 zero-shot 有害（19<32），
   但 SC-MCG +20%——significance gate + modular 执行避免了整条 OOD 教师的伤害。

## 六、关键认知与待你评判的决策点

**这是我们最担心的问题**：pilot 四任务 + 安全对照的正迁移，**主增益来自 warmup
bootstrap（call-and-return 注入回报数据），而 significance-calibrated modular
gate 在 gate 期 exec 仅 2-8%**。也就是说，论文想主张的核心机制（"modular +
显著性校准 gate"）目前**还没有被实验单独证实**——很可能审稿人会问"你的提升是不是
光靠 warmup 往 buffer 灌教师数据，跟你的 gate/modular 没关系"。

请你重点回答：

1. **如何设计 ablation 分离三个贡献**（warmup bootstrap / modular 分解 / 显著性
   gate）？我们计划的 PTF-full 基线（整条教师 warmup + 无 modular gate + 旧 PTF
   fixed distill）够不够？还需要哪些 cell（如 MCG-no-gate / MCG-sign-gate /
   bootstrap-only-no-distill）？

2. **既然 warmup bootstrap 是主力，论文的核心贡献该怎么重新定位**？是否应该把
   叙事从"significance-calibrated modular gate"调整为"reward-bearing teacher
   bootstrap + 安全校准"？还是坚持 gate 是关键、只是需要更难的任务（教师整条有
   局部有用+局部有害混合）才能体现 gate 价值？

3. **RIC 四支柱目前的覆盖**：Relevance（Transfer Map 选址，离线）✓；Consumption
   （warmup bootstrap）✓；Safety（gate 不引入负迁移）部分✓。但 **Initiation
   （chain warmup / initiation set）在这些单阶段 loco 任务上根本没用到**（它们不
   是链式任务，warmup 用 random 不是 chain）。Initiation 这条支柱是否只在 package
   类长链任务上才有意义？如果是，它和"package 降级为诊断"是否冲突？

4. **multi-seed 与统计**：四任务 + 安全对照，3 seed 还是 5 seed？哪些必须多 seed？

5. **package / EODT 怎么处理**：还做 intervention-driven EODT 吗，还是纯诊断章节
   （它贡献了"状态覆盖≠回报事件"的反例，已经很有价值）？

6. **临界任务**（truck/door）要不要纳入主实验扩大覆盖面？

7. **Transfer Map 作为方法论贡献**够不够强？要不要扩成完整 source×target×progress
   heatmap（加 critic significance / initiation coverage 维度）？

## 七、约束提醒（不变）

- 必须基于 PTF + FastTD3 + HumanoidBench，创新长在 PTF 内；FastTD3 官方代码不可改
  （train_ptf.py 是外挂副本可改）；目标 ICML，方法须通用、多任务，不可 package
  专项。
- 结论目前 1 seed（seed 2 跑中），方差大，正式声明前 paired 多 seed。
- 算力：8×V100 32G，单跑 100k≈2h，最多 4 个并行（6 个会把机器 swap 打满）。
