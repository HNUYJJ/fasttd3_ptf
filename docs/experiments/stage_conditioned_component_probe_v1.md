# Stage-conditioned component probe v1

> 冻结日期：2026-07-26  
> 状态：历史 v1；已被配置化的 target-evidence v2 取代  
> 目的：检验一个廉价的、student-relative 的候选迁移性指标；不实现在线 controller。

> 2026-07-26 一般化修订：v1 将聚合 feasibility 当作统一 veto。后续压力测试发现，
> “每个 reward component 都不得下降”会禁止 Hurdle 中有益 source 的合法任务权衡。
> 当前设计见 `stage_conditioned_target_evidence_generalization_v2.md`：target components
> 用于构造任务成就进度；只有 target MDP 显式声明的 hard constraint 才有 veto 权。

## 1. 科研立项门

1. **核心问题**：教师价值信号——当前 student 阶段应接纳哪个 source，何时严格弃权。
2. **唯一主要假设**：在当前 student occupancy 的同一物理状态上，source 相对 student
   的短段目标收益、任务进度与必要可行性差异，可以区分 Hurdle 的正迁移 source 和
   Crawl 的负迁移 source。
3. **决策影响**：正例与负例同时通过，才进入在线低频 admission 设计；任一失败即否证
   本候选，不调权重或阈值抢救。
4. **非重复性**：旧 Transfer Map v2 从 reset/zero baseline 出发，Crawl 没有 progress
   字段，也没有检查 crawling/tunnel 约束；本实验从真实 student occupancy 出发并做
   matched student counterfactual。
5. **最小实验**：Hurdle@10k 与 Crawl@10k，各 1 个 student checkpoint、32 个冻结状态、
   stand/walk/run 三个 source、每分支 25 步；不训练 controller。

## 2. 已冻结的因果标签

- **Hurdle 正例**：equal-dose RBO 的 3 seeds 均满足
  `run > walk > scratch`（5k–25k nAUC 与 30k source-free endpoint 两视角）。
- **Crawl 负例**：seed 1 的 stand/walk/run equal-dose RBO 均低于 scratch，三种 source
  的 32-episode source-free return/progress 对 scratch 均为 0/32 wins。

这些标签定义的是完整 RBO intervention package 的学习效果，不是动作片段本身的真值。
因此本实验只是检验廉价 proxy 能否预测标签；通过也不等于已经证明在线迁移性指标。

## 3. Matched-state probe

对每个 target student checkpoint：

1. 用冻结 student deterministic rollout 产生 4 条 occupancy stream；
2. 每条 stream 在 age `[0, 5, 10, 25, 50, 100, 150, 200]` 保存 MuJoCo
   `FULLPHYSICS` 状态，共 32 个状态；
3. 从完全相同的状态分别执行 student、stand、walk、run，固定 horizon `h=25`；
4. 对每个 source 计算逐状态 paired difference：
   - `ΔR`：25 步累计 target reward；
   - `ΔP`：root-x 位移；
   - `ΔF`：任务必要可行性的 25 步均值。

任务必要可行性直接来自官方 target reward components：

- Hurdle：`min(stand_reward, wall_collision_discount)`；
- Crawl：`min(crawling, crawling_head, in_tunnel)`。

## 4. 候选 admission rule

对 32 个 paired differences 做固定种子、5000 次 bootstrap，取 90% 区间下界
`LCB90`。source 仅在以下三项同时成立时接纳：

```text
LCB90(ΔR) > 0
LCB90(ΔP) > 0
LCB90(ΔF) >= 0
```

接纳 source 之间按 `LCB90(ΔP)` 降序排序。没有 source 通过时输出 exact abstention。

该规则没有任务专属权重、没有从实验结果拟合的 epsilon，也不使用 target critic。
任务只提供官方 reward 中“什么算进度、什么是必要可行性”的语义。

## 5. Primary gate 与停止规则

Primary 只看 10k student：

- Hurdle：run 与 walk 均被接纳，且 `run > walk`；
- Crawl：stand/walk/run 全部拒绝，输出 exact abstention。

两项同时满足才叫 `BIDIRECTIONAL_FEASIBILITY_PASS`。否则输出
`CANDIDATE_REJECTED`，停止该候选；不得改 confidence、horizon、组件权重或 epsilon
后在同一标签上重试。

更晚 checkpoint 只能作为 stage-dependence 描述，因为当前没有相应的局部干预因果标签，
不进入 primary gate。

## 6. 声明边界

- 通过：只说明“短段 matched component proxy 值得进入一次在线低频 feasibility test”。
- 失败：说明行为层短段 proxy 不能预测 RBO 学习效用，继续研究 transferability 时必须
  引入不同证据，而不是调整本 probe。
- 无论结果如何，`ΔR/ΔP/ΔF` 都不是最终论文指标，除非后续在新任务和在线闭环中验证。
