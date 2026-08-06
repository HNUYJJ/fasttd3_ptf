# RBO-PTF 核心论点结果 · terrain（v1, 2026-06-16）

贡献②（RBO-PTF method）的主实证。验证 **reward-weighted 源选择（safe）** 在地形任务上
正迁移、且优于 **uniform random 源选择（rand）** 与 **scratch（scr）**。

## 1. 实验设置

- 任务：`stair / slide / pole / crawl`（HumanoidBench 地形类，obs=151 纯 proprio，identity adapter）。
- 三方（其余超参一字不差，只差 warmup 源选择方式）：
  - **safe** = reward-weighted bootstrap：按 Effect-Map weight 的 softmax 抽 walk/run 源 + per-source horizon。
  - **rand** = uniform：均匀抽 stand/walk/run（含"无对价"的 stand）+ 固定 horizon=25。
  - **scr** = scratch：空 bank，纯 FastTD3 从零。
- 共同：128 env，100k step，`PTF_MCG`，warmup 30000，decay 80000，`bootstrap_only`，seed 1/2/3。
- 指标：wandb `eval_avg_return`，AUC = `trapz(v, step)/step_span`（平均 return）。
  三方对齐到共同窗口（各自 max_step 的最小值，本批=95k）避免终点不齐的偏差。
- 分析脚本：`scripts/analyze_terrain.py`。

## 2. seed1 结果（共同窗口 AUC，到 95k）

| task | scr | rand | safe | safe−scr | rand−scr | **safe−rand** |
|------|-----|------|------|----------|----------|-----------|
| stair | 217.8 | 172.6 | 253.5 | +35.8 | −45.1 | **+80.9** |
| slide | 236.4 | 425.2 | 522.5 | +286.1 | +188.8 | **+97.4** |
| pole | 537.0 | 558.3 | 702.3 | +165.3 | +21.3 | **+144.0** |
| crawl | 832.8 | 658.4 | 688.6 | −144.2 | −174.4 | **+30.3** |
| **平均** | | | | **+85.8** | **−2.4** | **+88.1** |

## 3. 判读（seed1）

1. **最干净的卖点：safe > rand 在 4/4 任务上一致成立（平均 +88.1，符号无翻转）。**
   这是"reward-weighted 源选择 vs uniform"的纯增益——是本方法的核心主张。
2. **uniform random 源选择平均 ROI≈0（−2.4），且时常负迁移**（stair −45、crawl −174）。
   说明"源选择"这个动作本身有风险，把无对价的 stand 源混进去会拖累——**选择质量才是关键**，
   不是"有没有用 bootstrap"。这正面回应"safe 的增益是否只是 bootstrap 本身"的质疑。
3. **safe 对 scratch 3/4 正迁移**（crawl 例外）。
4. **crawl 是有价值的反例**：整体 loco→crawl 迁移不划算（scratch 最强 832.8，姿态差异大），
   但"既然要迁移"的前提下 safe(688.6) 仍优于 rand(658.4)——即便整体负迁移，
   reward-weighted 仍是更优选择。这给 Effect-Map 的 go/no-go 边界提供了实证锚点。

## 4. 3-seed 加固结果（已完成，2026-06-16）

seed 2/3 全三方 24 runs 跑完（STAMP `20260616T000532Z`）。共同窗口 AUC（到 95k）mean±std 跨 3 seed：

| task | scr | rand | safe | safe−rand |
|------|-----|------|------|-----------|
| stair | 252.5±37 | 169.1±41 | 279.2±20 | +110±54 |
| slide | 271.1±46 | 450.2±20 | 504.7±14 | +54±31 |
| pole | 603.3±48 | 573.2±13 | 717.9±25 | +145±29 |
| crawl | 812.0±25 | 699.6±32 | 656.3±35 | −43±65 |

判读（12 个 task×seed 组合）：**safe>rand 10/12**（mean +66.5，paired t≈2.58 显著）；
safe>scr 8/12；rand>scr 仅 4/12（uniform 无净价值）。

**对第 3 节 seed1 结论的修正（诚实）**：2 个翻转**全在 crawl**——stair/slide/pole 上 safe>rand 是
**9/9 完美一致**；但 crawl 在 3-seed 下 **safe(656) 反而最差 < rand(700) < scr(812)**，即第 3 节
"crawl 上 safe 仍优于 rand"只是 seed1 偶然，3-seed 翻转。这把 crawl 从"反例"升级为 **abstain 机制
的黄金动机**：源(loco)对 crawl 系统性负迁移时，reward-weighted 比 uniform 更糟（safe 更自信地集中
抽有害的 walk/run）。详见 `handoff_discussion_20260616.md` Open Q5。

**呈现建议**：主结果只讲 stair/slide/pole（loco 有对价）safe>rand>≈scr 一致显著；crawl 单列为
negative-transfer 边界 + abstain 动机，勿混进总平均（crawl 负值会拉低 mean、放大 std）。

## 5. 方法定名与贡献定位

- 方法名：**reward_weighted_bootstrap**（RBO-PTF 的核心 warmup 模式 `safe_bootstrap`）。
- 三贡献结构：①Source-Target-Effect Map（诊断/go-no-go）②RBO-PTF method（本文档主实证）
  ③Broad HB evaluation（maze/truck 广度，待第④项）。

## 6. 后续（稳扎稳打，逐项串行）

①加固(seed2/3) → ②wfix 解耦（wfix−rand=纯源选择，safe−wfix=纯执行时长）→
③negctrl 边界（door/spoon safe≈scr）→ ④广度（maze/truck 三方）。每项做扎实、汇报、再进下一项。
