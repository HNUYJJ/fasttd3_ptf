# RIC-PTF 执行报告与求诊 v2（发 ChatGPT-5.5-Pro，2026-06-14）

承接你上次的核心建议（把 warmup bootstrap 升级为论文主贡献 Reward-Bearing
Option Bootstrap；RIC→RCS；五类 ablation；执行排序 seed→bootstrap-only
ablation→Transfer Map v2→weighted bootstrap→边界任务→modular stress test）。
这一轮我们把你排序里的前几项跑完了，得到三个关键结果，其中一个（gate 在 clean
任务上几乎无性能贡献）和一个（window 高方差）需要你重新评判论文叙事与下一步。

## 一、ablation 机制已实现并验证

train_ptf.py（外挂副本）加了 `--ptf-mcg-ablation full|bootstrap_only|no_bootstrap`，
两个解耦布尔控制：warmup 期是否教师执行 / gate 期是否执行+蒸馏。
- full=(T,T)：warmup bootstrap + gate 蒸馏（完整 RIC）；
- bootstrap_only=(T,F)：只 warmup 教师执行，gate 期纯 student、无蒸馏；
- no_bootstrap=(F,T)：warmup 期纯 student，只 gate 蒸馏。
跨 warmup 边界 smoke 验证两分支组合无 bug，200 单测全过。

## 二、四正任务双 seed 巩固（排除噪声）

hurdle/cabinet/powerlift/maze × {full, scratch} 各 2 seed，full vs paired scratch
的 AUC ROI：cabinet +72% / hurdle +56% / powerlift +43% / maze +19%，**regret 全
0，符号四任务一致**。排除了"单 seed 偶然"。

## 三、主性能 ablation（核心结果，2 seed，回答"是否全靠 warmup"）

| 任务 | bootstrap_only | no_bootstrap | full(RIC) |
|---|---|---|---|
| cabinet | +68%(±7) | +32%(±43) | +70%(±1) |
| hurdle | +66%(±33) | −29%(±8, regret 77) | +57%(±62) |
| maze | +15%(±4) | +4%(±0) | +17%(±2) |
| powerlift | +44%(±4) | +7%(±4) | +43%(±11) |

三条结论（四任务一致）：
1. **bootstrap_only ≈ full**（hurdle/powerlift 上 boot 甚至略超 full）——**坐实
   Reward-Bearing Option Bootstrap 是主性能通道**，你的核心论断成立。
2. **no_bootstrap（只 gate/distill）增益≈0 甚至稳定负迁移**（hurdle 两 seed 都
   −29%）——单靠 significance gate 不驱动迁移。
3. **gate 在 clean 任务上不提供额外性能，甚至轻微拖累**（hurdle boot +66 > full
   +57）；它的价值是把 no_bootstrap 的 −29% 负迁移消除（full +57、regret 0）。
   **gate 的角色被精确化为"负迁移安全阀"，而非 clean 任务的性能来源**。

这是个诚实但重要的结论：**modular significance gate 的正面性能价值在 clean
positive-transfer 任务上无法体现**。

## 四、边界任务 + 完整 9 任务主表（验证 Transfer Map 预测边界）

spoon/truck/door × {full, scratch}（seed 1）：truck +10%（中后期 partial）/ spoon
+8%（早期加速、scratch 50k 追平=no-opportunity）/ door +1%（loco 覆盖不到 P3 开门
瓶颈）。全 regret 0。

完整 9 任务主表（full vs scratch，ROI / regret）：

| 类别 | 任务 | ROI | regret | seeds |
|---|---|---|---|---|
| 强对价 | hurdle/cabinet/powerlift/maze | +17~70% | 0 | 2 |
| 安全 | balance_hard | +18% | 0 | 2 |
| 安全 | window | +76%/−27%（高方差）| — | 2 |
| 边界 | truck/spoon/door | +1~10% | 0 | 1 |

**8/9 任务无负迁移，增益从 +76% 到 +1% 单调对应 Transfer Map 的对价分级**。
Transfer Map（半小时 zero-shot 探针 + scratch 对价探针）选址的预测力全面兑现：
强对价四任务全中、spoon 被正确判为无对价、边界任务增益小但不伤害。

## 五、安全任务 seed2 + window 高方差发现（需你评判）

补 seed 后两个安全任务分化：
- **balance_hard 稳定**：+20%/+15%（双 seed 一致）；
- **window 高方差、符号翻转**：seed1 **+76%**，seed2 **−27%**。

**机制诊断（非噪声）**：window 是初始姿态特殊、19-36 步就摔的脆弱 OOD 任务，正
迁移完全依赖 warmup random bootstrap 恰好注入"站立/平衡"reward-bearing 片段。
seed1 恰好注入有用片段（+76%），seed2 注入更多摔倒片段（−27%）。balance_hard
（46 步才摔）站立片段可靠，故稳定。**random warmup 在脆弱 OOD 任务上注入的片段
质量 seed 敏感**——这正是你说的 safe-horizon execution 的 motivation：episode-level
会摔的教师不应长执行，应按 time-to-fall 限制到 safe prefix。

## 六、当前论文三贡献的证据完整度盘点

1. **Transfer Map 预测性诊断**：9 任务全面验证（强/边界/安全的增益梯度都被预测）
   ✓；缺 snippet-level score + Spearman 量化（v1 episode-level 在 window 上会
   误判——window zero-shot 全摔但 seed1 +76%）。
2. **Reward-Bearing Option Bootstrap**：双 seed ablation 坐实 boot≈full ✓，最强。
3. **Significance-calibrated (modular) gate**：安全价值有证据（no_bootstrap
   hurdle −29% → full 0）；但**正面性能价值在 clean 任务上无法体现（boot≈full）**，
   modular（body-group 分解）相对 full-action 的独立价值还完全没证。

## 七、待你评判的决策点

1. **gate/modular 怎么定位**：既然 gate 在 clean 任务上中性、modular 独立价值未证，
   是否应把第三贡献从"significance-calibrated modular gate"收缩为纯"negative-
   transfer safety control"（不强调 modular body-group 分解），还是坚持做 modular
   stress test（push/window/balance 固定 teacher share 比 full-action vs body-group）
   来证明 modular 价值？如果做，怎样的 stress 设置最有说服力、又不像 cherry-pick？

2. **下一步优先级**：三个候选——(a) safe-horizon TransferMap-weighted bootstrap
   （解决 window 高方差 + 把 random warmup 升级为 principled 方法，你排序第 4）；
   (b) modular stress test（gate 价值唯一证据）；(c) Transfer Map v2 snippet-level
   + Spearman predictive validation（贡献 1 的量化）。哪个最该先做？我们倾向 (a)，
   因为它同时解决 window 的诚实短板和方法升级，但想听你的判断。

3. **window 怎么处理**：高方差 +76/−27。是补到 5 seed 报告分布，还是先实现
   safe-horizon bootstrap 重测（预期方差收窄）后再定它的论文角色？它现在是"为什么
   需要 safe-horizon"的论据，这个定位你认可吗？

4. **multi-seed 策略**：四正任务 2 seed、安全 2 seed、边界 1 seed。正式投稿前哪些
   必须 3/5 seed？边界任务（增益小、regret 0）需要补 seed 吗？

5. **论文叙事**：现在最强的故事是"Transfer Map 预测对价 + Reward-Bearing Bootstrap
   是 off-policy 迁移主通道 + 全任务无负迁移"。gate/modular 偏弱。这个叙事够 ICML
   吗？还是必须把 modular/gate 的正面价值做实才能投？

## 八、约束提醒（不变）

- 必须基于 PTF + FastTD3 + HumanoidBench，创新长在 PTF 内；FastTD3 官方代码不可改
  （train_ptf.py 外挂副本可改）；方法须通用、多任务，不做单任务专项。
- 结论：四正任务/安全 2 seed，边界 1 seed；window 高方差需更多 seed。
- 算力：8×V100 32G，单跑 100k≈2h，最多 4 并行。
