# Stage-conditioned target evidence online v1：seed-1 结果

> 本文是 seed-1 中间记录。多种子终局裁决见
> `stage_conditioned_target_evidence_online_v1_multiseed_results_20260727.md`：
> Hurdle 3/3 正向，但 Crawl seed3 false admission 造成负迁移，因此当前量不能作为
> 独立、可靠的一般迁移性指标。

**当前裁决：`ONLINE_BIDIRECTIONAL_FEASIBILITY_PASS`；性能收益仍为
`SEED1_PROMISING_NOT_CONFIRMED`。**

## 1. 核心问题与唯一主要假设

核心问题是：能否在 student 当前训练阶段，用一个不依赖任务名称和历史 source
排名的通用规则，决定接纳哪个 source，或严格退化为 100% student？

主要假设是：从相同 student occupancy 状态出发，比较 source 短干预与 student
短干预；只有当 target return 与 target-achievement progress 的保守下界均为正时，
该 source 才值得在当前阶段进入 reward-bearing bootstrap。

该机制不把“向前位移”写死为普适迁移性。核心只读取 target evidence contract：

- 通用量：target return、target-achievement progress；
- 可选量：目标任务显式声明的 hard constraints；
- 任务适配器只负责说明“完成该 target 的进度是什么”，不得包含 source 名称、
  历史 source 排名或为当前结果拟合的参数。

Crawl 的 progress 因而是受匍匐姿态和隧道占用约束的前进，而不是裸 root-x；
Hurdle 使用受直立和越障语义约束的前进。换任务时替换 evidence contract，不修改
接纳、置信下界、排序、quarantine 或 replay 生命周期算法。

## 2. 冻结机制

- student occupancy：固定 reset seed 与 episode age 面板；
- matched-state intervention：每个状态分别执行 source 和当前 student，horizon=25；
- 统计：固定 bootstrap seed，5000 次 bootstrap，90% LCB；
- 接纳：`LCB90(ΔR)>0` 且 `LCB90(ΔP)>0`，并满足显式 hard constraints；
- 排序：在已接纳 source 内按 `LCB90(ΔP)` 排序，选 top-1；
- source/student 分配：top-1 与 student 的 logit 都为 0，即各 0.5；
- 无 source 通过：exact abstention，100% student；
- probe 数据：只进入 quarantine JSON，不进入主 replay；
- 决策时点：10k、20k；初始 0–10k 为 exact abstention；
- source 切换后：旧 source 数据仍保留用于审计，但立即退出 active replay。

这里衡量的是**当前阶段的局部 target intervention evidence**，不是已经得到验证的
延迟学习效用或完整训练 ROI。完整 RBO 训练性能是它必须通过的外部验证。

## 3. 在线选源结果

表中 `dR mean/LCB` 是 source 相对当前 student 的 target return 增量；
`dP mean/LCB` 是 target-achievement progress 增量。

| Task / step | Source | Admit | dR mean / LCB90 | dP mean / LCB90 |
|---|---|---:|---:|---:|
| hurdle / 10k | stand | yes | +0.186 / +0.060 | +0.032 / +0.017 |
| hurdle / 10k | walk | yes, **selected** | +1.481 / +1.096 | +0.224 / +0.180 |
| hurdle / 10k | run | yes | +1.409 / +1.067 | +0.205 / +0.166 |
| hurdle / 20k | stand | no | -0.452 / -0.805 | -0.066 / -0.101 |
| hurdle / 20k | walk | yes | +1.000 / +0.720 | +0.124 / +0.089 |
| hurdle / 20k | run | yes, **selected** | +0.881 / +0.455 | +0.142 / +0.093 |
| crawl / 10k | stand | no | -0.626 / -1.291 | -0.083 / -0.133 |
| crawl / 10k | walk | no | +0.973 / +0.306 | +0.018 / **-0.017** |
| crawl / 10k | run | no | +1.255 / +0.531 | +0.013 / **-0.029** |
| crawl / 20k | stand | no | -4.733 / -5.695 | -0.085 / -0.157 |
| crawl / 20k | walk | no | -2.377 / -3.022 | -0.078 / -0.125 |
| crawl / 20k | run | no | -2.455 / -3.116 | -0.037 / -0.105 |

关键观察：

1. Hurdle 在 10k 选择 walk，20k 切换为 run；stand 从弱正变为明确负，支持
   source value 随 student stage 改变。
2. Hurdle 20k 的 run/walk progress LCB 非常接近，单 seed 的 top-1 次序不应被
   解释成稳定的精确排名。
3. Crawl 10k 的 walk/run 如果只看即时 target return 都会被接纳；加入
   target-achievement progress 后均被拒绝。这正是“只前进但没有以目标要求的方式
   前进不算有益”的在线实例。
4. Crawl 在 10k、20k 均 exact abstention，没有为通过实验而修改阈值或任务规则。

## 4. Behavior / replay 生命周期审计

候选顺序均为 `[stand, walk, run, student]`。

### Crawl

- final candidate masses：`[0, 0, 0, 1]`；
- behavior execution counts：`[0, 0, 0, 3,840,000]`；
- main/active replay counts：`[0, 0, 0, 3,840,000]`；
- critic sample counts：`[0, 0, 0, 1,965,359,104]`。

因此这不是“多数时候没选 source”，而是 30k 全程 source behavior、source replay
与 source critic sampling 都严格为零。

### Hurdle

- 10k 后执行 walk，20k 后切换 run；
- final behavior counts：`[0, 634,725, 641,003, 2,564,272]`；
- main replay counts：`[0, 634,725, 641,003, 2,564,272]`；
- active replay counts：`[0, 0, 641,003, 2,564,272]`；
- final candidate masses：`[0, 0, 0.5, 0.5]`。

walk 的旧数据物理保留以便审计，但在 20k 被撤销后 active count 归零；run 与 student
按 0.5/0.5 继续采样。这验证了选源决策与 replay eligibility 的同步切换。

## 5. Seed-1 性能

### 5k–25k 在线评估曲线

| Task | Arm | 5k | 10k | 15k | 20k | 25k | normalized AUC |
|---|---|---:|---:|---:|---:|---:|---:|
| hurdle | online | 6.149 | 15.697 | 22.603 | 26.990 | 69.938 | 25.833 |
| hurdle | scratch | 6.957 | 11.049 | 13.559 | 15.817 | 14.646 | 12.807 |
| crawl | online, exact abstain | 417.447 | 395.454 | 618.858 | 673.236 | 812.047 | 575.574 |
| crawl | scratch | 330.930 | 542.874 | 509.856 | 641.776 | 678.755 | 549.837 |

Hurdle 的 seed-1 AUC 差为 `+13.027`，是值得补种子的正信号。Crawl online 全程
零 source 暴露，其与 scratch 的曲线差只能反映独立 GPU 运行的轨迹与数值噪声，
不能称为迁移收益，也说明不能用单 seed 曲线差直接做因果归因。

### 30k 冻结 source-free 评估（32 episodes）

| Task | Arm | return mean ± std | progress max-dx | posture |
|---|---|---:|---:|---:|
| hurdle | online | 115.07 ± 79.03 | 12.80 | NA |
| hurdle | scratch | 25.36 ± 6.70 | 1.00 | NA |
| crawl | online, exact abstain | 880.10 ± 75.14 | 24.58 | 0.797 |
| crawl | scratch | 831.38 ± 13.37 | 14.79 | 0.817 |

Hurdle 的终点差 `+89.71` 与进度差 `+11.80` 支持进一步确认，但尚不能用一个 seed
升级为稳定性能贡献。Crawl 的差异仍不具有 source 迁移归因，因为 source 暴露严格为
零。评估 JSON 中 Hurdle 的 `success_count` 字段不具备可靠成功语义，本裁决未使用。

## 6. 当前结论与下一步

### 已支持

- 相同核心规则可在正任务 Hurdle 接纳/切换 source，在负任务 Crawl exact abstain；
- Crawl 的任务适配器成功阻止“裸前进”冒充目标完成进度；
- 在线决策能同步控制 behavior authority 和 replay eligibility；
- seed-1 Hurdle 出现明显训练加速与 source-free 终点提升信号。

### 尚未支持

- 这还不是跨任务普适、已验证的最终迁移性指标；
- 尚未证明局部短干预证据稳定预测 RBO 的长期学习收益；
- 尚未证明 Hurdle 的性能增益跨 seed 稳定；
- 尚未证明 top-1 的细微排序差在接近并列时可靠。

### 最小确认矩阵

- Hurdle：online 与同 checkout scratch，各补 seed 2/3；
- Crawl：只补 online seed 2/3，检验 exact-abstention 决策跨 seed 稳定；
- 不重复 Crawl scratch，因为零 source 行为与零 source replay 已由机制审计直接保证；
- 不新增阈值、不换任务挽救、不扩参数网格。

Seed-1 raw artifacts:
`logs/train/stage_target_evidence_online_v1_20260726T104500Z/`.

Seed-2/3 confirmation root:
`logs/train/stage_target_evidence_online_v1_confirm_20260726T114555Z/`.
