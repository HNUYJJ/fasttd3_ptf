# Stability-Deconfounded Transfer Audit v1（预注册，尚未运行正式训练）

## 1. 决策问题

当前 RBO 的主要收益来自 reward-bearing bootstrap，但尚不清楚它迁移的是：

1. **稳定性/存活时间**：站得更久，因此看到更多状态并累计更多奖励；
2. **任务进展能力**：在相同暴露时间内更快接触、移动或完成目标；
3. **二者兼有**。

本审计不以“再得到一个正结果”为目标，而是决定论文应继续称为 task-skill
transfer，还是收窄为 stability-prior / replay-bootstrap。

## 2. 预注册假设

- **H-stability**：若收益只来自稳定性，则 RBO 的 raw progress 高于 scratch，
  但在 episode 配对并截断到较短存活长度后，task-progress 差值应接近 0。
- **H-task**：若存在任务能力迁移，则在共同存活前缀上，RBO 仍应产生更高的
  task-progress。
- **H-source**：stand、walk、run 单源会呈现不同的 stability/progress 结构；若
  三者只按存活率排序，而与 task-progress 无关，则当前源选择本质上是稳定性选择。
- **H-negative**：basketball 的负迁移应表现为共同前缀任务进展不升，且 early
  failure 或 stability 不改善；这用于检验方法何时必须 abstain。

## 3. 代表任务

| 任务 | 角色 | 主要硬指标 |
|---|---|---|
| cabinet | 既有 task-progress 正例 | `max(success_subtasks)` |
| maze | locomotion 到目标进展正例 | `max(success_subtasks)` |
| powerlift | 已知 stability-only 候选 | `max(reward_dumbbell_lifted)` |
| basketball | 最新负迁移反例 | `max(success)`，辅以 catch/throw stage 与球/手接近度 |

这四个任务覆盖“真进展、稳定性主导、负迁移”，比继续横向扩任务更有判别力。

## 4. 两阶段实验

### P0：回顾性去混杂（不重训）

使用已存在的 scratch/wfix 三 seed checkpoint，并固定检查 10k、30k（warmup
边界）和 100k；每个 checkpoint 使用完全相同的 eval seed 与 env rank。对每个
condition/scratch episode pair：

1. 取共同前缀 `h_pair=min(len_condition,len_scratch)`；
2. 分别在 `h_pair` 内计算主要 task-progress；
3. 同时报告 raw progress、episode length、early failure、stability；
4. episode 先在 train seed 内平均，再以 train seed 为统计单位。

这样，`raw Δ>0` 但 `common-prefix Δ≈0` 的结果可被明确判定为“多活带来的
暴露收益”，而不是任务能力加速。

资源修订（正式读取任何结果前）：共享服务器上 16-env smoke 将系统 load 从约
37 推高到约 74，因此正式 P0 改为 8 env × 4 eval seeds，即每个 train seed / step
共 32 个配对 episode。统计单位和共同前缀估计量不变，并固定 OMP/BLAS 单线程。

### P1：源身份因果对照

只新增 `stand-only / walk-only / run-only`，固定以下变量：

- `bootstrap_only`；
- warmup 30k；
- teacher execution probability 0.5；
- horizon 25；
- 单源 weight 1；
- 相同 action groups、FastTD3 超参和 target seeds。

先跑四任务 seed 1，共 12 runs。只有当某一来源在共同前缀指标上出现具有机制
意义的差异，才对相应 sentinel task/source 补 seed 2/3；不预先启动 36-run 全矩阵。

## 5. 主要估计量与报告顺序

主要估计量：

\[
\Delta P_{\text{common}}
=P_{\text{condition}}(1{:}h_{pair})
-P_{\text{scratch}}(1{:}h_{pair}).
\]

报告顺序固定为：

1. common-prefix task-progress；
2. raw task-progress；
3. episode length / early failure / fall（能可靠分类时）；
4. return 与 AUC；
5. task success 或 subtask milestone。

return 不再作为判断“任务技能是否迁移”的首要指标。

## 6. 停线与改线标准

- 若 cabinet、maze 在共同前缀上的优势均消失：停止“任务技能迁移”主张，主线
  收窄为 stability-prior replay bootstrap。
- 若 powerlift 只有 episode length/return 提升而举重进展不变：记为稳定性正例，
  不计作复杂任务完成能力提升。
- 若 stand 与 walk/run 的差异只由 survival 解释：T⁰/源选择不得再称为
  transferability metric，应重构为 stability opportunity estimator。
- 若 basketball 所有单源均负：当前源库覆盖不到其 bottleneck；后续优先扩充
  task-relevant primitives，而不是继续调 softmax 或 horizon。
- 只有在 leave-one-target-out 验证中稳定优于 `stand score`、fall rate、prefix
  reward 等简单基线，新的 source-data-value 指标才可进入方法贡献。

## 7. 工具和执行边界

- 实验定义：[configs/experiments/stability_deconfounded_audit_v1.json](../configs/experiments/stability_deconfounded_audit_v1.json)
- checkpoint 收集与共同前缀汇总：[scripts/stability_deconfounded_audit.py](../scripts/stability_deconfounded_audit.py)
- 单源 bank/命令生成器：[scripts/build_stability_audit_banks.py](../scripts/build_stability_audit_banks.py)

先执行只读检查：

```bash
python scripts/stability_deconfounded_audit.py collect \
  --spec configs/experiments/stability_deconfounded_audit_v1.json --dry-run
```

生成单源 bank 和 seed-1 命令，但不会启动训练：

```bash
python scripts/build_stability_audit_banks.py --write --emit-commands --seeds 1
```

正式训练必须在记录 GPU、commit/diff、source checkpoint 哈希和实验 stamp 后单独启动。
