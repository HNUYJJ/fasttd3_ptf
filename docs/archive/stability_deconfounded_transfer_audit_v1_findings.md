# Stability-Deconfounded Transfer Audit v1：P0 结果与科研裁决

日期：2026-07-10。状态：P0 完成，P1 单源因果对照待运行。

> **2026-07-11 效度修正**：旧 collector 未正确播种 Gymnasium reset RNG，因此条件均值与
> 跨训练种子方向仍可作描述性证据，但逐 episode 不是精确同初态反事实，配对统计不作因果
> 解释；cabinet 跌倒不 termination，等长 episode 也不能单独排除姿态机制。修正后的因果协议见
> [`source_intervention_mechanism_gate_v1.md`](source_intervention_mechanism_gate_v1.md)。

## 1. 数据完整性

- 任务：cabinet、maze、powerlift、basketball；
- 条件：scratch、静态 RBO/WFix；
- checkpoint：10k、30k、100k；
- 3 个训练 seed；每个单元 4 eval seeds × 8 env = 32 个配对 episode；
- 共 2,304 条 episode 记录；重复 0，缺失 0，四个采集进程退出码均为 0；
- 统计单位是训练 seed（n=3），episode 只在 seed 内平均；t 值仅作描述性证据。

资源修订发生在读取结果前：16 env 将共享服务器 load 从约 37 推至约 74，故改为
8 env × 4 eval seeds，并固定 OMP/BLAS 单线程。估计量和统计单位未改变。

## 2. 主要结果

主要估计量是 condition/scratch 相同初始 seed/rank 配对后，在两者较短存活长度内
的任务进展差：

\[
\Delta P_{common}=P_{RBO}(1:h_{pair})-P_{scratch}(1:h_{pair}).
\]

| task | step | scratch → RBO（共同前缀硬进展） | Δ | 描述性 t | 关键控制 |
|---|---:|---:|---:|---:|---|
| cabinet | 10k | 0.000 → 0.021 subtask | +0.021 | 1.00 | 两组均存活 1000 步 |
| cabinet | 30k | 0.021 → 0.260 | **+0.240** | **3.29** | 两组均存活 1000 步 |
| cabinet | 100k | 0.500 → 0.948 | **+0.448** | **2.25** | 两组均存活 1000 步 |
| maze | 10k | 1.000 → 1.542 checkpoint | **+0.542** | **4.61** | RBO 少活 169 步、failure +0.23 |
| maze | 30k | 1.271 → 1.896 | **+0.625** | **2.18** | 共同前缀已消除暴露时长 |
| maze | 100k | 1.865 → 2.000 | **+0.135** | **2.98** | 3/3 seeds 为正 |
| powerlift | 10k | 0.19044 → 0.19059 | +0.00016 | 0.75 | 实际量级可忽略 |
| powerlift | 30k | 0.19085 → 0.19049 | −0.00037 | −1.08 | stability 上升但 failure 也上升 |
| powerlift | 100k | 0.19038 → 0.19062 | +0.00024 | 3.55 | 统计稳定但无实际举重能力意义 |
| basketball | 10k | 0 → 0 success | 0 | 0 | 两组均失败 |
| basketball | 30k | 0.010 → 0 | −0.010 | −1.00 | RBO 未产生成功 |
| basketball | 100k | 0.323 → 0.135 | **−0.188** | **−3.93** | 3/3 seeds 为负 |

## 3. 次级指标交叉验证

### Cabinet：不是“只是站得久”

- 所有 checkpoint、两种条件的 episode length 都是 1000，early failure 都是 0；
- 30k `door_openness_reward` 最大值 0.138 → 0.517；
- 100k 0.644 → 0.961；
- 因此 cabinet 的学习加速不能由 survival/exposure 解释。

该结果证明 RBO 至少在部分任务上改变了 target learner 获得任务能力的速度，而不仅
是增加累计 alive reward。它仍不等于“源策略直接具备开柜技能”，更准确的解释是
源行为改变了早期 replay/state-action distribution，使目标策略更快学会开柜。

### Maze：任务进展甚至与稳定性方向相反

- 10k 时 RBO episode length 少 169 步、early failure 多 0.23，但 checkpoint
  仍从 1.00 提高到 1.54；
- 同一共同前缀内，stage conversion 100 → 208，checkpoint proximity 近 0 → 0.498；
- 100k 的优势缩小，符合“前期能力获取加速、后期趋于饱和”。

这构成“全是站稳收益”的反例：更早失败的 RBO 仍能在更短时间内推进更远。

### Powerlift：没有任务技能迁移证据

- `reward_dumbbell_lifted` 始终约 0.190，最大差异小于 4.5e−4；
- 30k return +29.6 伴随 stability +0.161，但 episode length −78、failure +0.23；
- 100k return 仅 +13.8±30.5，未形成稳健的最终收益。

因此不能把 powerlift 计作“举重能力迁移”。它至多是姿态/控制分量变化，而且当前
P0 甚至不支持简单的“站得更久”单一解释。

### Basketball：稳定性不是充分条件，且存在任务冲突

- catch/throw stage 两组从早期起均为 1，说明 `success_subtasks` 不是合适硬终点；
- 正式硬指标改为投篮 `success`：100k scratch 0.323，RBO 0.135；
- 100k RBO stability +0.163、episode length +12，但 success 明显下降；
- ball-success proximity 0.968 → 0.899，只有 hand proximity 小幅上升。

这说明“更稳定”并不保证迁移有用：源行为可以改善姿态，却把 target learner 推向
不利于投球/球轨迹控制的数据分布。现有 T⁰/stability score 无法区分这种冲突。

## 4. 对原担忧的裁决

1. **“主要只是 bootstrap”——成立。** P0 没有改变这一结论。
2. **“提升可能全来自 locomotion 稳定性”——部分成立但整体被否证。**
   powerlift 确实没有任务能力证据；但 cabinet 的等存活对照和 maze 的反向稳定性
   对照均显示真实 task-progress acceleration。
3. **“上限提升有限”——成立。** maze 优势从 10k/30k 的约 0.54/0.63 缩小到
   100k 的 0.14；当前主要贡献仍是能力获取速度。
4. **“缺少可靠 transferability metric”——进一步得到支持。** basketball
   展示 stability 更高但任务成功更低，说明任何只用 fall、upright 或 prefix return
   的指标都不充分。

## 5. 新的机制洞见

P0 支持的中心命题应改为：

> Source utility is effect-specific rather than policy-level: the same locomotion
> bootstrap can accelerate target-specific progress, alter only posture/control,
> or conflict with the target bottleneck. Stability is neither necessary nor
> sufficient for positive transfer.

当前应研究的不是一个源策略是否“整体可迁移”，而是它生成的数据对目标任务的哪个
effect channel 有价值：稳定性、状态到达、任务进展，还是负向动作/状态偏置。

## 6. P1 决策

按照预注册标准，cabinet 和 maze 在共同前缀内均保留优势，因此研究主线不应直接
收窄为 stability-only；进入 P1 单源因果对照：

- source ∈ {stand, walk, run}；
- task ∈ {cabinet, maze, powerlift, basketball}；
- 先跑 target seed 1，共 12 runs；
- 固定 bootstrap_only、warmup 30k、exec probability 0.5、horizon 25；
- P1 的目标不是选“冠军源”，而是判断 source identity 对 stability 与 task-progress
  两个通道是否产生可重复的差异。

原始与机器汇总：

- `logs/probe/stability_deconfounded_audit_v1_{task}.jsonl`
- `logs/probe/stability_deconfounded_audit_v1_summary.json`
- [机器生成结果表](stability_deconfounded_audit_v1_results.md)
