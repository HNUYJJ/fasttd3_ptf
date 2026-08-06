# Classic PTF × FastTD3：作者公开代码忠实模式的修复与单种子可行性验证

> 日期：2026-07-23  
> 研究问题：恢复作者公开 PTF 代码的关键 option/termination 语义后，
> 原始 PTF 调度机制在 HumanoidBench + FastTD3 上是否仍然没有可用信号？  
> 结论等级：**单种子机制与性能可行性证据，不是正式多种子性能结论。**

## 1. 为什么需要重新验证

此前 classic PTF 实验不是作者公开代码语义的直接复现，而是混入了多个项目适配：
null/student option、两层大网络、线性 Q、截断并重参数化的 beta、Polyak option target、
advantage clamp、beta-weighted transfer、线性 transfer schedule 等。已有负结果只能裁决
这个混合实现，不能直接推出“PTF 与 FastTD3/HumanoidBench 不兼容”。

本轮采用“作者公开源代码优先于论文描述”的基准。详细逐项审计见
`docs/agent_collab/chatgpt_ptf_official_code_fidelity_reaudit_20260723.md`。

## 2. 本轮恢复的作者代码语义

新增显式 `released_code_fidelity` 模式，不改变旧实验的默认路径。主要恢复：

1. option bank 只含冻结 source，至少两个 source，不含 null/student option；
2. episode reset 后立即 epsilon-greedy 选 source，之后由 beta 按 call-and-return
   语义决定何时重选；
3. 一层 `Linear -> ReLU6` 共享干，`Q=tanh(.)`，`beta=sigmoid(.)`，小权重零偏置初始化；
4. beta 使用当前 transition/current option，每个外层环境步只更新一次，并先于 Q 更新；
5. Q 使用作者代码的 online argmax/online beta、target Q 组合和 compatible-option
   求和后 batch mean；
6. option target 每 1000 次 option 更新硬拷贝，不做 Polyak；
7. 不使用 advantage clamp、beta warmup、beta logit clip 或 beta transfer 权重；
8. transfer weight 使用作者代码的 tanh 时间调度，探索概率从 1.0 衰减到 0.1。

仍保留的必要适配：

- FastTD3 actor/critic、双 critic update 和 deterministic actor；
- 随机策略交叉熵改为 deterministic masked Huber 蒸馏；
- 高维确定性动作的 soft Gaussian compatibility；
- HumanoidBench observation/action adapter、向量环境和 source-free evaluation。

因此它是**作者关键机制的 FastTD3/HumanoidBench 忠实适配**，不是逐条运算完全相同的
A3C/PPO 复刻，也不是新的迁移性指标。

## 3. 验证链

### 3.1 公式与回归测试

聚焦测试共 38 项通过，覆盖：

- 作者网络结构、初始化和输出；
- online/target option TD target；
- mixed-source compatible Q loss；
- 未截断 termination loss；
- tanh transfer schedule；
- reset 立即选 source；
- replay uniform 分支严格遵守 allowed mask 的真实 mixed-source 回归。

### 3.2 200-step wiring smoke

200 步训练完成，stand/walk/run 三个 source 均被构建，option 与 beta optimizer
各更新 189 次，所有参数有限，option online/target 已产生非零差异。该阶段只证明
接线和更新链可运行。

### 3.3 5k signal gate

单 seed 5k gate 通过“继续一次 30k 可行性验证”的最低门：

- beta 不再统一贴在 0.05 下轨；
- 在线日志中当前 argmax/non-argmax beta 分别约为 0.491/0.784；
- 冻结状态上 stand 的状态条件 beta 差约 0.528；
- option 发生真实切换，三个 source 都获得非零 rollout 占比。

这证明作者机制信号得到部分恢复，但不足以批准多种子实验。

### 3.4 30k 单种子 feasibility

训练成功完成，option 与 beta optimizer 各更新 29,989 次，checkpoint 参数有限。
source-free 在线评估如下：

| step | return |
|---:|---:|
| 5k | 5.21 |
| 10k | 15.41 |
| 15k | 25.32 |
| 20k | 67.31 |
| 25k | 94.97 |

5k--25k normalized trapezoidal AUC 为 **39.53**。

option 执行也随训练阶段发生变化：

| step | stand | walk | run | mean option age | termination rate |
|---:|---:|---:|---:|---:|---:|
| 5k | 0.336 | 0.289 | 0.375 | 0.17 | 0.778 |
| 10k | 0.195 | 0.352 | 0.453 | 8.43 | 0.794 |
| 15k | 0.141 | 0.484 | 0.375 | 16.59 | 0.746 |
| 20k | 0.039 | 0.477 | 0.484 | 60.82 | 0.650 |
| 25k | 0.156 | 0.156 | 0.688 | 59.34 | 0.569 |

在 30k checkpoint 的 7,245 个冻结评估状态上：

| source | Q argmax 占比 | beta（argmax 状态） | beta（非 argmax 状态） | 差值 |
|---|---:|---:|---:|---:|
| stand | 0.531 | 1.000 | 1.000 | ≈0 |
| walk | 0.025 | 0.417 | 0.764 | +0.347 |
| run | 0.444 | 0.011 | 0.918 | +0.908 |

这组结果的机制解释是：

- `Q_omega` 已经对状态进行 option-value 分区，而非所有 source 得到同一排序；
- run 在被 Q 判为最优时几乎持续，在非最优时高概率终止；
- walk 也表现出同方向的状态条件终止；
- stand 虽在不少状态上 Q 最大，却几乎总被终止，提示其 Q/beta 一致性仍不完美，
  不能把该 source 的 argmax 占比直接解释为有益迁移占比。

独立 8-episode source-free 固定面板的 return 为
`121.17 ± 32.49`（均值 ± population SD，中位数 107.69）。评估中出现一次 MuJoCo
`QACC` 数值警告，但 8 个 episode 全部返回结果，训练 checkpoint 参数也全部有限；
该警告需在正式对照实验中继续监控。

## 4. 与历史结果的描述性比较

历史同 seed 曲线并非当前 checkout 下的严格匹配对照，只能作为尺度参考：

| 5k--25k nAUC | 数值 |
|---|---:|
| 本轮 released-code fidelity | **39.53** |
| 历史 scratch | 11.04 |
| 历史旧 multi-teacher 混合实现 | 14.81 |
| 历史 fixed-walk | 56.16 |
| 历史 single-source learned | 58.04 |

因此本轮相对旧 multi-teacher 实现恢复了明显的性能可行性，并在 25k 达到 94.97；
但它仍低于历史 fixed-walk/single-source 曲线。由于代码、配置和运行批次不完全匹配，
这些差值**不能写成因果增益或论文主结果**。

## 5. 最终裁决

### 已支持

1. 先前 classic PTF 中的 beta 贴轨和弱调度信号，至少部分来自非必要实现偏差；
2. 恢复作者公开代码关键语义后，PTF option/termination 在
   HumanoidBench + FastTD3 上可以学习出非平凡、状态条件的切换信号；
3. 单 seed 30k 性能曲线具有继续做严格匹配对照的价值。

后续 option reward-scale gate 对第2点增加了重要限制：未缩放run中的状态条件beta
依赖已发生tanh饱和的Q动力学，不能再称为稳健健康的termination；去饱和后Q稳定偏向
已知有效的walk教师，但微小gap未能驱动beta。

### 尚未支持

1. 不能声称 released-code fidelity 显著优于 scratch 或 fixed teacher；
2. 不能声称自动教师选择已经学会“哪个 source 对 student 的未来学习最有益”；
3. 不能声称 `Q_omega` 是本课题最终需要的迁移性指标；
4. 不能直接启动完整多任务、多种子矩阵。

`Q_omega` 估计的是当前 call-and-return option 下的 target return。它的监督仍来自
student transition 的即时 reward/TD target，不是“接受 source 蒸馏更新”相对
“student 自学更新”的反事实未来学习增量。

## 6. 下一项最小实验

> 2026-07-23后续裁决：本节计划已被 option reward-scale gate 取代。

后续实验确认未缩放 fidelity 的 tanh Q 已发生量纲饱和；但使用理论固定缩放0.01
消除饱和后，Q gap和状态条件termination反而退化。原计划的scratch/fidelity/
fixed-walk三臂当前暂停；若论文需要严格匹配的PTF baseline，可作为独立baseline
补跑，而不是用于抢救termination。详见
`docs/experiments/classic_ptf_option_reward_scale_gate_20260723.md`。

## 7. 主要产物

- 配置：`configs/experiments/classic_ptf_hurdle_released_fidelity_v1.yaml`
- source bank：`configs/source_banks/pure_ptf/h1hand_hurdle_loco3_released_fidelity.yaml`
- 200-step log：`logs/classic_ptf_released_fidelity/smoke_s1_200.log`
- 5k log：`logs/classic_ptf_released_fidelity/gate_s1_5k.log`
- 30k log：`logs/classic_ptf_released_fidelity/feasibility_s1_30k.log`
- 离线诊断：`docs/data/classic_ptf_released_fidelity/`
- W&B 5k：`5mbmig99`
- W&B 30k：`9ihr3iju`
