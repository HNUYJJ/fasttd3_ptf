# Classic PTF + FastTD3 单教师实验结果 v2

> 日期：2026-07-21--22  
> 目标任务：`h1hand-hurdle-v0`  
> 冻结源策略：`h1hand-walk-v0`  
> 正式矩阵：learned PTF / fixed-transfer / scratch × seeds 1/2/3 × 100k outer steps  
> W&B project：`ptf_fasttd3_classic_revisit`

## 1. 本实验实际验证的机制

本实验只启用 PTF 的结构性主路径：冻结 walk source 作为一个 option，并加入
`no_transfer` option；student FastTD3 actor 始终控制目标环境；option-value 网络
学习选择 option，termination 网络学习 beta；当 walk option 被选中时，使用
`lambda * (1 - beta)` 加权的动作蒸馏损失更新 student。`lambda` 在 100k 内从
1 线性衰减到 0。

明确关闭：source action execution、reward-bearing source replay、warmup bootstrap、
MCG、admission、adaptive admission、replay reweighting、TTL 和当前 RBO 机制。
因此 source 不与 Hurdle 环境交互，也没有 source transition 注入 replay buffer。

这里的“classic PTF”是 PTF 的 option-value / termination / policy-transfer 结构在
FastTD3 与 HumanoidBench 上的数值适配版本，不声称逐行复现原论文实现。

### 1.1 原 PTF 的“策略迁移”究竟是什么

原论文中环境动作由 target policy 采样；被 option 选中的 source policy
作为一个补充模仿目标更新 target policy，而不是直接接管环境。原文明确把
这个迁移通道归为 policy distillation：随机策略用
`H(pi_source || pi_target)` 交叉熵，论文中再乘以
`f(t) * (1 - beta)`。官方连续动作代码对 teacher/student Gaussian
分布计算 cross-entropy，不是对单次采样动作直接做 L2。
需要注意，官方 PPO/A3C 发布代码虽计算并传入 termination，活跃的
continuous transfer-loss 行只乘时间衰减和 `c1`，`(1-beta)` 项反而留在
注释掉的表达式中。因此当前的 beta-weighted transfer 更接近论文公式，
但不是官方发布代码活跃数值路径的逐行复现。

当前 FastTD3 actor 是确定性策略，没有可供对齐的显式高斯分布。因此实现用
student 和冻结 teacher 在同一 target state 上的动作均值输出做
Huber/Smooth-L1 匹配，并加到 FastTD3 actor objective，是确定性策略下的
合理近似，但不是原式的字面复现。本实验实际使用的是 Huber，不是 L2；
其对 target-state 上的分布外 teacher action 比纯 L2 更不易被极端误差主导。

## 2. 工程与执行完整性

- 3k real-environment smoke 通过；source 成功加载，walk/no-transfer 均被选择，
  beta、兼容性和 transfer weight 均为有限值。
- learned PTF、fixed-transfer 与 scratch 各 3 条正式训练全部完成，
  无失败或残留进程。
- 每条训练产生 5k--95k 共 19 个在线确定性评估点和 100k final checkpoint。
- 9 个 final checkpoint 均额外使用同一冻结面板做结构性 source-free 评估：
  4 eval seeds × 8 ranks = 32 episodes；只加载 student actor 和 obs normalizer。
- 相关 option/replay 单元测试：25 passed。

## 3. 主要性能结果

### 3.1 早期样本效率：明确正向

5k--30k normalized AUC：

| seed | PTF | scratch | paired delta |
|---:|---:|---:|---:|
| 1 | 73.306 | 12.883 | +60.423 |
| 2 | 41.806 | 13.335 | +28.471 |
| 3 | 46.485 | 15.517 | +30.968 |
| mean | 53.866 | 13.912 | **+39.954** |

三个 seed 全部为正。30k return 为 `101.9 ± 41.1`（PTF）对
`25.3 ± 2.1`（scratch），paired mean delta = **+76.7，3/3 为正**。
所以在此正向 source-target 对上，纯 PTF 路径能够稳定提高早期样本效率。

### 3.2 全训练 AUC：均值为正但不稳定

5k--95k normalized AUC：

| seed | PTF | scratch | paired delta |
|---:|---:|---:|---:|
| 1 | 463.052 | 197.782 | +265.269 |
| 2 | 99.598 | 209.541 | -109.944 |
| 3 | 143.940 | 169.845 | -25.905 |
| mean | 235.530 | 192.389 | **+43.140** |

均值被 seed 1 的大幅收益主导，只有 1/3 seed 全程 AUC 为正；paired delta 的
seed SD 为 196.905。因此不能把 `+43.1` 解读为稳定的整体改进。

### 3.3 最终 source-free student：总体不如 scratch

100k final checkpoint，冻结 32-episode source-free panel：

| seed | PTF | scratch | paired delta |
|---:|---:|---:|---:|
| 1 | 875.189 | 629.035 | +246.154 |
| 2 | 177.690 | 695.585 | -517.895 |
| 3 | 439.759 | 608.354 | -168.596 |
| mean ± seed SD | 497.546 ± 352.322 | 644.325 ± 45.581 | **-146.779** |

PTF 只有 seed 1 提升，seed 2/3 均下降，并把跨 seed 方差显著放大。因此当前
实现不能支持“纯 PTF 稳定提升最终性能上限”的主张。

## 4. Option / termination 机制诊断

跨三个 seed 的 rollout option 均值：walk 从 5k 的 48.5% 增至 95k 的
68.7%；同时有效 transfer weight 从 0.902 降至 0.047。也就是说，训练末期
source 影响主要由预设的全局 lambda 衰减消失，而不是 termination 稳定地切换到
`no_transfer`。

更直接的反例是 seed 1：它是唯一最终显著优于 scratch 的 seed，但 95k 仍有
96.1% rollout 选择 walk，所选 walk 的 beta 仅 0.054。成功轨迹并没有表现出
“termination 判定教师已过时并退出”的理想 handoff。

seed 2 展示了 option/termination 确实在动态工作：25k 时 no-transfer 占 73.4%、
beta=0.602、return 降至 18.9；30k 又恢复选择 walk 61.7%，return 回升到 67.7。
但这种切换没有带来稳定训练，100k 最终仅 177.7。seed 3 在 75k 从约 220
短暂跌至 23.9，80k 又恢复至 272.1，也表明 learner 轨迹本身存在较强波动。

因此本实验支持：

1. option 和 termination 网络不是死代码，能够改变 source/no-transfer 调度；
2. walk 蒸馏能稳定加速 Hurdle 前期学习；
3. 当前 option/termination 信号尚未学出可靠的阶段性交接；
4. 预设 lambda decay 而非 learned termination 承担了主要的最终退场责任；
5. 早期正迁移并不自动转化成更好的 source-free 最终策略。

## 5. Fixed-transfer 因果对照前言

本实验否定了“PTF + FastTD3 在 HumanoidBench 上完全不起作用”的强说法；它在
walk -> hurdle 上给出 3/3 seed 的早期加速。但是，它也不支持“当前纯 PTF 已经
实现稳定自适应迁移或可靠教师终止”。单教师实验还不能证明 option-value 网络能在
多个教师之间选出当前最优教师。

为区分收益来源，我们执行了以下最小因果对照：

- learned PTF：当前 Q_o + beta；
- fixed-transfer control：同一个 walk 教师、同一 lambda/兼容性/蒸馏预算，但固定
  选择 walk，beta 不进入蒸馏权重；
- scratch：当前对照。

该消融用来回答“收益来自 walk imitation 本身，还是 learned
option/termination 的额外贡献”；结果见下文。

## 6. Fixed-transfer 因果对照

fixed-transfer 移除 `no_transfer` option，始终选择 walk，保留同样的
`lambda: 1 -> 0 / 100k` 和 Huber 动作蒸馏，但不让 beta 进入 transfer
weight。教师仍不接管环境，也不向 replay 注入 source transition。

### 6.1 样本效率：收益主要来自动作蒸馏，不是当前自适应调度

| metric | scratch | fixed-transfer | learned PTF |
|---|---:|---:|---:|
| 5k--30k nAUC | 13.912 | **58.292** | 53.866 |
| 5k--60k nAUC | 48.809 | **165.915** | 136.967 |
| 5k--95k nAUC | 192.390 | **293.383** | 235.530 |

| comparison | 5k--30k | 5k--60k | 5k--95k |
|---|---:|---:|---:|
| fixed - scratch | +44.380 (3/3) | +117.106 (3/3) | **+100.993 (3/3)** |
| learned - scratch | +39.954 (3/3) | +88.158 (3/3) | +43.140 (1/3) |
| learned - fixed | -4.426 (1/3) | -28.948 (1/3) | **-57.853 (1/3)** |

fixed 在全部三个时间范围的平均 AUC 均高于 learned PTF，并且 5k--95k
相对 scratch 三个 seed 全部为正。learned PTF 只有 seed 1 优于 fixed，seed
2/3 均更差。这说明已观察到的正迁移通道是 walk action imitation；当前
Q_o/beta 自适应模块没有产生稳定的额外收益，反而放大了训练方差。

60k return 是一个直观截面：fixed 为
`242.6 / 488.5 / 428.5`，learned PTF 为 `510.7 / 123.4 / 198.0`，
scratch 为 `175.9 / 167.3 / 161.9`。learned PTF 的 seed 1 非常强，但 seed
2/3 被不稳定的 option/beta 调度拖累；固定蒸馏则 3/3 高于 scratch。

### 6.2 最终 source-free student：基本恢复 scratch，但没有稳定提高上限

100k 冻结 32-episode source-free panel：

| seed | scratch | fixed-transfer | learned PTF |
|---:|---:|---:|---:|
| 1 | 629.035 | 557.754 | 875.189 |
| 2 | 695.585 | 733.195 | 177.690 |
| 3 | 608.354 | 612.189 | 439.759 |
| mean ± seed SD | 644.325 ± 45.581 | **634.379 ± 89.801** | 497.546 ± 352.322 |

fixed - scratch = `[-71.3, +37.6, +3.8]`，mean = **-9.9**，2/3 为正；
learned - fixed = `[+317.4, -555.5, -172.4]`，mean = **-136.8**，只有
1/3 为正。因此固定动作蒸馏实现了稳定的前中期加速，并在 lambda
归零后总体回到 scratch 终点水平；它还不支持“稳定提高最终上限”。

## 7. 更新后的科学裁决

本实验否定了“PTF 式策略蒸馏在 HumanoidBench/FastTD3 上不合适”：
walk -> hurdle 的 fixed-transfer 在 5k--95k AUC 上相对 scratch 是
3/3 正。但是它同时否定了“当前 Q_o/beta 已能可靠自适应交接”：
learned PTF 平均不如 fixed，并且方差大得多。

支持的主张是：

1. 在动作语义对齐的 humanoid source-target 对上，冻结源策略的确定性
   Huber 动作蒸馏是有效的样本效率迁移通道；
2. 当前的 option-value/termination 实现在这一 source-target 对上没有超过
   fixed teacher，不能把收益归因于自动选择或可靠 termination；
3. 该结论仅限于单教师 walk -> hurdle，不证明多教师选择能力，也不是
   通用迁移性指标。

若继续原始 PTF 路线，下一步应先修正“何时撤掉动作蒸馏”的调度问题，
而不是质疑蒸馏通道本身。一个最小后续消融可以区分：`no_transfer`
选择错误，以及同一 beta 同时控制 option termination 和蒸馏权重所带来的
“双重撤退”。本轮不自动扩展到多教师或新指标实验。

## 8. 产物索引

- 设计：`docs/experiments/classic_ptf_hurdle_single_source_v1.md`
- learned 配置：`configs/experiments/classic_ptf_hurdle_single_source_v1.yaml`
- fixed 配置：`configs/experiments/classic_fixed_transfer_hurdle_walk_v1.yaml`
- learned source bank：`configs/source_banks/pure_ptf/h1hand_hurdle_walk.yaml`
- fixed source bank：`configs/source_banks/pure_ptf/h1hand_hurdle_walk_fixed.yaml`
- 训练日志：`logs/train/classic_ptf_hurdle_single_source_v1/`
- 100k source-free eval：
  `docs/data/classic_ptf_hurdle_single_source_v1/final_eval/`
- learned/scratch 训练矩阵日志：
  `logs/train/classic_ptf_hurdle_single_source_v1/matrix_formal_20260721T1710Z/`
- fixed 训练矩阵日志：
  `logs/train/classic_ptf_hurdle_single_source_v1/matrix_fixed_formal_20260722T0435Z/`
