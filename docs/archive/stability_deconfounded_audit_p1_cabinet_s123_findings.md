# P1 Cabinet Single-Source Audit：三种子机制加固

日期：2026-07-11。状态：stand-only / run-only 三种子审计完成；run24 剂量裁决已完成。

> **2026-07-11 效度修正**：旧 collector 未正确播种 Gymnasium reset RNG，因此条件均值与
> 跨训练种子方向仍可作描述性证据，但逐 episode 不是精确同初态反事实，配对统计不作因果
> 解释；cabinet 跌倒不 termination，等长 episode 也不能单独排除姿态机制。修正后的因果协议见
> [`source_intervention_mechanism_gate_v1.md`](source_intervention_mechanism_gate_v1.md)。

## 1. 完整性与估计口径

- 条件：scratch、WFix、stand-only、run-only；walk-only 只有 seed 1，不进入本轮三种子裁决；
- 训练种子：1、2、3；checkpoint：10k、30k、100k；
- 每个 `condition × train seed × checkpoint` 使用 4 eval seeds × 8 env = 32 个 episode；
- 四条件合计 1,152 条记录；缺失 0、重复 0；
- 新补的 stand/run seed 2、3 四次训练均 `exit_code=0`，12 个目标 checkpoint 均可加载；
- 所有条件、所有 checkpoint 的 episode length 均为 1000，early failure 均为 0。

统计单位是训练种子；表中的 `±` 是三个训练种子均值的样本标准差。共同生存前缀
等于完整 1000 步，因此硬进展差异不能由“某条件只是活得更久”解释。

## 2. 三种子硬任务进展

主要指标：每 episode 的 `max(success_subtasks)`。

| condition | 10k | 30k | 100k |
|---|---:|---:|---:|
| scratch | 0.000 ± 0.000 | 0.021 ± 0.018 | 0.500 ± 0.360 |
| stand-only | 0.010 ± 0.018 | 0.073 ± 0.079 | 0.448 ± 0.065 |
| WFix | 0.021 ± 0.036 | 0.260 ± 0.118 | 0.948 ± 0.048 |
| **run-only** | **0.042 ± 0.036** | **0.521 ± 0.284** | **0.969 ± 0.054** |

次级 `max(door_openness_reward)` 给出相同排序：

| condition | 10k | 30k | 100k |
|---|---:|---:|---:|
| scratch | 0.036 ± 0.003 | 0.138 ± 0.044 | 0.644 ± 0.335 |
| stand-only | 0.101 ± 0.079 | 0.197 ± 0.141 | 0.724 ± 0.054 |
| WFix | 0.126 ± 0.060 | 0.517 ± 0.121 | 0.961 ± 0.013 |
| **run-only** | **0.204 ± 0.098** | **0.679 ± 0.232** | **0.969 ± 0.030** |

## 3. 直接条件对照

以下统计不再分别相对 scratch，而是按相同 `train seed × checkpoint × eval seed ×
env rank` 直接配对。`t` 仅为三个训练种子上的描述性 t 值，不作显著性检验承诺。

| contrast | step | hard-progress Δ (mean ± SD) | t vs 0 | seed-wise signs |
|---|---:|---:|---:|---|
| run − stand | 10k | +0.031 ± 0.031 | 1.73 | +, 0, + |
| run − stand | 30k | **+0.448 ± 0.253** | 3.07 | **+, +, +** |
| run − stand | 100k | **+0.521 ± 0.110** | 8.22 | **+, +, +** |
| run − WFix | 10k | +0.021 ± 0.036 | 1.00 | 0, 0, + |
| run − WFix | 30k | +0.260 ± 0.307 | 1.47 | +, +, − |
| run − WFix | 100k | +0.021 ± 0.100 | 0.36 | +, −, + |

run − stand 的逐种子 hard-progress 差值在 30k 为 `+0.719/+0.406/+0.219`，在
100k 为 `+0.625/+0.406/+0.531`。因此 seed-1 的方向在 seed 2/3 得到复现，但
run − WFix 的 30k 优势并非三种子一致，100k 更是基本持平。

## 4. Return 与目标能力的数据价值不等价

- 100k 时 run 相对 stand 的硬进展为 `+0.521 ± 0.110`，return 却只有
  `+3.43 ± 17.5`；两者 return 几乎相同，但开柜能力相差很大。
- 30k 时 run 相对 WFix 的硬进展均值为 `+0.260`，return 仅为
  `+8.77 ± 41.1`，同样没有给出可靠排序。

因此 target-task shaped return 可以反映姿态、控制和 dense shaping，却不能直接当作
一段源数据对后续目标能力学习的价值。若后续做在线源选择，目标应是 delayed learning
value，而不是当前 25 步片段的即时 return。

## 5. 当前可以与不可以声称什么

### 已支持

1. **评估期稳定站立不是 cabinet transfer 的充分解释。** run 与 stand 的评估生存长度
   完全相同，但 run 在 30k/100k 对三个训练种子都产生更高的硬任务进展。
2. **源身份会改变 bootstrap 数据价值。** 在其他训练配置固定时，run 明显优于 stand；
   “动态轨迹更早覆盖柜门交互区域”是与该现象一致、但仍需直接测量的机制假设。
3. **T0 更像 top-source screening，而不是校准的数据价值估计器。** 它把 run 排在首位
   是有用信号，但把 stand 估得接近 run 与目标学习结果不符。
4. **run-only 的主要优势是前期样本效率。** 到 100k，run 与 WFix 的硬进展基本持平，
   目前没有稳定的最终上限提升证据。

### 尚未支持或必须收窄

1. 不能据此声称“多源混合中的 stand/walk 主动有害”；run-only 在 warmup 中获得约
   50% 的 run 片段，而 WFix 只有约 24%，存在源剂量混杂。
2. 不能声称“每阶段最优源会随训练阶段变化”；当前只验证了一个静态 top-1 候选。
3. 不能用片段即时 return 做 winner-take-all；本轮恰好显示 return 与硬能力排序错位。
4. 不能把 cabinet 的动态覆盖机制直接推广到其他任务。

还需保留四个设计边界：这里相同的是**评估期** survival，并未直接统计 warmup buffer
中的 fall/termination；教师只控制 `legs_torso,arms`，hands 仍由学生控制，并非完整教师
轨迹；每种源只使用一个固定 source checkpoint；30k 只有三个训练种子且存在多 checkpoint/
多指标比较。因此 30k 的大效应应写成“跨种子同向”，不写成常规显著性结论。

## 6. 下一步：run 剂量匹配裁决

WFix 在教师分支内的 softmax 概率为 stand 0.4189、walk 0.1007、run 0.4804；再乘
`warmup_exec_prob=0.5`，run 占全部 warmup 片段的期望比例为 **0.240214**。

新条件 `run24` 仅保留 run 教师，并设 `warmup_exec_prob=0.24021406`：

- WFix：run 24.02% + stand 20.95% + walk 5.03% + student 50%；
- run24：run 24.02% + student 75.98%；
- run50：run 50% + student 50%。

30k 硬进展是主裁决指标，door openness 为支持性指标：

- `run24 > WFix`：被替换的 stand/walk 片段平均低于 student，支持“次优源主动稀释”；
- `run24 ≈ WFix < run50`：差距主要来自高价值 run 剂量不足，即机会成本而非毒性；
- `run24 < WFix < run50`：stand/walk 有互补价值，但单位预算价值低于 run；
- `run24 ≈ run50`：24% run 剂量已接近饱和，需要改写简单线性剂量解释。

由于只有三个训练种子，“≈”只作为机制筛选，不作正式等价检验。本轮把 30k hard-progress
平均差绝对值不超过 0.10 视为“实践上较小”的操作阈值；超过 ±0.10 仍需结合逐种子
方向及 door openness 是否同向。不能用 `p>0.05` 宣称相等，也不用 episode 数膨胀统计
置信度。

机器结果：

- [四条件相对 scratch](stability_deconfounded_audit_p1_cabinet_s123_results.md)
- [run 相对 stand](stability_deconfounded_audit_p1_cabinet_run_vs_stand_s123_results.md)
- [run 相对 WFix](stability_deconfounded_audit_p1_cabinet_run_vs_wfix_s123_results.md)
- [P2 run24 剂量匹配裁决](stability_deconfounded_audit_p2_cabinet_run24_findings.md)
