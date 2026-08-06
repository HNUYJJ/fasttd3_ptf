# P1 Cabinet Single-Source Audit：seed-1 机制筛选结果

日期：2026-07-10。状态：seed-1 筛选完成；run-vs-stand seed 2/3 加固待运行。

> **2026-07-11 效度修正**：旧 collector 未正确播种 Gymnasium reset RNG，因此条件均值与
> 跨训练种子方向仍可作描述性证据，但逐 episode 不是精确同初态反事实，配对统计不作因果
> 解释；cabinet 跌倒不 termination，等长 episode 也不能单独排除姿态机制。修正后的因果协议见
> [`source_intervention_mechanism_gate_v1.md`](source_intervention_mechanism_gate_v1.md)。

## 1. 完整性与控制

- 条件：scratch、WFix、stand-only、walk-only、run-only；
- checkpoint：10k、30k、100k；
- 每个单元 4 eval seeds × 8 env = 32 个配对 episode；
- 五条件合计 480 条用于汇总的记录（P0 scratch/WFix 192 + P1 单源 288）；
- 缺失 0、重复 0；三个单源训练和审计均退出码 0；
- 所有条件、所有 checkpoint 的 episode length 均为 1000，early failure 均为 0。

因此本表中的 task-progress 差异不含 survival/exposure 混杂。

## 2. 硬任务进展

主要指标：每 episode 的 `max(success_subtasks)`。

| condition | 10k | 30k | 100k |
|---|---:|---:|---:|
| scratch | 0.000 | 0.031 | 0.125 |
| stand-only | 0.031 | 0.063 | 0.375 |
| walk-only | 0.000 | 0.063 | 0.625 |
| **run-only** | **0.063** | **0.781** | **1.000** |
| WFix（多源加权） | 0.063 | 0.344 | 0.906 |

次级 `door_openness_reward` 方向完全一致：

| condition | 10k | 30k | 100k |
|---|---:|---:|---:|
| scratch | 0.032 | 0.094 | 0.276 |
| stand-only | 0.107 | 0.134 | 0.666 |
| walk-only | 0.041 | 0.203 | 0.821 |
| **run-only** | **0.261** | **0.864** | **0.983** |
| WFix | 0.196 | 0.653 | 0.949 |

## 3. Return 与硬进展发生错位

30k evaluation return：scratch 55、stand 69、walk 108、run 243、WFix 279；但硬
subtask 是 run 0.781、WFix 0.344。100k 同样是 WFix return 268 > run 225，然而
run 的硬 subtask 1.00 > WFix 0.91。

这说明：

1. shaped return 会混合 posture/control、dense door shaping 与离散 subtask；
2. 用 prefix return 给源赋权，不能直接等同于该源数据对目标能力学习的价值；
3. 多源混合即便提高 return，也可能稀释最有价值源的 task-progress channel。

## 4. 对 T⁰ 的精确裁决

当前 cabinet bank 的 T⁰ 权重约为 run 1.563、stand 1.426、walk 0。它正确找到了
run 是候选强源，但：

- 把 stand 估得接近 run，而实际 30k task-progress 相差约 12.5 倍；
- 把 walk 置零，但其 100k task-progress 高于 stand；
- WFix 混入 stand 后，30k 硬进展显著低于 run-only。

因此 T⁰ 在本任务上具有“top-source screening”信号，却不是校准良好的 data-value
estimator，也不能证明 softmax mixture 是最优使用方式。

## 5. 新机制假设

run 教师不具备开柜技能，且 run-only 与 stand-only 的 survival 完全相同。run 的
优势更可能来自动态源行为注入了更有利的全身动作、接近/姿态变化或 replay 覆盖，
使 target learner 更快发现柜门交互，而不是直接模仿开柜策略。

可检验命题：

> Dynamic source trajectories can have higher target learning value than stable
> trajectories even when both provide identical survival, because their induced
> state-action coverage reaches the target interaction basin earlier.

这仍是假设；当前只有一个 target seed，不能作为最终贡献。

## 6. 下一步裁决

不继续完整的 4-task × 3-source seed-1 矩阵。先补最小决定性加固：

- cabinet run-only：target seeds 2、3；
- cabinet stand-only：target seeds 2、3；
- scratch 与 WFix 已有三 seed，不重训；
- 完成后按同一共同前缀 `success_subtasks` 与 door openness 汇总。

若 run > stand 在 3/3 target seeds 保持，才进入“动态轨迹数据价值”机制分析；若
差异翻转，则把 seed-1 结果判为高方差，不再扩展该主张。

机器结果：[stability_deconfounded_audit_p1_cabinet_s1_results.md](stability_deconfounded_audit_p1_cabinet_s1_results.md)
