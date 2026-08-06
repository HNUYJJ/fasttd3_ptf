# P2 Cabinet Run-Dose-Matched Control

日期：2026-07-11。状态：启动前预注册已冻结；三种子训练和裁决已完成。

## 目的

区分 WFix 在 30k 落后于 run-only 的两个解释：

1. **run dose / opportunity-cost hypothesis**：WFix 只是减少了高价值 run 数据量；
2. **active dilution hypothesis**：stand/walk 片段不仅替代 run，也比 student 自采样更差。

本实验不检验在线 return 选源，也不声称阶段最优源会动态变化。

## 剂量计算

cabinet WFix 的源权重为 stand 1.426、walk 0、run 1.563，温度为 1。教师分支内：

`p(run | teacher) = softmax([1.426, 0, 1.563])_run = 0.480428119`

原 WFix 的教师概率为 0.5，因此全部 warmup 行为片段中的 run 期望占比为：

`p(run) = 0.5 × 0.480428119 = 0.240214060`

`run24` 使用单一 run bank，并把 `warmup_exec_prob` 固定为 `0.24021406`。

## 固定变量

- task：`h1hand-cabinet-v0`；训练种子：1、2、3；总步数：100k；
- MCG groups：`legs_torso,arms`；ablation：`bootstrap_only`；
- warmup：30k；mode：`safe_bootstrap`；segment horizon：25；
- source checkpoint、observation/action adapter、优化器和目标训练配置与 run50 完全相同；
- 唯一有意改变的训练变量：`warmup_exec_prob: 0.5 → 0.24021406`；
- checkpoints：10k、30k、100k；评估：4 seeds × 8 env × 最多 1000 步。

## 对照与主估计量

- 已有：scratch、WFix、run50，均为三个训练种子；
- 新增：run24，三个训练种子；
- 主要 checkpoint：30k warmup 边界；
- 主要指标：共同生存前缀的 `max(success_subtasks)`；
- 支持指标：`max(door_openness_reward)`；
- 诊断指标：return、episode length、early failure、stability composite；
- 统计单位：训练种子。episode 先在种子内平均，不能把 96 个 episode 当作独立训练重复。

## 预注册裁决

1. `run24 − WFix` 在三个种子方向一致为正，且 door openness 同向：支持 active dilution；
2. `run24 − WFix` 接近 0，而二者都低于 run50：支持 run-dose / opportunity cost；
3. `run24 − WFix` 方向一致为负，而 run50 仍最高：支持低强度互补、但 run 单位预算更高；
4. 方向混合：结论保持不确定，不启动在线 winner-take-all，先增加种子或做数据覆盖诊断。

100k 只判断优势是否保持或收敛，不用来替代 30k 的主裁决。

本轮操作性地把 30k hard-progress 的 `|mean seed-level Δ| ≤ 0.10` 记为实践上较小；这不是
正式等价界。若 `|mean Δ| > 0.10`，仍要求逐种子方向和 door openness 支持，不能用
`p>0.05` 代替等价，也不能把 96 个 episode 当作 96 个训练重复。

额外诊断会直接统计 warmup 训练日志中可获得的 fall/termination 与 source/student 片段
比例；若当前日志未记录这些量，则把该缺口列为后续 instrumentation，而不从评估期
survival 反推训练期稳定性。

## 启动记录（预注册冻结后追加）

| seed | physical GPU | experiment | W&B run |
|---:|---:|---|---|
| 1 | 4 | `h1hand_cabinet_dm_run24_s1_20260711T025916Z` | `m7g91tmt` |
| 2 | 5 | `h1hand_cabinet_dm_run24_s2_20260711T025928Z` | `hlt3rbbx` |
| 3 | 6 | `h1hand_cabinet_dm_run24_s3_20260711T025940Z` | `1ist9tlb` |

三路启动日志均确认：run/null bank、`warmup=30000`、`safe_bootstrap`、
`bootstrap_only`、`mcg_groups=legs_torso,arms`、`warmup_exec_prob=0.24021406`。

## 完成后结果索引（预注册冻结后追加）

严格主裁决落入第 4 分支：30k `run24 − WFix=+0.031±0.113`，但逐种子为
`−/0/+`；`run50−run24` 同样为 `+/+/−`。active dilution 未得到支持，均值轮廓与
run-dose / opportunity-cost 一致但尚未确认。完整解释见
[P2 机制裁决](stability_deconfounded_audit_p2_cabinet_run24_findings.md)。
