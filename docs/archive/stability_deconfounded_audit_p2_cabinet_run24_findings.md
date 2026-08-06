# P2 Cabinet Run-Dose-Matched Control：机制裁决

日期：2026-07-11。状态：三种子训练、统一评估与预注册裁决完成。

> **2026-07-11 效度修正**：旧评估 collector 使用
> `env.unwrapped.seed(seed)`，没有播种 HumanoidBench reset 实际使用的 Gymnasium
> `np_random`。因此，本报告的条件均值、真实 run 剂量及跨训练种子的方向比较仍可作为描述性
> 证据，且“run24 低于 run50”的预注册剂量裁决不依赖逐 episode 精确配对；但配对误差与
> “完全相同初态反事实”解释不再成立。另因 cabinet 跌倒不触发 termination，1000 步等长不能
> 单独排除姿态混淆。后续因果检验以
> [`source_intervention_mechanism_gate_v1.md`](source_intervention_mechanism_gate_v1.md) 为准。

## 1. 完整性

- `run24` seed 1/2/3 均训练至 100k，meta `exit_code=0`；
- 10k、30k、final 共 9 个 checkpoint 均可加载，final `global_step=100000`；
- 评估为 3 train seeds × 3 checkpoints × 4 eval seeds × 8 env = 288 条新记录；
- 与既有条件合并后的四条件汇总含 1,152 条记录，缺失 0、重复 0；
- 所有比较条件的 episode length 均为 1000，early failure 均为 0。

统计单位是三个训练种子；96 个配对 episode 先在每个训练种子内平均，不能当作
96 个独立训练重复。

## 2. 实际 run 剂量核验

本地 W&B history 在 `_step < 30000` 每 100 步记录一次 MCG 执行横截面。三种子均值为：

| condition | stand | walk | run | student |
|---|---:|---:|---:|---:|
| WFix | 0.207802 | 0.050838 | **0.238939** | 0.502421 |
| run24 | — | — | **0.238294** | 0.761706 |
| run50 | — | — | 0.497579 | 0.502421 |

run24 与 WFix 的实际 run share 只差 `−0.000645`，所以剂量匹配成立。该统计是
299 个时点 × 128 env 的横截面抽样估计，不是精确的 segment-start/transition 计数；
30k 后日志中的 MCG exec 值是未清空的 stale 值，未纳入计算。

## 3. 硬任务进展总览

主要指标：共同生存前缀内每 episode 的 `max(success_subtasks)`，表中为训练种子均值 ± SD。

| condition | 10k | 30k | 100k |
|---|---:|---:|---:|
| scratch | 0.000 ± 0.000 | 0.021 ± 0.018 | 0.500 ± 0.360 |
| WFix | 0.021 ± 0.036 | 0.260 ± 0.118 | 0.948 ± 0.048 |
| run24 | 0.000 ± 0.000 | **0.292 ± 0.172** | 0.896 ± 0.141 |
| run50 | **0.042 ± 0.036** | **0.521 ± 0.284** | **0.969 ± 0.054** |

30k 的逐训练种子结果揭示了明显异质性：

| seed | WFix | run24 | run50 | run24 − WFix | run50 − run24 |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.344 | 0.281 | 0.781 | −0.063 | +0.500 |
| 2 | 0.125 | 0.125 | 0.563 | 0.000 | +0.438 |
| 3 | 0.313 | 0.469 | 0.219 | +0.156 | −0.250 |

## 4. 预注册直接对照

| contrast | step | hard-progress Δ | door-openness Δ | return Δ |
|---|---:|---:|---:|---:|
| run24 − WFix | 10k | −0.021 ± 0.036 | −0.001 ± 0.040 | +5.1 ± 22.4 |
| run24 − WFix | **30k** | **+0.031 ± 0.113** | +0.097 ± 0.191 | +78.6 ± 130.0 |
| run24 − WFix | 100k | −0.052 ± 0.188 | −0.036 ± 0.035 | +14.5 ± 18.8 |
| run50 − run24 | 10k | +0.042 ± 0.036 | +0.078 ± 0.072 | +38.2 ± 35.5 |
| run50 − run24 | **30k** | **+0.229 ± 0.416** | +0.065 ± 0.336 | **−69.8 ± 90.0** |
| run50 − run24 | 100k | +0.073 ± 0.095 | +0.043 ± 0.037 | +3.5 ± 54.6 |

30k 的 `run24 − WFix=+0.031` 落在预注册的 `|mean Δ|≤0.10` 实践小效应区间，
逐种子方向为 `−/0/+`，door openness 也高度异质。因此不满足 active dilution 所要求
的三种子同向为正。

## 5. 严格按预注册裁决

本轮应归入预注册的**第 4 分支：逐种子方向混合，机制结论保持不确定**。

均值轮廓确实是 `WFix 0.260 < run24 0.292 < run50 0.521`，描述性上接近第 2 分支
run-dose / opportunity-cost；但 seed 3 完全反转为 `run50 0.219 < WFix 0.313 <
run24 0.469`。因此不能把均值轮廓升级成已经确认的剂量机制。

当前证据应拆成以下边界：

1. **active dilution 未得到支持。** 在实际 run 剂量相同的条件下，用 student 替换
   stand/walk 没有带来一致的硬进展提升；但近零均值不是等价证明，也不能反向断言
   stand/walk 与 student 完全等价。三个训练种子下该差值的描述性 95% 区间约为
   `[−0.249,+0.311]`，仍容纳有意义的正负效应。
2. **run-dose / opportunity-cost 是与均值最一致的候选解释，但未被严格确认。** 30k
   `run50−run24=+0.229±0.416`，逐种子为 `+/+/−`；不能声称单调剂量收益或 run50
   对所有种子都最优。
3. **WFix 的互补价值也未得到支持。** `run24−WFix` 为 `−/0/+`；100k WFix 均值
   稍高且方差更小，但三个种子不足以证明晚期互补或稳健性。
4. **较强的正面结果是低剂量 run 确实有目标学习价值。** 30k 时 run24 相对 scratch
   的 hard-progress 增量为 `+0.271±0.188`，三个种子分别为
   `+0.250/+0.094/+0.469`，door openness 也为 3/3 正向；这不能由评估期 survival 解释。

因此此前“混入次优源会稀释高价值数据”的表述必须收窄为：

> 在固定教师预算下，把预算分给其他来源必然降低 run 剂量；当前均值与机会成本稀释
> 一致，但跨种子数据尚不能区分机会成本、互补性与目标学习随机性的交互，且没有
> 证据表明 stand/walk 轨迹具有主动毒性。

## 6. Return 再次给出错误机制信号

30k 时 run50 相对 run24 的 hard progress 为 `+0.229`，但 return 却为 `−69.8`；
run24 相对 WFix 的 hard progress 仅 `+0.031`，return 却高出 `+78.6`。即时/shaped
return 不仅未校准数据价值，聚合方向甚至可能相反。

所以不能实施“每 25 步按当前 return 选赢家”的阶段最优注入。更合理的后续方法应估计
某类数据对后续 target learning progress 的边际贡献，并保留 student/null 作为比较臂。

## 7. 当前缺失的机制证据

现有 checkpoint 不保存 replay、MCG 调度器、环境轨迹或 behavior label；训练日志也没有
fall/termination、实际 segment length 和 source-conditioned 状态覆盖。因此：

- 相同的是评估期 survival，不能反推 warmup 训练期稳定性；
- 无法直接证明 run 更早到达 hand-handle/door interaction basin；
- 无法从现有数据计算 per-source transition 的 delayed learning value。

下一轮最小机制工作应先补 instrumentation：精确 segment/transition 计数、planned/realized
length、训练期 done/fall、真实 behavior label，以及 root height/tilt、hand-handle distance、
door angle、reward/progress 等小型 segment summary。完成后再预注册 30k-only 的剂量/覆盖实验，
而不是立即扩大 100k 训练矩阵。

机器结果：

- [四条件相对 scratch](stability_deconfounded_audit_p2_cabinet_run24_s123_results.md)
- [run24 相对 WFix](stability_deconfounded_audit_p2_cabinet_run24_vs_wfix_s123_results.md)
- [run50 相对 run24](stability_deconfounded_audit_p2_cabinet_run50_vs_run24_s123_results.md)
- [实际 warmup source-dose 审计](cabinet_p2_warmup_source_dose.md)
- [启动前协议](stability_deconfounded_audit_p2_run24_protocol.md)
- [下一步：阶段条件化 replay 数据价值探针](stage_conditioned_replay_data_value_probe_v1.md)
