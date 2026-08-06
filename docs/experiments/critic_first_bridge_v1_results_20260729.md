# Critic-First Bridge Bootstrap-and-Forget v1 结果

> 日期：2026-07-29。预注册与实现 commit：`72b947a`。
> 裁决：**ENGINEERING PASS / SCIENTIFIC FAIL**。这是单 learner-seed 的
> feasibility gate，不构成跨 seed 性能估计。

## 1. 要检验什么

从同一 10k pure-student anchor 出发，在 `[10k,12k)` 给三臂完全相同的
2k 时间预算：

- `student_freeze`：全 student 数据，冻结 actor；
- `interleaved`：50% source 行为与约 50% source critic replay，actor/critic
  同步更新；
- `critic_first`：与 interleaved 相同的 source bridge，但 bridge 期只训练
  critic，12k hard-exit 后再恢复 actor。

唯一假设是：critic-first grounding 能保留 Slide–walk 的正迁移，同时减少
Door–run 的负迁移。预注册要求同时通过正、负两个 gate，否则停止该分相机制。

## 2. 工程验收

全部通过：

| task/arm | bridge behavior source share | bridge critic source share | actor@12k | actor@20k | 12k 后 behavior/critic source 增量 |
|---|---:|---:|---:|---:|---:|
| Slide interleaved | 0.4697 | 0.4953 | 11989 | 19989 | 0 / 0 |
| Slide critic-first | 0.4674 | 0.4950 | 9989 | 17989 | 0 / 0 |
| Door interleaved | 0.4984 | 0.4954 | 11989 | 19989 | 0 / 0 |
| Door critic-first | 0.4997 | 0.4953 | 9989 | 17989 | 0 / 0 |

anchor actor update count 为 9989。三臂 critic 均由 19978 增至 23978（12k）
和 39978（20k）。source hard-exit 后 authority 为 false、active source buffer
计数为 0。因此失败不能归因于 actor 未冻结、bridge 剂量不匹配或 source 未退出。

## 3. 128-episode source-free 结果

| task | student-freeze | interleaved | critic-first | CF − INT | CF − SF |
|---|---:|---:|---:|---:|---:|
| Slide | 49.53 | **109.27** | 44.55 | **−64.72** | **−4.98** |
| Door | 266.58 | **282.68** | 248.21 | **−34.47** | **−18.37** |

冻结裁决位：

- Slide `J_CF > J_INT`：false；`J_CF > J_SF`：false；
- Door `J_CF > J_INT`：false；`J_CF >= J_SF`：false；
- 最终：`FAIL`。

相同 reset-seed 面板的 paired episode 诊断也表明差异远大于 evaluator 抽样误差：

| contrast | paired 90% interval |
|---|---:|
| Slide CF − INT | [−69.05, −60.39] |
| Slide CF − SF | [−7.57, −2.40] |
| Door CF − INT | [−39.16, −29.78] |
| Door CF − SF | [−24.08, −12.66] |

这些区间只描述同一 learner seed 下 128 个冻结 episode 的评估可靠性，**不是**
跨 learner-seed 的置信区间。

## 4. 科学解释边界

本 gate 支持一个明确的否证：在这两个预注册 source–target、10k 决策点和
2k bridge 预算下，**“先让 critic 吸收 source 数据、延迟 actor”不是有效的
bootstrap 分相**。Slide 中 interleaved 保留了强正迁移，而 critic-first
退化到略低于 student-freeze；这说明 source bridge 的收益至少在该设置下需要
actor 与 critic 同步适应，不能只归功于 critic 先建立 target-grounded value。

Door 也没有出现预期减害。值得注意的是，本轮更短、12k hard-exit 的
interleaved Door 点估计反而高于 student-freeze；这不能由单 seed 升级为
“Door 已转为正迁移”，但足以否定 critic-first 相对 interleaved 更安全的主张。

本结果**不否定**：

- reward-bearing bootstrap/RBO 在部分任务的既有正结果；
- 其他 bridge 长度或更新算法在宇宙中不可能有效；
- source utility 可被其他尚未提出的信号预测。

它只关闭本次预注册机制。按停止规则不改变 bridge 长度、source、task、剂量或
阈值，不启动正式多 seed，也不做救回搜索。

## 5. 可复现性说明

训练主改动绑定 commit `72b947a8730fc79f12d9dc997c31101ffdfb5402`。
当前仓库在该 commit 上仍依赖此前工作树中的未提交模块
`target_evidence.py` / `target_evidence_probe.py` 及 classic-PTF 相关修改；纯净
checkout 会在导入阶段失败。因此本结果同时绑定运行时文件哈希，不能误称为
仅凭该 commit 即可复现。完整科学输入与结果在：

- `docs/data/critic_first_bridge_v1/feasibility_result.json`；
- `docs/data/critic_first_bridge_v1/source_free_eval/`；
- `scripts/analysis/adjudicate_critic_first_bridge_v1.py`。

这一仓库打包缺口不改变本轮臂间比较，因为六臂使用同一运行时快照；但在论文
复现实验前必须把依赖文件选择性提交并冻结完整源码快照。
