# Door 顺序因果分解（behavior authority × replay eligibility）— 结果与裁决

> 日期：2026-07-28  
> 预注册：commit `1175833`（在任何臂被评估之前冻结）  
> **裁决：`UNRESOLVED`** → 按预注册规则停止：不追加 R-only、不加任务、不延长 horizon、
> 不做参数搜索。

## 1. 一句话结论

总效应 $U^{BR}$ 稳健为负（3/3 seed，90% CI 完全 <0），**但把它分解到两条通道后，
三个 learner seed 给出了方向相反的归因**：两个 seed 显示"行为有害、replay 反而补偿"，
一个 seed 显示"行为无影响、replay 有害"。两个分量的区间因此都跨 0。

按预注册，joint 显著而两分量各自不显著**只能**裁 `UNRESOLVED`，**不得**称为纯交互——
那也正是功效不足的样子。

## 2. 被修复的机制缺陷

Door@10k 表明两条迁移通道可能方向相反，而代码把它们焊在了一起：在
`warmup_mode=admission_bootstrap` 下，`McgBehaviorController.step`
（`fasttd3_ptf/ptf/mcg.py:547-560`）**直接对 student-inclusive categorical 做
multinomial 决定谁执行**，注释原文是 *"sample once from admitted sources + student,
with no outer teacher Bernoulli"*；而同一组 logits 又经
`rb.set_admission_policy` 决定 critic replay 的来源配额。

**一个标量同时决定 behavior authority 与 replay eligibility**——这是概念缺陷，不是实现细节。

本轮新增 `admission_replay_mode: shared | student_only`，**只**覆盖 replay 侧配额，
behavior 侧一律保持原分布。默认 `shared` 与历史行为逐位相同（26/26 既有
replay+admission 测试全过）。

> **一条被实测否决的捷径**：曾设想用极端 `student_logit` 模拟关闭 replay 通道。
> 30 秒探针显示它把 **behavior share 直接打到 0.000000**、buffer 里一条 source 数据都没有
> ——因为两条通道共用同一个 categorical。这反过来构成该缺陷存在的直接证据。

## 3. 分解定义（PI 冻结）

$$U^{B}=J_B-J_0,\qquad \Delta^{R\mid B}=J_{BR}-J_B,\qquad U^{BR}=U^{B}+\Delta^{R\mid B}$$

- $U^{B}$：固定交互预算下授予 behavior authority 的**总**效果，**包含** source 占用了
  原属 student 的交互机会。这是行为通道的真实机会成本，不是 confound。
- $\Delta^{R\mid B}$：在**同一** behavior authority 条件下，进一步允许 source transitions
  参与 critic replay 的条件增量。
- 恒等式为精确恒等式，实测最大误差 **0.00e+00**。

这不是纯 $U^R$，也不能单独识别交互项。R-only 臂在现有架构下不可实现：source 不开车就
不产生 source 数据，需要 matched-state shadow rollout，且会打破固定交互预算的可比性。

## 4. 执行完整性

3/3 臂 `Resumed core learner ... at step 10000`，bank 均为 `['run']`。
与 door gate 的 joint 臂逐项对照：

| seed | behavior share (B-only / joint) | buffer source (B-only / joint) | critic source 采样 |
|---|---|---|---|
| 1 | 0.4976 / 0.4981 | 636980 / 637539 | **0** |
| 2 | 0.5009 / 0.5004 | 641094 / 640530 | **0** |
| 3 | 0.4995 / 0.4986 | 639334 / 638168 | **0** |

**behavior 通道完全未受影响**（差异 <0.1%），而 critic source 采样严格为 0。
B-only 臂的干预是纯粹的：唯一变化就是 critic 不再采 source provenance。
`replay_mode=student_only` 已写入三份 checkpoint 的 `ptf_cfg`。

## 5. 主结果

| seed | $J_{\text{student}}$ | $J_{\text{B-only}}$ | $J_{\text{joint}}$ | $U^{B}$ | $\Delta^{R\mid B}$ | $U^{BR}$ |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 265.47 | 206.26 | 247.68 | **−59.21** | **+41.43** | −17.78 |
| 2 | 275.45 | 274.11 | 234.42 | **−1.34** | **−39.69** | −41.04 |
| 3 | 261.69 | 160.18 | 228.61 | **−101.50** | **+68.42** | −33.08 |

3 个 learner seed 的 90% 配对 t 区间（df=2）：

| 量 | mean | 90% CI | 符号 |
|---|---:|---|---|
| $U^{B}$ | −54.02 | [−138.79, +30.75] | unc |
| $\Delta^{R\mid B}$ | +23.39 | [−71.48, +118.25] | unc |
| $U^{BR}$ | −30.63 | [−50.56, −10.71] | **neg** |

## 6. 为什么是 UNRESOLVED：跨 seed 机制异质性，不是评估噪声

episode-level 诊断（**仅诊断，不参与裁决**）：

| seed | $\lvert U^{B}\rvert/\text{pairSE}$ | $\lvert\Delta^{R\mid B}\rvert/\text{pairSE}$ | 归因方向 |
|---|---:|---:|---|
| 1 | **12.48** | **10.68** | 行为有害 −59，replay 补偿 +41 |
| 2 | 0.50 | **12.39** | 行为几乎无影响 −1.3，replay 有害 −40 |
| 3 | **20.18** | **12.43** | 行为有害 −102，replay 补偿 +68 |

**每个 seed 内部的测量都高度可靠**（episode 层面比值 10–20，唯一的例外 s2 的 $U^B$
本身就接近 0）。所以三个 seed 的归因分歧**不是评估噪声，是真实的 learner-seed 机制异质性**。

这直接验证了 PI 对判据的修正：若按我原先建议的"用单 seed 的 128-episode SE 裁决"，
在 seed 1 上会得到"behavior 主导"、在 seed 2 上会得到"replay 主导"——两个互相矛盾且
各自看起来都极显著的结论。**episode SE 不能代替 learner-seed 不确定性。**

## 7. 对原假设的直接后果

预期中的判据 *"B-only 接近 student 而 joint 明显更差"* → 即
"source 行为可以保留，但其数据不应进入 replay" —— **未获支持**。

它只在 seed 2 上成立（$U^B=-1.34$、$\Delta=-39.69$）；seed 1 与 seed 3 **恰好相反**：
source 开车本身造成了 −59 与 −102 的伤害，而把 source 数据放进 replay 反而**补偿**了
+41 与 +68。

因此本轮**不能**支持"独立的 behavior authority 与 replay eligibility"作为已验证的核心机制。
解耦这一**能力**已经实现并通过验收（§4），但它所要支持的**科学主张**在 3 seed 下不成立。

同时必须说清楚：这也**不构成**对该主张的否定。$U^{BR}$ 在 3/3 seed 上稳健为负说明
joint 伤害是真实的；只是"伤害归于哪条通道"在当前证据下不是 (source, target, stage) 的
稳定函数，至少还依赖 learner 的具体轨迹。

## 8. 裁决与停止

`UNRESOLVED`。按预注册：**不追加 R-only 臂、不加任务、不延长 horizon、不做参数搜索、
不加 seed。** Hurdle 未参与本轮（其标签是 $t{=}0,K{=}30\text{k}$，与 Door 的
$t{=}10\text{k},K{=}10\text{k}$ 不可数值比较），Basketball 仍完全未动。

## 9. 产物

- 结果 JSON：`docs/data/door_channel_decomposition_v1/door_channel_decomposition_v1_results.json`
- 3 份冻结评估（各 128 episodes，与 door gate 同一面板）：
  `docs/data/door_channel_decomposition_v1/source_free_eval/*.json`
- 裁决脚本（**揭盲前定稿于 `1175833`**）：`scripts/analysis/analyze_door_channel_decomposition_v1.py`
- 训练/评估脚本：`scripts/run_door_channel_decomposition_v1.sh`、`scripts/analysis/run_door_channel_eval_v1.sh`
- 机制实现：`fasttd3_ptf/official_fasttd3_ptf/train_ptf.py` 的 `replay_candidate_masses`
- 单元测试：`tests/test_replay_channel_decoupling.py`（5 项）
- 训练日志：`logs/train/door_channel_decomposition_v1/`
