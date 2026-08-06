# 整篇论文核心贡献重构 v2：恢复迁移强化学习主线

> **2026-07-12 superseded**：SHU mandatory regression失败后，当前路线改为
> [`paper_core_contribution_reconstruction_v3.md`](paper_core_contribution_reconstruction_v3.md)中的
> 静态RBO主方法、source-bank条件规律与execution/replay双通道边界。本文件保留为被否证设计的审计记录。

日期：2026-07-12  
状态：**历史战略版本；首个机制门已失败；不再执行本文closed-loop设计**  
取代：[`paper_core_contribution_reconstruction_v1.md`](paper_core_contribution_reconstruction_v1.md)

---

## 0. 总裁决

论文继续研究 **HumanoidBench 上的跨任务策略迁移强化学习**，目标至少包括：

1. 在有用 source 上显著提高有限预算 sample efficiency；
2. 在部分任务上探索并验证有限预算 final-performance gain；
3. 在无合适 source 时严格退化为 student-only，而不是保留固定 teacher exposure；
4. 自动决定 source 是否、何时以及以何种 replay 权重参与目标学习。

v1 的 TBS 结论不撤回，但角色改变：

> **Transient Behavioral Scaffolding 解释 source 为什么能通过 target-reward experience 帮助
> off-policy learner；它不再单独充当完整方法贡献。**

新的方法中心暂称：

> **Stage-Conditioned Source Admission and Replay Control**

工作流程：

`matched source→student handoff probe`
→ `stage-conditioned admission`
→ `student-inclusive exact fallback`
→ `quarantined probing`
→ `reward-bearing bootstrap`
→ `admission-consistent replay`
→ `optional MCG/EPS authority`

这保留了 PI 希望的旧总体结构：

`迁移性评估 → 自动选教师 → bootstrap → T-weighted replay → MCG`，

但修复三个已知缺陷：静态 vs-zero 分数、固定 0.5 teacher floor、被拒绝 source 的旧数据长期驻留。

---

## 1. 旧结果没有被推翻，但 claim 必须分层

### 1.1 保留的正面证据

- **hurdle**：50k hard progress `move` 约 `0.731 vs 0.356`，是强 sample-efficiency 正例；
  100k `0.922 vs 0.923` 持平，不能写成 ceiling gain。
- **cabinet**：30k/100k hard progress 有明显优势，且 run 在 30k/100k 均优于 stand；source
  identity 与行为形态重要。
- **maze**：早期 checkpoint acquisition 加速，后期趋于饱和。
- **slide/pole/stair**：reward-weighted source allocation 相对 uniform 有正面证据，同时暴露
  crawl 与 horizon failure mode。
- **truck**：加入 hurdle source 后有限预算表现增加 `+229.9`，是 source diversity 与潜在 final
  gain 的重要候选。

### 1.2 保留的负面和边界证据

- **crawl**：locomotion source 整体有害；student-as-arm 能学到 student 最优排序却不能回到 scratch。
- **basketball**：posture/viability 改善而 success 下降，说明即时稳定性不等于迁移价值。
- **powerlift**：return 变化没有转化为举重技能。
- **door**：近似 null transfer；是 exact fallback 的自然对照。
- **window**：高方差，不作为稳定正例。

这些结果构成 selector/admission 方法的正、负、null 测试集，而不是被新路线删除。

---

## 2. 核心科学对象：Stage-Conditioned Handoff Utility（SHU）

令当前 student policy 为 `π_t`，source 为 `π_i`，当前 student occupancy 为 `d_t`。从同一个
simulator anchor `x~d_t` 构造两条 potential paths：

- `SS(x)`：student 执行 prefix `h`，随后 student 执行 follow-up `f`；
- `iS(x)`：source `i` 执行 prefix `h`，随后同一个 frozen student 执行 follow-up `f`。

分别定义配对效应：

`D_i(t)=E[G_prefix(iS)−G_prefix(SS)]`：source 直接 target-return effect；

`H_i(t)=E[G_follow(iS)−G_follow(SS)]`：source 交权后对 student 的 handoff effect；

`K_i(t)=E[Risk(iS)−Risk(SS)]`：fall/termination/unsafe-state effect。

`G` 使用 target reward；task hard progress 只作 paper audit 与 gate 交叉验证，不进入第一版通用
selector，避免针对任务写规则。所有差值均在相同 anchor、相同 student noise、相同 follow-up budget
下配对计算，并用 anchor-level bootstrap 给出 one-sided confidence bound。

### 2.1 为什么不是单一加权 scalar

旧 `T⁰` 允许 source prefix 的即时高回报补偿对后续 student 的伤害，crawl 正是这种 false positive。
第一版使用非补偿式准入：

1. **non-harm**：`LCB(H_i(t)) >= -δ_H`；
2. **positive evidence**：`LCB(D_i(t)) > δ_D` 或 `LCB(H_i(t)) > δ_H^+`；
3. **risk veto**：`UCB(K_i(t)) <= δ_K`。

任何一项不满足，source 不进入当前 stage 的 eligible set。prefix reward 不能抵消 handoff harm，
posture gain 也不能抵消 task-return follow-up loss。

阈值 `δ` 只允许由 outcome scale、paired-null/duplicate noise 与预注册 practical effect 定义；不按
task/source 搜索。

### 2.2 与已停止 DV/SIV 路线的边界

SHU 不声称估计 transition 对 K 次 learner update 的因果 data value，也不预测跨任务训练 ROI。
它只回答一个更窄且部署相关的问题：

> **在当前 student stage，source prefix 是否把 student 交到一个至少不更坏、并有直接或后续
> target-return 证据的状态分布？**

因此 cabinet 10k formal gate 的 null/negative result 不被推翻；它应使该 stage 的**同一
run-composite candidate**被拒绝，而不是继续训练更复杂 estimator。该标签不自动外推到full-action run。

---

## 3. Student-inclusive admission：删除固定 teacher floor

统一候选集合为：

`Π_t={π_t,π_1,...,π_S}`。

student 是 reference arm，utility 定义为 0，并永远 eligible。source eligible set 为 `A_t`。

- 若 `A_t=∅`：`p(student)=1`，source execution 为 0；
- 若 `A_t≠∅`：只在 student 与 `A_t` 中分配 segment；
- 没有外层固定 `teacher=0.5`；
- source 被后续 stage audit 撤销后立即停止执行；
- source exploration 只发生在独立 probe/quarantine 阶段，不通过主训练行为保留永久 floor。

第一版正 utility strength：

`s_i(t)=max(0, max(LCB(D_i(t))−δ_D, LCB(H_i(t))−δ_H^+))`。

若 source 已过三项 gate，则：

`p_i(t)=s_i(t)/(1+Σ_{j∈A_t}s_j(t))`，

`p_student(t)=1/(1+Σ_{j∈A_t}s_j(t))`。

这里的常数 1 是 student reference mass；effect 必须先按 paired student scale 做无量纲标准化。
如果该 allocation 在首个 gate 中过于敏感，停止 closed loop，不通过温度/grid 修补。

---

## 4. Quarantine：先证明 source 可用，再允许污染 learner

probe interaction 与 training replay 必须隔离：

- paired `SS/iS` trajectories 写入 immutable quarantine bank；
- quarantine 数据只用于 SHU、risk 和审计；
- 无论 source 是否通过，probe transition 第一版均不释放进 main replay；
- probe environment steps 计入总 interaction cost并单独报告；
- scratch/control 获得同等 probe budget，防止把额外交互隐去；
- source 只有通过 admission 后，才允许在正常训练 segment 中产生 main-replay transition。

这解决“为了判断 source 有害，先把有害数据训练进 critic”的逻辑错误。无法避免的是有限的 probe
交互成本；任何声称零 probe 又能识别未知 harmful source 的方法都需要额外 oracle。

---

## 5. Admission-consistent replay：不全历史 uniform，也不逐轨迹贪心

每条 main transition 必须记录：`behavior_source`、canonical `source_by_group`、stage/admission id、
segment id/step、learner step 与当前 admission snapshot hash。

main replay 按 behavior arm 分层。当前 stage 的 sampling source mass 与 admission distribution 对齐：

`q_z(t)=(1−ρ)p_z(t)+ρ·n_z(active)/N_active`。

- `p_z(t)` 是 §3 当前 student/source distribution；
- `ρ` 是覆盖下限，只作用于当前 active/eligible strata；
- 已撤销 source 的旧 transition 不属于 active set，采样概率严格为 0；
- 它们保留在 audit archive，不再训练 learner；
- 每个 active stratum 内优先从最近 window 均匀采样；TD-error priority 只作为后续消融，不进入第一版；
- actor 与 critic 使用同一组 strata distribution，避免已观察到的 AC mismatch；
- 记录每个 transition 的 sampling probability，若后续加入 PER，critic 必须有 importance correction。

这不是“只抽最高 return trajectory”：return 受初态/存活/长度混淆，失败 transition 对 critic 仍有价值。
方法在 **source/stage 层**做保守准入与撤销，在 active data 内保留 Bellman coverage。

---

## 6. Bootstrap、MCG 与 EPS 的新位置

### 6.1 Reward-bearing bootstrap

RBO/TBS 仍是性能主通道：eligible source 在 target environment 连续执行 segment，所有 transition
使用 target reward 进入 main replay；最终 student source-free。

改变的是 source budget 从固定 WFix/0.5 变为 stage-conditioned admission，并允许 source mass 为 0。

### 6.2 MCG

MCG 保留，但位于 policy-level admission 之后：未通过的 source 不进入 body-group gate，也不蒸馏。
post-warmup critic gate/distillation 当前仍是 supporting mechanism，不能与 bootstrap 并列为已证性能贡献。

### 6.3 EPS

EPS 降为已准入 source 的 action-authority 候选：只有 admission/replay 主机制过门后，才检验
full-action 与 `legs_torso` authority。旧 EPS gate 暂停，不删除。

---

## 7. 重构后的论文贡献

### Contribution 1：Stage-conditioned handoff utility

用 source-prefix→student-follow-up 的配对 target-environment intervention，区分直接收益、交权后的
student收益与风险；相对当前 student、随 stage 更新，不再用 vs-zero/static return 假装预测 ROI。

### Contribution 2：Conservative source admission with exact student fallback

student 是一等候选；source 未通过 non-harm/positive/risk gate 时不进入 training；无 source 适合时
行为严格退化为 student-only，probe 数据由 quarantine 隔离。

### Contribution 3：Admission-consistent replay control

source 撤销同时停止其行为执行和旧数据采样；replay 在 active source/stage 层按 admission utility 与
recency分配，actor/critic保持相同数据分布。

### Contribution 4：Humanoid transfer evidence and mechanism audit

用 hurdle/cabinet/maze/slide/pole/truck 正例、crawl/basketball 负例、door/powerlift null，以及
source-free hard progress/finite-budget final performance，验证何时加速、何时提高有限预算终点、何时
必须弃权。

TBS 是 Contribution 1–3 的机制解释；MCG/EPS 只有新证据支持后才升级。

---

## 8. 首个且唯一当前机制门

在任何闭环训练或 EPS 实现之前，必须先回答：

> **SHU 是否能在既有正/负/stage证据上，把有用 source 与有害/过时 source 分开？**

协议见 [`stage_conditioned_source_admission_gate_v1.md`](stage_conditioned_source_admission_gate_v1.md)。
这是 read-only/frozen-policy paired rollout gate，不更新 learner，不启动6–12个100k runs。

若该 gate 失败：停止 SHU closed loop，不调阈值搜索；论文回到 RBO/TBS 实证主线，source selection
只作 heuristic。若通过：才实现 quarantine、student exact fallback、active replay 和 provenance。

**2026-07-12结果：该gate已失败。** cabinet mandatory regression中SHU接受了旧formal downstream
intervention明确应拒绝的run-composite，且null robust scale塌到floor。aggregate route为
`STOP_CLOSED_LOOP`；hurdle/crawl不补跑，Contribution 1–3当前均是被否证的设计目标，不再作为论文
既成贡献。失败揭示SHU只测behavior/handoff utility，不能替代replay/update data utility。

---

## 9. 明确不做

- 不恢复跨任务 scalar ROI predictor；
- 不恢复 `DV/SIV`、gradient influence 或 K-step learner-value estimator；
- 不逐轨迹按 return 硬贪心；
- 不保留固定 0.5 harmful-teacher floor；
- 不把 probe interaction 免费藏在训练预算外；
- 不做 source/task/horizon/mask grid 搜索过门；
- 不把 hurdle early acceleration 写成 asymptotic ceiling；
- 不在 admission gate 前实现 EPS 或启动正式训练。

---

## 10. 执行顺序

1. 冻结 v2 与 SHU gate；
2. 复用现有 simulator anchor/paired path 基础设施，实现多 task/source read-only collector；
3. 工程 smoke：matched state/noise、duplicate、quarantine不写main replay；
4. 运行单个 SHU discrimination gate；
5. Go 后实现 student exact fallback 与 admission-consistent replay；
6. 单 seed closed-loop 对比 positive/negative/null；
7. 过门后才补3 seeds与MCG/EPS；
8. 最后整理 finite-budget gain 与 source-free hard-progress 主表。
