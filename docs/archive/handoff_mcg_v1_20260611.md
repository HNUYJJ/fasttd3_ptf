# MCG v1 执行报告与求诊（发 ChatGPT-5.5-Pro，2026-06-11）

你上次建议的 "Modular Critic-Guided Policy Transfer" 方向已被采纳并完成第一轮实现与三任务 pilot。以下是完整执行情况、实验结果与我们的诊断，请你分析并给出下一步方向建议。

## 一、采纳前的三条实证修正（你没见过的本仓库历史数据）

在你的方案落地前，我们注入了三条此前实验得出的修正：

1. **行为层执行是一等公民**：此前在 package（h1hand）上做过 distill-only PTF（你方案 3.3 的"teacher 只作 regularizer"形态），0% 成功——replay buffer 里永远没有教师可达的状态，蒸馏只在站桩分布上匹配单步动作。结论：教师必须真正控制环境（off-policy 合法）。
2. **critic gating 在 0% 成功任务上有鸡生蛋问题**：学习型 option-value U(s,o) 需要成功样本，成功需要正确调度。因此 MCG 不学 Q_o，调度直接从 target critic 读出（U(s,o) ≈ Q(s, ã_o(s))）。
3. **funneling**：package 上 oracle 完美调度的纯教师串联也 0%（approach 终态 ∉ push 初态分布），技能交接状态失配是 zero-shot 拼接的根本死因（四层实验证伪：单源/学习调度/oracle 调度/子目标接口）。

## 二、已实现的机制（MCG v1）

把 PTF 的 option 从"整条教师"推广为 (教师 i, 身体组 g)，g ∈ {legs_torso(11维), arms(10), hands(40)}，共 61 维动作。

**Gating**：masked candidate ã_{i,g} = 学生动作中第 g 组替换为教师 i 该组动作；Δ_{i,g}(s) = Qmin(s, ã_{i,g}) − Qmin(s, a_θ)（双 distributional head 取 min 的均值 Q）；每组取 argmax 教师，gate = 1[Δ > margin]，margin=0。未用你建议的 0.1 分位数——离线探针实测 q10 比较几乎永不放行（frac+ ≤ 0.19）。

**行为层**：per-env per-group 锁存执行（temporal smoothing，对应 PTF 的 termination 语义）：锁存到期时若 gate 放行且 Bernoulli(exec_prob=0.3) 命中则该组切到最优教师，锁 min_steps=10 步；done 全组回学生。**Warmup（前 15k 步）**：critic 不可信（见探针），退化为无条件 bootstrap——每 env 以 0.5 概率随机抽教师**整动作**执行（整动作是历史教训：教师闭环依赖全身协调），锁 25 步；蒸馏关闭。

**蒸馏**：actor_loss = −Q + λ(t)·Σ_g 1[Δ_{i*,g}>0]·huber(π^g, a_{i*}^g)，λ 0.2→0 线性衰减 80k，batch 内子采样 8192 行做 gating。

**与原版 PTF 的差异**：option 粒度（整教师→教师×身体组）；option-value 来源（学习 Q_o→直接读 target critic）；β/termination（学习→固定锁存）；λ 权重（全局时间→状态×部位 gated）。

## 三、机制前提的离线验证（训练前做的探针）

用已训好的 push run 各阶段 checkpoint（5k/25k/50k/100k，每点 4800 状态）离线算 Δ_{i,g}：

- **part≫full 稳定成立**（25k 起）：reach 教师 arms 组 Δ≈0/frac+=0.49，同教师整条替换 Δ=−6.9/frac+=0.05；各教师 full 替换几乎处处最差——"局部有用整体有害"有 critic 级证据；
- **5k 的 critic 不可信**：Q_student 虚高（267），所有教师全拒——恰在教师最有用时说不要（warmup 设计依据）；
- **gating 方向响应正确**：50k 学生退化期（Q=−161）教师全面翻正（frac+ 至 0.70）。

## 四、实验与结果（全部 1 seed、100k steps、128 envs、batch 32768、no-compile、paired scratch 对照——参数 diff 验证仅 exp_name 不同）

| 任务 | 教师库 | MCG 末段 eval | scratch 末段 | 参照 |
|---|---|---|---|---|
| push | reach+stand+walk（obs 适配可行） | ≈ +6 | +16（历史 3-seed mean，单 seed 方差 >±100） | 原版 PTF 3-seed mean −9 |
| door | stand+walk+run（仅 proprio 可适配） | 313 | 328 | **旧原版 PTF（同库，纯蒸馏）：320-344** |
| window | stand+walk+run | **336** | **489** | — |

**push**：持平。机制指标全自洽（warmup 执行率 0.48≈设定，gated 期 part 执行率 0.14=gate 0.5×exec 0.3，arms gate 全程最高与探针一致，eval 单调 −327→+6）。定性：scratch 无瓶颈，教师无增量。

**door**：持平但**低于旧原版 PTF**。MCG 的 gate_rate 单调衰减 0.53→0.11、执行率 0.46→0.05、Δ_best 转负——critic 判定教师无用后 MCG 主动退出迁移；而旧 PTF 的恒定 β 加权蒸馏全程保持，拿到了后期 +80~100 的稳定收益。**阶段漏斗评估**（approach→handle→open→passage，全从 obs 判定）：MCG 与 scratch 完全相同——P1 走近 100%、P2 转把手 100%、**P3 开门 0%**、P4 0%。瓶颈是"转住把手同时推门"的探索瓶颈，教师库无对应技能，return 330 都是 stand+approach+hatch 项堆出来的（success bar 600 无人接近）。

**window**：**净负迁移 −153**。诊断：(a) warmup 固定税——15k×50% 部位-时间被无关 loco 教师占据，学生自采数据近乎减半，起跑被砍（<15k 段 eval 8 vs 11，15-30k 段 28 vs 66）；(b) **margin=0 噪声蒸馏**——Δ_best 全程为负（critic 一直说教师不好）但 gate_rate 仍 0.2-0.3：per-sample Δ 噪声的右尾持续假阳性放行，把 actor 不断拉向无关教师。

## 五、我们的诊断（三层）

1. **教师-任务错配三形态**：push=任务无瓶颈教师无增量；door=瓶颈（P3 开门协调）存在但教师库覆盖不到；window=教师无关+机制缺陷放大为实际伤害。共同根源：loco 教师能教的（站稳/走近）FastTD3 scratch 在 10k 内自学完成（door 10k 时 scratch P1 已 94%）——**教师增量知识 < scratch 自学速度时，迁移没有对价，而行为预算的机会成本是实打实的**。
2. **机制缺陷（可修）**：warmup 是无条件固定税；margin=0 的 gate 不具备负迁移免疫力。且 gate 双向失灵：window 上因噪声错放（伤害），door 上因保守错关（错过旧 PTF 的后期稳定收益）。
3. **机制级发现（候选论文贡献）**：critic-gated transfer 的负迁移免疫性取决于 **Δ 判定的显著性（信噪比）而非符号**——离线探针中 Δ 量级仅 ±0.2-0.5，在线训练 critic 的估计噪声同量级或更大。这是对 CUP 一类 critic-guided 方法安全性声明的普适批评，我们有干净的失败实证（window）。

## 六、补充事实（影响方案设计）

- **obs 布局**：HB 的 manipulation 任务（reach/push/package）重写了 get_obs，显式给手部/目标笛卡尔坐标字段；door/window 用默认 get_obs（裸 qpos+qvel），无任何笛卡尔字段 → reach 教师接不进（已设计 ObservationWrapper 方案 h1hand-door-ext-v0 补字段，未实施；但 door 漏斗显示 reach 只能帮已经 100% 的 P1/P2，预期收益被削弱）。
- **可用源**：stand/walk/run/reach（自任务 solved）+ push_s1（push 上 59-63% 成功）+ approach v2（package 专训的接近技能，aux-reward 课程产物）。package 探针显示 push/reach 教师近身成功率 15-22%（教师与 package 瓶颈相关）。
- 硬约束：必须基于 PTF 框架 + FastTD3 算法 + HumanoidBench，创新长在 PTF 内；FastTD3 官方代码不可改（外挂式集成）；目标 ICML 水平。

## 七、我们的候选下一步（请你评判/排序/补充）

1. **v1.1 机制修复**：gate 改显著性判定（margin>0 或 Δ 的滑动统计/双 head 一致性）；加**保底蒸馏下限**（不让 gate 全关，吸收旧 PTF 恒定蒸馏的后期收益）；warmup 砍到 5k 或改需求驱动（学生进步停滞才触发）。
2. **window 重跑 v1.1** = 论文安全性实验（修复后预期 MCG≈scratch，证明 negative-transfer immunity——现在有真实失败基线做对照）。
3. **主攻任务换 package**：教师库与瓶颈匹配的唯一任务（approach/push 教师，近身 15-22%），MCG 的 warmup bootstrap + per-part 执行 + critic 调度恰好对症此前三个死结（不执行/鸡生蛋/整身切换）；但 funneling 风险仍在，且 0% 成功任务上 critic gating 的鸡生蛋只被 warmup 部分缓解。
4. door 的真瓶颈（P3 开门）需要专项技能源（near-door reset + dense opening reward 的课程式 aux 源）——投入较大，排后。

**最想请你回答的问题**：(a) 同意我们的三层诊断吗？(b) gate 显著性的具体设计（如何在线估计 Δ 的噪声尺度？分位数/bootstrap/双 head 分歧？）；(c) 行为预算（warmup/执行概率）的原则性设计，避免固定税；(d) 任务×教师库矩阵下一步怎么排——package 主攻是否正确？(e) 论文叙事：在"modular transfer"主贡献外，"critic-gated transfer 的安全性条件（显著性而非符号）"能否立为第二贡献？(f) 单 seed 方差 >±100 的现实下，pilot 阶段的实验设计该怎么改？
