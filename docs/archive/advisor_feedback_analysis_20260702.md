# 导师意见的批判性分析与统一方案（2026-07-02）

PI 于 2026-07-01 与导师进行科研进展交流，导师给出三条意见。本文档：逐条事实核查 +
批判性分析（含不同意的部分）→ 提出统一框架（在线修正的 transferability 度量 T）→
给出可计算公式（回应导师"不依赖实际交互"的补充要求）→ 执行计划 → 给 ChatGPT 的问题清单。

> **2026-07-11 状态更新（请审计者先读）**：§0–§7 保留 2026-07-02 至 07-03 的
> chronological research log、阶段假设与失败尝试，其中若干强因果解释已被后续 hard-task、
> 单源和剂量匹配审计否定或收窄。**当前证据边界、框架重构与请求 ChatGPT-5.6-Pro 裁定的
> 战略问题以 §8–§14 为准；若与旧文冲突，以新章节为准。**

> **2026-07-11 后续效度修正**：复核发现旧 stability audit 的
> `env.unwrapped.seed(seed)` 未播种 HumanoidBench reset 实际使用的 Gymnasium
> `np_random`，所以旧 P0/P1/P2 的条件均值与跨训练种子方向只能作描述性证据，不能再称为
> 精确同初态 episode counterfactual；cabinet 跌倒也不会提前 termination，等长 episode
> 不能单独排除姿态解释。当前正式的中心命题、两通道 estimands、完整 anchor 要求和 2×2
> 预注册协议见
> [`source_intervention_mechanism_gate_v1.md`](source_intervention_mechanism_gate_v1.md)。

> **2026-07-11 机制门已完成**：§8–§14 中的 source-intervention/DV 路线已经按正式
> 预注册执行并得到 `Engineering Go=true, Feasibility Go=false,
> STOP_COMPLEX_ESTIMATOR`。最终结果与后续禁止事项见本文 §15；若与 §13–§14 的待执行计划
> 冲突，以 §15 和机制门主文档第 14 节为准。

> **2026-07-11 论文核心重构已启动**：机制门之后的新中心 thesis、贡献裁剪、相关工作边界与
> 最小升级实验见本文 §16 及
> [`paper_core_contribution_reconstruction_v1.md`](paper_core_contribution_reconstruction_v1.md)。

---

## 0. 导师三条意见摘要

1. **replay 采样时序**：FastTD3 是 off-policy。当前每次选一个专家与目标环境交互 horizon 步、
   轨迹注入 replay buffer；切换专家后，上一阶段专家的轨迹仍驻留 buffer 并可能被采样，而这些
   轨迹对当前阶段是次优的。建议设计采样方式，尽量采样当前所选最优专家注入的轨迹。
2. **不加 null_option，直接加目标任务自己的策略**：目标任务自己的策略就是它自己在做 RL 探索。
   加 null_option 的目的本来就是防负迁移（无合适源时让学生自探索、至少不损害学习），
   直接把学生策略加入候选即可。
3. **提出可计算/度量的 transferability 方法或公式**（核心意见）：现有几点贡献不够强。
   要提出一个度量"源策略→目标任务可迁移性"的方法/公式，不必理论证明，但按它选出的源的
   实际迁移效果要比别人的方法好；做成半自动甚至全自动。
   **补充要求（2026-07-02）**：当前我们判断可迁移性是"把源策略放进目标环境实际交互算回报"；
   导师希望**不必每次选源都实际交互**——要有一个零交互或半交互即可计算的指标。
   参考文献：Sinapov et al. 2015、APT-RL、PMIC。

---

## 1. 事实核查（代码 + 文献）

### 1.1 代码层

- **意见1 的基础设施已存在**：`fasttd3_ptf/official_fasttd3_ptf/ptf_replay.py` 的
  `PTFReplayWrapper` **已为每条 transition 记录 option_ids**（−1=学生，≥0=教师索引），
  `sample()` 当前是 uniform `torch.randint`。改"按源加权采样"是我们自己文件的局部改动，
  不碰官方 FastTD3 代码。
- **意见2 的落点清晰**：`fasttd3_ptf/ptf/mcg.py` 的 `safe_bootstrap` 分支
  （line 291-307）：`use_teacher = rand < warmup_exec_prob(=0.5 固定)`，softmax 只在源上、
  无学生项、权重静态。"不选教师=学生行动（current=−1）"本来就是机制的一部分——null_option
  从来不是额外网络，导师方案的实质增量是**把学生变成平等竞争的 arm + 权重在线更新**。
- **意见3 的雏形已存在**：TransferMap v2 的 vs-zero probe 就是一个（交互式的）静态
  transferability metric，3-seed 已证其有效（weighted>uniform, wfix−rand +77.9, t=+3.08）。
  且我们已拿到它的**失败模式**：crawl 上 stand/walk vs-zero 分数高（13.9/13.7）但真实迁移
  为负——静态交互式 probe 在 stabilization 类任务上误判。
- **半交互度量的算子已存在**：train_ptf.py 的 `qheads_value` 与 `mcg_gating.deltas`
  （warmup 后 critic gate 在用）就是"学生 critic 评估源动作"的现成实现。

### 1.2 文献层（已查证）

| 文献 | 内容 | 与我们的关系 |
|------|------|------------|
| [Sinapov et al. AAMAS 2015](https://www.cs.utexas.edu/~sanmit/papers/AAMAS15-sinapov.pdf) | 用任务 meta-feature 学 transferability 预测器，无目标样本 | 立论"transferability 可学习"；但需参数化任务族+历史迁移数据训练，HB 任务异构、无此元数据集，不直接适用 |
| [APT-RL, arXiv 2311.06731](https://arxiv.org/abs/2311.06731) | advantage 作为正则权衡源知识 vs 新知识；提出 transfer performance measure + 任务相似度算法 | 最接近的框架级工作；单源、fixed-domain、小任务。可作 baseline 对比候选 |
| [Li & Zhang 2017, arXiv 1709.08201](https://arxiv.org/pdf/1709.08201) | 在线源策略选择形式化为 MAB，有最优性理论 | student-as-arm 就是 MAB 视角，必引；他们直接 policy reuse，我们是 experience injection + off-policy 蒸馏 |
| PMIC (ICML 2022) | MARL 互信息协作 | 关系较远：MI 估计成本高、不直接给"源→目标"分数。只借"用信息量度量契合度"的精神，**建议诚实向导师说明不直接搬** |
| [LEEP (ICML 2020)](http://proceedings.mlr.press/v119/nguyen20b/nguyen20b.pdf) / LogME / H-score | 监督学习的 transferability estimation 成熟谱系（无需微调即可打分） | RL 缺对应物——**正是导师指的空位**，我们的 critic-based 公式就是 RL 版对应物 |
| [Adaptive Replay Buffer, arXiv 2512.10510](https://arxiv.org/abs/2512.10510) | offline-to-online 按"on-policyness"动态加权采样 | 意见1 的近邻工作；我们按 per-source transferability 加权是新变体 |

---

## 2. 逐条批判性分析

### 2.1 意见1（replay 采样时序）：担忧部分成立，但优先级应放第二

**不同意的部分**：off-policy 的理论本性就是能从任意行为策略的数据学习。critic 学 Q(s,a)，
旧专家"次优轨迹"只要 reward 标注正确，就是**有用的负样本**（critic 借它学到"这些动作差"），
不必然有害。"buffer 里有旧专家轨迹=坏"这个判断在 off-policy 语境下不精确。

**同意的部分**：真正风险是 (a) 低质量源轨迹**稀释** buffer、降低有效样本频率；
(b) warmup 注入的坏轨迹在 warmup 结束后**长期驻留**。crawl 数据留有疑点：wfix
（最优源+短 horizon）仍比 scr 低 73——可解释为"所有源都不合适"（选择问题）或
"坏轨迹驻留毒害"（采样问题）。

**处置**：两个假设用一个实验区分——先做意见2（在线注入控制，从源头减少坏轨迹量）：
若 crawl 恢复 ≥scr，驻留问题已被解决大半，意见1 降为消融；若仍 <scr，再上 replay
重加权（基础设施已就绪，改 `PTFReplayWrapper.sample` 按源权重加权）。证据链干净，
符合"稳扎稳打一项一项"。

### 2.2 意见2（student-as-arm 替代 null_option）：完全同意，比 LCB abstain 更简洁

- 与 ChatGPT 的 abstain 设计语义等价，但形式更简洁：不需要特殊 abstain 逻辑，学生就是
  (S+1) 维 softmax 中的普通 arm；当学生的估计价值高于所有教师时，概率质量自然转向学生。
- **实质增量 = 权重在线更新**（现状是静态先验 + 固定 0.5 学生份额）。学生在进步，其 arm
  价值必须随训练上涨；坏源的 arm 价值被在线暴露。crawl 里站立教师在线 return 差 →
  student share 自动上升 → **负迁移自动关闭**。
- 正是 3-seed 数据指出的修复路径：crawl 源选择项 +39.6（加权方向没错）但被过量注入拖成
  总增益 −43.3——在线竞争保住 +39.6、关掉有害注入。
- 机制对学生/教师完全对称（都是"执行期 per-step reward 的 EMA"），无任务名分支，
  满足通用性硬约束。LCB 版可留作保守变体消融。

### 2.3 意见3（transferability 度量）：方向正确，我们已做了一半，公式见 §3

- 已有：交互式静态度量（vs-zero probe）+ 它的有效性证据（t=+3.08）+ 它的失败模式
  （crawl 误判）。失败模式正是"必须在线修正"的 motivation，是论文最硬的 insight 之一。
- 缺口（导师补充点明）：现度量**每次都要源策略进目标环境 rollout**。要一个零交互/半交互
  可计算的指标——见 §3 的 critic-based 公式。
- 对"贡献不够强"的回应：三条意见 + 已有 RBO-PTF 可统一为一个更聚焦的主张
  （"一个 transferability 度量 + 它在训练闭环中的三处使用"），见 §4。

---

## 3. 可计算的 transferability 度量 T（回应"不依赖实际交互"）

### 3.1 三级信号

**T⁰（离线先验，现有，交互式）**：vs-zero rollout probe。导师指出的问题所在——
每个新源都要进目标环境跑。降级为**可选初始化/离线 sanity check**。

**T^online（在线 EMA，Step A）**：arm i 执行期间的 per-step reward 的 EMA。
注意这也是"交互"，但它是**训练本来就产生的交互**（无额外 probe 成本），
且只对被选中的 arm 更新。

**T^critic（半交互核心公式，Step C，回应导师）**：

```
T_i^critic(t) = E_{s~B_t}[ min_{j=1,2} Q_j^stu(s, π_i(s)) − min_j Q_j^stu(s, π_stu(s)) ]
```

- `B_t` = 学生当前 replay buffer 的状态分布（目标任务数据，**源策略不控制环境**，
  只在这些状态上做 batch 前向传播，毫秒级）。
- 语义 = "源策略动作在学生 critic 下的期望 advantage"——APT-RL 的 advantage 概念
  × 我们已有的 MCG critic-gating Δ（`mcg_gating.deltas` 就是它的 per-group 版）。
- min double-Q 抑制过估计（TD3 本性）；跨 arm 做 z-score 归一化再 softmax（reward
  尺度不变，避免每任务调温度=变相任务分支）。
- **关键卖点**：对从未 probe 过的新源也能即时计算 → 源库扩张全自动化
  （放一个 ckpt 文件即可，无需人工 probe/标权重）——直接支撑导师"半自动/全自动"要求。

### 3.2 置信度融合（诚实处理已知失败模式）

我们自己的探针证据（2026-06-11）：**早期 critic 不可信**（q10 太保守、初期 Δ 噪声大）。
所以 T^critic 不能从第 0 步就接管：

```
T_i(t) = (1−c(t)) · [ w_p(t)·T_i⁰ + (1−w_p(t))·T_i^online ] + c(t) · T_i^critic
```

- `w_p(t)`：先验→在线的转移权重，warmup 前段从 1 线性衰减到 0（Step A 实现这半段）。
- `c(t)`：critic 置信度 ∈ [0,1]，随训练递增（候选：step-based ramp / TD-error EMA 的
  递减函数）。Step C 接入。
- 研究叙事完整：**MCG（6 月初的 critic-gating"失败"路线）以度量器身份复活**，与
  RBO（reward weighting）统一为"同一个 T 的两个估计器，按 critic 成熟度融合"。

### 3.3 关于"完全零交互"的诚实边界

完全零交互（目标任务一条数据都没有，只看任务描述/源 ckpt）在异构 HB 任务上信息量有限：
Sinapov 2015 需要参数化任务族+历史迁移数据训练预测器，我们没有这种元数据集。
T^critic 的定位是**半交互**：用学生自己反正要收集的数据 + 源策略前向传播，
**不需要源策略与环境交互**。这是在导师要求与可行性之间诚实的落点，
也与 LEEP/LogME 在监督学习中的定位同构（用目标数据前向传播打分，不微调）。

---

## 4. 统一框架：一个 T，三处使用

| 用处 | 对应意见 | 状态 |
|------|---------|------|
| ① 源选择：softmax(T) 抽注入源 | 现有 RBO-PTF | 已验证（wfix−rand +77.9, 11/12, t=+3.08）|
| ② student-as-arm：T 含学生 arm，负迁移自动关闭 | 意见2 | **Step A（本轮开工）** |
| ③ replay 重加权：驻留轨迹按当前 T 加权采样 | 意见1 | Step B（条件触发，基础设施已就绪）|

外加：全自动 pipeline（源 ckpt 集合 + 目标 env id → T^critic 自动评估 → 自动生成 bank →
训练），`build_source_bank.sh` 已是雏形。

**论文贡献重构（回应"贡献不够强"）**：
① 半交互 transferability 度量 T（公式 + 置信度融合 + 静态 probe 失败模式作 motivation）；
② T 驱动的统一训练闭环（选源 / student-as-arm / replay 重加权）on FastTD3+HB；
③ HB 广度验证 + 与 uniform/静态probe/APT-RL 式 advantage/MAB-UCB 的度量对比表。

---

## 5. 执行计划

### Step A（本轮开工）：`online_bootstrap` = student-as-arm + 在线 EMA

改动（全部在可改文件内）：
1. `fasttd3_ptf/ptf/mcg.py`：新增 `warmup_mode="online_bootstrap"`
   - 状态：`arm_value[S+1]`（index S = 学生 arm）+ `arm_count[S+1]`，count-based EMA
     （`v += (r−v)/min(count, N_ema)`，早期快收敛后期稳定）。
   - 每步结算：新方法 `update_arm_reward(rewards)`——`self.current` 正好记录本步各 env
     执行的 arm，reward 归属天然对齐（在 `envs.step` 之后调用）。
   - 选择（expired env 上）：以 `w_p(t)` 概率走**先验分支**（=现行 safe_bootstrap 行为：
     bernoulli(0.5) 学生 + softmax(bank 权重)），否则走**在线分支**：
     `p = (1−ε)·softmax(zscore(arm_value)/τ) + ε·uniform`，arms 含学生。
     `w_p(t)` 由内部步数计数器驱动，线性衰减（默认 warmup 前 30%）。
   - ε-floor 保证每个 arm 持续被探索（EMA 不死锁在先验）。
   - horizon 与 wfix 对齐：源 arm 用 bank horizon（=25），学生 arm 用 warmup_min_steps（=25）。
2. `train_ptf.py`：`envs.step` 后回传 `mcg_behavior.update_arm_reward(rewards)`
   （warmup 期 + online 模式时）；wandb log per-arm value / student share / w_p。
3. 配置：`mcg_warmup_mode="online_bootstrap"` + 4 个新键
   （`mcg_online_tau / mcg_online_eps / mcg_online_prior_frac / mcg_online_ema_n`）。

**验证协议（一项一项）**：
- 冒烟：crawl 短跑，确认 arm_value 更新、student share 随站立教师的低 reward 上升。
- 方向验证：crawl（负迁移 case，目标 **≥ scr**，即负迁移关闭）+ pole（正迁移 case，
  目标 **≈ wfix**，即不损害正迁移），各 1 seed。
- 通过后：4 任务 × 3 seed 正式对比（vs wfix/scr）。

**Step A 结果（2026-07-02，STAMP `20260702T025544Z`，共同窗口 95k AUC）**：

| task | scr(3seed) | rand(3seed) | wfix(3seed) | safe(3seed) | **onlineb(s1)** |
|------|-----------|-------------|-------------|-------------|-----------------|
| crawl | **812.0±25** | 699.6±32 | 739.2±6 | 656.3±35 | **726.0** |
| pole | 603.3±48 | 573.2±13 | **767.4±24** | 717.9±25 | **763.9** |

- **pole：PASS（不误伤）**。onlineb 763.9 vs wfix s1 794.6（−4%），落在 wfix 3-seed 带内
  （735–795），高于 safe 全带。arm 过程合理：walk 值最高（0.75）≈student（0.74），
  student share 稳在 0.53——教师有用时机制保持大量注入。
- **crawl：未达标（onlineb 726 vs scr 833，gap 107），但机制本身工作正常**——
  arm value 学到正确排序（student 0.455 > run 0.435 > walk 0.431 > stand 0.412），
  student share 从 50%→**76%**。onlineb≈wfix（726 vs 746，seed 噪声内）。
- **关键诊断（本轮最有价值的发现）**：把学生执行份额从 50% 提到 76% **并没有改善**
  crawl return → 负迁移的主要伤害**不在"执行坏教师的步数比例"，而在已注入 buffer 的
  坏轨迹被持续采样学习（残留毒害）**。这正是导师意见1指向的机制——
  **Step B（replay 按 T 降权）由"条件触发"升级为"必做"，且证据链干净**。
- **次级发现**：crawl 上各 arm 的 per-step reward EMA 挤在一起（0.41–0.46，分离度仅
  5–10%）→ 即时 reward 信号弱（站立教师执行期机器人未必立刻摔，伤害体现在把学生带进
  坏状态+坏数据，credit 延迟）→ 支持 §3 的论点：T^online 不够，需要 T^critic
  （半交互 advantage）的 c(t) 融合来提供更可辨识的信号。

### Step B（升级为必做）：replay 按 T 重加权

Step A 已证明"减少坏教师执行"不足以修复 crawl，残留 buffer 毒害是主因。改
`PTFReplayWrapper.sample`：per-transition 按其 option_id 的当前 T 加权采样
（低 T 源的驻留轨迹自动降权，学生轨迹权重恒 1）。crawl 是干净的检验场：
若 replay 降权后 crawl→scr 水平，同时证明 (a) 残留毒害假设 (b) 意见1 的价值
(c) T 闭环的第三处使用。

**Step B 实现（2026-07-02）**：`ptf_replay.py` 加权 sample（学生恒 1；源按
`exp((T_i−T_stu)/(std·τ))` 只降不升，floor=0.1 保留负样本价值；权重每 100 步从
arm_value 刷新，warmup 后冻结持续降权残留轨迹）。buffer 的 options 字段改记真实
执行 arm（MCG 模式下无其他消费方，安全）。单元自测 PASS（降权 0.1→采样占比
0.33→0.05 精确）；microbenchmark 加权路径 0.5ms/次（<2% 开销；冒烟 sps 下降经
同卡对照排除，为共享节点 CPU 高负载 load>60 所致，代码无罪）。

**Step B 结果（STAMP `20260702T045531Z`，95k AUC，seed1）**：

| task | onlineb | **obrw(=onlineb+replay降权)** | Step B 增量 | 参照 |
|------|---------|------------------------------|------------|------|
| crawl | 726.0 | **767.5** | **+41.6** | scr 832.8 / wfix 746.3 |
| pole | 763.9 | 733.2 | −30.8 | wfix 794.6（−8%，阈内）|

- **残留毒害假设证实（部分）**：crawl +41.6，gap 收窄 39%（106.8→65.3）。
  **obrw 已是 crawl 上所有迁移方法的最好成绩**（>wfix 746 > safe 689）。
  意见1（replay 采样设计）的价值坐实——一次实验同时支持 (a)(b)(c)。
- **诚实记录代价**：pole −30.8（vs wfix −8%，仍在不误伤阈内但非零）。机制解释：
  pole 上教师普遍有用（walk 0.748≈student 0.742），但 stand(0.548)/run(0.622)
  相对劣势被降权到 0.11/0.26——**在教师有用的任务上降权次优教师轨迹损失了
  有用数据**。改进方向（下轮迭代）：降权自适应激活——仅当所有源 < 学生
  （crawl 型全源劣势）时激活降权，有源 ≥ 学生（pole 型）保持 uniform。
- **crawl 残余 gap 65.3 的归因候选**：(i) floor=0.1 仍在采样坏轨迹；(ii) 24% 步数
  仍执行教师（ε 下限+softmax 尾部）；(iii) warmup 前 9k 先验注入。收口实验 =
  "硬 abstain"变体：在线判定全源劣势后 exec→0 + 权重→floor 最小，预期 ≈scr。

### Step C：T^critic 接入 + 度量对比表

- warmup 期把 `mcg_gating.deltas` 聚合为 T^critic，按 c(t) 融合进选择分布。
- 度量对比实验（导师要的"比别人好"证据表）：同一训练协议下比较
  uniform / 静态 vs-zero / T（在线融合版）/ APT-RL 式 advantage-only。
- 自动化 pipeline 收口。

**Step C 离线 logging 结果（2026-07-02，按 ChatGPT 第二优先级先离线验证，
`scripts/analyze_tcritic_offline.py`，onlineb s1 中间 ckpt 10k→100k）——重要否定性发现**：

| 时序表 | crawl（期望全源负→abstain）| pole（期望 walk≥0→transfer）|
|--------|---------------------------|------------------------------|
| T_mean 实测 | 全源 −0.2~−0.7 全程 ✓方向 | **walk 也全程负**（−0.2~−0.8）✗**误判** |
| 排序 | 三源无分化（三源确实都没用，合理）| walk > run/stand 排序 ✓（@15k：−0.3 vs −2.8/−2.1）|

1. **T^critic 的符号不可用作 transfer/abstain 判据**。根因是系统性负偏，非噪声：
   学生 actor 本来就在最大化学生 critic 的 Q（policy improvement），任何 off-policy
   单步动作在成熟 critic 眼里都 ≈略差——所以 pole 上真实有用的 walk 也被判负
   （幅度 <1%·Q_stu 的小负数）。若用它 gate，pole 会被错杀成 abstain。
2. **T^critic 的源间相对排序有信息**（pole 上 walk 最高，正确）→ 角色重定位：
   **降级为选源排序辅助**（softmax 相对权重），**符号/abstain 判据由 T^online
   （执行期 reward EMA）承担**——后者在 crawl/pole 上双向正确（已验证）。
3. **这个否定性结果反而补全了度量故事**：T^0（静态 probe）误判 crawl；
   T^critic（单步 advantage）误判 pole；只有 T^online（执行期 reward-bearing）
   双向正确——因为源的价值在"执行一段 horizon 的实际后果+注入的数据"，
   不在"单步动作的 Q 比较"。**这正是 Reward-Bearing 主线必要性的直接证据**，
   也是三信号失败模式对比表（贡献1）的实验支柱。
4. 待 ChatGPT 复核的改进方向：horizon-aware 的 critic 评估（如 target-critic
   n-step / 执行 h 步后状态的 V）能否修复负偏——成本与收益需权衡。

---

## 6. 给 ChatGPT 的问题清单

1. **T^critic 公式**（学生 critic 下源动作的期望 advantage，min-double-Q，跨 arm z-score）
   作为"半交互 transferability"定义是否合理？有无更稳健的形式
   （percentile/CVaR 代替均值？uncertainty-aware 版本？）
2. **c(t) 置信度调度**怎么定：step-based ramp 还是 TD-error EMA 驱动？我们的探针证据是
   早期 critic 不可信、q10 分位太保守。
3. **student arm 初始化与探索下限**：学生 arm 无先验（bank 权重是 vs-zero 尺度、在线 EMA
   是 per-step reward 尺度，不可直接混用）。我们的方案是"先验分支/在线分支按 w_p(t) 混合"
   绕开尺度问题——是否有更好做法？ε-floor 取多少（0.1?）？
4. **Step A 验证协议**：crawl（≥scr = 负迁移关闭）+ pole（≈wfix = 不损害）先行，
   通过后 4×3seed——判据是否合理？
5. **意见1 的处置**：replay 重加权作为条件触发（Step A 不达标才上）vs 直接并入主方法做消融，
   哪个更好？off-policy 下"坏轨迹是有用负样本"的反驳是否成立？
6. **论文故事重构**：从"diagnosis + RBO-PTF + broad eval"改为"一个 transferability 度量 +
   三处使用"，哪个对 ICML 更强？MCG(critic) 以度量器身份复活的叙事是否可信？
7. **度量对比的 baseline**：APT-RL 原实现是单源 SAC-based，移植成本高——用
   "APT-RL 式 advantage-only 变体（去掉在线融合）"作为对比是否公允？
   还应加 MAB-UCB（Li & Zhang 2017 式）吗？
8. **fall-avoidance 结论**（上轮遗留）：接受"HB 上 robustness=fall-avoidance
   （safe 摔倒终止率 0–3% 最低）"作 behavioral insight，"跌倒后恢复"留 future work？
9. **（Step A/B 结果后新增）下轮迭代优先级**：硬 abstain 收口（全源劣势 exec→0，
   归因 crawl 残余 gap 65，预期 ≈scr）vs 降权自适应激活（仅全源劣势时激活 replay
   降权，修 pole −31 且不伤 crawl）——两个都做（各 1 run 便宜）还是有更优先的？
10. **pole −31 的呈现**：replay 降权在"教师有用"任务上有真实代价（次优但有用的
    stand/run 轨迹被降到 0.11/0.26）。论文里作为 trade-off 诚实呈现 + 用自适应激活
    修复，还是把降权强度 τ 调温和（弱化差异）更好？

---

## 6.5 ChatGPT-5.5-Pro 裁定（2026-07-02，PI 转达）

方向判定：**"方向正在变强而不是变散"**。论文主线正式升级为
**T-RBO-PTF**（Transferability-Calibrated Reward-Bearing Option Bootstrap）——
一个在线修正的 T，三处闭环使用（选源执行 / student-as-arm abstention / T-weighted replay）。

**关键裁定**（对 §6 十个问题的回答，全文见 PI 转达记录）：

1. **第一优先级 = T-gated transfer/abstain 模式切换**（把"硬 abstain"与"降权自适应
   激活"**合并为一个机制**，不要拆开，也不要先调温度 τ）：
   - **Transfer mode**（存在源 T_i ≥ T_stu − δ）：源执行按 T 加权；**replay 保持
     uniform 不强降权**（修 pole −31）。
   - **Abstain mode**（全源 T_i < T_stu − δ 持续 K 个窗口）：teacher exec→~0.02
     probe floor；源 replay 权重降到 floor；student 权重 1（修 crawl 残余 gap）。
   - 验证 crawl+pole 各 1 seed。成功标准：crawl 接近 scratch 且显著 > obrw(767)；
     pole ≥ 0.95×wfix(≈755)；crawl 后期 exec share→0、replay 源采样占比显著降。
2. T^critic 公式合理，但稳健化：winsorized mean 或 0.5·mean+0.5·q25（**不要 q10**，
   已实测过保守）+ head-disagreement penalty（λ_q·Q75(|Q1−Q2|)）；z-score 用
   **median/IQR** 而非 mean/std（3-8 个 arm 时更稳）。
3. c(t)：第一版 **step ramp**（t0=0.3·warmup, t1=0.8·warmup，端值 0.5-0.7 不到 1）
   + critic health veto（TD-error EMA、head disagreement）作为后续加法，先别复杂化。
4. **T^critic 先离线 logging 不控制训练**（第二优先级）：在已有 runs 上离线算
   T^0/T^online/T^critic 排名与实际 per-arm return 的一致性，crawl 上若能更早分辨
   student>walk/run 再接入控制。
5. 三分支融合公式：p = (1−c)·[w_p·p_0 + (1−w_p)·p_online] + c·p_critic，
   各分支内部 robust z-score，不跨尺度直接相加（认可现方案）。
6. 探索下限：source ε=0.05 起；abstain mode 降到 0.01-0.02;student 无需 floor。
7. baseline 措辞："advantage-only transferability score **inspired by** APT-RL"
   （不 claim 复现）；另加 MAB-EMA/UCB 与 static-vs-zero-only 两个便宜 baseline；
   给出"额外交互/student-arm/critic/replay-aware"四列对比表。
8. pole −31：若 adaptive activation 修复则作为开发过程 diagnostic 不进主文；
   **最终主方法不允许停在 pole 明显低于 wfix 的版本**。
9. fall-avoidance：接受，措辞用 "fall avoidance / upright robustness"，不 claim
   fall recovery。
10. **优先级序列**：①T-gated abstain+adaptive replay（先修 crawl 不掉 pole）
    ②T^critic 离线 logging ③4 terrain×3seed 主表（scratch/uniform/wfix/onlineb/
    T-gated）④negctrl door/spoon（final method，验证自动 abstain）⑤扩源扩任务
    （**现在不要过早扩源**）。

**规则在已有数据上的预演**（δ=0.5·std(arm_values)）：
- crawl：std≈0.018，阈=0.455−0.009=0.446 > 全部源(0.412/0.431/0.435) → **正确判 abstain**
- pole：std≈0.083，阈=0.742−0.041=0.701 < walk(0.748) → **正确留 transfer**
- δ_frac=0.5 有数据支撑，作默认值。

### T-gated 验证结果（STAMP `20260702T070723Z`，95k AUC，seed1）——阈值脆弱性暴露

| task | onlineb | obrw | **tgated** | 判读 |
|------|---------|------|-----------|------|
| crawl | 726.0 | **767.5** | 727.1 | ✗ ≈onlineb，比 obrw 低 40.4 |
| pole | 763.9 | 733.2 | 749.5 | PASS（−6% vs wfix；比 obrw 好 16.3）|

**根因（wandb `mcg/abstain_mode` 曲线证实）：crawl 的 abstain 从未触发（全程 0）**。
- 预演用的是 onlineb run 收敛后的 arm 快照（gap 0.020 > δ 0.009）；但 tgated 自己的
  轨迹里 student−max_src 只有 **0.002~0.009**，恰好卡在 δ≈0.0095 之下——判定从未成立，
  tgated 全程退化为 onlineb（727 vs 726 完全吻合，pole 未触发也正确）。
- **机制级教训**：crawl 的 per-step reward 信号本来就弱（arm 分离度 <2%），任何依赖
  "分离度超过阈值"的**二值硬判定都脆弱**（run 间方差就能翻转）；obrw 的**连续 exp
  降权**不依赖阈值（任何小差距→按比例降权），所以稳健拿到 +41.6。
- **更深的缺口（带给 ChatGPT 的核心问题）**：T^online 读数上，pole 的"walk 略差于
  学生"（0.758 vs 0.783）与 crawl 的"全源略差于学生"（0.320 vs 0.329）**几乎无法
  区分**，但真实效果天差地别（pole 数据有用/crawl 数据有毒）。**执行期 reward ≠
  数据价值**——现有三信号（T^0/T^online/T^critic）都不度量"源轨迹作为 replay 数据
  的价值"，这是下一步的核心缺口。
- 两机制各赢一边的格局：obrw 赢 crawl（连续降权对弱信号稳健），onlineb/tgated 赢
  pole（不降权不误伤）。候选出路（待裁定，不自行连跑变体）：
  (a) δ/K 调松——阈值工程，已被警告；(b) 降权强度乘连续劣势系数 g=σ((T_stu−maxT)/s)
  ——仍是同一信号，区分不了两种"略差"；(c) 引入"数据价值"信号（如源轨迹上的
  TD-error / critic 拟合残差，PER 有文献根基）；(d) 务实收口：obrw 为主方法 +
  降权温和化（floor 0.1→0.3）缓解 pole。

---

## 6.6 ChatGPT 第二轮裁定（2026-07-02 晚，看完 T-gated 失败诊断后）

**核心升级：T 从标量升级为 profile (T^exec, T^data, T^rank)；下一步主修复 =
actor/critic split replay**（源数据对 critic 是有用负样本、对 actor 状态分布是污染）：

1. **停止硬 abstain 阈值路线**（tgated 已证二值判定在弱信号下脆弱）；一律改连续控制。
2. **replay 拆双路径**：critic 权重保守（floor 0.3-0.5，保留负样本价值）；actor 权重
   强降（floor 0.02-0.1，防状态分布污染）。理论依据：TD3 actor 在 replay 状态上最大化
   Q(s,π(s))，坏源状态大量进入 actor batch = 在学生永远不该去的状态上浪费优化。
3. **第一优先级实验**：crawl+pole × {onlineb已有 / obrw-both已有 / actor_only /
   critic_only / split} → 定位污染来自哪条路径。
4. T^critic 短期不接入控制（只作排序/baseline）；不做 horizon-aware critic（会失去
   半交互优势）。T^data 第一版不显式定义，split sampling 本身即隐式承认数据价值分流。
5. 不要先调 floor/τ 遮问题；先 split 归因，再有依据地调。
6. crawl 残余 gap 修复顺序：split replay →（仍有 gap 再）warmup calibration
   （前 2-5k 低预算 probe，源执行预算 ≤10-20%）→ 最后才调 floor。
7. pole −31 不进主文（开发诊断）；最终验收保持双条件：
   **crawl ≈ scratch 且 pole ≥ 0.95×wfix**，不满足不扩任务。
8. 论文三贡献定稿方向：①Transferability is operational, not static（三信号失败
   模式=实验支柱）②T-RBO-PTF（execution 加权 + student-as-arm + **非对称 replay**
   + critic 排序）③terrain 4 任务扎实 → door/spoon → 再扩。

**我的批判性审视（执行前明确两点）**：
- **变量控制改进（与 ChatGPT 建议略有不同）**：actor_only / critic_only 沿用与
  obrw **完全相同**的 exp 公式与 floor=0.1，只差作用路径——这样 both(0.1/0.1) vs
  actor_only(0.1/–) vs critic_only(–/0.1) 三者的差异**只能**归因于路径，归因链干净；
  split(actor 0.05 / critic 0.4) 作为基于归因结果的"设计版"。若直接按 ChatGPT 给的
  差异化 floor 跑 actor_only，floor 与路径两个变量会纠缠。
- **"critic 需要坏负样本、floor 要高"目前是假设非事实**——若 crawl 的 +41.6 主要来自
  critic 降权，split 的保守 critic floor(0.4) 反而会丢收益。critic_only 实验正是检验
  这一点；split 的最终 floor 应依据归因结果，必要时回调。

---

## 6.7 split replay 归因矩阵结果（2026-07-02 深夜，STAMP `20260702T124238Z`）

（首轮矩阵因 `data_pol` 漏 normalize_obs 全部作废并重跑；修复经 eval 点对比验证，
教训已入 memory。）**修复版终局（95k AUC，seed1）**：

| task | scr | wfix | onlineb | **both(obrw)** | aonly | conly | split(0.05/0.4) |
|------|-----|------|---------|---------------|-------|-------|-----------------|
| crawl | 832.8 | 746.3 | 726.0 | **767.5** | 676.2 | 629.6 | 715.9 |
| pole | 537.0 | 794.6 | 763.9 | 733.2 | **773.5** | **771.0** | 684.5 |

**核心发现：ChatGPT 的 split 假设被数据推翻——对称性主导，非对称降权全面失败**：

1. crawl 上**单路径降权比不降权更差**（aonly −49.8 / conly −96.4 vs onlineb），
   split（非对称强度）居中受伤（−10.1），**只有对称的 both 获益（+41.6）**。
2. 机制解释（覆盖全部数据点）：TD3 的 actor 梯度 ∇Q(s,π(s)) 依赖 critic 在采样
   状态附近的准确性——**单/非对称降权让 actor 与 critic 看到不同的数据分布**，
   AC 失配的伤害超过降权收益。crawl 降权量大（全源压到 floor）→ 失配大 → 灾难；
   both 对称 → 分布自洽 → 安全且受益。
3. pole 上 aonly/conly(+7~+10)≈onlineb 而 both(−31)/split(−79) 偏低——但这些差异
   （±30-80）与 pole 单 seed 噪声带（scr 3seed 537-651、wfix 735-795）重叠，
   **单 seed 不足以裁决 pole 上的方法排序**。
4. "critic 需要坏源负样本（floor 高）"假设未获支持：conly 在 crawl 是全场最差
   （629.6）——降权 critic 同时不动 actor 是最糟组合。

**结论与下一步**：
- 对称连续降权（both/obrw）是唯一在 crawl 上稳健获益的形态，**主方法候选 = obrw**；
- 当前最大不确定性不是机制而是 **seed 方差**（pole 的 −31 是否真实）——已启动
  seed 加固（onlineb/obrw × crawl/pole × s2/s3 = 8 runs，STAMP `20260702T163209Z`）；
- 若 3-seed 后 pole 的 both ≈ onlineb（−31 是噪声），方法直接收口：
  **T-RBO-PTF = student-as-arm 执行控制 + 对称 T-weighted replay**，
  然后进 4 任务×3seed 主表。
- 这是本方向第二次"顾问方案被实验修正"（T-gated 阈值脆弱 → split 非对称失配），
  两次都产出了机制级洞察：前者=弱信号下二值判定脆弱；后者=**off-policy AC 的
  replay 干预必须保持 actor/critic 数据分布一致**（可作为论文的独立 finding）。

---

## 6.8 seed 加固收口（2026-07-02 夜，STAMP `20260702T163209Z`，95k AUC 3-seed）

| task | scr | wfix | onlineb | **obrw（主方法候选）** |
|------|-----|------|---------|----------------------|
| crawl | 812.0±24.6 | 739.2±6.5 | 634.8±77.5 | **729.5±27.6** |
| pole | 603.3±48.3 | 767.5±24.4 | 761.3±3.3 | **755.0±15.5** |

**两个裁决都完成**：

1. **pole 的 −31 确认是单 seed 噪声**：obrw−onlineb 配对差 = −30.8/+11.6/+0.3，
   mean −6.3，t=−0.50（不显著）。obrw 755.0 ≥ 0.95×wfix(729.1) → **验收条件 PASS**。
2. **crawl 上 obrw 的收益稳健且比 s1 更大**：配对差 +41.6/+76.0/+166.4，
   mean **+94.7，t=+2.54，3/3 正**。且 obrw 把方差压小 3 倍（27.6 vs onlineb 77.5）
   ——replay 对称降权同时带来均值提升与稳定性。
3. **新信息（诚实）**：onlineb（无 replay 保护的 student-as-arm）在 crawl 3-seed
   其实**不稳**（634.8±77.5，比 wfix 739.2 差）——s1 的 726 是偏乐观采样。
   replay 对称降权是把它拉回 wfix 水平的**必要件**，不是可选优化。
4. **未回避的问题**：crawl 上 obrw(729.5)≈wfix(739.2) 打平、pole 上也打平
   （755 vs 768）——**T-RBO 相对静态 wfix 的净增益还没在这两个任务上体现**。
   它的预期卖点在 stair：wfix 在 stair 是负迁移（174±43 vs scr 252±37，静态权重
   没发现源不合适），若 onlineb/obrw 的在线自适应能在 stair 自动降教师份额→≥scr，
   "在线 vs 静态"的价值就坐实。**这正是 4 任务主表要回答的**。
5. crawl 离 scr 仍有 ~82 gap（迁移法的共同上限，源库无匍匐技能）——按 §7 的
   诚实预期，机制目标是"关负迁移逼近 scr"，obrw 已把 onlineb 的 177 gap 收到 82。

**下一步 = ChatGPT Step 2 主表补全**：stair/slide × onlineb/obrw × 3seed（12 runs），
补齐后主表 = 4 任务 × {scr/rand/wfix/onlineb/obrw} × 3seed 完整成型。

### stair 3-seed 结果（STAMP `20260702T195055Z`）——期望落空 + T^online 第三盲区

| 方法 | stair 95k AUC (3seed) |
|------|----------------------|
| **safe(h50)** | **279.2±20.3**（全场最好，超 scr）|
| scr | 252.5±36.7 |
| obrw | 184.2±23.9 |
| wfix | 174.1±43.3 |
| rand | 169.1±41.1 |
| onlineb | 157.7±49.0 |

1. **"在线自适应修 stair"落空**：onlineb 比 wfix 还差（配对 −16.3，t=−3.42 显著）；
   obrw ≈ wfix（+10.1，t=0.38），仍显著低于 scr（−68.4，t=−2.47）。
2. **stair 的负迁移根因不是"源不好"而是"执行方式不对"**：safe（同源、长 horizon=50）
   反而超 scr——教师有用但需要**长执行**；所有短 horizon 方法（h25：wfix/onlineb/obrw）
   全部负迁移。与 wfix 解耦的"horizon 任务依赖、stair +105"完全一致。
3. **T^online 的第三个盲区**：arm value 读的是"执行 25 步的 per-step reward"，
   读不到"换长 horizon 会更好"这个维度——执行时长不在 arm 空间里。
   （前两个盲区：T^0 误判 crawl；T^critic 误判 pole；现在 T^online 盲于 horizon。）
4. 候选出路（待 slide 齐后与全表一起交 ChatGPT 裁定）：(source, horizon) 联合 arm
   空间（如每源两档 h∈{25,50}，arm 数 2S+1）——机制上是 online_bootstrap 的自然
   扩展，无任务名分支；或接受"horizon 敏感任务用 safe 配置"作为已知边界诚实呈现。

---

## 6.9 完整主表（2026-07-03 凌晨，4 任务 × 6 方法 × 3seed，95k AUC）

| task | scr | rand | wfix | safe(h50) | onlineb | **obrw(T-RBO)** |
|------|-----|------|------|-----------|---------|-----------------|
| stair | 252.5±37 | 169.1±41 | 174.1±43 | **279.2±20** | 157.7±49 | 184.2±24 |
| slide | 271.1±46 | 450.2±20 | 522.7±20 | 504.7±14 | 551.3±5 | **614.8±16** |
| pole | 603.3±48 | 573.2±13 | **767.4±24** | 717.9±25 | 761.3±3 | 755.0±16 |
| crawl | **812.0±25** | 699.6±32 | 739.2±6 | 656.3±35 | 634.8±77 | 729.5±28 |

**关键配对（obrw vs 基线，per-seed 配对 t）**：

| task | vs scr | vs wfix（在线vs静态） | vs onlineb（replay 的贡献） |
|------|--------|----------------------|---------------------------|
| stair | −68.4 (t=−2.47) | +10.1 (t=+0.38) | +26.4 (t=+0.93) |
| **slide** | **+343.7 (t=+10.5)** | **+92.1 (t=+14.7)** | **+63.5 (t=+7.4)** |
| pole | +151.7 (t=+6.3) | −12.5 (t=−0.48) | −6.3 (t=−0.50) |
| crawl | −82.5 (t=−6.4) | −9.8 (t=−0.63) | **+94.7 (t=+2.5)** |

**全景判读（论文主张的底稿）**：

1. **slide = "在线 vs 静态"的决定性胜利**：obrw > onlineb > wfix 每级都极显著
   （+92.1 over 最强静态基线，t=14.7；且方差 16 vs scr 46）。在线竞争选出更好的
   执行组合，对称 replay 降权进一步纯化——此前悬而未决的"T-RBO 净增益在哪"有了答案。
2. **crawl = 负迁移防护**：obrw 比无 replay 保护的 onlineb +94.7（t=2.5），与最佳
   静态法打平（−9.8 不显著），离 scr 的 82 是源库无匍匐技能的共同上限。
3. **pole = 不误伤**：与 wfix/onlineb 打平（差异均不显著），≥0.95×wfix 验收持续 PASS。
4. **stair = 已知边界（诚实呈现）**：horizon 敏感型任务，safe(h50) 是该象限唯一
   超 scr 的配置；所有短 horizon 方法负迁移。T^online 盲于执行时长维度。
5. **obrw vs wfix 总账**：slide 决定性赢 + pole/crawl/stair 统计打平 → 总体占优；
   vs safe：3/4 任务赢（slide +110/pole +37/crawl +73），仅 stair 输。

**给 ChatGPT/导师的最终裁定问题**：
1. horizon-arm 扩展（每源两档 h∈{25,50}，arm 空间 2S+1，online_bootstrap 的自然
   推广、无任务名分支；stair 有 safe=279 作上界参照）——现在做（若成，四任务全绿、
   故事完美收口）还是留 future work（主表已足够成文）？
2. negctrl door/spoon（Step 3）与论文成稿的先后。
3. 主表方法列的最终呈现：obrw 定名 T-RBO-PTF；safe(h50) 作为"horizon 边界"的
   对照行还是并入 ablation？

---

## 6.10 ChatGPT 第三轮裁定（2026-07-03，看完完整主表后）+ 批判性审视 + horizon-arm 实现

### 裁定核心（原文要点）

1. **先做 horizon-arm，再做 door/spoon negctrl**——"不是要不要继续做，而是最后一块机制要不要补齐"。
2. arm 空间：`{student} ∪ {(π_i, h): h∈{25,50}}`，第一版只两档，不加 10/75。
3. **不要继续用 per-step reward EMA 作唯一 arm value**——改 snippet-level return
   `G_a = (1/h)Σγ^k r − λ_f·1[fall]`，整段归属一个 arm。
4. replay option_id 记 (source, horizon) arm 级；权重按 arm value；保持对称 reweight，不拆 split。
5. student 保持单 arm；先验同源两档同先验，让在线竞争区分；**禁任务名决定 h50**。
6. 最小验证矩阵：4 任务 × 1 seed mh-obrw，验收 stair≥scr(252.5)/slide≥584(0.95×obrw)/
   pole≥729(0.95×wfix)/crawl≥obrw−30(≈699.5)。过了再 3 seed；stair 不过就不调参，写 limitation。
7. 主表 6 方法列（scr/uniform/wfix/safe(h50)/onlineb/final），aonly/conly/tgated 进 appendix。
8. crawl 不死磕；negctrl 用最终方法 3 seeds；现在不扩源；T^critic 定位=baseline+分析图。
9. 一句话：最终方法回答"哪个 source option、以多长 temporal extent 执行、其数据是否继续影响 replay"。

### 批判性审视（第三次"顾问方案需实验修正"的预登记）

**采纳**：第 1/2/4/5/6/7/8 条全部采纳——与我们 §6.9 的候选设计一致，验收数字具体可执行。

**修正两点（第一版不采纳，登记为 fallback）**：

1. **per-step EMA 不换成 snippet-level return**。ChatGPT 的诊断"per-step reward 在短
   horizon 内读不出长 horizon 的价值"针对的是**旧 source-only arm 空间**（h 固定 25，
   EMA 确实收不到第 26-50 步）；一旦 arm 空间含 horizon，**per-step 归属本身就分辨了**——
   h50 arm 的 EMA 天然包含第 26-50 步的 reward（如登阶 burst），h25 arm 不含。
   且其公式的 γ^k 折扣有内在矛盾：γ=0.99 时 γ^49≈0.61，恰好系统性压低"第 40-50 步
   才出现的价值"——与 horizon-arm 的动机自相冲突。per-step 平均无折扣，无此偏差。
2. **fall penalty 第一版不加**。保持与 obrw 的唯一差异 = arm 空间（PI 的"一项一项做
   扎实"）；HB 摔倒即终止，done 后 reward 流自然中断，摔倒源的 EMA 已被部分压低。
   诚实承认盲区：摔倒的机会成本（episode 重置）不在 per-step reward 里。

**Fallback 预登记**：若 stair 1-seed 不过，第二版换 snippet-level mean return
（不带 γ 折扣）加 fall penalty，且能干净归因"修复来自 arm 空间还是 value 估计方式"。

### 实现（2026-07-03，方法代号 **mh** = mh-obrw）

- `fasttd3_ptf/ptf/mcg.py`：构造参数 `online_horizons`（如 `(25,50)`）；arm 空间
  S×H+1（source-major：arm=src·H+h_idx，学生=最后）；新增 `current_arm` 状态
  （arm 级 id，`current` 恒存 source id 供动作组装）；先验分支 bank 权重抽源 +
  档间均匀；在线分支 softmax over 全 arm；horizon 锁存查 per-arm 表；
  info 键 `mcg/arm_value_src{j}_h{h}`、`mcg/exec_share_src{j}_h{h}`。
- `train_ptf.py`：CLI `--ptf_mcg_online_horizons 25,50`；`option_ids` 消费
  `current_arm`（replay 权重按 arm 级降权——同源 h25/h50 独立）；`_w` 权重路径
  按 `arm_value` 长度自动适配（零改动）；multi-horizon 限 `bootstrap_only`
  （gated 分支 arm 语义未定义，诚实拒绝）。
- **不需要新 bank**：wfix bank 的源权重继续作先验，其 horizon 字段被 per-arm 表取代。
- 单元自测 8 项全过：构造/映射(src=arm//H)/锁存/EMA 归属 arm 级/在线分辨（好 arm
  份额 0.89）/单 horizon 回归（current_arm==current，旧 obrw 行为不变）/先验档均匀
  （h25 占比 0.49-0.50，学生保底 0.50）/done 重置。

### mh 1-seed 验收结果（2026-07-03，STAMP `20260703T071959Z`，95k AUC）

| task | mh-obrw | 验收线 | 判定 | 备注 |
|------|---------|--------|------|------|
| stair | **256.7** | ≥252.5（参照 safe 279.2） | **PASS** | vs obrw 184.2 = **+72.5**；尾段 477→563→618 强劲爬升 |
| slide | 544.3 | ≥584 | 边缘 FAIL（−6.5%） | 掉到 onlineb 551 水平；候选解释=先验档均匀的探索税 |
| pole | 721.5 | ≥729 | 边缘 FAIL（−1.0%） | 差 7.5，在单 seed 噪声带内（前例 pole −31=噪声） |
| crawl | **716.8** | ≥699.5 | **PASS** | 尾段 970 接近 scr 水平，防护无退化 |

**warmup 期机制证据（30k 冻结值）——horizon 偏好任务间自动分化，无任务名分支**：

- stair：walk h50 value 0.532 > h25 0.500，执行份额 h50=48.4% vs h25=12.5%——
  arm 竞争自动发现"walk 要连续执行 50 步"，正是 stair 修复的机制通道；
- slide/pole：在线竞争正确偏向 h25（slide 0.25 vs 0.14；pole 0.20 vs 0.09），
  与静态证据方向一致（两任务 wfix(h25) 均优于 safe(h50)）；
- crawl：arm value 窄带 0.20-0.26（弱信号，与 onlineb 观察一致），replay 降权承担防护。

**判读与决策**：stair 修复兑现（+72.5 远超噪声带）且 mh 是唯一四任务无负迁移的配置
（obrw 在 stair 是 184 的负迁移）；slide/pole 的边缘掉线是单 seed 数字——按预注册纪律
（stair 不过才停；不调参）走 **3-seed 裁决**（与 pole −31 事件处理一致）。
slide 若 3-seed 后仍掉 ~60，则"探索税"成立：先验期同源两档均匀，把一半教师预算花在
较差的 h50 档上——届时作为 trade-off 诚实呈现（stair +72 换 slide −70，总账打平但
消灭负迁移），或裁决先验档间是否允许用静态 T⁰ 分数（wfix/safe 分数已存在，非任务名）。
3-seed 加固 STAMP：`20260703T091439Z`（s2/s3 × 4 任务 = 8 runs）。

### mh 3-seed 终裁（2026-07-03 午，95k AUC）——**第一版全局 horizon-arm 被否决，obrw 保持最终主方法**

| task | mh 3-seed | per-seed | vs obrw 配对 | vs scr 配对 |
|------|-----------|----------|--------------|-------------|
| stair | 224.6±51.6 | 256.7 / 151.8 / 265.2 | +40.4 (t=+0.86) | −28.0 (t=−0.45) |
| slide | 456.6±94.3 | 544.3 / 499.7 / 325.8 | **−158.2 (t=−2.87)** | +185.5 (t=+2.84) |
| pole | 715.8±48.9 | 721.5 / 772.6 / 653.3 | −39.2 (t=−1.09) | +112.5 (t=+2.02) |
| crawl | 666.1±44.8 | 716.8 / 607.8 / 673.6 | **−63.4 (t=−2.63)** | −145.9 (t=−4.04) |

**裁决**：两个显著恶化（slide −158/crawl −63）对一个不显著改善（stair +40），总账明确
为负——mh 不能替代 obrw。按预注册纪律（stair 3-seed 均值 224.6 < 验收线 252.5）停止
调参；fallback（snippet return + fall penalty）不启动——它不对症：失败原因不是 value
估计方式，而是下述结构性代价。

**机制归因（三个独立证据，全部有正面价值）**：

1. **探索税**：先验期同源两档均匀，h25 已最优的任务（slide/pole）把一半教师预算浪费
   在较差的 h50 档上，纯付税无收益；slide 上税额 −158 且 s3 崩至 325.8。
2. **毒害扩大**：crawl 的 h50 档让坏教师单次执行步数翻倍（s2 中期 walk_h50 份额 66%），
   毒害注入量增加，replay 降权只能部分挽回（−63 vs obrw）。
3. **统计代价**：arm 空间 7 vs 4，30k warmup 预算内每 arm 样本减半——mh 四任务 std
   （51.6/94.3/48.9/44.8）全面大于 obrw（23.9/16.4/15.5/27.6），seed 敏感性上升。

**保留的正面发现（进论文 discussion/ablation）**：
- 在线竞争能自动发现任务的 horizon 偏好且方向全部正确（stair→h50 份额 48%，
  slide/pole→h25，双 seed 复现，无任务名分支）——**T 的 horizon 维度可在线辨识**；
- stair 上 mh 是唯一配对 vs scr 不显著为负的迁移法（−28 t=−0.45；obrw 是 −68
  t=−2.47），且 2/3 seed 越过 scr——horizon 修复方向存在，但在 30k 预算下不稳；
- 反面结论同样成立："transferability 依赖 horizon"不等于"全局加 horizon 档免费"——
  把不需要的档强塞进 arm 空间有结构性代价。这与 ChatGPT 预案的 limitation 写法一致：
  temporal extent must be explicitly modeled——但 how 仍是 open problem。

**收口决定**：主表不变（obrw=T-RBO-PTF 最终方法，safe(h50) 作 horizon 对照行，stair
写 limitation）；mh 3-seed 全套数据进 appendix；下一步按 ChatGPT 序列 Step 4 =
negctrl door/spoon（用 obrw 跑，3 seeds）。

---

## 6.11 negctrl door/spoon 结果（2026-07-03 晚，STAMP `20260703T124531Z`，{obrw,scr}×3seed，95k AUC）

bank：`h1hand_loco_wfix_{door,spoon}.yaml`（safe bank 改 h25，与主表 obrw 口径一致）。

| task | obrw | scr | 配对 Δ (3seed) | per-seed Δ |
|------|------|-----|---------------|------------|
| door | 295.8±8.5 | 295.0±5.4 | **+0.7 (t=+0.10)** | −1.1 / +13.8 / −10.5 |
| spoon | 354.2±1.4 | 315.4±13.9 | **+38.8 (t=+3.69)** | +42.2 / +55.0 / +19.1 |

**机制指标（ChatGPT 要求的三个报告量，3 seed 一致）**：

- door：学生执行份额自动升至 **86-88%**（学生 arm value 0.275 > 全部源 0.197-0.228），
  replay 源权重全部压到 0.10-0.25——**自动 abstain 生效且零代价**（+0.7≈0）。
- spoon：学生份额 73-84%，replay 权重 0.10-0.31——保留的 ~20% 教师执行提供了
  显著正迁移，且 obrw 方差被压缩 10 倍（1.4 vs scr 13.9）。

**判读**：door=教科书式 negctrl（源库覆盖不到任务瓶颈时，T-RBO-PTF 的执行/replay
份额自动坍缩到学生，性能与 scratch 统计不可分）；spoon 意外成为额外正例——loco 源
在 spoon 上有真实对价（与 2026-06-15 边界 pilot 的 SC-MCG +8% 相互印证），论文呈现时
spoon 应归入"弱对价正迁移"而非 negctrl，door 是纯 negctrl。**无对价→自动关闭、
弱对价→自动利用，两种行为出自同一机制、无任何任务级配置差异**——这是 T-RBO-PTF
自适应主张的最强直接证据。

**预注册序列执行状态**：①horizon-arm 1seed ✅ ②3-seed 裁决 ✅（否决第一版，诚实收口）
③4任务×6方法×3seed 主表 ✅ ④negctrl ✅（本节）⑤扩源扩任务=待 ChatGPT/导师裁定。

---

## 7. 风险与诚实边界（提前写给未来的自己）

- T^critic 的三个已知风险：早期 critic 不可信（→c(t) 调度）；源动作在学生 buffer 状态上
  OOD 导致 Q 过估计（→min-double-Q + 可加保守分位）；critic 只反映"当前学生的价值观"，
  可能低估"能带学生去新区域"的探索型源（→T^online 项补偿，实际执行 reward 含探索收益）。
- 完全零交互度量在 HB 异构任务上不可行（无任务族元数据），T^critic 的"半交互"定位要
  在论文里说清楚，不过度承诺。
- crawl 上即使 student-as-arm 生效，上限可能就是 ≈scr（源库确实没有匍匐技能）——
  这是诚实的预期：机制目标是"关掉负迁移"，不是"无中生有正迁移"。

---

## 8. 2026-07-11 接手后工作总结与战略转向

### 8.1 执行摘要

接手后，本轮工作的重点不是继续堆叠算法变体，而是检查现有结果究竟支持什么科学结论。
完成代码语义核验、hard-task stability-deconfounded 审计、cabinet 单源三种子复核以及
run-dose-matched control 后，当前最重要的判断是：

1. **现有主要性能通道确实是 warmup bootstrap，而不是已经得到独立验证的 MCG gate、
   scalar transferability `T` 或后期蒸馏。**此前 `bootstrap_only ≈ full` 的消融结论仍然成立，
   因而不能把所有框架组件并列包装成三个同等扎实的贡献。
2. **bootstrap 的收益不能统一解释为“源策略教会了目标任务技能”，也不能统一解释为“只是站得更稳”。**
   cabinet 和 maze 存在 survival 去混淆后仍然成立的早期 hard-task progress；powerlift
   几乎没有技能收益；basketball 则出现 stability 更高但成功率更低。
3. **源身份重要，但现有静态 return 权重不是校准良好的迁移性度量。**cabinet 中 run
   显著优于 stand，说明收益不只是共同的站立平衡；但静态 `T^0` 把 stand 与 run 估得很接近，
   shaped return 与 hard progress 还会反向。
4. **P2 没有证明“混入次优源会主动毒害 replay”。**在实际 run 剂量几乎完全匹配时，
   `run24 − WFix` 的种子方向为 `−/0/+`。均值轮廓更像高价值 run 的剂量/机会成本，
   但同样没有达到可以确认机制的证据强度。
5. **目前最有论文级 insight 潜力的重构不是即时 return winner-take-all，而是把源策略视为
   “目标环境中的数据生成干预”，并把 transferability 定义为该干预对目标学习器后续能力的
   stage-conditioned、effect-conditioned 边际学习价值。**这一路线已经完成实验协议设计，
   但尚未得到实证验证，不能提前宣称为贡献。

因此，本项目从现在起停止围绕 cabinet 单点剂量、阈值、floor、temperature、horizon 档位等
局部问题进行无边界扩展。下一阶段只做能够决定整篇论文中心命题是否成立的最小实验。

### 8.2 本轮工作范围

本节所称“接手后”，指开始对“收益是否只是 locomotion 站立稳定性”“多源是否稀释高价值数据”
以及“return 能否作为迁移性指标”进行系统审计后的连续工作。具体完成内容包括：

- 重新核验 maintained official 路径中的训练、MCG、replay 和 checkpoint 语义；
- 建立严格 episode-paired 的 stability-deconfounded hard-progress 评估工具；
- 完成四任务 P0 审计、cabinet P1 单源审计和 P2 run-dose-matched control；
- 从 W&B cross-sectional history 核验 warmup 期间实际 source/student 占比；
- 补齐配置、bank 生成脚本、结果分析脚本、完整性检查和单元测试；
- 审计 checkpoint/replay provenance 缺口；
- 设计 stage-conditioned replay data-value probe，作为可能的核心机制实验，而非立即再开
  一轮大规模训练矩阵。

---

## 9. 接手后完成的具体工作与结果

### 9.1 代码与实验协议语义核验

重点核验了以下三处 maintained official code anchors：

- `fasttd3_ptf/official_fasttd3_ptf/train_ptf.py`：warmup bootstrap、source/student 执行、
  `full|bootstrap_only|no_bootstrap` 消融、replay 写入与 checkpoint 保存；
- `fasttd3_ptf/ptf/mcg.py`：source option、模块化动作组合、segment latch 与 gate 逻辑；
- `fasttd3_ptf/official_fasttd3_ptf/ptf_replay.py`：transition 存储、option-id 字段和
  source-conditioned replay sampling。

核验后的一个重要修正是：replay wrapper **具备** option-id 存储字段，不等于旧实验已经保留了
可用于事后因果归因的真实行为来源。普通 WFix 运行中 `mcg_track_options=False`，大量 transition
的 option 被写为 `-1`；现有主要 checkpoint 也没有保存 replay、actor/critic optimizer、scheduler、
RNG、MCG controller 或环境状态。因此，不能从旧 checkpoint 精确重建某一训练阶段的完整 learner
state，也不能从旧 replay 做严格的逐来源反事实分析。

### 9.2 P0：stability-deconfounded hard-task audit

为直接检验“收益是否只来自站稳和活得更久”，新增并使用了：

- `configs/experiments/stability_deconfounded_audit_v1.json`；
- `scripts/stability_deconfounded_audit.py`；
- `scripts/build_stability_audit_banks.py`；
- `scripts/run_stability_audit_p1.sh`；
- `tests/test_stability_deconfounded_audit.py`。

评估按 `(train seed, checkpoint, eval seed, env rank)` 精确配对，在共同存活前缀上计算任务硬进展，
并检查重复、缺失和 reference condition。P0 共得到 **2,304 条 episode 记录**，覆盖 4 个任务、
scratch/WFix、10k/30k/100k、3 个训练种子和每 checkpoint 32 个评估 episode；重复 0、缺失 0。

核心结果如下。数值是 scratch → WFix 的 hard progress：

| 任务 | step | hard progress | 配对 Δ | 稳定性去混淆后的含义 |
|---|---:|---:|---:|---|
| cabinet | 30k | 0.021 → 0.260 | **+0.240** | 两组均存活 1000 步，不能由曝光时长解释 |
| cabinet | 100k | 0.500 → 0.948 | **+0.448** | door openness 方向一致，是真实任务进展 |
| maze | 10k | 1.000 → 1.542 | **+0.542** | WFix 反而少存活 169 步且 failure 更高 |
| maze | 30k | 1.271 → 1.896 | **+0.625** | common-survival prefix 上仍然成立 |
| maze | 100k | 1.865 → 2.000 | **+0.135** | 后期差距收窄，主要是早期加速 |
| powerlift | 10k–100k | 均约 0.190 | < 4.5e−4 | 实际量级可忽略，未学会举重 |
| basketball | 100k success | 0.323 → 0.135 | **−0.188** | 3/3 seeds 为负，稳定性提高也未转化为成功 |

P0 排除了“所有收益只是活得更久”这一单一解释，但也排除了“稳定 locomotion source 普遍迁移
目标技能”这一更强叙事。更准确的结论是：**stability 既不是正迁移的必要条件，也不是充分条件；
同一个 source bootstrap 在不同 target/effect 上可能带来任务进展、只改变姿态、没有有效技能收益，
甚至与目标瓶颈冲突。**

### 9.3 P1：cabinet 单源 run/stand 三种子审计

P1 用相同协议比较 scratch、stand-only、run-only（下文亦称 run50）和 WFix。所有条件在评估期
episode length 都是 1000、early failure 都是 0，因此 cabinet 的差异不是评估期存活时间造成的。

关键 hard progress 均值为：

| condition | 10k | 30k | 100k |
|---|---:|---:|---:|
| scratch | 0.000 | 0.021 | 0.500 |
| stand | 0.010 | 0.073 | 0.448 |
| WFix | 0.021 | 0.260 | 0.948 |
| run | 0.042 | **0.521** | **0.969** |

最重要的配对结果：

- `run − stand`：30k 为 `+0.448 ± 0.253`，100k 为 `+0.521 ± 0.110`，均为 3/3 seeds 正向；
- `run − WFix`：30k 为 `+0.260 ± 0.307`，逐种子 `+/+/−`；100k 仅 `+0.021 ± 0.100`，
  方向混合；
- 100k 时 run 与 stand 的 hard progress 相差 `+0.521`，但二者 return 差只有
  `+3.43 ± 17.5`。

P1 支持三点：第一，站立/平衡本身不足以解释 cabinet 提升；第二，source identity 对目标学习
很重要；第三，run 的优势目前主要表现为 early sample efficiency，而非已经证实的最终能力上限。
同时它暴露了 `T^0` 的局限：`T^0` 可以把 run 粗略筛到 top source，却把 stand 估得过于接近
run，因而不是校准的 learning-value estimator。

### 9.4 P2：run-dose-matched control

为区分“stand/walk 主动有毒”与“WFix 中高价值 run 剂量不足”，构造 `run24`：只保留 run 教师，
将 `warmup_exec_prob` 设为 `0.24021406`，使其 run 注入量与 WFix 中的 run 份额匹配；同时保留
run50 作为高剂量参照。

通过 W&B warmup cross-sectional history 核验的三种子实际占比为：

| condition | run | stand | walk | student |
|---|---:|---:|---:|---:|
| run24 | **23.829% ± 0.019%** | 0 | 0 | 76.171% |
| WFix | **23.894% ± 0.096%** | 20.780% ± 0.177% | 5.084% ± 0.171% | 50.242% ± 0.161% |
| run50 | 49.758% ± 0.161% | 0 | 0 | 50.242% ± 0.161% |

`run24` 与 WFix 的实际 run share 只差约 0.065 个百分点，因而本实验是 realized run-dose
匹配，而不是仅按配置计算期望剂量。新增 `scripts/analyze_warmup_source_dose.py`、实验清单与测试，
与 stability audit 测试合计 **8 tests passed**。

30k hard progress：

| condition | hard progress (mean ± train-seed std) |
|---|---:|
| WFix | 0.260 ± 0.118 |
| run24 | 0.292 ± 0.172 |
| run50 | 0.521 ± 0.284 |

严格按预注册分支，P2 的正式裁决是：**方向混合，机制仍不确定**。

- `run24 − WFix = +0.031 ± 0.113`，逐种子为 `−/0/+`：不支持“stand/walk 主动毒害”；
- `run50 − run24 = +0.229 ± 0.416`，逐种子为 `+/+/−`：均值轮廓符合剂量/机会成本，
  但不能声称已确认单调剂量机制；
- `run24 − scratch = +0.271 ± 0.188`，hard progress 和 door openness 均为 3/3 seeds 正向：
  **低剂量 run 本身具有可检测的目标学习价值**；
- 100k 三种 source 条件明显收敛：没有稳定的 asymptotic ceiling 提升证据。

P2 还给出了当前方法设计最关键的反例：30k 时 `run50 − run24` 的 hard progress 是 `+0.229`，
return 却是 `−69.8`；`run24 − WFix` 的 hard progress 只有 `+0.031`，return 却是 `+78.6`。
因此，**即时或 shaped return 不能直接作为源数据对目标学习器价值的代理，更不能据此启动
return-based winner-take-all。**

上述所有不确定性判断以 **3 个训练种子**为统计单位；同一种子内的 96 个 episode pair 只提高
评估精度，不能伪装成 96 个独立训练重复。本文中的均值、标准差和种子方向主要用于机制筛选，
不把 n=3 的描述性 t 值包装成确定性显著性结论。

### 9.5 仍然存在的 instrumentation 缺口

当前 W&B 审计能可靠回答 warmup 不同时点的横截面 source/student share，但不能恢复：

- 精确 segment 数、逐 segment 实际长度和 transition 数；
- warmup 训练期的 fall/termination 与早停片段；
- 逐来源 state coverage、接触状态和 target bottleneck coverage；
- replay 中每条 transition 的真实 behavior provenance；
- 某一阶段完整可克隆的 learner state（网络、optimizer、scheduler、replay、RNG、MCG 状态）。

评估期全程存活也不能反推 warmup 训练期同样全程存活。若下一步研究 replay data value，必须先
补足 paper-grade provenance 和 checkpointing；否则只能声称 matched-distribution evidence，不能
声称逐状态反事实或严格的因果效应。

### 9.6 已设计但尚未验证：stage-conditioned replay data-value probe

已完成 `docs/stage_conditioned_replay_data_value_probe_v1.md`，提出在同一 learner stage 下进行
等数据量、等更新次数的 run/stand/student replay 干预。其核心对象暂定义为：

`DV_s(t,e) = E[P_e(U^K(L_t, B_t ∪ D_s)) − P_e(U^K(L_t, B_t ∪ D_base))] / N`

其中：

- `L_t` 是阶段 `t` 的**完整**目标 learner state，而不只是 actor 权重；
- `D_s` 是 source `s` 在目标环境、固定 modular action semantics 下生成的 `N` 条数据；
- `D_base` 是同预算 student 数据或预注册的共同基线；
- `U^K` 表示完全相同的 `K` 次更新和采样日程；
- `P_e` 是独立评估种子上的 hard progress/effect `e`，不是 shaped return。

v1 最小 pilot 建议只做 cabinet 10k 的 run/stand/student 三臂。只有当 `DV` 在训练种子间可复现、
能预测后续 held-out learning outcome，并且比 return、`T^0` 或 critic advantage 更可靠时，才值得
扩展到多阶段、多任务并用于闭环选源。**当前它是候选研究方向，不是已经完成的方法贡献。**

---

## 10. 新证据对旧叙事的正式裁决

### 10.1 支持、未支持与必须收窄的主张

| 研究问题 | 当前证据 | 当前裁决 | 现在不能声称什么 |
|---|---|---|---|
| bootstrap 是否只提供稳定性 | cabinet/maze/powerlift/basketball hard metrics | 稳定性既非必要也非充分；部分任务存在真实早期进展 | 不能说所有增益只是站稳，也不能说普遍技能迁移 |
| source identity 是否重要 | cabinet run vs stand | 重要；run 在 30k/100k 均 3/3 seeds 更优 | 不能从一个 target 推广到全部任务 |
| 多源是否主动有毒 | run24 vs WFix | **未支持**；方向 `−/0/+` | 不能声称 top-1 或阶段最优注入已优于 mixture |
| 高价值 source 剂量是否重要 | WFix/run24/run50 | 有描述性轮廓，但 learner-seed 方向混合 | 不能声称已确认单调 dose-response |
| return 是否可作迁移性/选源指标 | P1/P2 return–hard progress 错位 | 只能作 behavior diagnostic，不能等同 data value | 不能做即时-return WTA，也不能用 AUC 代替任务能力 |
| 是否提高最终上限 | cabinet/maze 100k 收敛趋势 | 主要证据是 early sample efficiency | 不能声称稳定 asymptotic gain |
| student-as-arm 是否自动关闭负迁移 | door/crawl 等历史结果 | 提供连续退让通道，但没有普遍安全保证 | 不能写“自动 abstain 即保证无损” |
| `T^critic` 是否是核心迁移度量 | pole 等符号失效、仅有限排序信息 | 只可作诊断或 baseline | 不能支撑自动 transferability estimator |
| 一个 scalar `T` 能否统一执行/选源/replay | 三类信号失效模式不同 | **当前不成立** | 不能用同一数值同时解释 behavior 与 learning utility |
| MCG 是否有独立性能贡献 | `bootstrap_only ≈ full` | 架构价值存在，独立性能增益未获支持 | 不能与 bootstrap 并列为已证主贡献 |
| replay composition 是否影响学习 | crawl reweight、cabinet dose 审计 | task-dependent；因果来源未分清 | 不能泛化成“坏数据残留毒害”的一般规律 |

### 10.2 对旧章节中强表述的覆盖说明

为避免审计者误读，以下旧表述均降级为历史假设或阶段性观察：

1. `T^0` 不再称“已验证的 transferability metric”，只保留为静态先验或 coarse top-source
   screening signal。
2. student-as-arm 不再称“自动关闭负迁移”；它只是可学习的 fallback/execution budget 通道。
3. `T^critic` 的符号不能用于 transfer/abstain；原“一个 `T` 三处统一使用”的核心方案不再作为
   当前论文主张。
4. crawl replay 结果不再表述为“残留毒害已证实”。当前无法区分主动毒性、优质来源机会成本、
   剂量效应与 learner-seed 交互。
5. actor/critic 非对称 replay 在特定 seed 下较差，不升级为 off-policy actor-critic 的普遍规律。
6. 旧 95k shaped-return AUC 只作为 optimization/behavior evidence；不能用“决定性胜利”、
   “零代价”“最强直接证据”等措辞替代 hard-task evidence。
7. `obrw` 改称**当前强工程实现基线与经验锚点**，不再称已经足以支撑论文的“最终方法”。
8. “完全零交互不可行”收窄为：在当前异构任务、缺少 task metadata 与历史 meta-transfer
   dataset 的条件下，可靠零交互估计目前不可辨识、也没有证据支持。

---

## 11. 对整篇论文大框架的重新审计

### 11.1 建议审计的中心命题

当前最值得检验、但尚未被完全证实的中心命题是：

> **Cross-task source policies need not contain a target-task solution. Their transferable value lies in
> the stage- and effect-conditioned learning value of the target-environment data distributions they induce.**

中文表述：

> **跨任务源策略不必已经会做目标任务；其可迁移价值在于它们作为行为干预，在目标环境中诱导的
> 数据分布，能否在特定学习阶段、针对特定任务效应，提高目标学习器后续能力。**

这一定义能诚实解释当前结果：run 不是 cabinet teacher，却能产生有利的早期状态/动作/接触分布；
stand 同样稳定却未产生等价学习价值；powerlift 没有对应目标瓶颈收益；basketball 中稳定行为甚至
可能与投篮成功冲突。它也把研究问题从“哪个 source rollout return 高”提升为“哪种干预数据会
改变目标 learner 的学习结果”。

但该命题目前仍缺一个决定性环节：我们尚未直接测得 matched learner state 下的 marginal data
value，也尚未证明用该估计闭环分配 source/student/replay budget 能优于静态或均匀方案。

### 11.2 必须分开的四类量

旧方案试图用 scalar `T` 同时驱动 source selection、在线执行和 replay sampling；现有反例说明这几类
对象不能默认等价。建议改为条件化 profile，而非强行合并为一个分数：

| 分量 | 回答的问题 | 可观测信号 | 应控制的机制 |
|---|---|---|---|
| `T_behavior(s,t,e)` | source 此刻执行会发生什么？ | 即时 progress、return、coverage | behavior execution |
| `T_safety(s,t,e)` | source 是否提高跌倒/终止/冲突风险？ | fall、termination、constraint | gate / fallback |
| `T_data(s,t,e)` | 这些 transition 会否改善后续 learner？ | matched-update 后 hard-progress 增量 | replay allocation / source budget |
| `T_effect(s,t,e)` | source 影响的是目标的哪个瓶颈？ | posture、locomotion、contact、manipulation 等 | modular composition / map |

其中最可能形成新贡献的是 `T_data`；`T_behavior` 与 `T_safety` 可以快速在线估计，但不能冒充 delayed
learning value。`stage t`、target、effect 与 intervention dose 都应显式进入定义。

### 11.3 机制—贡献—证据矩阵

| 框架机制 | 科学角色 | 当前最强证据 | 论文贡献潜力 | 仍缺少的决定性证据 |
|---|---|---|---|---|
| RBO warmup bootstrap | 在目标环境中注入 source-conditioned reward-bearing data | bootstrap 是主通道；cabinet/maze 有去混淆后的真实早期进展；run24 低剂量 3/3 正向 | **当前最强机制基础**；应重构为 target replay/data intervention，而非 teacher skill cloning | matched data-value 与跨任务可预测性 |
| PTF source options | 提供多个冻结行为干预 | source identity 对 cabinet 学习结果重要 | 作为干预集合和 source library substrate | 不同 source/effect 的系统性因果图谱 |
| MCG modular composition | source 控制指定 body groups、student 保留其余动作；限制干预范围 | 代码和 critic-level modularity 证据；当前配置中 legs/torso/arms 与 hands 分工 | 若能证明“局部有用、整体有害”并隔离增益，可成为 controlled behavior compositor；否则是支持组件 | 相对 pure bootstrap/full-action 的独立、跨任务收益 |
| MCG gate / student-as-arm | safety/fallback 与执行预算调节 | 某些任务可退让；但 `bootstrap_only ≈ full`，且无普遍无损保证 | 安全组件或 supporting mechanism | gate 相对 bootstrap 的独立 regret/safety 改善 |
| `T^0` / reward-weighted source selection | 静态 source allocation prior | 能粗筛 cabinet 的 run；但严重失准且 return 与 hard progress 可反向 | baseline 或 cold-start prior | calibration、held-out prediction；目前不足以做主贡献 |
| `T^online` | 在线 behavior feedback | 能调节部分 source/student share | execution allocator | 与 delayed learning value 的关系未建立 |
| `T^critic` | 当前 learner 的局部 action preference | 有有限相对排序信息 | diagnostic/baseline | 符号和 OOD calibration 均不足，不应作核心 |
| source-weighted replay | 改变 off-policy 训练数据分布 | 特定任务有收益；不同配置/seed 结果不一 | 可成为 data intervention 的执行手段 | 与 `T_data` 对齐的目标函数及跨任务复现 |
| horizon arms | 建模 temporal extent | 能发现部分 horizon 偏好，但扩大 arm space 后统计代价高、总体失败 | discussion/open problem | 不宜作为当前主方法 |
| Source–Target–Effect Map + 去混淆审计 | 区分 survival、target progress 与冲突 | P0 已揭示 effect-specific transfer | 有潜力成为 evaluation/diagnostic contribution | 需裁定是否足够新，及能否指导而不只是解释结果 |
| stage-conditioned `DV` probe | 直接测 marginal target learning value | **仅完成协议设计** | 若可预测且可闭环使用，可能成为核心方法贡献 | 可复现 pilot、held-out prediction、closed-loop gain |

### 11.4 当前最有潜力的三点论文贡献候选

以下是需要 ChatGPT-5.6-Pro 审计的候选，而不是既成结论：

1. **Behavior-source data intervention（机制）**：把跨任务 source policy 从“要模仿的 teacher”
   重释为目标环境中的受控数据生成器；RBO 通过这些干预改变 early replay 与可达状态分布。
2. **Stage-/effect-conditioned target learning value（度量/决策）**：用相同 learner state、相同数据量、
   相同更新预算下的 hard-progress 增量定义 transferability，并据此分配 source/student/replay
   budget，而不是用 rollout return 或单步 critic advantage。
3. **Modular/effect-aware intervention（结构与诊断）**：用 MCG 限制 source 对不同 action/effect channel
   的干预，并通过 Source–Target–Stage–Effect Map 判断何处有益、无效或冲突。

三点只有在逻辑上形成闭环时才不是模块堆砌：MCG 定义可控干预单位，RBO 产生目标数据，`T_data/DV`
衡量这些数据对 learner 的边际价值，map 则用于解释与泛化。如果 MCG 的独立必要性或 `DV` 的预测性
无法成立，应主动降级对应组件，而不是继续用工程复杂度掩盖贡献不足。

### 11.5 三条可能的论文路线

**路线 A：保守、现有证据可支持。**以 RBO 的 early replay intervention 为主，配合 effect-aware
transfer taxonomy 和 stability-deconfounded evaluation。优点是诚实、已有结果较完整；缺点是若没有
新的 transferability object，可能仍被审稿人视为有效 bootstrap heuristic 加大规模实证分析。

**路线 B：概念重构、优先建议审计。**以 stage/effect-conditioned data value 为中心，把 RBO/MCG
变成可控的数据干预框架，再证明 `DV` 能预测 held-out 学习效果并闭环改善分配。若成立，方法、度量、
分析三者可形成统一故事；风险是它需要高质量 instrumentation 和少量但严格的因果实验。

**路线 C：维持 static WFix / reward-weighted `T`。**当前证据已经显示 return 与 hard progress 错位、
scalar `T` 失准、上限收益有限。除非有新的理论或强预测结果，否则不建议再把它作为论文中心路线。

---

## 12. 从现在起明确停止的低价值实验

除非 ChatGPT-5.6-Pro 证明某项是核心主张的最小决定性检验，否则暂不继续：

- cabinet-only 的更多 source dose 点或更细的 mixture grid；
- 根据即时/shaped return 做 winner-take-all 或“当前阶段最优源”在线切换；
- 继续调 `δ/τ/floor/temperature` 以修补 scalar `T`；
- naive 地增加 horizon arms 或扩大 source arm space；
- 为解释 n=3 的小差异再开大批 100k runs；
- 继续扩充 source library、增加一两个随机任务或单 seed 案例；
- 用 return AUC、稳定性或评估 episode length 代替 hard-task progress；
- 为每个失败变体分别包装一个“新 insight”。

实验的准入标准改为：**若结果的正、负两种方向都不会改变中心命题、贡献结构或方法定义，则不做。**

---

## 13. 建议的下一步：只做能裁决中心命题的实验

### 13.1 第一优先级：`DV` 可测性 pilot（先过生死门）

只选 cabinet、一个明确 stage（建议 10k）、run/stand/student 三臂：

1. 重新生成并保存 paper-grade anchor：网络、optimizer、scheduler、replay、normalizer、RNG、
   MCG/controller 状态，以及可重放的 matched reset/prefix anchors；
2. 从相同 learner state 克隆三臂，生成相同数量、显式带 behavior provenance 的数据；
3. 共享同一 background replay、相同 batch indices、相同更新次数，并设置 no-update/base-only control；
4. 用独立 eval seeds 测量 cabinet hard progress 与 door openness；return 只作诊断；
5. 预先定义最小可检测效应、训练种子数和 kill criterion。

**生死门**：如果 `DV_run > DV_stand/student` 不能跨训练种子复现，或零更新/采样控制无法通过，立即停止
该路线，不扩任务、不做闭环算法。

### 13.2 第二优先级：预测性，而不是继续解释 cabinet

只有 pilot 成立后，才在少量不同 stage 和 held-out source/target 上检验：早期估计的 `DV` 排名能否
预测固定预算后的 hard-task learning gain。至少与 uniform、static `T^0`/return、critic advantage
以及 student-only 比较。此步回答“它是否真是 transferability estimator”，而不是只回答
“cabinet 中 run 再次比 stand 好”。

### 13.3 第三优先级：一次闭环用途

只有预测性成立后，才用估计的 `DV` 分配 source/student execution 或 replay budget，并与 uniform、
static weighted mixture 和 top-source oracle/post-hoc upper bound 比较。无需一开始做大矩阵；一个正例、
一个无对价任务和一个冲突任务足以判断方法是否具有普适决策价值。

### 13.4 是否追求最终上限

现有证据最诚实的卖点是 early sample efficiency。论文不必为了看起来更强而虚构 asymptotic gain，
但必须回答：早期加速是否在固定低数据/低交互预算下稳定、是否跨任务可预测、是否没有以目标成功率
或安全性为代价。如果目标期刊/会议要求最终能力提升，应由审计者明确指出需要怎样的机制改变；不能
仅靠继续延长当前 warmup 或微调 source 权重期待自然出现上限提升。

---

## 14. 请求 ChatGPT-5.6-Pro 进行的战略审计

### 14.1 审计任务

请把 §0–§7 视为历史研发记录，把 §8–§14 以及下列 evidence packet 视为当前状态。不要默认接受
作者提出的 `DV` 重构；请以严格审稿人/研究导师身份优先寻找不可辨识、概念偷换、与已有工作的重合、
缺失 baseline 和无法被证伪之处。我们希望得到的是论文级方向裁决，不是再推荐一组阈值实验。

请重点回答：

1. 基于现有证据，整套框架最有 insight 且可证伪的**单一中心命题**应如何精确定义？
2. 哪 2–3 点足以成为相互依赖而非模块堆砌的核心贡献？哪些必须降级到 supporting component、
   diagnostic 或 appendix？
3. “source policy as target-data intervention”是否真正超出普通 policy reuse、offline-to-online RL、
   replay prioritization、skill composition 和 curriculum learning？差异应如何形式化？
4. transferability 应定义为 policy-level scalar，还是 source–target–stage–effect 条件化 profile？
   当前 `DV` 定义是否具备因果和操作意义？有没有更好的对象，如 gradient alignment、influence、
   held-out Bellman improvement、representation coverage 或 counterfactual policy improvement？
5. `L_t` 必须包含哪些状态，`D_base` 应选 student data、no-extra-data 还是 matched random data，才能
   避免把 exploration、state distribution、optimizer history 和更新次数混为一谈？
6. MCG 应保留为主贡献、受控干预/安全组件，还是降级为实现细节？需要哪个最小实验才能裁决？
7. Source–Target–Effect/Stage Map 是真正的方法贡献、评估贡献，还是仅是结果分析工具？怎样才能从
   post-hoc taxonomy 变成可预测、可决策的对象？
8. 现有证据中哪些可以进入主文，哪些只能进入 appendix，哪些旧 claim 应完全删除？
9. 如果下一轮最多只允许 **2–3 个决定性实验**，应选择什么？请对每个实验写明：支持/反对结果
   分别怎样改变论文中心命题，以及明确的停止条件。
10. early sample efficiency 是否足以成为主问题？若不足，必须新增什么机制才能合理追求
    asymptotic gain，而不是继续扩大训练预算？
11. 应比较哪些强 baseline？至少请审视 uniform mixture、static return/`T^0`、student-only、
    critic-advantage、bandit allocation、oracle/post-hoc source ranking，以及 APT-RL/PMIC 类方法中
    真正可比的部分。
12. 如果当前大框架仍不足以撑起整篇论文，请给出**最小概念重构**，并明确建议放弃哪些既有机制，
    而不是提出一个更大的组件清单。

### 14.2 希望的输出格式

请按以下顺序给出审计结论：

1. **总裁决**：当前能否形成一篇方法论文，最大 fatal flaw 是什么；
2. **一句话 thesis**：精确、可证伪、不过度承诺；
3. **贡献裁剪表**：保留 / 重构 / 降级 / 删除；
4. **核心对象的正式定义**：变量、干预、估计目标、假设和不可辨识项；
5. **最小决定性实验**：不超过 3 个，含 baseline、指标、种子、成功/失败/停止标准；
6. **论文叙事**：问题设定、方法模块依赖、主图和主表应各回答什么；
7. **相关工作边界与新颖性风险**；
8. **Go / Pivot / Stop 建议**。

### 14.3 当前 evidence packet

- P0 总结：`docs/stability_deconfounded_transfer_audit_v1_findings.md`
- P0 协议：`docs/stability_deconfounded_transfer_audit_v1.md`
- P1 总结：`docs/stability_deconfounded_audit_p1_cabinet_s123_findings.md`
- P2 预注册：`docs/stability_deconfounded_audit_p2_run24_protocol.md`
- P2 裁决：`docs/stability_deconfounded_audit_p2_cabinet_run24_findings.md`
- realized source dose：`docs/cabinet_p2_warmup_source_dose.md`
- 候选 data-value probe：`docs/stage_conditioned_replay_data_value_probe_v1.md`
- 早期主通道消融：`docs/wide_pilot_v1_results.md`
- 当前代码入口：`fasttd3_ptf/official_fasttd3_ptf/train_ptf.py`
- MCG：`fasttd3_ptf/ptf/mcg.py`
- replay：`fasttd3_ptf/official_fasttd3_ptf/ptf_replay.py`

### 14.4 当前作者立场（供审计者反驳，而非要求认同）

当前倾向路线 B：不再把 static WFix、即时 return 权重或“一个 scalar `T` 三处复用”作为论文中心；
把 RBO 解释为 source-conditioned target-data intervention，把 transferability 重定义为
stage/effect-conditioned marginal target learning value，并要求 MCG 证明其作为受控干预器的独立必要性。

但在 `DV` pilot、held-out prediction 与一次闭环分配实验完成之前，我们只把它视为**最有潜力的
研究假设**。当前已充分支持的结论仍限于：bootstrap 是主要通道；部分任务存在真实早期进展；
source identity 重要；stability/return 都不能单独代理目标学习价值；尚无稳定的最终上限提升证据。

---

## 15. Source-intervention 机制门正式结果与审计后裁决（2026-07-11）

### 15.1 完成了什么

依据 ChatGPT-5.5-Pro 审计中“先识别行为/reachability 与 replay/update 两条通道，不要直接把
source transition value 当作总 transferability”的关键意见，同时保留了我们自己的约束：只做
一个会改变路线的 cabinet/scratch10k/run-composite pilot，不扩任务、不搜索正例。

已经完成：

- 中心命题、total/local `B0/R0/I/T` estimands 与 oracle/deployable-estimator 边界；
- 相关工作与新颖性边界审计；
- paper-grade full learner anchor、正确 Gymnasium seed plumbing、MuJoCo branch-state parity；
- 512 个共同 anchors 的 `Z^S/Z^R/F^S/F^R` paired trajectory bank；
- 固定 50% D0 + 25% Z + 25% F sampler、相同 target noise、相同更新预算的六分支实验；
- no-update、D0-only、duplicate00、K=0、provenance、hash、seed-leak 与 checkpoint controls；
- 全仓测试 `218 passed`。

完整协议、实现审计和结果见
[`source_intervention_mechanism_gate_v1.md`](source_intervention_mechanism_gate_v1.md)，原始结果见
[`gate_report.json`](../artifacts/mechanism_gate/cabinet_s1/gate_report.json)。

### 15.2 正式结果

`00/10/01/11` 在 K=400 的 source-free max-door mean 分别为：

`0.10038 / 0.04932 / 0.07001 / 0.05202`。

因此：

- reachability/bridge-only `B0=-0.05106`；
- prefix replay/update-only `R0=-0.03037`；
- interaction `I=+0.03307`；
- full local intervention `T=-0.04836`。

预注册实践阈值为 `0.10`；没有一个机制 contrast 过门。归一化 K=0/100/200/400 AUC 的
`B0/R0/I/T` 也分别只有 `-0.01156/-0.00548/-0.00584/-0.02288`。`00` 与
`duplicate00` 的模型、全部 episode JSON 与 update logs 精确一致，故 duplicate noise=0，
Engineering Go=true；结果不是 GPU 漂移、不同采样或初态不一致造成的。

### 15.3 对原始担忧的回答

run-composite 确实强烈改变 25 步 endpoint state（observation L2 median `61.94`），提高 head
height/upright，并让 hand 短期更接近 door；但 endpoint door fraction 只增加 `0.00036`，没有
subtask progress。K=400 时，包含 run 的 full cell 相比 student baseline 仍然是：

- head height `+0.0557`、upright `+0.0687`；
- max-door `-0.0484`；
- door≥0.25 time fraction `-0.0394`；
- success subtasks `-0.015625`。

所以当前最有证据的解释不是“run 教会了 cabinet manipulation”，而是它改变了 locomotion/posture
与 reachable geometry；这种变化在本 stage 并没有转化为足够的目标技能学习价值。该结果实质上
支持了 PI 的核心担忧，同时又避免把单 learner seed 外推为“run 永远有害”。

### 15.4 路线裁决

正式裁决为：**`STOP_COMPLEX_ESTIMATOR`**。

不再执行：

- 当前 oracle 2×2 的多 learner-seed 扩展；
- `hat(SIV)`/DV/gradient-influence estimator；
- held-out source-target-stage prediction 与 closed-loop budget allocation；
- winner-take-all/阶段最优源注入；
- 通过改 K、h、dose、source 或 task 搜索过门结果。

这不是宣告整项研究失败，而是裁掉一个未过可测性生死门的高复杂度支线。当前 bootstrap
early-acceleration 证据仍可保留，但不能再被包装成“目标技能迁移”或 asymptotic gain。下一轮战略
讨论必须回到整篇论文的大框架：选择一个有独立理论对象、清晰 failure mode 和跨条件正面证据的
核心机制；不能用更多 cabinet 小消融、source weighting 或阈值实验填补贡献强度。

---

## 16. 整篇论文核心贡献重构启动（2026-07-11）

### 16.1 新中心问题

不再问“怎样估计哪个 source 最可迁移”，而改问：

> **跨任务 source policy 实际给 humanoid target learner 提供了什么？**

当前最能解释全部正、负结果的候选答案是：source 主要是 cold-start 阶段的
**transient behavioral scaffold**，而不是 target-task teacher。它通过在目标环境中产生带目标奖励的
可行 whole-body experience 加速学生获得目标技能；当 source 占用了目标 bottleneck 所需的控制权时，
同一个机制也会产生 interference。

工作标题：

> **Beyond Skill Transfer: Transient Behavioral Scaffolding for Humanoid Reinforcement Learning**

### 16.2 新贡献结构

1. **机制发现**：区分 source competence、whole-body viability、source-free target progress、
   post-handoff retention 与 final ceiling，说明稳定性可能提供 enabling scaffold，但不是目标技能、
   也不是充分条件。
2. **最小方法 TBS**：冻结 source 仅作为 cold-start behavior policy；source/student segment 交替，
   transition 使用 target reward 进入统一 replay；warmup 后 source 完全撤出，最终 student 独立执行。
3. **hard-progress mechanism audit**：用 cabinet/maze/hurdle 正例、powerlift null、basketball conflict、
   source identity/dose 与 10k late-stage mechanism gate 划定帮助和干扰边界。

TBS 本身与 PTF、JSRL、SFP/behavior priors 存在明显重叠，因此 base paper 定位是机制型 CCF-B，
不把简单 prior mixing 冒充算法首创。

### 16.3 唯一方法升级门

只允许检验 **Effect-Preserving Scaffolding (EPS)**：在 manipulation warmup 中，同一个 WFix source
只控制 anatomy-defined `legs_torso`，student 始终控制 `arms,hands`；与 scratch 和 full-action WFix
比较。只用 cabinet 正例与 basketball conflict，不搜索新 task。

单 learner-seed feasibility 必须同时满足：

- cabinet 保留 full scaffold 至少 80% 的 hard-progress gain，或绝对提高 ≥0.10；
- basketball 相比 full scaffold 回收至少 50% success regret，且不低于 scratch−0.05；
- posture treatment 仍可检测，equal dose/update、source-free eval 和 duplicate control 全通过。

任一失败即停止 EPS，不换 mask、不调 horizon、不扩 source。EPS 通过才允许 3 seeds；EPS 多种子通过
后才做一个 early-vs-delayed same-dose stage-locality gate。

### 16.4 继续禁止

- `DV/SIV` estimator、held-out prediction、closed-loop allocation；
- winner-take-all 或阶段最优 source；
- horizon-arm、更多 source、killer-task 搜索；
- 用 shaped return 或 episode length 替代 target hard progress；
- 把 early acceleration 写成 asymptotic ceiling gain。

### 16.5 EPS 实验契约已冻结

后续代码实现与实验不得仅依赖“相同 RNG seed”匹配 full/EPS，因为两种 authority 会改变 termination，
进而使当前 controller 的重抽源时机分叉。正式 gate 使用与 episode termination 无关的固定 25-step
schedule tape，并逐 transition 写入真实 source/body-group provenance。完整 E0/E1/E2 Go/Stop 条件见
[`eps_feasibility_gate_v1.md`](eps_feasibility_gate_v1.md)；E0 工程门通过前不启动科学 run。

---

## 17. PI 修正裁决：恢复迁移强化学习方法主线（2026-07-12）

PI 明确指出：单独把论文收缩为 TBS/EPS 机制论文偏离“HumanoidBench 上自动选择 source、通过
迁移强化学习加速或提高有限预算性能”的研究目标；uniform replay、固定 teacher probability 与
student-as-arm 不能 exact fallback 都是仍需解决的方法问题。该批评成立。

新的统一路线不是原样恢复旧 `T⁰→onlineb→OBRW→MCG`，而是：

1. **Stage-Conditioned Handoff Utility**：在当前 student occupancy 上比较
   `source prefix→student follow-up` 与 `student→student`，把直接收益、handoff收益与风险分开；
2. **Student-inclusive conservative admission**：student为reference arm；无source过门时
   `p(student)=1`，删除固定0.5 teacher floor；
3. **Quarantine probe**：source probe不进入main replay，通过准入后才允许产生训练数据；
4. **Admission-consistent replay**：source撤销时同时停止其行为与旧数据采样；active strata按当前
   admission与recency采样，actor/critic保持同分布；
5. **RBO/TBS保留为性能通道/机制解释**；MCG/EPS降为source准入后的可选authority层。

旧 cabinet 10k formal gate 的 `STOP_COMPLEX_ESTIMATOR` 不被推翻：不再估计K-step learner data value
或跨任务ROI；它现在是stage-conditioned admission应当拒绝late source的负标签。

当前战略主文档：
[`paper_core_contribution_reconstruction_v2.md`](paper_core_contribution_reconstruction_v2.md)。首个
read-only辨识力门：
[`stage_conditioned_source_admission_gate_v1.md`](stage_conditioned_source_admission_gate_v1.md)。
旧EPS协议暂停；SHU gate前不修改训练闭环、不启动正式100k实验。

### 17.1 2026-07-12 实现进展

SHU read-only collector、quarantine schema、analysis/report与通用HumanoidBench branch diagnostics已经实现；
9项聚焦测试及cabinet/hurdle/crawl各1-anchor真实checkpoint smoke通过。实现中特别修正了两类会污染
measurement的风险：FastTD3训练worker维度的瞬时`noise_scales`不直接映射到collector，而是按anchor
从冻结`std_min/std_max`确定性重采样；risk同时纳入termination/truncation与固定姿态fall proxy。

这些结果只构成Engineering smoke，不是SHU辨识力证据。正式512-anchor mandatory gate仍未运行；在它
同时满足hurdle positive、crawl negative、cabinet late-null标签前，不实现closed-loop admission、
replay撤销或新100k训练。

### 17.2 SHU v1 mandatory regression裁决：Stop（2026-07-12）

cabinet/scratch10k/run-composite/h25+f25正式512-anchor cell在reference protocol parity、duplicate exact、
quarantine-only和完整provenance通过后，仍把run判为eligible：direct `+0.0253`、handoff `+0.1728`、
risk `−0.0156`，三项non-compensatory checks全部通过。旧formal 2×2却显示该candidate经过FastTD3
update后的max-door `T=-0.04836`、normalized AUC `T=-0.02288`。同时三个null robust scale均塌到
`1e-6` floor。

因此aggregate为`mandatory_label_contradiction / STOP_CLOSED_LOOP`；hurdle/crawl不再补跑，closed-loop
admission、active replay撤销和新100k训练均禁止实现。该失败给出的方法学结论是：短期行为/交权效用
不等于source transition的数据/更新效用，不能用一个SHU分数同时决定“让谁控制环境”和“让哪些数据
训练critic”。下一版核心重构必须显式分离behavior transfer与replay transfer；若不能得到低成本、可辨识
的data-utility机制，则论文退回RBO/TBS有限预算实证主线，source selection只能作有边界的heuristic。

---

## 18. v3中间裁决：从transferability分类转向双通道暴露控制（2026-07-12，已被§19校正）

> 本节保留SHU失败后第一轮重构的审计轨迹。其中把OBRW/CE-RBO升为默认主方法的判断，没有纳入
> 2026-07-05已完成但当时文档未终裁的WFix九任务裁决与第二批本地结果，现已被§19校正。

SHU失败后重新审计既有onlineb、OBRW、split replay、terrain主表、door/spoon与cabinet 2×2，发现已有
数据足以支持一个更扎实的中心命题：source执行造成瞬时occupancy exposure，而source transition驻留
replay造成持久update exposure；减少前者不能自动消除后者，且actor/critic对后者的控制必须保持相同
sampling distribution。

关键证据链：

1. crawl中onlineb把student执行份额提高到76%仍无法恢复；三种子AUC仅`634.8±77.5`；
2. 对称OBRW把crawl提高到`729.5±27.6`，相对onlineb配对`+94.7`且3/3正；
3. actor-only/critic-only/split在crawl分别只有`676.2/629.6/715.9`，均低于both `767.5`，支持
   actor–critic replay distribution coherence；
4. slide中OBRW `614.8±16`，高于onlineb `551.3±5`和wfix `522.7±20`；
5. door自动把student份额提高到86–88%且与scratch打平，spoon则保留弱source对价并提高；
6. cabinet formal 2×2和SHU反例共同证明behavior/handoff与update utility不可合并成一个标量。

本轮曾暂定工作名称**Coupled-Exposure RBO-PTF（CE-RBO）**。后续总账证明OBRW不足以承担默认
主方法，因此该名称不再使用；代码/实验名OBRW仅保留为历史标签和在线扩展。`T^online`降级为
operational feedback、不再叫完整transferability的判断仍然有效。

当前先完成machine-readable result registry、配置追溯、论文主表与机制图，不启动新100k实验。详细证据
矩阵见[`dual_channel_transfer_evidence_matrix_v1.md`](dual_channel_transfer_evidence_matrix_v1.md)，当前
战略主文档见[`paper_core_contribution_reconstruction_v3.md`](paper_core_contribution_reconstruction_v3.md)。

---

## 19. 最新总裁决：静态RBO主方法 + 条件化迁移规律 + 双通道边界（2026-07-12）

### 19.1 为什么必须再次校正

§18只审计了onlineb/OBRW与SHU链，没有纳入更晚的两组已完成证据：

1. WFix五任务裁决与九任务全局对账：OBRW相对静态WFix在slide决定性大胜、spoon小幅胜；WFix
   在cabinet显著胜出，其余六任务打平；
2. 第二批本地W&B binary审计及恢复：48个逻辑run slot现已全部有效；`balance_hard`两个首次OOM
   cell于2026-07-12按原配置补齐，历史失败仍保留为attempt provenance。basketball上WFix与OBRW
   均3/3负迁移。

因此“执行+replay在线耦合=默认主方法”的复杂度没有被全局数据支持。论文不能因为该机制在crawl/slide
有洞察，就忽略静态方法在更多任务上不差或更好，也不能在basketball硬反例后继续宣称在线层普遍安全。

### 19.2 第二批重算结果

严格使用每个本地run的19个评估点（5k到95k）、归一化梯形AUC和同seed配对：

| task | scratch | rand | WFix/RBO | OBRW | WFix−scratch |
|---|---:|---:|---:|---:|---:|
| bookshelf_simple | 679.4±67.4 | 654.7±19.9 | 711.6±50.1 | — | +32.1, t=1.13 |
| basketball | 188.9±70.6 | 116.0±23.7 | 87.4±20.6 | 114.8±21.0 | **−101.5, t=−2.58，3/3负** |
| window | 269.0±41.8 | 206.4±71.0 | 237.2±58.1 | — | −31.9, t=−0.99 |
| powerlift | 177.6±9.6 | 254.2±6.0 | **255.2±0.8** | — | **+77.6, t=14.78，3/3正** |
| balance_hard | 84.4±4.9 | 91.1±4.2 | 90.5±15.4 | — | +6.1, t=0.75 |

本表`±`为sample SD。`balance_hard rand-s3`与`WFix-s2`首次在replay buffer分配时CUDA OOM，
后于2026-07-12原配置恢复成功；完整账本是50次attempt、48个有效逻辑slot。复核入口为
`artifacts/breadth_batch2_local_audit/analysis.json`和`scripts/analyze_breadth_batch2_local.py`。

两个科学结论尤其重要：powerlift在所有50-step probe分数<1时仍有稳定大增益，而basketball在弱正
probe下产生大负迁移；因此`T^0`低分区没有方向判别力。OBRW虽然相对WFix减少basketball损失，却仍
3/3低于scratch；student-as-arm与对称replay降权不是exact fallback。

### 19.3 最终方法层级

1. **默认主方法：静态RBO/WFix**——`T^0`相对权重 + softmax source allocation + 0.5 student
   branch + h25 segment + target-reward replay + `bootstrap_only`；
2. **在线扩展：OBRW**——解释execution/replay双通道，在slide型执行伤害任务可能额外有益；不作为
   默认配置，不承诺自动安全；
3. **supporting：MCG gate/distillation**——代码保留，因`bootstrap_only≈full`不作headline贡献；
4. **failed/appendix：SHU、hard abstain、split、multi-horizon**——分别贡献measurement boundary、
   阈值脆弱、AC distribution coherence和arm-space exploration tax。

### 19.4 论文真正可撑住的四点贡献

1. **Reward-bearing Option Bootstrap**：冻结source通过目标环境交互改变有限预算数据获取，目标
   FastTD3从真实target reward transition学习，最终student source-free；
2. **source-bank条件规律**：加权价值随bank内强弱分化增大；扩源价值由skill complementarity与
   remaining learning headroom共同决定（truck +229.9 vs maze +0.3）；
3. **双通道负迁移诊断**：execution/occupancy与replay/update持续时间不同；crawl、split、cabinet
   2×2与SHU共同支持该分解及actor--critic sampling coherence；
4. **跨任务regime map**：同时报告positive、weak/null、negative、horizon-sensitive和saturation，
   并把return与hard progress分层；hurdle/cabinet/maze有任务进展佐证，powerlift只有稳定AUC而无
   hard-skill证据，不再把所有return提升解释为完成目标skill或提高上限。

### 19.5 当前禁止继续消耗的方向

- 不为挽救SHU继续调阈值或跑更多分类cell；
- 不把OBRW重新包装成万能transferability estimator；
- 不补跑大规模source×task网格来追求“全绿”；
- 不把basketball硬负例隐藏，也不把不完整balance_hard写成3seed；
- 在配置/result registry、论文主表、abstract和method figure完成前，不启动新100k训练。

下一步是成稿工程与claim audit，而不是继续用无关紧要的小实验寻找局部胜场。
