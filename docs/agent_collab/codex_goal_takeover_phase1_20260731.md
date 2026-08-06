# Codex 接管长期科研目标：Phase 1 研究问题与方法蓝图

> 日期：2026-07-31  
> 状态：Stage 1 / Scoping；不启动新实验、不改变既有预注册。  
> 依据：`docs/RESEARCH_EXECUTION_GUARDRAILS_20260721.md`、  
> `docs/EVIDENCE_STATE_20260731.md` 以及截至 `d90d9b1` 的实验记录。

## 1. Goal 的正确解释

长期目标不是“必须发明一个零交互迁移性标量”，而是：

> 在 HumanoidBench 上，把冻结跨任务策略转化为对 FastTD3 student
> 有实际样本效率价值、能够自动选择或拒绝、最终不依赖 source 的迁移机制，
> 并形成一条经得起论文审稿的因果证据链。

因此以下结果都可以构成 Goal 的有效进展：

1. 新的 teacher-value / transferability 机制；
2. 不依赖静态迁移指标的低成本 source-selection protocol；
3. 与 reward-bearing bootstrap 结合的安全干预或数据利用机制；
4. 在困难任务上可重复的样本效率或上限提升；
5. 对负迁移和 learner-path dependence 给出方法层面的解决方案。

## 2. 当前最强事实基础

### 已支持

- Reward-bearing bootstrap（RBO）是目前最可靠的性能通道。
- Hurdle 上 run 源相对 scratch 的早期达阈速度为 3.5–4.4×，但 100k
  优势缩小，且训练稳定性仍需改善。
- Hurdle 上，3 个 source 各进行 10k 真实 bootstrap 干预后，racing 在
  两批共 6 个 learner 上均选中 run；K=5k 不可靠，K=10k 才稳定。
- Slide 上长期 source 排序的 argmax=walk 已在 6 个独立 learner 上一致，
  与 Hurdle 的 argmax=run 构成真正 crossover。
- Exact abstention、provenance、replay lifecycle、source-free evaluator
  已具备支撑新方法的工程条件。

### 未支持

- 没有可靠的零交互绝对迁移性指标。
- 没有证明 racing 选中的 source 比任意 source 更有训练价值。
- 没有把 racing 与长期 bootstrap 串成一个端到端算法。
- Slide 尚未证明长期加速；当前跨任务正面证据仍不足。
- Door 表明绝对 transfer utility 甚至符号可能依赖 learner trajectory，
  不能把一次实验的 `U(source,target)` 当成 target 的固定属性。

## 3. Research Question Brief

### Primary Research Question

在固定总干预预算下，能否通过对当前 FastTD3 learner 进行短期、
stage-conditioned 的 reward-bearing bootstrap racing，选择一个能显著降低
长期 source-free 达阈步数的冻结 source policy，并在证据不足时 exact abstain？

### FINER Assessment

| 维度 | 分数 | 理由 |
|---|---:|---|
| Feasible | 4/5 | racing、RBO、anchor、source-free evaluator 已存在；端到端串联尚需设计 |
| Interesting | 5/5 | 直接回应“谁来教、何时不教”，且已有 proxy 系统失败的强动机 |
| Novel | 3/5 | 在线多 source policy selection / bandit policy reuse 已有先例；候选增量只可能是用 target-reward bootstrap 后的 student learning response 作 arm outcome，并允许 exact abstention |
| Ethical | 5/5 | 仿真研究，不涉及人类受试者或现实机器人伤害 |
| Relevant | 5/5 | 直接服务 PTF × FastTD3 × HumanoidBench 的主问题 |
| **平均** | **4.6/5** | 达到推进门槛 |

### Sub-questions

1. **Selection value**：racing 选中的 source 是否优于同一 bank 中较差但仍有益的 source？
2. **Cross-target performance**：该 source 在至少第二个 argmax 不同的 target
   上能否带来长期 source-free 加速？
3. **Abstention under learner dependence**：当短期相对排序或绝对收益不稳定时，
   能否用 replicated/risk-sensitive 判据退化为 student，而不假装存在固定的
   `U(source,target)`？

### Scope

**In scope**

- 冻结 source policy；
- HumanoidBench；
- FastTD3 student；
- target-reward-labeled online source trajectories；
- behavior/replay bootstrap；
- source-free student evaluation；
- 有成本、可并行的短期 interventional measurement。

**Out of scope**

- 继续搜索第十二个零交互 proxy；
- 把一次 per-seed U 当作全局 ground truth；
- 复活已经裁决失败的 SIV/SHU/P0/QMP/critic-sign 路线；
- 在本阶段承诺“大部分 HumanoidBench 任务”；
- 把工程基础设施本身包装成性能贡献。

## 4. 候选路线蓝图（尚未冻结）

以下两条路线都只处在研究设计阶段。它们回答不同问题，不能合并包装成
一个已经成立的方法。

### 4.0 对 R3B 的关键纠正

暂定名 **Racing Reward-Bearing Bootstrap (R3B)** 目前只能表示一种
有成本的 source-selection protocol，不能作为已成立的核心创新。原因是：

1. 同一批分支共用 student 基线时，
   \(\arg\max_i[J_i-J_{\mathrm{student}}]\equiv\arg\max_iJ_i\)，
   “transfer utility 排序”在代数上退化为选择当前回报最高的分支；
2. 继续训练同批 winner 会引入 best-of-\(N\) / order-statistic 效应；
3. 若没有等预算 scratch population，对比无法区分 source transfer 与
   多跑几个 learner 再挑最好者。

因此 `docs/agent_collab/race_then_run_design_20260731.md` 的作废裁决继续有效。
Hurdle stand 对照只检验“run 是否比较差 source 更有训练价值”，不自动恢复
RACE-then-RUN。

### 4.1 核心对象

Transfer utility 不建模为固定标量：

\[
U_i = U(\pi_i, \mathcal{T}, \theta_t, \rho_t, K, d),
\]

其中 \(\theta_t\) 是当前 learner，\(\rho_t\) 是其 occupancy，\(K\) 是
短期干预窗口，\(d\) 是 bootstrap 剂量。

真正的决策对象是同一 learner 上的相对干预结果：

\[
\hat i_t = \arg\max_i J_{\mathrm{sf}}
  \bigl(\mathcal{U}_{K,d}(\theta_t;\pi_i)\bigr),
\]

并显式包含 student-only arm。若 source 相对 student 的证据不足，则输出
exact abstention，而不是强行给出 source。

### 4.2 与现有路线的区别

| 工作 | 核心信号 | 与本项目拟议方法的区别 |
|---|---|---|
| PTF | option value + learned termination | 依赖共享 reward 下的 option/termination 信号；本项目已有信号量级失配证据 |
| QMP | target critic 的 per-state action value | 优化当前行为价值；本项目已观察到它可能安全地退化为 student，却识别不了 delayed learning value |
| JSRL | 预设 guide-policy prefix curriculum | 假设 guide 合理；不解决多个冻结 source 的 learner-specific 选择 |
| RLPD | offline/online 数据的稳定混合 | 说明 prior data 可提高在线 RL；不决定跨任务 source policy 应选谁 |
| Energy-Based Transfer | teacher state-support / OOD familiarity | 控制 teacher 在熟悉状态介入；familiarity 不等于下游 student learning utility |
| Online Source Policy Selection (2017) | multi-armed bandit + policy reuse | 已覆盖“在线选择多个 source policy”；本项目不能把 racing 本身当创新，只能检验 delayed student-response outcome 与 RBO 接口是否有额外价值 |
| R3B（拟议） | 对当前 learner 的短期真实 bootstrap 干预 | 不预测 proxy；直接比较 delayed student learning response |

相关一手来源：

- [QMP, ICLR 2025](https://openreview.net/forum?id=aUZEeb2yvK)
- [JSRL](https://arxiv.org/abs/2204.02372)
- [RLPD](https://proceedings.mlr.press/v202/ball23a.html)
- [Energy-Based Transfer for RL](https://openreview.net/forum?id=Tcm2Mmaw61)
- [An Optimal Online Method of Selecting Source Policies for RL](https://arxiv.org/abs/1709.08201)
- [PTF, IJCAI 2020](https://www.ijcai.org/proceedings/2020/0428.pdf)

### 4.3 路线 A 若成立时可能支持的贡献

**C1 — Target-reward online policy reuse**

把跨任务冻结策略当作 target-domain、reward-bearing 的在线数据生成器，
通过 balanced replay/bootstrap 训练最终 source-free student。这区别于直接动作蒸馏、
纯 offline data 和仅行为 prefix。

**C2 — Transferability as intervention, not proxy**

实证说明在已测试的 HumanoidBench 设置中，行为回报、critic value、梯度、
task/reward structure 等廉价 proxy 不能稳定替代 delayed student utility；
据此把 source selection 改写为低预算 interventional racing。

**C3 — Learner-conditioned and abstaining selection**

将迁移效用视为 learner/stage-conditioned 分布，而非 source-target 固有属性；
当 racing 证据不足时 exact abstain，避免把不稳定绝对符号强制转成选择。

**C4 — HumanoidBench evidence**

在 argmax 不同的多个 target 上验证 source selection 与学习加速，并分别报告
交互成本、并行墙钟、source-free 性能、稳定性和失败边界。

### 4.4 路线 B：Counterfactual Branching Replay（CBR）

该候选不再尝试从即时信号预测 source 的长期学习价值，而是改变 source
经验的生成接口：

1. 主环境始终由 student 行为推进；
2. 从 student 实际访问的 MuJoCo 状态复制 `INTEGRATION` 快照；
3. 辅助环境由冻结 source rollout \(h\) 步；对照环境使用同一个冻结 source，
   但从 target reset distribution 独立采集；
4. 两类辅助 rollout 都产生真实 target reward transition；
5. 主实验采用 frozen source identity（walk/stand）× branch origin
   （independent/student-state fork）的 2×2，对比 state conditioning 是否
   提供超出通用策略多样性的增量，并把所有辅助模拟步计入总 target interaction。

它要检验的不是“哪个 source 的 proxy 分数高”，而是：

> 在固定总模拟交互预算与相同 frozen source 下，从 student occupancy
> 出发的 source 分支经验，是否比从 target reset distribution 独立采集的
> source 经验带来更高的 source-free 学习效率？

这能形成当前架构缺失的真正 `R-only` 对照：source 不接管主轨迹，
但其 target-reward transition 可以进入 replay。现有
`target_evidence_probe.py` 已能用 `mjSTATE_FULLPHYSICS` 运行 matched
branches，不过目前只返回聚合审计量，不记录 transition，也没有接入 replay。
2026-07-31 的复现实测表明 `FULLPHYSICS` 在 Door 上首步 observation 可产生
约 `2.1e-4` 的恢复偏差；改用 `mjSTATE_INTEGRATION` 后，Slide/Door 的同
snapshot、同 100-step action 序列逐 transition 一致。因此若实现路线 B，
必须升级 snapshot contract，不能直接复用旧 probe 的状态定义。

该候选的硬性公平条件：

- 总 interaction 必须包含主轨迹和全部辅助分支；不得只按 main learner step
  报告“样本效率”；
- 必须用两个 frozen source 做 identity × origin 的 2×2，排除
  student-fork 数据冗余与“任何不同策略都行”的平凡解释；
- 第一版预先固定 walk/stand 两个 source 仅用于因果对照，不引入 selector、
  阈值或多 source controller；
- 先只在 Slide 做两个新 learner seed 的最小 feasibility gate；失败即关闭，
  不调 branch horizon 或 replay 比例抢救。

与邻近工作的边界尚需继续核实：Branching RL 研究树状交互模型；
RLPD/REPAINT/Lapse 研究已有或旧策略数据复用；GuDA/HiER 研究数据增强或
高价值 replay。尚未确认有工作在连续控制迁移中，从当前 student 的精确
模拟器状态分叉冻结跨任务策略、用真实 target reward 生成 replay，并以
等交互 student branch 作因果对照。因此当前只能称“有潜力”，不能声称
新颖性已经成立。

## 5. 当前实验的决策作用

### Hurdle selection-value

当前正在运行的 stand 对照直接回答：

> racing 选中的 run 是否优于“随便选一个仍有益的 source”？

- `SELECTION_VALUABLE`：C2 获得必要但非充分的正证据。
- `SELECTION_NULL`：自动选源的实用价值被削弱；不应继续包装 racing，
  应把论文中心退回 RBO/data-interface 与跨任务性能。

### Slide speedup

- `SPEEDUP_CONFIRMED`：获得第二个 argmax 不同 target 的长期性能证据；
  随后才有资格设计端到端 R3B。
- `SPEEDUP_REFUTED`：说明稳定 source ranking 不保证长期加速；
  需要把 R3B 的 horizon/continuation 假设收窄，而不是改阈值抢救。

## 6. 最小后续方法验证

只有 Hurdle selection-value 与 Slide speedup 至少不否定主假设后，才进入：

1. 在同一 learner snapshot 上建立 `{student, source_1,...,source_n}` 分支；
2. 每个 source 使用相同 \(K,d\) 和 source-free evaluator；
3. 以相对排名和 student arm 为基准，选择 winner 或 abstain；
4. 继续训练 winner branch，而不是用同一批结果事后证明自己；
5. 总环境步数必须计入所有 racing arms；并行加速只能另报 wall-clock；
6. 至少一个 target 使用独立 learner 批次验证选择稳定性。

不得声称 K=10k 是通用值；它目前只是 Hurdle 上的经验上界。

## 7. Devil's Advocate Checkpoint 1

### Verdict: REVISE BEFORE METHOD FREEZE

没有发现需要放弃整个 RQ 的 Critical 问题，但有三个 Major 问题必须由当前实验解决。

#### Major 1：racing 的交互成本可能吃掉全部收益

Hurdle 的账面结果是 3×10k=30k 的选择成本换约 67k 的达阈节省。
但这只在单 target、三个候选和特定阈值成立。若候选数增大或 K 上升，
方法可能在 environment-step 指标上失去优势。

**处理**：论文必须同时报告 total interaction、main-learner interaction 和
parallel wall-clock；不得只报墙钟。

#### Major 2：同 learner 的 race 分支并不等于部署时单一路径

若每个 source 分支更新出不同 learner，选 winner 相当于 population selection，
不是在单 learner 上学得一个 selector。它仍可能是有效算法，但定位必须诚实。

**处理**：把方法称为 intervention/racing protocol，不称零成本 transferability metric；
部署对象是获胜 learner branch。

#### Major 3：site selection 与现有证据很薄

Hurdle 和 Slide 都是事先知道存在好 source 的 target。即使两者成功，也只能证明
“在存在可迁移 source 的场地，racing 可选择并利用”，不能证明普遍适用。

**处理**：后续必须包含一个具有真实 abstention 需求、且标签可推广的 target；
在找到这种场地之前，C3 只能是机制能力，不是性能结论。

### Strongest counter-argument

> R3B 只是用 N 倍训练成本跑多个 learner，然后挑最好的；任何超参数搜索都能做到，
> 它既不是迁移指标，也未必比直接增加 scratch seeds 更高效。

### 必须正面回答

后续方法需要与“同预算多 scratch learner 后选优”比较。否则无法证明收益来自
source transfer，而不是 population selection / lucky seed。

## 8. Stage 1 裁决

当前保留两条互斥候选：

1. **路线 A**：RBO 是性能通道；interventional racing 是尚未成立的候选
   选择协议；exact abstention 是安全出口。
2. **路线 B**：固定主轨迹为 student，用公平计费的 counterfactual source
   branch 生成 replay，直接检验 source 数据通道的增量价值。

两条路线都没有方法冻结。下一步顺序为：

1. 先按既有预注册收口 Hurdle selection-value；
2. 再按既有预注册裁决 Slide 长程加速；
3. 路线 A 只有在两项结果均未否定其前提，并加入等预算 scratch-population
   baseline 后才可重写；
4. 路线 B 先完成文献碰撞与只读实现审查，再决定是否值得一个最小 gate；
5. 两条路线都必须报告总 interaction，不能用并行 wall-clock 隐去样本成本。

在上述条件满足前，不实现新的 controller，不启动新的高成本矩阵。
