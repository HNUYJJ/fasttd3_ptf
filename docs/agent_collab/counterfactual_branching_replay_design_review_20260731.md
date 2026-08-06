# Counterfactual Branching Replay：文献碰撞与设计前审查

> 日期：2026-07-31  
> 状态：**REVISE / 尚不可实现或开跑**  
> 范围：只读文献与代码审查；不改变 Hurdle/Slide 已冻结实验，不启动新训练。  
> 工作名：Counterfactual Branching Replay（CBR）；名称未冻结。

## Material Passport

- `material_type`: research design review
- `evidence_scope`: 当前仓库代码、既有 RBO/通道分解/QMP/critic-first 结果、
  2011–2026 一手论文与预印本（预印本单独标注）
- `claim_status`: candidate only
- `cross_model_status`: Claude Opus 只读复核完成，裁决 `REVISE`；三项
  Critical/Major 已由 Codex 独立裁定并纳入本文
- `authoritative_checkout`: `d90d9b1` + 当前工作树（工作树非干净，未用于正式运行）

## 1. 五句立项门

1. **核心问题**：行为/数据双通道迁移——能否利用 source 经验而不让 source
   接管 student 的主 occupancy？
2. **唯一假设**：在固定总 target-simulator interaction 下，从 student 当前
   访问状态分叉出的 source transition，比同预算 student 分支 transition
   更能提高后续 source-free student 性能。
3. **决策作用**：正结果会支持一种新的 RBO 数据生成接口；负结果会关闭
   “source 数据有用、主要伤害来自 behavior takeover”这一方法方向。
4. **与旧实验区别**：Door 只做过 `student / B-only / joint`，没有真正的
   `R-only`；critic-first 仍让 source 接管行为；matched-state probe 只输出
   聚合审计量、不把分支 transition 用于学习。
5. **最低成本证伪**：先做 paired-fork 工程恒等 smoke；通过后只在一个已知正例
   和一个已知负例各做一次 10k→20k、source-branch vs equal-cost
   student-branch feasibility gate。

## 2. 方法对象

### 2.1 不公平的朴素版本（禁止）

```
128 个 main student env 正常训练
+ 额外 source shadow rollout
→ 只按 main step 报样本效率
```

该版本免费增加 simulator interaction；任何收益都可能只是“多拿了数据”。
它不能进入实验。

### 2.2 公平数据接口：paired auxiliary rollout

保持每个 vector step 的总环境 transition 数为 128：

```
64 个 anchor env：始终由 student 推进主 occupancy
64 个 fork env：每 h=25 步从配对 anchor 复制完整环境状态

fork 处理：auxiliary env 每 h=25 步从配对 main env 复制状态
independent 处理：auxiliary env 按普通 target reset 独立推进

两种处理都由同一个冻结 source 驱动 auxiliary env，
并把 64 main + 64 auxiliary transition 写入 replay
```

给定同一个 source \(i\)，起点处理的条件增量为：

\[
\Delta_i
=J_{\mathrm{sf}}(i\text{-fork})
-J_{\mathrm{sf}}(i\text{-independent}).
\]

这样两臂具有相同 main occupancy 数、相同总 simulator interaction、相同更新
次数、相同 auxiliary policy 与 replay 容量；主要差异是 auxiliary rollout
从当前 student state 分叉，还是从 target reset distribution 独立采集。

`h=25` 不是调参结果，而是沿用当前 RBO segment horizon。第一版固定单 source，
不加入 selector、阈值、termination 或多 source bandit。

### 2.3 修订后的 2×2 因果矩阵：source identity × branch origin

Claude 复核正确指出：把 `student-fork` 当主 control 会在 fork 行引入结构性
冗余——main 与 auxiliary 同起点、同 student policy，仅探索噪声不同；而
source-fork 天然具有更高策略多样性。这个偏差不会被原来的 difference-in-
differences 消掉。

因此主矩阵改为两个**冻结 source** × 两种起点，student-fork 只保留为工程
sanity check，不进入主因果 estimand：

| auxiliary frozen policy | independent reset | student-state fork |
|---|---|---|
| `stand`（Slide 上既有中性/无益参照） | `stand-independent` | `stand-fork` |
| `walk`（Slide 上既有正迁移参照） | `walk-independent` | `walk-fork` |

四臂都由 64 个 main student env 与 64 个 auxiliary env 组成；每个 vector step
恰好产生 128 条 target transition、执行相同数量 learner update。主量为：

\[
\Delta_{\mathrm{walk}}
=J(\text{walk-fork})-J(\text{walk-independent}),
\]
\[
\Delta_{\mathrm{stand}}
=J(\text{stand-fork})-J(\text{stand-independent}),
\]
\[
I_{\mathrm{conditioning}}
=\Delta_{\mathrm{walk}}-\Delta_{\mathrm{stand}}.
\]

继续条件必须同时满足：

1. \(\Delta_{\mathrm{walk}}>0\)：student-state fork 确实优于等预算独立 walk 数据；
2. \(I_{\mathrm{conditioning}}>0\)：这个增量显著大于 stand 的通用
   policy-diversity/reset 效应；
3. 结论来自 learner seeds，而不是 episode-level 重采样。

这样不能证明任意 source 都会受益，但可以排除“任何不同于 student 的策略分支
都会赢”这一平凡解释。若只有 `walk-fork > stand-fork`，结果只支持 source
identity 主效应，不支持 state conditioning。

## 3. 它与现有 RBO 的关系

当前 joint RBO 在主环境中用 source action 替换 student action，因此同时改变：

1. main occupancy；
2. replay 中的数据内容。

CBR 把 source 限制在辅助 fork：

- main occupancy 始终由 student 形成；
- source transition 仍携带 target reward，进入 FastTD3 replay；
- 最终评估仍是结构性 source-free student。

因此它不是新的 transferability score，而是一个 **target-state-conditioned
source data generator**。若成立，论文主张应是“改变跨任务 prior policy 的数据接口”，
不是“准确预测 source 的长期迁移性”。

## 4. 与既有失败线的碰撞

| 既有线 | 已回答的问题 | CBR 是否重复 |
|---|---|---|
| T0 / SHU / SIV / P0 | 用短期行为或短训结果预测延迟效用 | 否；CBR 不先预测或裁 source |
| update-space influence | 单步 source update 是否预测长期效用 | 否；CBR 直接训练，主比较是终点反事实 |
| QMP | target critic 是否能逐状态选 source action | 否；CBR 不使用 critic 选源 |
| Door B-only | source behavior、source replay 条件增量 | 部分邻近；CBR 补的是此前不可实现的 true R-only |
| critic-first | source bridge 中冻结 actor 是否减害 | 否；其 behavior takeover 仍存在，且已 SCIENTIFIC FAIL |
| matched-state target-evidence | source/student 短 rollout 的聚合 reward/progress | 工程复用；CBR 不用聚合 probe 作准入判据 |

关键边界：Door 三 seed 的 replay 条件增量方向不一致
（`+41.43/-39.69/+68.42`）。CBR **没有**先验证明一定安全；它必须被设计成
真正可失败的实验。

## 5. 一手文献碰撞

### 邻近但不相同

- [REPAINT, ICML 2021](https://proceedings.mlr.press/v139/tao21a.html)：
  迁移 teacher representation，并用 advantage 选择 teacher experience。
  它最接近“教师样本进入 off-policy learning”，但不是从当前 student 状态
  精确分叉生成 target transition。
- [Branching Reinforcement Learning, ICML 2022](https://proceedings.mlr.press/v162/du22a.html)：
  把同一状态下多个 base action 产生的树状 trajectory 作为新的交互模型，
  主要研究 regret/RFE；不是 frozen cross-task policy reuse。
- [RLPD, ICML 2023](https://proceedings.mlr.press/v202/ball23a.html)：
  研究 prior/offline data 与 online replay 的稳定混合；不生成
  student-state-conditioned source branch。
- [Replay Across Experiments (RaE), ICLR 2024](https://openreview.net/pdf?id=Nf4Lm6fXN8)：
  直接把旧实验经验复用到新实验 replay，并在 locomotion/manipulation 上验证
  bootstrap 与渐近收益。它进一步说明“跨实验 replay reuse”本身已有强结果；
  CBR 不能把跨任务经验进入 replay 作为新颖性，只能检验**在线、当前
  student-state-conditioned 的数据生成**是否优于复用既有/独立 teacher 数据。
- [How to Spend Your Robot Time, IROS 2022](https://arxiv.org/abs/2205.03353)：
  直接研究 target task 上 teacher/student 数据预算的分配，并发现混合两者通常
  最好；teacher 还可在任意 student state 被查询。它是 CBR 在**问题设定和公平
  数据预算**上的强近邻。区别是其 teacher 数据由独立 teacher episodes
  预收集，student-state 查询用于 action supervision；CBR 候选则从在线 student
  snapshot 生成真实动力学 teacher branch，并只把该 branch 用作 off-policy
  transition。若该 state-conditioning 没有胜过等预算 teacher episode 混合，
  CBR 的方法增量很弱。
- [DAgger, AISTATS 2011](https://proceedings.mlr.press/v15/ross11a.html)：
  在 student 自己诱导的状态分布上查询 expert action，并聚合为监督数据；
  它已经建立了“student-state-conditioned teacher query”这一基本思想。
  CBR 与它的区别不能写成“在 student 状态调用 teacher”，只能写成：
  **从同一 simulator state 真实执行 source 多步动力学分支、取得 target
  reward transition，并只用于 off-policy RL，而不是 expert action label 或
  imitation loss**。若实验最终只证明 action label/蒸馏也能达到同样效果，
  CBR 的增量不成立。
- [Replayed-Prefix On-Policy Distillation (ReOPD), 2026
  preprint](https://arxiv.org/abs/2607.04763)：
  研究对象是多轮 LLM agent 蒸馏，不是机器人连续控制或 replay-based RL；
  因而不构成 CBR 的直接方法先例。但其“two-sided distribution shift”指出了
  CBR 必须显式面对的概念风险：让 teacher 分支更贴近 student occupancy，
  同时可能把 frozen teacher 查询到其训练支持域之外、降低 teacher
  reliability。CBR 的 student-state conditioning 因此不是单调优势；
  2×2 interaction 若为负，不能只解释为 replay 机制失败，也可能是
  occupancy relevance 与 source reliability 的此消彼长。该文仅作风险建模
  参照，不作为机器人迁移方法的新颖性依据。
- [Branching Policy Optimization](https://arxiv.org/abs/2607.14171) 与
  [Process-Scorer Guided Adaptive Tree Rollout](https://arxiv.org/abs/2607.15610)
  （均为 2026 preprint）：
  在可 snapshot 的 agent sandbox 中从共享 prefix 分叉 rollout，并分别利用
  sibling return 或过程分数提高采样/credit assignment 效率。它们不是
  cross-task robot teacher replay，但进一步说明“精确快照 + 等预算树状
  rollout”已是独立算法方向；CBR 不能把分叉拓扑或共享 prefix 本身算作
  新颖性，必须证明 frozen source branch 对 FastTD3 replay 的特定增量。
- [Contextual Policy Transfer, UAI 2021](https://proceedings.mlr.press/v161/gimelfarb21a.html)
  与 [Energy-Based Transfer, ICLR 2026 submission](https://openreview.net/forum?id=yx00QWzKEF)：
  前者在目标 transition 上学习 source dynamics 的 state-dependent belief，
  后者用 teacher state-visitation density/OOD familiar states 控制 advice；
  二者说明“source 在当前状态是否可靠”本身已有直接方法路线。CBR 第一版
  不得把 fork-state familiarity 包装为新 selector，也不因某个 source
  在 student state 可执行就默认可靠。若 CBR gate 失败，不能事后加入
  dynamics/energy threshold 抢救；那会同时改变数据接口和选源机制，必须另立假设。
- [Guarded Policy Optimization / TS2C, ICLR 2023](https://arxiv.org/abs/2303.01728)：
  teacher 在 student 在线轨迹中基于 trajectory value 介入，并把 demonstration
  用于 off-policy learning；这覆盖了“任意质量 teacher + student-state
  intervention + replay”的大部分问题设定。CBR 的剩余边界是 shadow branch
  **不改变主轨迹**、不依赖 intervention value gate、且以 exact student-fork
  作等交互反事实。若去掉 shadow branch 后 TS2C 式 shared control 同样有效，
  则不能把安全性或教师复用本身算作 CBR 的贡献。
- [REBOOT, CoRL 2023](https://proceedings.mlr.press/v229/hu23a/hu23a.pdf)：
  用跨任务/对象旧数据初始化 replay，证明 prior data 可提高真实机器人样本效率；
  数据是既有静态数据，不是当前 learner occupancy 上的在线分支。
- [IBRL, RSS 2024](https://arxiv.org/abs/2311.02198)：
  让 imitation policy 为在线探索和 target bootstrap 提议备选动作，再由 critic
  选择；CBR 不以当前 critic 判断 source action，也不把 source action放进主轨迹。
- [SnapshotRL, 2024](https://arxiv.org/abs/2403.00673)：
  从 teacher trajectory snapshot 重置 student，再由 student 采集；
  CBR 的方向相反——从 student snapshot 出发，让 teacher 仅生成 replay 数据。
- [LiDER, Neural Computing and Applications 2021](https://arxiv.org/abs/2009.13736)：
  重置到 replay 中的历史状态，再由当前 policy 重走并用更好的新 experience
  刷新 replay。它已覆盖“精确/近似状态重置 + 分支 rollout + replay
  refresh”这一算法骨架。CBR 的候选差异仅是**锚点来自当前 student occupancy，
  分支策略是 frozen cross-task source，并有等成本 student-fork 对照**；
  simulator reset、dreaming rollout 或 replay refresh 本身都不是新贡献。
- [The Power of Resets in Online RL, 2024](https://research.google/pubs/the-power-of-resets-in-online-reinforcement-learning/)：
  从理论上研究 local simulator access，即重置到已访问状态并沿动力学继续。
  这使 CBR 必须明确标注 simulator/local-reset 假设；“可从当前状态分叉”不是
  普适机器人接口，也不能作为自身贡献。
- [SR², IJCAI 2025](https://www.ijcai.org/proceedings/2025/970)：
  从离线轨迹中选择状态，重置到不完美 simulator 后继续探索，并把离线数据与
  simulator rollout 混合训练 sub-policy。它说明“可重置状态 + 分支采集 +
  mixed replay”这一大框架已有明确先例；CBR 若有增量，只能来自
  **当前 student occupancy 上的 frozen cross-task policy 分支、source
  不接管主轨迹，以及 equal-cost student-fork 反事实对照**，不能把
  simulator branching 本身写成新贡献。
- [GuDA, RLC 2024](https://openreview.net/forum?id=rtJmC83c0r)：
  人工规则引导的 offline trajectory augmentation；不是在线跨任务 source fork。
- [ACAMDA, AAAI 2024](https://ojs.aaai.org/index.php/AAAI/article/view/29442)
  与 [CAIAC, 2024](https://arxiv.org/abs/2405.18917)：
  都生成 counterfactual RL transition，但依赖学习到的因果模型或对
  action-unaffected state factors 的交换。CBR 使用真实 simulator dynamics，
  因此不承担 model-bias 问题；反过来，它承担更强的 local-reset/simulator
  access 假设，不能泛称“counterfactual augmentation”创新。
- [Lapse, ICML 2025](https://proceedings.mlr.press/v267/zhang25ck.html)：
  在 state-evolution 场景复用旧 policy 与旧 experience，并自动集成；
  问题设定和数据生成机制均不同。

### 新颖性结论边界

本轮限定检索中，**未找到完全相同的组合**：“在连续控制迁移中，从当前
student 的精确 simulator state 分叉 frozen cross-task source、取得真实 target
reward transition、只用于 replay，并以 equal-interaction student branch 为因果
对照”。但 *How to Spend Your Robot Time* 已研究 teacher/student target-task
数据预算，RaE 已覆盖跨实验 replay reuse，DAgger/TS2C 已覆盖
student-state teacher query/intervention，LiDER、Branching RL、SnapshotRL
与 SR² 又分别覆盖“历史状态重放”“同状态分支”
“teacher snapshot reset”和“trajectory-state revisit + simulator
re-exploration + mixed replay”的大部分构件。因此当前新颖性只是
**student-state-conditioned frozen cross-task teacher dynamics branch +
main-trajectory isolation + equal-cost student-fork** 这一组合级候选差异，
不是已确立的算法原语新颖性。

这不是全领域不存在的证明。正式写 novelty 前还需补：

- policy reuse / demonstrations / DAgger / teacher intervention / Dyna /
  reset-based exploration 的引文追溯；
- 上述论文（尤其 SnapshotRL、SR²）的引用与被引文献的 backward/forward
  snowball；
- 代码层面确认 SnapshotRL、REPAINT 是否含等价变体。

正式方法实验若进入设计，还必须加入一个简单 teacher-episode mixed replay
参照；否则即使 CBR 胜过 student-fork，也无法证明“从 student state 分叉”比已有
teacher/student 数据预算分配多提供了什么。

## 6. 当前代码可行性审查

### 已有可复用部分

- `target_evidence_probe.py::_capture/_restore` 已用
  `mjtState.mjSTATE_FULLPHYSICS` 捕获与恢复物理状态；但它只能作为旧 probe
  的代码起点，不能直接充当 CBR 的 exact-branch contract；
- source observation/action adapter、冻结 source bank 已存在；
- `PTFReplayWrapper.extend` 已能写 transition、option id 和完整 provenance；
- source-free evaluator、anchor、admission audit 已存在。

### 阻塞性缺口

1. 训练使用 `SubprocVecEnv`。SB3 已提供 `env_method` worker RPC，因此不必修改
   SB3 worker 协议；但项目 wrapper 还没有可序列化的
   `capture_branch_state()/restore_branch_state()` 环境方法。
2. 2026-07-31 的真实环境恢复探针发现：
   `FULLPHYSICS + elapsed_steps` **不是逐 transition 精确的**。固定四个 seed、
   同一 snapshot、同一 100 步 action 序列重复两次时，Slide 首步最大
   observation 偏差约 `3.0e-13`，Door 达 `2.1e-4`；后者足以否定把现有
   probe 直接称作 exact counterfactual。根因是 `FULLPHYSICS` 不包含
   `mjSTATE_WARMSTART/CTRL/QFRC_APPLIED/XFRC_APPLIED` 等 integration state。
   改用 `mjtState.mjSTATE_INTEGRATION` 后，同一测试在 Slide 与 Door 的
   observation/reward/info/terminated/truncated **全部逐值一致**。
   因而 gate task 的最小 contract 必须是：
   `mjSTATE_INTEGRATION + TimeLimit._elapsed_steps +
   env.unwrapped.np_random.bit_generator.state + worker np.random state`。
   静态实例审计确认 Slide 与 Door 的 `task.__dict__` 除
   `robot/_env/unwrapped` 引用外没有额外动态字段；这不能外推到全部
   HumanoidBench task，运行时若 task 出现未声明动态字段必须拒绝启动。
3. replay 的 `extend` 每次期望 `[n_env]` 一整列；paired fork 必须保持 128 条
   transition/step，不能额外插入 64 条破坏更新与容量口径。
4. 自动 reset、TimeLimit truncation 和 fork 提前终止不能只要求“规则相同”：
   branch policy 会改变 done 率，从而改变真正来自 student snapshot 的有效剂量。
   fork env 在 segment 内提前 done 后，terminal transition 可以写入 replay，
   但 auto-reset 后的 transition **不得**作为 fork 数据继续写入；下一次写入前
   必须从当前配对 main env 重新 capture/restore。逐臂报告 effective segment
   length、early-done 次数、reclone 次数与 post-reset transition share（必须为
   0）。处理剂量不满足预冻结容差时裁 `INCOMPLETE`。
5. fork/independent 两种处理都使用与既有 RBO 一致的 deterministic frozen
   source action；不得只给某一臂额外探索噪声。

## 7. 工程恒等 gate（任何训练前）

先只在 Slide 的**训练同款 wrapper 栈**执行：

1. 递归枚举 wrapper 栈并保存每一层有状态字段，至少包含两层
   `TimeLimit._elapsed_steps`；运行时出现未声明动态字段即拒绝；
2. 用 `mjSTATE_INTEGRATION` 把 worker A 的完整 snapshot 复制到 worker B，
   同时恢复 `env.unwrapped.np_random`、worker `np.random` 与 wrapper state；
3. A/B 施加完全相同的 100 步 action 序列；
4. 检查每一步 observation、reward、terminated、truncated；
5. 至少跨一个 `h=25` segment 边界，并分别覆盖自然 done、内外层 TimeLimit
   较早触发以及 SubprocVecEnv auto-reset 后重新 clone；
6. 若不能逐 transition 一致，则 snapshot contract 不完整，科学实验不得开始。

这里的“逐 transition 一致”直接验证干预身份，和过去无关紧要的全训练逐 bit
审查不同：失败会让所谓 matched fork 根本不是同一反事实起点。

## 8. 唯一最小科学 gate（工程 gate 通过后）

仓库中虽已有 Slide/Door 的 10k pure-student anchor bundle，但其 manifest
分别记录 `git.dirty=True`（历史 HEAD `45e5821` / `5944792`），不能证明与未来
CBR 实现的普通训练语义一致。feasibility 若获准，必须在冻结实现 HEAD 上为
Slide 的两个新 seed 重建 pure-student 10k anchor；不得把历史 dirty anchor
直接升级为方法证据。

最小 gate 使用新建的 Slide 10k pure-student anchors：

- 场地：Slide；
- sources：walk（既有正迁移参照）与 stand（既有中性/无益参照）；
- 两个新 learner seeds；
- `10k→20k`；
- 四臂：`stand-independent / stand-fork / walk-independent / walk-fork`；
- source-free deterministic 128-episode panel；
- 记录总 interaction、fork transition 数、有效 segment 长度、done/reclone 次数、
  post-reset transition share 与 replay provenance。

继续条件：

1. `walk-fork > walk-independent`；
2. \(I_{\mathrm{conditioning}}>0\)，且方向在两个 learner seed 一致；
3. 所有处理身份与剂量 gate 通过。

任一失败：`CBR_FEASIBILITY_FAIL`，不调 `h`、fork fraction、replay mass 或 source
抢救。两个 seed 只用于 feasibility，不能进入论文主结果。

Door 从 feasibility 主判据移除：CBR 第一版没有 selector，Door 对未知有害 source
只能提供安全性/失败边界，不能检验正向数据接口是否有效；若 Slide gate 存活，
Door 可作为后续独立安全性实验，不能反过来作为“方法有效”的证据。

## 9. Devil's Advocate Checkpoint

### Verdict: REVISE

没有足够证据 `CLOSE`，因为 true R-only 从未被实现；也不能
`PROCEED_TO_DESIGN_FREEZE`，因为有三个 Major 问题：

1. **Novelty collision**：与 DAgger、TS2C、LiDER、SnapshotRL、SR²、
   REPAINT、REBOOT 的组合边界尚未完成 snowball 检索；现阶段只能声称
   “未找到完全相同组合”，不能声称“branching replay 是新的”。
2. **Snapshot completeness**：`mjSTATE_INTEGRATION` 已在 Slide/Door 的固定
   action 100-step 探针上恢复逐 transition 一致，但这只验证了两个 task；
   正式实现仍需对 task/wrapper 动态字段做运行时白名单与拒绝策略，不能把
   MuJoCo integration state 自动等同于所有 HumanoidBench task 的完整 MDP
   state。
3. **Learner-path dependence**：Door 已表明 replay 效应跨 seed 反向；因此
   feasibility 已改为两个新 learner seed，正式结论仍必须来自上述 2×2 的
   多 seed interaction，而不是某一臂的绝对提升。

### Strongest counter-argument

> CBR 只是把 128 个独立 student 环境中的一半换成相关的 teacher rollout。
> 若它变好，可能只是 teacher 提高了 replay 的奖励密度；若它变差，可能只是
> 有效 student occupancy 减半。它既没有自动选源，也无法迁移到真实机器人，
> 只是 simulator-only data augmentation。

必须用 source identity × branch origin 的 2×2 主矩阵消除
student-fork redundancy；student/scratch 与 joint RBO 只作外部既有基线。
贡献仍须严格限制为 simulator-enabled cross-task policy reuse。

## 10. 当前裁决

**REVISE，不实现、不运行。**

先完成已在执行的 Hurdle selection-value 与已冻结的 Slide speedup。等待期间只做：

1. 文献 snowball；
2. 完整 snapshot contract 的静态设计；
3. Claude Opus 只读交叉复核（已完成，裁决 `REVISE`）。

只有复核后仍不出现 Critical 问题，才起草真正 run card。

## 11. Claude 交叉复核与最终裁定

Claude 的三项阻塞意见均成立并已纳入：

1. 原 `student-fork` 主 control 存在结构性冗余偏差；
2. source-dependent early termination 会稀释真正的 fork dose；
3. 旧 snapshot 探针没有覆盖训练栈中的双层 TimeLimit、worker RNG 与
   SubprocVecEnv auto-reset。

但不直接采用其“Slide 三臂 student/mismatched/source fork”建议，因为该设计
仍只能比较 fork policy identity，不能识别 student-state fork 相对独立 source
数据的增量。本文采用更严格但仍最小的 frozen-source 2×2。

**最终状态仍为 `REVISE`**：科学设计已收敛，但在 Hurdle selection-value 与
Slide speedup 两个优先实验闭环前不实现 CBR。之后若启动，只允许按 §7–§8
冻结一次 run card；工程 gate 或科学 gate 任一失败即停止。
