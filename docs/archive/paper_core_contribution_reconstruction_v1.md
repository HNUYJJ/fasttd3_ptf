# 整篇论文核心贡献重构 v1：从“迁移性估计”转向“暂时性行为脚手架”

日期：2026-07-11  
状态：**历史战略版本；已依次由v2与当前v3取代，保留用于审计决策演化**  
适用证据截止：`source_intervention_mechanism_gate_v1` 正式裁决之后

> 2026-07-12 PI 裁决：TBS 保留为机制解释，但不再单独承担整篇迁移强化学习论文的方法中心；
> source admission、student exact fallback 与 replay control 曾在v2恢复为主方法问题；SHU失败和
> 后续全局结果裁决后，当前版本见
> [`paper_core_contribution_reconstruction_v3.md`](paper_core_contribution_reconstruction_v3.md)。

---

## 0. 总裁决

论文不再以以下对象为核心：

- 一个统一的 scalar transferability `T`；
- source return / target critic / replay value 的组合评分；
- `DV`、`SIV` 或低成本 learning-value estimator；
- winner-take-all、阶段最优 source 或更多 source library；
- MCG、student-as-arm、replay weighting 等组件清单。

正式机制门已经证明：在 cabinet/scratch10k/run-composite/h25/f25 的受控条件下，
reachability、prefix replay 和 interaction 都没有达到继续开发复杂 estimator 的实践信号门。
因此，ChatGPT-5.5/5.6-Pro 共同建议的 data-value → prediction → closed-loop 路线已经完成其
生死检验，并被正式停止。

新的论文核心问题改为：

> **What do cross-task source policies actually provide to a target humanoid learner?**

当前最能闭合全部证据的回答是：

> **Cross-task policies act primarily as transient behavior scaffolds during the cold-start phase.
> They need not contain a target-task solution; by producing target-reward experience under viable
> whole-body behavior, they can accelerate acquisition of target-specific progress. Their benefit is
> stage-limited and bottleneck-dependent, and can turn into task interference when the scaffold
> occupies action authority needed by the target skill.**

中文：

> **跨任务源策略的主要作用不是把目标任务技能教给学生，而是在冷启动阶段暂时提供可行的全身行为
> 脚手架，使目标学习器获得带目标奖励的训练经验并更早学到任务特异行为。该收益只在特定阶段和
> 瓶颈下成立；当脚手架占用了目标技能所需的动作控制权时，它也会转化为任务干扰。**

工作标题首选：

> **Beyond Skill Transfer: Transient Behavioral Scaffolding for Humanoid Reinforcement Learning**

备选：

- *Source Policies as Scaffolds: Understanding Cross-Task Reuse in Humanoid RL*；
- *When Priors Help—and When They Interfere: Behavioral Scaffolding in Humanoid RL*。

---

## 1. 为什么必须重构

### 1.1 旧中心对象已经被证据否决

| 旧候选中心 | 当前证据 | 裁决 |
|---|---|---|
| zero-shot / `T^0` transferability | crawl false positive、powerlift false negative、maze/truck/cabinet 排序与收益错位 | 只作 coarse diagnostic |
| online return | cabinet return 与 hard progress 可反向；stair 对 horizon 盲 | 不作 learning-value metric |
| target critic advantage | pole false negative，早期 OOD/sign calibration 不可靠 | baseline/diagnostic |
| `DV/SIV` data value | 正式 2×2 中 `B0=-0.0511,R0=-0.0304,I=+0.0331`，均低于 0.10 门 | `STOP_COMPLEX_ESTIMATOR` |
| reward-weighted source selection | terrain 有用，但 breadth 多任务与 uniform 打平；大源库可稀释 | supporting heuristic |
| MCG gate/distillation | `bootstrap_only≈full`，`no_bootstrap` 近零或负 | safety/supporting component |
| horizon-arm | 3-seed 总账为负，扩大 arm 空间产生探索税与方差 | appendix negative result |
| stage-best / WTA | run24 vs WFix 方向混合；没有 active toxicity 证据 | 删除 |

### 1.2 当前真正稳定的证据链

1. `bootstrap_only≈full`：主要性能通道是 source/student 行为产生的 target-reward experience，
   不是蒸馏或后期 gate。
2. cabinet、maze、hurdle 存在去除 survival 混淆后仍成立的早期 hard progress；所以不只是 alive
   reward。
3. run 明显优于 stand：泛化的“站稳就够了”不成立，源行为形态仍然重要。
4. powerlift 几乎没有技能收益，basketball 中姿态更稳定但成功更低：稳定性不是充分条件。
5. 100k 时多项差距收敛：当前证据支持 early sample efficiency，不支持稳定 ceiling gain。
6. 10k mechanism gate 中 source intervention 强烈改变状态，却没有后续学习价值：至少说明 source
   value 不能假定为 stage-invariant，并直接推动后续 stage-locality gate。

这六点共同支持“行为脚手架”的重构，并提示其可能是暂时性的；stage-locality 仍需 §9 的直接检验。
它们不支持“目标技能教师”或“通用 transferability estimator”。

---

## 2. 候选论文路线比较

| 路线 | insight | 当前证据 | 方法新颖性 | 风险 | 决策 |
|---|---:|---:|---:|---:|---|
| A. Data-centric transferability estimator | 高 | 低 | 中 | 正式机制门未过 | **停止** |
| B. Reward-weighted policy selection | 中低 | 中 | 低 | 与 PTF/MAB/PER 重叠，跨任务不稳定 | 降级 |
| C. Transient behavioral scaffolding | 高 | **高** | 中低 | 与 JSRL/SFP 相近 | **选为科学主线** |
| D. Effect-preserving modular scaffolding | 高 | 低 | 中高 | 当前 warmup 实际为整动作 source | **只保留一个升级门** |
| E. 纯 benchmark/negative-result taxonomy | 中 | 高 | 无方法 | 容易被视为经验报告 | 作为 C 的贡献 3 |

路线 C 是唯一同时满足以下条件的选择：

- 能解释已有正、负结果；
- 接受 early acceleration 而不虚构 ceiling；
- 不依赖已经失败的 estimator；
- 能把“主要收益来自稳定性”从弱点转化为可检验机制；
- 即使后续方法升级门失败，仍可形成诚实的 CCF-B 级机制论文。

---

## 3. 核心对象的正式定义

### 3.1 Target learner 与 source scaffold

令目标 MDP 为 `M_T=(S,A,P,r_T,γ)`，目标 learner state 为 `L_t`，冻结的跨任务 source
集合为 `Π_S={π_s}`。目标 actor 为 `π_θ`。

在 scaffold window `t<W` 内，行为策略不是要被蒸馏的 teacher，而是：

`μ_t(a|x) = (1-p_t)π_θ(a|x) + p_t Σ_s w_s π_s(a|x)`，

其中 source/student 按 segment 锁存执行，所有 transition 都由目标环境奖励 `r_T` 标记并进入
同一 off-policy replay。`t≥W` 后 `p_t=0`，source 完全撤出；最终评估也必须 source-free。

当前代码中的 `bootstrap_only + safe_bootstrap/WFix` 是该对象的工程实例。论文不把 softmax weight、
h25 或 warmup=30k 本身称为理论贡献。

### 3.2 三个必须分开的效果

对 hard task-progress 指标 `P_T` 定义：

1. **即时 viability effect**：source 执行期对 upright、fall、survival、root motion 的影响；
2. **post-handoff learning effect**：source 撤出后，纯 target actor 的 hard progress 增量；
3. **ceiling effect**：足够训练预算后的最终能力差异。

脚手架假说只要求第 2 项在早期为正；它不要求 source 自己完成目标，也不要求第 3 项为正。

### 3.3 可证伪的量

令 `π_T^scaf(k)` 和 `π_T^scr(k)` 为训练到 `k` 后的 source-free target policies：

`G_scaf(k)=E[P_T(π_T^scaf(k))−P_T(π_T^scr(k))]`。

定义：

- **handoff retention**：source 撤出若干更新后 `G_scaf(k)>0`；
- **stage locality**：等 source 剂量下，cold-start intervention 的 `G_scaf` 大于 delayed
  intervention；
- **skill/scaffold separation**：viability 增益与 hard progress 可以分离，且 target actor 在
  source-free evaluation 中保留任务进展；
- **interference**：viability 改善但 hard progress 或 success 下降。

这些是论文 outcome/estimand，不是部署前 transferability estimator。

---

## 4. 重新组织后的贡献

### Contribution 1：科学发现——跨任务策略复用是暂时性行为脚手架，而非技能克隆

我们系统区分 source competence、viability、hard task progress、post-handoff retention 和 ceiling，
并用 bootstrap/no-bootstrap、source identity、dose control、hard-progress 去混淆与 10k 因果机制门
证明：

- source 不必会做 target task；
- 收益主要通过冷启动行为数据进入 off-policy learner；
- 稳定性可以是必要的 enabling condition，却不是目标技能本身，也不是充分条件；
- cabinet 10k 的 null local effect 表明 source value 不是可默认跨 stage 保持的常数；
- 同一类 scaffold 可在 cabinet/maze 中加速，在 basketball 中干扰目标技能。

这是论文最有 insight 的部分，也是所有方法设计的因果动机。

### Contribution 2：方法——Target-reward Transient Behavioral Scaffolding（TBS）

TBS 是一个刻意简单的 cross-task behavior scaffold：

1. 冻结 source policies，仅作为 target-environment behavior policies；
2. source/student segment-level 交替，保持闭环动作时间一致性；
3. transition 全部使用 target reward，进入 target learner 的统一 replay；
4. source 只在预定 cold-start window 出现，之后完全撤出；
5. 不要求 imitation、transfer loss 或永久 hierarchical controller；
6. 最终 target policy 独立执行，不依赖 source library。

当前 RBO `bootstrap_only` 是 TBS 的实现基础。静态 source weights、safe horizon、student-as-arm、
replay weighting 和 MCG 都降级为实现变体或 safety ablation。

必须诚实承认：TBS 单独的方法新颖性有限。它与 JSRL、PTF、SFP 和 behavior-prior exploration 有
明显重叠；其论文价值来自“humanoid whole-body scaffolding 的机制识别 + hard-progress 证据”，
而不是声称首次用 prior policy 探索。

### Contribution 3：评估与机制图谱——何时 scaffold、何时 interference

建立 source-free、hard-progress-centered 的 humanoid transfer audit：

- task progress 与 shaped return 分开；
- task progress 与 posture/survival 分开；
- early sample efficiency 与 final ceiling 分开；
- source identity、dose、stage 与 action authority 分开；
- 正例、无效例和冲突例使用同一协议。

Source–Target–Effect Map 只作为该审计的结果组织工具，不再声称 ROI predictor 或独立算法。

### 三项贡献的依赖关系

`Contribution 1` 定义为什么需要 scaffold；`Contribution 2` 给出最小干预；`Contribution 3` 检验
scaffold 是否真正转化为 source-free target skill，并划定失败边界。三者不是独立模块清单。

---

## 5. 组件保留、重构、降级与删除

| 现有对象 | 处理 | 新角色 |
|---|---|---|
| FastTD3 | 保留 | target learner backbone |
| PTF source bank | 保留 | frozen behavior scaffold library |
| RBO warmup bootstrap | 重构 | TBS 主方法实例 |
| target reward replay | 强调 | source 与 target task 发生联系的唯一监督通道 |
| segment latch / horizon | 保留 | temporal consistency；不是单独贡献 |
| static WFix / `T^0` | 降级 | scaffold prior / baseline |
| online student-as-arm | 降级 | optional fallback baseline |
| symmetric replay weighting | appendix | 特定 crawl/slide 的安全变体，不作通用机制 |
| MCG post-warmup gate/distill | 降级 | safety/supporting component |
| MCG body-group composition | 未证候选 | 只通过 §8 的升级门决定是否回主文 |
| Transfer Map | 降级 | safe-horizon/configuration diagnostic |
| Source–Target–Effect Map | 保留 | effect audit / result taxonomy |
| `T^critic/T^online/DV/SIV` | 删除主线 | baseline、失败分析或 appendix |
| winner-take-all / stage-best source | 删除 | 无证据且会重启被否决路线 |
| asymptotic improvement | 删除 claim | 当前只主张 early sample efficiency |

---

## 6. 与最相近工作的边界

### PTF

[PTF](https://www.ijcai.org/Proceedings/2020/428) 将 source policies 建模为 options，学习选哪个
source 以及何时终止，并用 transfer loss 优化 target policy。TBS 不声称 option reuse 首创；差异在于
把 source 明确降为暂时 behavior scaffold，用 target-reward replay 学习，并以 source-free hard
progress 检验它是否产生了 target skill。

### JSRL

[JSRL](https://proceedings.mlr.press/v202/uchendu23a.html) 使用 guide policy roll-in 形成 starting-state
curriculum，再由 exploration policy 接管。它是 TBS 最强概念基线。我们的 source policies 是
cross-task、可能无法接近目标甚至会冲突；研究问题不是“一个 reasonable guide 如何靠近 goal”，
而是“whole-body behavior support 何时只提供 viability、何时能转化为 target skill”。若最终实验
不能超出这一差异，不能声称比 JSRL 更一般。

### SFP / behavior priors

[SFP](https://openreview.net/forum?id=qYNfwFCX9a) 已经从简单任务数据学习 state-free temporal priors，
并在 off-policy RL 中动态混合 prior 与 policy 来加速 unseen downstream tasks。
[Behavior Priors](https://www.jmlr.org/papers/v23/20-1038.html) 也系统研究从既有行为提取探索先验。
因此，论文不能声称首次提出“跨任务行为 prior + off-policy exploration”。我们的差异必须落在
humanoid-specific whole-body viability/task interference 的可证伪机制与 effect-preserving authority。

### RaE / prior replay

[Replay across Experiments](https://deepmind.google/research/publications/50575/) 复用旧实验数据；TBS
让 frozen source 在新 target environment 中在线生成带 `r_T` 的 transition。两者数据来源不同，
但都说明“旧计算通过 replay 加速新 learner”本身不是新颖点。

### Teacher–student intervention / shared control

[TS2C](https://openreview.net/forum?id=O5rKg7IRQIO) 已研究 imperfect teacher 的 intervention 与
teacher-student shared control。若升级到 body-group authority，必须对比 full-action、shared-control
和 source-free student，而不能只与 scratch 比。

### 新颖性结论

当前 base route 的新颖性是**科学问题与机制证据**，不是 TBS 算法结构。若目标是更强方法论文，
必须让 §8 的 effect-preserving modular scaffold 通过；否则以机制型 CCF-B 定位最诚实。

---

## 7. 现有证据如何进入论文

### 主文保留

- `bootstrap_only≈full` 与 `no_bootstrap`：行为脚手架是主通道；
- cabinet/maze/hurdle 的 source-free hard-progress early gain；
- cabinet run vs stand：source behavior 形态重要；
- cabinet run24 dose control：低剂量 source 也有 early learning effect；
- powerlift null 与 basketball negative：viability 不充分、存在 interference；
- 10k 2×2 mechanism gate：强 behavior treatment 但无 late-stage learning value；
- 100k 收敛：不主张 ceiling。

### Supporting table / appendix

- terrain/breadth return AUC；
- wfix/obrw/onlineb/replay weighting；
- horizon-arm negative result；
- Transfer Map v1/v2；
- package chain、door、window 等研发过程。

### 删除或改写

- “9/9 全绿”“零代价”“自动避免 harmful reuse”；
- “source-generated data value 可预测”；
- “run 学会了 cabinet skill”；
- “大源库交给 `T^0` 自动管理”；
- “稳定性就是全部增益”或“稳定性与增益无关”两种极端说法。

### 投稿前必须修复的证据债务

旧 P0/P1/P2 evaluation 使用的 `env.unwrapped.seed(seed)` 没有正确播种 Gymnasium `np_random`。
它们的 condition mean 与 training-seed 方向仍可用于路线判断，但旧 episode-paired delta/t 值不能直接
进入论文。若 headline 使用 cabinet/maze/run-vs-stand 等旧 checkpoint，必须用已经修复的 seeded
wrapper 对最小主表重新做 source-free evaluation；这属于效度修复，不得借机换 checkpoint、指标或
任务。正式 10k mechanism gate 的 reset/pairing 已使用新协议，不受该问题影响。

---

## 8. 唯一方法升级门：Effect-Preserving Scaffolding（EPS）

### 8.1 为什么只允许这一项

当前最尖锐的失败是：source 能改善 whole-body posture，却可能占用 manipulation 所需动作控制权。
如果 source 只负责 shared support subspace，而 student 保留 task-specific effectors，可能把
“稳定性脚手架”与“目标技能学习”结构化分开。这直接回应当前核心问题，而不是再优化权重。

候选组合动作：

`a_EPS = M_support ⊙ a_source + (1−M_support) ⊙ a_student`，

其中 manipulation tasks 固定使用 anatomy-defined `M_support=legs_torso`，arms/hands 始终属于
student。mask 不按任务结果调节，不允许事后选择。

### 8.2 工程事实与风险

当前 warmup 的主实现是整动作 source；body-group modularity 主要存在于 warmup 后的 MCG gate。
历史代码注释还记录过“早期学生控制部分关节会破坏 source 全身闭环”的失败风险。因此 EPS 是真正
的新假设，不是现有方法改名。

### 8.3 最小 feasibility gate

只使用两个已有、机制方向相反的 manipulation targets：

- cabinet：已有 early hard-progress 正例；
- basketball：已有 posture 改善但 success 下降的冲突例。

每个 task 使用其现有 WFix source bank 与冻结的 source-allocation 分布。正式 gate 不依赖“相同控制器
seed 会产生相同源序列”这一错误假设：full-action 与 EPS 的 termination 可能不同，现有控制器会在 done
后重抽 source，从而使两条件的 source identity/dose 分叉。为只改变动作权限，必须预生成与 episode
termination 无关的 25-step schedule tape，并让 full 与 EPS 共享同一个 tape、训练预算和 eval seeds；
schedule hash、实际 source/student dose 及每组执行来源必须进入 provenance。比较：

1. scratch/student-only；
2. full-action transient WFix scaffold；
3. EPS：同一个被抽中的 source 只控制 `legs_torso`，student 控制 `arms,hands`；
4. duplicate EPS 短程工程控制（只验证确定性与 provenance，不作为额外科学样本）。

第一阶段只做 1 learner seed 工程/信号 feasibility；只有通过以下门槛才做 3 seeds：

- cabinet：EPS source-free hard progress 至少保留 full scaffold 增益的 80%，或相对 scratch
  绝对提高 ≥0.10；
- basketball：EPS 相比 full scaffold 至少回收 50% 的 success regret，且不比 scratch 低 0.05 以上；
- posture/viability treatment 仍可检测，证明不是简单关掉 source；
- duplicate 精确、schedule hash/equal dose/equal updates、source-free eval 全通过；
- 两任务不能只凭 shaped return 过门。

若 cabinet 不保留增益，或 basketball 不减少冲突：**EPS Stop**。不改 mask、不加新 source、不调
horizon 寻找正例，TBS 保持 base paper 方法，modularity 留 future work。

### 8.4 实现就绪审计（2026-07-11）

- **动作组合已具备。** 当前 `McgBehaviorController` 按 `group_masks` 逐组用 source action 替换
  student action；配置 `mcg_groups=[legs_torso]` 时，0–10 维来自 source，11–60 维保持 student，
  与 EPS 定义一致。
- **训练语义已具备。** `mcg_ablation=bootstrap_only` 会关闭 warmup 后的 critic gate 与蒸馏，因此
  gate 测到的是行为数据脚手架，而不是额外 imitation loss。
- **精确调度尚未具备。** 当前 safe-bootstrap 在 segment 到期或 done 后重抽 source；必须增加固定
  schedule tape，不能只复用 RNG seed。
- **正式 provenance 尚未具备。** 当前 paper-anchor 路径虽分配了
  `behavior_source/source_by_group/executed_group_mask`，训练循环却仍把每条 transition 填成全 student。
  正式 gate 前必须按 canonical groups `(legs_torso, arms, hands)` 写入真实执行来源，并把
  `segment_id/segment_step` 改为 schedule segment 语义。
- **显式 No-Go。** 在 schedule tape、真实 provenance、短程 duplicate 与正确 seeded evaluation
  四项通过前，不得启动 EPS 科学 run。

完整预注册与实现契约见 [`eps_feasibility_gate_v1.md`](eps_feasibility_gate_v1.md)。

### 8.5 通过后的贡献升级

若 EPS 通过，Contribution 2 升级为：

> **Effect-preserving transient scaffolding assigns source authority only to shared embodiment-control
> dimensions while preserving target autonomy over task-specific effectors.**

届时需要补 full-action PTF、JSRL-style roll-in 和 shared-control baseline；否则不能声称结构优越。

---

## 9. 第二个且最后一个机制门：Stage-locality（仅在 EPS 通过后）

用相同 run source steps 比较：

- early scaffold：只在 cold start 注入；
- delayed scaffold：target learner 已训练一段后再注入；
- scratch。

只选 cabinet 一个已知正例，3 learner seeds，主指标为 source-free hard progress；source steps、更新、
replay dose 和 evaluation seeds 必须相同。

支持条件：early−delayed ≥0.10 且 3/3 seeds 同向，同时 early gain 在 source 撤出后保持。若不满足，
“cold-start-specific scaffold”必须收窄为普通 behavior prior，论文不再强调 stage-locality。

本实验不与 EPS 并行，不扩 task，不做 handoff-time grid。

---

## 10. Baseline 与指标

### 必须 baseline

- scratch FastTD3；
- original/full-action PTF；
- uniform source bootstrap；
- current static TBS/WFix；
- JSRL-style guide roll-in；
- 若 EPS 通过：full-action scaffold 与 anatomy-matched modular scaffold。

SFP/behavior-prior 方法若因 observation/action prior 训练接口不可直接复现，应作为最相近相关工作，
并提供明确的不可比原因；不能假装不存在。

### 主指标

- source-free task-specific hard progress；
- time-to-first-meaningful-progress；
- fixed-budget progress AUC；
- post-handoff retention；
- final/ceiling difference作为边界而非必须正向。

### 机制指标

- posture/upright/fall/survival；
- target bottleneck progress；
- source/student realized action authority 与 transition dose；
- source 关闭前后曲线；
- task-effect conflict，例如 basketball success、cabinet door progress。

### 禁止替代

- shaped return 不能替代 hard progress；
- episode length 不能替代 posture/safety；
- eval episode 不能伪装 learner seeds；
- 单任务正例不能支撑通用 transferability。

---

## 11. 论文结构

### Introduction

1. humanoid RL 冷启动同时面对高维控制、viability 和目标探索；
2. policy transfer 通常把 source 当作 teacher/guide；
3. 我们发现 source competence、stability 和 target learning progress 系统性错位；
4. 提出 transient behavioral scaffolding 视角；
5. 用 hard-progress 与 post-handoff audit 划定何时帮助、何时冲突。

### Method

- problem formulation；
- TBS behavior mixture、segment latch、target-reward replay、withdrawal；
- 若过门，再加入 EPS authority mask；
- FastTD3 update 本身不改，不把 backbone 包装成贡献。

### Experiments

1. channel ablation：bootstrap vs distillation/gate；
2. scaffold → target progress：cabinet/maze/hurdle；
3. viability ≠ skill：powerlift/basketball；
4. source identity/dose：run/stand/run24/WFix；
5. stage-locality：10k mechanism gate + 可选 early/delayed；
6. broad sample-efficiency table；
7. failures/limits。

### 主图建议

- Figure 1：teacher transfer vs behavior scaffold 的概念图；
- Figure 2：source execution → target-reward replay → source-free target actor；
- Figure 3：hard progress、posture、source withdrawal 三条时间线；
- Figure 4：cabinet positive / powerlift null / basketball interference 三联图。

### 主表建议

- Table 1：核心 hard-progress + time-to-progress，而非只放 return AUC；
- Table 2：bootstrap/full/no-bootstrap channel ablation；
- Table 3：stability/task-progress separation；
- Table 4：与 PTF/JSRL/uniform/TBS/EPS 的对比（EPS 通过后才生成）。

---

## 12. Claim contract

### 可以写

- source policies can scaffold early target learning without containing a target-task solution；
- behavior bootstrap, rather than post-warmup distillation, is the dominant observed channel；
- stability/viability can enable learning but is neither sufficient nor universally beneficial；
- current gains are primarily early sample-efficiency gains；
- hard-progress and source-free evaluation are necessary to distinguish scaffold from skill transfer。

### 不能写

- TBS 是第一个 prior-guided exploration 方法；
- source-generated data 已被证明具有可预测 learning value；
- 方法自动选择正确 source 或避免所有负迁移；
- locomotion policy 教会了 manipulation skill；
- MCG modularity 已有独立贡献；
- 方法提高 asymptotic ceiling；
- 一个 cabinet 结果能推广到全部 humanoid transfer。

---

## 13. 执行顺序与资源纪律

1. 先冻结本文档并让 PI 审核 thesis/贡献结构；
2. 不启动新任务、不扩 source、不修改主指标；
3. 只实现 EPS 最小 feasibility 所需的 anatomy mask、完整 provenance 和 duplicate control；
4. 先做 smoke，再做两个 target 的单 learner-seed gate；
5. EPS 不过立即停止方法升级，转论文成稿；
6. EPS 通过才做 3 seeds；
7. 只有 EPS 多种子通过，才允许 stage-locality gate；
8. 不再返回 `DV/SIV/WTA/horizon-arm` 路线。

该顺序的目标不是寻找最好数字，而是用最多两个实验决定：论文是“机制型 behavioral scaffolding”
还是能升级为“effect-preserving modular scaffolding”方法论文。
