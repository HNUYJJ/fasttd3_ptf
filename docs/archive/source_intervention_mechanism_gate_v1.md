# Source Intervention Value Mechanism Gate v1

日期：2026-07-11  
状态：**单 learner-seed 正式机制门已完成；Engineering Go，Feasibility Stop**  
当前阶段：已裁决 `STOP_COMPLEX_ESTIMATOR`；不扩任务、不扩源、不启动三种子、低成本 estimator 或闭环分配。

> 本文取代 `stage_conditioned_replay_data_value_probe_v1.md` 作为下一项正式工作协议。
> 旧文保留为思路演化记录。关键修正是：不再默认 source 的价值等同于 source transition
> 的 replay value，而是先区分 reachability/collection 与 replay/update 两条通道及其交互。

---

## 0. 战略决定

当前不执行以下路线：

- 不继续 cabinet dose grid；
- 不依据即时/shaped return 做 winner-take-all；
- 不搜索 scratch 失败、transfer 成功的“杀手任务”；
- 不增加 source library、horizon arms、阈值或 replay floor；
- 不把昂贵的分叉训练结果直接称为可部署 transferability estimator。

当前唯一目标是回答：

> 在固定 target learner 阶段，run-composite intervention 的收益究竟来自
> （1）把 student 带到不同的后续状态并改变其后续数据，
> （2）run-composite transition 本身更适合更新 learner，
> 还是（3）两者的非加性交互？

该问题的结果会决定后续论文是 replay-value、reachability bridge、segment-level coupled
intervention，还是停止高复杂度 estimator 路线。

---

## 1. 当前证据起点与必须修正的效度边界

### 1.1 仍可保留的方向性证据

- warmup bootstrap 是当前主要性能通道，后期 gate/distillation 的独立正面增益不足；
- maze 存在与更长 survival 方向相反的 early hard-progress gain；
- cabinet 中 run、stand、WFix 的后续学习结果明显不同；
- P2 的 shaped return 与 hard progress 可以反向；
- basketball 表明 posture/stability 改善不保证目标成功；
- 现有证据主要支持 early sample efficiency，不支持稳定 asymptotic ceiling gain。

这些事实支持继续研究 source intervention，但不识别 intervention 通过哪条通道生效。

### 1.2 新发现的 seed plumbing 漏洞

当前 HumanoidBench wrapper 和旧 stability audit 使用：

`env.unwrapped.seed(seed)`

但 `HumanoidEnv.seed()` 只调用全局 `np.random.seed()`；Gymnasium reset 随机性实际来自
`self.np_random`。本地复核：

- 两个 fresh cabinet env 使用旧方式设置相同 seed 后，首次 reset observation
  `max_abs_diff ≈ 0.0194`；
- 使用 `env.reset(seed=123)` 后，`max_abs_diff = 0`。

因此，P0/P1/P2 中“相同 eval seed/rank 的精确 episode pairing”不成立。条件均值、训练种子
方向和大效应仍可作为方向性证据，但以下旧表述必须收窄：

- episode-paired delta；
- 配对 t 值；
- “完全相同初态下”的反事实措辞。

新 probe 的任何数据采集前必须先修复 seed plumbing，并通过 reset/step parity test。

### 1.3 Cabinet episode length 不能排除 stability

`Cabinet.get_terminated()` 只在四个 subtasks 全部完成后终止；跌倒不会触发 early termination。
因此 cabinet 中所有条件 `episode_length=1000`、`early_failure=0`，不能证明条件具有相同姿态
稳定性，也不能单独排除 stability-mediated progress。

旧审计已经记录 `stand_reward/small_control` composite，但 run 与 stand 的 stability composite
并不相同。新 probe 必须直接记录 head height、torso upright、lying/fallen fraction 和控制诊断，
不能继续用 episode length 代替稳定性。

### 1.4 旧 10k checkpoint 不是完整 learner state

现有 `save_ptf_params()` 不保存或不能忠实恢复：

- actor/Q optimizer；
- LR scheduler、AMP scaler；
- replay tensor、pointer 与真实 behavior provenance；
- Python/NumPy/Torch CPU/CUDA RNG；
- reward normalizer；
- environment state；
- 训练数据采样日程。

resume 路径还会创建新 replay、新 optimizer 和新环境。因此已有 10k `.pt` 只能用于 policy
评估或 surrogate collector，不能作为 paper-grade learner fork。必须重新训练一个带完整 anchor
bundle 的 scratch-10k learner。

---

## 2. 中心命题、estimand 与 estimator

### 2.1 当前候选 thesis

> **We identify the learner-relative value of cross-task modular policy interventions by
> separating how they alter reachable experience from how their transitions alter learning.**

中文：

> **我们研究跨任务模块化源策略干预对当前目标学习器的条件化价值，并区分它通过改变可达的
> 后续经验和通过改变 learner update 两条通道产生的作用。**

这是一项待验证命题，不是当前已成立的论文贡献。

### 2.2 完整 learner state

令 target learner 在阶段 `t` 的状态为：

`L_t = (theta_actor, theta_Q, theta_Q_target, optimizers, schedulers, scaler,
normalizers, B_t, RNG, global_step)`。

`B_t` 是完整有效 replay；source value 不是 source policy 自身属性，而依赖 `L_t`、target、
action-composition mask、dose、segment horizon、follow-up horizon、update operator 和 outcome。

### 2.3 总干预价值

实际在线系统希望预测的是：

`SIV_s(L_t; d, h, f, K, U, P)`

即在给定 dose `d`、source segment `h`、student follow-up `f`、更新窗口 `K`、FastTD3 update
operator `U` 和 source-free hard outcome `P` 下，source intervention 相对 student intervention
造成的目标学习差异。

暂不除以 transition 数 `N`。逐 transition 归一化会暗含线性剂量与可加性，而 P2 尚未支持
单调剂量机制。

### 2.4 Oracle 与可部署估计器必须分开

- `SIV/DV oracle`：真的构造各个 counterfactual learner branch、更新并评估后得到的因果标签；
- `hat(SIV)`：在分配 source budget 前或用远低于完整训练的 micro-probe 成本即可获得的预测；
- hard progress：oracle 的 outcome，不是 estimator；
- `T0`、return、critic advantage、TD、gradient similarity、coverage：候选 proxy/baseline。

本轮 2×2 只测 oracle mechanism labels。即使成功，也还没有产生可部署的新算法。

---

## 3. 共享 potential-trajectory bank 的 2×2 设计

### 3.1 Treatment 语义

当前只使用：

- target：`h1hand-cabinet-v0`；
- learner stage：fresh scratch 10k；
- source：run；
- composed action：run 替换 `legs_torso,arms`，hands 始终由 frozen 10k student 控制；
- source segment：`h=25`；
- frozen-student follow-up：`f=25`。

论文和代码均必须写 `run-composite` 或 `source-conditioned composed behavior`，不能写成
“完整 run 教师控制”。

### 3.2 Paired potential trajectories

对每个共同 simulator anchor `x_j`，冻结 10k student 和 normalizer，分别生成：

- `Z_j^S`：student 从 `x_j` 执行 25 步；
- `Z_j^R`：run-composite 从同一 `x_j` 执行 25 步；
- `F_j^S`：从 `Z_j^S` 终点由同一 frozen student 再执行 25 步；
- `F_j^R`：从 `Z_j^R` 终点由同一 frozen student 再执行 25 步。

在四条 potential path 全部收集完成前禁止 learner update。这样 source/student 的行为差异不会
通过更新后的 learner 反向污染 collector。

### 3.3 四个数据分支

四个 learner branch 都从逐字节相同的 `L_t` 和 `D0=B_t` 开始：

| reachability factor `B` | prefix replay factor `R` | branch dataset | 解释 |
|---|---|---|---|
| student | student | `D0 + Z^S + F^S` | `00` baseline |
| run | student | `D0 + Z^S + F^R` | `10` bridge/reachability-only |
| student | run | `D0 + Z^R + F^S` | `01` replay/update-only |
| run | run | `D0 + Z^R + F^R` | `11` full local intervention |

`10` 中 learner 不看 run transition，只看 run 把 frozen student 带到的新终点之后产生的数据；
`01` 中 learner 看 run transition，但 follow-up occupancy 固定为 student baseline。

当前 replay 是 `n_steps=1`，所以两个有效 transition pool 的交叉组合不要求它们在 replay 中保持
物理连续。若未来使用 n-step/sequence replay，本设计不能直接复用。

### 3.4 不能接受的伪 2×2

以下实现均不具有所声明的因果含义：

- 将实际 run transition 只改 behavior label 后称为 student data；
- run 执行后从无关 student rollout 随机补数据；
- 四个 cells 使用不同 reset distribution；
- collection 与 learner update 交错；
- source/student cells 的 termination 被静默丢弃；
- 四个 cells 使用不同 batch indices、target noise 或 update counts。

---

## 4. Factorial contrasts

令 `mu_br` 为 branch `br` 在 source-free independent evaluation 上的 hard-progress 均值：

`B0 = mu_10 - mu_00`：bridge-only simple effect。  
`R0 = mu_01 - mu_00`：replay-only simple effect。  
`I  = mu_11 - mu_10 - mu_01 + mu_00`：interaction。  
`T  = mu_11 - mu_00 = B0 + R0 + I`：完整 local intervention contrast。

同时报告 averaged main effects：

`B_bar = 0.5 * [(mu_10 - mu_00) + (mu_11 - mu_01)]`

`R_bar = 0.5 * [(mu_01 - mu_00) + (mu_11 - mu_10)]`

若 `I` 较大，不把 averaged main effect 当作可独立部署的机制；应承认 source value 是
segment 与 follow-up 的 coupled sequence-level effect。

本设计识别的是固定 stage/frozen collector 下的 **local data mechanism**，不等于完整在线 RBO
的长期 total effect。真实在线系统还含 learner update→future behavior→future data 的反馈环。

---

## 5. Paper-grade anchor 与 simulator parity

### 5.1 Scratch-10k learner anchor

重新训练 `empty source bank + MCG off` 的 scratch learner，并在 10k 保存：

- actor、Q、target Q；
- actor/Q optimizer；
- actor/Q scheduler；
- AMP scaler；
- obs/reward normalizer；
- replay valid chronological slice、ptr、schema；
- 每条 transition 的 `behavior_id`；
- Python/NumPy/Torch CPU/CUDA RNG；
- `global_step`、完整 args/PTF config；
- git head、dirty status、配置和代码 hash。

cabinet 128 env、10k stage 的有效 replay 约 2.35 GiB；只保存 valid chronological slice，
不保存约 12 GiB 的空 capacity。

该 bundle 只支持当前 scratch/non-MCG probe；不在本轮泛化实现任意 MCG 训练的完全 resume。

### 5.2 Simulator anchor bank

优先实现 SubprocVecEnv state RPC，保存/恢复：

- MuJoCo full-physics/integration state；
- cabinet `current_subtask`；
- `TimeLimit._elapsed_steps`；
- environment `np_random.bit_generator.state`；
- observation 与必要 task mutable state。

若 full state RPC 成本或兼容性不可接受，可使用相同 `reset(seed=...)` + deterministic student prefix
重建 anchor，但必须同时验证：

- full observation parity；
- qpos/qvel parity；
- `current_subtask` parity；
- 固定 probe action 下的 next_obs/reward/done parity。

任一 parity 不通过即停止完整 2×2，不允许把 matched-distribution donor 冒充逐 anchor counterfactual。

### 5.3 Anchor-state 分布（采集前锁定）

状态 bank 不是“挑选看起来有希望的接近柜子状态”。它按
[`cabinet_source_intervention_2x2_v1.yaml`](../configs/experiments/cabinet_source_intervention_2x2_v1.yaml)
固定为：

- 从 reset seed `410000` 起按升序取候选 seed，另预留 128 个 replacement seeds；
- 每个候选 seed 的 prefix length 由独立 PCG64 seed `20260711` 在 `[0,950)` 均匀产生；
- frozen 10k student 使用独立 noise seed `20260712` 执行 prefix；探索尺度使用
  anchor actor 的 `noise_scales[anchor_id mod 128]`；
- 得到的是 frozen 10k student 在 cabinet episode 前 950 步的均匀 time-slice occupancy，
  不是原训练进程中某个 in-flight state 的恢复；
- 若到 anchor 前已 termination/truncation，或任一 `student/run-composite` 50 步 potential path
  termination/truncation，则整个 pair 失效；按候选 seed 升序用 reserve 补齐，直至 512 个有效 pair；
- 失效原因和所有被替换 seed 必须保留，不能根据结果人工筛 state。

路径的显式 exploration noise 使用 seed `20260713`；同一 `(anchor,relative_step)` 在两个 potential
paths 中复用同一噪声，run-composite 只在当前状态 student action 的 `legs_torso,arms` 维度上
替换 source action。

---

## 6. Seed 修复与旧结果边界

正式 collection 前必须完成：

1. `HumanoidBenchEnv` 接收并使用 `args.seed`；
2. 通过 SB3 `envs.seed(seed)` 的 next-reset 语义，或在 worker 中使用 Gymnasium
   `reset(seed=seed+rank)`；
3. stability audit 的 eval factory 不再调用无效的 `env.unwrapped.seed()`；
4. 添加 same-seed/different-seed reset parity test；
5. 添加 same-anchor fixed-action next-step parity test。

P0/P1/P2 暂不因该问题自动重跑。它们继续作为条件均值和方向性背景，但新论文材料不得沿用
“精确 episode pairing”及其配对显著性措辞。只有当后续核心 claim 需要这些旧数值时，才按修复后
协议重新评估相应最小子集。

---

## 7. 数据预算、固定 sampler 与更新协议

### 7.1 单 learner-seed feasibility 配置

- learner seed：1；
- anchor states：`J=512`；
- 每个 anchor 两条 potential path，各 `25+25` steps；
- 总物理 transition：`2 × 512 × 50 = 51,200`；
- 每个 cell 的逻辑 candidate data：`12,800 Z + 12,800 F = 25,600`；
- 四个 cells 共享同一 paired trajectory bank，不重复采集环境数据。
- 所有数值常量与 artifact 路径以
  [`cabinet_source_intervention_2x2_v1.yaml`](../configs/experiments/cabinet_source_intervention_2x2_v1.yaml)
  为机器可读真值。

### 7.2 Batch composition

每个 branch 的每次 critic batch 固定为：

- 50% `D0`；
- 25% `Z`；
- 25% `F`。

四个 cells 使用逐 update 预生成并共享的 `(pool, anchor_id, step_id)` sampling schedule。相同位置
只替换 treatment 对应的 transition 内容。actor update 使用同一 branch batch，保持 symmetric
sampling；当前 probe 不研究 actor/critic split replay。

该 50/25/25 是显式 probe dose，不声称等同自然 online uniform replay。

### 7.3 Updates

- critic updates：400；
- actor updates：200，严格复现 `policy_frequency=2`；
- target soft update：每 critic update 后一次；
- evaluation checkpoints：K=0/100/200/400 critic updates；
- primary endpoint：K=400；
- update-curve AUC：辅助；
- obs/reward normalizer：probe update 阶段冻结；
- target-policy smoothing noise：四 cells 使用相同预生成 noise；
- branch 执行顺序：随机化并记录；
- 四 cells 在同型号 GPU 上顺序执行，避免并行 resource contention。

### 7.4 Controls

- `no-update`：验证评估器漂移；
- `D0-only`：同样 K 次更新但没有 candidate data；
- duplicate `00`：相同 snapshot、sampler 和 noise 的完全重复；
- K=0 四 cells：参数和评估必须一致；
- extra-student/equal-dose 已由 `00` 的 `Z^S+F^S` 提供；
- 所有 evaluation source-free。

---

## 8. 指标

### 8.1 主要 hard-progress outcome

主连续指标直接读取第一阶段 pulling-door joint：

`P_door = max_k clip(abs(q_pull_door(k)) / 0.4, 0, 1)`。

不使用 shaped total return 作为主终点。

确认指标：

- `success_subtasks >= 1`；
- `max(success_subtasks)`；
- door fraction 首次达到 0.25/0.50/0.95 的时间；
- door fraction ≥0.25/0.50 的持续步数比例，区分偶然碰撞与稳定打开。

### 8.2 Stability/safety

- head height；
- torso upright；
- stand reward；
- lying/fallen time fraction；
- control/actuator diagnostic；
- termination/truncation 和发生阶段。

### 8.3 Reachability/behavior diagnostics

- root 到 cabinet/door 的水平距离与朝向；
- hand–handle proxy distance；
- hand–door contact；
- door joint displacement；
- `Z^R` 与 `Z^S` endpoint state distance；
- `F^R` 与 `F^S` 的 task-state coverage；
- source segment 与 follow-up 的 reward/return，仅作诊断。

其中 hand-contact 是按 MuJoCo contact geom 名称得到的 proxy，不把它写成已验证的语义接触标签。

如果 run-composite 没有产生可检测的 endpoint/occupancy treatment，不能把 `B0≈0` 解释成
“reachability 没有学习价值”；该结果是 treatment failure。

---

## 9. 统计单位与推断边界

- 当前单 learner seed 只用于工程与信号筛选，不形成科研结论；
- 512 anchors、injection data seeds 和 eval episodes 都不是独立 learner repeats；
- future scientific screen 的单位是独立 scratch-10k learner seed；
- 若进入三种子，先在每个 learner seed 内聚合 paired eval，再跨 learner seeds 汇总；
- 三种子仍只作机制筛选；论文确认是否补到 5 seeds 由观察到的 seed-level variance 做 power analysis；
- eval seed bank 与 anchor/data seed bank 必须完全不重叠。
- 本轮 source-free evaluation 固定使用从 `610000` 起的 64 个 seeds；不能在看到结果后替换。

本轮成功最多识别 cabinet/scratch10k/run-composite/h25/f25/固定 dose/FastTD3 operator 下的
oracle channel value，不能外推为通用 transferability。

---

## 10. Go / Stop / Pivot

### 10.1 Engineering Go（全部满足）

- full learner snapshot save/load roundtrip parity；
- replay valid slice、ptr、schema 与 hash parity；
- same-seed reset parity，different-seed 非退化；
- same-anchor obs/qpos/qvel/task-state/next-step parity；
- run 只改变 `legs_torso,arms`，hands 与 student action 逐维相同；
- 四 cells 的 D0、data count、sampling slots、updates、noise 完全一致；
- `Z^R` 在两个 R=run cells 中 hash 相同；`F^R` 在两个 B=run cells 中 hash 相同；
- K=0 四 cells 一致；
- duplicate `00` 参数或评估差异在预设 numerical tolerance 内；
- 无 eval-seed/data-seed 泄漏。

任一项失败：不启动正式 K=400 branch。

### 10.2 单 learner-seed feasibility Go

只有同时满足以下条件，才建议进入多 learner-seed：

1. run treatment 在 endpoint/occupancy diagnostic 上可检测；
2. `B0`、`R0` 或 `I` 至少一个在主 door fraction 上满足
   `abs(delta) >= max(0.10, 3 × duplicate-branch noise)`；
3. hard threshold 指标方向不与主连续指标强烈矛盾；
4. 结果不完全由 posture collapse 或 differential termination 解释；
5. full local intervention `T` 不是稳定的大幅负向。

该门仅决定是否值得做多种子，不证明机制成立。禁止根据哪个 cell 最好再改主指标、K、h、f 或 dose。

### 10.3 路线裁决

- `R0>0, B0≈0, I≈0`：进入 replay/update value，研究低成本 `hat(DV)`，必须对比 VER/IIF；
- `B0>0, R0≈0`：转向 reachability/occupancy credit，以 JSRL 为强基线；
- `B0,R0>0` 且 interaction 小：双通道 source intervention；
- `I` 主导：放弃逐 transition value，研究 segment+follow-up sequence value；
- 只有 `T` 有效：定位为 coupled intervention，不拆成可独立模块；
- 全部小于实践阈值或主要方向不稳定：停止复杂 estimator，收口 RBO early-transfer empirical route；
- treatment diagnostic 失败：不是科学零效应，先判 treatment/collector 不适合。

即使多种子 oracle 成立，也必须再提出计算成本显著低于完整分叉的 `hat(SIV)`，验证 held-out
source-target-stage 预测性后，才允许启动闭环 allocation。

---

## 11. 相关工作与新颖性边界

以下对象已经存在，不能声称首次：

- source/current policy guided exploration 与按 return/UCB 选源：PRQL、OPS-TL、MAPSE；
- guide policy 改变 starting-state curriculum：JSRL；
- target critic 选 source/current policy：CUP、APT-RL；
- option-level source selection/termination 与 adaptive transfer：PTF；
- prior/offline/cross-experiment replay bootstrap：RaE、RLPD、Shared Experience Replay；
- experience value 与 learned replay policy：PER、VER、ERO；
- actor/critic 数据需求不一致：Actor-PER；
- stage/current-policy-conditioned learning potential：PLR；
- online RL influence/filtering：A Snapshot of Influence / IIF；
- algorithm-relative trajectory value：Algorithm-Relative Trajectory Valuation；
- multi-policy/primitive composition：MCP、MULTIPOLAR。

关键 primary references：

- PTF: <https://www.ijcai.org/Proceedings/2020/428>
- JSRL: <https://proceedings.mlr.press/v202/uchendu23a.html>
- CUP: <https://proceedings.neurips.cc/paper_files/paper/2022/hash/b09df3a10e26204136540ca59bc5a646-Abstract-Conference.html>
- APT-RL: <https://arxiv.org/abs/2311.06731>
- RaE: <https://arxiv.org/abs/2311.15951>
- RLPD: <https://arxiv.org/abs/2302.02948>
- VER: <https://arxiv.org/abs/2102.03261>
- ERO: <https://arxiv.org/abs/1906.08387>
- PLR: <https://proceedings.mlr.press/v139/jiang21b.html>
- A Snapshot of Influence: <https://proceedings.neurips.cc/paper_files/paper/2025/hash/377d0752059d3d4686aa021b664a25dd-Abstract-Conference.html>
- Algorithm-Relative Trajectory Valuation: <https://arxiv.org/abs/2511.07878>

当前仍可能具有新颖性的组合被严格限制为：

> **cross-task modular source-policy intervention + paired equal-budget identification of
> reachability and update channels + learner/stage/effect conditioning + a practical
> pre-intervention total-value estimator + held-out closed-loop allocation.**

本轮 2×2 只可能完成其中的机制识别部分，不能单独支撑完整方法贡献。

### 11.1 后续 estimator 的最低 proxy baselines

- target rollout return / `T0`；
- return-UCB/PRQL-style allocation；
- CUP/APT critic advantage；
- PER `abs(TD)`；
- VER `abs(TD) × on-policyness`；
- IIF/gradient-similarity influence；
- occupancy/reachability features；
- random/uniform；
- post-hoc oracle ranking。

### 11.2 后续闭环的最低方法 baselines

- scratch/student-only；
- uniform RBO；
- static WFix；
- best fixed source；
- return-weighted/UCB；
- critic-guided reuse；
- oracle source allocation；
- reachability route需加 JSRL-style roll-in；
- replay route需加 RLPD-style symmetric mixing；
- MCG claim需比较 full-action source / modular source / no source。

---

## 12. 实现工作包与启动顺序（已全部执行）

1. [x] 修复 HumanoidBench training/eval seed plumbing并添加 parity tests；
2. [x] 新增 compact replay snapshot/restore 与 full learner anchor bundle；
3. [x] 新增 seeded simulator anchor-state RPC 与 reset/state parity；
4. [x] 新增 paired `Z^S/Z^R/F^S/F^R` collector 与 behavior provenance；
5. [x] 新增 three-pool fixed sampler 与 deterministic FastTD3 update kernel；
6. [x] 新增 no-update/D0-only/duplicate-00/equal-dose/hash tests；
7. [x] 先执行少量 anchors 的 smoke，不看科研方向；
8. [x] 全部 Engineering Go 后，执行单 learner-seed `J=512,K=400` feasibility；
9. [x] 生成机器可读 JSON、Markdown 报告和 Go/Stop 裁决；
10. [x] 未经新的明确裁决，不自动扩到三 learner-seed；正式裁决为不扩展。

---

## 13. 当前允许与禁止的 claim

### 当前允许

- source intervention 的 downstream learning utility 尚未被 return/stability 校准；
- reachability 与 replay/update 是需要区分的候选通道；
- 2×2 是机制审计，旨在决定后续研究对象；
- 当前主要证据是 early acceleration，非 ceiling gain。

### 当前禁止

- source as data generator 是我们的首创；
- stage-conditioned data value 是我们的首创；
- cabinet 提升已经排除了 stability mediation；
- P0/P1/P2 是精确相同初态的 episode-paired causal evidence；
- run 已被证明产生 approach/interaction states；
- replay contamination/active toxicity 已证实；
- `DV` 已经是一个在线可计算 metric；
- MCG 已有独立性能贡献；
- 当前方法自动避免 harmful reuse；
- 一个 cabinet 机制实验即可验证通用 transferability estimator。

---

## 14. 正式执行、结果与路线裁决（2026-07-11）

### 14.1 完成性与工程效度

本协议已经完整执行，不再是计划文档。正式 artifact 为：

- paper-grade learner anchor：
  [`cabinet_scratch_s1_step10000`](../artifacts/anchors/cabinet_scratch_s1_step10000/)；
- simulator state bank：
  [`state_bank.pt`](../artifacts/mechanism_gate/cabinet_s1/state_bank.pt)；
- paired potential-trajectory bank：
  [`trajectory_bank.pt`](../artifacts/mechanism_gate/cabinet_s1/trajectory_bank.pt)；
- 全部 branch 原始评估：
  [`results.json`](../artifacts/mechanism_gate/cabinet_s1/results.json)；
- 机器可读门控与简表：
  [`gate_report.json`](../artifacts/mechanism_gate/cabinet_s1/gate_report.json)、
  [`gate_report.md`](../artifacts/mechanism_gate/cabinet_s1/gate_report.md)。

正式完成审计如下：

- anchor 为 scratch cabinet seed 1、10,000 vector steps、128 env，含 1,280,000 transitions；
  replay `ptr=valid_size=10,000`，per-transition behavior provenance 全部为 student；
- `learner.pt/replay.pt/rng.pt` SHA-256 分别为
  `4c3adf65...`、`dec81e21...`、`32efce03...`，重新验签通过；
- state bank 640/640 有效，seed 为 `410000..410639`，640 个 state digest 全部唯一；
  prefix length 与 PCG64 seed `20260711` 逐项一致，所有状态与观测有限；
- trajectory bank 接受前 512 个 anchors、拒绝 0 个，student/run 两条 path 起始观测逐元素相同，
  50 步连续性误差为 0，termination/truncation 均为 0；
- trajectory content digest 为
  `c442bc0568f7000a39b1ef8e68e7612ef89be1ebc1f0c6debaddd6115ca43c67`，独立复算一致；
- 正式 branch 顺序为 `10 → 00 → 11 → d0_only → 01 → duplicate00`；六个更新分支均为
  400 critic / 200 actor / 200 scheduler steps，schedule digest 唯一值相同；
- 七个 K=0 initial model digest 和 source-free evaluation 全部相同；eval seeds 精确为
  `610000..610063`，与 state/data seeds 无交集；
- `00` 与 `duplicate00` 的最终 actor/critic/target digest、K=0/100/200/400 全部 episode JSON、
  update logs 均逐字节一致，duplicate noise 为 0；
- 六个 K=400 checkpoint 的模型 hash 均与 `results.json` 一致，模型、optimizer 与结果无 NaN/Inf；
- 全仓测试重新运行：`218 passed, 11 warnings`；bash 与 Python syntax checks 通过。

因此正式 rerun 的 **Engineering Go = true**。

#### 首轮 CUDA 非确定性工程失败

第一轮六分支虽然 seeds、sampler、target noise、K=0 与预算都一致，但 `00` 和 `duplicate00`
在 K=400 的 max-door mean 相差 `0.06437`，最终模型 digest 也不同，故 Engineering Go=false。
直接 reproducer 定位到未锁定运行时下官方 distributional projection 的 CUDA 浮点归约：同一输入
可产生约 `1e-9` 的差异，经过 400 次更新与非线性动力学后放大。修复为：

- `torch.use_deterministic_algorithms(True)`；
- `CUBLAS_WORKSPACE_CONFIG=:4096:8`；
- cuDNN deterministic、benchmark off；
- CUDA matmul/cuDNN TF32 off。

修复后的真实 CUDA smoke 和正式 duplicate 均精确复现。首轮结果完整封存于
[`failed_nondeterministic_v0`](../artifacts/mechanism_gate/cabinet_s1/failed_nondeterministic_v0/)，
不得与正式 rerun 合并或用于科研结论。

### 14.2 Treatment 是否真的改变状态

是。run-composite 干预不是“没有执行成功”的空处理：

- 25 步 endpoint observation L2：median `61.94`，mean `62.81`；
- endpoint changed fraction (`L2>0.05`)：`1.00`；
- 被替换动作组 action L2 mean：`5.85`。

但 endpoint 的语义变化主要是姿态/几何，而非 cabinet hard progress：

- head height：`+0.2371`；
- torso upright：`+0.0978`；
- lying proxy：`-0.0254`；
- hand-to-pulling-door distance：`-0.1211`（更近）；
- root-to-cabinet distance：`+0.0737`（更远）；
- door fraction：仅 `+0.00036`；
- completed subtask：`0` 差异。

因此 treatment detectable，但它首先改变的是 locomotion/posture-conditioned reachable state，
不是已经完成 cabinet skill 的 source behavior。

### 14.3 主结果

K=400 source-free max-door mean 与归一化 K=0/100/200/400 curve AUC：

| cell | K=400 max door | normalized AUC |
|---|---:|---:|
| `00` student prefix + student follow-up | 0.100380 | 0.076971 |
| `10` student prefix + run-conditioned follow-up | 0.049315 | 0.065408 |
| `01` run prefix + student-baseline follow-up | 0.070014 | 0.071491 |
| `11` run prefix + run-conditioned follow-up | 0.052022 | 0.054091 |

| estimand | K=400 | normalized AUC |
|---|---:|---:|
| `B0` reachability/bridge-only | -0.051065 | -0.011563 |
| `R0` prefix replay/update-only | -0.030366 | -0.005480 |
| `I` interaction | +0.033073 | -0.005836 |
| `T` full local intervention | -0.048357 | -0.022880 |

duplicate noise 为 0，故预注册实践阈值为 `max(0.10,0)=0.10`。`B0/R0/I` 均未过门，
`T` 也为负。K=100 曾出现 `R0=+0.04762` 与 `I=-0.08593`，但仍未过阈值，并在 K=200/400
反向；不能把该短暂波动写成 early positive mechanism。

辅助 controls：

- no-update K=0：`0.053582`；
- D0-only K=400：`0.087383`；
- `00` K=400：`0.100380`。

所以 student candidate data 相对纯 D0 update 只有 `+0.0130` 的小差异；本结果不是简单的
“所有新数据都有大收益，而 run 数据更小”。

### 14.4 姿态稳定与目标技能的分离

`00 → 11` 的完整 local intervention 在 K=400：

| metric | `00` | `11` | `T=11-00` |
|---|---:|---:|---:|
| max door fraction | 0.100380 | 0.052022 | -0.048357 |
| success subtasks | 0.015625 | 0.000000 | -0.015625 |
| door ≥0.25 time fraction | 0.097266 | 0.057844 | -0.039422 |
| door ≥0.50 time fraction | 0.043250 | 0.014781 | -0.028469 |
| head height mean | 0.299707 | 0.355420 | +0.055713 |
| torso upright mean | 0.045510 | 0.114220 | +0.068710 |
| lying fraction | 0.942734 | 0.948625 | +0.005891 |
| min hand-to-door distance | 0.244697 | 0.252156 | +0.007459 |

更强的分离出现在 `01`：它显著提高 head/upright、降低 lying fraction、让 root/hand 更接近，
但 max-door 仍比 `00` 低 `0.03037`，door≥0.25 time fraction 低 `0.06597`，且没有 subtask
success。即使不把 lying proxy 当作完美 fall label，多个 posture/geometry 指标与 hard task progress
方向分离。

这支持一个窄而重要的解释：在该 learner stage 和 action composition 下，run source 的主要内容是
**稳定性/姿态/可达域调制**，没有提供足够的 cabinet manipulation skill；稳定得更好或更接近目标
并不自动转化为目标任务学习价值。这与用户对现有增益来源的担忧一致。

### 14.5 预注册裁决

- Engineering Go：**true**；
- treatment detectable：**true**；
- mechanism signal above `0.10`：**false**；
- feasibility Go to multi-seed：**false**；
- route：**`STOP_COMPLEX_ESTIMATOR`**。

立即停止的工作：

- 不把该 oracle 2×2 扩到 3/5 learner seeds；
- 不训练 `hat(SIV)`、DV、gradient/value estimator；
- 不启动 held-out source-target-stage prediction 或 closed-loop allocation；
- 不通过更换 K、h、dose、source、task 或主指标寻找过门结果；
- 不据此提出 winner-take-all/阶段最优教师注入算法。

### 14.6 支持、不支持与必须收窄的 claim

**当前支持：**

- 已建立一个可复现的 learner-relative、paired、equal-budget source-intervention mechanism assay；
- run-composite 明确改变了 downstream state distribution；
- 在 cabinet/scratch10k/run/h25/f25/当前 FastTD3 update operator 下，reachability、prefix replay 与
  interaction 均没有达到继续开发复杂 estimator 的实践信号门；
- 姿态/可达性改善与目标 manipulation progress 可以系统性分离。

**当前不支持：**

- 所有 source interventions 都无效或有害；
- run 在所有 stage/task 都没有 transfer value；
- 已经发现可部署的 source transferability metric；
- 多源稀释应由 winner-take-all 自动解决；
- 单 learner seed 可形成统计显著的通用科学结论。

**论文必须收窄：**

- 该 experiment 是机制筛选与负结果，不是新 estimator 的 positive validation；
- 现有主线最多继续声称 bootstrap 带来部分 early acceleration，不能声称 source 已迁移目标技能、
  提升 asymptotic ceiling 或自动识别 harmful source；
- 若论文核心贡献仍要求“有 insight 且 solid”的新机制，不能用更多 cabinet 小消融填补本门未过；
  应回到整体框架层重新选择一个有独立理论对象、可证伪机制和跨条件证据的核心命题。
