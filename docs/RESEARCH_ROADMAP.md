# 科研路线（RESEARCH ROADMAP）

> 更新：2026-07-16（本文档由散落的历史战略/handoff 文档整合而来；原始文档全部在
> [`docs/archive/`](archive/README.md)，本文引用处均给出原件链接）
>
> **硬约束（PI 多次确认）**：基座算法锁定 **FastTD3**，创新只能长在 **PTF**（Policy
> Transfer Framework）框架内，benchmark 为 **HumanoidBench (h1hand)**。不允许中途
> 更换 backbone，也不做第二 backbone 的"可移植性验证"。目标 ICML。
>
> **执行宪章（PI 2026-07-21 冻结）**：所有新机制、代码修改、审查和实验在启动前
> 必须通过 [`RESEARCH_EXECUTION_GUARDRAILS_20260721.md`](RESEARCH_EXECUTION_GUARDRAILS_20260721.md)
> 的立项门。不得让反复审计、逐 bit 等价、小数阈值追调或无主假设实验挤占核心机制
> 与困难任务验证；不能直接推进“自动选教师/终止、双通道迁移或困难任务性能”的工作停止。

---

## 1. 总目标（原始路线，未放弃）

> **迁移性评估 → 自动选择教师 → bootstrap → T-weighted replay → MCG**

正式重构为六组件统一框架（[archive/paper_core_contribution_reconstruction_v2.md](archive/paper_core_contribution_reconstruction_v2.md)），
中心问题：

> Humanoid learner 能否在训练过程中识别哪个跨任务 source **当前**真正有利，
> 只在有利时允许其改变数据分布，并在 source 失效后同时消除行为与 replay 残留影响？

| 组件 | 内容 | 状态（2026-07-16） |
|---|---|---|
| ① Stage-conditioned handoff utility | 相对当前 student、随训练阶段变化的迁移性指标 | **未解决——当前核心断点**（行为 reward 信号族已三重否定，见 §4） |
| ② Student-inclusive conservative admission | student 一等候选、无固定 0.5 teacher floor、可全退 | 基础设施已建成并验证（exact abstention + student-inclusive categorical） |
| ③ Quarantine probe | probe 轨迹隔离，不污染主 replay | 已实现 |
| ④ Reward-bearing bootstrap | 准入源在目标环境短 segment 执行、target-reward 轨迹入 replay | 已实现并验证（性能主通道） |
| ⑤ Admission-consistent replay | 撤销同步退出行为/replay/critic exposure | 已实现并验证（authority-coupled handoff + 原子撤销） |
| ⑥ Optional MCG/EPS authority | 通过准入后再决定 full-action / body-group authority | 后移待命，未删除 |

**版本关系**：v2 = 原始科研目标（active）；
[v3](archive/paper_core_contribution_reconstruction_v3.md) = SHU 失败后的**保底收缩方案**
（静态 RBO + lifecycle 四贡献），只有组件①最终攻不下来才启用——**不能把 v3 说成
默认终点或原计划完成态**（PI 2026-07-15 明确纠正）。

---

## 2. 路线演化时间线

### Phase 0：起步与 PTF 接线（2026-05 中 → 05-29）

官方 FastTD3 + HumanoidBench 接线；PTF（Yang et al. 2020）机制移植到 TD3 backbone
（masked action distillation + β termination + λ 线性衰减 + null option）。
push 目标上 Force-PTF / Decay-PTF 诊断（3v3 t=−0.34 无显著差异），β-clamp 修复
（β sigmoid 饱和 → clamp [0.05, 0.95]）。
**结论**：PTF 机制健康，但 stand/walk/run/reach→push 无显著正迁移——问题不在机制
实现，在"源-目标可迁移性"本身。

### Phase 1：表征统一路线（2026-05-30 → 06-08，全系判 null）

假设"跨任务 obs 不统一是迁移瓶颈"，依次尝试：
- **entity encoder**（entity-token + hypernetwork 共享 obs 前端；~2× 吞吐税）；
- **z-native source**（step-2：冻结共享 E 作 adapter；A=−386 / A2=−271，A2 消融
  决定性——冻结单任务 E 本身就是坏前端）；
- **anchored readout**（proprio-anchored cross-attention 池化；v1≈−425、v2-c 仍
  −359~−486，vs scratch +472）；
- **ED-SF**（SF/GPI value 迁移独立线；push transfer 1M 步全程负 ≈−489，w_task 爆炸）。

**框架复盘**（06-07，[archive/step2_framework_review.md](archive/step2_framework_review.md)）：
失败在 L1（技能重叠低）/L2（冻结表征不可适配），不在 L3（池化方式）——
**"表征统一 ≠ 技能可迁移"**。方向从表征层重定位到迁移性/选源层。
（文献扫描 06-06 同期完成：problem 坐实、CARE/SkillBlender 撞车定性。）
本次代码整理（2026-07-16）已删除该路线全部代码，结论存档于
[archive/step2_research_briefing.md](archive/step2_research_briefing.md)。

### Phase 2：MCG 与 package 专项（2026-06-10 → 06-12，判负但产出关键机制洞察）

- 06-10 repo 大整理 + **PTF 主线硬约束确立**；
- **MCG**（Modular Critic-Guided：option=(教师,身体组)，critic gating Δ）：三任务
  pilot 暴露 window −153 负迁移；SC-MCG（显著性校准 gate）把伤害压到 −80——
  **"critic-gated transfer 的安全性取决于 Δ 显著性而非符号"**；
- **package 主攻**三轮全 0%：oracle 链探针上界 0~9%（approach 卡死+推飞黑洞）、
  chain warmup 能展开链但 eval 仍 0%（TD 长程 credit assignment 瓶颈）——
  **"状态覆盖 ≠ 回报事件"**。package 定为 hard case，专项停止。

### Phase 3：RBO 主线成型（2026-06-13 → 07-01）

- **RIC v1 宽 pilot**（06-13）：项目至今最强正结果（hurdle ROI +71%、cabinet +53%、
  powerlift +42%，regret 0），但机制诊断首次暴露主增益来自 **warmup bootstrap**
  而非 gate；
- **核心 ablation**（06-14）：bootstrap_only≈full、no_bootstrap≈0——
  **bootstrap 是主性能通道，gate 只是安全阀**；TransferMap 跨任务 ROI 预测判
  ill-posed（ρ=−0.22）；方法定调 **RBO**（Reward-Bearing Option Transfer）；
- **terrain 核心三方**（06-15~）：stair/slide/pole/crawl，safe>rand 10/12
  （t≈2.58）；crawl 全翻转（safe<rand<scr）= abstain 的黄金动机；
- **wfix 解耦 3-seed 定论**（06-26）：**源选择 +77.9（11/12，t=3.08）是稳健主因**；
  horizon 中性任务依赖（−11.4，t=−0.46）。主方法定名
  **reward_weighted_bootstrap**（weighted 源 + h25）。

### Phase 4：Transferability 统一框架 + 广度（2026-07-02 → 07-05）

导师意见（07-02，[archive/advisor_feedback_analysis_20260702.md](archive/advisor_feedback_analysis_20260702.md)）：
统一为"度量 T + 三处使用"（选源 / student-as-arm / replay 加权）。执行结果：
- **Step A** online_bootstrap（student-as-arm）+ **Step B** obrw（replay 按 T 降权）：
  crawl +94.7（t=2.54）稳健、slide obrw 决定性大胜（+92.1，t=14.7）；
- T-gated abstain 失败（弱信号下二值阈值脆弱）；mh horizon-arm 3-seed 否决
  （arm 空间扩张的结构性代价）；
- **breadth 三批 + wfix 裁决**：广度 3/3 显著（maze/truck/cabinet），但 wfix 与
  obrw 全线打平 → **主方法简化为静态 RBO，OBRW 降为局部扩展**；
  第二批 std9：powerlift +77.6（t=14.78）、basketball 3/3 负（负迁移 hard case）；
- **stability-deconfounded audit P0/P1/P2**：cabinet run>stand 3/3（源身份决定数据
  价值）；return 与 hard progress 方向错位 → **execution return 不可作在线选源信号**
  （T^critic 方向的直接动机）。

### Phase 5：Admission Core 与 lifecycle 定局（2026-07-06 → 07-13，ChatGPT 接管周 + 我接手审查）

- **SIV 2×2 机制门失败**（07-11）：per-source 因果打分 T=−0.048 < 0.10 →
  论文重构 v1（TBS 中心）；
- **SHU gate 失败**（07-12）：cabinet mandatory regression 接受了应拒绝的
  run-composite → STOP_CLOSED_LOOP → v2→v3 降级。关键病理：**SHU 只测
  behavior/handoff utility，不能替代 replay/update data utility**；
- **Admission Core v1**：student-inclusive categorical、exact abstention、
  quarantine、runtime revocation、provenance replay 全部落地；basketball
  exact-none 安全门 PASS；powerlift 30k 加速显著（t=4.318）但 100k retention FAIL；
- **我接手审查**（07-13）：诊断出 retention FAIL 的两层真相——
  (a) **repetition divergence**（fixed quota 在源退役期 oversample 43×，80k 崩点
  三 seed 同崩）；(b) powerlift **headroom 耗尽**是 retention 败因主体（勿再拿
  powerlift 裁 retention，正确场地是 truck）；
- **authority-coupled physical handoff 修复**：6/6 + 4/4 全 PASS，中介预测命中
  （30k→60k source critic share 预测 33.7%/实测 33.65%）。
  **论文贡献③定稿 = provenance-consistent source data lifecycle**。

### Phase 6：Adaptive revocation（2026-07-14 → 07-15，预注册 FAIL，高信息量负结果）

时间维弃权自动化（stage-window segment 级 UCB/LCB 单向撤销）。18/18 runs 完成，
预注册裁决 **FAIL**：crawl 收益 gate 未达（+41.5/−66.8/+53.9）；truck 禁撤 gate
3/3 违反（hurdle 被误撤，代价 −120/−205）；powerlift 保持 PASS（9k 精确撤
crawl/reach）；basketball 大量触发但无系统性改善。
详见 [archive/adaptive_admission_v1_results.md](archive/adaptive_admission_v1_results.md)。

---

## 3. 当前坐标（2026-07-28，PI 正式收束）

**组件①（learned cross-task scalar transferability metric）降为未来工作**，不再作为
当前论文必须补齐的组件。这不是放弃迁移强化学习，而是承认：一个**便宜、万能、注入前的
单一迁移分数**这个更强也更天真的目标，已被系统性证据否定。

- **已攻下**：组件②③④⑤（执行与生命周期机制）全部建成并经干预实验验证；
  性能主通道 = 静态 RBO；安全与审计通道 = admission lifecycle。
- **组件①的真实状态**：被估量定义清楚，但**当前无法低成本可靠预测**——
  真值需反事实训练分支才能观察，而所有廉价代理（8 个信号族）已全部否定，
  且 Door 分解表明连"归因到哪条通道"都不稳定（见 §4.3）。
- **论文定位**：不是"我们提出了完美迁移指标"，而是一个**有正结果、有机制消融、
  也有系统性负迁移边界**的完整迁移强化学习研究。重构稿见
  [`PAPER_CONTRIBUTION_RESTRUCTURE_20260728.md`](PAPER_CONTRIBUTION_RESTRUCTURE_20260728.md)。

## 4. 为什么收束：三层否定

### 4.1 行为 reward 信号族（07-11 → 07-15，三重否定）

| 尝试 | 形式 | 否定方式 |
|---|---|---|
| SIV 2×2 | per-source 因果干预打分 | 机制信号未过实践阈值（T=−0.048 < 0.10） |
| SHU | 阶段条件化准入判据 | 行为正/下游更新负的 mandatory contradiction |
| adaptive revocation | 时间维聚合撤销 | 预注册：3/3 误撤已证好源，代价 −120~−205 |

共同病灶：引导型好源执行段做"脏活"，与劣源在行为 reward 下不可区分。

### 4.2 非行为信号族（07-27，同样否定）

换信号族的三个候选全部走完：

- **update-space influence**：FAIL 且**排序反转**（最有益的 cell 被判最有害）——
  它度量即时分布错配，不是延迟学习价值；
- **T^critic sign**、**P0 lease oracle**：此前已封存；
- **zero-shot 行为探针**：Door 给出**同任务内**反向证据（run 行为 +58% 却 harmful，
  walk 行为 −61% 却最不负），就此关闭。

**所有 8 个信号族的失败机制相同**：度量**即时**量，而被估量是**延迟**学习价值。

### 4.3 归因本身不稳定（07-28，新边界）

Door 顺序因果分解在 source/target/stage/剂量/anchor/噪声种子**全部固定**下，
仍得到跨 learner seed **方向相反**的通道归因（s1/s3：行为致害、replay 补偿；
s2：行为无害、replay 致害），且**不是评估噪声**（episode 层面每 seed 内比值 10–20）。

> **learner-path dependence**：迁移效用不是 (source, target, stage) 的稳定标量，
> 而是依赖具体 learner state 与训练路径的**分布**：
> `U ~ p(U | source, target, θ_t, D_t, occupancy_t, channel, d, K)`。

这是本项目此前未记录的失败模式，也是当前最有价值的新洞察——它意味着**安全迁移必须
处理效用的不确定性，而不能把 source transferability 当作固定属性**。

## 5. 当前决策（PI 2026-07-28）

1. **停止运行新实验**，转入论文核心贡献重构稿。
2. **不再追加**：R-only 臂、source replay 比例扫描、新 target、新 seed、第九个 proxy。
3. **`admission_replay_mode` 保留**，但在贡献表中标为**诊断/控制基础设施**，
   不声明性能收益。
4. learned cross-task scalar transferability metric → **未来工作**。若要重启，
   它是一个独立的大型项目：需要更多 target × stage × source × learner seeds
   构成真正的 meta-transfer dataset，而非补一个实验。
5. Basketball 始终未参与任何选择与收集，保留为完全未见的外部 abstention 测试。

---

## 6. 协作与执行约定

- 协作文件：[`docs/agent_collab/claude_chatgpt_20260713_rbo_admission.md`](agent_collab/claude_chatgpt_20260713_rbo_admission.md)
  （追加式轮次；我=独立审查/方法设计/实验执行，ChatGPT=实现/编排/统计复算）；
- 高成本实验先出 run card 经 PI 审批；预注册 gate 字面执行；
- 实验执行权在 Claude（PI 2026-07-15 指示"后续跑实验都由你来执行"）。
