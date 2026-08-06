# 实验记录与分析（EXPERIMENT LOG）

> 更新：2026-07-16。本文档整合了原 `experiment_registry.md` 与全部散落的实验结果
> 文档，按时间线记录每个实验的目的、设计、关键数字与裁决。原始结果文档（含完整
> 统计表与审计细节）在 [`docs/archive/`](archive/README.md)。
> 机制术语与代码对应关系见 [`REPO_MAP.md`](REPO_MAP.md)。

---

## 1. 机制栈：当前方法由哪些层组成

RBO-PTF = FastTD3 + 逐层叠加的迁移机制（口径以 `fasttd3_ptf/ptf/mcg.py` 为准）：

| 层 | 机制 | 说明 |
|----|------|------|
| **L0** | FastTD3 backbone | off-policy、128 并行 env、distributional critic |
| **L1** | PTF source options | 冻结源策略作 call-and-return option；transfer loss（模仿）+ β termination + λ 衰减 |
| **L2** | MCG modular + gate | option=(教师, body-group)；warmup 后 critic significance gate 决定执行/蒸馏（**仅安全阀，非性能来源**） |
| **L3** | **RBO warmup bootstrap（核心）** | warmup 期执行 source option，把 reward-bearing transitions 注入 replay |
| **L4** | transfer decay | transfer_lambda 线性衰减到 0 |

L3 内部子机制（真正被消融的变量）：

| 子机制 | 选项 | 当前定论 |
|--------|------|---------|
| 3a 源选择 | uniform / reward-weighted（T⁰ probe softmax） | **weighted 是稳健主因**（强分化 bank +77.9） |
| 3b 执行 horizon | 固定短 h25 / per-source h50 | 总体中性、任务依赖（stair 是 h50 边界例外） |
| 3c teacher/student 混合 | legacy=0.5 Bernoulli；admission_bootstrap=单一 categorical | 当前主路径为 **student-inclusive categorical**（无外层 0.5） |
| 3d admission / exact abstain | all/none/static/manifest/schedule + adaptive(已否) | 被动正确性已验证；**自动判据未解决** |
| 3e replay lifecycle handoff | fixed_quota / physical_after_authority | **physical handoff 修复 repetition divergence** |

公共超参：warmup=30000、segment=25、decay=80000、total=100000、128 env、
buffer 51200/env、eval 每 5k。gate 统计量 = 10k–95k evaluation-grid mean return（18 点均值）。

## 2. 方法代号字典

| 代号 | 配置 | 含义 |
|------|------|------|
| **scr** | 空 bank | scratch 纯 FastTD3 基线 |
| **rand** | random + h25 | uniform 源选择（PTF 朴素版对照） |
| **safe** | safe_bootstrap + h50 | 历史"主方法"（weighted + safe-horizon） |
| **wfix** | safe_bootstrap + h25 | **当前静态 RBO 主方法**（weighted + 短 horizon） |
| **onlineb** | online_bootstrap | student-as-arm 在线权重（T 线 Step A） |
| **obrw** | onlineb + replay 对称降权 | 在线扩展（仅局部收益） |
| **admission-\*** | admission_bootstrap | student-inclusive categorical + lifecycle |
| **adaptive** | admission + AdaptiveAdmissionController | 时间维撤销（已判 FAIL） |
| bootstrap_only / no_bootstrap / full | L3/L2 消融 | 证明 L3 是主通道 |

---

## 3. 实验总表（按时间线）

### Phase 0-1：PTF 诊断与表征路线（2026-05 → 06-08，全 null）

| 实验 | 关键结果 | 裁决 |
|------|---------|------|
| Force-PTF push 100k | scratch +148 vs Force-PTF −20 | 常数 λ 过约束 actor |
| Decay-PTF 3v3 push | PTF −9 vs scratch +16，t=−0.34；PTF 方差小 3× | 无显著迁移增益 |
| entity encoder A/B | A=−386、A2=−271（冻结 reach-only E 自身是坏前端） | 判 null |
| anchored readout v1/v2-c | ≈−425 / −359~−486（scratch +472） | 判 null |
| ED-SF push transfer | 1M 步全程负（last-10≈−489），w_task 爆炸 | 判 null，代码已删 |

### Phase 2：MCG 与 package（2026-06-10 → 06-12）

| 实验 | 关键结果 | 裁决 |
|------|---------|------|
| MCG v1 三任务 | push ≈平；door 313<328；**window −153 负迁移**；Δ 量级与噪声同级 | gate 无负迁移免疫 |
| SC-MCG v2 | window 伤害 −153→−80 | 显著性校准是必要条件 |
| package 三轮 + EODT probes | R1/R2/R3 全 0%；oracle 上界 0-9%；chain warmup 展开链但 eval 仍 0% | 学习端瓶颈（credit assignment），专项停止 |
| zero-shot push→package | 重叠=0（push 源站桩） | 失败在 approach 段 |

### Phase 3：RBO 成型（2026-06-13 → 07-01）

| 实验 | 关键结果 | 裁决/意义 |
|------|---------|----------|
| RIC v1 宽 pilot（9 任务） | hurdle ROI +71%/cabinet +53%/powerlift +42%，regret 0；window 增益来自 warmup 暴涨而非 gate | 首次暴露 bootstrap 主通道 |
| 核心 ablation ×2seed | boot≈full（cabinet +68 vs +70）；no_bootstrap≈0 或负（hurdle −29） | **bootstrap=主通道，gate=安全阀** |
| Transfer Map v1/v2 | snippet score vs ROI ρ=−0.22；"scratch 卡住度"才与相对 ROI 相关（+0.73，分母 confound） | 跨任务 ROI 预测 ill-posed |
| terrain 核心三方 ×3seed | safe>rand 10/12（t≈2.58）；**crawl 全翻转**（safe 656<rand 700<scr 812） | crawl=abstain 黄金动机 |
| **wfix 解耦 ×3seed** | **源选择 +77.9（11/12，t=+3.08）；horizon −11.4（t=−0.46 中性）** | 主方法=weighted+h25（定论） |
| fall-recovery probe | safe 摔倒终止率最低（0-3%）；HB 一摔即 done | 增益机制=fall-avoidance 非 recovery |
| hurdle→stair zero-shot | hurdle 22.3/100% fall（stair_scr 516） | zero-shot 探针系统性低估 dynamic 源 |

### Phase 4：Transferability + 广度（2026-07-02 → 07-05）

| 实验 | 关键结果 | 裁决/意义 |
|------|---------|----------|
| Step A onlineb（crawl/pole） | pole PASS（−4%）；crawl 726<scr 833（share 50→76%、排序学对） | 残留 buffer 毒害是主因 |
| Step B obrw | **crawl +41.6**（毒害假设证实）；pole −30.8 | replay 降权有效但有代价 |
| T-gated abstain | crawl abstain 从未触发（分离度 0.002-0.009<δ） | 弱信号下二值阈值脆弱；连续降权稳健 |
| split 归因矩阵 v2 | 仅对称 both 获益（aonly −50/conly −96/split −10） | AC 数据分布一致性=独立 finding |
| seed 加固 + 主表补全 | slide obrw 614.8 全场第一（vs wfix +92.1，t=14.7）；stair=h50 边界 | 在线收益集中于 slide 型 |
| mh horizon-arm ×3seed | stair 不稳（t=0.86）、slide −158（t=−2.87）、crawl −63 | **否决**：arm 空间扩张的结构性代价 |
| negctrl door/spoon | door +0.7（零伤害）、spoon +38.8（t=+3.69） | soft 退让局部成立 |
| breadth 3-seed | obrw vs scr：maze +74.0(t=3.63)/truck +148.5(t=3.25)/cabinet +98.5(t=4.78)；vs rand 全打平 | 广度成立；选源增益取决于 T⁰ 权重分化度 |
| **wfix 裁决 15 runs** | wfix 与 obrw 在 maze/truck/door 打平；cabinet wfix 胜（+15.7） | **主方法简化为静态 RBO** |
| hurdle 增量 truck/maze | **truck +229.9（t=+3.47，1421 全场新高）**；maze +0.3（饱和） | T⁰ 增量=必要非充分 |
| 第二批 std9（5 任务） | powerlift wfix−scr **+77.6（t=14.78）**；**basketball wfix −101.5 / obrw −74.0 均 3/3 负** | basketball=负迁移 hard case |
| stability audit P0/P1/P2 | cabinet run>stand 3/3 同向（survival 全同）；run24 剂量匹配后方向混合；return 与 hard progress 错位 | 源身份决定数据价值；**return 不可作在线选源信号** |
| 扩源 v2 正交源 probe | hurdle maze 16.4/truck 13.6 双冠；reach 全线≤2.1 | hurdle 入 bank 的依据 |

### Phase 5：Admission Core + handoff（2026-07-06 → 07-13）

| 实验 | 关键结果 | 裁决/意义 |
|------|---------|----------|
| SIV 2×2 机制门 | B0=−0.0511/R0=−0.0304/I=+0.0331，均<0.10 门 | **行为信号第一次否定**；STOP，勿调阈值重启 |
| SHU 判别 gate | cabinet mandatory regression 接受应拒绝的 run-composite | **第二次否定**：behavior utility ≠ update utility |
| Admission Core v1（FINALV2） | basketball exact-none 安全门 PASS（execution/replay/distillation 全零，回 scratch 分布）；powerlift 30k 加速 t=4.318；100k retention FAIL（0.1996） | 安全门+加速成立；retention 归因待查 |
| 我的 T0002 复审（零训练） | **repetition divergence**：80k 源物理残留 1.2% 仍拿 50% 配额（oversample 43×），3/3 同崩、81.2k 自愈；**headroom 耗尽**是 retention 败因主体（wfix−scr 从 +127 收敛到 +6.5） | 裁决 gate 改为"暂态伤害消除+崩点消失"；retention 场地改 truck |
| **admission handoff v1（6 runs）** | powerlift 6/6 PASS：修复窗口 +20.1（3/3 正）、80k 崩点消失（+77/+138/+167）、critic share 中介预测命中（33.65% vs 预测 33.7%、7.19% vs 7.2%）、剂量-响应 r≈0.96；truck 4/4 PASS：fix−scr +227.8（t=4.74），保留 legacy gap 84% | **贡献③定稿：provenance-consistent source data lifecycle** |

### Phase 6：Adaptive revocation（2026-07-14 → 07-15，预注册 FAIL）

18/18 runs（stamp `20260714T110054Z`），机制=stage window 3000 步、segment 级
UCB/LCB（z=1.645）、每源每窗一票、连续 3 窗撤销（最早 9k）、不可逆、原子应用。

| 任务 | per-seed Δ（adaptive−对照） | gate | 裁决 |
|---|---|---|---|
| crawl | +41.5 / **−66.8** / +53.9 | Δ≥+30 且 3/3 正且有撤销 | **FAIL**（s2 无撤销且为负；placebo 差 −66.8 否证因果归因） |
| truck | −6.0 / **−119.7** / **−204.9** | 禁撤 hurdle/walk/run | **FAIL**（hurdle 3/3 被误撤） |
| powerlift | −4.7 / −2.0 / −4.9 | Δ≥−20 | PASS（9k 精确撤 crawl/reach，3/3 一致） |
| basketball | −23.7 / +36.2 / −34.7 | 描述性 | 大量撤销但无系统性改善 |

**定论 = 行为 reward 信号第三次独立否定**：引导型好源执行段做"脏活"，即时 reward
低，与劣源不可区分；判据只能识别明显无关源。完整机制解读见
[archive/adaptive_admission_v1_results.md](archive/adaptive_admission_v1_results.md)、
独立复算见 [archive/adaptive_admission_v1_codex_independent_audit.md](archive/adaptive_admission_v1_codex_independent_audit.md)。

### Phase 7：迁移性指标的系统性边界（2026-07-16 → 07-28，全部预注册）

本阶段的全部实验都在回答同一个问题：**能否用一个低成本、注入前的量预测 source 的延迟
学习价值**。答案是系统性的否定，且每一步都收窄了下一步的可能空间。

| 实验 | 关键结果 | 裁决/意义 |
|------|---------|----------|
| classic PTF 复现审计（07-22/23） | 三方对照（论文/官方码/本项目）无 BUG 级缺陷；β 失败机制=**sigmoid 表征死区**（logit −15~−17，σ′≈1e-6），非动力学 | 停修复线；`fixed-walk 3/3 加速`是唯一稳健正结果 |
| σ / current_transition / released_fidelity 三修复 | 全部 FAIL；去饱和后 Q_ω argmax ~90% 正确指向 walk 但 gap 仅 ~2% | 信号存在但**不足以驱动 termination** |
| update-space influence gate（07-27） | FAIL 且**排序反转**：最有益的 `hurdle_s2_run` 被判最有害 | 度量的是即时分布错配,非延迟学习价值；**信号族封存** |
| 标签清单审计（07-27，零训练） | 裁 PARTIAL：最佳标签在 t=0、唯一可得特征在 t=10k，阶段错配；6 个 EQD30K cell 全部会被误判准入 | 特征族判别力**从未被证明** |
| 标签可识别性审计（07-27，零训练） | 建立可测性判据 `U/trend`；锚点 crawl 0.83（成功）/ cabinet 10.31（失败）；筛出 door | 可测性成为**采集前置条件** |
| **Cabinet@10k gate**（07-27） | `CABINET_UNCERTAIN`：三源区间全跨 0，`\|U\|/SE` 无一超 1.74 | 罕见事件主导→标签在该 stage 不可分辨（**测量失败**） |
| **Door@10k gate**（07-27） | `DOOR_ALL_SAME_SIGN`：stand −32.64 / run −30.63 harmful，walk −22.20 unc，**9/9 per-seed 负**；`\|U\|/SE` 中位 9 | **真结论非测不出**；且**学习效用与行为先验反向**（run 行为 +58% 却 harmful，walk 行为 −61% 却最不负） |
| **Door 通道分解**（07-28） | `UNRESOLVED`：U^BR −30.63 [−50.56,−10.71] neg，但 U^B [−138.79,+30.75] 与 Δ^{R\|B} [−71.48,+118.25] 均 unc | **learner-path dependence**（见 §4.3）；解耦机制本身通过全部验收 |

**Phase 7 的统一教训**：行为即时效果、critic advantage、短期 handoff、单步 influence
**都不能**稳定代替延迟学习价值；而 Door 分解进一步表明，即使把通道拆开，**归因本身**
也不是 (source, target, stage) 的稳定函数。

### Phase 8：从"预测"转向"测量"，与跨任务边界的暴露（2026-07-29 → 08-04）

Phase 7 关闭了零成本预测，本阶段回答两件事：**最小测量代价是多少**，
以及**已有的正面结果能否跨任务**。第二个问题的答案是否定的。

| 实验 | 关键结果 | 裁决/意义 |
|------|---------|----------|
| per-state Q-switch / QMP（07-29） | `QMP_FIDELITY_PARTIAL`：退化为 student（source share 0.3–5.5%）；critic 对源的即时优势 **18/18 全负**、2/2 任务排序错 | **取消聚合救不了信号族**——问题在被聚合的量本身 |
| **hurdle 加速**（07-30） | `SPEEDUP_CONFIRMED`：θ=200 中位 **4.38×**、θ=300 **3.59×**，各 3/3 | 但 100k 衰减到 **1.24×**；训练不稳定为 source 臂独有 |
| **RACING_K**（07-30，两批 24 条） | `RACING_VIABLE`，**K\*=10000**；两批各 3/3；K=5000 不稳健（批1 3/3 → 批2 1/3） | 不可能性刻画的**另一半**：预测不可行但**测量可行**。辨别证据 12/12 排出 walk>stand（与 zero-shot 反向）|
| Competence-Gated Transfer（07-30） | **实现前关闭** | 第四次行为代理换皮；三条证据全在仓库内 |
| RACING_MULTI / RACE-then-RUN（07-30/31） | **作废，未执行** | 候选集合不同 → 全局固定排序即可解释；`argmax U ≡ argmax J` 代数退化 |
| **RACING_REJECT v1–v4**（07-30/31） | v1 作废（sanity 蕴含主判据）→ v2 `REPLICATION_DIVERGED` → v3 撤回（判据切换）→ v4 **`PARTICIPANT_DIVERGED`** | **door 的 ground truth 本身跨 learner 反转**（18/18 负 → 新批 2/9 正）；判决场关闭，`M31` |
| **slide 可推广性审计**（07-31） | `GEN_OK`：`argmax=walk` 在 **6 个独立 learner** 上一致 | 首次具备**真正的 crossover**（同候选集合 argmax 反转）；发现基线相消机制（`M32`）|
| **hurdle 选源价值**（08-01） | `SELECTION_VALUABLE`：θ=200 中位 **4.95×**（3/3），θ=300 **3/3 右删失** | 首次证明"**选对源 > 选错源**"，此前只证过"用源 > 不用源" |
| **slide 加速**（08-01 数据 / 08-04 裁决） | **`SPEEDUP_REFUTED`**：三阈值中位 0.851/0.627/0.758 全 <1.5，100k 被 scratch 反超（792.4 vs 293.3）| **跨任务加速不成立**；据预注册 §7 收紧 hurdle 结论为"仅 hurdle" |
| **slide 30k 硬退出**（08-01 数据 / 08-04 裁决） | `HARD_EXIT_SUPPORTED`：终点 **+631.8 ± 9.3**（3/3），293.3 → 929.1 | 损害**完全来自 30k 后继续注入源**且可逆；**工程基线非贡献**（PTF 原文 λ 衰减即为此设计）|
| **零训练 progress 粗筛**（08-04，族 12） | **`HOLDOUT_FAILED`**：crawl 上有害的 run 位移 14.302，slide 上有用的 walk 位移 1.814，**反向 7.9×** | 阈值须满足 `14.302<θ<1.814` = **空集**；**连"单向排除"这个最弱用法也不成立** |

**Phase 8 的统一教训**：

1. **预测不可行 ⇒ 测量是唯一剩下的路**，且其最小代价可量化（K\*=10000，30k 成本 / 节省 67k）；
2. **跨任务是真正的边界**——同一机制、同一剂量、同一候选集合，hurdle 正而 slide 毁灭性负；
3. 全程恒定剂量是本项目**自设的**配置缺陷（`WARMUP_STEPS` = 总步数），
   不是 PTF 原文用法，修它属于工程基线；
4. 降低用法强度（排序 → 单向排除）**救不了**已失败的信号空间（族 12）。

---

## 4. 当前定论汇总（2026-07-28 重构）

按证据强度分区。**分区本身是结论的一部分**——把诊断工具写成性能贡献，或把未支持的
主张写成已支持，是本项目反复出现过的错误。

### 4.1 已支持（有干预实验与预注册裁决）

- **性能主方法 = 静态 RBO / wfix**：T⁰ reward-weighted 源选择 + h25 + target-reward
  replay + `bootstrap_only`。正结果覆盖 hurdle / truck / maze / cabinet / powerlift / slide。
- **source identity 与分配方式重要**：wfix 3-seed 解耦给出源选择主效应 +77.9（11/12，t=3.08）。
- **admission lifecycle 限制负迁移**：exact abstention（basketball 安全门 execution/replay/
  distillation 全零）、quarantine、provenance-consistent physical handoff（powerlift 6/6 +
  truck 4/4 全 PASS，剂量-响应 r≈0.96，80k 崩点消失）。
- **joint RBO 的总效应可以稳定为负**：door-run `U^BR` 在 3/3 seed 上为负，
  90% CI [−50.56, −10.71]。负迁移是真实且可复现的现象，不是噪声。
- **behavior authority 与 replay eligibility 在工程上已可独立控制**
  （`admission_replay_mode`，见 4.2）。

### 4.2 诊断/控制基础设施（**不得**声明为性能贡献）

- **`admission_replay_mode: shared | student_only`**：修复了"一个 student-inclusive
  categorical 同时决定谁执行与 critic 采什么"的概念缺陷。验收完整（behavior share
  与 joint 臂差 <0.1%，critic source 采样严格 0，26/26 既有测试全过），
  但**它所要支持的科学主张未被证实**（4.3），故只能计为诊断与控制工具。
- **标签可测性判据**（`U/trend`，锚点 crawl 0.83 / cabinet 10.31）：采集迁移标签前的
  前置筛选，本身不是迁移性指标。
- 冻结 source-free evaluator（128-episode 面板，前 32 与历史面板逐位兼容）。

### 4.3 未支持（做过实验，证据不足以支撑）

- **"保留 source 行为、关闭 source replay 更安全"**：只在 door seed 2 成立；
  seed 1/3 恰好相反（行为致害 −59/−102，而 replay **补偿** +41/+68）。
- **replay 是 Door 负迁移的稳定根因**：同上。
- **双通道解耦可直接提升性能**：本轮未测得任何性能收益。
- **channel-specific metric 已验证**：未验证。

> **新边界：learner-path dependence（本项目此前未记录的失败模式）**  
> 即使 source、target、stage、剂量、anchor、噪声种子**全部固定**，通道归因仍会随
> learner trajectory 翻转。且这**不是**评估噪声：episode 层面
> `|U^B|/pairSE` = 12.5 / 0.5 / 20.2、`|Δ|/pairSE` = 10.7 / 12.4 / 12.4，
> 每个 seed 内部的测量都高度可靠。  
> 描述性观察：三个 seed 上 `corr(U^B, Δ^{R|B}) = −0.98`。**但须注意其中很大一部分是
> 代数必然**——`U^BR = U^B + Δ` 是恒等式，而 `sd(U^BR)/sd(U^B) = 0.235`，故
> `Δ ≈ const − U^B`。该相关不能单独作为"两通道反馈耦合"的独立证据。

### 4.4 已封存的信号族（禁止变体抢救）

T⁰ 行为重叠 · T^critic sign · SIV · SHU · adaptive reward revocation · P0 lease oracle ·
update-space influence · **zero-shot 行为探针**（Door 反向证据）。

**共同失败机制**：全部度量**即时**量（行为像不像、Q 值高不高、这批数据当下是否拉正梯度），
而被估量是**延迟**学习价值。

### 4.5 被估量的正确形式（收窄后）

不能再写成 `U = f(source, target, stage)`。现有证据支持的是一个**分布**：

```
U ~ p( U | source, target, θ_t, D_t, occupancy_t, channel, d, K )
```

相应地，迁移性指标若要定义，应是分布量而非标量：
`T_i(t,K,d) = ( E[U_i], Var(U_i), P(U_i>0) )`。
**定义得出来，但当前无法低成本可靠预测**：真值需反事实训练分支才能观察，而现有
任务数与 learner trajectory 数不足以训练可泛化的估计器。

### 4.6 hard case 地图

basketball（负迁移，探索瓶颈型，**始终未参与任何选择,保留为外部 abstention 测试**）·
package（回报事件稀疏）· crawl（好源=毒数据）· stair（horizon 边界）·
cabinet（罕见事件主导→标签不可测）· door（三源全负 + 行为先验反向 + 通道归因不稳）。

## 5. 证据索引

- 训练日志：`logs/train/<experiment>_<stamp>/`；W&B project `fasttd3_ptf`
- 审计产物：`artifacts/`（admission_core_v1 / admission_handoff_v1 / adaptive_admission_v1 / mechanism_gate 等）
- 预注册配置：`configs/experiments/`（SHA256 冻结）
- 原始结果文档：[`docs/archive/`](archive/README.md)
