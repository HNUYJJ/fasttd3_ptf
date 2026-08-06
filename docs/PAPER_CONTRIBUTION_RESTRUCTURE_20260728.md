# 论文核心贡献重构稿（2026-07-28）

> 状态：PI 2026-07-28 正式收束当前论文路线后的重构。**不再运行新实验。**  
> learned cross-task scalar transferability metric **降为未来工作**，不再作为本论文
> 必须补齐的组件。  
> 证据来源：[`EXPERIMENT_LOG.md`](EXPERIMENT_LOG.md) §3–§4（含 Phase 7 全部预注册裁决）。

## 0. 论文一句话（英文，投稿用）

> Frozen source policies can substantially accelerate humanoid learning through
> reward-bearing off-policy experience, but their learning value is intervention- and
> learner-dependent rather than captured by behavioral similarity or immediate value.
> We therefore combine static source allocation with exact abstention and explicit
> experience lifecycle control, and characterize when transfer succeeds or fails.

中文要点：**冻结源策略确实能大幅加速人形学习，但其学习价值既依赖干预方式、也依赖
learner 自身轨迹，无法由行为相似度或即时价值刻画。** 因此我们把静态源分配、精确弃权
与显式的经验生命周期控制组合起来，并刻画迁移在什么条件下成功、什么条件下失败。

这不是"我们提出了完美迁移指标"，而是一个**有正结果、有机制消融、也有系统性失败边界**
的完整迁移强化学习研究。

## 1. 问题定义

在 HumanoidBench（61-DoF H1-hand）上，给定一组**冻结的** locomotion 源策略
{stand, walk, run, …} 与一个新 target 任务，问：如何利用这些源加速 target 的
在线强化学习（FastTD3 骨干），且**不引入负迁移**？

三条约束贯穿全文，也是与多数迁移 RL 工作的区别：

1. **最终评价永远是 source-free student**——源在评估时不在场。我们要的是学生学会了，
   不是源在替它开车。
2. **固定环境交互预算**——源执行会占用原属 student 的交互机会，这是行为通道的真实
   机会成本，不能不计。
3. **必须能"什么都不做"**——存在源确实有害的任务，机制必须支持精确弃权而非只支持加权。

被估量（迁移效用）：

$$U_i(t,K,d)=J_{\text{sf}}\bigl(\theta^{(i)}_{t+K}\bigr)-J_{\text{sf}}\bigl(\theta^{(\text{student})}_{t+K}\bigr)$$

即在 stage $t$ 以剂量 $d$ 注入源 $i$、再训 $K$ 步后，source-free 性能相对纯 student 基线
的增量。**本文的核心经验发现之一是：这个量不是 $(source, target, stage)$ 的稳定标量**（§5.3）。

## 2. 方法总图

```
                    ┌─────────────────────────────────────────┐
   frozen sources   │  Source-Target Effect Map (静态先验)     │
   {stand,walk,run} │  zero-shot 行为对价 → allocation prior   │  ← 只作分配先验
                    │  ★ 不宣称预测 learning ROI               │     不预测学习 ROI
                    └──────────────────┬──────────────────────┘
                                       │ 源选择 + 权重 (WFix)
                                       ▼
   ┌───────────────────────────────────────────────────────────────┐
   │  Reward-Bearing Bootstrap (RBO) —— 主要迁移通道                │
   │  冻结源在 target env 执行 h=25 步，产生的 transition           │
   │  带 **target reward** 注入 replay；学生从这些状态继续          │
   └───────────────┬───────────────────────────────┬───────────────┘
                   │ behavior authority            │ replay eligibility
                   │ (谁执行)                       │ (critic 采什么)
                   ▼                               ▼
   ┌───────────────────────────────────────────────────────────────┐
   │  Admission Lifecycle —— 限制负迁移                             │
   │  · student-inclusive exact abstention（可精确关到零）          │
   │  · quarantine（未验证数据不污染主 buffer）                     │
   │  · provenance-consistent replay（来源分层配额）                │
   │  · authority-coupled physical handoff（authority 结束即交接）  │
   └───────────────────────────┬───────────────────────────────────┘
                               ▼
              frozen source-free evaluator (128-episode 面板)
```

**关键机制事实**（本文修复的概念缺陷）：在 `admission_bootstrap` 下，
"谁执行"与"critic 采什么"原本共用**同一个** student-inclusive categorical——
一个标量同时决定两条通道。本文提供 `admission_replay_mode` 使二者可独立控制
（详见 §4，**计为诊断/控制基础设施，非性能贡献**）。

## 3. 贡献列表（按证据强度分区）

### C1. Reward-bearing bootstrap 是有效的主要迁移通道 ★已证性能贡献

冻结源在 target 环境执行、带 **target reward** 的 transition 注入 replay，
在多个任务上产生显著加速（§5.1 表）。这是本文的性能主通道。

### C2. Source identity 与分配方式重要 ★已证

不是"有源就行"：WFix 3-seed 解耦给出**源选择主效应 +77.9（11/12 seed 同向，t=3.08）**，
且 horizon 的作用是任务依赖的、中性的——**源选择才是稳健主因**。

### C3. 显式经验生命周期控制可限制负迁移 ★已证

`student-inclusive exact abstention` + `quarantine` + `provenance-consistent replay` +
`authority-coupled physical handoff`。干预实验证据见 §5.2：安全门可精确关到零；
handoff 修复 powerlift 6/6 + truck 4/4 全 PASS，剂量-响应 $r\approx0.96$，
80k 崩点消失，且 critic share 的**中介预测命中**（预测 33.7% / 实测 33.65%）。

### C4. 迁移成功与失败的系统性刻画 ★本文的第二类主贡献

不是附录里的 limitation，而是与 C1–C3 同等的正文内容（§5.3）。包含三层否定：
行为 reward 信号族、非行为信号族、以及**归因本身的不稳定性**。

### I1. Behavior/replay 通道解耦 ☆诊断与控制基础设施（**不声明性能收益**）

`admission_replay_mode: shared | student_only`。修复了两条通道共用一个标量的概念缺陷，
工程验收完整（behavior share 与 joint 臂差 <0.1%，critic source 采样严格 0，
26/26 既有测试通过）。但它所要支持的科学主张（"保留行为、关闭 replay 更安全"）
**未获证实**，故只能计为工具。

### I2. 标签可测性前置判据 ☆基础设施

在投入昂贵的迁移标签采集之前，先用无源臂数据判断标签在该 stage 是否可分辨
（判据 `U/trend`，已实测锚点：crawl 0.83 成功 / cabinet 10.31 失败）。
**本身不是迁移性指标。**

### F1. Learned cross-task transferability metric ○未来工作

被估量定义清楚（§1），且应是**分布量**而非标量：
$\mathcal T_i=(\mathbb E[U_i],\operatorname{Var}(U_i),P(U_i>0))$。
当前不可低成本可靠预测（§5.3）。重启它需要一个真正的 meta-transfer dataset
（更多 target × stage × source × learner seeds），是独立的大型项目。

## 4. 主要实验表

### 4.1 正结果（C1/C2：RBO 加速与源选择）

| target | 结果 | 统计 |
|---|---|---|
| truck | +229.9（1421，全场新高） | t = +3.47 |
| powerlift | wfix − scr **+77.6** | t = 14.78 |
| slide | obrw 614.8（vs wfix +92.1） | t = 14.7 |
| maze | +74.0 | t = 3.63 |
| cabinet | +98.5 | t = 4.78 |
| hurdle | 三源全正（等剂量单源标定，`FULL_ORDER_REPLICATED`） | 3 seed 复现 |
| **源选择主效应** | **+77.9** | 11/12，t = 3.08 |

### 4.2 安全性与生命周期（C3）

| 实验 | 结果 | 裁决 |
|---|---|---|
| basketball exact-none 安全门 | execution / replay / distillation **全零**，回归 scratch 分布 | PASS |
| powerlift handoff v1 | 修复窗口 +20.1（3/3 正）；80k 崩点消失（+77/+138/+167） | 6/6 PASS |
| truck handoff v1 | fix − scr **+227.8** | t = 4.74，4/4 PASS |
| 剂量-响应 | $r \approx 0.96$ | 机制因果证据 |
| critic share 中介预测 | 预测 33.7% / 实测 33.65%；预测 7.2% / 实测 7.19% | 中介命中 |

### 4.3 失败边界（C4）

| 任务/实验 | 结果 | 边界类型 |
|---|---|---|
| basketball | wfix −101.5 / obrw −74.0，**均 3/3 负** | 负迁移（探索瓶颈型） |
| crawl | 三源全负 | 好源=毒数据 |
| **door** | stand −32.64 / run −30.63 harmful；walk −22.20 unc；**9/9 per-seed 负** | 负迁移，且**行为先验反向** |
| cabinet@10k | 三源区间全跨 0，$\lvert U\rvert/\text{SE}$ 无一超 1.74 | **标签不可测**（罕见事件主导） |
| door 通道分解 | $U^{BR}$ neg，但 $U^B$ 与 $\Delta^{R\mid B}$ 均 unc | **learner-path dependence** |

## 5. 失败边界详述（论文正文，非附录）

### 5.1 行为效用 ≠ 学习效用（同任务内直接证据）

Door 提供了本项目第一个**同任务内**、控制了 source/target/stage/剂量/anchor/噪声种子的
直接反例：

| source | zero-shot 行为（zero=64） | 行为相对 | RBO 学习效用 $U$ |
|---|---:|---:|---:|
| run | 101 | **+58%** | **−30.63**（harmful） |
| stand | 59 | −8% | −32.64（harmful） |
| walk | 25（62% 摔） | **−61%** | **−22.20**（三者中最不负） |

此前的证据都是跨任务的（hurdle 全正、crawl 全负，行为量根本区分不了二者）。
**后果：zero-shot 行为探针作为廉价迁移性指标就此关闭。**

### 5.2 八个信号族的共同失败机制

T⁰ 行为重叠 · T^critic sign · SIV · SHU · adaptive reward revocation · P0 lease oracle ·
update-space influence · zero-shot 行为探针。

其中 update-space influence 不仅 FAIL 而且**排序反转**（最有益的 cell 被判最有害）。

**共同机制**：全部度量**即时**量（行为像不像、Q 值高不高、这批数据当下是否拉正梯度），
而被估量是**延迟**学习价值。这与在线 RL 数据归因的核心困难一致——训练样本不仅改变参数，
还会改变之后收集的数据分布，因此固定数据集式的局部归因无法覆盖完整学习效应。

### 5.3 Learner-path dependence（本文最重要的新边界）

Door 顺序因果分解：$U^{BR}=U^{B}+\Delta^{R\mid B}$（精确恒等式，实测误差 0.00e+00）

| seed | $U^{B}$（行为通道） | $\Delta^{R\mid B}$（replay 通道） | $U^{BR}$ |
|---|---:|---:|---:|
| 1 | −59.21 | **+41.43** | −17.78 |
| 2 | −1.34 | **−39.69** | −41.04 |
| 3 | −101.50 | **+68.42** | −33.08 |

总效应 $U^{BR}$ 在 3/3 seed 上稳健为负（90% CI [−50.56, −10.71]），
但两个分量的区间均跨 0。**且这不是评估噪声**——episode 层面每 seed 内部的测量高度可靠
（$\lvert U^{B}\rvert/\text{pairSE}$ = 12.5 / 0.5 / 20.2，$\lvert\Delta\rvert/\text{pairSE}$ = 10.7 / 12.4 / 12.4）。

> **即使 source、target、stage、剂量、anchor、噪声种子全部固定，行为与 replay 的作用
> 归因仍会随 learner trajectory 翻转。**

因此迁移效用应写成分布而非标量：

$$U \sim p\bigl(U \mid \text{source},\text{target},\theta_t,\mathcal D_t,\text{occupancy}_t,\text{channel},d,K\bigr)$$

**方法学推论**：安全迁移必须处理效用的**不确定性**，而不能把 source transferability
当作固定属性。这正是 C3（exact abstention + lifecycle control）在方法论上的正当性来源——
当效用不可靠预测时，可精确关闭与可审计的数据生命周期比更精细的加权更重要。

> **一处必须标注的统计 caveat**：三个 seed 上 $\operatorname{corr}(U^B,\Delta^{R\mid B})=-0.98$，
> 但其中**很大一部分是代数必然**——$U^{BR}=U^B+\Delta$ 是恒等式，而
> $\operatorname{sd}(U^{BR})/\operatorname{sd}(U^{B})=0.235$，故 $\Delta\approx c-U^{B}$。
> 该负相关**不能**单独作为"两通道反馈耦合"的独立证据。

### 5.4 标签本身可能不可测

Cabinet@10k：三源区间全跨 0，且**没有任何 per-seed 效应超过 1.74 个面板 SE**。
机制是 source-free 回报被罕见成功 episode 主导（median 11–28 而 max 33–706）。

由此得到一条可复用的前置判据（I2）：**在投入源标定之前，先用无源臂数据判断标签在该
stage 是否可分辨**。实测锚点：crawl `U/trend`=0.83（标签干净）、cabinet 10.31（不可测）。

## 6. 方法学教训（可迁移到其它 RL 实证工作）

1. **episode-level SE 不能代替 learner-seed 不确定性。** 在 Door 分解上，若用单 seed 的
   128-episode SE 裁决，seed 1 会"证明" behavior 主导、seed 2 会"证明" replay 主导，
   **两个矛盾结论各自看起来都有 12σ 显著性**。
2. **"总效应显著而分量不显著"必须裁为 UNRESOLVED，不得称纯交互**——那也正是功效不足的样子。
3. **可测性应当先于可解释性**：先确认标签能被分辨，再讨论它由什么解释。
4. **预注册必须在揭盲前落盘**（本项目全部 Phase 7 实验均先提交裁决脚本再产出数据）。

## 7. 与已封存路线的关系

本文**不**包含：learned transferability predictor、自动源退出/撤销、
channel-specific metric、zero-shot 探针作为选源依据。
它们的否定过程构成 §5，是论文的组成部分，而非被隐藏的失败。

## 8. 待补（下一步的文档工作，非实验）

- [ ] 外部 baseline 的定位说明（JSRL single-guide / PTF-distillation / best-single；
      36-run 设计已存档，PI 曾指示暂缓）
- [ ] 相关工作章节：与 importance-weighted transfer of samples、在线 RL 数据归因、
      SF/GPI 线的关系（本项目 ED-SF 分支的负结果可并入 §5）
- [ ] 主表的统一重绘（当前数字散落在 Phase 4–7，需统一评估协议口径后合表）
