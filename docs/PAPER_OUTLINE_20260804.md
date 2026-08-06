# 论文结构与图表清单

> 2026-08-04。配套 `PAPER_CLAIMS_20260804.md`（主张—证据—边界逐条映射）。
> 本文只管**组织**：每节讲什么、用哪张图、图的数据在不在。
> 正文定稿用英文；本文用中文以便 PI 审阅论证结构。
>
> 图表状态：`HAVE`=数据齐可直接出图 / `NEED`=需新画但数据齐 / `MISSING`=数据不齐

---

## 标题（工作稿）

> **Transfer Utility Is Not a Property of the Task Pair:
> An Impossibility Characterization and the Minimum Measurement That Suffices**

## Abstract 的五句话

1. 跨任务迁移 RL 的核心决策是"用哪个源、用不用"，主流做法是设计一个**廉价指标**去预测；
2. 我们在 HumanoidBench 上系统性地证伪了这条路——**十二个信号族、七个信号空间**，
   每族独立预注册独立裁决；
3. 而且不止是经验失败：给出**两条原理性反例**，说明这类量在原理上无法承载该决策；
4. 由此推出唯一剩下的路径是**直接测量**，并给出其最小代价（`K*=10000`，
   `3K` 步换取节省 `67k` 步），同一次测量可同时决定**准入**与**选源**；
5. 边界如实给出：跨任务的正面加速目前只有一个 target，全部判决场的真值均已知。

---

## 1. Introduction

**论证链**（每一步都必须挡住一个反驳）：

| 步 | 说什么 | 挡住的反驳 |
|---|---|---|
| 1 | 决策问题：`(源集合, target, learner 状态) → 用不用 / 用哪个` | — |
| 2 | 自然做法是预测：设计廉价指标 `X` 并假设 `X → U` | — |
| 3 | 我们试了 12 族全败 | "你们没试对指标" → 见 §3 的空间覆盖表 |
| 4 | **但经验失败不构成不可能性** | "再多试几个就有了" → 由第 5 步挡 |
| 5 | 两条原理性反例：输入相同而输出不同 / 可行区间为空 | "换个阈值/归一化" → 空集论证与取法无关 |
| 6 | 故只能测量；给出最小代价与它能同时决定的两件事 | "测量太贵" → `3K` vs 节省 `67k` |

**贡献声明**（严格对应 PAPER_CLAIMS，不多说一个字）：

- C1 十二族/七空间的系统性证伪 + 两条原理性反例（支柱 I）
- C2 最小测量代价的量化，及"一次测量同时给出准入与选源"（支柱 II）
- C3 端到端系统的实证（**待 P2 裁决；若为负则本条删除，退化为各零件结果**）

## 2. Setup

- 因果干预标签 `U_i(t,d,K) = J_sf(θ_i @ t+K) − J_sf(θ_student @ t+K)`
- `J_sf` = **source-free** 评估（源在评估时不在场）——全项目不变口径
- 冻结面板：16 eval seeds × 8 ranks = 128 deterministic episodes
- **图 1**：`U` 的定义示意（同 anchor 分叉 → 等剂量干预 → source-free 评估）  `NEED`

## 3. 支柱 I：不可能性刻画

### 3.1 十二族 / 七空间

- **表 1**：十二族 × (测什么 / 怎么失败 / 出处)  `HAVE`（`impossibility` §2）
- **图 2**：七个信号空间的覆盖示意（行为/即时 reward/梯度/critic/reward 结构/
  任务定义+静态规格/任务进度）  `NEED`

### 3.2 两条原理性反例（核心）

- **反例 A（输入相同，输出不同）**：slide 与 stair 共用同一份
  `ClimbingUpwards.get_reward`、常量逐字节相同；walk 的效用为
  `+56.95` vs `+0.19 [−5.35,+5.72]`（后者**跨零**）。
  任何只读 `(source,target)` 静态规格的量，在两者上读到的输入完全一样。
  - **图 3**：两个 target 的 reward 源码 diff（空）+ 效用对比条形图  `NEED`
- **反例 B（可行区间为空）**：单向排除需 `P(run,crawl) < θ < P(walk,slide)`，
  即 `14.302 < θ < 1.814`，**空集**；与阈值取法无关。
  - **图 4**：三个 target × 三个源的位移 vs 真实效用散点，标出空集区间  `HAVE`
    （`progress_screen_v1/probe.json` + 各 target 真值）

### 3.3 统一解释

`U ~ p(U | source, target, θ_t, D_t, occupancy_t, channel, dose, K)`；
十二族都在估计只含 `(source,target)` 的**点函数**。
支撑证据三类：方向非对称（sibling gate，**只用负方向那一半**，见 CLAIMS I-5）、
通道归因跨 seed 反向（door 分解）、符号跨 learner 反转（M31）。

## 4. 支柱 II：最小充分测量

### 4.1 racing：直接测 `U` 的短跑近似

- 与十二族的关键区别：**estimand 未变**——不做跨量类外推，
  只问 `U(小K) → U(大K)` 的同量 horizon 一致性，可直接验证
- **表 2**：`K ∈ {2k,5k,10k}` 的 top-1 命中（两批各 3 seeds）  `HAVE`
- **图 5**：`K` vs 命中率 + 成本-收益曲线（`3K` vs 节省 `67k`）  `NEED`

### 4.2 辨别力：测的不是行为质量

zero-shot 行为排序 `run > stand > walk` vs 真实 `run > walk > stand`；
racing 在 K≥5000 的 **12/12** 运行排出 `walk > stand`。
- **表 3**：三种排序并列  `HAVE`

### 4.3 准入：同一次测量还能决定"要不要用"

- `admit = ∃i: U_i > 2·SE_i`；`false_admit=0, false_reject=0`
- **图 6**：三个 target × 3 seeds 的 `U ± 2SE` 森林图，标出阈值线  `HAVE`
  （`racing_admission_v1/results.json`）
- **关键子图**：crawl 上 argmax 跨 seed 完全不一致（stand/run/walk 各一次）
  → **全负时 argmax 是噪声，准入不可被选源替代**  `HAVE`

### 4.4 端到端系统

- 决策链：racing → 自动决策 → 主训练（带源则 30k 退出）
- **图 7**：三臂 × 三 target 的 100k 终点配对比较  `MISSING`（P2 进行中）
- **图 8**：成本-性能曲线，x 轴为**含 racing 的总交互**  `MISSING`（P2）
- 成本两个口径都报（理论最小 `3K` / 本实验实现 `4K`，见 CLAIMS §2.1）

## 5. Limitations（**单独成节，不塞进 discussion**）

逐条来自 PAPER_CLAIMS §3 与 §4：跨任务加速只有 hurdle 一个；
hurdle 100k 衰减到 1.24× 且回撤为 source 臂独有；恒定剂量是本项目自设缺陷；
`U` 符号跨 learner 反转；per-seed 数值不可当真值；准入不提升上限只避免灾难；
未解决"前人解不了的任务"；**全部判决场真值已知 → 检验判据而非前瞻发现**。

- **表 4**：适用范围表（机器人/target/源/算法/seeds/评估/真值状态）  `HAVE`

---

## 图表缺口汇总

| 图表 | 状态 | 阻塞项 |
|---|---|---|
| 图 1 `U` 定义示意 | `NEED` | 无，画图即可 |
| 图 2 七空间覆盖 | `NEED` | 无 |
| 图 3 reward diff + 效用对比 | `NEED` | 无（diff 为空是卖点） |
| 图 4 位移 vs 效用散点 + 空集 | `HAVE` | 无 |
| 图 5 K vs 命中 + 成本收益 | `NEED` | 无 |
| 图 6 准入森林图 | `HAVE` | 无 |
| 图 7 三臂配对比较 | `MISSING` | **P2 训练中** |
| 图 8 成本-性能曲线 | `MISSING` | **P2；另需 60k 评估点才能做严格总交互对齐** |
| 表 1–4 | `HAVE` | 无 |

**唯一的真实数据缺口是 P2**；图 8 若要做严格的总交互对齐，还需补 60k 评估点
（当前用 50k 作保守下界，见 endtoend 预注册 §4）。
