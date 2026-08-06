# reward 侧指标的天花板在哪里：源干预后的技能吸收探针

> **2026-07-29 晚，外部审核后重大修订。** 本文初稿有三处表述过强，其中一处
> （"BTE 是 U 的中介变量"）是实质性科学错误，已撤回。修订后的结论比初稿弱得多，
> 但方向判断保留。三处纠正见 §0。

## 0. 三处纠正（外部审核 ChatGPT 指出，我已独立复算确认）

### 0.1 【实质性错误，已撤回】"BTE 是 U 的中介变量"

初稿把"源臂相对无源臂的 `progress_max_dx` 提升"当作独立的中介变量。**这是同义反复。**

slide/stair 的 reward 是 `stand_reward × small_control × move`，
而 `progress_max_dx` 就是位移，与 `move` 近乎同一个量。因此 ΔJ 与 Δprogress
高度相关是**该 reward 定义的推论，不是独立发现**。我独立复算：

```
6 个 task-source 单元    r = 0.967
18 个 seed 级单元        r = 0.941
```

**我没有解释任何东西，只是换了个坐标重新描述"有没有迁移效应"。**
该量降级为**事后的 skill-uptake outcome**，既不是中介变量，也不是预测器。

### 0.2 "源在两个 target 上做的事几乎一样"—— 不成立，已收窄

初稿自相矛盾：文中列出 episode 长度 82.6 vs 61.2（**差 35%**）却仍称"几乎一样"。
另有 `move` 差 8%、per-step 差 10%、NET 差 12%，**且地形动力学完全不同**。

准确表述：**静态 reward 分量画像相似**，但源产生的状态轨迹、接触模式与动力学效果
从未被比较过。

### 0.3 "两个 student 自主学习速率逐位相同"—— 均值巧合，已收窄

均值 +44.3% vs +44.4% 掩盖了极大的 seed 内变异：

| | s1 | s2 | s3 |
|---|---:|---:|---:|
| slide | +58.6% | +52.9% | +26.0% |
| stair | +39.7% | +17.0% | **+103.4%** |

**因此不能据此否证"任务可学性差异"这一类解释。** 初稿的"否证了冗余解释"过强。

### 0.4 "student 复制了教师行为"—— 未被观察，已改名

student 可能学到一种完全不同的步态却同样走得更远。全文中的"行为被复制/传递"
一律改称 **post-intervention skill uptake（源干预后的技能吸收）**，
不断言发生了行为模仿。

---

（以下为初稿正文，保留原始记录；凡与 §0 冲突处以 §0 为准。）

# 初稿：reward 侧指标的天花板在哪里

> 2026-07-29。零训练成本（只评估既有 10k anchor checkpoint + 复用 12 份 20k 面板）。
> 起因：PI 追问 BAC 方向是否真的不能深挖——"从 reward 拆解源策略对目标任务的价值效用
> 很符合直觉"。本文回答该直觉在哪里成立、在哪里触到天花板。

## 1. 先纠正一个我上一轮造成的误解

上一轮我报告"BAC 相对简单基线没有增量"，容易被读成"reward 分量分解无效"。
**这个读法是错的**，因为四个被比较的预测器里，

```
P3 = main progress component = 取非通用分量中边际敏感度最大的单个分量
```

**它本身就是 reward 分量分解**，只是最简形式。而它的成绩是：

| 预测器 | 全序命中 | 最差源命中 |
|---|---:|---:|
| P1 episodic return（标量） | 1/4 | 1/4 |
| P2 per-step reward（标量） | 2/4 | 3/4 |
| **P3 主任务进度分量（分量级）** | **2/4** | **3/4** |
| P4 BAC / NET（分量级复合） | 2/4 | 3/4 |

且在 crawl 上 **P3 是唯一全序命中的预测器**（BAC 全序错）。

**准确结论：被否的是 BAC 的过度工程化（瓶颈集 + 正负不对称 + 乘性敏感度），
不是"从 reward 分量看源价值"这个思路。** 该思路的最简实现已经拿到了全部可得的预测力。

## 2. 但 reward 侧存在天花板，本文给出机制级证据

slide 与 stair 是同一 reward 函数族（`ClimbingUpwards`：`stand_reward × small_control × move`）、
同一源库、同一瓶颈分量 `move`。所有 reward 侧指标对二者的判断几乎一致：

| 源在 target 上的 zero-shot 行为 | slide walk | stair walk |
|---|---:|---:|
| `move` 分量 | 0.821 | 0.760 |
| episode 长度 | 82.6 | 61.2 |
| 摔倒率 | 1.00 | 1.00 |
| NET (BAC) | 0.5153 | 0.4617 |
| per-step reward | 0.589 | 0.537 |

**两个 student 的自主学习速率也相同**（10k anchor → 20k 无源臂）：

| | J（自主增量） | progress_max_dx（自主增量） |
|---|---|---|
| slide | 29.50 → 51.18（+21.67） | 0.766 → 1.105（**+44.3%**） |
| stair | 28.96 → 44.30（+15.34） | 0.713 → 1.030（**+44.4%**） |

自主学习速率几乎逐位相同（+44.3% vs +44.4%）。
**这否证了"stair 的 student 自己学得快、所以源冗余"这一解释。**

而源的实际效果天差地别（源臂 vs 无源臂，同 seed 配对）：

| target | src | U | Δprogress_max_dx |
|---|---|---:|---:|
| slide | stand | −1.21 | +19.6% |
| slide | **walk** | **+56.95** | **+135.5%** |
| slide | run | +16.90 | +17.9% |
| stair | stand | −5.74 | −24.9% |
| stair | walk | +0.19 | **+0.1%** |
| stair | run | −1.11 | −4.0% |

**slide 上 walk 把 student 的行进距离教到了 2.6 倍；stair 上同一个 walk 源
对 student 的行为没有产生任何改变（+0.1%）。**

## 3. 因此缺失的维度不是 reward 分解的粒度

reward 侧的一切指标（return / per-step / 进度分量 / BAC）度量的都是

> **源做了什么**

而决定学习效用的是

> **源做的事情能否被 student 复制**

slide 与 stair 上源做的事几乎一样、student 的自学能力也一样，
**唯一不同的是源的行为有没有传过去**。U 跟随的是后者。

把这个量命名为**行为传递效力**（behavioral transfer efficacy）：

```
BTE_i = 源臂在关键行为维度上的水平 − 无源臂在同维度的水平   （同 seed 配对）
```

在两个 locomotion target 上，BTE 与 U 的关系比任何 reward 侧指标都紧：
唯一具有大 U 的单元（slide-walk，+56.95）也是唯一具有大 BTE 的单元（+135.5%）；
stair 三源 BTE 全部 ≈0 或负，U 也全部 ≈0。

## 4. 三条必须写清的限制

1. **BTE 是事后量。** 它需要跑完源臂才能计算，因此目前只是一个**中介变量**，
   不是预测器。要变成可用的东西，必须找到它的事前代理——这才是真正的科学问题，
   而且该问题不在 reward 侧。

2. **只有两个 locomotion target。** door 是反例：三源的 Δprogress 全部 ≈0
   （−1.4% / −0.2% / +2.9%），U 却全部显著负（−32.64 / −22.20 / −30.63）。
   合理的限定是 `progress_max_dx` 对 manipulation 任务不是关键行为维度
   （door 要开门，不是走路）；但这意味着 BTE 需要 per-target 地选择行为维度，
   这一步目前是人工的。door 的负 U 仍然无解释（`C(dose)` 假说已于 07-29 撤回）。

3. **本节全部是事后观察，未经前瞻验证。** 不得按此改写任何已冻结的裁决，
   也不得据此宣称新指标成立。

## 5. 与已封存信号族的关系

BTE 不属于此前封存的八族（zero-shot return、T⁰、T^critic、SIV、SHU、
P0 lease oracle、update-space influence、zero-shot 行为探针）——那八族全部在
reward/value 侧或纯行为侧。最接近的是 update-space influence，
但它度量的是**即时分布错配**，方向恰好相反（错配大意味着信息多）；
BTE 度量的是**行为是否被吸收**。

`fasttd3_ptf/ptf/compatibility.py` 的 `gaussian_action_compatibility_all`
是训练循环内的动作支撑判据，从未作为事前预测器被检验过。
它是 BTE 事前代理的一个候选起点，但"动作空间接近"未必等价于"行为可被学会"。

## 6. 数据

```
10k anchor 面板   docs/data/{slide,stair}_bac_gate_v1/source_free_eval/anchor_s*_step10000.json
20k 四臂面板      docs/data/{slide,stair,door}_*_gate_v1/source_free_eval/*_step20000.json
zero-shot 探针    logs/probe/transfer_map_v1.jsonl（ep_len_mean / fall / info_means）
预测器比较        docs/experiments/predictor_baseline_comparison_v1_results_20260729.md
```
