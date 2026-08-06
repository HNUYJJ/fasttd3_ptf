# 预测器正面比较：BAC 相对简单基线**没有增量**

> 2026-07-29。回应外部审核（ChatGPT）提出的阻塞项。零训练成本，只用既有数据。
> **结论对本项目提出的 BAC 指标不利，如实记录并据此收缩主张。**

## 1. 被检验的问题

外部审核指出一个此前未做的检验：

> slide 上 mean per-timestep reward 也预测 `walk>run>stand`；stair 上同样如此。
> 因此尚未证明"reward 分量分解"比"去掉 episode 长度"这一简单修正有任何增量。

若成立，BAC 的全部复杂度——17 个任务的结构核准、瓶颈集、Coverage/Damage
正负不对称、乘性边际敏感度——都是无增量的。

## 2. 四个预测器（均为零额外交互）

| | 定义 |
|---|---|
| P1 episodic return | 八信号族之一，已知被生存时长污染 |
| P2 per-step reward | **最简修正**：只去掉 episode 长度 |
| P3 main progress component | 取非通用分量中边际敏感度最大的**单个**分量 |
| P4 BAC / NET | 本项目提出的指标 |

评分对象：有配对学习效用真值**且该真值可判定**的 target。
stair 被排除——其三源 U 全部跨零，无判决力，不能用于区分预测器。

## 3. 结果

```
### hurdle          实测 U 排序 run>walk
    P1/P2/P3/P4     全部 run>walk          四者均命中（此 target 无区分力）

### crawl           实测 U 排序 run>walk>stand
    P1_return       stand>walk>run         全序✗ 最差✗
    P2_per_step     stand>walk>run         全序✗ 最差✗
    P3_progress     run>walk>stand         全序✓ 最差✓
    P4_BAC          walk>run>stand         全序✗ 最差✓

### door            实测 U 排序 walk>run>stand
    P1_return       run>stand>walk         全序✗ 最差✗
    P2_per_step     run>walk>stand         全序✗ 最差✓
    P3_progress     walk>stand>run         全序✗ 最差✗
    P4_BAC          run>stand>walk         全序✗ 最差✗

### slide           实测 U 排序 walk>run>stand
    P1_return       stand>walk>run         全序✗ 最差✗
    P2_per_step     walk>run>stand         全序✓ 最差✓
    P3_progress     run>walk>stand         全序✗ 最差✓
    P4_BAC          walk>run>stand         全序✓ 最差✓
```

| 预测器 | 全序命中 | 最差源命中 |
|---|---:|---:|
| P1 episodic return | 1/4 | 1/4 |
| P2 per-step reward | **2/4** | **3/4** |
| P3 main progress component | **2/4** | **3/4** |
| **P4 BAC / NET** | **2/4** | **3/4** |

## 4. 裁决

**P2、P3、P4 三者完全打平。BAC 没有在任何一个 target 上严格优于两个简单基线。**

更不利的是：在 crawl 上，**P3（只看主任务分量，一行代码）全序命中，而 BAC 全序错**
（BAC 把 walk 排在 run 前，实测 run 略优）。即 BAC 的额外复杂度在此处不仅无增量，
表现还差于更简单的形式。

按外部审核预先给出的分支裁决执行：

> 如果 BAC 没有明显优于 per-step/progress 基线：停止把 BAC 当迁移性指标，
> 只保留"生存时长校正与任务分量诊断"的结果。

**BAC 作为迁移性指标的主张就此停止。** 不做以下任何抢救：
不调 `BOTTLENECK_MASS` / `SIGN_EPS` / `SEPARATION_MIN`、
不改瓶颈集定义、不换 Coverage/Damage 的组合方式、不加权重拟合。

## 5. 什么被保住了（这部分证据未受影响）

1. **episodic return 是错误的聚合，因为它混入了生存时长。**
   最差源命中率从 P1 的 1/4 提升到 P2 的 3/4，只需把 return 换成 per-step reward。
   slide 是这一点的直证：stand 的 `move` 仅 0.188（几乎不移动），
   靠不摔跑满 episode 刷出全场最高 return 88.5，实测却是唯一没有正迁移的源。

2. **reward 分量比标量 return 更符合任务语义。**
   P3（主任务分量）同样达到 3/4，且在 crawl 上是唯一全序命中的预测器。

3. **17 个任务的 reward 组合结构核准**（`configs/reward_structure/humanoidbench_v1.py`）
   与四种 return 失效机制的实例（min 算子 / 乘性归零 / 生存时长 / 权重错配）
   是从源码和既有 probe 直接读出的事实，不依赖 BAC 的预测力。

4. **slide gate 的实验结果本身有效**：walk/run 显著优于 stand，
   `D_walk=+58.16 [+44.32,+72.00]`、`D_run=+18.11 [+3.31,+32.91]`，3/3 seed 一致，
   而 return 的事前排序完全反向。这一条不因 BAC 无增量而失效——
   它证伪的是 episodic return，不是证实 BAC。

**因此正确的主张变成一个更简单、也更容易被采用的修正**：
用 per-step reward 或主任务进度分量替代 episodic return；
而不是一个新的复合指标。

## 6. 三处表述纠正（采纳外部审核）

### 6.1 "相差约 300 倍"是非法的效应表述 —— 已撤回

我此前写"slide 与 stair 的 U(walk) 相差约 300 倍（+56.95 vs +0.19）"。
但 stair 的 +0.19 区间为 [−5.35, +5.72]，**统计上与零不可区分**；
用一个不可区分于零的估计作分母，倍数可任意放大。

改为：**slide 的 walk 效应是明确的正迁移 +56.95；stair 的 walk 效应接近零且无法判定。
高且相近的 BAC 分数没有对应到一致的绝对迁移效用。**

### 6.2 slide 与 stair 不因结果不同而成为独立机制验证 —— 已修正

我此前写"结果差 300 倍，说明二者确实独立"。这是逻辑错误。
两者共享同一 `ClimbingUpwards` reward 函数、同一 stand/walk/run 源库、
同一 BAC 瓶颈分量 `move`、高度相似的 zero-shot 分量剖面，**只有 terrain dynamics 不同**。

准确表述：**stair 是同 reward family 下的跨地形 robustness test。
结果表明仅凭 reward structure 与 zero-shot 分量覆盖，
无法预测 source intervention 是否会产生有实际意义的学习增益。**

### 6.3 "可测性筛选本可排除 stair" —— 部分成立，但事实比我说的更严重

外部审核正确指出：该筛选只能估计 baseline 噪声、自然学习趋势与所需最小效应量，
**无法在不看源臂结果的情况下知道真实效应本身接近零**，故不能断言它一定能预先识别本次失败。

但核实后发现事实比我原先承认的更严重：
`docs/experiments/label_identifiability_audit_20260727.md` 中

- 第 89 行：stair 保守 `U/trend = 3.57`（锚点：crawl 0.83 可测、door 1.42 选中、cabinet 10.31 不可测）
- 第 100 行：**stair 与 slide 一并被明确列入排除名单**
  （"保守 U/trend ≥ 1.17 且噪声或 seedCV 明显劣于 door"）

**即我选作判决场的两个 target，都在本项目自己 07-27 审计的排除名单上，而我没有去查。**
准确表述：stair 事前已是统计效率较差的判决场，不应优先于更可测的任务。

（附注：slide 同在排除名单却给出了清晰效应，说明该筛选偏保守，会误排一些可用 target；
但它对 stair 的判断是正确的。）

## 7. 数据与复核

```
比较脚本    scripts/analysis/predictor_baseline_comparison_v1.py
输出        docs/data/predictor_baseline_comparison_v1.json
真值来源    各 gate 的 128-ep 冻结面板（hurdle/crawl/door/slide）
可测性审计  docs/experiments/label_identifiability_audit_20260727.md:89,100
```
