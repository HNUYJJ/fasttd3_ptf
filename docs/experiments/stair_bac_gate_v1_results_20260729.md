# Stair BAC 重复验证结果：BAC_PARTIAL —— 但真实情况是**该判决场无判决力**

> 2026-07-29。预测冻结于 `33d4c92`（与 slide 同一次），执行 `32f66d0`。
> 本文件的结论比裁决标签更严格：**stair 不构成对 BAC 的有效重复验证**，
> 且暴露了我在选择判决场时的一处方法论疏忽。

## 1. 裁决输出

```
VERDICT: BAC_PARTIAL
```

| seed | student | stand | walk | run | D_walk | D_run |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 44.77 | 41.96 | 45.11 | 40.36 | +3.14 | **−1.61** |
| 2 | 45.86 | **30.13** | 42.69 | 46.64 | +12.56 | **+16.51** |
| 3 | 42.28 | 43.59 | 45.67 | 42.57 | +2.09 | **−1.01** |

```
D_walk = +5.93   90%CI [ −3.79, +15.65]   不显著
D_run  = +4.63   90%CI [−12.72, +21.98]   不显著
U(stand) = −5.74 [−20.73, +9.25]  uncertain
U(walk)  = +0.19 [ −5.35,  +5.72]  uncertain
U(run)   = −1.11 [ −5.94,  +3.72]  uncertain
```

表面上排序 `walk>run>stand` 与 NET 预测逐位一致、与 return 预测反向。
**但这个"命中"不成立**，理由见 §2。

## 2. 为什么这不是一次有效的重复验证

### 2.1 排序命中完全由单个 seed 驱动

`D_run` 的三个 per-seed 值是 **−1.61 / +16.51 / −1.01**：
**3 个 seed 中有 2 个方向与预测相反**，均值转正完全靠 seed 2。
而 seed 2 的 stand = 30.13，明显偏离 seed 1/3 的 41.96 / 43.59。

去掉 seed 2 后：`D_walk` 均值 +2.6，`D_run` 均值 **−1.3（负）**。

裁决判 `BAC_PARTIAL` 的依据是 `stand_is_worst`，而 stand 之所以最差，
同样来自 seed 2 的那一个低值。

### 2.2 stair 上迁移效应本身几乎不存在

| | slide | stair |
|---|---:|---:|
| \|U\|max | **56.95** | **5.74** |
| D_walk 三 seed | +67.5 / +52.1 / +54.9 | +3.1 / +12.6 / +2.1 |
| D_run 三 seed | +28.1 / +11.8 / +14.4 | −1.6 / +16.5 / −1.0 |
| episode SE | 1.00–2.47 | **0.38–1.59** |
| stand 跨 seed sd | 11.15 | 7.35 |

**评估本身更干净**（stair 的 episode SE 更小），问题不是测量噪声，
而是**效应量比 learner-seed 离散还小**：|U| ≤ 5.74，而 stand 的跨 seed sd = 7.35。

这正是教训 M16 的另一面：episode SE 小不代表能测出效应，
决定判决力的是 learner-seed 离散。

### 2.3 我的方法论疏忽

项目里早有**标签可测性前置筛选**（`docs/experiments/label_identifiability_audit_20260727.md`，
锚点：crawl U/trend = 0.83 可测、cabinet 10.31 不可测），
其存在目的就是避免选到效应量小于噪声的 target。

**我在把 stair 选为重复验证场时没有应用这个筛选。**
若事前跑一遍，stair 很可能会被判为不适合作判决场。这是流程上的疏漏，
不是结果不利之后的辩解——记录在此以便复核。

## 3. 一个比 walk/run 更严重的定量失效

slide 与 stair 的 reward 结构完全相同（`ClimbingUpwards`：`stand_reward × small_control × move`），
源相同，zero-shot 分量也高度相似：

| | zero | stand | walk | run | NET(walk) |
|---|---:|---:|---:|---:|---:|
| slide | 9.3 | 88.5 | 45.7 | 27.8 | 0.5153 |
| stair | 9.3 | 84.9 | 31.8 | 26.2 | 0.4617 |

NET 对两者的预测几乎相同（walk：0.5153 vs 0.4617，差 12%），
**而实测学习效用相差约 300 倍**（U(walk)：+56.95 vs +0.19）。

**结论：BAC 在跨 target 比较上完全没有校准。**
它此前已被证明不能定量分辨同一 target 内的诸源（slide §3.1），
现在进一步表明它也不能预测不同 target 之间迁移效应的量级。

一个副产品：这否证了审核包 §5 疑点 2 的担忧方向——slide 与 stair
**不是**近乎复制（同结构同源却效用差 300 倍），它们确实独立；
但这个独立性暴露的是 BAC 的缺陷，而非增强了证据。

## 4. 当前证据状态的准确表述

- **slide**：`BAC_SUPPORTED`，强命中，效应量大、3/3 seed 一致 —— 有效
- **stair**：`BAC_PARTIAL`，但**无判决力**，既未证实也未证伪 —— 应计为无效判决场
- 因此 **BAC 目前只有一个有效的前瞻验证点**

可以说的只有：在 slide 上，当源在瓶颈分量的覆盖度相差 40 倍时，
BAC 正确识别了 return 完全反向的排序。
**不能**说 BAC 已获重复验证。

## 5. 不做什么

- **不换一个新 target 重试**。冻结预测集里 pole 的 spread 最大（0.2604），
  看起来诱人，但在 stair 失利后改挑 pole 就是事后挑靶。
- **不调 `SEPARATION_MIN` / `SIGN_EPS`** 让 stair 变成命中。
- **不去掉 seed 2**。它不是故障，是真实的 learner-seed 变异。

若要继续，唯一正当的路径是：**先用只依赖 student/scratch 数据的可测性判据
筛选候选判决场**（该筛选不得读取任何源臂结果），把筛选规则与候选名单
一并预注册，再从中选取。这一步本应在 stair 之前做。

## 6. 数据

- 12 份冻结评估：`docs/data/stair_bac_gate_v1/source_free_eval/*.json`
- 裁决输出：`docs/data/stair_bac_gate_v1/stair_bac_gate_v1_results.json`
- 剂量验收：behavior share 0.4651–0.4829（跨源最大差 1.8%），
  critic share 0.4988–0.4992（差 0.04%）。
  最低的是 walk（0.4651），即 BAC 预测最优的源拿到的剂量反而最少，
  剂量方向对结论是保守的。
