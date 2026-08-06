# 结果：零训练 progress 粗筛 —— `HOLDOUT_FAILED`

## 并给出迄今最强的"行为量与学习价值反向"反例

> 2026-08-04。预注册 `docs/experiments/progress_screen_v1_prereg_20260804.md`，
> 探针 `scripts/analysis/probe_source_progress_screen_v1.py`，
> 裁决 `scripts/analysis/analyze_progress_screen_v1.py` ——
> 三者均在任何探针数据产出之前于 `788afa0` 冻结提交。
> 裁决输出 `docs/data/progress_screen_v1/results.json`。

## 1. 裁决

```
VERDICT: HOLDOUT_FAILED

design data
  lo = max_i P(i, crawl) = 14.302  (run)
  hi = P(run, hurdle)    = 22.521 ± 2.057
  theta = sqrt(lo × hi)  = 17.947          （lo < hi 成立，design 阶段未失败）

holdout
  P(walk, slide) = 1.814 ± 0.126  ≤  theta = 17.947      ← 已知最优源被误杀
```

## 2. 完整测量（zero-shot，32 episodes，deterministic，1000 步）

`P` = 每 episode `max_t(x_t − x_0)` 的均值，单位米。

| 环境 | stand | walk | run | 该 target 的真实最优源 |
|---|---:|---:|---:|---|
| crawl（三源**全有害**） | 0.221 | 3.664 | **14.302** | 无（U = −448 / −217 / −208）|
| hurdle | 0.188 | 8.717 | **22.521** | run（U = +379.66）|
| slide | 0.183 | **1.814** | 1.753 | walk（U = +56.95，`GEN_OK` 6 learner）|
| walk-v0（平地参考，不进判据）| 0.173 | 31.388 | 115.470 | — |

## 3. 核心反例：行为量与学习价值**反向**，倍率 7.9×

```
crawl 上的 run ：位移 14.302  →  真实效用 −208.070   （有害）
slide 上的 walk：位移  1.814  →  真实效用  +56.95    （最有用）
```

**有害源的前进距离是有用源的 7.9 倍。**

这不是"信号弱"或"排序偶尔出错"，是系统性反向。它比 M19 的 door 反例
（run 位移 +58% 却有害 / walk −61% 却最不负）强一个数量级，
可作为族 1（zero-shot 行为 return / 位移）失败的最强证据补充。

**位移在 slide 上连排序都做不到**：walk 1.814 vs run 1.753，差距 0.061，
而二者的真实效用是 +56.95 vs +16.90。

### 3.1 失败与阈值取法无关：任何单调阈值都不可能成立

`HOLDOUT_FAILED` 是按预注册的具体 θ 取法裁出的，但失败并不依赖那个取法。
一个只用两个测量值的纯代数论证：

```
粗筛要成立，必须同时满足
    P(run,  crawl) < θ        （拒绝有害源）
    P(walk, slide) > θ        （保留有用源）
即需要
    14.302 < θ < 1.814        ← 空集
```

**不存在任何阈值能同时做对这两件事**，因为有害源的位移比有用源高 7.9 倍。
换 θ 的取法、换分位数、换归一化方式都不能改变这个不等式的方向。
这是本实验最硬的一条结论，且与 §4 中位移由何种行为产生**完全无关**。

### 3.2 需要区分的两件事："不相关"与"反向"

"走得远不等于 return 高"是显然的——crawl 的 reward 是
`(0.1·small_control + 0.25·min(crawling,crawling_head) + 0.4·move + 0.25·xquat) × in_tunnel`，
位移只与 `move` 有关，姿态项与门控项都可能同时下降。**这一点不是本实验的发现。**

本实验的发现是更强的一件事：位移与学习效用不只是弱相关或不相关，
而是在这三个 target 上**系统性反向**。若只是不相关，噪声会让阈值时对时错；
反向意味着任何单调阈值都会**系统性地**选中有害源、排除有用源，即 §3.1。

## 4. 预注册时的机制假设被证伪（如实记录）

预注册 §2 的机制推理是：

> crawl 隧道顶高仅 1.15 m，站立源**进不去**，任务进度为零。

**实测证伪**：run 源在 crawl 上前进了 **14.302 m**，并非"进度为零"。

**它是怎么前进的**——由同批探针采集的 `in_tunnel` 分量给出直接答案：

| crawl | `in_tunnel` | `crawling` | `move` | dx | return |
|---|---:|---:|---:|---:|---:|
| stand | **1.0000** | 0.849 | 0.181 | 0.221 | 300.15 |
| walk | **0.6115** | 0.807 | 0.338 | 3.664 | 257.96 |
| run | **0.4175** | 0.829 | 0.414 | 14.302 | 174.56 |

`in_tunnel = rewards.tolerance(imu_y, bounds=(-1,1), margin=0)` 是**硬门控**
（`margin=0` ⇒ 取值只有 0 或 1），故其均值即"处于隧道横向范围内的时间比例"。
`stand` 恒为 1.0000（不移动，始终在中线）为该读法提供了内部校验。

**`run` 有 58% 的时间步处在隧道横向范围之外**，`walk` 为 39%。
即位移主要来自**横向偏离出任务区域后在隧道外前进**，而不是穿过隧道。
`crawling = 0.829` 也不支持"摔倒后翻滚"。

> **更正**：本文初稿写的"它穿过了大半条隧道"与"摔倒后翻滚前进"
> **均无证据支撑**，已按上表更正。位移前进不但不等于任务前进，
> 在本例中它恰恰**来自绕开任务**。

反过来，slide 上 walk 源只走 1.814 m（平地能走 31.388 m，被斜坡阻断 94%），
却是三个源里唯一稳定正迁移的。**"被 target 结构阻断"不蕴含"对学习无用"。**

**这不改变裁决，也不改变 §3 的结论**——§3 的不可能性论证只用到位移本身，
与位移由何种行为产生无关。

## 5. 这对方向的意义

结合十一族的既有失败，本实验把"零训练/零交互准入"这条路补完了最后一块：

| 空间 | 状态 |
|---|---|
| 即时 reward / return | 族 1–8、10 已失败；crawl 上 `reward_gain` 把有害源判为 `+6.35/+5.43`（真实 −217/−208）|
| 静态任务规格 | 族 11 已失败；slide 与 stair 共用同一份 `ClimbingUpwards.get_reward`，效用 +56.95 vs +0.19 |
| **任务进度（本实验）** | **`HOLDOUT_FAILED`**，且与效用反向 7.9× |

按预注册 §7/§8，**不得**换指标（速度 / 关节幅度 / 姿态 / 覆盖度…）
在同一批 target 上抢救。**零训练准入方向到此关闭。**

正面推论：准入判断只能来自**实测**——即在真实 learner 状态上花交互代价去测
（`RACING_VIABLE`，K\* = 10000）。这与 `impossibility_characterization` §3.3 的
刻画一致：`U ~ p(U | source, target, θ_t, D_t, occupancy_t, channel, dose, K)`，
而任何只读 (source, target) 的 zero-shot 量都在估计一个不存在的点函数。

## 6. 能与不能声称

**能**：

- 在这三个 locomotion target 上，zero-shot 前进进度**不能**承担绝对准入，
  且与真实效用系统性反向（7.9×）；
- 该反例强于 M19，可加强族 1 的失败刻画。

**不得**：

1. 不得声称"任何进度类指标都不行"——本实验只测了前进位移这一种进度定义；
   但按 §8 的预注册约束，**不得在同一批 target 上换指标继续尝试**；
2. 不得外推到 manipulation target（本实验只覆盖 locomotion，见预注册 §3）；
3. §4 的姿态解释是**未验证假设**，引用时必须标注；
4. 本实验无 learner seed，不产生 learner 级统计结论。

## 7. 数据

```
预注册 / 脚本    788afa0（均先于探针数据冻结）
探针原始输出     docs/data/progress_screen_v1/probe.json      （12 组 × 32 episodes）
裁决输出         docs/data/progress_screen_v1/results.json
运行日志         logs/probe/progress_screen_v1.log
真实效用来源     crawl  docs/experiments/crawl_equal_dose_source_calibration_v1_results_20260723.md
                 hurdle docs/experiments/hurdle_equal_dose_source_calibration_multiseed_v1_results_20260723.md
                 slide  docs/experiments/slide_generalizability_v1_results_20260731.md
```
