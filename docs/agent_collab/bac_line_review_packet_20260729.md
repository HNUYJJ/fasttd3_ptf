# BAC 主线审核包（持续更新，供外部独立审核）

> 起草 2026-07-29。本文件是**自包含**的审核材料：不读仓库其余部分也应能判断
> 主张是否被证据支持。每条主张都标注了证据、提交哈希与时间顺序
> （哪些写在揭盲之前、哪些在之后）。
>
> **审核者请重点攻击第 5 节（我自己列出的疑点）与第 4 节（已被我否证的东西）。**
> 我最担心的不是结论错，而是"预注册"这层保护是否真的成立。

---

## 0. 本轮最终状态（2026-07-29 更新，外部审核后）

**BAC 作为迁移性指标的主张已停止。** 正面比较表明它相对简单基线没有增量：

| 预测器 | 全序命中 | 最差源命中 |
|---|---:|---:|
| P1 episodic return | 1/4 | 1/4 |
| P2 per-step reward | 2/4 | 3/4 |
| P3 main progress component | 2/4 | 3/4 |
| **P4 BAC / NET** | **2/4** | **3/4** |

P2/P3/P4 三者打平；crawl 上 P3（只看主任务分量）全序命中而 BAC 全序错。
详见 `docs/experiments/predictor_baseline_comparison_v1_results_20260729.md`。

**保住的主张收缩为**：episodic return 因混入生存时长而错误；
把它换成 per-step reward 或主任务进度分量即可获得全部预测力——
这是一个简单修正，不是一个新指标。

以下 §1–§5 保留原始记录（含已被推翻的部分），供审核复盘推理链。

## 1. 一句话主张（**已收缩，见 §0**）

> 冻结源的迁移效用，取决于它覆盖了目标 reward 的哪个**分量**，而不是它拿到多少
> **总分**；标量 return 因为混入了组合算子与生存时长，会在一类任务上系统性反向。

## 2. 为什么会走到这一步（问题的来历）

本项目此前封存了八个迁移性信号族，全部失败：zero-shot return、T⁰、T^critic、
SIV、SHU、P0 lease oracle、update-space influence、zero-shot 行为探针。

原本的写法是"迁移效用不可预测"——这是投降式结论。
2026-07-28 重新检查时发现这八族有一个共同点：**都把 target 的 reward 聚合成一个标量**。

而 HumanoidBench 的 reward 是带组合算子的分量结构。标量 return 混合了三样东西：

```
return  =  每步分量质量  ×  分量组合算子  ×  生存时长
```

四种会让 return 反向的机制，均有仓库内实测支持：

| 机制 | 实例 | 现象 |
|---|---|---|
| `min` 瓶颈算子 | crawl `0.25·min(crawling, crawling_head)` | stand 把 crawling 抬到 0.845 却把 crawling_head 压到 0.343，min 取小者反比 zero(0.526) 差，return 却最高 |
| 乘性归零 | sit_hard `sc × sit_reward × dont_move` | stand 把 sit_reward 打到 0.005（zero 为 0.113） |
| 生存时长 | slide `sr × sc × move` | stand 每步 reward 0.178 最低，因不摔跑满 episode 而 return 88.5 最高 |
| 权重错配 | door `0.45·door_openness + 0.35·passage` | run 推进 passage，占最大权重的 door_openness 三源零覆盖 |

## 3. 指标与证据链

### 3.1 指标定义（零额外交互）

以 zero-action 基线 `x[zero]` 作 student 起点代理：

```
m_c        = ∂R/∂x_c 在 x[zero] 处
             加性项  m_c = w_c（被门控则再乘 gate[zero]）
             乘性因子 m_c = Π_{c'≠c} x_{c'}[zero]
B          = 按 m_c·(1 − x_c[zero]) 降序累计到 ≥50% 的分量集（瓶颈集）
Coverage_i = Σ_{c∈B}   m_c · max(0, x_c[i] − x_c[zero])
Damage_i   = Σ_{c∈all} m_c · min(0, x_c[i] − x_c[zero])
NET_i      = Coverage_i + Damage_i
```

代码 `scripts/analysis/bottleneck_aligned_coverage_v1.py`；
17 个任务的 reward 结构逐个从 `humanoid_bench/envs/*.py` 核准，
写在 `configs/reward_structure/humanoidbench_v1.py`；
单元测试 5/5（`tests/test_bottleneck_aligned_coverage.py`）。

### 3.2 时间顺序（这是审核的关键）

| 时间 | 事件 | 提交 |
|---|---|---|
| 07-28 | 17 任务结构核准 + 指标实现 + **7 个 target 的前瞻预测冻结** + slide 三级裁决冻结 | `33d4c92` |
| 07-28 | slide bank / 训练 / 评估 / **裁决脚本**（含硬编码的事前预测）提交 | `75772e8` |
| 07-28 | slide 12 臂训练与评估执行 | — |
| 07-28 | slide 揭盲 + 结果 | `e0f07a1` |
| 07-29 | stair 落地执行（预测早在 `33d4c92` 已冻结） | `32f66d0` |

**slide 与 stair 的预测是同一次冻结的**（`33d4c92`，07-28），
不存在"看到 slide 结果后再挑 stair"的可能。

### 3.3 回溯检验（事后，我不把它当证据）

| target | NET 排序 | 实测 U 排序 | 判定 |
|---|---|---|---|
| hurdle | run>walk>stand | run(+380)>walk(+105) | 命中 |
| crawl | walk>run>stand | run(−208)≈walk(−217)≫stand(−448) | 主判别命中，细序不命中 |
| door | run>stand>walk | walk(−22)>run(−31)>stand(−33) | **不命中** |

排序 1/3，主判别 2/3。且其中有一处事后调整（见 §4.1），故 crawl 不算独立验证。

### 3.4 slide 前瞻判决（已闭环，`BAC_SUPPORTED`）

| seed | student | stand | walk | run | D_walk | D_run |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 39.38 | 37.54 | 105.04 | 65.67 | +67.50 | +28.12 |
| 2 | 63.61 | 59.13 | 111.22 | 70.88 | +52.09 | +11.75 |
| 3 | 50.54 | 53.23 | 108.11 | 67.67 | +54.88 | +14.44 |

```
D_walk = +58.16   90%CI [+44.32, +72.00]   3/3 seed 为正
D_run  = +18.11   90%CI [ +3.31, +32.91]   3/3 seed 为正
U(stand) = −1.21  90%CI [−7.33, +4.91]     uncertain —— 唯一没有正迁移的源
```

事前 return 说 stand(88.5) 最好，实测 stand 垫底；事前 return 说 run(27.8) 最差，
实测正迁移 +16.90。Spearman ρ = −0.5。

**剂量验收**（决定这次实验能否用于裁决）：三源 behavior share 0.4766–0.4846
（跨源最大差 0.8%），critic share 0.4988–0.4991（差 0.03%）。排序差异不可由剂量失配解释。

**评估**：128 deterministic episodes（16 eval seeds × 8 ranks），前 32 与既往面板逐位兼容；
纯 source-free student（源在评估时不在场）。episode SE 1.00–2.47，效应量 +18~+58。

### 3.5 stair 重复验证（已闭环：`BAC_PARTIAL`，但**无判决力**）

详见 `docs/experiments/stair_bac_gate_v1_results_20260729.md`。

```
D_walk = +5.93   90%CI [ −3.79, +15.65]   不显著
D_run  = +4.63   90%CI [−12.72, +21.98]   不显著
三个源的 U 全部 uncertain（stand −5.74 / walk +0.19 / run −1.11）
```

表面排序 `walk>run>stand` 与 NET 一致，**但不成立**：

1. `D_run` 的 per-seed 值是 **−1.61 / +16.51 / −1.01**，
   3 个 seed 中 2 个方向与预测相反，均值转正全靠 seed 2（其 stand=30.13 明显偏离 41.96/43.59）。
   去掉 seed 2 后 `D_run` 均值为 **−1.3（负）**。
2. stair 上迁移效应本身几乎不存在：|U|max = 5.74，而 stand 的跨 seed sd = 7.35，
   **效应量小于 learner-seed 离散**。评估反而更干净（episode SE 0.38–1.59，优于 slide）。
3. **我的疏忽（核实后比原先承认的更严重）**：
   `label_identifiability_audit_20260727.md:89` 已记录 stair 保守 `U/trend = 3.57`
   （锚点 crawl 0.83 可测 / door 1.42 选中 / cabinet 10.31 不可测），
   第 100 行更把 **stair 与 slide 一并列入排除名单**。
   **我选作判决场的两个 target 都在本项目自己 07-27 审计的排除名单上，而我没有去查。**
   限定（采纳外部审核）：该筛选只能估计 baseline 噪声与所需最小效应量，
   无法在不看源臂结果时知道真实效应接近零，故不能断言它一定能预先识别本次失败；
   准确表述是"stair 事前已是统计效率较差的判决场，不应优先于更可测的任务"。

**因此 BAC 目前只有 slide 一个有效的前瞻验证点，不能声称已获重复验证。**

### 3.6 stair 暴露了什么（**表述已按外部审核修正**）

~~原写法："实测学习效用相差约 300 倍"~~ —— **撤回**。stair 的 U(walk)=+0.19
区间为 [−5.35, +5.72]，统计上与零不可区分；以不可区分于零的估计作分母，
倍数可任意放大。

**修正后**：slide 的 walk 效应是明确正迁移 +56.95；stair 的 walk 效应接近零且无法判定。
**高且相近的 BAC 分数没有对应到一致的绝对迁移效用。**

~~原写法："结果不同说明 slide 与 stair 确实独立"~~ —— **撤回，属逻辑错误**。
二者共享同一 `ClimbingUpwards` reward 函数、同一源库、同一瓶颈分量 `move`、
高度相似的 zero-shot 剖面，**只有 terrain dynamics 不同**。结果不同说明
terrain dynamics 对学习效用重要、且 BAC 漏掉了这一信息，
但不会使二者在"reward decomposition 是否普适"层面成为独立验证。

**修正后**：stair 是同 reward family 下的跨地形 robustness test。
结果表明仅凭 reward structure 与 zero-shot 分量覆盖，
无法预测 source intervention 是否会产生有实际意义的学习增益。

**真正的新信息**（采纳外部审核的重述）：

> 高 BAC 只表示源在**行为层面**覆盖了目标 reward 的主要分量，
> 并不表示这种覆盖会转化为非零的 student **学习**收益。

slide（BAC 高、per-step 高、学习效用显著正）与 stair（BAC 高、per-step 高、
学习效用接近零）共同分离出：**behavioral component compatibility ≠ delayed learning utility**。
这与本项目此前的核心发现一致——BAC 改善了行为语义判断，
仍未跨过"行为价值 → 学习价值"的鸿沟。

---

## 4. 我已经自己否证/撤回的东西（请重点复核这些是否撤得干净）

### 4.1 一处事后调整（已在预注册文档记录）

指标初稿只用 `Coverage` 作主量，它在 crawl 上给 walk/run 打正分，与实测（全负）矛盾。
加入 `Damage` 后 NET 才与实测一致。**此调整发生在看到 crawl 结果之后**，
因此 crawl 不构成对 NET 的独立验证。slide/stair 是在此调整之后冻结的，不受影响。

### 4.2 `C(dose)` 机会成本假说 —— 已被 slide 否证并撤回

door 三源 NET≈0 而 U 全负（−22~−33），我曾据此提出 `U_i ≈ α·NET_i − C(dose)`，
即存在与源身份无关的固定交互预算机会成本。

slide 直接反驳：stand 的 NET = 0.0129（≈0），实测 U = −1.21，90%CI **跨零**。
若存在通用固定机会成本，stand 应显著为负。假说撤回，不进论文。

**后果**：door 的"三源全负"重新成为 open 问题，目前无解释。

### 4.3 NET 的定量刻度 —— 不成立

```
NET  walk/run = 0.5153 / 0.5086 = 1.013   （差 1.3%）
U    walk/run = 56.95  / 16.90  = 3.371   （差 237%）
```

walk > run 这一位虽然对上了，但 1.3% 的 NET 差不具备分辨 3.4 倍效用差的能力，
**我不把这一位计为命中**。稳健的只有 `stand vs {walk, run}` 主判别（NET 差 40 倍）。

α 在 target 内部也不自洽（door 内 844→38300，slide 内 33→110）。

**后果**：BAC 只支持"谁不覆盖瓶颈"的判别，不支持定量效用预测。
**按 NET 加权 replay 这条下游用法目前没有证据基础，不能上。**

### 4.4 Door prefix handoff —— 已停

框架预测 `Δ_placement ≈ 0`（door 瓶颈分量三源零覆盖，改变时间放置不改变覆盖哪个分量）。
此前的停止理由是工程性的（behavior share 0.3896 未达 [0.45,0.55] 门槛），现替换为上述科学理由。

---

## 5. 我自己列出的疑点（请优先攻击）

1. **`BOTTLENECK_MASS = 0.50` 这个阈值是怎么定的？** —— 已做敏感性分析，部分成立

   该阈值确实是在看过 door 数据之后写下的，属于 §4.1 同类风险。
   我对 mass ∈ {0.30, 0.40, 0.50, 0.60, 0.70} 重跑了全部 target：

   | target | 0.30 | 0.40 | 0.50 | 0.60 | 0.70 | |
   |---|---|---|---|---|---|---|
   | slide | wa>ru>st | wa>ru>st | wa>ru>st | wa>ru>st | wa>ru>st | 稳定 |
   | stair | wa>ru>st | wa>ru>st | wa>ru>st | wa>ru>st | wa>ru>st | 稳定 |
   | crawl | wa>ru>st | wa>ru>st | wa>ru>st | wa>ru>st | wa>ru>st | 稳定 |
   | hurdle | ru>wa>st | ru>wa>st | ru>wa>st | ru>wa>st | ru>wa>st | 稳定 |
   | pole / sit_hard / maze | — | — | — | — | — | 均稳定 |
   | **door** | ru>st>wa | ru>st>wa | ru>st>wa | ru>wa>st | ru>wa>st | **★随阈值变** |

   结论：**7/8 稳定；唯一敏感的 door 恰是唯一不命中的 target**，
   且 door 在任何阈值下都不命中实测（实测 walk>run>stand，各阈值下预测为
   ru>st>wa 或 ru>wa>st，头名均错）——即调阈值救不了 door，
   我也没有借选阈值优化结果。

   **判决场 slide 与 stair 的预测对该阈值完全不敏感**，故本轮结论不依赖这个可疑参数。
   残留风险：该分析只覆盖排序，未覆盖 `SIGN_EPS` / `SEPARATION_MIN` 两个阈值。

2. **slide 与 stair 是否构成独立重复？** 两者的 reward 结构同为
   `ClimbingUpwards`（`stand_reward × small_control × move`），源相同，
   zero-shot 分量也高度相似。stair 通过可能只是 slide 的近乎复制，
   而非独立证据。若如此，"重复验证"的说服力应打折。

3. **正负不对称是否为拟合手段？** `Coverage` 只在瓶颈集上取正向，`Damage` 在全分量上取负向。
   我给的语义理由（非瓶颈已饱和 / 乘性下压垮不可补偿）是事后构造的，
   虽然自洽，但需要独立判断它是否只是为了让 crawl 对上。

4. **样本量**：每个 target 只有 3 个 learner seed，df=2。90% t 区间在 df=2 时非常宽，
   D_run 的下界 +3.31 已相当接近 0。

5. **只有两个 target 的前瞻证据，且都是乘性结构** —— 已查明结构性原因，边界应改写

   按 reward 结构分组看 NET 的分离度 `spread = max(NET) − min(NET)`：

   | 结构 | target（spread） |
   |---|---|
   | 加性 | powerlift **0.0000**、room **0.0041**、spoon 0.0200、**door 0.0255**、window 0.0337 |
   | 乘性/门控 | slide **0.5023**、stair **0.4571**、pole 0.2604、balance_hard 0.1523、hurdle 0.0547、maze 0.0453、crawl 0.0398、sit_hard 0.0334 |

   **加性任务的 spread 系统性地小（全部 ≤ 0.034），乘性任务可大到 0.5**，
   相差一个半量级。原因是结构性的：加性 manipulation 任务的瓶颈分量
   （`door_openness` / `dumbbell_lifted` / `room_object_organized` / `spoon_in_cup`）
   三个 loco 源**全都零覆盖**，于是 Coverage 全≈0、spread 全≈0；
   而乘性任务的瓶颈往往是 `move`，恰是 loco 源天然能大幅推进的因子。

   **这重新解释了 door 的"不命中"**：不是指标给错了排序，而是 door 的 spread
   仅 0.0255、勉强超过 `SEPARATION_MIN = 0.02` 这道门槛，才被允许输出排序。
   若门槛设为 0.05，door 会与 powerlift / room 一样判为"不定序"，
   "不命中"就不存在了。

   **但我不调这个阈值**——事后调阈值消除自己的不命中，正是项目纪律禁止的操作。
   如实保留 door 为不命中，同时记录这一结构性观察。

   适用边界应改写为：**BAC 只在源能推进瓶颈分量的任务上有分辨力**；
   对 loco 源 + 加性 manipulation 任务，它正确地报告"无分辨力"而非给出错误排序。
   这也让预注册中 powerlift / room 的 `ALL_NEGLIGIBLE` 预测有了机制解释，
   且该预测是可证伪的（若三源在这两个任务上出现显著效用差，则本节论断被推翻）。

   附带一个应记录的例外：**balance_hard 的瓶颈集是 `stand_reward`，是通用姿态项**
   （唯一如此的 target）。该任务上 BAC 退化为"谁站得稳"，与 return 高度相关，
   没有增量信息——不应把它算作 BAC 的适用场。

6. **UNMEASURABLE 的一致点是否为循环论证？** 指标把 cabinet/package/push/truck 判为
   不可测，而这四个恰是既有数据里标签测不出来的。我称之为"独立一致点"，
   但两者可能同源于"return 尺度无界"这一事实，不构成独立确认。

---

## 6. 当前状态与下一步

- slide：`BAC_SUPPORTED`，已闭环（`e0f07a1`）
- stair：anchor 训练中（07-29 起）
- 其余冻结预测（pole / sit_hard / maze / powerlift / room）：**本轮不跑**，
  以防事后挑选有利 target
- open 问题：door 三源全负的机制（§4.2 撤回后无解释）

## 7. 复核入口

```
预注册与冻结预测   docs/experiments/bottleneck_aligned_coverage_v1_prereg_20260728.md   33d4c92
slide 结果         docs/experiments/slide_bac_gate_v1_results_20260728.md               e0f07a1
指标实现           scripts/analysis/bottleneck_aligned_coverage_v1.py
reward 结构规格    configs/reward_structure/humanoidbench_v1.py
单元测试           tests/test_bottleneck_aligned_coverage.py
裁决脚本           scripts/analysis/analyze_{slide,stair}_bac_gate_v1.py
原始面板           docs/data/{slide,stair}_bac_gate_v1/source_free_eval/*.json
zero-shot probe    logs/probe/transfer_map_v1.jsonl（72 cell，含 info_means 分量）
```
