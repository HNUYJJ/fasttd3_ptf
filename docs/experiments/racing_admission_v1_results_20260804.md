# 结果：racing 的准入能力 —— `ADMISSION_VIABLE`

## 一次 K=10000 的测量可同时决定"要不要用源"与"用哪个"

> 2026-08-04。预注册 `docs/experiments/racing_admission_v1_prereg_20260804.md`，
> 训练/评估/裁决脚本链 `55d1423` —— 四者均在 crawl / slide 的**任何**评估结果
> 产出之前冻结提交。裁决输出 `docs/data/racing_admission_v1/results.json`。

## 1. 裁决

```
VERDICT: ADMISSION_VIABLE        false_admit = 0     false_reject = 0
```

补齐了 `M31(d)` 点名、`racing_reject_door` 系列四轮未能裁决的唯一机制缺口。

## 2. 主结果：`U ± paired SE`（K=10000，逐 episode 配对，128 ep）

`admit(T,s) = ∃i: U_i > 2·SE_i`（`*` 标记满足该条件的源）

| target | seed | stand | walk | run | admit |
|---|---|---:|---:|---:|---|
| **crawl**（负例，期望全拒） | s1 | −44.06 ± 7.84 | −258.77 ± 18.45 | −79.17 ± 9.66 | **False** |
| | s2 | −186.76 ± 12.51 | −196.22 ± 15.21 | −181.86 ± 12.49 | **False** |
| | s3 | −162.98 ± 10.13 | −60.31 ± 13.86 | −70.35 ± 11.64 | **False** |
| **hurdle**（正例） | s1 | +2.57 ± 0.87\* | +62.95 ± 2.03\* | +102.19 ± 3.82\* | **True** |
| | s2 | +18.67 ± 0.90\* | +42.32 ± 2.78\* | +110.51 ± 3.70\* | **True** |
| | s3 | +7.16 ± 0.90\* | +36.67 ± 2.79\* | +81.16 ± 4.41\* | **True** |
| **slide**（正例） | s1 | −0.65 ± 1.27 | +48.59 ± 1.77\* | +6.67 ± 1.70\* | **True** |
| | s2 | +7.01 ± 1.80\* | +73.50 ± 2.29\* | +15.55 ± 1.22\* | **True** |
| | s3 | +2.81 ± 0.81\* | +75.83 ± 1.86\* | +26.13 ± 1.45\* | **True** |

**crawl 9/9 全部显著负；hurdle 9/9 全部显著正；slide 6/9 显著正（stand 两次跨零）。**

## 3. 没有任何一个决策处于边界

每个 seed 的准入由**最强的源**决定。列出该源与其阈值的距离：

| target | seed | 最强源 | `U` | 阈值 `2·SE` | 裕度 |
|---|---|---|---:|---:|---:|
| crawl | s1 | stand | −44.06 | 15.69 | **−59.75** |
| crawl | s2 | run | −181.86 | 24.99 | **−206.85** |
| crawl | s3 | walk | −60.31 | 27.72 | **−88.03** |
| hurdle | s1 | run | +102.19 | 7.65 | +94.55 |
| hurdle | s2 | run | +110.51 | 7.40 | +103.11 |
| hurdle | s3 | run | +81.16 | 8.81 | +72.34 |
| slide | s1 | walk | +48.59 | 3.54 | +45.06 |
| slide | s2 | walk | +73.50 | 4.58 | +68.92 |
| slide | s3 | walk | +75.83 | 3.72 | +72.11 |

最小裕度是 crawl s1 的 59.75（负方向）与 slide s1 的 45.06（正方向）——
均为阈值本身的 3.8×–12.7×。

**这一点回应了预注册 §8 第 3 条声明的多重比较问题**：判据在每个 target 上做
9 次检验（3 seeds × 3 源）而不控制族错误率，但由于所有效应都远离阈值，
结论不依赖于边界情形，因此多重比较不构成实际威胁。

## 4. 预注册标记的唯一风险没有发生

预注册 §8.4 写明：

> `ADMISSION_FALSE_REJECT` 的最大风险是 **slide 在 K=10000 时 walk 的 `U` 尚未
> 超过 `2·SE`**（其真值 +56.95 测于 K=30000）。

实测 walk 在 K=10000 已达 **+48.59 / +73.50 / +75.83**，与 K=30000 的真值同量级，
全部远超阈值。**K=10000 对 slide 的准入判断已经足够。**

## 5. 附带发现（**均非主判据**，仅作记录）

### 5.1 crawl 上 argmax 跨 seed 完全不一致 —— 准入不可被选源替代

```
s1  argmax = stand (−44.06)
s2  argmax = run   (−181.86)
s3  argmax = walk  (−60.31)
```

三个 seed 选出**三个不同**的"最不坏"源。这直接说明：**当所有源都有害时，
argmax 是噪声**。只有选源（argmax）而没有准入（符号判断）的系统，
在 crawl 上会随机挑一个有害源用下去。

这与 `M32` 并不矛盾：M32 说的是**存在有用源时**源间差稳健（故 argmax 稳定）；
本例说的是**不存在有用源时** argmax 无意义。两者共同表明
**准入与选源是两个独立的决策，不能互相替代**。

### 5.2 slide 上 racing 的选源也选对了（3/3）

`argmax = walk` 在 3/3 seed 上成立，与 `slide_generalizability_v1` 的
`GEN_OK`（6 learner）一致。这是 racing 的**选源**能力在**第二个 target** 上
首次得到验证（此前 `RACING_VIABLE` 只在 hurdle 上测过）。

**但本实验未预注册该判据**，故只能作为附带观察记录，
不得作为"racing 选源跨任务成立"的裁决。

### 5.3 crawl 的标签从 1 seed 补到 3 seeds

此前 crawl 的"三源全负"只有**单 seed**支撑（`−448 / −217 / −208`，K=30000），
按 `M31` 不足以判断符号稳定性。本实验在 K=10000 上给出 **9/9 全负**，
且三个 seed 均显著。**crawl 的符号跨 learner 稳定**，与 door 形成对照
（door 是 18/18 负后新批出现 2/9 正）。

## 6. 工程验收（层1，全部通过）

```
剂量        24 条训练的 behavior share 全部落在 [0.45, 0.55]
臂间 share 差  crawl 0.0000   hurdle 0.0019   slide 0.0036      （M26 上限 5pp）
臂身份       每条 checkpoint 的 source_names 非 null 部分 == [arm]（M28）
面板         全部 36 个评估点与 panel128 常量逐位一致（128 ep，seed 11000…241007）
identity     36/36 通过 --expect-global-step / --expect-seed / --expect-admission-mode
hurdle 复用   从 correct_lr 的 per-episode 数据重算，与已发表 U 最大偏差 0.0004
```

## 7. 能与不能声称

**能**：

- 在这三个 target 上，一次 `K=10000` 的 racing 测量可同时给出**准入**与**选源**两个决策；
- 准入判据 `∃i: U_i > 2·SE_i` 在 9 个 (target, seed) 组合上**全部正确**，且无边界情形；
- crawl 的"全部源有害"这一标签**跨 3 个 learner 稳定**。

**不得**：

1. **不得声称跨任务普适**——三个 target 的真值均已知，本实验是
   **在已知真值的场地上检验判据**，不是发现新事实（§8.5）。
   跨任务推广须在真值未知的新 target 上前瞻验证；
2. **不得省略 `M24`**——单批 3 seeds，为正须注明"待独立重复"；
3. 不得把 `2·SE` 解释为统计显著性检验（不控制多重比较，见 §3）；
4. 不得据 §5.2 声称"racing 选源跨任务成立"——那未经预注册；
5. **不得声称准入在坏 target 上带来性能提升**——它的代价是 `N × K = 30k` 步，
   在 crawl 上正确拒绝的结果是"退化为 scratch 并损失 30k 步"。
   准入的价值是**避免灾难性负迁移**，不是提升上限。

## 8. 数据

```
预注册 / 脚本链   55d1423（先于 crawl/slide 任何评估数据冻结）
训练              24 条 × 10k（crawl 12 + slide 12）；hurdle 复用 racing_min_horizon_v1
评估              docs/data/racing_admission_v1/{crawl,slide}/source_free_eval/  24 点
hurdle 复用       docs/data/racing_min_horizon_v1/correct_lr/source_free_eval/   12 点
裁决输出          docs/data/racing_admission_v1/results.json
真值来源          crawl  docs/experiments/crawl_equal_dose_source_calibration_v1_results_20260723.md
                 hurdle docs/experiments/hurdle_equal_dose_source_calibration_multiseed_v1_results_20260723.md
                 slide  docs/experiments/slide_generalizability_v1_results_20260731.md
```

---

## 补注（2026-08-06，场地普查后追加）

crawl 作为"准入正确拒绝"的例证有一条此前未记录的背景：

```
crawl 的 J_best_known = 984.9，而理论回报上限 = 1000（reward 为 [0,1] 项相乘 × 1000 步）
→ 98.5% of theory max，scratch 已能高质量解决该任务
```

**这不改变本文件的裁决**（`ADMISSION_VIABLE` 与 9/9 决策仍然正确），
但引用时必须并列，且**不得**把归因改写成"任务已解决，故任何源必然有害"——
那个表述过强。"最终无 headroom"不蕴含"迁移必然无价值"，源仍可能更早达阈或降低方差。

crawl 上这些可能性是被**实测数据**排除的，而非被 headroom 论证排除：
早期（K=10000）9/9 显著负，终点（100k）盲目用源 809.2 vs scratch 960.2（−151.0）。
**早期与终点同时为负**，故 source mismatch 的解释依然成立。

出处：`docs/experiments/post_transfer_autonomy_site_screen_results_20260806.md` §3。
