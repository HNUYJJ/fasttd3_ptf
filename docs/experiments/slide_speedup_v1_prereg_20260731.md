# 预注册：slide 上的样本效率加速倍数（第二个 target）

> 2026-07-31。**本文必须在任何长程臂被评估之前提交。**
> 定位：把 `hurdle_speedup_v1` 的口径原样搬到**第二个 target**，
> 直接回应"跨任务正面结果只有 hurdle 一个"这条缺口。

## 1. 立项门

1. **核心问题**：§2 问题 3——"方法能否在困难任务上加速学习"。
2. **唯一主要假设**：在 slide 上，已验证的最优源 `walk` 带来 **≥2×** 早期样本效率提升。
3. **正负后果**：正 → 迁移加速不是 hurdle 特有，跨任务有第二个案例；
   负 → `SPEEDUP_CONFIRMED` 须限制为 hurdle 单例，且需解释为何 slide 不成立。
4. **是否重复**：不重复。slide 此前只测到 20k（BAC gate），**从未测长程加速**。
5. **最小成本**：6 条 100k 训练 + 36 点评估。

## 2. 为什么现在做 slide

`slide_generalizability_v1` 刚裁决 **`GEN_OK`**（`efcc882`）：
`argmax = walk` 在 **6 个独立 learner** 上一致。故 slide 已通过 `M31` 要求的
**标签可推广性**前置审计——不会重蹈 door 在一个 ground truth 会翻转的场地上白跑四轮。

且 hurdle 与 slide 在**同一候选集合** `{stand, walk, run}` 上 argmax 反转
（hurdle=run，slide=walk），故本实验若成立，
得到的是"**在两个 argmax 不同的 target 上，各自选对源后都带来加速**"，
而非"walk 到处都好"。

## 3. 协议（冻结，与 `hurdle_speedup_v1` 逐项同构）

```
target        h1hand-slide-v0
臂 A(对照)    scratch    SOURCE_BANK=configs/source_banks/empty.yaml, ADMISSION_MODE=legacy
臂 B(处理)    walk 源    SOURCE_BANK=configs/source_banks/calibration/h1hand_slide_rbo_walk.yaml
                         PTF_MCG=1, MCG_ABLATION=bootstrap_only, WARMUP_MODE=admission_bootstrap
                         ADMISSION_MODE=all, EXPECTED_SOURCE_MASS=0.5, WARMUP_MIN_STEPS=25
起点          t=0（从头训练，不用 anchor）
长度          100k
seeds         1, 2, 3（两臂配对同 seed）
评估          source-free student, deterministic, 128 episodes
评估点        10k, 20k, 30k, 50k, 75k, 100k
其余          NUM_ENVS=128 BATCH=32768 BUFFER=51200 NUM_UPDATES=2 COMPILE=0 AMP=1
```

**剂量验收**：源臂 behavior share ∈ `[0.45, 0.55]`（slide 的 BAC 协议实测 0.4771–0.4845）。

## 4. 阈值（跑前冻结，来源可核实）

`docs/experiments/label_identifiability_audit_20260727.md:83` 记录
**slide 的 `r@end = 749.7`**（与 hurdle 的 `597.5` 同表同口径）。
按 `hurdle_speedup_v1` 的同一取法（`r@end` 的 34% / 50% / 67%）：

```
749.7 × {0.34, 0.50, 0.67} = {255, 375, 502}   →   冻结为 θ ∈ {250, 375, 500}
```

**该取法与本实验任何结果无关**，仅依赖已发表的 `r@end`。

## 5. 判据（冻结，与 `hurdle_speedup_v1` 同构）

```
speedup(θ) = steps_scratch(θ) / steps_walk(θ)
steps_X(θ) = 首次 ≥ θ 的步数（相邻评估点线性插值；全程未达到记 100k 并标右删失）
```

| 条件 | 裁决 |
|---|---|
| ≥2/3 阈值上 speedup 中位数 ≥ 2.0，且这些阈值上 3/3 seed 的 per-seed speedup ≥ 1.5 | `SPEEDUP_CONFIRMED` |
| 全部阈值 speedup 中位数 < 1.5，或 walk 臂在 100k 被 scratch 反超 | `SPEEDUP_REFUTED` |
| 其余 | `SPEEDUP_PARTIAL` |

## 6. `CLAUDE.md` §8 设计层自查

- **8.1 辨别力**：平凡解释="slide 本来就容易，什么都能快"——由**同 seed 配对的 scratch 臂**排除。
  另一平凡解释="任何源都能加速"——本实验**不主张**排除它；
  那是 `hurdle_selection_value_v1` 的职责，本文不得声称"选源有价值"。
- **8.2 混淆**：两臂除 bank 外逐项同参数；剂量逐 checkpoint 验收。
- **8.3 独立重复**：单批 3 seeds。按 `M24`，若为正须注明"待独立重复"，不得报为定论。
- **8.4 前提蕴含**：阈值来自外部已发表值；`SPEEDUP_REFUTED` 真实可达。
- **8.5 site selection**：slide 是已知 walk 有用后选的——本实验只回答
  "在一个已知有好源的 target 上，加速是否也成立"，不得推广到任意 target。
- **8.6 本轮教训对照**：M25（不涉跨 target 排序）、M26（剂量验收 + 两臂同参）、
  M27/M31（不做跨批数值比对，判据是本批内的达阈步数）、M28（bank 配置已核实
  `h1hand_slide_rbo_walk.yaml` 存在且 `null_option: false`）、M29/M30（判据先于数据冻结）、
  M32（本实验测的是绝对加速，不涉源间差）。
- **8.7 判据切换红线**：全新数据 + 沿用已冻结的 speedup 口径。

## 7. 能与不能声称

**能**（若 `SPEEDUP_CONFIRMED`）：迁移加速在**第二个 target** 上成立；
连同 hurdle，构成两个 argmax 不同的 target 上各自选对源后均加速的证据。

**不得**：不得称"大部分 HumanoidBench 任务"（两个 target）；
不得省略 `M24` 的单批限制；不得声称"选源有价值"（那需要 argmin 对照臂）；
不得与 hurdle 的倍率直接平均（不同 target 的 reward 尺度不同，M15）。

## 8. 不得做的事

- 裁决后不得调 θ、seed 数或剂量带。
- 若 `SPEEDUP_REFUTED`，如实报告并据此限制 hurdle 结论的推广范围。
