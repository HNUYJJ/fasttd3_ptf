# 预注册：hurdle 上 reward-bearing bootstrap 的**样本效率加速倍数**

> 2026-07-30。**本文必须在任何长程臂被评估之前提交。**
> 定位：这**不是**新机制，而是把一个已验证的 A 级正迁移结果
> （`EQD30K.hurdle.run`，U=+379.66，CI90 [+271.5,+487.9]，3 seeds）
> 从 30k 窗口放大到长程，**量化它到底加速了多少倍**。

## 1. 为什么做这个

本项目已有十个"迁移性预测"信号族全部失败，最近一次是规格匹配假设（同日 REFUTED）。
但**在源已知有用时，迁移本身的收益从未被量化过**：

| 事实 | 出处 |
|---|---|
| hurdle + run 源：U(t=0,K=30k) = **+379.66**，CI90 [+271.5,+487.9]，3 seeds，剂量实测 0.500–0.502 | `transfer_effect_label_inventory` A 级 cell |
| hurdle scratch：`r@20k = 16.5`，`r@end = 597.5` | `label_identifiability_audit_20260727.md:84` |

也就是说 run 源在 30k 时就把回报推到了接近 scratch 训练末期的量级，
**但没有人测过 scratch 要跑多久才能追上**。那个比值才是"加速"。

## 2. 假设与主指标

**H**：给定一个已知有用的源，reward-bearing bootstrap 在 hurdle 上提供
**≥ 2×** 的样本效率提升。

**主指标**（对每个阈值 θ）：

```
speedup(θ) = steps_scratch(θ) / steps_source(θ)
steps_X(θ) = 臂 X 的 source-free student 回报首次 ≥ θ 的评估步数
             (线性插值于相邻评估点之间；若全程未达到,记为 >100k 并标注右删失)
```

**阈值在跑前冻结为绝对值**：`θ ∈ {200, 300, 400}`。
选它们的依据只有已公开的 `r@end = 597.5`（约 34% / 50% / 67%），
**不依赖本实验任何结果**。

## 3. 协议（冻结，单因素：只改是否有源）

```
target        h1hand-hurdle-v0
臂 A(对照)    scratch      SOURCE_BANK=configs/source_banks/empty.yaml, ADMISSION_MODE=legacy
臂 B(处理)    run-source   SOURCE_BANK=configs/source_banks/calibration/h1hand_hurdle_rbo_run.yaml
                           PTF_MCG=1, MCG_ABLATION=bootstrap_only, WARMUP_MODE=admission_bootstrap
                           ADMISSION_MODE=all, EXPECTED_SOURCE_MASS=0.5, WARMUP_MIN_STEPS=25
                           —— 与 EQD30K.hurdle.run 逐项相同,只把干预窗口延长到全程
起点          t=0(从头训练,不用 anchor)
长度          100k
seeds         1, 2, 3(两臂配对同 seed)
评估          source-free student, deterministic, 128 episodes
评估点        10k, 20k, 30k, 50k, 75k, 100k
其余          NUM_ENVS=128 BATCH=32768 BUFFER=51200 NUM_UPDATES=2 COMPILE=0 AMP=1
```

**剂量验收**：behavior source share 必须落在 [0.48, 0.52]，否则该 seed 作废重跑
（EQD30K 实测为 0.500–0.502）。

## 4. 判据（冻结）

| 条件 | 裁决 | 含义 |
|---|---|---|
| 至少 **2/3** 个阈值上 `speedup ≥ 2.0`，且每个达标阈值上 3/3 seed 的 per-seed speedup 均 ≥ 1.5 | `SPEEDUP_CONFIRMED` | 可作为论文的正面结果 |
| 有阈值达标但不满足上述 | `SPEEDUP_PARTIAL` | 报告曲线，不作强声明 |
| 所有阈值 `speedup < 1.5`，或 source 臂在 100k 被 scratch 反超 | `SPEEDUP_REFUTED` | 早期优势不转化为样本效率，如实记录 |

**次级观察（报告但不作判据）**：两臂在 100k 的终点回报；source 臂是否出现后期停滞
（行为预算被源持续占用的机会成本）。

## 5. 本实验**不能**声称什么

1. **不能**声称解决了"如何自动选源"——run 源是**人工指定**的，
   本项目十个信号族的失败恰恰说明自动选源仍是 open problem。
   论文里必须并列陈述，不得暗示该加速是自动获得的。
2. **不能**外推到其他 target——hurdle 是已知的正迁移场，
   door/crawl 上同样配置是明确负迁移。
3. **不能**与 HumanoidBench / FastTD3 论文的数字直接比较——
   本实验的 128 envs / batch 32768 / 无 compile 配置与它们不同；
   唯一合法的对照是**同 checkout 的 paired scratch 臂**。

## 6. 已知风险

- 100k 可能不足以让 scratch 达到 θ=400，届时 speedup 右删失，只能给下界；
- source 臂全程 0.5 剂量可能在后期成为负担（EQD30K 只验证到 30k）；
  若出现反超，那本身是重要发现，**不得**改剂量重跑。
