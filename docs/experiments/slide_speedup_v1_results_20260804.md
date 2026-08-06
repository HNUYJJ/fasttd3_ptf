# 结果：slide 上的样本效率加速 —— `SPEEDUP_REFUTED`

## 跨任务加速不成立，且不是收益衰减而是绝对损害

> 预注册 `docs/experiments/slide_speedup_v1_prereg_20260731.md`（`a5fcd9b`），
> 裁决链 `e7f11c4`，均先于任何长程臂评估冻结。
> 裁决输出 `docs/data/slide_speedup_v1/slide_speedup_v1_results.json`。
> 数据产出于 2026-08-01，结论文档补记于 2026-08-04（Codex 线程断线导致的收尾缺口）。

## 1. 裁决

```
VERDICT: SPEEDUP_REFUTED        —— 预注册的两个否定条件**同时**命中

条件 a  全部阈值 speedup 中位数 < 1.5
        θ=250  中位 0.851   per_seed [0.851, 1.168, 0.805]
        θ=375  中位 0.627   per_seed [0.627, 0.800, 0.505]
        θ=500  中位 0.758   per_seed [0.758, 0.911, 0.562]

条件 b  walk 臂在 100k 被 scratch 反超
        scratch 792.406   vs   walk 293.345      scratch_overtook_walk = true
```

阈值来源与本实验任何结果无关：`label_identifiability_audit_20260727.md:83` 记录
slide 的 `r@end = 749.7`，按 hurdle 同一取法 `× {0.34, 0.50, 0.67}` = `{255, 375, 502}`
→ 冻结为 `θ ∈ {250, 375, 500}`。

## 2. 曲线：walk 臂在 30k 之后停止学习

source-free、deterministic、128-episode 冻结面板：

| 步数 | 10k | 20k | 30k | 50k | 75k | 100k |
|---|---:|---:|---:|---:|---:|---:|
| walk s1 | 112.0 | 152.2 | 234.4 | 246.8 | 257.5 | 279.8 |
| walk s2 | 119.3 | 201.2 | 223.5 | 272.1 | 283.3 | 311.4 |
| walk s3 | 122.5 | 221.7 | 248.1 | 248.4 | 280.7 | 288.8 |
| scratch s1 | 26.6 | 49.3 | 86.3 | 259.1 | 486.4 | 912.9 |
| scratch s2 | 37.3 | 41.1 | 55.7 | 274.6 | 318.0 | 601.4 |
| scratch s3 | 31.7 | 45.3 | 103.4 | 364.3 | 915.6 | 862.9 |

**早期领先是真实的**（10k 时 walk 112–123 vs scratch 27–37，约 3–4×），
**但 30k 之后 walk 臂的斜率几乎归零**：50k→100k 仅从约 247–272 爬到约 280–311，
而同期 scratch 从约 259–364 升到 601–913。

这与 hurdle 的形态**不同**：hurdle 上倍率从 25.2×（20k）衰减到 1.24×（100k），
始终为正；slide 上是**绝对损害**——终点仅为 scratch 的 37%。

## 3. 工程身份（干预有效，非配置失败）

```
walk 臂 behavior share    s1 0.4788   s2 0.4766   s3 0.4774      带 [0.45, 0.55]
walk 臂 critic replay share  三 seed 均 0.5000
```

剂量落在预注册带内，两臂除 source bank 外逐项同参数，故差异不由剂量解释。

## 4. 对 hurdle 结论的影响（预注册 §7 要求执行）

预注册第 8 条写明：「若 `SPEEDUP_REFUTED`，如实报告并据此限制 hurdle 结论的
推广范围」。据此：

> `hurdle_speedup_v1` 的 `SPEEDUP_CONFIRMED` **不得**再表述为"迁移加速"这一
> 一般现象，只能表述为"**在 hurdle 上**成立"。在第二个 target 上，
> 同一机制、同一剂量、同一候选集合下的**已验证最优源**造成了绝对损害。

这直接否定了 `slide_speedup_v1_prereg` §7 期望的正面主张
（"迁移加速不是 hurdle 特有"）。

## 5. 损害的归因已由后续实验确定

`slide_hard_exit_v1`（`docs/experiments/slide_hard_exit_v1_results_20260804.md`）
从同一协议的 30k branch anchor 分叉，证明本实验的损害**完全来自 30k 之后
继续注入源**：源在 30k 硬退出后，终点从 293.3 恢复到 929.1（`+631.8 ± 9.3`，3/3）。

因此本实验的正确读法不是"walk 源对 slide 有害"，而是
**"全程恒定 50% 剂量对 slide 有害"**——而全程恒定是本项目为等剂量对照的
干净性人为设定的配置（`PTF_MCG_WARMUP_STEPS` = 总步数），
不是 PTF 原文的用法（原文为 λ 线性衰减）。

## 6. 能与不能声称

**能**：

- slide 上"选对源 + 全程恒定 50% 剂量"在 100k 被 scratch 反超 2.7 倍；
- 因此 hurdle 的加速结论必须限制在 hurdle 上，不构成跨任务现象；
- 早期加速（10k 约 3–4×）本身是真实的，与后期损害并存。

**不得**：

1. 不得据此声称"walk 源对 slide 有害"——`slide_bac_gate` / `slide_generalizability`
   已在 6 个 learner 上确立 walk 是最优源（`GEN_OK`），本实验的损害来自剂量调度；
2. 不得与 hurdle 的倍率平均或合并（不同 target 的 reward 尺度不可比，M15）；
3. 单批 3 seeds，按 M24 不得作为定论，需独立重复；
4. 不得因该负结果回头调整 θ 或剂量带（预注册 §8）。

## 7. 数据

```
预注册 / 裁决链   a5fcd9b / e7f11c4（均先于评估冻结）
评估              docs/data/slide_speedup_v1/source_free_eval/   36 点 × 128 episodes
裁决输出          docs/data/slide_speedup_v1/slide_speedup_v1_results.json
裁决日志          docs/data/slide_speedup_v1/adjudication.log
运行时快照        docs/data/slide_speedup_v1/runtime_manifest_20260801T0440Z.json
```
