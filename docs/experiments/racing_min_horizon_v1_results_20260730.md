# 结果：自动选源的最小测量代价 —— RACING_VIABLE，K* = 10000

> 2026-07-30。预注册 `docs/experiments/racing_min_horizon_v1_prereg_20260730.md`（`6776c03`），
> 裁决脚本 `scripts/analysis/analyze_racing_min_horizon_v1.py`（`88a5e26`），
> 二者均在任何臂被评估之前提交。
> **共跑了两批完全独立的 12 条训练（合计 24 条训练 + 72 点评估）**，原因见 §2。

## 1. 最终裁决

**`RACING_VIABLE`，K\* = 10000。** 选源代价 = 3 源 × 10k = **30k 步**（并行则墙钟 10k 步）。

```
主判据(3/3 seed 满足 argmax_i U_i(K) = run):
   K= 2000    批1 0/3    批2 0/3    合并 0/6
   K= 5000    批1 3/3    批2 1/3    合并 4/6      ← 不稳健
   K=10000    批1 3/3    批2 3/3    合并 6/6      ← 两批独立测量皆通过
```

**预注册未覆盖"两批数据如何合并"**（原计划只跑一批）。此处采取**保守解释**：
取两批都满足 3/3 的最小 K，即 `K* = 10000`。不采用批 1 单独的 `K*=5000`。

## 2. 为什么跑了第二批：一个被独立重复推翻的结论

批 1 单独看是 **`RACING_CHEAP`，K\*=5000**。若就此收工，结论会**过强一倍**（成本 15k vs 实际 30k）。

第二批本是为排查另一件事而跑的（见 §6），跑完才发现它推翻了 K\*=5000：

```
K=5000 的 per-seed U(run / walk / stand)
  批1  s1: +56.41 / +16.65 / +7.68   HIT      批2  s1: +59.71 / +30.33 / +2.03   HIT
       s2: +28.94 /  +9.18 / −3.03   HIT           s2:  +1.29 /  +2.54 / +2.50   MISS
       s3: +33.33 /  +5.85 / +1.54   HIT           s3: +15.43 / +36.94 / +1.85   MISS
```

**seed 3 在两批中给出完全相反的排序**（批1 run 大胜 27.5；批2 walk 大胜 21.5），
不是边界抖动。

**定量解释**（合并 6 个独立 learner 运行）：

| K | run | walk | run−walk | 合并 SE | t |
|---|---|---|---|---|---|
| 2000 | −0.41 ± 1.19 | +3.42 ± 1.50 | −3.83 | 1.92 | −2.0（方向错误） |
| 5000 | +32.52 ± 8.48 | +16.91 ± 5.19 | +15.61 | 9.94 | **1.57（不显著）** |
| 10000 | +105.56 ± 6.40 | +48.42 ± 3.38 | +57.14 | 7.24 | **7.89** |

K=5000 时 run 与 walk 的真实间隔相对 **learner 间方差**不够大；单批 3/3 是运气。
**episode 面板 SE 在这里完全不是正确的不确定性尺度**——批 1 内部
run 领先次优达 8.4–14.8 个 episode-SE，看起来极显著，却经不起换一批 learner。

## 3. 辨别判据：racing 测的不是行为质量（通过）

预注册 §3 冻结的辨别设计——zero-shot 行为排序与真实 U 排序在 walk/stand 上恰好相反：

```
zero-shot 行为(32 ep)    : run 169.21 > stand 146.94 > walk  96.35    walk 垫底
真实 U (EQD30K, K=30k)   : run 379.66 > walk  104.89 > stand  51.28    walk 第二
```

| K | walk > stand 的运行数 |
|---|---|
| 2000 | 3/6 |
| 5000 | **6/6** |
| 10000 | **6/6** |

**在 K ≥ 5000 的全部 12 个独立 learner 运行中，racing 都把 walk 排在 stand 之上**，
与 zero-shot 行为排序相反、与真实 U 排序一致。
这坐实了 racing 测的是延迟学习效用，不是源的行为质量——
它做到了族 1（zero-shot 行为 return / 位移）做不到的事。

## 4. 成本—收益（公式冻结于预注册 §5）

```
racing 成本 = 3 源 × 10000                      = 30000 步（并行墙钟 10000 步）
选源收益   = steps_scratch(θ=300) − steps_source = 67020 步（hurdle_speedup_v1 per-seed 中位数）
净收益                                          = +37020 步
```

收益约为成本的 **2.2 倍**。若按批 1 的 K\*=5000 计则是 4.5 倍——**这正是必须用重复后的
K\*=10000 的原因**。

## 5. K=2000 的失败是系统性的，不是信号太弱

```
K=2000 合并 6 运行:  run = −0.41 ± 1.19    walk = +3.42 ± 1.50    stand = +2.26 ± 1.78
```

run 不是"分不开"，而是**排在最后**（批 1 内 run 落后次优 5.3–14.3 个 episode-SE，3/3 一致）。

机制（批 1 数据，`progress_max_dx` 为前进距离）：

```
K=2000   student dx=0.447   run dx=0.962   walk dx=0.830   stand dx=0.430
K=5000   student dx=0.776   run dx=6.085   walk dx=2.233   stand dx=1.163
K=10000  student dx=0.743   run dx=15.238  walk dx=6.119   stand dx=2.143
```

极早期 run 已经跑得最远，但 return 最低——前进带来的增益被摔倒/姿态惩罚抵消
（run 源自身 zero-shot 摔倒率 62%）。到 K=5000 前进差距放大到 6.09 vs 2.23，return 才分离。

**因此存在一个"信号出现阈值"，在此之前 racing 会给出自信而错误的答案。**
这是本方法必须声明的失效模式，不能靠缩小 K 来省成本。

### 5.1 一个被重复否定的观察项（记录，不作结论）

批 1 中 `progress_max_dx` 口径在 K=2000 就 3/3 命中且完整排序正确，
一度看起来能把 K\* 降到 2000。**批 2 复现为 0/3，合并 3/6。**

即便它复现了也不该采纳，理由在跑批 2 之前就已写下：
(a) ground truth 是 `U(return, K=30k)`，用 `dx` 去预测它属**跨量类外推**，
而"zero-shot 行为 return / **位移**"正是已被否定的族 1；
(b) `dx` 是**任务特定**读出量（hurdle 是前进任务，door/cabinet 上无意义）；
(c) 事后在多个读出量中挑最好的，是多重比较。

## 6. 一处未解释的差异（如实记录）

同为 `t=0, K=10k, dose 0.5, run 源`（同一 protocol family），三批的 U 绝对水平不一致：

```
hurdle_speedup_v1  U@10k = +57.99   per-seed [53.8, 80.9, 39.3]
racing 批1         U@10k = +113.16  per-seed [114.6, 130.6, 94.2]
racing 批2         U@10k = +97.95   per-seed [102.2, 110.5, 81.2]
```

已排除的解释（逐项对比 checkpoint 内的 `args` / `ptf_cfg`）：

- **不是学习率日程**。`actor/critic_learning_rate == *_learning_rate_end == 0.0003`，
  `eta_min == base_lr`，`CosineAnnealingLR` 恒为常数，`total_timesteps` 不进入学习率。
- **不是 `mcg_warmup_steps`**。批 2 已将其对齐为 100000（与 speedup 相同），
  U@10k 仍为 98.0，接近批 1 的 113.2 而非 speedup 的 58.0。
- **不是 global_step 语义**。三批均满足 `exec_total = global_step × num_envs`，
  且 `critic_update_count=19978` / `actor_update_count=9989` 逐位相同。
- 剩余差异仅 `exp_name` / `project` / `eval_checkpoint_steps` / `run_stop_step`。

批 1↔批 2 的运行间差约 15，而 speedup↔批 2 差约 40，偏大。**此差异未获完全解释。**
它**不影响主判据**（主判据是排序，不是绝对水平），
但提示跨批次的 U 绝对值不可直接比较——与 `hurdle_speedup_v1` 结果文档 §5 的结论一致。

## 7. 剂量与盲化验收（两批均通过）

```
批1 behavior share  0.4990–0.5047     批2  0.4990–0.5047     带 [0.48, 0.52]
源臂 source_names   ['<源>', 'null'] + 有 admission_audit
student 臂          ['null']          + 无 admission_audit
```

全部 54 个源臂 checkpoint（3 源 × 3 seed × 3 K × 2 批）均在带内。
72 个评估点全部通过完整性校验（128 episodes / `global_step` / seed / 臂 / ckpt 路径 /
`identity_checked` / sha256 两两不同）。

## 8. 能声称什么、不能声称什么

**能声称：**

- 在 hurdle 上，用 **30k 步真实交互**（3 源各 10k，可并行至墙钟 10k）
  即可从 {run, walk, stand} 中**稳健地选出最佳源**（6/6 独立 learner 运行），
  且该代价显著小于选对源带来的 67k 步节省。
- racing 测到的是延迟学习效用而非行为质量：它在 12/12 个运行中排出了
  zero-shot 行为排序的**反向对**（walk > stand）。
- 与十一族的区别是**estimand 未变**：直接测 `U` 本身，只缩短 `K`，
  不做跨量类外推。

**不能声称：**

1. **不能声称 K\*=10000 是理论下界**。只测了三个 K；真实的最小可用 horizon
   可能落在 (5000, 10000) 之间。K=5000 已证不可靠。
2. **不能声称跨任务成立**。单 target（hurdle）、单源集合（3 个 loco 源）。
   door 上三个 loco 源一致有害（9/9 per-seed 负），方向依赖是本项目反复确认的事实。
3. **不能声称解决了通用自动选源**。racing 需要真实交互，
   它是"最小测量代价"的上界，不是零成本预测器——
   十一族证明的零成本预测不可行**依然成立**。
4. **不能声称 ground truth 是外部真值**。`K=30k` 的 U 是本项目自测的，
   且 `EQD30K.hurdle.stand` 仅 `single_seed`。
5. **不能声称 U 的绝对水平可跨批比较**（§6）。

## 9. 方法学教训

**单批 3/3 + 大 episode-SE 不足以支撑结论。** 批 1 的 K=5000 显示 run 领先
8.4–14.8 个 episode-SE，看起来无可争议，却被独立重复推翻。
正确的不确定性尺度是 **learner 间方差**，而 3 个 seed 估计它本就很粗。
`progress_dx` 观察项（批 1 3/3 → 批 2 0/3）是同一教训的第二个实例。

## 10. 数据与复现

```
预注册        docs/experiments/racing_min_horizon_v1_prereg_20260730.md   (6776c03)
脚本冻结      scripts/analysis/analyze_racing_min_horizon_v1.py           (88a5e26)
训练          scripts/run_racing_min_horizon_v1.sh + drive_racing_min_horizon_v1.sh
评估          scripts/eval_racing_min_horizon_v1.sh   (128 ep, deterministic, source-free)
批1 数据      docs/data/racing_min_horizon_v1/compressed_lr/    (results.json + 36 点 + dose_audit)
批2 数据      docs/data/racing_min_horizon_v1/correct_lr/       (results.json + 36 点 + dose_audit)
源天花板探针   docs/data/hurdle_speedup_v1/source_ceiling_probe.json        (469c1fb)
加速对照      docs/experiments/hurdle_speedup_v1_results_20260730.md       (20f1e11)
```

目录名 `compressed_lr` / `correct_lr` 是批 1 期间一个**错误归因**的遗留
（我曾误判 `total_timesteps` 压缩了 LR 日程，已在 `11ff656` 更正）。
两批实际配置对训练动态等价，故构成**独立重复**而非"修复前后"。
名称保留以匹配已提交的数据路径。
