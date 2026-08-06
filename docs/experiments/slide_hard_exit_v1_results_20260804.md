# 结果：slide 上的 30k 硬退出 —— `HARD_EXIT_SUPPORTED`

## 全程恒定剂量造成的损害完全可逆，且退出后方差压小 8 倍

> 冻结协议 `docs/run_card_interventional_bootstrap_racing_v1.md`，
> 脚本链 `62a6199` / `cab7c50`（训练 + 评估 + 裁决三件均先于任何评估冻结）。
> 训练于 2026-08-01 完成；评估与裁决于 2026-08-04 补做
> （Codex 线程在 2026-08-01 13:55 因 `stream disconnected` 中断，训练在后台跑完但无人收尾）。
> 裁决输出 `docs/data/slide_hard_exit_v1/slide_hard_exit_v1_results.json`。

## 1. 裁决

```
VERDICT: HARD_EXIT_SUPPORTED

D_exit_endpoint（100k 终点，exit − cont）
    per_seed  +631.46 / +641.35 / +622.72     mean +631.84   sd 9.32   lcb90 +621.69
A_exit_nAUC_30k_100k（30k–100k 归一化 AUC，exit − cont）
    per_seed  +416.79 / +439.79 / +437.97     mean +431.52   sd 12.78  lcb90 +417.60
D_scratch_endpoint（100k 终点，exit − scratch）
    per_seed   +15.61 / +348.42 /  +46.03     mean +136.69   sd 184.00 lcb90 −63.62
```

3/3 seed 同向，`sd = 9.32` 相对效应量 631.84 仅 1.5%——本项目迄今最干净的效应之一。

## 2. 设计：同一 anchor 分叉的配对干预

```
prefix   0 → 30k    walk 源，剂量 0.5，在 30k 存 branch anchor（learner + replay + RNG）
  ├─ cont   从该 anchor 续训至 100k，ADMISSION_MODE=all，MASS=0.5   （继续注入源）
  └─ exit   从**同一** anchor 续训至 100k，ADMISSION_MODE=none，MASS=0.0（源硬退出）
两条 continuation 使用同一 resume-noise seed；评估用同一 128-episode 冻结面板
```

分叉点的等同性由数据直接验证——两臂 30k 的评估值**逐 seed 完全相同**：

```
s1  206.640    s2  259.548    s3  247.247        （cont 与 exit 同值）
```

## 3. 曲线：损害不是"变慢"，是"停住"

| 步数 | 30k | 50k | 75k | 100k |
|---|---:|---:|---:|---:|
| cont s1 | 206.6 | 284.1 | 285.6 | 297.1 |
| cont s2 | 259.5 | 271.1 | 289.4 | 308.5 |
| cont s3 | 247.2 | 265.4 | 267.7 | 286.2 |
| **exit s1** | 206.6 | **490.1** | **951.5** | **928.5** |
| **exit s2** | 259.5 | **590.0** | **913.1** | **949.9** |
| **exit s3** | 247.2 | **540.7** | **934.8** | **908.9** |

从**完全相同的**学习状态出发，唯一差别是 30k 之后是否继续注入源：
继续注入的一臂在 70k 步里只涨了约 40–90 分；退出的一臂涨了约 660–720 分。

## 4. 工程审计：干预严格生效

`treatment_audit`（三 seed 一致）：

```
anchor_execution              [1839880, 2000120]   → source share 0.4791   带 [0.45,0.55] ✓
continuous_delta_execution    [4286200, 4673800]   → source share 0.4784   （cont 确实继续用源）
hard_exit_delta_execution     [      0, 8960000]   → source 执行增量**严格为 0**
hard_exit_delta_critic        [      0, 4587520000] → source critic 采样增量**严格为 0**
hard_exit_active_buffer_counts[      0, 6553600]   → active replay 中 source 槽位**严格为 0**
```

即 exit 臂的行为通道与 replay 通道**同时**归零，无残留。

## 5. 与 scratch 的关系：追平，且方差小 8 倍

```
slide 100k source-free return
  scratch        792.4 ± 167.3 (sd)      [912.9, 601.4, 862.9]
  cont（全程用源） 297.3 ±  11.2          [297.1, 308.5, 286.2]
  exit（30k 退出） 929.1 ±  20.5          [928.5, 949.9, 908.9]
```

- **均值高出 scratch 136.7，但不显著**（`lcb90 = −63.6`），因 scratch 自身跨 seed
  波动极大（601–913）。**不得声称超越 scratch。**
- **exit 的跨 seed sd 是 scratch 的 1/8**（20.5 vs 167.3）。三条 exit 曲线
  终点落在 909–950 的窄带内，而 scratch 落在 601–913。这是一个稳健的观察，
  但本实验未预注册方差比较，故只作**描述性**记录，不进主张。

## 6. 定位：这是工程基线，不是贡献

必须与 `competence_gated_transfer_design_20260730.md` §2 的既有判定一致：

> **恒定剂量在后期有害，本身不是新东西。** PTF 原文（Yang 2020）的 λ 线性衰减
> 就是为此设计的……**修掉它属于修正实验配置，不构成贡献。**

同文 §0.1 预先规定了本实验的地位：

> 下一项最低成本基线应是预注册的固定 bootstrap schedule / hard exit……
> **只能先称工程基线，不得包装成 learned 调度贡献。**

本结果**证实**了该基线有效，但按上述条款，它只修正了本项目自设的配置缺陷
（`PTF_MCG_WARMUP_STEPS` = 总步数的全程恒定剂量），**不作为方法贡献**。

## 7. 能与不能声称

**能**：

- `slide_speedup_v1` 的绝对损害**完全来自 30k 之后继续注入源**，且完全可逆；
- 在 slide 上，30k 硬退出使终点从 293.3 恢复到 929.1（`+631.8 ± 9.3`，3/3）；
- 干预在行为与 replay 两个通道上均严格生效（增量恒为 0）。

**不得**：

1. **不得声称超越 scratch**——`D_scratch_endpoint` 的 `lcb90 = −63.6`，跨零；
2. **不得作为方法贡献**（§6）；
3. **不得分离通道归因**——exit 臂的 behavior 与 replay 同时退出，本设计
   无法区分二者各自的贡献；而 `phase1_bounded_bank_lease` 已证明单独退出
   replay 通道在 truck/basketball 上是 `HETEROGENEOUS`；
4. 单 target、单批 3 seeds，按 M24 需独立重复；
5. **不得据此声称"30k 是普适的退出时机"**——退出点未做敏感性分析，
   30k 的选择来自既有 anchor 协议而非优化。

## 8. 数据

```
协议 / 脚本链   62a6199 / cab7c50（均先于评估冻结）
anchor          artifacts/slide_hard_exit_v1/anchors/slide_s{1,2,3}_walk_k30000/
                （各含 learner.pt / replay.pt / rng.pt / manifest.json / checksums.json）
评估            docs/data/slide_hard_exit_v1/source_free_eval/   21 点 × 128 episodes
裁决输出        docs/data/slide_hard_exit_v1/slide_hard_exit_v1_results.json
运行时快照      docs/data/slide_hard_exit_v1/runtime_manifest_20260801T131916Z.json
scratch 对照    docs/data/slide_speedup_v1/source_free_eval/（同 target 同面板同协议）
```
