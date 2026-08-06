# 证据状态总表

> 建于 2026-07-31，**最后更新 2026-08-06**（文件名保持不变以维持既有引用）。
> 单页速览：手上**确实成立**的是什么、**边界**在哪、**还缺**什么。
> 每行都指向可复现的裁决输出。写这份表是因为结果已分散在十余份文档里，
> 容易在汇报时把"已裁决"与"待验证"混为一谈。

## 1. 已裁决的正面结果

| 结论 | 裁决 | 关键数字 | 必须同时引用的边界 |
|---|---|---|---|
| hurdle 上迁移带来早期加速 | `SPEEDUP_CONFIRMED` | θ=200 中位 **4.38×**，θ=300 中位 **3.59×**，各 3/3 seed | 100k 衰减到 **1.24×**；训练不稳定为 source 臂独有（−66%/−84% 回撤，scratch 零回撤）；**run 源系人工指定**；**在 slide 上不成立**（见 §2）|
| 用 30k 步交互可自动选对源 | `RACING_VIABLE` | K\*=**10000**，两批独立测量各 3/3；成本 30k vs 节省 67k | 单 target（hurdle）、单源集合；K=5000 **不稳健**（批1 3/3 → 批2 1/3） |
| racing 测的不是行为质量 | 辨别判据通过 | K≥5000 的 **12/12** 运行排出 `walk > stand`，与 zero-shot 行为排序相反 | 仅 hurdle |
| **选对源确实比选错源快** | `SELECTION_VALUABLE` | θ=200 中位 **4.95×**（3/3）；θ=300 **3/3 右删失**（stand 到 100k 未达阈） | 单 target（hurdle）、单批 3 seeds；剂量差 ≤0.0020 已验收 |
| slide 的选源决策可推广 | `GEN_OK` | `argmax = walk` 在 **6 个独立 learner** 上一致；偶然通过率 3.7% | 仅验证**标签**可推广，未验证 racing 短 K |
| hurdle 与 slide 构成**真正的 crossover** | — | 同一候选集合 `{stand,walk,run}`，argmax 反转（run ↔ walk） | 排除了"全局固定源排序"，但只有 2 个 target |
| slide 上 30k 硬退出源可完全逆转损害 | `HARD_EXIT_SUPPORTED` | 终点 `+631.8 ± 9.3`（3/3）；293.3 → 929.1；跨 seed sd 为 scratch 的 1/8 | **工程基线非贡献**（PTF 原文 λ 衰减即为此设计）；**不得称超越 scratch**（`lcb90 = −63.6`）；behavior/replay 通道未分离 |
| **准入能力**：一次 K=10000 测量可同时判是否该整体拒绝 | `ADMISSION_VIABLE` | `false_admit=0`、`false_reject=0`；crawl 9/9 显著负；最小裕度为阈值的 3.8×–12.7× | 三 target 真值均已知 → 检验判据而非前瞻发现；单批 3 seeds |
| **端到端系统**：9/9 自动决策与真值一致，组合后优于两基线 | `ENDTOEND_SUPPORTED` | 口径1（learner 步数对齐）：C>A 与 C>B 各 3/3；hurdle C 840.4±11.4 vs A 387.4±63.1 vs B 479.1±331.8 | **口径2（全额计入 racing 40k）推翻两个 target**：slide 0/3、crawl 1/3，仅 hurdle 仍赢（+313.3）。故只能称最终更高更稳，**不得称提升样本效率** |

## 2. 已裁决的负面结果（系统性）

| 结论 | 证据 |
|---|---|
| **零成本迁移性预测不可行** | **十二**个信号族全部失败（行为 / 即时 reward / 梯度 / critic / reward 结构 / 任务定义 / 静态规格 / 任务进度 **七个空间**）；`docs/impossibility_characterization_of_transfer_prediction_20260730.md` |
| 连"单向排除"这个最弱用法也不成立 | 族 12（`progress_screen_v1`，`HOLDOUT_FAILED`）：crawl 上有害的 run 位移 **14.302**，slide 上有用的 walk 位移 **1.814**，**反向 7.9×**；阈值需满足 `14.302 < θ < 1.814` = **空集** |
| **跨任务加速不成立** | `slide_speedup_v1` = **`SPEEDUP_REFUTED`**，两个否定条件同时命中：三阈值中位数 0.851/0.627/0.758 全 <1.5，且 100k 被 scratch 反超（792.4 vs 293.3）|
| 恒定剂量在 student 超越源后有害 | hurdle：源 zero-shot 169.21，student 20k 即达 241.72、75k 达 3.82×；slide：全程恒定剂量使终点仅为 scratch 的 37%，而 30k 退出即恢复 |
| learned 自适应调度劣于固定 schedule | `classic_ptf_hurdle`：fixed−scratch `+100.993 (3/3)`，learned−fixed `−57.853 (1/3)`，seed 方差 ×4 |

## 3. 未裁决 / 已作废（如实列出）

| 项 | 状态 | 原因 |
|---|---|---|
| racing 能否**提前拒绝**（door） | **主终点未裁决，判决场已关闭** | v2 `REPLICATION_DIVERGED`（容差量纲错配）→ v3 撤回（判据切换）→ v4 `PARTICIPANT_DIVERGED`：**door 的 ground truth 本身不跨 learner 稳定**，原理上不适合作判决场。M31(d) 要求另找 `U` 符号稳定的全负 target |
| Competence-Gated Transfer | **实现前关闭** | 第四次行为代理换皮；三条证据全在仓库内（自写 guardrail、adaptive 先例、fixed>learned） |
| RACING_REJECT v1 | **作废，未揭盲** | 主假设逻辑上不可证伪（sanity 蕴含主判据） |
| RACING_MULTI | **作废，未执行** | 候选集合不同 → 全局固定排序即可解释；剂量混淆；无独立重复 |
| RACE-then-RUN | **作废，未执行** | `argmax_i U_i ≡ argmax_i J_i` 代数退化；best-of-N order statistic；成本核算错误；单批 n=3；site selection |
| 零训练准入（族 12 之后） | **方向关闭** | 按 `progress_screen_v1` 预注册 §8，不得换指标在同一批 target 上抢救 |

## 3.1 一个被推翻的既有结论（2026-07-31）

`door_at10k_gate_v1` 的"三个 loco 源一致有害（9/9 per-seed 全负）"
**须限制为"在 seeds 1–6 上"**：新批 `s7–9` 出现 2/9 为正，
其中 `s9` 的 `run = +36.32 ± 3.95` **显著正**（原结论 −30.63，跨度 67）。
gate 在其自身 learner 上的测量可靠（`s4–6` 复现符号，18/18 负），
但那是 **learner 子总体**的性质，不是 target 的性质。见 `M31`。

## 4. 两条影响全部标签的方法学发现（M27 / M31）

> **此前全部 per-seed `U` 标签都只有单次运行，从未刻画 run-to-run 不确定性。**
> 首次同协议重跑（固定 source/target/stage/dose/anchor/noise seed）：
> door `|ΔU|` 中位 **24.23**、最大 **43.78**；hurdle 同 seed 两批约 **15**。
> 而这些标签的效应量本身只有 `−7 ~ −43`。

**M31（更强）**：连**符号**都可能跨 learner 显著反转（door 上 18/18 负 → 新批 2/9 正）。
故 `M18` 的"标签可测性"审计之外，还缺一层**标签可推广性**审计——
现有全部 A 级标签都只在 3 个 learner 上测过，从未检验跨 learner 的符号稳定性。
hurdle 是对照（`run` 的 U=+379.66 远离零点，两批 6 learner 全选中 run），
故符号稳定性**因 target 而异，不能默认**。

**M27 的后果**：符号/排序可用；**per-seed 数值不可当作可复现真值**
（`EQD30K` / `sibling gate` / `door gate` 的点值均只有单次运行支撑）。
凡"与已发表值比对"的复制检查，容差必须基于 run-to-run 漂移。

## 4.1 target 覆盖的实际情况（2026-08-06 更新）

| target | 源 | 已有结论 | seeds | 可用性 |
|---|---|---|---|---|
| **hurdle** | run / walk / stand | 加速 `SPEEDUP_CONFIRMED`；选源价值 `SELECTION_VALUABLE`；racing `RACING_VIABLE` | run·walk 各 3+，stand 3 | **完整链条，唯一的正面 target** |
| **slide** | walk / run / stand | 标签 `GEN_OK`(6 learner)；加速 **`SPEEDUP_REFUTED`**；退出 `HARD_EXIT_SUPPORTED` | 6（标签）/ 3（加速、退出） | 标签可用；**加速为负** |
| **crawl** | stand / walk / run | **全负**：K=30k 时 −448/−217/−208（1 seed）；K=10k 时 **9/9 显著负**（3 seeds） | 3 | **符号跨 3 learner 稳定**，与 door 形成对照；已作准入判决场的负例 |
| door | stand / walk / run | 全负但**不可推广**（M31）| 9 | **判决场已关闭** |

**跨任务的样本效率优势实际只有 hurdle 一个**。slide 通过了标签审计但加速为负；
端到端在 slide 上的价值在**终点与方差**（929.1±20.5 vs scratch 792.4±167.3），
不在速度——全额计入 racing 成本后 slide 反而输 0/3（`endtoend_v1` 口径 2）。

## 5. 距离课题目标还缺什么（诚实清单）

| 缺口 | 现状 |
|---|---|
| **跨任务的正面加速** | **仍缺**。hurdle 成立，slide `REFUTED`；端到端在 slide 上赢在终点与方差而非速度。样本效率优势仍只有 1 个 target |
| **准入能力**（要不要用源）| **已补齐** `ADMISSION_VIABLE`；crawl 亦由该实验从 1 seed 补到 3，9/9 全负且显著 |
| **端到端方法** | **已补齐** `ENDTOEND_SUPPORTED`，决策由脚本自动导出且先于训练冻结。**但净收益依赖成本口径**（见 §1）；racing 仍是外层编排而非算法内组件 |
| **解决前人解不了的任务** | 无。hurdle 虽是 TD-MPC2 解不了的（64.68/700），但 FastTD3 本身能解 |
| **新颖迁移性指标** | **已确定不存在**（十二族）。这从缺口转为**主张**：racing 是测量不是指标 |
| 独立重复 | RACING_K 两批同 seeds（CUDA 重复）；door 有独立的 4–6 与 7–9；slide 有独立的 4–6 |

## 6. 方法学产出（非实验结果，但决定了上述结论的可信度）

- 强制执行点 `CLAUDE.md`：执行层 §1–§7 + 设计层 §8（含"写出平凡解释并排除"）
- 教训 `M20–M33`：规范需强制加载点 / grep 一处≠读懂 / 缺失数据必须 INCOMPLETE /
  并行输出集合须不相交 / 单批 3/3 不足 / "答案不同"≠"排序反转" /
  共变剂量不能靠放宽容差 / U 标签 run-to-run 漂移 / 核实须到本场景实际值 /
  86 秒内重演刚否决的推理 + gate switching 红线 / U 符号跨 learner 反转 /
  源间差消去 student 基线漂移
- **四个设计在跑实验之前被拦下**（CGT、REJECT v1、MULTI、RACE-then-RUN）
- **两处会写进论文的转抄损耗由自查抓出**（racing 成本口径 30k/40k 混用；sibling gate 两方向稳健性不对称）
- **一个方向被一小时的廉价实验关闭**（族 12），代价约为 racing 方案的 1/1000

## 7. 复现入口

```
docs/data/hurdle_speedup_v1/hurdle_speedup_v1_results.json               SPEEDUP_CONFIRMED
docs/data/hurdle_selection_value_v1/results.json                         SELECTION_VALUABLE
docs/data/racing_min_horizon_v1/{compressed_lr,correct_lr}/results.json   RACING_CHEAP / RACING_VIABLE
docs/data/slide_generalizability_v1/results.json                         GEN_OK
docs/data/slide_speedup_v1/slide_speedup_v1_results.json                 SPEEDUP_REFUTED
docs/data/slide_hard_exit_v1/slide_hard_exit_v1_results.json             HARD_EXIT_SUPPORTED
docs/data/progress_screen_v1/results.json                                HOLDOUT_FAILED（族 12）
docs/data/racing_admission_v1/results.json                               ADMISSION_VIABLE
docs/data/endtoend_v1/results.json                                       ENDTOEND_SUPPORTED
docs/data/endtoend_v1/decisions.json                                     自动决策（先于训练冻结）
docs/data/racing_reject_door_v4/results.json                             PARTICIPANT_DIVERGED
docs/impossibility_characterization_of_transfer_prediction_20260730.md   十二族 + racing 的两半主张
docs/ISSUES_AND_LESSONS.md                                               E1–E15, M1–M33
docs/PAPER_CLAIMS_20260804.md                                            主张—证据—边界逐条映射
docs/PAPER_OUTLINE_20260804.md                                           章节结构与图表清单
CLAUDE.md                                                                强制执行规范
```
