# Adaptive Admission v1（Phase A）：正式结果与预注册裁决

> 日期：2026-07-15
> 冻结 stamp：`20260714T110054Z`；预注册：`configs/experiments/adaptive_admission_v1.yaml`（发车前 SHA256 冻结）
> 完整性：18/18 runs 完成、exit code 全 0；双并发队列无重试；W&B 在线
> **总裁决：Phase A FAIL（crawl 收益 gate 与 truck 无伤害 gate 双 FAIL；powerlift 保持 gate PASS；basketball 描述性）。按 run card A.6 止损条款，adaptive behavioral-source revocation 不进入主方法。**
> 定性：这是一次**高信息量的预注册负结果**——它用单变量干预实验精确刻画了"行为层即时 reward 信号"驱动源退出决策的能力边界，并构成该信号族的第三次独立否定（SIV、SHU 之后）。

## 1. 预注册 gate 裁决（字面执行）

统计量 = 10k–95k evaluation-grid mean return（5k 网格 18 点均值）。crawl/basketball 使用本批同 seed、same-launch static；truck/powerlift 使用预注册并通过哈希认证的 `20260713THANDOFFV1Z` 历史 fix 对照。科学配置的预期差异是 adaptive 开关，但 CUDA 跨进程 learner 非确定性意味着它们不是逐 bit counterfactual。

| 任务 | per-seed Δ（adaptive − 对照） | mean（t, df=2） | gate | 裁决 |
|---|---|---|---|---|
| crawl | +41.5 / **−66.8** / +53.9 | +9.5（t=0.25） | Δ≥+30 且 3/3 正 且有撤销事件 | **FAIL**（s2 无撤销且为负；mean 未达标） |
| truck | −6.0 / **−119.7** / **−204.9** | −110.2（t=−1.91） | \|Δ\|≤60 且 hurdle/walk/run 禁撤 | **FAIL**（禁撤 3/3 违反 + s2/s3 超限） |
| powerlift | −4.7 / −2.0 / −4.9 | −3.9（t=−4.09） | Δ≥−20 | **PASS** |
| basketball | −23.7 / +36.2 / −34.7 | −7.4（t=−0.34） | 描述性 | 判据触发（推翻"可能不触发"的预登记预期），无系统性改善 |

## 2. 撤销事件全记录（从 replay policy_events 重建，时点为 global step）

| run | 撤销链 | 最终保留 |
|---|---|---|
| crawl_s1 | 24k: walk, run | stand |
| crawl_s2 | **无撤销** | stand, walk, run |
| crawl_s3 | 18k: run；21k: walk | stand |
| truck_s1 | 12k: walk；**15k: hurdle** | stand, run |
| truck_s2 | **15k: walk, hurdle**；21k: run | stand |
| truck_s3 | 12k: walk；**15k: hurdle**；18k: run | stand |
| powerlift_s1/s2 | 9k: crawl, reach | 其余 7 源 |
| powerlift_s3 | 9k: crawl, reach；27k: slide | 其余 6 源 |
| basketball_s1 | 9k: crawl, reach；12k: slide；21k: stair；24k: pole | stand, walk, run, hurdle |
| basketball_s2 | 9k: crawl, reach；15k: slide | stand, walk, run, hurdle, stair, pole |
| basketball_s3 | 9k: crawl, reach；12k: slide；18k: stair；21k: pole；27k: hurdle；30k: run | stand, walk |

机制运行本身与预注册完全一致：全部撤销发生在 3000 的整数倍步、最早 9k（persistence=3 窗的理论下界）、无跳窗、无跨窗 stale vote；"最终保留 stand"的模式是最小证据门的保守语义所致（stand 份额近零 → 每窗攒不够 20 segments → 永无投票资格 → 永不撤）。

## 3. 机制解读：判据的能力与边界被精确刻画

### 3.1 系统性偏差：正迁移 source bank 的即时 reward 排序与学习价值错位（truck 的教训）

truck 的 stand/walk/run/hurdle **整体 bank** 已被 `admission_handoff_v1` 证明能保留显著正迁移（95k fix−scratch `+227.8`）；本轮判据却在 12k–21k 撤掉 walk/hurdle（3/3）和 run（2/3），并在 s2/s3 产生 `−119.7/−204.9` 的 evaluation-grid 代价。这里不能进一步声称 walk、run、hurdle 每个单独都是已证好源，因为既有实验没有单源训练归因；可以严格声称的是：**局部即时 segment reward 判据未能保护一个已知有整体学习价值的 source bank**。

一种与证据一致、但仍属机制解释而非直接测量的原因是：引导型 source 可能承担低即时 reward 的过渡/覆盖工作，而价值在后续 student update 与 source-free 学习中实现。关键时序支持这种“延迟学习价值”解释：source 在 12k–21k 被撤，30k 后 adaptive 与 fix 都没有 source behavior authority，但 truck 差值从 10–30k 的三 seed 均值 `−62.4` 扩大到 35–95k 的 `−128.6`。大样本使置信区间很窄，只会让一个有偏 estimand 更稳定地触发，并不能修复 estimand 错位。

### 3.2 能力边界的另一侧：判据能稳定移除低即时 reward strata（powerlift 的兼容性证据）

powerlift 3/3 在最早允许点（9k）一致撤掉 crawl/reach，且保持 gate PASS（per-seed Δ=`−4.7/−2.0/−4.9`）；basketball 也 3/3 首撤 crawl/reach。它支持的最强结论是：**判据可以重复识别并移除相对 student 即时 reward 明显偏低的 source strata，而且在 powerlift 上与原性能兼容**。由于没有 crawl-only/reach-only 的 powerlift 学习价值标签，这还不能升级为“已可靠识别零 learning-utility source”，更不能验证通用 source selector。

### 3.3 crawl 的部分信号

s1/s3 的全程均值差为 `+41.5/+53.9`，事件对齐后的 post-minus-pre 分别为 `+129.2/+88.6`，属于与“撤销截断伤害”一致的探索性信号。但它不能被写成真实因果机制：s2 从未触发，adaptive/static 的 30k/60k/90k/final execution、replay、critic strata 均完全一致，却仍出现 `−66.8` 的 placebo AUC 差；短 smoke 也已证明同算法 CUDA learner 会跨进程分叉。因此正式分析器将 `mechanism_attribution_supported=false`，crawl 只能表述为“2/3 triggered seeds 有相符信号，但预注册效应与因果归因均未成立”。

### 3.4 basketball：预期被推翻的方式本身有信息

预登记预期是“存活 reward 掩护 → 判据可能不触发”；实际判据大量触发（s3 最终只保留 stand/walk），但相对 static 的三 seed 效应为 `−23.7/+36.2/−34.7`，没有系统性改善。这说明“发生大量撤销”不等于找到了造成负迁移的 source。现有实验没有 basketball 单源训练标签，因此不能把伤害主体因果归于 walk/run，也不能用 FastDSAC 的 body-rebound 发现替代这种 source-level 归因；严格结论只是：**即时 reward 排序没有产生可复现的 basketball 负迁移修复**。

## 4. 对研究主线的意义：行为信号族的第三次独立否定

| 路线 | 形式 | 否定方式 |
|---|---|---|
| SIV 2×2（2026-07-11） | per-source 因果干预打分 | 机制信号未过实践阈值（T=−0.048 < 0.10） |
| SHU（2026-07-12） | 阶段条件化准入判据 | 行为正/下游更新负的 mandatory contradiction |
| **adaptive revocation（本轮）** | **时间维聚合撤销（segment 级置信比较）** | **预注册干预实验：3/3 误撤已证好源，代价 −120~−205** |

三种不同设计（打分/准入/撤销）和统计形式（因果分解/置信门/序贯窗口投票）指向同一边界：**目前试过的行为 reward 代理不足以支撑可靠的自动 source admission/revocation**。这里的“不同设计”不是统计独立重复，也不能证明所有可能的行为信号都无效。现有证据更支持把 source 的行为表现与其对后续 learner/replay 的价值分成两个 estimand。

论文定位由此确定：
- 主方法维持**静态 RBO + admission lifecycle（被动正确性：exact abstention / quarantine / revocation / handoff——全部已验证）**；
- adaptive revocation 作为预注册负结果写入，与 SIV/SHU 组成"行为信号三重否定"证据链——这**强化**了双通道论点（behavior score ≠ learning utility）并将其推进到"任何已试形式"的强度；
- powerlift 的选择性撤销（识别无关源的能力）如实报告为判据能力的正面边界；
- "自动源退出"列为 open problem，且现在有三重证据说明它为什么难：可行的下一个方向只能来自**非行为通道**的信号（如学生侧的 learning progress、replay 通道的 TD 统计），不应再消耗预算于行为 reward 的第四种变体。

## 5. 工程审计记录

- 18/18 exit=0；orchestrator 无重试；每 run meta 含 bank/protocol/implementation SHA；
- 撤销时点/内容由 replay `policy_events` 完整重建（第 2 节表）；
- `admission_history` 已作为 checkpoint `admission_audit.decision_history` 持久化，包含全部窗口统计、LCB/UCB、persistence、撤销与 apply-time replay/execution 快照；独立 finalizer 已直接从 final checkpoints 离线重建全部撤销及统计。因此不存在“窗口统计只在 W&B、无法离线复算”的缺口。`replay.policy_events` 另行保存 admission policy 变化，二者共同支持生命周期审计。
- W&B API 独立核验 18/18 run 均为 `finished`，最终 `_step=99900`；final checkpoints 的 `global_step=100000`。

## 6. 证据索引

- 训练日志：`logs/train/adaptive_admission_v1_20260714T110054Z/`（18 runs + orchestrator）
- checkpoints：30k/60k/90k/final × 18（`models/*adaptive_admission_v1*`）
- W&B：project `fasttd3_ptf`，run 名含 `adaptive_admission_v1_*_20260714T110054Z`
- 预注册：`configs/experiments/adaptive_admission_v1.yaml`（launch 前冻结）
- 协作记录：`docs/agent_collab/claude_chatgpt_20260713_rbo_admission.md` T0007–T0017
