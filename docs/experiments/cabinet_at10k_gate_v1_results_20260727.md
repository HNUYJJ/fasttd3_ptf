# Cabinet@10k standardized equal-dose calibration gate — 结果与裁决

> 日期：2026-07-27  
> 目的：在**同一个 target task 内**制造 source 异质性标签，打破 hurdle(全正)/crawl(全负)
> 的"标签符号与任务身份完全共线"。  
> **裁决：`CABINET_UNCERTAIN`** → 按预注册规则停止本轮指标建模，不追加任务/seed、不改阈值。  
> **裁决措辞（PI 冻结）**：在 3 个训练 seed、每臂 32 个评估 episode 下，无法可靠判定
> Cabinet@10k→20k 的 source-specific transfer effect；**不代表三种 source 的真实效应相同**。
> 这是一个**有效的负结果**，不是执行失败。

## 1. 冻结协议与执行完整性

| 项 | 设计 | 实测 |
|---|---|---|
| anchor | 每 seed 一个 10k exact-abstention 纯 student anchor | 3/3 完成，env/seed/步数校验通过 |
| 臂 | stand / walk / run / student-only × seeds 1,2,3 | 12/12 完成，全部 `Resumed core learner ... at step 10000` |
| dose | behavior 0.5 / replay 0.5 | **behavior 0.4985–0.5002；critic 0.4988–0.4990** |
| 同 seed 内跨源剂量一致性 | 必须一致 | **完全一致**（s1 全为 0.4985/0.4990，s2 0.5002/0.4989，s3 0.4990/0.4988） |
| horizon / 执行 | h=25、full-action、bootstrap_only | 一致，`sampling_phase=authority_quota` |
| 噪声重采样 | 同 seed 四臂共享 `resume_noise_seed` | 一致（90000+seed） |
| 评估 | 20k 冻结 source-free，4 eval seeds × 8 ranks | 12/12 × 32 episodes |

执行层无缺陷：臂间**唯一差异就是 source 身份**。

### 1.1 执行中发现并修复的接线缺陷（影响复现，必须记录）

`train_ptf.py` 支持 `--ptf-anchor-resume` / `--ptf-resume-noise-seed`，但
`scripts/official_fasttd3_train_target_ptf.sh` **未转发**这两个环境变量（历史 P0
实验直接调 `python -m ...train_ptf`，绕过 wrapper，故缺口从未暴露）。首次启动时进度条
从 0 而非 10000 起，被当场发现并在约 1 分钟内截停（浪费约 2 分钟算力）。已做最小
additive 修复（只转发既有 CLI flag，不改训练语义），重跑 smoke 确认
`Resumed core learner ... at step 10000` 后才开始正式矩阵。首次 smoke 的"通过"是
假阳性：它同样从 0 跑，admission share 0.49 看似正常，掩盖了 resume 未发生。

## 2. 主结果

标签：\(U_i(10\text{k},10\text{k})=J^{sf}_{@20\text{k}}(\text{source}_i)-J^{sf}_{@20\text{k}}(\text{student})\)，同 seed 配对；
90% t 区间（df=2）。

| source | per-seed U (s1/s2/s3) | U mean | lo90 | hi90 | 分类 |
|---|---|---:|---:|---:|---|
| stand | −7.61 / −6.50 / +5.96 | −2.717 | −15.426 | +9.992 | uncertain |
| walk | +57.69 / −4.58 / −4.41 | +16.232 | −44.297 | +76.762 | uncertain |
| run | −23.63 / +37.27 / +23.27 | +12.305 | −41.473 | +66.083 | uncertain |

student 基线 \(J^{sf}_{@20k}\)：42.93 / 33.84 / 18.16。

三者区间全部跨 0 → 既无 helpful 也无 harmful → **`CABINET_UNCERTAIN`**。

注意：`uncertain` 的含义是**未能判定**，不是**判定为零**。本表不支持"三种 source
效应相同"这一结论。

## 3. 失败机制：标签在该 stage/horizon 上不可分辨

这不是"三个源效果相同"，而是**结果测量本身无法分辨**。

Cabinet 在 20k 的 source-free 回报是**罕见事件主导的重尾分布**：中位数仅 11–28，
但最大值 33–706，超过 50 分的 episode 占比 0–41%。均值由少数"真正打开柜门"的
episode 支配。

| arm/seed | mean | 面板 SE | median | p90 | max | frac>50 |
|---|---:|---:|---:|---:|---:|---:|
| walk s1 | 100.62 | 28.24 | 28.43 | 316.44 | 706.44 | 0.41 |
| run s2 | 71.11 | 20.08 | 20.93 | 154.46 | 473.21 | 0.28 |
| walk s3 | 13.75 | 1.20 | 11.49 | 21.60 | 33.10 | 0.00 |
| student s1 | 42.93 | 17.49 | 11.89 | 22.01 | 391.54 | 0.09 |

把每个 per-seed U 与其两臂面板 SE 合成噪声比较：

| cell | U | panel SE | \|U\|/SE |
|---|---:|---:|---:|
| stand s1/s2/s3 | −7.61 / −6.50 / +5.96 | 24.04 / 19.26 / 4.76 | 0.32 / 0.34 / 1.25 |
| walk s1/s2/s3 | +57.69 / −4.58 / −4.41 | 33.22 / 22.74 / 3.25 | **1.74** / 0.20 / 1.36 |
| run s1/s2/s3 | −23.63 / +37.27 / +23.27 | 18.72 / 26.33 / 16.45 | 1.26 / 1.42 / 1.41 |

**没有任何一个 per-seed 效应超过约 1.74 个面板标准误**。因此本实验的准确结论是：
**当前数据无法区分 episode-level 重尾评估噪声与 learner-seed 之间的真实效应异质性。**
walk 在 s1(+57.69) 与 s2/s3(≈−4.5) 之间的落差，既可能来自面板恰好抽到更多成功
episode，也可能来自不同 learner seed 上确实不同的干预效应——本实验的分辨力不足以
分离这两种解释，**不得据此断言任一方向**。

因此 `CABINET_UNCERTAIN` 的准确含义是：

> 在 3 个训练 seed、每臂 32 个评估 episode 的预算下，**无法可靠判定
> Cabinet@10k→20k 的 source-specific transfer effect**；这**不代表**三种 source
> 的真实效应相同。

即：U 标签在 cabinet@20k 这一 stage/horizon 上的信噪比低于本实验的分辨力，
结论是关于**可测性**的，不是关于**效应值**的。

## 4. 裁决与停止

按预注册规则执行：

- **停止本轮 SCTU/指标建模**；
- **不追加 seed、不延长 K、不换评估指标、不改分类阈值**；
- **不新增 target task 抢救**；
- Basketball 继续保留为完全未见的外部 abstention 测试，未参与本阶段任何收集与选择。

任务内异质性**未能建立**，因此 §"标签符号与任务身份完全共线"的问题仍然存在：
当前 A 级标签仍只覆盖 hurdle(全正)/crawl(全负) 两个任务，leave-one-target-task-out
依旧不可做。

## 5. 本轮产生的一条可复用约束（属结果记录，不是补救提案）

标签可测性应当成为**未来任何 U 标签采集任务的前置筛选条件**：

> 若某 target 在标签 horizon 上的 source-free 回报由罕见事件主导（中位数远低于均值、
> 成功 episode 占比很低），则其 U 标签的面板噪声会淹没干预效应，在可行的 seed 规模下
> 不可分辨。

Hurdle 与 Crawl 之所以能给出干净标签，正是因为它们在对应 horizon 上的回报分布密集；
Cabinet 在 20k 不满足该条件。是否、以及如何使用这条约束，由 PI 决定。

## 6. 产物

- 结果 JSON：`docs/data/cabinet_at10k_gate_v1/cabinet_at10k_gate_v1_results.json`
- 12 份冻结评估：`docs/data/cabinet_at10k_gate_v1/source_free_eval/*.json`
- 训练日志：`logs/train/cabinet_at10k_gate_v1/`
- anchors：`artifacts/cabinet_at10k_gate_v1/anchors/s{1,2,3}/`
- 裁决脚本（揭盲前定稿）：`scripts/analysis/analyze_cabinet_at10k_gate_v1.py`
- 矩阵/评估启动脚本：`scripts/run_cabinet_at10k_gate_v1.sh`、
  `scripts/analysis/run_cabinet_at10k_eval_v1.sh`
- 新增任务契约与 bank：`configs/target_evidence/humanoidbench_cabinet_v1.yaml`、
  `configs/source_banks/calibration/h1hand_cabinet_rbo_{stand,walk,run}.yaml`
- W&B project：`ptf_fasttd3_source_calibration`
