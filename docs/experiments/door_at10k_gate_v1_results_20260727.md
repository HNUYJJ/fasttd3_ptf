# Door@10k standardized equal-dose calibration gate — 结果与裁决

> 日期：2026-07-27  
> 定位（PI 冻结，见预注册 commit `5944792`）：**有历史行为先验的定向 RBO 学习效用标定**，
> 不是盲测，也不是外部验证。  
> **裁决：`DOOR_ALL_SAME_SIGN`** → 按预注册规则停止本轮指标建模，不追加任务/seed、
> 不延长 K、不换指标、不改阈值。

> **⚠️ 2026-07-31 推广范围修正（`0d594ad`）**：本文的"三个 locomotion 源一致有害（9/9 per-seed 全负）"
> 经 `RACING_REJECT v4` 检验，**须限制为"在 seeds 1–6 上"**。
> 新 learner 批 `s7–9` 出现 2/9 为正，其中 `s9` 的 `run = +36.32 ± 3.95` **显著正**
> （本文对 run 的结论是 −30.63，跨度 67）。
> 本文在其自身 learner 上的测量仍然可靠（holdout `s4–6` 复现符号，共 18/18 负），
> 但"一致有害"是该 **learner 子总体**的性质，不是 door 这个 target 的性质。
> 详见 `docs/experiments/racing_reject_door_v4_results_20260731.md` 与 `M31`。


## 1. 一句话结论

Door 的标签**测得非常干净**（`|U|/配对面板SE` 中位 ≈ 9，Cabinet 当时 ≤ 1.74），
但三个 locomotion 源在 door@10k→20k 上**一致有害**：9/9 个 per-seed 效应全为负。
**任务内异质性未能建立。**

同时得到一个比裁决本身更重要的结果：**学习效用的排序与行为层先验反向**（§5）。

## 2. 执行完整性

| 项 | 设计 | 实测 |
|---|---|---|
| anchor | 每 seed 一个 10k exact-abstention 纯 student anchor | 3/3，学习器/replay/rng/manifest/checksums 齐全 |
| 臂 | stand / walk / run / student × seeds 1,2,3 | **12/12 全部 `Resumed core learner ... at step 10000`** |
| bank 身份 | 逐臂单源 | `['null']`×3、`['stand']`×3、`['walk']`×3、`['run']`×3 |
| dose | behavior 0.5 / replay 0.5 | **behavior 0.4978–0.5005；critic 0.4987–0.4990** |
| horizon / 执行 | h=25、full-action、bootstrap_only | 一致，`sampling_phase=authority_quota` |
| 噪声重采样 | 同 seed 四臂共享 `resume_noise_seed` | 一致（91000+seed） |
| 评估 | 20k 冻结 source-free，**128** deterministic episodes | 12/12 × 128（16 eval seeds × 8 ranks） |

同 seed 内跨源的 behavior share 有 ≤0.3% 的微小差异（Cabinet 当时是完全相同）。
原因是 door 的轨迹会随源不同而分叉、episode 边界不再一致，而 cabinet 不因摔倒终止。
这是**源身份的必然后果，不是协议缺陷**——偏差比任何观测到的效应小两个数量级，且真正
进入学习的 critic 采样占比在同 seed 内仍近乎逐位一致。

### 2.1 执行中的一次故障（必须记录）

首次启动的 12 臂矩阵在约 27 分钟后被整体中断：**整个 tmux server 被杀**，而非单个会话退出。
`student_s1` 已完成，`stand_s1`(85%) 与 `student_s3`(94%) 被杀。

**不是 OOM**——内存 566G 总量仅用 31G，可用 535G，日志无任何 traceback。
根因：`tmux new-session -d` 启动的 server 继承了工具调用的进程组/会话，调用结束即被一并清理。
改用 `setsid tmux new-session -d` 后经实测确认隔离生效：训练进程祖先链终于 `tmux: server`
且其 PPID=1（被 init 收养），SID 与工具 shell 不同，启动它的 bash 退出后训练照常存活。

三个 anchor 全部完好未受影响，实际损失约 25 分钟算力。**协议与预注册未受任何影响**
（预注册早于故障提交）。

## 3. 主结果

标签：\(U_i(10\text{k},10\text{k})=J^{sf}_{@20\text{k}}(\text{source}_i)-J^{sf}_{@20\text{k}}(\text{student})\)，同 seed 配对；
90% t 区间（df=2）。

### PRIMARY 面板（128 episodes，裁决依据）

| source | per-seed U (s1/s2/s3) | U mean | lo90 | hi90 | 分类 |
|---|---|---:|---:|---:|---|
| stand | −23.43 / −42.83 / −31.66 | **−32.64** | −49.05 | −16.23 | **harmful** |
| walk | −7.06 / −38.58 / −20.96 | −22.20 | −48.83 | +4.43 | uncertain |
| run | −17.78 / −41.04 / −33.08 | **−30.63** | −50.56 | −10.71 | **harmful** |

student 基线 \(J^{sf}_{@20k}\)：265.47 / 275.45 / 261.69。

### SECONDARY 子面板（前 32 episodes，与既有面板逐位兼容，不参与裁决）

| source | U mean | lo90 | hi90 | 分类 |
|---|---:|---:|---:|---|
| stand | −32.37 | −46.40 | −18.35 | harmful |
| walk | −17.92 | −48.36 | +12.52 | uncertain |
| run | −34.15 | −56.45 | −11.85 | harmful |

**两个面板给出完全相同的分类**，结论对面板规模稳健。

裁决：两个可判定的源同为 harmful → **`DOOR_ALL_SAME_SIGN`**。

## 4. 这次是"真结论"，不是"测不出来"

与 Cabinet 的决定性区别在于测量分辨力。把每个 per-seed 效应与其两臂配对面板噪声相比：

| source | s1 | s2 | s3 |
|---|---:|---:|---:|
| stand | 4.41 | 11.95 | 9.30 |
| walk | 1.92 | 13.88 | 7.29 |
| run | 4.80 | 18.79 | 9.02 |

`|U|/配对面板SE` 中位约 **9**，最小 1.92。Cabinet 当时**没有任何一个** cell 超过 1.74。
零训练审计对 door 可测性的筛选**被实测验证**。

student 臂的回报分布也确实密集：median 265–277，128-episode 面板 SE 仅 0.93–3.26
（cabinet 当时 SE 1.20–28.24、median 仅 11–28）。

### 4.1 一条方法学记录（属结果记录，不是补救提案）

区间宽度并非由面板噪声主导。以 stand 为例：per-seed U 的 sd = 9.72，而配对面板 SE 平均仅 4.1。
即 **sd(U) 主要由真实的 learner-seed 异质性构成**。

推论：把面板从 32 加到 128 让每个 per-seed U 的估计非常可靠（这是本次能明确判定的原因），
但要**收窄 3-seed 区间**，应该加 seed 而不是加 episode。本轮不执行——预注册禁止加 seed，
且加 seed 也改变不了结论：9/9 个 per-seed 效应已经全为负。

## 5. 核心发现：学习效用与行为先验**反向**

| source | Transfer Map v1 行为(zero=64) | 行为相对 | 学习效用 U | 分类 |
|---|---:|---:|---:|---|
| **run** | 101 | **+58%** | **−30.63** | harmful |
| stand | 59 | −8% | −32.64 | harmful |
| **walk** | 25（62% 摔） | **−61%** | **−22.20** | uncertain（三者中最不负） |

- 行为层排序：run ≫ stand ≈ zero > walk
- 学习效用排序：walk > run ≈ stand

**行为层表现最好的 run，在学习效用上是 harmful；行为层最差、摔倒率 62% 的 walk，
反而是三者中唯一未被判为有害的。**

这是本项目"**行为即时效果 ≠ 延迟学习价值**"这一论断的**第一个同任务内直接证据**。
此前的证据都是跨任务的（hurdle 全正、crawl 全负，行为量根本无法区分二者），
本次则在固定 target、固定 stage、固定剂量、配对设计下取得，控制了全部其他变量。

### 5.1 对"迁移性指标"的直接后果

预注册时列出的三个分支中，实际落在了第二个：

- ~~与先验同向 → zero-shot 行为探针可能是廉价可用的指标~~ → **已被否决**
- **与先验反向 → 行为层信息不仅无用而且误导，必须换非行为信号族** ← 本次结果
- ~~全同号 → 概念本身有问题~~ → 同时也发生了（见 §6）

因此 **zero-shot 行为探针作为迁移性指标候选，就此关闭**。它与 T⁰、T^critic sign、
SIV、SHU、adaptive revocation、P0 lease oracle、update-space influence 归入同一失败族，
且失败机制相同：度量的是即时量，而 U 是延迟学习价值。

## 6. 裁决与停止

按预注册规则执行：

- **停止本轮指标建模**；
- **不追加 seed、不延长 K、不换评估指标、不改分类阈值、不新增 target 抢救**；
- Basketball 继续保留为完全未见的外部 abstention 测试。

任务内异质性**未能建立**。当前标签格局：

| target | 标签 | 可测性 |
|---|---|---|
| hurdle | 三源全正 | 可测 |
| crawl | 三源全负 | 可测 |
| **door** | **三源全负**（2 harmful + 1 uncertain，9/9 per-seed 负） | **可测（本次证实）** |
| cabinet | 不可判定 | 不可测 |

**标签符号与任务身份仍然完全共线**，leave-one-target-task-out 依旧不可做。

而且这次的信息比 Cabinet 强得多：Cabinet 留下的是"测不出来"的悬念，Door 关掉了这个悬念——
在一个**已证实可测**的场地上，任务内异质性**确实不存在**。这构成对
\(U \approx f(\text{target})\) 的正面证据：若 U 主要由 target 决定、source 只起缩放作用，
那么"迁移性指标"就没有可预测的内容——只要知道 target 是什么就能预测 U。

是否据此裁定停止跨任务 learned transferability metric 路线、把论文主线退回已证的
静态 RBO / exact abstention / 有限暴露 / replay lifecycle，由 PI 决定。**本轮不再启动任何训练。**

## 7. 产物

- 结果 JSON：`docs/data/door_at10k_gate_v1/door_at10k_gate_v1_results.json`
- 12 份冻结评估（各 128 episodes）：`docs/data/door_at10k_gate_v1/source_free_eval/*.json`
- 训练日志：`logs/train/door_at10k_gate_v1/`
- anchors：`artifacts/door_at10k_gate_v1/anchors/s{1,2,3}/`
- 裁决脚本（**揭盲前定稿并提交于 `5944792`**）：`scripts/analysis/analyze_door_at10k_gate_v1.py`
- 矩阵/评估脚本：`scripts/run_door_at10k_gate_v1.sh`、`scripts/analysis/run_door_at10k_eval_v1.sh`
- 单源 bank：`configs/source_banks/calibration/h1hand_door_rbo_{stand,walk,run}.yaml`
- W&B project：`ptf_fasttd3_source_calibration`
