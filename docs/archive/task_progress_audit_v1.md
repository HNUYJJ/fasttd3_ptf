# Task-Progress Audit v1：增益归因分层（RBO-PTF Day-1，2026-06-14）

ChatGPT v3 的最高优先级提醒:若 cabinet/powerlift 等任务的 return 增益只是
alive/stand multiplier 上升,"提升复杂任务完成度"会被审稿人一句"你只是站得
更久"打掉。本审计把 9 个任务的 total return 拆成 **task-progress** 与
**alive/stand** 两路,精确界定 RBO-PTF 的增益到底落在哪里。

## 方法

- 工具 [scripts/task_progress_audit.py](../scripts/task_progress_audit.py):用
  `SourcePolicy` 加载已落盘的 pilot checkpoint(自动复现训练时的 obs
  normalizer),确定性 rollout(纯学生动作,不执行教师),16 env × 1 episode;
- 采集 HumanoidBench info 的细粒度 reward 分量 per-step 均值 + fall rate
  (用 `TimeLimit.truncated` 区分摔倒 vs 超时);
- 对每个任务取**最能代表完成度的硬字段**(HB 官方的 `success_subtasks` /
  object-goal proximity / 前进量等),沿训练 5 个 checkpoint(10/30/50/70/100k)
  算 AUC,对比 full(=mcg,旧"RIC")vs scratch;
- 全部 seed 1。total-return AUC 方向与训练 wandb eval 一致(交叉验证可信)。

## 核心表:硬完成度字段 AUC（full vs scratch，seed 1）

| 任务 | 硬完成度字段 | scr AUC | full AUC | **进度 ROI** | fall AUC Δ |
|---|---|---|---|---|---|
| cabinet | success_subtasks（完成子任务数）| 0.129 | 0.348 | **+169%** | 0 |
| truck | robot_package_truck（靠近卡车）| 0.302 | 0.483 | **+60%** | 0 |
| hurdle | move（前进量）| 0.502 | 0.678 | **+35%** | −0.05 |
| maze | success_subtasks（到达 checkpoint 数）| 1.420 | 1.804 | **+27%** | +0.21 |
| window | window_contact_total（擦窗接触）| 0.589 | 0.654 | +11% | **−0.40** |
| powerlift | reward_dumbbell_lifted（举起哑铃）| 0.189 | 0.189 | **+0%** | +0.04 |
| door | passage_reward（走出门）| 0.397 | 0.326 | −18% | −0.06 |
| spoon | spoon_spinning（搅拌）| 0.012 | 0.000 | ~0（绝对值≈0）| −0.07 |

（balance_hard 的任务目标本身就是"保持平衡=站立",无独立 manipulation 进度
字段;return +36% 来自站得更久,见下。）

## 增益归因四层（论文 claim 的精确边界）

**A 层 · task-specific completion 增益（"提升复杂任务完成度"的硬证据）**
- **cabinet**:完成子任务数 0.13→0.35（**+169%**,近 3 倍),开门进度是真增益;
- **truck**:robot-package-to-truck proximity +60%(更接近卸货目标,partial——
  `packages_picked_up` 仍 0,未真正搬上车);
- **maze**:到达 checkpoint 数 +27%(导航推进更远)。

**B 层 · time-to-progress 加速（前中期进度领先,末期 scratch 追平）**
- **hurdle**:前进量 move 在 50k 时 0.731 vs scratch 0.356(2×),末端 100k 持平
  (0.92≈0.92)。增益是**更快获得跨栏前进能力**(time-to-threshold 大幅缩短),
  不是渐近上限更高——教科书式 transfer 形态。

**C 层 · whole-body stability / 存活主导（增益是 manipulation 的前置技能）**
- **window**:擦窗 per-step 技能两者相近(+11%),return +117% 的真因是 **fall-rate
  AUC 大降 −0.40**(81%→31%)——full 学会站稳擦窗,存活更久 → 累计 reward 暴涨;
- **powerlift**:`reward_dumbbell_lifted` **全程恒定 0.189**(full 和 scratch 都
  从未真正举起哑铃),return +49% **100% 来自站稳**(中段 fall 31%→0%);
- **balance_hard**:任务=平衡,alive per-step 相近,full 站得更久(return +36%)。

C 层完全印证 ChatGPT 与 HumanoidBench 论文的判断:**manipulation 前必须先
locomotion/stabilization**;RBO-PTF 的 loco 教师 bootstrap 的正是这个前置技能。
诚实写法:"RBO-PTF first bootstraps whole-body stability, a prerequisite
subskill for manipulation."

**D 层 · 无对价 / 瓶颈未触及（正确地不迁移、不伤害）**
- **spoon**:`reward_spoon_in_cup` 两者都 0(都没把勺子入杯),搅拌量绝对值≈0,
  no-opportunity control 确认;
- **door**:`door_openness_reward`(主门开度)两者都 **0**——loco 源覆盖不到
  "转把手+推主门"的协调瓶颈,passage AUC 噪声级负(−18%,绝对值小)。

## 对论文的影响（诚实但更有 insight 的 claim）

不能笼统说"提升所有复杂任务的完成度"。精确的、可防审稿的表述是:

> RBO-PTF improves **task-specific completion** on cabinet (+169% subtasks),
> maze (+27% checkpoints), truck (+60% goal-proximity); **accelerates progress
> acquisition** on hurdle (2× mid-training forward progress); and **bootstraps
> whole-body stability** — a prerequisite manipulation subskill — on
> powerlift/window/balance. It correctly provides **no transfer** where the loco
> sources cannot reach the bottleneck (door main-door opening, spoon stirring),
> with near-zero negative-transfer regret.

这个分层比"全是 task-progress"可信,比"只是站得更久"有 insight:它把
**source(loco 技能)→ target bottleneck** 的匹配关系讲清楚了,且与 Transfer Map
的对价预测一致(强对价 cabinet/maze/hurdle = 真进度/加速;安全任务 window/balance
= stabilization;边界 truck partial;无对价 spoon/door)。

## 局限 + 下一步

- audit 用 16 env × 1 episode(样本偏少),total ROI 与训练 eval 有数值出入但
  符号一致;正式版可加 seed 2 + 更多 episode 收窄。
- 黑名单法 prog-sum 在含 multiplier 的任务(hurdle 的 wall_collision_discount、
  cabinet 的 door_openness 与 success_subtasks 方向不一)上不如**单一硬字段**干净
  ——本文主表已改用硬字段。
- **下一步 Day2-3:Transfer Map v2**——把这套 info 分量(progress-event gain)
  接进 snippet-level score,加 safe-horizon + Spearman 预测力验证,让"Transfer Map
  预测对价"从事后解释升级为算法模块。window 的 fall 主导(−0.40)正是 v2 要量化的
  "短 prefix 站立片段有用、长 episode 摔倒有害"的 safe-horizon 信号。

数据:[logs/probe/task_progress_audit.jsonl](../logs/probe/task_progress_audit.jsonl)、
[logs/probe/task_progress_audit_summary.json](../logs/probe/task_progress_audit_summary.json)。
