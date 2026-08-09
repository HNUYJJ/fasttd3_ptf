# N1 预注册：Non-Displacing Transfer 最小因果门

> 冻结日期：2026-08-09。本文必须先于 seeds 4–8 的任何 N1 正式训练和评估
> 提交。实验只回答 replay 配额与行为子空间两个机制问题；不搜索 source、
> stage、dose、horizon、阈值或任务。

## 1. 核心问题与理论修正

Truck 的 10k→20k scaffold 在旧 seeds 1–3 上相对 scratch 为负。T4-R 后续
诊断中，ordinary physical replay 相对 fixed quota 的差值为 3/3 正，但旧工程
gate 错把累计 `q_cumulative` 与终点 `rho_endpoint` 合成了所谓 `A=0.545`。

正确语义是：

- `physical`：每一时刻在当前 allowed physical slots 上均匀采样，瞬时
  `q_t = rho_t`；
- `fixed`：source/student provenance strata 按固定 0.5/0.5 quota 采样；
- `q_cumulative < rho_endpoint` 是 late-entry cohort 的寿命效应，不是欠采样；
- 本实验禁止新增累计 `A=1` / catch-up replay 臂，因为那将是另一种
  age-prioritized replay 算法，而非自然对照。

## 2. 三个可证伪假设

在每个 seed 自己的同一 0→10k student anchor 上，令 20k source-free return
为 `J`：

1. **H_R（replay quota）**：`J_FP - J_FF > 0`。普通 physical replay 比
   固定 50% provenance quota 更少伤害。
2. **H_A（arms protection）**：`J_LP - J_FP > 0`。在相同 physical replay
   下，不让 locomotion bank 接管 arms 能改善目标学习。
3. **H_REC（recovery）**：`J_LP - J_S > 0`。保护 arms 后的 scaffold 至少不再
   低于纯 student。

这里 `LP-FP` 是“保护 arms 子空间”干预的**总效应**：它包含由动作变化引起的
occupancy 与 replay 内容变化，不冒充纯参数效应。Truck 的 hands 在 FF/FP 中
本来就由 student 控制；LP 新保护的只有 arms。

## 3. 冻结矩阵

目标固定 `h1hand-truck-v0`，learner seeds 固定 `{4,5,6,7,8}`，每个 seed
四臂共享同一 A0 anchor 与同一 `resume_noise_seed=95000+seed`。

| 臂 | behavior source groups | replay | admission |
|---|---|---|---|
| S | 无 source 执行 | exact student / uniform | none |
| FF | `legs_torso,arms` | fixed quota (`shared`) | all |
| FP | `legs_torso,arms` | ordinary physical | all |
| LP | `legs_torso`；arms 由 student | ordinary physical | all |

其余全部固定：同 hurdle4 bank、source mass 0.5、`h=25`、
`admission_bootstrap`、`bootstrap_only`、10k→20k、`num_updates=2`、
buffer 51200、batch 32768、128 envs、总 LR 日程 100k、provenance 两组。
四臂均不蒸馏、不训练 option/termination，不使用 MCG critic gate。

## 4. 工程 gate

任何一项失败，科学裁决记 `ENGINEERING_INVALID`：

1. 每 seed 四臂的 anchor 路径与 resume noise seed 完全相同；checkpoint
   `global_step=20000`，配置逐项匹配冻结矩阵。
2. source 臂 candidate execution share 在 `[0.45,0.55]`；S 的 source
   execution 与 replay counts 严格为零。
3. FF：`replay_physical=false`、`sampling_phase=authority_quota`、累计 critic
   source share 在 `[0.45,0.55]`。
4. FP/LP：`replay_physical=true`、`sampling_phase=physical_allowed`。在 buffer
   未 wrap、H=u=10k、mass=0.5 时，累计 physical replay 预期
   `0.5*(1-ln 2)=0.153426`，允许 `[0.13,0.18]`。该检查不与终点 rho 混算。
5. group provenance：FF/FP 的 `legs_torso` 与 `arms` endpoint source share
   均非零且二者相等；LP 的 legs 非零、arms **严格为零**；S 两组均为零。
6. source-free evaluator 固定 panel128、deterministic、128 episodes，环境、
   seed、checkpoint step 与 arm 身份正确。

## 5. 科学判序

每个 contrast 先报告五个 paired learner-seed 差值、均值、sample SD、90% t-CI
（df=4；episode 不是独立重复）。主判据只用预先冻结的 learner-seed signs：

- `DIRECTIONAL_SUPPORT`：mean > 0 且至少 4/5 seed > 0；
- `DIRECTIONAL_REFUTATION`：mean <= 0 或至多 2/5 seed > 0；
- `UNRESOLVED`：其余（主要是 3/5 正且 mean > 0）。

只有 H_R、H_A、H_REC **三项同时** `DIRECTIONAL_SUPPORT`，才允许把
Non-Displacing Transfer 升级为算法主线。任何一项不满足都不自动增加 seed、
不换 stage/task/source、不调 dose/horizon，不开发 catch-up replay。

## 6. 结论边界

- 成功只支持 Truck、10k→20k、当前 hurdle4 locomotion bank 的机制可行性，
  不是通用迁移性指标。
- 失败将关闭“ordinary physical replay + arms protection”这条最小组合，而非
  否定所有可能的结构化 source reuse。
- 旧 seeds 1–3 只用于提出假设；本轮确认性裁决只认 fresh seeds 4–8。
