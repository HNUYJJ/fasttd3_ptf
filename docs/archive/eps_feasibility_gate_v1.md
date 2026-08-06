# EPS feasibility gate v1：预注册与实现契约

日期：2026-07-11  
状态：**暂停；不得实现或启动。先通过 stage-conditioned source-admission 机制门，再决定是否恢复**  
上位决策：[`paper_core_contribution_reconstruction_v1.md`](paper_core_contribution_reconstruction_v1.md)

> 2026-07-12 PI 裁决：EPS 不再是下一步唯一方法门。它降为“source 已通过准入之后”的可选
> action-authority 模块；当前优先级由
> [`paper_core_contribution_reconstruction_v2.md`](paper_core_contribution_reconstruction_v2.md) 与
> [`stage_conditioned_source_admission_gate_v1.md`](stage_conditioned_source_admission_gate_v1.md) 接管。

## 1. 本 gate 只回答一个问题

在 manipulation target 的 transient bootstrap 中，若冻结 source 只控制 shared embodiment-support
subspace（`legs_torso`），而 target student 始终控制 task-specific effectors（`arms,hands`），能否：

1. 保留 cabinet 的 early hard-progress acceleration；同时
2. 减少 basketball 中“posture 改善但投篮 success 下降”的 task interference？

这不是 source ranking、transferability prediction 或 mask search。唯一 treatment 是 source action
authority：full 61-D 对比 anatomy-fixed 11-D support mask。

## 2. 固定方法定义

所有方法使用同一个 FastTD3 learner、target reward、uniform replay sampling 与 update budget。正式条件：

| condition | warmup behavior | source authority | warmup 后 |
|---|---|---|---|
| scratch | student | none | student |
| full | schedule tape 指定 source/student | source: all 61 dims | student |
| EPS | 与 full 同一 schedule tape | source: dims 0–10；student: dims 11–60 | student |

共同固定项：

- `mcg_ablation=bootstrap_only`；不启用 post-warmup MCG gate/distillation；
- warmup 30k environment steps；segment horizon 25；teacher probability 0.5；
- cabinet bank：`configs/source_banks/h1hand_loco_wfix_cabinet.yaml`；
- basketball bank：`configs/source_banks/h1hand_std9_wfix_basketball.yaml`；
- source probability：对 bank 中冻结的 bootstrap weights 按现有 softmax 规则计算；
- EPS mask：`legs_torso`，不得按结果改为 legs-only、torso-only 或加入 arms；
- 训练到 100k，保存 10k/30k/100k；最终 policy 的 evaluation 不加载、不调用 source。

## 3. 为什么必须使用固定 schedule tape

现有 safe-bootstrap controller 在 segment 到期或 environment done 后重抽 arm。full 与 EPS 会改变
termination，因此即便 controller seed 相同，后续 source identity 与 teacher dose 也会分叉。这个分叉会把
“动作权限差异”和“数据来源差异”混在一起。

正式 gate 必须在训练前按 `(task, learner_seed, env_rank, global_segment)` 生成离散 tape：

- 每个 global segment 固定 25 environment steps；
- tape item 为 `student(-1)` 或一个 source id；
- episode reset 不推进、不重抽 tape；
- full 与 EPS 读取完全相同的 tape；
- tape 文件和 canonical serialization 的 SHA-256 写入 run metadata/checkpoint；
- scratch 不消费 source action，但保留相同 update/checkpoint cadence。

这样估计的是 authority mask 的 total effect，而不是 source-selection realization 的差异。EPS gate 中所有
bank horizon 均为 25；本协议不扩展 variable-horizon tape。

## 4. 必须补齐的 transition provenance

provenance 使用 canonical group 顺序 `(legs_torso, arms, hands)`，不能随 CLI group 数变化：

- `behavior_source`：student 为 `-1`；teacher segment 为 tape source id；
- `source_by_group`：
  - scratch/student segment：`[-1,-1,-1]`；
  - full teacher segment：`[src,src,src]`；
  - EPS teacher segment：`[src,-1,-1]`；
- `executed_group_mask` 与上项是否非负逐位一致；
- `segment_id = global_segment * num_envs + env_rank`；
- `segment_step = global_step % 25`；
- `learner_step/env_rank` 按实际 transition 写入；
- snapshot/export 前调用 replay completeness assertion。

额外 run-level 审计量：schedule hash、每个 source 的期望/实际 transition 数、student transition 数、
各组 source action L2 displacement、critic/actor update count、checkpoint hash。

## 5. 工程门（E0，科学 run 之前）

只做短程 smoke，不看科学指标：

1. unit test：手工 action tensor 验证 EPS 只替换 0–10 维；
2. schedule test：full/EPS 在模拟不同 done 序列下仍返回同一 tape item；
3. provenance test：scratch/full/EPS 三种 canonical 标签逐 transition 正确；
4. 200-step full/EPS smoke：schedule hash 与 source/student counts 精确相同；
5. duplicate EPS smoke：固定 seed 的 action/reward/provenance/checkpoint hash 精确一致；
6. seeded source-free evaluator test 通过。

任一失败均为 Engineering Stop，不得把 smoke return 当科学信号。

## 6. 单 seed 信号门（E1）

固定 learner seed 1。每个 task 运行 scratch/full/EPS，共 6 个 100k runs；E0 duplicate 不重复为
科学样本。所有条件使用同一训练配置，除 source authority/tape consumption 外不得变化。

正确 seeded source-free evaluation：4 eval seeds × 8 envs；训练 seed 是统计单位，episode 只用于该
seed 内估计。E1 只作 feasibility，不报告显著性。

主 checkpoint 与指标：

- cabinet @30k：`max(success_subtasks)`，支持指标 `max(door_openness_reward)`；
- basketball @100k：`max(success)`，支持指标 `max(reward_ball_success)`；
- 两任务均报告 return、stand/small-control、episode length/early failure，但不得替代主指标。

先检查 contrast 是否被本协议复现：

- cabinet：`P_full − P_scratch >= 0.10`；
- basketball：`P_full − P_scratch <= -0.10`。

任一不满足则 E1 为 **Inconclusive / contrast not reproduced**，不调 mask/source/horizon补救；先审计
实现与 seed 代表性，再由 PI 决定是否直接回到 base TBS 成稿。

在 contrast 成立时，EPS 同时满足才算 E1 Go：

- cabinet：`P_EPS − P_scratch >= 0.8*(P_full − P_scratch)`，或绝对增益 `>=0.10`；
- basketball：EPS 至少回收 full regret 的 50%，即
  `P_EPS − P_full >= 0.5*(P_scratch − P_full)`，且 `P_EPS >= P_scratch − 0.05`；
- cabinet secondary task metric 同向；basketball `reward_ball_success` 不恶化；
- tape/dose/update/provenance 全部通过，且 teacher segment 的 support-action displacement 非零。

## 7. 多 seed 裁决门（E2，仅 E1 Go 后）

补 learner seeds 2、3，仍只运行 scratch/full/EPS；若严格相同 scratch checkpoint/config 已存在，可在
核验 checkpoint metadata 与 evaluator 后复用，禁止挑 checkpoint。

最终支持 EPS contribution 需要：

- cabinet retention 与 basketball regret recovery 的训练-seed mean 均过 E1 阈值；
- 两个关键 contrast 至少 2/3 seeds 同向；
- 不用 episode 数膨胀显著性，不以 shaped return 翻盘；
- source-free policy 在 source 撤出后仍保留相应 task effect。

E2 失败即 **EPS Stop**。不做 mask grid、source search、horizon search 或 killer-task search。

## 8. 结果对应的论文裁决

| 结果 | 论文动作 |
|---|---|
| E0 失败 | 修工程，不产生科研结论 |
| E1 contrast 未复现 | EPS 证据不足；优先 base TBS 成稿 |
| E1 EPS 失败 | 删除 EPS contribution，modularity 留 future work |
| E1 Go / E2 失败 | 只报告 pilot，不写一般方法 claim |
| E2 Go | 将 Contribution 2 升级为 effect-preserving transient scaffolding |

无论结果如何，旧 DV/SIV estimator、WTA/stage-best、更多 T/source/horizon 调参都不恢复为论文核心。
