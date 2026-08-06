# Critic-First Bridge Bootstrap-and-Forget v1 — 最小可证伪 run card

> 状态：CLOSED — single-seed feasibility `FAIL`。结果见
> `docs/experiments/critic_first_bridge_v1_results_20260729.md`。
> 日期：2026-07-29。
> 目标：把已证有效的 reward-bearing bootstrap 从“同步经验注入”推进为
> “短暂 target-grounded bridge → critic-first grounding → student takeover
> → source forget”的两阶段迁移机制。

## Material Passport

- artifact_type: experiment_plan
- evidence_status: preregistered
- target_algorithm: FastTD3 + PTF reward-bearing bootstrap
- final_policy: source-free student
- primary_tasks: `h1hand-slide-v0`（正例）、`h1hand-door-v0`（负例）
- formal_multiseed_authorized: false

## 1. 强制立项门

1. **核心问题**：行为/数据双通道迁移与困难任务真实收益。
2. **唯一主要假设**：在完全相同的短暂 source 行为、replay 暴露和硬退出下，
   先只更新 target critic、延迟 target actor 更新，比同步 actor–critic bootstrap
   更能保留正迁移并减少负迁移。
3. **决策影响**：正负 gate 同时支持，则形成 critic-first cross-task warm-start
   主方法并进入 3 seeds；否则关闭 critic-first 分相，不改窗口或任务抢救。
4. **非重复性**：既有 RBO、Door 通道分解和 QMP 均在 source 数据到达时同步
   更新 actor；没有实验隔离“source bridge 期间只训练 critic”。
5. **最小方案**：一个已知正迁移 source–target 与一个已知负迁移
   source–target，各做 3 个配对分支，先单 seed feasibility。

## 2. 与既有失败线的边界

- 不估计 source utility，不恢复 `T^0/T^critic/SIV/SHU/P0/QMP`。
- 不学习 selector 或 termination；source 身份与 2k 暴露预算事前固定。
- 不改 replay 权重、UTD、TD3 loss、MCG 蒸馏或动作兼容性。
- exact abstention、provenance 和 replay hard exit 只负责实现“forget”，不单独
  声称性能贡献。

## 3. 被测机制

共同起点为同 seed 的 10k pure-student anchor。令 bridge 区间为
`[10000, 12000)`：

1. 单个 frozen source 在 target 环境执行；source 与 student 的行为概率均为
   0.5，critic replay 配额均为 0.5，segment horizon=25。
2. `interleaved`：FastTD3 actor 与 critic 均照常更新。
3. `critic_first`：critic、target critic、normalizer、replay 与 scheduler 照常；
   只禁止 `update_pol`，target actor 参数不接收梯度。
4. completed step 12000 时 schedule exact-abstain：source 行为增量和 source
   critic-sample 增量此后必须严格为 0；历史 source transition 留存审计记录，
   但立即退出 active replay。
5. `[12000, 20000)` 为 100% student takeover，actor 恢复正常更新；20k 用
   128-episode frozen source-free evaluator。

选择 2k 是协议预算，不是待调超参：128 envs 下产生 256k target transitions，
名义约 128k source transitions，同时提供 4000 次 critic update，并保留 8k
student-only takeover 步。结果出来后不得调整该窗口救结论。

## 4. 三臂配对设计

| arm | bridge 数据 | 10k–12k actor | 作用 |
|---|---|---:|---|
| `student_freeze` | 100% student | 冻结 | 控制“延迟 actor”本身 |
| `interleaved` | 50% source + 50% student | 更新 | 当前同步 RBO 对照 |
| `critic_first` | 与 interleaved 相同 | 冻结 | 被测机制 |

除该列外，同一 task/seed 的 anchor、resume-noise seed、LR 时间、source bank、
admission schedule、batch、UTD、checkpoint 和 evaluator 面板均相同。

## 5. 场地

- **Slide–walk**：历史 `U(10k,10k)=+56.95`，3/3 seed 正；承担保留正迁移。
- **Door–run**：历史 `U(10k,10k)=-30.63`，3/3 seed 负；承担减害/免疫。

任务与 source 由既有冻结证据选定，不根据本实验结果更换。

## 6. 工程 gate

任一失败均为 `ENGINEERING_INVALID`：

1. 三臂从对应 10k anchor 恢复，completed step 身份正确。
2. bridge source behavior/critic share 均在 `[0.45, 0.55]`。
3. `critic_first` 与 `student_freeze` 在 12k 的 actor update count 等于 anchor；
   `interleaved` 必须增加。三臂 critic update count 均增加。
4. 12k decision 后至 20k，source execution count 与 source critic sample count
   增量严格为 0。
5. 20k evaluator 为 deterministic、source-free、128 episodes、固定 seed 面板。

## 7. 单 seed feasibility 裁决

记 `J_CF`、`J_INT`、`J_SF` 分别为 critic-first、interleaved 和
student-freeze 的 20k source-free 回报。

- `DUAL_GATE_PASS`：Slide `J_CF > J_INT` 且 `J_CF > J_SF`；Door
  `J_CF > J_INT` 且 `J_CF >= J_SF`。
- `POSITIVE_ONLY`：Slide 双条件成立；Door 仅 `J_CF > J_INT`，但仍低于
  `J_SF`。只允许报告“减害趋势”，不得进入正式矩阵，交 PI 决定。
- `FAIL`：Slide 未同时超过两对照，或 Door 不优于 interleaved。机制关闭；
  不改 bridge 长度、source、任务、replay 配额或阈值抢救。

单 seed 只作 feasibility，不进入论文效应主张。`DUAL_GATE_PASS` 后另行冻结
3-seed 统计方案。

## 8. 本轮实施范围

允许：一个默认关闭的 actor-update start step、聚焦单测、真实 smoke、上述
单 seed 六臂及冻结评估。

禁止：正式多 seed、MCG、curriculum/reset、自动 source metric、超参搜索。
