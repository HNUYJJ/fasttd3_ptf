# Stage-Conditioned Replay Data-Value Probe v1

日期：2026-07-11。状态：实现前协议；不启动正式训练。

> **2026-07-11 superseded**：本文件保留为早期 replay-only 思路记录。后续代码与实验以
> `docs/source_intervention_mechanism_gate_v1.md` 为准。新协议先区分 source execution
> 改变后续 reachability/occupancy 的通道、source transition 的 replay/update 通道及二者交互；
> 不再预设全部收益都可归因于 replay data value。

## 1. 研究问题

P2 已表明即时 shaped return、stability composite 与后续 cabinet hard progress 可以错位，
而 run24/run50 的训练种子方向也会反转。下一步不再问“哪个源当前 return 最高”，而问：

> 在同一个目标 learner 阶段，等量加入某个行为来源的数据并执行等量更新后，哪类数据
> 真正提高了独立评估集上的目标能力？

定义来源 `s` 在 learner stage `t` 的数据价值：

`DV_s(t) = [P(theta_{t,s}^{K}) - P(theta_{t,base}^{K})] / N`

其中 `N` 是注入 transition 数，`K` 是完全相同的梯度更新数，`P` 是独立 eval bank 上
的目标任务进展。`base` 分支只从共同基础数据更新，`s` 分支从同一基础数据加等量来源
数据更新。return 只作诊断，不进入 `DV` 定义。

## 2. 当前 checkout 的限制

现有 `save_ptf_params()` 只保存网络、normalizer、option 模块和 `global_step`，不保存：

- replay buffer 与真实 behavior label；
- actor/Q optimizer、scheduler、AMP scaler 和 RNG；
- MCG controller 状态；
- vector env/MuJoCo/task mutable state。

因此已有 10k checkpoint 不能作为忠实的“暂停后继续”分叉点。普通 WFix/run 实验还因
`mcg_track_options=False` 把 replay option 写成 `-1`，无法事后恢复来源。v1 pilot 必须把
这些限制显式纳入估计口径，不能声称逐状态 counterfactual。

## 3. 最小 pilot

### 固定项

- task：cabinet；learner stage：scratch 10k；先做一个 learner seed 的工程可行性筛选；
- 行为臂：`run`、`stand`、`student`；源仍只覆盖 `legs_torso,arms`，hands 由学生控制；
- segment horizon：25；每臂注入完全相同的 transition 数 `N`；
- 每分支使用相同初始化网络、normalizer、fresh optimizer 配置、随机种子和更新数 `K`；
- eval 使用与训练/收集分离的固定 seeds；禁止用 eval 数据训练或选 checkpoint。

### 共同起始状态

v1 不修改 HumanoidBench worker 去序列化完整 simulator state。对每个配对 env seed：

1. 各行为臂分别创建相同 seed 的环境；
2. 用同一 frozen 10k student actor 执行相同 deterministic prefix；
3. 验证 prefix 末观测在容差内一致；不一致或提前终止的 seed 整组丢弃；
4. 从该共同起点分别执行 run/stand/student 的 25 步锁定片段。

这样得到可复现的配对起点，但只代表“10k learner 诱导的固定 seed 状态分布”，不等同于
真实训练中任意 in-flight state。若该假设验证失败，才实现完整 `get_sim_state/set_sim_state`
（qpos/qvel、elapsed steps、task mutable state 与 RNG）。

### 共同基础 replay

由于旧 checkpoint 没有 replay，先用 frozen 10k student actor 和独立 seed bank 收集共同
surrogate buffer `D0`。所有分支使用逐字节相同的 `D0`：

- `base-only`：只从 `D0` 执行 `K` 次更新；
- `base+run`：`D0` + `N` 条 run transitions；
- `base+stand`：`D0` + `N` 条 stand transitions；
- `base+student`：`D0` + `N` 条额外 student transitions；
- `no-update`：不更新，只用于检查评估器漂移。

每个数据集位置使用同步的预生成 sampling-index schedule，使分支间只改变 transition 内容，
不改变更新次数或抽样噪声。fresh optimizer 使比较在分支间公平，但结论必须写成 surrogate
offline data value，而不是原训练 optimizer 状态下的精确继续训练效应。

## 4. 数据与指标

每条 transition 必须保存 `behavior_id`、obs/action/reward/next_obs、done/truncation、segment
与 env id。每段另存：

- planned/realized length、termination/fall reason；
- root height、tilt/upright；
- hand-handle distance、handle contact；
- door angle/openness、subtask progress；
- segment return（仅诊断）。

pilot 的连续主指标为 eval `max(door_openness_reward)`，避免 hard metric 在 10k 的 floor；
确认指标为 `max(success_subtasks)`。同时报告 episode length/early failure，保持 stability
deconfounding。真正统计单位是 learner seed；segment/env 数不能伪装成训练重复。

## 5. v1 验证和停止规则

工程 pilot 先回答四件事：

1. 相同 reset seed + deterministic prefix 是否产生可验证的一致起点；
2. 三臂能否收集相同 `N` 且保留正确 behavior label；
3. 相同权重、optimizer 和 sampling schedule 的重复 base 分支是否数值一致；
4. `DV_run(10k)` 是否在 door openness 上稳定高于 `DV_stand(10k)` 与额外 student 数据。

一个 learner seed 只能验证管线，不形成科研结论。只有 pilot 通过数值复现与方向筛选后，
才预注册 learner seeds 1/2/3、多个 injection-data seeds，并补 20k stage。若来源排序在不同
stage 可复现地变化，才有证据讨论“阶段最优注入”；若不变化，则保留静态预算分配叙事。

若 continuous progress 无可检测变化、重复 base 分支不一致，或结果主要由 termination 数量
决定，则停止扩大实验，先修复 estimator/instrumentation，不启动在线 WTA。

## 6. 最小代码工作包

1. 给 replay 增加明确的 `behavior_id` 与可选 CPU snapshot/export；
2. 将 actor/Q update kernel 从 `train_ptf.py` 主函数中抽成可复用、可固定 sampling schedule
   的离线更新单元；
3. 新增 segment collector 与 `scripts/probe_replay_data_value.py`；
4. 新增 synthetic replay、branch determinism、equal-dose、prefix-state parity 测试；
5. pilot 只生成数据和报告，不修改主训练默认行为。

该探针如果成立，会把当前“静态 return 权重”推进为可验证的 **target learning-value map**：
来源、目标任务、learner stage 和作用效果四者联合决定数据价值。这比按片段即时 return
动态选源更有机制依据，也更接近可独立成文的方法贡献。
