# Phase 0 预注册：接口 positive control（Push）

> 冻结日期：2026-08-12，**先于任何正式训练**。
> 性质：**工程与方法 positive control，不是创新实验**。
> 它只回答"当前 FastTD3 栈能否利用一个已知可靠的连续控制接口"，
> 结果不构成任何迁移学习主张。

## 1. 已完成的前置验证（先于本预注册）

| 项 | 结果 |
|---|---|
| 接口 controllability | **RESPONSIVE**：+x setpoint 手位移 `+0.9472`，−x 为 `−0.3540`，符号相反、差 `1.3012`（阈值 0.02） |
| 并行环境契约 | **CONTRACT_OK**：flat `num_actions=61` / interface `=3`；`num_obs=163`、`max_steps=500`、各张量形状两臂完全一致 |
| 训练链路 | 两臂各 300 步 smoke 跑通并保存 checkpoint |

## 2. 任务与为什么选 Push

`h1hand-push-v0`，`max_episode_steps = 500`（**不是 1000**，见
`humanoid_bench_env.py:25-34` 的短 episode 表）。

reward（`envs/push.py:76-95`，逐字读出）：

```
reward = −0.1 · hand_dist − 1.0 · target_dist + 1000·[target_dist < 0.05]
terminated = (target_dist < 0.05)        # 即 terminated 语义是"成功"
success_bar = 700
```

选 Push 的理由：早期回报由 `hand_dist`（左手到 box 的距离）主导，
而**这正是 reaching interface 覆盖的能力**。官方 HumanoidBench 也正是在
Push 上报告了 hierarchy 优于 flat，在 Package 上因低层不会 lifting 而收益有限。
所以 Push 是这条路线的**正对照**——若这里都不成立，路线应立即关闭。

## 3. 两臂（唯一差别是动作接口）

| 臂 | `INTERFACE_POLICY_TYPE` | 动作空间 |
|---|---|---|
| `flat` | 未设 | 61 维关节目标 |
| `iface` | `reach_single` | 3 维末端 setpoint 增量（低层为官方冻结 controller） |

其余全部相同：同 seed 集合 `{1,2,3}`、`num_envs=128`、`batch_size=32768`、
`buffer_size=51200`、`total_timesteps=100000`、`num_updates` 与 LR 日程用
vendor 默认、`compile` 关闭（与主线口径一致）、wandb 开启。

**不设 residual、不训练任何低层技能、不做 Lift、不做多技能组合。**

## 4. 冻结判据

主指标：训练结束时（100k）的 eval return，逐 seed 配对差 `J_iface − J_flat`。
`n=3` 只报方向与 learner 间 SD，**不做显著性声称**（M16/M24）。

- **`INTERFACE_VIABLE`**：3/3 seed 为正，且均值差 > 0。
- **`INTERFACE_NOT_VIABLE`**：≤1/3 seed 为正，或均值差 ≤ 0。
- **`INTERFACE_UNRESOLVED`**：2/3 为正。

诊断指标（**描述性，不进裁决**）：`hand_dist` 的 episode 均值（approach 能力）、
`target_dist`（box progress）、`success` 触发率、达到 `success_bar=700` 的步数。

### 停止规则（先于结果写定）

- 判为 `INTERFACE_NOT_VIABLE` → **立即关闭整条 interface 路线**。
  不训练 locomotion skill、不加 residual、不做 Lift、不换任务抢救、不加 seed。
- 判为 `INTERFACE_VIABLE` → 才允许进入 Phase 1（source-free consolidation gate）。
- 判为 `INTERFACE_UNRESOLVED` → 补 seeds 4–5 一次，仍不决则按 NOT_VIABLE 处理。

## 5. 这个实验不能说明什么

1. **不是迁移结果。** 两臂的差异是"控制接口"，不涉及任何 source→target 知识迁移。
   `INTERFACE_VIABLE` 只意味着 FastTD3 能用这个接口，**不意味着接口是好的迁移载体**。
2. **不构成 novelty。** 官方 HumanoidBench 已有此 hierarchy 与其 Push 结论；
   本轮是在我们的栈里复现，属基础设施验证。
3. **persistent HRL ≠ source-free。** 本阶段 interface 全程在位，
   最终策略依赖冻结低层模块，与主线"最终 student 必须 source-free"的标准不同。
   这个区分留待 Phase 1 用 consolidation 臂处理。
4. 不推翻主线任何 verdict；特别是 truck 的 **20k 负 / 95k 正（+229.9, t=3.47）**
   意味着"固定 policy 载体不行"尚未成立，本路线是并行探索而非替代。

## 6. 资源与边界

3 seeds × 2 臂 × 100k 步。按 smoke 实测约 30–38 sps（4 env），
正式 128 env 的吞吐需实测后估算。仅 Push 单任务，不扩展。
