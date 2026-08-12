# Phase 0 预注册 **v2**（取代 v1，v1 作废但保留）

> 冻结时间：2026-08-12，**仍先于任何正式训练**（v1 冻结后未跑过任何一步正式训练，
> 未产生任何结果）。按 CLAUDE.md §4.1，需要调整时**新写一份，不改旧的**。

## 0. 为什么需要 v2

v1 把 `total_timesteps` 写成 100k。启动前复核 vendor 超参时发现：

```python
class H1HandPushArgs(HumanoidBenchArgs):
    v_min, v_max = -1000.0, 1000.0
    total_timesteps = 1000000        # 官方 push 默认是 100 万步
```

v1 的 100k 只有官方默认的 **1/10**。若两臂在 100k 时都还远未学会，
v1 的判据会给出 `INTERFACE_NOT_VIABLE`，并按停止规则**关闭整条 interface 路线**——
那将是一次**假阴性关闭**（"预算不足"被读成"接口无效"），
正是 CLAUDE.md §4 首条所禁止的"没跑完被读成方向被否定"。

**本次修正让实验更难通过、不更易通过**，且在零结果状态下做出。

## 1. 相对 v1 的三处改动

1. **预算与其理由显式化**：仍取 `total_timesteps = 100000`（受本机共享资源限制，
   6 条臂 × 1M 步不可行），但由此新增第 3 条判据分支。
2. **新增 `INCONCLUSIVE_BUDGET` 分支**（见 §3），使"预算不足"无法被误读为"方向否定"。
3. **`eval_interval = 5000`**（v1 未指定），强制留下学习曲线；
   曲线是第 2 条分支唯一可依据的证据，不留就无法区分两种失败。

其余（任务、两臂定义、seeds、主指标、停止规则的其余部分）与 v1 完全一致。

## 2. 冻结矩阵

`h1hand-push-v0`，`max_episode_steps = 500`。
reward `= −0.1·hand_dist − 1.0·target_dist + 1000·[target_dist<0.05]`，
`terminated` 语义为成功，`success_bar = 700`（逐字读自 `envs/push.py:76-95`）。

| 臂 | `INTERFACE_POLICY_TYPE` | 动作空间 |
|---|---|---|
| `p0_flat_s{1,2,3}` | 未设 | 61 维关节目标 |
| `p0_iface_s{1,2,3}` | `reach_single` | 3 维末端 setpoint 增量 |

其余逐项相同且**全部采用 `H1HandPushArgs` 官方默认**（含 `v_min/v_max=±1000`）：
`num_envs=128`、`batch_size=32768`、`buffer_size=51200`、`num_updates` 与 LR 日程默认、
`compile` 关闭、`eval_interval=5000`、`save_interval=0`、`render_interval=0`、wandb 开启。
唯一被 CLI 覆盖的是 `total_timesteps=100000` 与 `seed`/`exp_name`。

**不设 residual、不训练任何低层技能、不做 Lift、不做多技能组合。**

## 3. 判据（主指标：100k 处 eval return 的逐 seed 配对差 `J_iface − J_flat`）

`n=3`，只报方向与 learner 间 SD，**不做显著性声称**（M16/M24）。

1. **`INTERFACE_VIABLE`**：3/3 seed 为正且均值差 > 0 → 允许进入 Phase 1。
2. **`INTERFACE_UNRESOLVED`**：2/3 为正 → 补 seeds 4–5 一次；仍不决按第 4 条处理。
3. **`INCONCLUSIVE_BUDGET`**：判据 1、2 均不满足，**但**两臂在最后 20%
   训练步（80k→100k）的 eval return 均单调不减 → **不关闭路线**，
   如实记为"预算不足以判别"，是否加预算交 PI。
4. **`INTERFACE_NOT_VIABLE`**：判据 1、2 不满足，**且**至少一臂的曲线在
   最后 20% 已平台或下降 → **立即关闭整条 interface 路线**：
   不训练 locomotion skill、不加 residual、不做 Lift、不换任务抢救、不加 seed。

诊断指标（**描述性，不进裁决**）：`hand_dist` episode 均值、`target_dist`、
`success` 触发率、首次达到 `success_bar=700` 的步数。

## 4. 这个实验不能说明什么（与 v1 相同，重申）

1. **不是迁移结果**——两臂差异是控制接口，不涉及任何 source→target 知识迁移。
2. **不构成 novelty**——官方 HumanoidBench 已有此 hierarchy 及其 Push 结论。
3. **persistent HRL ≠ source-free**——本阶段接口全程在位，最终策略依赖冻结低层模块。
4. **不推翻主线任何 verdict**——truck 的 20k 负 / 95k 正（+229.9, t=3.47）
   意味着"固定 policy 载体不行"尚未成立。

## 5. 已完成的前置验证

接口 controllability `RESPONSIVE`（+x 手位移 `+0.9472` / −x `−0.3540`，差 `1.3012`）；
并行契约 `CONTRACT_OK`；两臂各 300 步 smoke 跑通。
