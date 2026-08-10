# 预注册：N1 辨别力探针（arms 动作对 truck 早期 reward 是否有通路）

> 冻结时间：2026-08-10，**先于运行任何探针**。
> 性质：DIAGNOSTIC 诊断探针，**不改动 N1 的任何 verdict**，
> 不是新机制、不新增训练。目的只有一个——检验 N1 的 `H_A` 是否具备辨别力。

## 1. 被检验的命题

N1 的 `H_A`（`J_LP − J_FP > 0`，即"不让 locomotion source 接管 arms 能改善学习"）
被判 `DIRECTIONAL_REFUTATION`（mean = −1.9，2/5 正）。

对这个否证存在一个**平凡解释**（CLAUDE.md §8.1）：

> 在 truck 的 0→20k 阶段，reward 对 arms 的动作根本没有直接通路，
> 因此"保护 arms"与否本就不该产生 return 差异。零效应是设计的预期产物，
> 而非对"arm authority displacement"假设的检验。

源码依据（`humanoid_bench/envs/truck.py:87-190`）：在没有任何 package 被拾起时
（T3 已实测 0–20k 期间 `|r|>3` 事件数为 0），每步 reward 恰好是

```
reward = upright × (1 + reward_robot_package_truck)
upright                    = tolerance(torso_upright, …)         # 仅躯干姿态
reward_robot_package_truck = tolerance(‖xpos[pkg] − qpos["free_base"][:3]‖, …)  # 仅根部位置
```

两项都不含 arm/hand 的位置或姿态；`arms` = action dim 11–21
（`action_schema.py:54-58`），只能经动力学间接影响平衡。

## 2. 探针设计

在**已训练好的 N1 checkpoint** 上做纯前向 rollout，不训练、不更新任何参数。
对同一 checkpoint、同一 eval seed，按动作子空间做扰动对照：

| 条件 | 处理 |
|---|---|
| `intact` | 原始动作 |
| `rand_arms` | dim 11–21 替换为 U(−1,1) |
| `rand_legs_torso` | dim 0–11 替换为 U(−1,1) |
| `zero_arms` | dim 11–21 置 0 |

`rand_legs_torso` 是**方法有效性对照**：若它不导致大幅下降，说明扰动本身没生效，
整个探针作废（判 `PROBE_INVALID`）。

## 3. 冻结判据

令 `Δ_x = mean_return(x) − mean_return(intact)`，相对幅度 `δ_x = |Δ_x| / mean_return(intact)`。

1. **有效性前置**：`δ_rand_legs_torso ≥ 0.20`。不满足 → `PROBE_INVALID`，
   不得解读其余条件。
2. **主判据**：`δ_rand_arms < 0.05` 且 `δ_zero_arms < 0.05`
   → `ARMS_PATHWAY_NEGLIGIBLE`：确认平凡解释成立，`H_A` 在本场地缺乏辨别力。
3. `δ_rand_arms ≥ 0.15` → `ARMS_PATHWAY_SUBSTANTIAL`：平凡解释被排除，
   `H_A` 的否证保持其原有效力。
4. 其余 → `ARMS_PATHWAY_PARTIAL`：不足以支持任一端。

阈值 0.05 的依据：`H_A` 的 learner 间 sd = 18.1，约为均值 return（≈943）的 1.9%；
取 5% 已是该噪声尺度的 2.6 倍。阈值在看到任何探针输出之前写定。

## 4. 矩阵与边界

- checkpoint：N1 的 `s`（scratch）与 `lp` 两臂，seeds 4–8，step 20000。
  用两臂是因为若只用 scratch，可能被质疑"scratch 本就没学会用 arms"。
- 每 checkpoint × 每条件 16 episodes，1000 步，deterministic actor，
  eval seed 与 `p0_evaluator.py` 的 panel 前 16 个一致。
- **边界**：本探针只说明 arms 动作在 **truck 0→20k、当前 checkpoint** 上
  对 return 的因果通路强弱，**不**说明 arms 在完整 truck 任务（含搬箱阶段）
  上不重要，**不**改变 N1 已冻结的任何 verdict。
