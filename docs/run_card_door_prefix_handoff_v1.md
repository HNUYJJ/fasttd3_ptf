# Run card：Door fixed-horizon option handoff feasibility ablation

> 批准：PI 2026-07-28。**预注册**——本文件与裁决脚本在任何 prefix 臂被评估之前提交。  
> 定位：**PTF-derived fixed-horizon option handoff feasibility ablation**。
> 不是 curriculum（第一版固定 H、无退火），不是 faithful PTF（无 Q_ω、无 learned β），
> 也不是迁移性指标实验。

## 1. 唯一要回答的问题

在 **source 身份、训练阶段、总行为剂量、replay eligibility 全部相同**的条件下，
把 source 的介入方式从「随机 25 步碎片」换成「完整 episode 前缀 handoff」，
能否缓解 Door 上已确认的负迁移？

## 2. 四条件框架（前三组复用既有结果，只新跑第四组）

| 条件 | 行为方式 | source replay | 状态 |
|---|---|---|---|
| Student | 无 source | 无 | 已有（door_at10k_gate_v1） |
| Joint RBO | 随机 25 步 segment | **有** | 已有（door_at10k_gate_v1） |
| Segment B-only | 随机 25 步 segment | 无 | 已有（door_channel_decomposition_v1） |
| **Prefix B-only** | **episode 前缀连续 H 步** | 无 | **本轮新跑，3 seeds** |

## 3. 主/次比较

```
主比较（唯一决定裁决）：
    Δ_placement = J(prefix B-only) − J(segment B-only)      同 seed 配对

次级（单独报告，不参与裁决）：
    prefix vs student      判定是正迁移 / 仅缓解 / 仍负迁移
    prefix vs joint RBO    与完整 RBO 的关系
    跨 seed 方差           描述性；3 seed 不足以把"方差下降"升级为机制贡献
```

`J` = 20k 冻结 source-free evaluator，128 deterministic episodes
（16 eval seeds × 8 ranks，前 32 与历史面板逐位兼容）。

## 4. 协议（除 prefix 外与 Segment B-only 逐项相同）

- 复用 door 的三个 10k 纯 student anchor 与同一 `resume_noise_seed`（91000+seed）
- bank：`calibration/h1hand_door_rbo_run.yaml`（单源 run）
- `PTF_ADMISSION_REPLAY_MODE=student_only`（source 数据不进 critic）
- `PTF_MCG_EPISODE_PREFIX_STEPS=H`（本轮新增机制）
- 其余：`bootstrap_only`、`admission_bootstrap`、`admission_mode=all`、
  `student_logit=0.0`、`expected_source_mass=0.5`、warmup 30000、训练到 20k
- **不加** curriculum 退火、G_i、多源 bandit、learned termination

## 5. 允许的一次盲化剂量校准

door 在 10k–20k 的 eval episode 长度实测 **873–997 步**，故名义 50% 前缀取 **H=500**。

1. 用 H=500 跑 smoke，**只查看 behavior share 与数据流，不查看任何性能数字**；
2. 若 behavior share ∉ [0.45, 0.55]，**仅依据占比**调整一次 H，随后冻结；
3. smoke 必须覆盖 prefix→student handoff 与至少一次 episode reset。

> 这是**协议校准**（对齐剂量这一协议参数），不是结果导向搜索。分界线是
> **是否已经看到 J**——校准阶段严禁读取任何性能量。

## 6. 工程验收（只查这五项）

1. 10k anchor 正确恢复（日志 `Resumed core learner ... at step 10000`）
2. episode 前 H 步由 source 执行，此后**严格** student（handoff 生效）
3. `critic source sample count = 0`
4. 实际 source behavior share 落在 [0.45, 0.55]
5. 其余训练参数与 Segment B-only 相同（batch/num_updates/步数/anchor/noise seed）

## 7. 预注册裁决

```
PREFIX_SUPERIOR       LCB_90(Δ_placement) > 0
                      → 强证据，可进论文

PREFIX_PROMISING      3/3 seed 的 Δ_placement > 0 且 mean(Δ_placement) ≥ +30
                      → 仅报告并等待 PI 决定；不自动扩展，
                        不作为论文确认性结论

PREFIX_NOT_SUPPORTED  其余情况
                      → 停止该路线；不调 H、不加 bandit、不复活 termination
```

区间为 3 个 **learner seed** 的配对 t 区间（df=2，与 Door 系列一致）。
episode-level SE 只作评价可靠性诊断，**不得**代替 learner-seed 不确定性（教训 M16）。

prefix 相对 Student 的结果**必须单独报告**：显著高于 = 正迁移；区间跨零 = 仅缓解/不确定；
显著低于 = 仍是负迁移。

## 8. 统计力的事前声明

用 Door 分解的实测方差估计：`sd(Δ_placement) ≈ 60–80`，故
`LCB_90 > 0` 需要 `mean(Δ) ≳ +101 … +135`，即 prefix 需达到 315–349
（segment B-only 均值 213.5，student 267.5，door 95k 上限约 332）。

**因此 PREFIX_SUPERIOR 的门槛很高，事前即知**。这是设置 PREFIX_PROMISING 这一中间档的
原因，而非事后放宽。三档的后果均已写死，不因结果调整。

## 9. 成本

3 臂 × 10k 步 ≈ 40 分钟训练 + 3 × 128-episode 评估。复用现成 anchor 与面板。
