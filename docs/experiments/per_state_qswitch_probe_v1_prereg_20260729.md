# 预注册：per-state × per-body-group Q-switch 机制探针（door）

> 2026-07-29。**本文必须在任何 Δ 被计算之前提交。**
> 方向依据见 `docs/direction_per_state_qswitch_20260729.md`。
> 这是**离线探针**，零训练成本；它检验机制前提，**不能**推断迁移效果。

## 1. 要回答的唯一问题

**H2：在 target critic 判定"整条源策略替换有害"的状态上，是否仍存在大量身体组
使得局部替换被判为有益？** 即"整体有害"是否蕴含"处处有害"。

若不蕴含，则 per-state × per-body-group 的 Q-switch 存在可用信号，值得投入训练；
若蕴含，则 door 上没有可用的局部信号，**整线停止**。

## 2. 为什么选 door

door 是目前**唯一**同时满足三条的 target：

1. **full-action 的学习效用已知为负且测量干净**——Door@10k gate：三 loco 源
   9/9 per-seed 负，|U|/SE 中位 9（`door_at10k_gate_v1_results_20260727.md`）；
2. student checkpoint 与源 bank 都在，可零成本复用；
3. 它正是 MCG 当年判负的场地之一（313 vs 328），构成直接对照。

**这不是在挑对我们有利的场地**：door 对 full-action 是已知的**最不利**场地。
探针问的是"在最不利的场地上，局部信号是否还存在"。

## 3. 设定（全部冻结）

```
student ckpt   models/h1hand-door-v0__door_at10k_student_s{1,2,3}__{1,2,3}_20000.pt
源              configs/source_banks/calibration/h1hand_door_rbo_{stand,walk,run}.yaml
                （逐个取 sources[0]，与 Door@10k gate 逐位同源）
身体组          mcg.DEFAULT_GROUPS = (legs_torso[11], arms[10], hands[40])
状态采集        student actor + 噪声 0.1，16 env × 300 步 = 4800 状态/seed
                （与 2026-06-11 push 探针的 4800 状态口径一致）
seed            3 个 learner seed 各自独立计算，不合并
```

Δ 的定义沿用 `mcg.ModularGating.deltas()` 的 paired head delta（v1.1 口径）：

```
Δ_full,i(s)  = min_h [ Q_h(s, π_i(s))    − Q_h(s, π_stu(s)) ]
Δ_{i,g}(s)   = min_h [ Q_h(s, ã_{i,g}(s)) − Q_h(s, π_stu(s)) ]
ã_{i,g}(s)   = 学生动作，第 g 组维度替换为源 i 的对应维度
```

null margin 沿用 `mcg.null_margins()`：把 batch 内**其他状态**的源动作拼给当前状态，
得到"与当前状态无关的建议"的 Δ 分布，取高分位作为噪声右尾。
full 与每个 group **分别**估计各自的 null 分布。

## 4. 多重比较校正（必须，否则比较不公平）

`max` 的取值范围两侧不同：

- group 侧：`max_{i,g}` 覆盖 3 源 × 3 组 = **9** 个比较
- full 侧：`max_i` 覆盖 3 源 = **3** 个比较

不校正就直接比 frac，group 仅凭比较次数多就会占优。用 Šidák 把两侧的
**family-wise 名义假阳性率**对齐到

```
α_fw = 1 − 0.95³ ≈ 0.1426
```

- full 侧：每比较用 null 分位 **q95**（α = 0.05，3 个比较 → α_fw = 0.1426）
- group 侧：每比较用 null 分位 **q98.30**
  （解 1 − (1−α)⁹ = 0.1426 ⟹ α = 1 − 0.8574^{1/9} = 0.0170）

于是

```
frac_sig_group = P_s[ max_{i,g} ( Δ_{i,g}(s) − m_g^{q98.30} ) > 0 ]
frac_sig_full  = P_s[ max_i    ( Δ_full,i(s) − m_full^{q95}  ) > 0 ]
```

两者在同一名义假阳性率下可比。

## 5. 判据（冻结）

**主判据**

1. `frac_sig_group ≥ 0.30`（≈ 2 × α_fw；低于此则机制几乎无发挥空间）
2. `frac_sig_group > frac_sig_full` 在 **3/3** seed 上成立

**裁决**

| 条件 | 裁决 | 后果 |
|---|---|---|
| 主判据 1 与 2 同时满足 | `PROBE_SUPPORTED` | 进 Step 1（通道解耦），随后另行预注册训练对照 |
| `frac_sig_group < 0.1426` 或 方向一致性 ≤ 1/3 | `PROBE_REFUTED` | **整线停止**，写入负结果，不做变体 |
| 其余 | `PROBE_WEAK` | **不投训练**，记录后停 |

**次级观察（报告但不作判据）**

- `Δ_full` 的均值符号预期为**负**，与 Door gate 的 9/9 学习效用负同向。
  **若为正**，说明 critic 的 per-state 判断与最终学习效用方向矛盾，
  探针本身可信度存疑，必须在结果中显著标注。
- 未做多重比较校正的 sign 版 `frac+ = P[Δ>0]`，仅用于与 2026-06-11
  push 探针（reach-arms 0.49 vs reach-full 0.05）保持可比，不参与裁决。
- 每个身体组各自的 `frac_sig`，用于观察是否存在组间分化。

## 6. 明令禁止

跑完后**不得**调整以下任一项再重跑：null 分位数、判据阈值、身体组定义、
状态采集步数/噪声、student checkpoint 步数。

若结果落在 `PROBE_WEAK`，**不得**通过换 target、换源、换 checkpoint 步数来"再试一次"——
`PROBE_WEAK` 的后果就是停。

## 7. 这个探针不能说明什么

- 不能说明 per-group Q-switch 会带来正迁移（Δ 是 critic 的判断，不是学习效用；
  本项目已多次证明二者可以反向）；
- 不能说明 critic 的判断是对的；
- 只能说明"局部信号在统计上是否存在"。若不存在，后续训练无从谈起。
