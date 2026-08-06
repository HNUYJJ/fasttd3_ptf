# Run card：最小 QMP-fidelity behavior-only 验证（v1，修订版）

> 2026-07-29 起草，同日按外部审查 `CONDITIONAL_APPROVE` 修订。
> 缘起：per-state Q-switch 探针被判 `INCONCLUSIVE`
> （`docs/experiments/per_state_qswitch_probe_v1_results_20260729.md` §0）。
>
> **本轮目的：把 QMP 的官方核心思想与本项目的 MCG 扩展彻底分开，先验证前者。**
> 本轮**不引入身体组**、不引入蒸馏、不引入 null margin。
>
> 修订项（v1 → v1 修订版）：①删除"CI 上界 > 0 = 非劣效"的错误判据；
> ②显式 `qmp_enabled` 并彻底绕开 classic PTF；③修正探索噪声时序；
> ④删除结果触发的未定义随机剂量臂；⑤补机制诊断量；⑥Door 对照复用前置等价性检查。

## 1. 要回答的问题（estimand）

> 在 source 即将介入的**决策时点（10k）**，用 target 自己的 critic 做
> **per-state 完整策略选择**、且**只改变行为采集**，能否
> (A) 在源确实有益的任务上**保留**正迁移，
> (B) 在源确实有害的任务上**减少**相对固定 source 的伤害？

终点是**在线学习效用**（source-free student 在 20k 的回报），
**不用离线 Q 统计代替**——本项目已多次证明 critic 判断与学习效用可以反向。

## 2. 与上一轮探针的区别（四条阻塞缺陷逐条修复）

| 上一轮的缺陷 | 本轮的处理 |
|---|---|
| 用 20k student，estimand 不匹配 | 从**真实 10k anchor** resume（learner + replay + rng） |
| null margin 为负、未按训练端 clamp | **完全不用 null margin**——直接 argmax Q，无阈值 |
| Šidák 独立性不成立 | **不做多重比较检验**——argmax 不需要显著性判据 |
| MCG 拼接动作 ≠ QMP 完整策略 | 候选**仅完整策略**，动作是某策略的实际输出 |

## 3. 机制定义（精确，冻结）

候选集合（**含 student 本身**）：

```
C = { π_student, π_stand, π_walk, π_run }
```

打分函数——与 FastTD3 actor 的优化目标**同口径**
（`fast_td3/train.py:485-491`：`use_cdq=True` 时 actor 用 `min(qf1,qf2)`；
本项目 `train_ptf.py:1753` 同）：

```
score_i(s) = min_h Q_h( s, π_i(s) )          h ∈ {1,2}
i*         = argmax_i score_i(s)
```

**并列时选 student**：把 student 放在候选索引 0，`argmax` 并列返回最小索引即自动满足。

这**不是** MCG 的 `min_h [ Q_h(a_cand) − Q_h(a_stu) ]`：本轮直接比较 Q 值本身，
与 QMP 的 `argmax_{π'} E[Q(s,a)]` 对应。

### 3.1 探索噪声时序（**关键，v1 的伪码在此处有 bug**）

`train_ptf.py:964` 的 `policy = actor_detach.explore` 返回的动作**已含探索噪声**。
若拿它与**无噪声**的 source action 比 Q，会**系统性偏向 source**（噪声压低 student 的 Q）。

正确顺序，冻结：

```
1. student_action = policy(obs=norm_obs, dones=dones, deterministic=True)   # 无噪声
2. src_actions, _ = source_bank.act_all(obs)                                # 无噪声
3. 用 critic 对上述【全部无噪声】候选打分，argmax 选出 a_sel
4. noise = torch.randn_like(a_sel) * actor_detach.noise_scales
   actions = a_sel + noise                       # 只采样一次，加在选中的动作上
```

**RNG 等价性**：`Actor.explore` 的 `noise_scales` 重采样发生在 `if deterministic: return act`
**之前**（`fast_td3.py`），因此 `deterministic=True` 仍保留 episode-level 重采样与其 RNG 消耗；
随后手动 `randn_like` 一次，与 `deterministic=False` 路径的 RNG 消耗**逐位一致**。
该等价性由 §7 的 forced-student smoke 验证。

不加 `clamp`——原 `explore` 返回 `act + noise` 无 clamp，保持一致。

### 3.2 其余全部保持 FastTD3 原样

- **无时间锁存**——QMP 是 per-state per-timestep 选择，逐步重选；
- **无蒸馏**——actor loss 不加任何模仿项；
- **replay 正常记录**——被选中的 source 产生的 transition 照常入 buffer，
  **uniform sampling**，不做屏蔽/加权/reweight；
- selected source **只**用于 provenance 与诊断，**不**触发 admission / reweight。

## 4. 实现方案

### 4.1 复用已验证的隔离路径

代码里已有 `target_only_behavior`（`train_ptf.py:1459`），它已实现本轮需要的全部隔离：

| 隔离要求 | 现有实现 |
|---|---|
| rollout 不走 classic option selector | `:2509` 分支 |
| actor 的 transfer loss 严格为 0 | `:1757` |
| 不更新 option/termination 网络（Q_ω/β） | `:2995` |
| 不构造 beta current-transition | `:2711` |
| classic rollout 诊断不启用 | `:3084`、`:3148` |

但它的定义是 `source_bank.num_sources == 0 or static_exact_abstention`，
**只在 bank 为空时为真**，而 QMP 需要非空 bank 作候选。

因此新增显式开关，**复用**而非重写这条隔离：

```python
qmp_enabled = bool(ptf_cfg.get("qmp"))
# QMP 复用 target-only 的全部 classic-PTF 隔离,只把 rollout 的动作选择换掉
isolate_classic_ptf = target_only_behavior or qmp_enabled
```

把 `:1757` / `:2711` / `:2995` / `:3084` / `:3148` 的 `target_only_behavior`
替换为 `isolate_classic_ptf`；rollout 分支改为：

```python
if target_only_behavior:
    actions = policy(obs=norm_obs, dones=dones)      # 纯 student,原样
elif qmp_enabled:
    actions = qmp_step(...)                          # §3 + §3.1
elif mcg_enabled:
    ...
```

**启动期硬断言**（不满足即 `raise`，不静默降级）：
`qmp_enabled` 时必须 `mcg_enabled=False`、`admission_enabled=False`、
`mcg_replay_mode="off"`、`transfer_loss ≡ 0`、`source_bank.num_sources ≥ 1`。

### 4.2 诊断量（只作机制诊断，**不参与任何 gate**）

- `qmp/source_share`：非 student 被选中的比例；
- `qmp/share_per_source`：每个源各自被选中的比例；
- `qmp/score_gap`：`max_i score_i − score_student` 的均值；
- `qmp/switch_rate`：相邻两步选择发生变化的比例；
- `qmp/run_len_per_source`：每个 source 的平均连续执行长度；
- **非有限 Q 拒绝启动**：任一候选的 score 出现 NaN/Inf 即 `raise`，不静默跳过。

## 5. 实验矩阵（资产已核实存在）

10k anchor（`completed_vector_steps=10000`，含 `learner.pt`/`replay.pt`/`rng.pt`）：

```
artifacts/slide_bac_gate_v1/anchors/s{1,2,3}/     ✓ 已核实
artifacts/door_at10k_gate_v1/anchors/s{1,2,3}/    ✓ 已核实
```

历史对照臂（source-free eval，各 128 episodes，**复用**）——已核实数值：

| task | student | stand | walk | run |
|---|---:|---:|---:|---:|
| **door** | **267.54** | 234.90 | **245.34** | 236.90 |
| **slide** | 51.18 | 49.97 | **108.12** | 68.07 |

配对差（相对 student，3 seed）：

```
door :  stand −32.64 (0/3)   walk −22.20 (0/3)   run −30.63 (0/3)   → walk 最不负
slide:  stand  −1.21 (1/3)   walk +56.95 (3/3)   run +16.90 (3/3)
```

**本轮新增的唯一臂**：

| task | 臂 | seeds | 角色 |
|---|---|---|---|
| slide | `qmp` | 1,2,3 | 正例：walk/run 已知有益 |
| door | `qmp` | 1,2,3 | 负例：loco bank 已知 9/9 有害 |

共 **6 个 run**，均 10k→20k。按 `feedback-node-memory-limit`：**并行 ≤3，分两批串行**。
评估口径与历史面板逐位一致：**source-free student**，128 episodes，step 20000。

## 6. 判据（冻结，跑前提交）

配对到 learner seed。`J_x` = 该臂 source-free student 在 20k 的回报。

> **v1 的 A2/B2 用"90% CI 上界 > 0"声称非劣效，这是错的**——那只能说明
> "无法排除有益"，一个均值严重为负但方差很大的结果也会通过。
> 改为不引入新 δ 的方向 gate。CI 全部报告，但**不用于**声称非劣效。

**Slide（正例，保留正迁移）**

```
A:  mean_seed[ J_qmp − J_student ] > 0   且  3/3 seed 为正
```

**Door（负例，减害 + 免疫）**

```
B1: mean_seed[ J_qmp − J_walk ] > 0      且  3/3 seed 为正
    （walk = 245.34，已冻结结果中最不负的固定 source）
B2: J_qmp − J_student  在 3/3 seed 上非负      ← 才配称"负迁移免疫"
```

**裁决**

| 条件 | 裁决 | 后果 |
|---|---|---|
| A 且 B1 且 B2 | `QMP_FIDELITY_SUPPORTED` | **才**解禁身体组扩展，另行预注册 |
| Door 过 B1 但不过 B2（优于固定源、仍低于 student）；或仅一侧通过 | `QMP_FIDELITY_PARTIAL` | 记录；**不解禁**身体组；下一步交审查 |
| A 与 B1 均不成立 | `QMP_FIDELITY_REFUTED` | per-state 完整策略选择在本设定下失败，整线停止 |

**禁止**：跑完后调整候选集合、打分口径、ties 规则、锁存设置、判据再重跑。

## 7. 历史对照复用与 Door 前置等价性检查

- **Slide**：对照臂对应的训练版本与当前 `train_ptf.py` 差距小，**直接复用**。
- **Door**：对照臂对应旧版本与当前代码差异较大，**复用前必须先过一次等价性 smoke**：

```
200-step forced-student 检查(同一 10k anchor,qmp_enabled 且强制选 student):
  对比对象 = 普通 target-only 路径
  比对项   = 逐步动作、transition/replay 写入 checksum、update counts、RNG 消耗
  通过     → 复用历史 Door 对照臂
  不通过   → 只补跑 Door 的 student 控制臂(不需要重跑全部历史 source 臂)
```

该检查同时验证 §3.1 的 RNG 等价性。

## 8. 剂量：本轮的立场（已按审查收口）

QMP 臂的 source 份额是**内生的**（argmax 决定），固定源臂是**外生固定的**，两者剂量不可比。

**本轮只主张"完整 QMP-style 机制整体是否有效"，不单独主张"状态条件选择"的因果贡献。**
`qmp/source_share` 等诊断量照常记录并报告，但**不设结果触发的补充臂**——
v1 的"`≈` 时补随机剂量臂"已删除：`≈` 未定义，且只匹配总 share 也控制不了
per-source、状态与时间结构。若本轮通过，未来另行**单独预注册**
yoked / shuffled-selector 对照来做因果归因。

## 9. 已知局限（写进结果，不事后追加）

1. **FastTD3 是确定性策略，无 QMP 的熵项**。Q-switch 退化为纯 argmax，
   仅靠 FastTD3 自带 exploration noise。
2. **QMP 的理论保证不适用于本设定**。Theorem 5.1 建立在 tabular SAC、有限动作空间上；
   本设定是连续 61 维动作、深度逼近、分布式双 critic、**冻结跨任务源**。
   理论只作动机，**不作安全保证**。
3. **无时间锁存的风险**：本项目经验是"教师闭环依赖全身协调"，逐步重选可能行为不连贯。
   本轮忠实于 QMP 的 per-timestep 语义；若出现明显抖动，记录为发现，
   **不在本轮加锁存抢救**。
4. 3 个 learner seed，df=2。

## 10. 本轮不做什么

- 不做身体组拼接——**A 与 B 全部通过前不解禁**；
- 不做蒸馏、null margin、显著性 gate；
- 不做跨状态聚合的分数（那正是被否定的八信号族的形式）；
- 不重跑历史对照臂（Door 除非 §7 等价性检查不通过）；
- 不把上一轮探针的脚本换任务重跑。
