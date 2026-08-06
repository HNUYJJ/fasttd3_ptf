# 设计草案（待 review，未实现）：RACING_REJECT v2

> 2026-07-30。v1 已作废（`da32b13`），作废时未揭盲。本草案按 Codex review 的五条意见重做。
> **本文只是设计，尚未写任何代码、尚未跑任何新实验。**

## 1. v1 死于什么（已核实）

`sanity(K=10000 时 9/9 全负)` ⇒ `∀seed: max_i U_i(10000) < 0` ⇒ `K*_reject ≤ 10000` 必然成立；
而 K=10000 不全负时走 `VOID` 而非 `REJECT_REFUTED`。**主假设不可证伪。**

## 2. 一个 v1 和 Codex 都没点破、但决定 v2 形态的事实

Codex 建议把拒绝条件改成 `max_i UCB_i(K) < −δ`。**但这个判据在 door 上先验地不可能通过**：
已发表的 ground truth 里 walk 的 learner 层 90% CI 是 `[−48.83, +4.43]`，**跨 0**——
即便用 K=10000 的完整测量、n=3 learner，walk 的 UCB 也 > 0。
要求"所有源 UCB < 0"等于要求一个 ground truth 自己都达不到的标准，
那不是严格，是**判据与场地不匹配**（同 `feedback_decision_field_effect_sign` 的教训）。

**推论**：door 上 n=3 的 learner 方差不足以支撑"统计显著地全负"。
v2 必须换一个既可证伪、又与场地功效相称的主终点。

## 3. v2 的主终点：**决策一致性**，且只问"能否提前"

racing 在实用中输出一个决策：

```
decide(K) = argmax_i U_i(K)      若 max_i U_i(K) > 0    （用这个源）
          = REJECT               否则                    （不用任何源）
```

ground truth 决策：
- hurdle（`RACING_K`）：`USE run`（U=+379.66）
- door（本实验）：`REJECT`（三源 K=10000 全负，9/9 per-seed）

**主终点（可证伪）**：

```
H: 在 K ≤ 5000 时，decide(K) = REJECT 在 3/3 seed × 2 批 上成立
```

K=10000 **不参与主终点**，只作复制检查（§5）。这样"提前拒绝失败"是真实可达的结果。

| 结果 | 裁决 | 含义 |
|---|---|---|
| K=2000 或 5000 在两批各 3/3 | `EARLY_REJECT_CONFIRMED` | 拒绝比选源更便宜（≤15k vs 30k 步） |
| 仅 K=10000 达标 | `EARLY_REJECT_REFUTED` | **有意义的负结果**：拒绝与选源代价同量级，不存在"便宜的安全网" |
| 连 K=10000 也不达标 | `REPLICATION_DIVERGED` | 复制失败，不解释为机制结论（见 §5） |

## 4. 点估计判据的错误率：预先声明，不回避

主判据用点估计 `max_i U_i(K) < 0`，与 racing 的实用形式一致。其偶然通过率必须预先算清：

**零效应零假设**下（student 与三源同分布、独立），单个 seed 上 `max_i U_i < 0`
等价于 student 恰为四臂最大 = `1/4`；3 seed 独立 → `(1/4)³ = 1/64 = 1.56%`；
两批 6 seed 全中 → `(1/4)⁶ = 0.024%`。

**因此要求"两批各 3/3"把偶然通过率从 1.56% 压到 0.024%**——这正是 M24
（单批 3/3 不足以定论）的量化落实，也是不引入 learner-UCB 的补偿。

同时**必须报告**（不参与裁决）：每个源的 learner 层 `mean ± t(0.90,2)·SE_learner`，
让读者自行判断哪些源是"显著负"、哪些只是"点估计负"。

## 5. sanity 拆两层（Codex 建议，采纳）

**层 1 · 工程硬检查**（任一不过 → `VOID_ENGINEERING`，不输出任何主结果）：

- 剂量 behavior share ∈ `[0.48,0.52]`（**写进裁决脚本**，v1 缺）
- 每个源臂 checkpoint 的 `source_names` == 臂名（防 stand/run 臂对调，v1 检测不到）
- student 臂 `source_names == ['null']` 且无 `admission_audit`
- 12/12 `Resumed core learner ... at step 10000`
- 每个 eval json：`episode_count==128`、`identity_checked==True`、
  `global_step` 匹配、`checkpoint.path` 含正确臂名与 seed、sha256 两两不同

**层 2 · 复制检查**（K=10000 与已发表 gate 比对）：

- **逐 seed** 比较（v1 只比 mean，故 stand↔run 对调也能通过）
- 用**真正的 paired SE**：`p0_evaluator` 面板逐位配对
  （注释："(seed, rank) → 唯一 reset seed；面板冻结，分支间逐位相同"），
  故取逐 episode 差值序列的 SE，而非 `sqrt(se₁²+se₂²)`
- 失败称 `REPLICATION_DIVERGED`，**不自动等于实现 bug**（CUDA 非确定性亦可致）

**层 2 不再是主终点的前提**：主终点只用 K≤5000，故层 2 通过与否不蕴含主结论。

## 6. 独立重复（M24，硬性）

- **批 1**：v1 已产出的 12 条训练 + 36 点评估（未揭盲，剂量 PASS，结构校验 36/36）。
- **批 2**：新跑 12 条，仅 `EXP_PREFIX` 不同，其余逐项相同。
- **合并规则（预先冻结）**：取两批**都**满足 3/3 的最小 K；单批达标不采纳。

**批 1 复用的合法性**：v2 判据在**查看任何 U 数值之前**冻结，故判据选择未受数据影响。
此事实须写进结果文档。

## 7. 待 review 的问题

1. §3 的"决策一致性"主终点是否真的可证伪？有没有我没看见的、使 `EARLY_REJECT_REFUTED`
   不可达的隐含蕴含（v1 就是死在这里）？
2. §2 的判断——"要求所有源 UCB<0 与 door 的功效不匹配"——是否成立？
   还是我在为回避统计严格性找借口？
3. §4 的零效应基准率（1/64 单批、1/4096 两批）推导是否正确？
   独立性假设（4 臂同分布独立）在配对面板下是否成立？
4. §6 复用批 1 是否构成 selective reporting？
   我认为不构成（判据先于揭盲冻结），但请指出反面理由。
5. 还有哪些 v1 式的"前提蕴含结论"结构隐藏在本设计里？

## 8. 出处

```
v1 作废          docs/experiments/racing_reject_door_v1_prereg_20260730.md (da32b13)
ground truth     docs/experiments/door_at10k_gate_v1_results_20260727.md
RACING_K         docs/experiments/racing_min_horizon_v1_results_20260730.md (a744adb)
教训 M20-M24     docs/ISSUES_AND_LESSONS.md
强制规范         CLAUDE.md (67071e1)
```
