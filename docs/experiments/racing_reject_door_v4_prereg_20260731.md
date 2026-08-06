# 预注册：RACING_REJECT v4 —— split-sample 重做，裁决 door 的拒绝主终点

> 2026-07-31。v2 `REPLICATION_DIVERGED`（主终点未裁决）、v3 因
> **outcome-contingent gate switching** 被撤回（`b66e6cb`）。
> 本版按 Codex 裁定的**唯一合法路径** split-sample 重做。
> **提交本文时：seeds 4–6 与 7–9 的任何 `K≤5000` 数据均未被查看**（§3 可自证）。

## 1. split-sample 划分（Codex 裁定，本设计的核心）

| seeds | 角色 | 状态 |
|---|---|---|
| **1–3** | **design data** —— 明确承认其结果已被我看到，且**已被用于形成本判据** | 已揭盲（K=10000 层2 的 9 个值） |
| **4–6** | **holdout** —— 未揭盲，不参与判据形成 | 数据已产出，`K≤5000` 从未计算 |
| **7–9** | **新增批** —— 本轮新跑 | 待跑 |

**确认性裁决只使用 seeds 4–9。seeds 1–3 不参与主终点。**

这是 split-sample 的标准逻辑：在一部分数据上形成方案，在未分析的 holdout
与新数据上检验。我不再声称"判据未受已看到结果影响"——
**恰恰相反，我明确承认它受了影响，因此把那批数据整体排除出裁决。**

## 2. v3 错在哪（防止重蹈）

v3 试图在**同一数据**上把已知失败的数值/排序门换成已知通过的符号门。
即使主终点数据仍盲、即使如实披露，也不能恢复确认性地位。
`CLAUDE.md` §8.7 已把它固化为红线。

## 3. 盲态可自证

```
docs/data/racing_reject_door_v2/results.json 顶层键：
  ['dose','layer2_replication','prereg','prereg_commits','reject_rule','run_id','verdict']
  含 per_K ? False        含 layer3_batch2_reference ? False
```

v2 脚本在层2 失败处 `dump()`+`SystemExit`（`:273`），`per_K` 在 `:293` 才计算，
`layer3`（批2 = seeds 4–6）更在其后。故 **seeds 4–6 的任何 U 从未被计算或输出**。
v3 撤回时我**主动放弃**了零成本的 `POST_HOC_SENSITIVITY` 重算，正是为保住这一地位。

## 4. 主终点（与 v2 逐字相同，未做任何修改）

```
decide(K) = REJECT  ⟺  max_i U_i(K) ≤ 0
H：存在 K ∈ {2000, 5000}，使 decide(K)=REJECT 在 seeds 4–6 全部 3/3
   且 seeds 7–9 全部 3/3 上成立
```

| 结果 | 裁决 |
|---|---|
| K=5000 两批各 3/3 | `EARLY_REJECT_CONFIRMED` |
| 仅 K=2000 两批各 3/3，K=5000 失败 | `EARLY_REJECT_NONMONOTONIC`（不得称"稳定安全网"） |
| 二者均未达标，且两批 K=10000 决策均为 REJECT | `EARLY_REJECT_REFUTED`（有意义的负结果） |
| 任一批 K=10000 决策不是 REJECT | `PARTICIPANT_DIVERGED` |

偶然通过率上界 ≈ **1/32 (3.1%)**：复合零假设 `(1/2)⁶` × look-elsewhere（2 个 K）。

## 5. 前置门：改为**批内自洽**，不再与 gate 数值比对

**理由是数据结构决定的，不是判据切换**：gate 的 ground truth 只有 seeds 1–3，
而本轮裁决只用 seeds 4–9，**根本无法与 gate 比对**。
这同时消除了 v2/v3 卡住的容差问题——不再需要任何跨批数值容差。

```
层1 · 工程硬检查     与 v2 完全相同（含 R8 修正后的 source_names 判据）
层2 · 批内自洽       seeds 4–6 与 7–9 的 decide(K=10000) 均须为 REJECT
                     否则 PARTICIPANT_DIVERGED
层3 · 无             （取消与 gate 的数值比对）
优先级               VOID_ENGINEERING > PARTICIPANT_DIVERGED > 主终点
异常分类             缺产物→INCOMPLETE；产物无效/未预期异常→VOID_ENGINEERING
```

**报告项（不参与裁决）**：每批每源的 learner 层 `mean ± t₀.₉₅,₂·SE_learner`
（两侧 90%，`t₀.₉₅,₂ = 2.919986`，每批 n=3 分别报告，不合并 df）。

## 6. 协议

```
target        h1hand-door-v0
seeds 4–6     anchor = artifacts/door_at10k_gate_v1/anchors/s{4,5,6}（已建）
seeds 7–9     anchor = 同上目录 s{7,8,9}（本轮新建，协议逐项相同）
臂            student / stand / walk / run（四臂配对同 seed）
noise 重采样  PTF_RESUME_NOISE_SEED = 91000 + seed
剂量          behavior 0.5 / replay 0.5，h=25，bootstrap_only；验收带 [0.48,0.52]
K 取值        2000, 5000, 10000 → checkpoint 于 global_step 12000/15000/20000
评估          source-free student, deterministic, 128 episodes（16 eval seeds × 8 ranks）
```

seeds 4–6 的 12 条训练与 36 点评估**已完成**（v2 产出，层1 已通过）；
本轮只需新跑 seeds 7–9 的 anchor + 12 条训练 + 36 点评估。

## 7. `CLAUDE.md` §8 设计层自查

- **8.1 辨别力**：平凡解释="door 上任何决策都会是 REJECT，因为 student 也在退步"。
  排除方式：主终点判的是 `max_i U_i ≤ 0`，`U` 是**相对 student 的配对差**，
  student 自身的涨落已被消去。
- **8.2 混淆**：剂量严格带 + 逐 checkpoint 验收；四臂共享 anchor 与 noise seed。
- **8.3 独立重复**：两批**不相交**的 learner seeds（4–6 与 7–9），满足 M24。
- **8.4 前提蕴含**：层2 只判 `K=10000`，主终点只用 `K≤5000`；
  `RACING_K` 已实证 horizon 间可反向（K=2000 时 run 系统性排最后），故不蕴含。
- **8.5 site selection**：door 是**已知全负**后选定的，只能声称该特定案例。
- **8.6 是否重演本轮教训**：M25–M30 已逐条对照（见提交记录）。
- **8.7 判据切换红线**：本设计**不复用** design data（seeds 1–3）做裁决。

## 8. 能与不能声称

**不得声称**：统计显著地证明所有源有害；通用负迁移免疫；与 hurdle 加速合并核算；
`NONMONOTONIC` 时称"稳定安全网"。
**必须声明**：seeds 1–3 已被用于判据形成、故整体排除出裁决；
door 系已知全负后选定的场地。

## 9. 不得做的事

- 裁决后不得再改任何判据。
- **若本轮前置失败，不得再修改判据**——v2/v3 已各改一次，红线已达。
- 在本文冻结前，不得查看 seeds 4–9 的任何 `K≤5000` 数据。
- 若 `EARLY_REJECT_REFUTED`，不得改用代理量补救。
