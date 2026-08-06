# QMP-fidelity v1 结果：`QMP_FIDELITY_PARTIAL`（身体组**不解禁**）

> 2026-07-29。run card 与裁决脚本提交于 `1648b9c` / `0d93e6d`，
> **先于任何 QMP 臂被训练或评估**。
> 结论：**per-state 完整策略 Q-switch 在冻结跨任务源设定下退化为 student。**

## 1. 裁决

```
VERDICT: QMP_FIDELITY_PARTIAL      解禁身体组: False
```

| 判据 | 量 | mean | 90% CI | per-seed | 结果 |
|---|---|---:|---|---|:--:|
| **A** | slide `J_qmp − J_student` | **+0.12** | [−14.27, +14.51] | 2/3 正 | **FAIL** |
| **B1** | door `J_qmp − J_walk` | **+21.67** | [+10.85, +32.49] | 3/3 正 | **PASS** |
| **B2** | door `J_qmp − J_student` | **−0.53** | [−16.44, +15.39] | 1/3 非负 | **FAIL** |

（CI 照常报告，但按外部审查裁定**不用于**声称非劣效。）

`J_sf@20k`（source-free student，128 episodes/臂）：

| task | QMP | student | walk | run | stand |
|---|---:|---:|---:|---:|---:|
| slide | 45.87 / 54.04 / 53.98 | 39.38 / 63.61 / 50.54 | **105.04 / 111.22 / 108.11** | 68.07* | 49.97* |
| door | 274.40 / 265.51 / 261.12 | **265.47 / 275.45 / 261.69** | 258.41 / 236.88 / 240.72 | 236.90* | 234.90* |

\* 三 seed 均值。

按 run card §6：Door 过 B1 但不过 B2 → `PARTIAL`，**不解禁身体组扩展**。

## 2. 机制诊断：QMP 几乎从不选源

训练期 `qmp/*` 全程序列（每 run 100 个记录点，从 wandb 本地流读出）：

| task | seed | `source_share` 均值 | 最大 | `score_gap` 均值 |
|---|---|---:|---:|---:|
| door | 1 / 2 / 3 | 0.0125 / 0.0034 / 0.0057 | 0.070 / 0.023 / 0.039 | 0.0016 / 0.0010 / 0.0005 |
| slide | 1 / 2 / 3 | 0.0549 / 0.0495 / 0.0332 | 0.148 / 0.156 / 0.094 | 0.0087 / 0.0083 / 0.0065 |

- 源被选中时平均只连续执行 **1.04–1.12 步**；student 连续执行 **94–202 步**；
- `score_gap = max_i score_i − score_student` 平均 0.0005–0.0087，
  相对 Q 值量级（door ≈25、slide ≈13–16）是 **0.004%–0.07%**；
- slide 上三个源的选择率几乎均等（stand 1.3–1.6%、walk 1.1–2.0%、run 0.9–1.9%），
  **而它们的真实学习效用是 walk +56.95、run +16.90、stand −1.21**。

**因此 B1 的"通过"必须如实解读**：QMP 在 door 上优于所有固定源，
是因为它**几乎不使用被测机制**，行为上等同于 student——而 door 的 student
本就优于所有固定源。这不是"负迁移免疫成功"，是机制未被激活。

A 的失败是同一枚硬币的另一面：在源确实有益的 slide 上，
QMP 同样退化为 student，因此**错过了 walk 源 +56.95 的正迁移**。

## 3. 为什么 critic 不选源：两个独立的失败

事后诊断（`qmp_ood_pessimism_diagnostic_v1.py`，**探索性，不参与裁决**），
打分口径与训练端完全一致 `score_i = min_h Q_h(s, π_i(s))`，各 2400 状态：

### 3.1 系统性低估：18/18 组合为负

`score_source − score_student` 的均值，在 **两个任务 × 三个源 × 三个 seed = 18 个组合上全部为负**：

| task | walk | run | stand |
|---|---:|---:|---:|
| slide | −0.469 | −0.337 | −0.608 |
| door | −0.493 | −0.290 | −0.375 |

包括 slide 上真实效用 **+56.95** 的 walk 源。

### 3.2 排序也错：2/2 任务不一致

| task | critic 的 Δ 排序 | 真实 U 排序 | 一致 |
|---|---|---|:--:|
| slide | run > walk > stand | walk > run > stand | ✗ |
| door | run > stand > walk | walk > run > stand | ✗ |

即使忽略绝对值的系统性负偏，**相对排序也不能用**。
两个任务上 critic 都把 run 排在 walk 之前，而 walk 才是真实效用最高的源。

## 4. 对 OOD 悲观性假设的检验：**混合证据，不下定论**

事前写下的假设 H_ood 是：冻结跨任务源的动作对 target critic 分布外，
CDQ 的 `min(Q1,Q2)` 系统性压低它们。预测 P1 = 源动作上的双 head 分歧显著更大。

实测（head 分歧比 = 源/学生）：

| task | head 分歧比 | 源胜过学生的状态比例 | student 的 head 分歧绝对值 |
|---|---:|---:|---:|
| door | **1.26×** | 0.032 | 0.29–0.30 |
| slide | **1.02×** | 0.300 | 2.05–2.90 |

- **door 支持 P1**（1.26×），**slide 不支持**（1.02×，几乎无差异）；
- 因此 **H_ood 不能作为确定解释**。§3 的两个失败是稳健的观察事实，
  但它们的成因尚未被本轮确定。
- 附带观察：slide 的 critic 自身 head 分歧绝对值比 door 大约 7–9 倍
  （相对 Q 值约 18% vs 1.2%），说明 slide 的 critic 在 20k 时远未收敛——
  这是另一条可能的线索，本轮未检验。

### 4.1 一处必须标注的不一致

诊断给出 slide 上单个源胜过 student 的状态比例是 **0.30**，
而训练期实测 `source_share` 只有 **0.033–0.055**。二者相差近一个量级。

可能来源（本轮**未**区分）：诊断用的是 20k 的 critic 与 20k student 的
rollout 状态，而训练期是 10k→20k 演化中的 critic 与 QMP 自身行为诱导的状态分布。
**在澄清之前，不得用诊断的 win_rate 去反推训练期的选择行为。**

## 5. 本轮的科学结论

前八个信号族的统一诊断是"标量聚合抹掉了状态维度"，本轮据此做了**取消聚合**的
最忠实实现——per-state、完整策略、只改行为、无阈值、无蒸馏。结果是：

> **取消聚合并不能拯救该信号族。问题不在聚合方式，而在被聚合的量本身：
> target critic 对冻结跨任务源动作的即时价值估计，既不能定位（18/18 为负）
> 也不能排序（2/2 任务排序错误）其延迟学习效用。**

这与 M19（"行为即时效果 ≠ 延迟学习价值"）同源，但**首次在 critic 空间、
per-state 粒度上取得直证**：此前的反例都建立在某种聚合量上，本轮没有聚合。

同时这也精确定位了 QMP 从原设定迁到本设定失效的位置：QMP 的 mixture 是
**同时训练**的多任务策略，其动作在各自 replay 分布内且随 critic 共同演化；
本项目的源是**冻结的跨任务**策略。Theorem 5.1 的 argmax 步骤本身不依赖这一点，
但它保证的是"不比当前策略差"，**并不保证 argmax 能识别出有价值的源**——
本轮的结果正落在这个缝隙里：机制安全（door 未受害），但无能（slide 未获益）。

## 6. 限制

1. 3 个 learner seed，df=2；A 与 B2 的 CI 都很宽（±15 左右），
   本轮只能说"QMP 与 student 无可检测差异"，不能说"二者相等"。
2. §4 的 H_ood 未被确证，§4.1 的不一致未被澄清。
3. 只测了 10k→20k 一个窗口、两个 target、一个三源 loco bank。
4. FastTD3 为确定性策略，无 QMP 的熵项；无时间锁存（忠实于 QMP 的 per-timestep 语义）。
   本轮未检验这两处差异是否影响结论。

## 7. 后续（不自行启动）

按 run card §6，`PARTIAL` 的后果是**不解禁身体组扩展**，下一步交审查。
可选方向（均需另行预注册）：

- **A**：接受 §5 的结论，把"即时 critic 优势与延迟学习效用无关"
  作为论文的问题刻画写入，与八信号族的负结果链合并；
- **B**：澄清 §4.1 的不一致，并检验 slide critic 未收敛（head 分歧大 7–9 倍）
  是否是选择率低的主因——若是，则应在**更晚的决策时点**重测；
- **C**：放弃"用 critic 选源"，转向不依赖即时价值估计的信号族。

## 8. 数据与复现

```
run card      docs/run_card_qmp_fidelity_v1.md                    (1648b9c)
裁决脚本      scripts/analysis/analyze_qmp_fidelity_v1.py         (0d93e6d)
实现          fasttd3_ptf/ptf/qmp.py, train_ptf.py                (be7cedf)
测试          tests/test_qmp.py                                   11/11
训练          scripts/run_qmp_fidelity_v1.sh   TASK={door,slide} SEEDS='1 2 3'
评估          scripts/eval_qmp_fidelity_v1.sh  (panel128, source-free student)
裁决输出      docs/data/qmp_fidelity_v1/qmp_fidelity_v1_results.json
诊断输出      docs/data/qmp_fidelity_v1/ood_pessimism_diagnostic.json
脏树 provenance  docs/data/qmp_fidelity_v1/provenance.md
```

隔离的实证（见 `be7cedf`）：200-step forced-student smoke 的 option/beta
optimizer state **完全为空**（PyTorch 仅在首次 `step()` 时创建 state），
而同 bank/anchor/步数的 classic PTF 阳性对照为 `option: n=6 steps=[400]`。
