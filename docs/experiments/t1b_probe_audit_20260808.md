# T1b：T1 probe 的 POST-HOC AUDIT

日期：2026-08-08 · 数据：`docs/data/pdau_probe_v1/t1b_corrected_audit.json`
**不修改冻结的 `NO_CONFLICT_STOP`。** 原 artifact 保留不覆盖。

结论：**T1 的核心判断在修正后依然成立**（无梯度符号冲突，PDAU 不实现），
但原报告的三处表述过头，且有一个真实的标签 bug。逐条更正如下。

---

## 1. 标签 bug：student 被错标为 `"null"`

`admission_execution_counts` 的结构是 `[real sources..., student]`，
长度 = `num_sources + 1`（`train_ptf.py:1408-1410`）。
而 `SourcePolicyBank.names()` 在 `null_option=True` 时会**追加 `"null"`**
（`source_bank.py:74-78`）。原 T1 脚本直接 zip 两者，于是 truck（4 源 +
`null_option: true`）的**最后一槽 student 被标成了 `"null"`**。

原 artifact 里那个 `"null": 0.5015` 实际是 **student 的 50.15%**。

- **不影响** `behavior_share = ec[:-1].sum()/ec.sum()`（`ec[0:4]` 确是 4 个真实源）。
- **不影响** T1 的 verdict（cosine 计算完全不依赖此标签）。
- stair 未受影响（`null_option: false`，`names()` 只有 1 个元素）。

修正后（truck，按结构显式命名）：

| seed | behavior source | behavior student | 真实源分布 |
|---|---|---|---|
| 1 | 0.4985 | 0.5015 | hurdle .2747 / walk .1238 / run .0999 / stand .0000 |
| 2 | 0.5002 | 0.4998 | hurdle .2762 / walk .1238 / run .1000 / stand .0001 |
| 3 | 0.4990 | 0.5010 | hurdle .2749 / walk .1233 / run .1009 / stand .0000 |

---

## 2. 补齐 critic 侧曝光（上轮遗漏项）

actor 在默认路径**直接复用 critic 的 replay batch**，所以 actor 真正看到多少
source 状态取决于 **replay sampling**，而不只是 behavior execution share。

从 replay snapshot 的 `admission_sampling.sample_counts` 读取：

| 任务 | critic source share | actor role |
|---|---|---|
| truck s1/s2/s3 | 0.4957 / 0.4955 / 0.4956 | **从未被采样** |
| stair s1/s2/s3 | 0.4991 / 0.4989 / 0.4989 | **从未被采样** |

`actor` role 的计数全为 0，直接证实了"actor 复用 critic batch"：
Gate A 用默认 `mcg_replay_mode='off'`，`rb.sample(..., role="actor")` 从未被调用。

**critic source share ≈ behavior source share ≈ 0.496–0.500**，三者一致，无异常。

bank horizon 核对（`VERIFIED`，文件可读）：truck 四源 horizon 均为 25，
weight = stand 5.2 / walk 12.816 / run 12.613 / hurdle 13.623；stair 的 slide
horizon 25、weight 0.0。`stand` 的 share≈0 与其 weight 最低经 softmax 的预期一致。

---

## 3. student estimand 错位：修正后 cosine 下降但仍全正

原 probe 的 `g_stu` 只取 10k–20k 的 `z=0`。但 20k 的 replay 里还完整保留
**0–10k 的纯 student prefix**，而 actor 是从**整个有效 replay** 采样的。
所以原口径测的是"当期 source-authority vs 当期 student-authority"，
不是"source 组分 vs actor 实际采到的 student 组分"。

补算 `actual_replay_student`（全部 provenance-written 的 `z=0`，含 prefix）：

| 任务 | seed | contemporaneous（原口径） | **actual_replay_student** | ‖g_src‖/‖g_stu‖ |
|---|---|---|---|---|
| truck | 1 | 0.9665 | **0.8646** | 1.555 |
| | 2 | 0.9478 | **0.8712** | 1.499 |
| | 3 | 0.8016 | **0.5534** | 1.916 |
| stair | 1 | 0.8744 | 0.8313 | 1.485 |
| | 2 | 0.9125 | 0.8346 | 1.445 |
| | 3 | 0.7442 | 0.6421 | 1.324 |

**修正后 cosine 仍然全部为正（0.553–0.871），无一负值** ——
T1 的核心判断成立，PDAU 不实现。

但这个错位**不是可忽略的细节**：truck s3 从 0.802 降到 0.553。
今后凡是"actor 看到什么分布"的论断，都必须按 actor 实际采样口径算。

**source 侧两个口径完全相同**，这是被验证而非假定的：脚本显式统计
"`z=1` 出现在 scaffold 窗口外"的数量，六个 (task, seed) 组合**全部为 0**——
0–10k 是 empty-bank 纯 student，确实不可能有 source-authority transition。

一个附带观察（**不作新假设推进**）：在正确口径下 `‖g_src‖/‖g_stu‖` 升到
**1.32–1.92**。方向仍一致，但 source 状态贡献的梯度幅度明显更大。

---

## 4. 措辞收紧

| 原表述 | 更正后 |
|---|---|
| `actor-distribution-contamination hypothesis falsified` | **`NO_ENDPOINT_AUTHORITY_CONDITIONED_GRADIENT_CONFLICT_AT_20K`** |
| "10k 诊断排除了『actor 已经适应』" | **`OOD_COUNTERFACTUAL_DIAGNOSTIC`**，见下 |
| "source-shaped 状态没有冲突" | "**source-authority-conditioned** 状态在 20k endpoint 上没有平均梯度符号冲突" |
| "不存在冲突被噪声掩盖的空间" | 加限定：**仅就当前 endpoint estimator 的有限样本噪声而言** |

### 4.1 10k 诊断的地位（我上轮说过头了）

我上轮写"用 10k 的 actor/critic 排除了『20k actor 已适应』"。**这个说法不成立。**

10k 的 critic **从未见过**随后 10k–20k scaffold 分支产生的那些状态，
拿它给这些状态上的动作打 Q 梯度，本质是 **OOD / counterfactual 外推**。
它得到正 cosine 有参考价值，但**不能**证明"10k–20k 训练期间任何时刻都没出现冲突"，
也不能排除中途（如 11k、14k）出现冲突、随 critic 更新旋转、到 20k 又重新对齐。

因此"12 个组合无一冲突"**不能**被理解为 12 个独立训练时刻的证据。

### 4.2 `z=1` 不等于 "source-shaped state"

`executed_group_mask[t]` 描述的是**在状态 s_t 上这一步动作由谁执行**
（current action authority），不描述**状态 s_t 是谁造成的**（causal occupancy origin）。
source 连续执行若干步后 handoff，handoff 后第一步的状态显然由前面的 source 行为造成，
但该 transition 会被记为 `z=0`。

### 4.3 split-half 排除了什么

0.97–0.995 的 split-half 只说明：给定当前 checkpoint、当前标签定义、当前数据分布，
**这个平均梯度估计的有限样本噪声很小**。它**不排除** critic 的系统性偏差、
provenance 分组偏差、endpoint-only 的时间偏差，以及上面第 3 条的 estimand 错位
（后者已实测确实影响数值）。

---

## 5. 最终裁决状态

- T1 冻结 verdict `NO_CONFLICT_STOP` **不变**，PDAU **不实现**。
- 科学表述改为
  **`NO_ENDPOINT_AUTHORITY_CONDITIONED_GRADIENT_CONFLICT_AT_20K`**：
  PDAU 所依赖的直接梯度反向机制没有得到支持，故不开发 PDAU；
  **但这不等于所有 actor-side behavior/occupancy harm 都已被排除。**

这两个层次必须分开写，不得合并成"整体证伪"。
