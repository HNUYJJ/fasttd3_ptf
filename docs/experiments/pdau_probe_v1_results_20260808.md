# T1 机制 probe 结果 —— `NO_CONFLICT_STOP`

日期：2026-08-08 · 判据：`scripts/analysis/probe_provenance_actor_gradient_v1.py` 文档串（先于结果冻结）
数据：`docs/data/pdau_probe_v1/gradient_probe.json`（判定口径）
　　　`docs/data/pdau_probe_v1/gradient_probe_actor10k_diagnostic.json`（post-hoc 诊断）

**结论：actor-distribution-contamination 假设被证伪。不实现 PDAU。**

零环境交互，只读已有的 20k branch anchor，总耗时约 20 分钟。

---

## 1. 被检验的命题

DPG 下 actor 更新是在某个状态分布上求 action-value 梯度：

$$
\psi_\theta(s) = \nabla_\theta \pi_\theta(s)\,\nabla_a Q(s,a)\big|_{a=\pi_\theta(s)},
\qquad g_d = \mathbb{E}_{s\sim d}[\psi_\theta(s)]
$$

scaffold 期 replay 的状态分布是混合 $d_M = (1-\alpha)d_0 + \alpha d_S$，故

$$
g_M - g_0 = \alpha\,(g_S - g_0)
$$

若 $\langle g_0, g_S\rangle < 0$，source-shaped 状态就在给 actor 一个与
student-authority 状态相冲突的更新方向——**而这并不要求 source transition
对 critic 无用**。这是"source experience 对 value learning 与对 policy
improvement 作用不同"这一想法最直接、最便宜的可证伪版本。

---

## 2. 剂量 sanity check（先于梯度分析）

名义 admission mass 0.5 是概率质量，MCG 有 latch/horizon，实际占用时间未必相等。
实测 `admission_execution_counts`（scaffold 期 1,280,000 次 behavior 决策）：

| 任务 | 实际 behavior source share | per-source |
|---|---|---|
| truck s1/s2/s3 | 0.4985 / 0.5002 / 0.4990 | hurdle ≈0.275, walk ≈0.124, run ≈0.100, stand ≈0.000 |
| stair s1/s2/s3 | 0.4964 / 0.4987 / 0.4976 | slide ≈0.497 |

与名义 0.5 一致，**无异常**。`stand` 的 share≈0 也与其 bootstrap weight 最低
（5.2 vs 12.6–13.6）经 softmax 后的预期一致，不是 bug。

---

## 3. 判定口径结果（20k actor/critic，冻结判据）

判据：PRIMARY = truck，进入 T2 需**同时**满足
(a) 至少 2/3 seed 的 $\cos(g_{src}, g_{stu}) < 0$；
(b) 这些 seed 两组的 split-half cosine 均 > 0.5。

| 任务 | seed | $\cos(g_{src},g_{stu})$ | split-half src | split-half stu | $\lVert g_{src}\rVert/\lVert g_{stu}\rVert$ |
|---|---|---|---|---|---|
| **truck** | 1 | **+0.9665** | 0.9950 | 0.9934 | 1.128 |
| | 2 | **+0.9478** | 0.9944 | 0.9936 | 1.092 |
| | 3 | **+0.8016** | 0.9752 | 0.9739 | 1.294 |
| stair | 1 | +0.8744 | 0.9880 | 0.9780 | 1.107 |
| (diagnostic) | 2 | +0.9125 | 0.9880 | 0.9840 | 1.152 |
| | 3 | +0.7442 | 0.9870 | 0.9840 | 1.141 |

**0/3 truck seed 满足冲突条件，且全部是强正对齐。** `verdict = NO_CONFLICT_STOP`。

每组样本量约 64 万（source）/ 64 万（student），每个 seed 只产出一个 cosine——
batch 重复不是 learner replication。

### 为什么这个结论不是噪声

组内 split-half cosine 达 **0.97–0.995**：把同一组随机对半，两半的梯度方向几乎完全一致，
说明梯度估计本身极其稳定。在这种噪声水平下，跨组的 +0.80~+0.97 是真实的强一致，
不存在"冲突被噪声掩盖"的空间。**没有这个对照，单看跨组 cosine 无法排除噪声解释。**

---

## 4. Post-hoc 诊断：排除"20k actor 已经适应了"

判定口径用的是 scaffold **结束**时刻的 actor/critic。一个合理的质疑是：
20k 的 actor 已在混合分布上训练了 10k 步，此时两组梯度对齐可能是**适应的结果**，
而不是"从未冲突"。

于是用 **10k anchor**（scaffold 尚未开始、actor 完全未被 source 影响）的
actor/critic 评估**同一批** scaffold 期状态：

| 任务 | s1 | s2 | s3 | split-half |
|---|---|---|---|---|
| truck | +0.9896 | +0.9881 | +0.7154 | 0.998–0.999 |
| stair | +0.9351 | +0.9358 | +0.8925 | 0.999 |

**同样全部强正。** 该质疑被排除：在 source 刚要介入的那一刻，
source 状态与 student 状态要求的 actor 改进方向就已经高度一致。

此口径标记为 `DIAGNOSTIC_ONLY`，**不参与 T1 裁决**——冻结判据只认 20k 口径。
两个口径同向，结论是加固而非改判。

合计 **2 个任务 × 3 seed × 2 个时刻 = 12 个组合，无一出现负对齐。**

---

## 5. 这排除了什么，没排除什么

**排除**：source-shaped 状态**不是**通过给 actor 一个冲突的 policy-improvement
方向来造成伤害。truck 20k 的 $\Delta = -227.6$（t=−5.77）另有原因。

**没排除**：cosine 高只说明**方向**一致，不等于 source 数据无害。
本 probe 只测了 actor 梯度方向这一个维度。

一个附带观察（**不作为新假设推进**）：$\lVert g_{src}\rVert / \lVert g_{stu}\rVert$
恒 > 1（1.07–1.51）。方向相同但幅度更大，意味着 actor 更新在混合 batch 里
被 source 状态**加权更重**——这不是冲突，但也不是中性。

按 T1 的约定，**到此停止，不发明算法**。剩余候选解释（例如 student 自身
on-policy 数据减半、critic 侧对 source 动作的价值估计偏差、探索多样性下降）
各自都需要新的可证伪设计，不在本轮范围内，也不应挂在 PDAU 名下。

---

## 6. 方法学价值

这个 probe 零环境交互、约 20 分钟，把一个原本需要数周训练才能检验的算法方向
直接证伪了。它之所以有效，是因为假设被写成了**一个可以在已有 checkpoint 上
直接计算的量**（两组状态的 actor 梯度夹角），而不是"跑出来看谁的曲线高"。

这个模式值得复用：**新机制假设先找有没有零训练的判别式，再考虑实现算法。**
