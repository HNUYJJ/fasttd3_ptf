# T2 预注册：Truck 10k→20k 的 behavior / replay 因果分解

冻结时间：2026-08-08 · **先于 B-only 臂的任何运行**
判据冻结后只允许改路径参数。本轮**只作根因定位，不作论文 confirmatory result**。

---

## 0. 要回答的问题

Gate A 测得 truck 在 20k 点 `Δ = J_joint − J_scratch = −227.6`（3/3 负，t=−5.77）。
这个伤害来自**哪条通道**？

- **behavior 通道**：source 拿走了一半的行为权，改变了 student 的状态访问与
  自身探索机会（"student 自己动作机会减半"、"探索多样性下降"都属于这一条）；
- **replay-learning 通道**：source 产生的 transition 进入 critic/actor 的学习批次。

Gate A 的 joint 臂把两条通道**捆在一起**，无法归因。仓库已有现成接口可以拆开：

```
shared       = source behavior + source replay      （Gate A 的 joint 臂）
student_only = source behavior 照常，但 source 的 replay 配额恒为 0
```

`train_ptf.py:653-677` 的 `replay_candidate_masses` docstring 原文：
"`student_only`：critic 只采 student provenance 的 slot（source 配额恒 0），
而 source 仍照常获得 behavior authority、照常写入 physical buffer。
**这实现 B-only 臂 (B=1, R=0)**"。

**不需要新算法。** 只缺一条臂。

---

## 1. 设计

三臂全部从 Gate A **已有的同一个** A0 anchor（`truck_s{seed}_k10000`）分叉，
10k→20k，共享 `PTF_RESUME_NOISE_SEED = 92000 + seed`：

| 臂 | 来源 | replay mode | 状态 |
|---|---|---|---|
| scratch | Gate A `pgav1_scratch_truck_s*` @20k | — | **已有** |
| joint (BR) | Gate A `pgav1_scaf_truck_s*` @20k | `shared` | **已有** |
| **B-only** | 本轮新增 | `student_only` | 待跑 |

B-only 臂除 `PTF_ADMISSION_REPLAY_MODE=student_only` 外，与 Gate A 的 joint 臂
**逐项相同**：同 `h1hand_hurdle4_wfix_truck.yaml`、同 `student_logit=14.216676716804526`
（mass 0.5）、同 `MCG_GROUPS=legs_torso,arms`、同 `PROVENANCE_GROUPS=2`、
同 horizon、同 `PTF_ADMISSION_MODE=all`、同 noise seed。

- seeds = 1, 2, 3
- 评估：20k 的 source-free `panel128`，只用 return
- **不继续到 50k/100k**（Gate A 的 restart confound 见
  `pare_gate_a_posthoc_interpretation_20260808.md` §2；本轮只比 20k 单点，
  三臂在 10k 都经历同一次 matched resume，故 20k 比较是干净的）

---

## 2. 工程 gate（先于科学判定，任一不过则本轮作废）

| # | 检查 | 通过条件 |
|---|---|---|
| E1 | behavior source share | B-only 与 joint 在同一 seed 上一致（\|Δ\| ≤ 0.01） |
| E2 | critic source sample 增量 | **严格为 0**（10k→20k 期间 critic 采到的 source 样本数不增加） |
| E3 | source physical transitions | **非 0**（source 仍在写 buffer，证明 behavior 通道确实开着） |
| E4 | provenance 完整性 | `assert_complete_provenance` 通过 |

E2 与 E3 必须**同时**成立——这正是 B-only 臂的定义：
行为权在、replay 学习不在。任一不成立说明拿到的不是 B-only。

---

## 3. 分解与判定

逐 learner seed 计算（n=3，**用 learner 间方差，不得用 episode 面板 SE**）：

$$
U^B = J_{B} - J_{0},\qquad
\Delta^{R\mid B} = J_{BR} - J_{B},\qquad
U^{BR} = J_{BR} - J_{0}
$$

恒等式 $U^{BR} = U^B + \Delta^{R\mid B}$ 必须在数值上成立（作为一致性自检）。

| 条件 | 裁决 |
|---|---|
| $U^B$ 3/3 < 0 且 $\Delta^{R\mid B}$ 非 3/3 < 0 | `BEHAVIOR_SIDE_CANDIDATE` |
| $\Delta^{R\mid B}$ 3/3 < 0 且 $U^B$ 非 3/3 < 0 | `REPLAY_SIDE_CANDIDATE` |
| 两者均 3/3 < 0 | `JOINT_HARM_CANDIDATE` |
| 任一关键分量**跨 seed 翻符号** | `CHANNEL_UNRESOLVED` |
| 任一 (arm, seed) 评估缺失 | `INCOMPLETE`（非零退出） |

`CHANNEL_UNRESOLVED` 是**实质裁决之一，不是失败**：Door 的先例
（`project_channel_decoupling`：通道归因跨 seed 反向，且非噪声，episode \|U\|/SE 达 10–20）
已经证明单 seed 或 episode SE 不足以做通道归因。若本轮也翻符号，**直接停**，
不得靠加 seed 或换指标抢救。

---

## 4. 本轮不做什么

- 不开发新算法、selector、proxy、dose search、exploration mechanism
- 不启动 early-vs-late timing 实验（须等本轮结果才决定方向）
- PARE v1 与 PDAU 均保持 CLOSED
- 不继续到 50k/100k

---

## 5. 已知边界

- 本轮只测 **truck**、只测 **10k→20k 这一个 stage**、只测 mass 0.5。
  结论不得外推到其他 target、其他注入时机或其他剂量。
- 三臂比较是 20k 单点。它回答"伤害来自哪条通道"，
  **不**回答"这条通道的长期后果是什么"。
- `student_only` 只改 replay 采样配额；source transition 仍在 physical buffer 中。
  故 B-only 并非"source 数据从未存在"，而是"source 数据不被学习批次采到"。
  这个区别在解释结果时必须保留。
