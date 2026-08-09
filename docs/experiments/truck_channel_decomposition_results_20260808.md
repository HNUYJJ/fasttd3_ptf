# T2 结果：Truck 10k→20k 的 behavior / replay 分解 —— `JOINT_HARM_CANDIDATE`

日期：2026-08-08 · 判据：`truck_channel_decomposition_prereg_20260808.md`（先于运行冻结）
数据：`docs/data/truck_channel_v1/channel_verdict.json`

> ⚠️ **先读 §7 表述更正**（2026-08-08 追加）。本文正文中"两条通道各自独立有害"
> 与"约六四开"两处表述**已被收回**——三个 cell 不足以支持独立主效应的因果分解。
> 冻结 verdict `JOINT_HARM_CANDIDATE` 与全部数值**均不变**。

**结论：source behavior 在关闭 source replay 时有负效应；在 source behavior 已存在的
条件下，再启用当前 source replay scheme 会进一步降低性能。**
本轮只作根因定位，非论文 confirmatory result。

---

## 1. 工程 gate（先于科学判定，全过）

| seed | E1 behavior share (B-only vs joint) | E2 critic source samples | E3 source physical | E4 provenance |
|---|---|---|---|---|
| 1 | 0.498535 vs 0.498535 | **0** | 638,125 | ✓ |
| 2 | 0.500176 vs 0.500176 | **0** | 640,225 | ✓ |
| 3 | 0.499043 vs 0.499043 | **0** | 638,775 | ✓ |

E1 是**完全相等**而非"在容差内"——三臂共享同一 A0 anchor 与同一 `resume_noise_seed`，
而 `student_only` 只改 replay 采样配额，不触碰 behavior 路径。

E2 与 E3 同时成立，正是 B-only 的定义：**source 照常执行动作并写入 physical buffer
（63.8 万条），但 critic 一个 source 样本都没采到**。

---

## 2. 分解

$$
U^B = J_{B} - J_{0},\qquad
\Delta^{R\mid B} = J_{BR} - J_{B},\qquad
U^{BR} = J_{BR} - J_{0}
$$

| 分量 | per-seed | mean | sd(learner) | t | 3/3 同号 | 符号翻转 |
|---|---|---|---|---|---|---|
| $U^B$ **behavior** | [−194.6, −85.8, −118.0] | **−132.8** | 55.9 | **−4.11** | ✓ 全负 | 无 |
| $\Delta^{R\mid B}$ **replay** | [−31.2, −74.4, −178.9] | **−94.8** | 75.9 | −2.16 | ✓ 全负 | 无 |
| $U^{BR}$ 联合（Gate A） | [−225.8, −160.2, −296.9] | −227.6 | 68.4 | −5.77 | ✓ 全负 | 无 |

**恒等式 $U^{BR} = U^B + \Delta^{R\mid B}$ 的最大绝对误差 = 0.0**（数值一致性自检通过）。

原始 return（20k，source-free panel128）：

| seed | scratch | B-only | joint |
|---|---|---|---|
| 1 | 1094.6 | 900.0 | 868.8 |
| 2 | 1089.5 | 1003.7 | 929.3 |
| 3 | 1206.1 | 1088.1 | 909.2 |

---

## 3. 这排除了什么

预注册前 ChatGPT 列出的三种单侧情形**全部被排除**：

| 假想情形 | 含义 | 实测 |
|---|---|---|
| $J_B \approx J_{BR} \ll J_0$ | replay 无关紧要，behavior 主导 | ✗ replay 也贡献 −94.8 |
| $J_B \approx J_0 > J_{BR}$ | behavior 无害，replay 主导 | ✗ behavior 贡献 −132.8 |
| $J_B < J_{BR} < J_0$ | behavior 有害但 replay 在补偿 | ✗ replay 同向加害，不补偿 |

实际是第四种：$U^B<0$ **且** $\Delta^{R\mid B}<0$（均 3/3）。

**含义：没有一个单独修掉就能全部回收的病灶。** 屏蔽 source replay（即 B-only）
仍留下 $U^B=-132.8$；而在 behavior 已存在的条件下，当前 replay scheme 又再减 94.8。

> **§7.1 更正**：这里**不能**说成"两条通道各自独立有害"或给出份额比。
> 缺 $J_{01}$ 时无法做独立主效应分解，$\Delta^{R\mid B}$ 只是
> **conditional replay harm**，且份额依赖分解顺序、跨 seed 从 86:14 到 40:60。

---

## 4. 与 T1b 结合：behavior 侧的伤害机制不是梯度方向冲突

两个结果必须放在一起读：

- **T1b**：source-authority 状态与 student 状态给 actor 的改进方向**高度一致**
  （正确 estimand 下 cos = 0.55–0.87，无一负值）；
- **T2**：behavior 通道**确实有害**（−132.8，t=−4.11，3/3）。

所以 behavior 侧的伤害**不是**"source 把 actor 往错误方向拉"。
两者并存唯一自洽的读法是：伤害来自**机会成本**——source 拿走一半行为权，
student 自身的 on-policy 交互与探索机会随之减半——而不是来自方向性污染。

这是本轮最有信息量的一点，也是 T1b 那个"看似无用"的负结果真正的价值所在：
**它把 behavior 侧的候选机制从"方向冲突"缩到了"机会成本"。**

**注意**：这是对现有证据最自洽的解读，**不是已验证的结论**。
"机会成本"本身尚未被独立检验（例如与"同样减半 student 交互、但用随机策略或
student 自身旧策略代管"的对照相比）。不得当作已确立的机制写进论文。

---

## 5. 必须同时说明的边界

**5.1 replay 侧的证据弱于 behavior 侧。**
$\Delta^{R\mid B}$ 的 t = −2.16，**未达** df=2 单侧 0.05 的临界值 2.92。
预注册的判据是符号（3/3 < 0）而非 t 检验，故 `JOINT_HARM_CANDIDATE` 按判据成立；
但**不得**把"3/3 同号"表述为"显著"（CLAUDE.md §5：n=3 的 3/3 不足以定论）。

**5.2 通道的相对贡献跨 learner 很不稳定。**

| seed | behavior 占比 | replay 占比 |
|---|---|---|
| 1 | 86% | 14% |
| 2 | 54% | 46% |
| 3 | 40% | 60% |

符号在三个 seed 上一致（这已强于 Door 的先例——那里通道归因跨 seed **反向**），
但**幅度分配从 86:14 到 40:60**。因此可以说"两条通道都有害"，
**不能**说"behavior 是主因"。若要主张主次，需要更多 learner seed。

**5.3 只测了一个点。** truck、10k→20k 这一个 stage、mass 0.5、20k 单点。
结论不得外推到其他 target、其他注入时机、其他剂量，也不回答长期后果——
Gate A 的 truck 在 50k/100k 转正，但那个比较有 restart confound
（见 `pare_gate_a_posthoc_interpretation_20260808.md` §2），不能用来推断本轮通道的长期走向。

**5.4 `student_only` 不等于"source 数据从未存在"。** 它只把 source 的 replay
采样配额置 0；source transition 仍在 physical buffer 中（E3 实测 63.8 万条）。
故 $\Delta^{R\mid B}$ 度量的是"source 数据被学习批次采到"的边际效应，
不是"source 数据存在"的效应。

---

## 7. 表述更正（2026-08-08 追加，外部 review 后）

**冻结 verdict `JOINT_HARM_CANDIDATE` 与全部数值不变。** 以下两处表述收回。

### 7.1 "两条通道各自独立有害" —— 收回

本实验有三个 cell：

$$
J_{00}=\text{scratch},\quad
J_{10}=\text{source behavior} + \text{student-only replay},\quad
J_{11}=\text{joint}
$$

$$
U^B = J_{10}-J_{00},\qquad \Delta^{R\mid B} = J_{11}-J_{10}
$$

$U^{BR} = U^B + \Delta^{R\mid B}$ **只是代数恒等式**（我在 §2 把它当"一致性自检"是对的，
但不能用它论证独立性）。完整的交互项需要第四个 cell：

$$
I = J_{11} - J_{10} - J_{01} + J_{00}
$$

而 $J_{01}$（无 source behavior 但有 source replay）在自然实验中**无法构造**——
没有 source 执行动作，就不存在同一轨迹上的 source-generated target transition。

因此正确表述只能是：

- $U^B$：source behavior **在关闭 source replay 时**的效应；
- $\Delta^{R\mid B}$：**在 source behavior 已存在的条件下**，再启用当前 source replay
  scheme 的效应 —— 即 **conditional replay harm**，不是"replay 独立有害"。

### 7.2 "约六四开" —— 收回

sequential decomposition 的份额**依赖分解顺序**，且实测跨 seed 从 86:14 到 40:60。
它不是一个稳定的量，不应报告。§5.2 已列出这个不稳定性，但 §3 仍写了"六四开"，前后矛盾。

### 7.3 统计强度的准确表述

$|t|$：behavior 4.11、replay 2.16，$n=3$（$df=2$）。
双侧 5% 的临界值约 **4.303**，故**两者都未跨过常规显著性门槛**。
预注册用的是"3/3 同号"而非 t 检验，故 verdict 成立；
但**不得**写成"已证明 behavior 与 replay 都有害"。

128 个 evaluation episode 只提高每个 learner 的测量精度，**不把 $n=3$ 变成 $n=384$**。

### 7.4 保留不变的部分

工程 gate（E1 完全相等、E2 严格 0、E3 非 0、E4 完整）、全部原始数值、
恒等式误差 0.0、§4 关于"behavior 侧伤害不是方向性污染"的推论
（该推论只依赖 $U^B<0$ 与 T1b 的 cosine，不依赖独立性主张），
以及 §5 的全部边界声明。

---

## 6. 处置

按预注册，本轮**只作根因定位**，到此停止：

- 不开发新算法、selector、proxy、dose search、exploration mechanism；
- 不启动 early-vs-late timing 实验；
- PARE v1 与 PDAU 保持 CLOSED。

下一步方向的选择权交回 PI / 外部 review。就本轮证据：$U^B$ 与 $\Delta^{R\mid B}$
同为负意味着只修一条通道无法全部回收；而 timing 假设（stage-dependence）目前
仍只有非受控的历史对比支持（`phase1_mechanism_audit_20260719.md:87` 已明确记录
历史 +227.8 与中期负值"为不同 estimand，非受控窗口对比"）。

**后续（2026-08-08 追加）**：$\Delta^{R\mid B}$ 度量的**不是**"source transition
正常进入 replay"，而是"只占约 25% 物理 buffer 的 source transition 拿到约 50% 的
replay quota"。缺的关键 cell 是 $q_S=\rho_S$。见
`docs/experiments/dual_displacement_audit_prereg_20260808.md`。
