# T2 结果：Truck 10k→20k 的 behavior / replay 因果分解 —— `JOINT_HARM_CANDIDATE`

日期：2026-08-08 · 判据：`truck_channel_decomposition_prereg_20260808.md`（先于运行冻结）
数据：`docs/data/truck_channel_v1/channel_verdict.json`

**结论：两条通道各自独立有害，没有单一病灶。** 本轮只作根因定位，非论文 confirmatory result。

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

实际是第四种：**两条通道各自独立有害，约六四开**（behavior 58% / replay 42%）。

**含义：不存在一个可以单独修掉的病灶。** 只改 replay 侧（无论是屏蔽 source 数据
还是给它加权）最多回收 42%，behavior 侧的 −132.8 原封不动；反之亦然。

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

## 6. 处置

按预注册，本轮**只作根因定位**，到此停止：

- 不开发新算法、selector、proxy、dose search、exploration mechanism；
- 不启动 early-vs-late timing 实验；
- PARE v1 与 PDAU 保持 CLOSED。

下一步方向的选择权交回 PI / 外部 review。就本轮证据，两条路都不是显然的：
"两条通道都有害且六四开"意味着单通道的修法上限有限，
而 timing 假设（stage-dependence）目前仍只有非受控的历史对比支持
（`phase1_mechanism_audit_20260719.md:87` 已明确记录历史 +227.8 与中期负值
"为不同 estimand，非受控窗口对比"）。
