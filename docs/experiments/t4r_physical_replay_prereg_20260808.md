# T4-R 预注册：physical-replay arm（$q_S = \rho_S$）

冻结时间：2026-08-08 · **先于任何 T4-R 运行**
前置：T3 `AMPLIFICATION_CONFIRMED`（实测 $A$ = 2.945–3.018，理论 3.0，6/6 偏差 < 2%）

---

## 0. 要回答的问题

T3 证明了**放大存在**：晚期介入的 source 只占 24.9% 的物理 buffer，
却拿走 49.6% 的 critic 采样配额，$A \approx 3$。

但 T3 **不能**证明这个放大就是 T2 中 $\Delta^{R\mid B}<0$ 的原因。
T2 实际比较的是

$$
q_S = 0 \;(\text{B-only}) \quad\text{vs}\quad q_S \approx 0.5 \;(\text{fixed quota})
$$

**中间缺了最重要的一格**：source transition 按其真实物理占比自然进入 replay，

$$
\boxed{\;q_S = \rho_S\;}
$$

T4-R 只补这一格。

---

## 1. 实现（已完成，先于本预注册提交）

`admission_replay_mode` 新增第三个取值 `physical`：

- `PTFReplayWrapper.set_admission_replay_physical(True)` 让 `draw_indices`
  走既有的 **physical-uniform over allowed slots** 路径
  （allowed 全 True 时退化为 FastTD3 原生 `randint`，与 legacy 逐位一致）；
- **behavior authority 完全不变**：source 照常按 mass 0.5 执行动作、照常写 physical buffer；
- rejected-source 槽位仍被精确排除，与退休路径一致；
- **无新 loss、无 threshold、无 schedule、无新超参数**。

即：把 replay authority 从 behavior authority 中解耦。
原本 `_admission_source_authority_active` 一个 flag 同时管两者。

## 2. 设计

三臂共享 Gate A 的**同一** A0 anchor（`truck_s{seed}_k10000`）与**同一**
`PTF_RESUME_NOISE_SEED = 92000+seed`，10k→20k，seeds = 1,2,3：

| 臂 | replay mode | $q_S$ | 状态 |
|---|---|---|---|
| R0 = B-only | `student_only` | 0 | **已有**（T2） |
| **Rphys** | `physical` | $\rho_S$ | 待跑 |
| Rfixed = joint | `shared` | ≈ 0.5 | **已有**（Gate A） |

三臂的 source behavior 完全相同。评估：20k `panel128` source-free，只用 return。
**不跑 50k/100k。**

## 3. 工程 gate（先于科学判定）

| # | 检查 | 通过条件 |
|---|---|---|
| G1 | behavior source share | 与 joint 一致（\|Δ\| ≤ 0.01） |
| G2 | source physical transitions | 非 0 |
| G3 | **critic source sample share 跟随物理占比** | $\lvert q_S - \rho_S\rvert \le 0.03$ |
| G4 | 放大比 | $A \le 1.3$（对比 fixed quota 的 ≈3.0） |
| G5 | provenance 完整 | `provenance_written` 全覆盖 valid slice |

G3/G4 是本臂的**定义性检查**：若 $q_S$ 仍≈0.5，说明 physical 模式没生效，本轮作废。

## 4. 关键预测（先于结果冻结）

记 $J_0$ = B-only、$J_p$ = physical、$J_f$ = fixed（joint）。

| # | 模式 | 判据 | 解释 |
|---|---|---|---|
| **P1** | $J_p > J_f$ 3/3 | `ENTRY_AMPLIFICATION_SUPPORTED` | 过采样确实是伤害来源 |
| **P2** | 且 $J_p > J_0$ | `SOURCE_DATA_VALUABLE` | **最强**：source 数据本身有学习价值，过采样把它翻成伤害 |
| **P3** | 且 $J_0 \ge J_p > J_f$ | `SOURCE_DATA_NEUTRAL_OR_NEGATIVE` | source replay 本身非正，但放大加重伤害 |
| **P4** | $J_p \approx J_f$ 或更差 | `ENTRY_AMPLIFICATION_REFUTED` | **否定** replay-amplification 作为性能根因，停止该线 |

$\approx$ 的操作化：$\lvert \overline{J_p} - \overline{J_f}\rvert$ 小于
$U^{BR}$ 的 learner 间 SE（39.5）即视为无差别。

主判据是 **P1 的 3/3 符号**（与 T2 同口径，不用 t 检验；$n=3$ 下
$t$ 只作报告，见 T2 §7.3）。跨 seed 翻符号 → `REPLAY_MODE_UNRESOLVED`，停。

任一 (arm, seed) 缺失 → `INCOMPLETE` 且非零退出。

## 5. 边界

1. 本轮只测 truck、10k→20k、mass 0.5、20k 单点。
2. 即使 P1/P2 成立，也只说明**在这个 stage 与剂量下**放大是伤害来源；
   $H=0$（早期介入）情形未测。
3. $q_S=\rho_S$ 是"不放大"，不是"最优"。本轮不搜索 $q_S$ 的最优值——
   那会退回被禁止的阈值搜索。
4. behavior 侧的 $U^B=-132.8$ 不在本轮范围内，physical replay 不可能修复它。
