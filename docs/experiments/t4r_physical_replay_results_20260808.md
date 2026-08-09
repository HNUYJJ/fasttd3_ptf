# T4-R 结果：`ENGINEERING_GATE_FAILED` → 全部标 **DIAGNOSTIC**

日期：2026-08-08 · 判据：`t4r_physical_replay_prereg_20260808.md`（先于运行冻结）
数据：`docs/data/t4r_phys_v1/t4r_verdict.json`

> **本轮全部数值为 DIAGNOSTIC，不构成确认性证据。**
> 工程 gate 的 G3 按字面 FAIL，判决脚本据此拒绝出科学裁决（退出码 1）。
> G3 的失败根因是**我把对照值算错了**（§2），不是数据问题。
> 按 CLAUDE.md §4.1，本轮如实标 DIAGNOSTIC，**不修改 G3 使其通过**。

---

## 1. 工程 gate

| gate | s1 | s2 | s3 | 结果 |
|---|---|---|---|---|
| G1 behavior share（ρ_S vs T3 的 0.2493） | 0.249268 | 0.250088 | 0.249521 | **PASS** |
| G2 source physical transitions | 638,125 | 640,225 | 638,775 | **PASS** |
| G3 \|q_S − ρ_S(终点)\| ≤ 0.03 | 0.0959 | 0.0968 | 0.0959 | **FAIL** |
| G4 A ≤ 1.3 | 0.5451 | 0.5431 | 0.5459 | **PASS** |
| G5 provenance 完整 | ✓ | ✓ | ✓ | **PASS** |

`replay_physical_flag = true`（三 seed 均是），physical 路径确实生效。

---

## 2. G3 失败的根因：**`physical` 不给 A=1**（修正一个前提）

先排除实现 bug：`executed_group_mask.any()`、`behavior_source ≥ 0`、`options ≥ 0`
三个口径在 joint anchor 上**完全一致**（均为 638,125 = 0.249268），ρ_S 的定义无误。

真正原因：physical 在**每个时刻**均匀采样，但 `valid_n` 随训练增长——
早期 buffer 里只有 student prefix（source 占比 0），source 占比是逐渐爬升的。
故累计 $q_S$ 是 $\rho_S(t)$ 的**时间平均**，而非终点值：

$$
\bar\rho_S=\frac1u\int_H^{H+u}\frac{m(t-H)}{t}\,dt
= m\left[1-\frac Hu\ln\left(1+\frac uH\right)\right]
$$

代入 $m=0.5,\;H=u=10^4$：$\bar\rho_S = 0.5(1-\ln 2) = \mathbf{0.153426}$。

实测 $q_S$ = **0.153261 / 0.153339 / 0.153617** —— 吻合到小数点后 3–4 位，三 seed 全中。

**所以 physical 实现完全正确，是 G3 的对照值选错了**：我拿终点 $\rho_S$ 去比累计 $q_S$。

### 由此修正的前提

外部 review 假定 physical 是"自然基准"（$q_S=\rho_S \Rightarrow A=1$）。
在 late-entry 场景下**这不成立**：

| 模式 | $q_S$ | $A$ | 性质 |
|---|---|---|---|
| fixed quota | 0.4957 | **2.96** | 过采样 |
| physical uniform | 0.1533 | **0.545** | **欠采样** |
| "公平"（A=1） | — | 1 | 两者之间，**没有现成模式** |

source transition 出现得晚、在 buffer 中存在的时间短，所以在**累计**意义上，
均匀采样给它的曝光自然低于同期 student 数据。A=1 需要按存在时长加权，
当前代码里没有这个模式。

---

## 3. 性能（DIAGNOSTIC，不作确认性结论）

20k source-free panel128，三臂共享同一 A0 anchor 与同一 resume noise seed，
source behavior 完全相同。

| seed | scratch | B-only ($q_S=0$) | **phys** ($A$=0.55) | fixed ($A$=2.96) |
|---|---|---|---|---|
| 1 | 1094.6 | 900.0 | 926.0 | 868.8 |
| 2 | 1089.5 | 1003.7 | 947.5 | 929.3 |
| 3 | 1206.1 | 1088.1 | 1176.4 | 909.2 |

| 对照 | Δ | sd | t | 符号 |
|---|---|---|---|---|
| **phys − fixed（P1）** | **+114.2** | 133.9 | +1.48 | **3/3 正** |
| phys − bonly | +19.4 | 72.5 | +0.46 | **跨 seed 翻符号** |
| phys − scratch | −113.4 | 73.7 | −2.66 | 3/3 负 |
| fixed − scratch（Gate A） | −227.6 | 68.4 | −5.77 | 3/3 负 |

### 可以说的

- **P1 的方向成立（3/3 正）**：把 source 的 replay 曝光从 $A$=2.96 降到 0.55，
  相对 scratch 的伤害从 −227.6 收窄到 −113.4，**回收了约一半**。
- 但 $t$=1.48，$n$=3，**远不构成统计证据**；且工程 gate 未过。

### 不能说的

1. **不能说"过采样是伤害来源"已被证实。** phys 臂同时改变了两件事：
   消除过采样（$A$: 2.96→1）**和**引入欠采样（$A$: 1→0.55）。
   没有 $A$=1 的臂，这两者**无法区分**。P1 的 +114.2 可能来自其中任一。
2. **P2/P3 无法区分**：phys − bonly 跨 seed 翻符号（+26.0 / −56.2 / +88.3）。
   即 source replay 在"不放大"时既不显著帮助也不显著伤害。
3. **残余伤害仍在**：phys − scratch = −113.4（3/3 负），与 T2 的
   $U^B$ = −132.8 量级相当。**replay 侧的修正无法触及 behavior 侧的伤害**，
   这与 T2 的结论一致。

---

## 4. 处置建议（决定权交回 PI / 外部 review）

本轮**不**自行重跑、**不**改判据、**不**开发新机制。可选的下一步：

- **若要正式化 P1**：须新写预注册，用 $\bar\rho_S$ 作 G3 的对照值，并独立重跑。
  数据本身干净（三 seed 一致、与解析式吻合 4 位），但按 §4.1，
  在已知结果的同一批数据上换 gate 不能恢复确认性地位。
- **若要分离"消除过采样"与"欠采样"**：需要第四个臂，
  即按 transition 存在时长加权使 $A\approx1$。这是**新机制**，不在当前授权范围。
- **若认为 replay 侧上限已见**：phys − scratch = −113.4 说明即使完全修好 replay，
  behavior 侧仍有约一半伤害。可转向 behavior 侧——但 T3 Part B 已表明
  20k 处两侧 manipulation 事件率均为 0，需要换检验方式或换训练阶段。

---

## 5. 边界

1. 全部数值为 DIAGNOSTIC。
2. 只测 truck、10k→20k、mass 0.5、20k 单点。
3. $\bar\rho_S$ 的推导假设 buffer 未 wrap（实测 `buffer_not_wrapped=true`）
   且采样均匀覆盖全部有效槽位。
4. T3 的 $A$≈2.96 用的是**终点** $\rho_S$。对 fixed quota 而言 $q_S=m$ 恒定、
   与时间无关，故该口径下的"过采样"结论不受本节修正影响；
   但若要与 physical 并列比较，两者应统一到同一时间口径。
