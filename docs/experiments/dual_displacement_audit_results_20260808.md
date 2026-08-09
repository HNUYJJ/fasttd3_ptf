# T3 结果：Dual Displacement Audit —— `AMPLIFICATION_CONFIRMED`

日期：2026-08-08 · 判据：`dual_displacement_audit_prereg_20260808.md`（先于计算冻结）
数据：`docs/data/dual_displacement_v1/audit.json` · 零环境交互

---

## 1. Part A：replay amplification 与理论吻合

放大律（预注册 §0 冻结，无自由参数，直接由 replay sampling rule 推出）：

$$
\rho_S(u) = \frac{mu}{H+u},\qquad q_S = m,\qquad
A = \frac{q_S/\rho_S}{(1-q_S)/(1-\rho_S)} = 1 + \frac{H}{(1-m)u}
$$

Gate A / T2 的 $H=10{,}000$、$m=0.5$、$u=10{,}000$ ⟹ 理论 $\rho_S=0.25$、$q_S=0.5$、$A=3.0$。

| 任务 | seed | $\rho_S$ | $q_S$ | $q_S/\rho_S$ | $A$ | 与理论 $A$ 的偏差 |
|---|---|---|---|---|---|---|
| truck | 1 | 0.2493 | 0.4957 | 1.989 | **2.9605** | −1.32% |
| | 2 | 0.2501 | 0.4955 | 1.981 | **2.9450** | −1.83% |
| | 3 | 0.2495 | 0.4956 | 1.986 | **2.9555** | −1.48% |
| stair | 1 | 0.2482 | 0.4991 | 2.011 | 3.0178 | +0.59% |
| （对照） | 2 | 0.2494 | 0.4989 | 2.001 | 2.9966 | −0.11% |
| | 3 | 0.2488 | 0.4989 | 2.005 | 3.0059 | +0.20% |

**6/6 组合与理论偏差均 < 2%。** `verdict = AMPLIFICATION_CONFIRMED`
（判据：truck 3/3 满足 $A\ge2.5$ 且 $|\rho_S-0.25|\le0.02$，实测 ρ 偏差 ≤ 0.0018）。

前提核对：`buffer_not_wrapped = True`（valid 20000 < buffer 51200/env），
故 $\rho_S$ 的理论式成立。B-only 臂的 $q_S=0.0$，证实 `student_only` 生效
（$A$ 因除零未定义，符合预期）。

**stair 同样吻合**，说明这是 replay sampling rule 的**结构性性质，与任务无关**。

### 含义

在当前 fixed provenance quota 下，晚期介入的 source **每条 transition 获得的
gradient exposure 约为 student transition 的 3 倍**：source 只占 24.9% 的物理 buffer，
却拿走 49.6% 的 critic 采样配额。

$H=0$（source 从训练起点介入）时 $A=1$，replay 曝光与数据生成比例自然匹配。
这为"source utility 依赖 learner stage"提供了一个**结构性的、可解析的**来源，
而不是又一个 proxy 假设。

它与仓库已实证的**退出侧**病理（source 退休后残留数据仍拿固定 quota →
repetition divergence → Powerlift 80k collapse → `physical_after_authority` 修复）
构成对称的两端。

---

## 2. Part B：在 20k 点**无法检验**（不是证伪）

按预注册，用 truck reward 的真实离散事件定义
$e^+=\mathbf 1[r>50]$、$e^-=\mathbf 1[r<-50]$、$e^{\text{succ}}=\mathbf 1[r>900]$。

**实测：全部计数为 0 —— source 侧与 student 侧都是 0。**

先核实这不是读错了量：`reward_normalization = False`，replay 存的是 raw reward。
实测 20k 全窗口的 reward 分布：

| 统计量 | 值 |
|---|---|
| min / max | 0.0000 / **1.7977** |
| mean / std | 0.7651 / 0.2706 |
| 99.99% 分位 | 1.7078 |
| $\lvert r\rvert>3$ 的条数 | **0** |

即：**在整个 0–20k 期间，truck 从未发生过任何 ±100 的 package 事件。**
这与 20k 的 return ≈ 900–1200 一致（1000 步 × 平均 0.765 ≈ 765，
全部来自 `upright × (若干 [0,1] 项相加)` 的连续项）。

### 结论与更正

**"source 抢走 arm authority ⟹ manipulation 机会减少"这一假设在 20k 点无法用
事件产出检验**，因为 **student 自己的事件率也是 0**。这是 measurement-level 的
无法判定，**不是方向证伪**。

同时这修正了一个推测：20k 处 $U^B=-132.8$ 的伤害**不可能**来自"少抓了几次 package"——
那个阶段两边都没抓到过。差异只能来自**连续项**（`upright` 与接近程度），
即姿态维持与靠近 package 的能力。

若要检验 arm-authority displacement，须在**事件真正开始发生**的训练阶段测，
或改用连续项的分解（如 `upright` 与各 tolerance 项的分项均值）。**本轮不做。**

---

## 3. 附带确认：source 同时接管 legs_torso 与 arms

provenance groups = `[legs_torso, arms]`（truck 的 2 组配置）。实测：

| seed | group_0 (legs_torso) | group_1 (arms) |
|---|---|---|
| 1 | 0.249268 | **0.249268** |
| 2 | 0.250088 | **0.250088** |
| 3 | 0.249521 | **0.249521** |

两组占比**完全相同**（逐位相等），证实 `admission_bootstrap` 的
`self.current[...] = new.expand(num_groups)` 语义：**source 一旦被抽中，
就同时接管全部身体组，不存在按组独立抽取。**

故"source 被选中时接管 arms 的比例"= 100%；
换算到全 buffer 即 24.9% 的槽位、约 49.9% 的 scaffold 窗口步数中，
肩/肘/腕由 locomotion source 控制。

---

## 4. 裁决与后续

- Part A **`AMPLIFICATION_CONFIRMED`** → 按预注册 §3 允许进入 **T4-R**
  （仅新增一个 physical-replay arm，$q_S=\rho_S$）。
- Part B **无法检验** → 按预注册，**本轮不启动 behavior 侧实验**
  （legs-only / group protection），也不据此实现 group selector。

---

## 5. 边界（必须与结论同引）

1. **`AMPLIFICATION_CONFIRMED` 只证明放大存在，不证明它是 $\Delta^{R\mid B}<0$ 的原因。**
   因果检验需要 T4-R 的 $q_S=\rho_S$ 对照。
2. $q_S$ 读自累计 `sample_counts["critic"]`，覆盖 anchor resume 之后的全部采样；
   B-only 的 0 值反证了该计数确实只记录本 run 的采样，未混入历史。
3. Part B 的 0 事件率是 **20k 这一时点**的性质，不能外推到更晚的训练阶段。
4. 放大律假设 buffer 未覆盖历史数据；本配置已核对成立，
   若在更长训练或更小 buffer 下复用该式，须重新核对 wrap 情况。
