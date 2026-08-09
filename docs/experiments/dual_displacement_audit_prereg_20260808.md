# T3 预注册：Dual Displacement Audit（零环境交互）

冻结时间：2026-08-08 · **先于任何 displacement 数值的计算**
只读已有的 Gate A / T2 20k replay 与 anchor，**不训练、不 rollout**。

---

## 0. 动机：一个来自代码结构的可解析机制

`ptf_replay.py` 在 source authority 活跃期间**不按物理占比**采样，
而是按 admission candidate mass 给每个 provenance stratum **固定 quota**：

$$
q_S = m \quad(\text{与 behavior mass 相同，与物理占比无关})
$$

仓库已实证过它的**退出侧**病理：source 退休后少量残留仍拿 50% quota，
造成 repetition divergence（Powerlift 80k collapse，促成 `physical_after_authority` 修复）。

本审计要检验它的**入口侧镜像问题**。

### 放大律（先于计算冻结）

设 learner 在 source 引入前已有 $H$ 步 student history，
之后 source 以 behavior mass $m$ 执行 $u$ 步，且 buffer 未覆盖历史数据。则物理占比

$$
\rho_S(u) = \frac{mu}{H+u}
$$

而 fixed quota 仍给 $q_S = m$。定义 source 相对 student 的
**per-transition replay amplification**

$$
A = \frac{q_S/\rho_S}{(1-q_S)/(1-\rho_S)}
$$

代入并化简（本文件独立复核过该推导）：

$$
A = \frac{q_S(1-\rho_S)}{\rho_S(1-q_S)}
  = \frac{m\left(1-\frac{mu}{H+u}\right)}{\frac{mu}{H+u}(1-m)}
  = \frac{H+u-mu}{u(1-m)}
  = \frac{H+u(1-m)}{u(1-m)}
$$

$$
\boxed{\;A(u) = 1 + \frac{H}{(1-m)\,u}\;}
$$

**推论**：$H=0 \Rightarrow A=1$（早期介入时 replay 曝光与数据生成比例自然匹配）；
$H>0 \Rightarrow A>1$（晚期介入时 fixed quota 系统性放大新 source transition）。

**这不是启发式，是从 replay sampling rule 直接推出的。** 无自由参数。

### 本实验的定量预测

Gate A / T2 的配置为 $H=10{,}000$、$m=0.5$、$u=10{,}000$，故

| 量 | 理论值 |
|---|---|
| $\rho_S$（20k 时的物理 source 占比） | **0.250** |
| $q_S$（critic source 采样占比） | **0.500** |
| $q_S/\rho_S$ | **2.00** |
| $A$ | **3.00** |

---

## 1. Part A —— Replay displacement

对 truck 与 stair 的每个 seed，从 20k replay 计算：

- $\rho_S$ = physical source slots / all valid physical slots
- $q_S$ = critic source samples / all critic samples
- $q_S/\rho_S$
- $A = \dfrac{q_S/\rho_S}{(1-q_S)/(1-\rho_S)}$

并按 `learner_step` 每 1k 重建 $\rho_S(t)$ 与由冻结公式推出的 $A(t)$，
与实测 endpoint 对照。**理论与实测的偏差如实报告，不调参去拟合。**

### 判据（先于计算冻结）

- **`AMPLIFICATION_CONFIRMED`**：truck 3/3 seed 满足 $A \ge 2.5$
  且 $|\rho_S - 0.25| \le 0.02$（即物理占比与理论一致，放大确实来自 quota 而非别的）
- **`AMPLIFICATION_REFUTED`**：truck 出现 $A < 1.5$ 的 seed
- 介于两者之间 → **`AMPLIFICATION_PARTIAL`**，单列，不自动进入 T4-R

$A \ge 2.5$ 这个门是相对理论值 3.0 留约 17% 余量定的，
**先于任何实测数值写下**；stair 只作对照，不参与判定。

---

## 2. Part B —— Behavior event yield

truck 的 reward 含离散事件项（已核实 `humanoid_bench/envs/truck.py`）：
`reward += 100`（package 抬起 / 放上 table）、`reward -= 100`（drop）、
成功另加 `reward += 1000`；其余为 `upright × (若干相加项)` 的连续小值。

故直接在 replay 的 reward 上定义（**不是新 proxy，是目标任务的真实离散事件**）：

$$
e^+_t = \mathbf{1}[r_t > 50],\quad
e^-_t = \mathbf{1}[r_t < -50],\quad
e^{\text{succ}}_t = \mathbf{1}[r_t > 900]
$$

按 provenance 分组统计 $P(e^+ \mid z=\text{source})$ 与 $P(e^+ \mid z=\text{student})$，
joint 与 B-only 三 seed 都算。同时输出 `executed_group_mask` 的
**arms overwritten fraction**（source 被选中时 arms 组被接管的比例）。

### 报告规则（先于计算冻结）

**只报告，不设"通过阈值"。** 具体报告：三个 seed 是否同向、幅度、
以及 event rate 的比值。**不得**据此直接实现 group selector，
也**不得**把 event rate 当作新的 transfer proxy。

---

## 3. 裁决与后续

- Part A 若 `AMPLIFICATION_CONFIRMED` → 允许进入 **T4-R**（仅一个 physical-replay arm）；
- 若 `AMPLIFICATION_REFUTED` → **停止 replay-lifecycle-entry 假设**；
- Part B 无论结果如何，**本轮都不启动 behavior 侧实验**（legs-only / group protection）。

---

## 4. 已知边界

1. $\rho_S$ 的理论式假设 **buffer 未覆盖历史数据**。
   本配置 buffer 容量 51200/env × 128 env 远大于 20k 步，故成立；脚本仍显式核对 `valid_size`。
2. $q_S$ 读自 `admission_sampling.sample_counts["critic"]`，是**累计**计数，
   覆盖 anchor resume 之后的全部采样。若 resume 导入了历史计数，需在报告中说明。
3. 本审计只解释 **replay 侧**的一个结构性偏差。即使 `AMPLIFICATION_CONFIRMED`，
   也**不**证明它就是 T2 中 $\Delta^{R\mid B}<0$ 的原因——那需要 T4-R 的
   $q_S=\rho_S$ 对照来检验。
4. Part B 的 event 统计基于 replay 中的 reward，反映的是**采集时**的事件产出，
   不等于策略的最终能力。
