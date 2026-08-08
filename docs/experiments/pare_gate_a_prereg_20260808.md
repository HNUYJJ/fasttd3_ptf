# PARE Experiment A —— 判决准则（结果盲，先于任何评估冻结）

冻结时间：2026-08-08 · 对应实现：`scripts/run_pare_gate_a_v1.sh`（commit 见本文件所在提交的父链）

**冻结时的状态声明**：stair 三条链的训练已跑完，**但我尚未查看任何 return / 评估数据**——
只读过 orchestrator 日志的 `START` / `DONE` 行。truck 尚未启动。
本文件提交之后才允许运行 `p0_evaluator.py`。

---

## 0. 这个实验回答什么

**只回答一个问题**：有没有值得 PARE 去解决的 residual problem？

即某任务是否**同时**表现出
(a) early scaffold gain —— source 的 10k 临时代管确实把 student 带到了更好的地方；
(b) post-exit residual headroom —— hard exit 之后仍有可观的未吃到的性能空间。

两个任务都不满足 → 不存在 post-release expansion 现象 →
**按 `docs/PARE_ALGORITHM_SPEC_v1.md` §8 F1 诚实关闭 PARE，不硬造算法。**

**这个实验不是论文结果**，不用于任何性能主张。

---

## 1. 数据来源与评估口径

四段结构（每 seed 一条串行链，source 曝光固定 10k）：

| 段 | 区间 | 配置 | 产出 |
|---|---|---|---|
| prefix | 0→10k | empty bank，纯 student | anchor A0 |
| scaffold | 10k→20k | 真 bank，`ADMISSION_MODE=all`，source mass 0.5 | anchor A1 + 20k ckpt |
| exit | 20k→100k | 从 A1，`ADMISSION_MODE=none` | 50k / 100k ckpt |
| scratch | 10k→100k | 从 A0，empty bank | 20k / 50k / 100k ckpt |

- **exit 臂的 20k 点就是 scaffold run 的 20k checkpoint**（release 点，二者同一状态）。
- exit 与 scratch 的 target interactions 相同（都到 100k），且共享同一段 0–10k prefix；
  唯一差别是 10k–20k 有无 scaffold。
- 评估：`scripts/p0_evaluator.py --eval-seeds panel128`，128 个 deterministic episode，
  source-free。**只用 `return`**。
  `success_count` 在 locomotion 上读的是 `terminated`（摔倒早停），与 return 强反向，
  本实验一律不引用（CLAUDE.md §6）。

---

## 2. 三道判据（阈值先于结果冻结）

记 $J_{\text{arm}, s}(t)$ 为臂 arm、seed $s$、步数 $t$ 的 128-episode 平均 return。

### G1 — early scaffold gain

配对差值 $\Delta_s = J_{\text{exit},s}(20\mathrm{k}) - J_{\text{scratch},s}(20\mathrm{k})$。

- **PASS**：3/3 同为正 **且** $t = \overline{\Delta} / (\mathrm{sd}(\Delta)/\sqrt3) > 2.92$
  （$df=2$ 单侧 $\alpha=0.05$）
- **WEAK_GAIN**：3/3 同为正但 $t \le 2.92$ —— 单列，**不自动进入 PARE**，交 PI 判断
- **FAIL**：其余

不确定度必须用 **learner 间方差**，不得用 episode 面板 SE（M16 / CLAUDE.md §5：
RACING_K 批 1 曾因用 episode-SE 而把 8.4–14.8 SE 的"无可争议"结论做成 1/3 复现）。

### G2 — residual headroom

$\overline{J_{\text{exit}}(100\mathrm{k})} < 500$，即理论上限的 50%。

理论上限论证：HumanoidBench 的 per-step reward 为若干 $[0,1]$ 项相乘，
episode 长 1000 步，故 return 上界为 1000。该论证独立于 `success_bar`。

> **诚实声明（CLAUDE.md §8.5）**：我在定这个阈值时**已经知道** stair 的历史水平极低
> （旧实验 20k 约 67），所以 G2 对 stair 近乎形式性通过，**对 stair 没有辨别力**。
> 真正有辨别力的是 G1 与 G3。truck 的绝对水平我未查，对 truck 而言 G2 是实质判据。

### G3 — 不被"训练量不够"解释

$$
r = \frac{J_{\text{exit}}(100\mathrm{k}) - J_{\text{exit}}(50\mathrm{k})}
{\max\bigl(J_{\text{exit}}(50\mathrm{k}) - J_{\text{exit}}(20\mathrm{k}),\ \varepsilon\bigr)}
$$

- **PASS**：$r < 0.5$ —— 后半程增量不到前半程一半，已明显放缓
- **STILL_IMPROVING**：$r \ge 0.5$ —— 仍在快速上升

$r \ge 0.5$ 时 spec §8 的 F6（`num_updates=2` vs `4`）挑战更强：
"plateau 只是更新不够"这个平凡解释尚未被排除，应先做该 falsification control
再谈 PARE。分母 $\le 0$（前半程没涨或倒退）时不取比值，直接标 `NON_MONOTONE` 并单列。

---

## 3. 任务级与全局裁决

| 条件 | 任务裁决 |
|---|---|
| G1 PASS ∧ G2 PASS ∧ G3 PASS | `PARE_CANDIDATE` |
| G1 FAIL | `NO_SCAFFOLD_EFFECT`（无 source 塑造的 occupancy 可言） |
| G2 FAIL | `SATURATED`（hard exit 已经做满） |
| G3 STILL_IMPROVING / NON_MONOTONE | `CONFOUNDED_BY_BUDGET` |
| G1 WEAK_GAIN 且 G2/G3 PASS | `WEAK_CANDIDATE`（交 PI，不自动推进） |
| 任一 seed 的任一评估点缺失 | `INCOMPLETE` |

**任何数据缺失一律 `INCOMPLETE` 且判决脚本非零退出**（CLAUDE.md §4）——
缺失路径绝不允许落进 `PARE_CANDIDATE` 或 `SATURATED` 这类实质裁决分支。
缺失统计独立扫描全部 (task, arm, seed, step) 组合，不因前置缺失而 `continue`。

**全局**：

- 两任务都不是 `PARE_CANDIDATE` / `WEAK_CANDIDATE` → **F1 触发，关闭 PARE**
- 恰好一个 candidate → 它作 development task
- 两个都是 candidate → 取 G1 的 $t$ 较大者作 development task，
  另一个**冻结为 holdout**，不得依其结果调 PARE（spec §11 8.5 / CLAUDE.md §8.7）

---

## 4. 冻结后不得更改的内容

- G1/G2/G3 的判据形式与阈值（2.92 / 500 / 0.5）
- "learner 间方差，不用 episode SE"
- 只用 return，不用 success_count
- 缺失即 INCOMPLETE 且非零退出

允许更改的只有**路径参数**（模型/输出目录）。
若判据在执行中被发现无法实现，须走 CLAUDE.md §4.1 的四步：
标 DIAGNOSTIC → hotfix 只含代码 → 另写新预注册（不改本文件）→ 独立重跑。
