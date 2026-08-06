# per-state × per-body-group Q-switch 探针结果

> 2026-07-29。预注册 `afdc001`，**先于任何 Δ 被计算**。
>
> - **机械记录**：按预注册字面判据执行，输出 `PROBE_REFUTED`。该记录保留，不修改。
> - **科学裁决（2026-07-29 外部审查后更正）**：`INCONCLUSIVE / INVALID_FOR_DIRECTION_DECISION`。
>   **不得据此关闭 QMP / per-state 路线。**

## 0. 外部审查更正（2026-07-29，晚于 §1–§6 写作）

外部审查指出四条阻塞性缺陷，我逐条核实**全部成立**。原文 §1–§6 保留为原始记录，
但其方向性结论作废。

### 0.1 测错了时间点（阻塞性）

Door 的因果标签是 **10k 时决定投源、10k→20k 干预**。应被检验的是 **10k learner 的 critic**
能否正确路由。本脚本却硬编码 20k student checkpoint（`per_state_qswitch_probe_v1.py:47`）。

真实的 10k anchor 存在且完整——核实结果：

```
slide / stair / door  ×  s1,s2,s3   completed_vector_steps = 10000
files = ['learner.pt', 'replay.pt', 'rng.pt']    (learner 32.9MB, replay 1.98GB)
```

所以本探针实际回答的是"纯学生又训练 10k 步后，critic 怎样评价 source"，
**不是**"source 即将介入时，critic 能否正确路由"。estimand 不匹配。

### 0.2 负 margin 未按训练端 clamp

训练端 `select()` 会把 margin 截断到至少 `self.margin`（默认 0）：

```python
# fasttd3_ptf/ptf/mcg.py:153
m = margins.view(1, -1).clamp_min(self.margin)
```

本探针直接使用原始 null margin，实测**全为负**：

| seed | full margin | legs_torso | arms | hands |
|---|---:|---:|---:|---:|
| 1 | −0.1639 | +0.0268 | +0.0153 | +0.0036 |
| 2 | −0.2893 | −0.0154 | +0.0083 | −0.0463 |
| 3 | −0.1231 | +0.0017 | +0.0410 | +0.0490 |

用负阈值判"显著正"会放行部分 `Q_source < Q_student` 的动作。
本探针度量的是"优于错配动作 null"，**不是** QMP 的"候选策略 Q 最大"。

### 0.3 Šidák 校正缺少独立性基础

9 个 (source, group) 候选共享同一批状态、同一 student baseline、同一 critic，
且各候选动作与 student 动作共享其余维度，**高度相关**。按独立比较推导 Šidák 阈值无依据。

正确做法：直接构造"每次置换后在所有候选上取最大值"的**经验 max-null 分布**，
不需要任何独立性假设。§1 的 α_fw 与 q98.30 因此不成立，§4 的"校准后分化消失"
也不能按原口径解读。

### 0.4 MCG 不是"已实现的 QMP"

两处实质差异：

1. **候选动作性质**：QMP 在**完整策略**间选择，候选动作是某个策略的实际输出；
   MCG 拼接不同教师与学生的身体组动作（`mcg.py:80` 起），
   该混合动作**可能不属于任何策略的动作流形**，critic 在该点可能无数据支持。
2. **打分函数**：MCG 用 `min_h [ Q_h(a_cand) − Q_h(a_stu) ]`，
   它既不等于 QMP 的 soft policy value，也不等于 `min_h Q_h(a_cand) − min_h Q_h(a_stu)`。

因此 MCG 的准确定位是 **"QMP-inspired、带身体组混合与保守双 Q 的新 heuristic"**，
**不能借用 QMP Theorem 5.1 作为其安全保证**。
（QMP 的理论本身也建立在 tabular SAC、有限动作空间上，
对 HumanoidBench + FastTD3 的连续动作 / 深度逼近 / 分布式双 critic / 冻结跨任务源
只能作为**动机**。）

### 0.5 §3 的"critic 判断正确"不成立

原 §2/§3 用"20k 的 Δ_full 平均符号与 10k→20k 学习效用同向"推断 critic 判对了。
**该推断作废**：时间点与 estimand 均不匹配，符号一致只是描述性巧合。

### 0.6 更正后的裁决与后续

- `PROBE_REFUTED` 保留为机械记录；
- 科学结论为 `INCONCLUSIVE`，**不关闭** QMP / per-state 路线；
- **不执行**"同一脚本换 Slide 重跑"——0.1–0.4 的缺陷在 Slide 上同样存在；
- 唯一后续 = 重新设计的最小 **QMP-fidelity** 验证，
  见 `docs/run_card_qmp_fidelity_v1.md`；**先不引入身体组**。

---

> 以下 §1–§6 为原始记录，保留不改。其方向性结论已被 §0 作废。

## 1. 裁决

```
VERDICT: PROBE_REFUTED
Šidák: α_fw = 0.142625 | full q=0.9500 | group q=0.98305 (9 比较)
```

| seed | `frac_sig_group` | `frac_sig_full` | group > full？ |
|---|---:|---:|:--:|
| 1 | 0.1060 | 0.1165 | ✗ |
| 2 | 0.1006 | 0.1033 | ✗ |
| 3 | 0.1177 | 0.1240 | ✗ |

- 判据 1（全部 ≥ 0.30）：**False**——三个 seed 全部落在 α_fw = 0.1426 **以下**
- 判据 2（方向 3/3）：**False**——实际 **0/3**，group 每个 seed 都略**低于** full

每个身体组单独看（各组内 3 源取 max，名义假阳性率 5.0%）：

| seed | legs_torso | arms | hands |
|---|---:|---:|---:|
| 1 | 0.0373 | 0.0429 | 0.0448 |
| 2 | 0.0333 | 0.0469 | 0.0340 |
| 3 | 0.0496 | 0.0408 | 0.0463 |

**每一组的显著正比例都低于它自己的名义假阳性率。** 这是干净的"无信号"。

## 2. 次级观察：critic 的判断方向是对的

`Δ_full` 均值（全部为负）：

| seed | stand | walk | run | Q_student |
|---|---:|---:|---:|---:|
| 1 | −0.420 | −0.500 | −0.347 | 25.5 |
| 2 | −0.592 | −1.103 | −0.645 | 26.5 |
| 3 | −0.455 | −0.291 | −0.505 | 24.5 |

与 Door@10k gate 的学习效用 **9/9 per-seed 负**同向。预注册 §5 的一致性检查**通过**：
critic 的 per-state 判断没有与最终学习效用矛盾。

## 3. 判决力缺陷（必须披露）

**预注册选 door 的理由是"最不利场地上局部信号是否还存在"。这个理由有逻辑漏洞。**

door 上三个 loco 源的学习效用**已知确实为负**（9/9，测量干净）。
在源确实无用的场地上，一个**正确工作**的 Q-switch 必然给出"无信号"——
拒绝无用的源正是它该做的事。

因此本探针无法区分：

- **(A)** 机制无效：身体组粒度没有可用信号；
- **(B)** 机制有效：它正确地拒绝了确实无用的源。

而 §2 的证据（critic 判断与学习效用同向）反而更支持 (B)。

**这与 stair 被选作 BAC 判决场时"无判决力"是同一类错误**
（`feedback-stepwise-experiments` 的反面案例）。我在预注册里把"最不利"
误当成了"最强测试"。判据 2（group > full）还隐含假设了"full 有信号"，
当两侧都无信号时，方向比较退化为噪声比较，0/3 是**无信息的**。

**后果**：按预注册停止；`PROBE_REFUTED` 记录在案；
但本文**不主张**"per-state Q-switch 无效"，因为本实验没有测出这一点。

## 4. 方法论发现：未校准的 sign 口径会造出约 10× 的虚假分化

同一批数据，未做 null 校准的 sign 口径（`Δ > 0`）：

| seed | sign 版 group（max/9） | sign 版 full（max/3） | 比值 |
|---|---:|---:|---:|
| 1 | 0.1617 | 0.0152 | 10.6× |
| 2 | 0.0885 | 0.0054 | 16.4× |
| 3 | 0.3965 | 0.0331 | 12.0× |

**校准后这个分化完全消失**（0.106 vs 0.116）。

### 4.1 对 MCG 设计依据的追溯

`fasttd3_ptf/ptf/mcg.py` docstring 把下列 2026-06-11 push 探针结果列为身体组粒度的设计依据：

```
25k: reach-arms Δ≈0 / frac+=0.49  vs  reach-full Δ=−6.9 / frac+=0.05
```

核对原始数据 `logs/probe/modular_gating_push.json`（25k）：

| 教师 | legs_torso | arms | hands | full |
|---|---:|---:|---:|---:|
| reach | 0.116 / −3.72 | **0.490 / −0.12** | 0.148 / −3.45 | **0.048 / −6.90** |
| stand | 0.259 / −1.34 | 0.352 / −0.93 | 0.313 / −0.81 | 0.177 / −2.96 |
| walk | 0.220 / −1.71 | 0.409 / −0.49 | 0.444 / −1.48 | 0.250 / −3.55 |

两点必须说准确：

1. **这不是多重比较造成的**——push 探针是 per-(teacher, group) 的 **1v1** 比较，
   不取 max。（我在分析中一度推断为多重比较假象，核对原始数据后**收回**该推断。）
2. **真正的缺陷是缺少扰动尺度校准**：`frac+` 在不同扰动幅度间不可比。
   arms 只改 10/61 维，扰动小 → Δ 集中在 0 附近 → 约一半越过 0；
   full 改 61/61 维，扰动大 → Δ 显著负移。
   注意**所有 Δ 均值都是负的**，包括 arms 的 −0.12——
   `frac+=0.49` 描述的是一个**均值为负**的分布刚好有一半在 0 以上。

MCG v1.1 引入的 null margin（"源动作 × 打乱状态"）正是为消除这个尺度效应而设计的，
但 docstring 中作为设计依据的 part≫full 结论**是 v1 未校准口径下得出的，从未在校准口径下复核**。

**能说的**：未校准 sign 口径在 door 上直接产生了 10–16× 的虚假分化，该口径不足以支撑 part≫full。
**不能说的**：push 上的 part≫full 也是假的——不同任务、不同源、不同 checkpoint，
本实验没有测这个。要判定它需要在 push 上用校准口径重跑，而那是新实验。

## 5. 数据

```
预注册    docs/experiments/per_state_qswitch_probe_v1_prereg_20260729.md  (afdc001)
方向依据  docs/direction_per_state_qswitch_20260729.md
脚本      scripts/analysis/per_state_qswitch_probe_v1.py
结果      docs/data/per_state_qswitch_probe_v1/per_state_qswitch_probe_v1_results.json
student   models/h1hand-door-v0__door_at10k_student_s{1,2,3}__{1,2,3}_20000.pt
源        configs/source_banks/calibration/h1hand_door_rbo_{stand,walk,run}.yaml
```

## 6. 待人工裁定的决策（不自行启动）

按预注册，本线停止。以下属于**方向级决策**，交 PI 与外部审查裁定：

1. §3 的判决力缺陷是否构成"用有判决力的场地（源确实有用，如 slide/stair）重测一次"的
   正当理由——还是应视为变体抢救而彻底关闭本线；
2. 若重测，判决场应满足什么前置条件（源效用已知为正 + 标签可测性通过），
   以及如何避免"两侧都无信号 → 方向判据退化"的同类缺陷；
3. §4.1 的追溯是否需要在 `mcg.py` docstring 上加标注，避免该未校准结论继续作为设计依据。

**在裁定前不启动任何新实验，也不修改本次判据。**
