# PARE Experiment A 结果 —— `F1_TRIGGERED_CLOSE_PARE`

日期：2026-08-08 · 判据：`docs/experiments/pare_gate_a_prereg_20260808.md`（结果盲，先于评估冻结）
数据：`docs/data/pare_gate_a_v1/gate_a_verdict.json` + `source_free_eval/`（36 个评估点，全齐）

**结论：两个任务都判 `NO_SCAFFOLD_EFFECT`，触发 `docs/PARE_ALGORITHM_SPEC_v1.md` §8 的 F1。
按预注册，PARE 关闭。**

---

## 1. 判决

| 任务 | G1 early gain | G2 headroom | G3 非训练量不足 | 裁决 |
|---|---|---|---|---|
| stair | **FAIL** t=−0.33 | PASS (413) | STILL_IMPROVING (r=0.84) | `NO_SCAFFOLD_EFFECT` |
| truck | **FAIL** t=−5.77 | FAIL (1516.6) | PASS (r=0.16) | `NO_SCAFFOLD_EFFECT` |

G1 FAIL 优先于 G2/G3（预注册 §3 表的顺序），故两任务裁决均由 G1 决定。

`development_task = null`，`holdout_task = null`。

---

## 2. 完整配对差值（`exit − scratch`，learner 间方差，n=3）

两臂共享同一段 0–10k prefix，target interactions 相同，唯一差别是 10k–20k 有无 scaffold。
20k 点的 exit 状态即 scaffold run 的 20k checkpoint（release 点）。

### stair（source = slide，mass 0.5）

| 步 | exit | scratch | Δ | sd | t | per-seed Δ |
|---|---|---|---|---|---|---|
| 20k | 49.4 | 52.4 | −3.0 | 15.7 | −0.33 | [+14.8, −8.9, −14.9] |
| 50k | 247.0 | 329.0 | −81.9 | 127.1 | −1.12 | [−162.3, −148.2, +64.7] |
| 100k | 413.0 | 639.4 | **−226.4** | 123.3 | **−3.18** | [−336.6, −249.4, −93.2] |

**模式：早期无效应（符号都不一致）→ 后期显著有害。**

### truck（source = hurdle-enhanced bank，mass 0.5）

| 步 | exit | scratch | Δ | sd | t | per-seed Δ |
|---|---|---|---|---|---|---|
| 20k | 902.4 | 1130.0 | **−227.6** | 68.4 | **−5.77** | [−225.8, −160.2, −296.9] |
| 50k | 1429.6 | 1274.2 | +155.5 | 179.3 | +1.50 | [+281.4, +234.8, −49.8] |
| 100k | 1516.6 | 1368.1 | +148.5 | 131.3 | +1.96 | [+185.3, +257.4, +2.8] |

**模式：早期显著有害 → 后期转正但不显著（t=1.96 < 2.92）。**

---

## 3. 判据缺陷：G2 的 `theory_max = 1000` 对 truck 无效

预注册 §2 的 G2 写了"HumanoidBench 的 per-step reward 为若干 [0,1] 项相乘，
episode 长 1000 步，故 return 上界为 1000"。**这个论证只对 locomotion 类任务成立。**

核实 `humanoid_bench/envs/truck.py`：

```python
reward = 0
...  reward += 100        # 离散事件奖励，多处
...  reward -= 100
reward += upright * (reward_robot_package_truck + reward_robot_package_picked_up
                     + reward_package_table)      # 相加，不是相乘
if <success>: reward += 1000
success_bar = 3000
```

truck 的 reward 含 ±100 的离散事件项与完成时 +1000，**不是 [0,1] 项相乘**，
上界远超 1000（实测 J=1516.6，`success_bar` 本身就是 3000）。
这是我在写预注册时把 locomotion 的 reward 结构外推到 manipulation 造成的错误。

**处置：不修改该阈值。** 在同一批数据上把已知失败的门换成已知通过的门，
即使如实披露也不能恢复确认性地位（CLAUDE.md §8.7）。原判据的输出原样保留。

**为什么不影响裁决**（三条独立成立）：

1. truck 的裁决由 **G1 FAIL** 决定，G1 在预注册 §3 表中优先于 G2；
2. 即使用 `success_bar = 3000` 重算，G2 的判据是**绝对阈值 500**，
   而 J=1516.6 > 500，G2 依然 FAIL；
3. 即便假设 G2 改判 PASS，truck 仍因 G1 FAIL 落在 `NO_SCAFFOLD_EFFECT`。

G1 的证据本身很强：3/3 全负，t=−5.77，远超临界值 2.92。

---

## 4. 选场依据被 fresh 重跑推翻

选 stair 的唯一依据是旧实验 Slide→Stair 在 20k 点三个 seed 全正
（+21.91 / +13.23 / +11.05）。fresh 重跑得到 **[+14.8, −8.9, −14.9]**——连符号都不一致。

预注册 §2 与 spec §11 8.5 事先声明过"旧结果存在 source-dose confound，
只作选场理由、不作论文证据"，现在证实这条谨慎是必要的。
这也与仓库里已记录的"U 的符号会跨 learner 反转"一致。

---

## 5. 为什么这两个任务都不是 PARE 的场地

PARE 的前提是 source **早期有用、后期束缚**：先把 student 带进有用区域，
再把它困在 source-shaped occupancy 里。两个任务显示的都不是这个模式：

- **stair**：早期无效应（Δ=−3.0，符号不一致），后期显著有害（Δ=−226.4）。
  这是负迁移，不是"被 source 困住"。
- **truck**：早期显著有害（Δ=−227.6），后期转正但不显著。
  这是"早期害、后期益"，与 PARE 前提**恰好相反**。

**不得把"scaffold 有害"包装成"正好需要 PARE 来逃离"。**
PARE 解决的是"已经获得的能力把策略锁在 source 分布附近"，
前提是那份能力真实存在。两个任务上它都不存在。

另外 stair 的 G3 是 `STILL_IMPROVING`（r=0.84），说明它 100k 远未 plateau——
其 G2 的"headroom"相当一部分只是训练量不够，而不是被 source occupancy 困住
（spec §8 F6 的近亲情形）。

---

## 6. 处置

**按 spec §8 F1 关闭 PARE。**

准确的 scope：在**这两个任务、这一 scaffold 配置（10k 曝光、source mass 0.5）**下，
不存在 PARE 要解决的 post-release expansion 现象，因此没有可做判定的场地。
这**不**等于证明了"PARE 机制在任何条件下都无效"——它没有被测试过，
因为前提条件不成立。但按预注册，我们不去另找场地来让它成立
（那就是 site selection after seeing results）。

PARE 的实现、spec、smoke（`SMOKE_PASS`）与单元测试原样保留在仓库中，
不删除；但**不再推进 PARE 主实验、不做消融、不搜 scaffold 预算**。

### 值得单独留意的观察（不作为 PARE 的抢救理由）

truck 上出现了**延迟收益**：20k 显著落后 227.6，到 50k/100k 反超约 150
（3/3 为正但 t=1.96 不显著）。这个"先付代价、后回收"的形状本身有意思，
但它与 PARE 的问题陈述无关，且未达显著。
若日后要追这条线，须作为**新方向新预注册**，不得挂在 PARE 名下复用本批数据。
