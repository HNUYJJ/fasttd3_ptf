# 【已关闭】设计草案：Competence-Gated Transfer —— 源何时该退出

> **2026-07-30 裁决：CLOSED，未实现、未跑任何臂。**
> 外部 review（Codex）判定为"第四次用行为回报代理延迟学习效用"，
> 我独立核实了它引用的三条证据，**全部为真**，因此接受裁决。
> 本文保留全文以记录推理链，但 §4 的协议**不得执行**。

## 0. 关闭理由（三条独立核实的证据）

**(a) 违反本项目自己的停止条款。** `docs/RESEARCH_EXECUTION_GUARDRAILS_20260721.md` §4 原文：

> 不再把 `T^0`、即时 source return 或 `T^critic` 符号重新包装成可靠迁移性指标。

本设计的 `d_t = clip(0.5·g(J_src/J_stu), 0, 0.5)` 正是"即时 source return"的比值形式。
该条款是我自己 2026-07-21 写的，**今日再次违反**（同日第二次 M15：先是合并
slide/stair 掩盖反例，再是不查自己的 guardrail）。

**(b) 已有直接失败先例，我完全漏查。** `adaptive_admission_v1`（已归档）用
局部即时 segment reward 决定源的撤销，在 truck 上撤掉了一个已被
`admission_handoff_v1` 证明有整体正迁移的 source bank（95k fix−scratch `+227.8`），
代价 `−119.7/−204.9`。且时序证据支持"价值在后续 update 中延迟实现"：
source 于 12k–21k 被撤，差值反而从 10–30k 的 `−62.4` 扩大到 35–95k 的 `−128.6`。
**CGT 与它的差别仅是 segment reward 换成 episode 回报比，estimand 未变。**

**(c) 同一 target 上"固定 vs 自适应"的对照已经做过，固定胜出。**
`classic_ptf_hurdle_single_source_v1`：

```
5k–95k nAUC:   fixed(λ:1→0/100k) − scratch = +100.993 (3/3)
               learned 自适应     − fixed   = −57.853  (1/3)
100k source-free:  scratch 644.3±45.6   fixed 634.4±89.8   learned 497.5±352.3
```

动作蒸馏权重的固定衰减在该实验中 3/3 有效、100k 未崩溃（634 ≈ scratch 644）；learned 自适应不仅更差，
seed 方差还放大约 4 倍。**CGT 是又一个 learned 自适应调度器。**
（该实验走动作蒸馏通道、配置不同，绝对值与 `hurdle_speedup_v1` 不可比；
可迁移的是"固定 vs 自适应"这个对照的方向，不能据此声称 bootstrap 行为/replay
剂量的固定衰减已经验证。）

## 0.1 保留的可执行结论

关闭方向不等于否定问题。`hurdle_speedup_v1` 的长程崩溃仍需修，
但下一项最低成本基线应是**预注册的固定 bootstrap schedule / hard exit**。
它在当前行为/replay bootstrap 通道上仍是未验证假设；若补齐，只能先称工程基线，
不得借动作蒸馏的 3/3 结果预先宣称有效，也不得包装成 learned 调度贡献。

## 0.2 教训

- "在线可观测"≠"观测到的就是决策量"。提出任何新信号前，必须先写出
  代理量与反事实学习效用之间缺失的因果箭头。
- 换时间尺度／换在线离线位置／换聚合方式，都不构成新信号族——**estimand 没变就是换皮**。
- 提新机制前先问："最简单的固定基线能不能解决？本项目是否已经测过？"

---

> 以下为 2026-07-30 的原始草案，**仅供记录推理链，协议不得执行**。

## 1. 触发这个设计的两个已确立事实

**事实 A（今日已裁决）**：`hurdle_speedup_v1` = `SPEEDUP_CONFIRMED`。
run 源给出早期 3.5–4.4× 样本效率提升（θ=200 中位数 4.38，θ=300 中位数 3.59，
各 3/3 seed per-seed ≥1.5），剂量实测 0.4994/0.4983/0.4995。

**但长程收益几乎全部流失**：

```
倍率:  20k=25.2×   30k=18.4×   50k=6.1×   75k=3.5×   100k=1.24×
回撤:  source s1 30k→50k −66%,  s2 75k→100k −84%
       scratch 三 seed 六个评估点全程零回撤
```

**事实 B（今日探针，`469c1fb`）**：源自身在 target 上很弱。

```
run 源 zero-shot on hurdle: 169.21 ± 17.18 (SE), 摔倒率 62%
student@20k = 241.72  已超过源
student@75k = 645.88  = 源的 3.82×
```

而 speedup 实验的配置是 `PTF_MCG_WARMUP_STEPS = 总步数`、
`EXPECTED_SOURCE_MASS = 0.5`，即**全程恒定 50% 剂量**。
交叉点在 ~15k，因此**约 85% 的训练时间里，一半的行为来自一个比 student 差
1.4–3.8 倍的策略**。早期加速、后期衰减、最终崩溃，三个现象在同一原因下自洽。

## 2. 先承认这不是新发现的部分

**恒定剂量在后期有害，本身不是新东西。** PTF 原文（Yang 2020）的 λ 线性衰减
就是为此设计的；本项目 2026-05-20 的 force-PTF 实验也已得出
"机制健康、恒定 λ 过度约束 actor"。我全程恒定是为了等剂量对照的干净性，
代价是长程崩溃。**修掉它属于修正实验配置，不构成贡献。**

因此本设计**必须**包含一个"PTF 原文式固定 schedule 衰减"臂作为基线。
如果它就足够好，结论就是"照抄原文即可"，本方向应当就此关闭。

## 3. 想主张的区分（这才是待检验的 insight）

本项目已有**十一个**迁移性预测信号族全部失败
（`docs/impossibility_characterization_of_transfer_prediction_20260730.md`）。
统一失败原因：它们都在估计一个只含 `(source, target)`（至多加 `t`）的点函数，
而真实的 `U ~ p(U | source, target, θ_t, D_t, occupancy_t, channel, dose, K)`。

**待检验的区分**：

| 问题 | 目标量是否可观测 | 本项目证据 |
|---|---|---|
| 预测一个**未知**源的延迟学习效用 | 否——需要外推到未发生的训练 | 十一族全败 |
| 判断一个**已在场**的源何时该退出 | **是**——student 与源都在环境里，相对能力可直接 rollout | 待检验 |

关键差别：退出判断**不需要预测任何未知量**。student 和源都在当期环境中，
各自的 target 回报是可以直接测的行为量。而十一族失败的是"从即时量外推延迟学习价值"——
这里不做外推，测的就是决策所需的量本身。

**这个区分在本项目此前的框架里是混在一起的**，因而所有信号族都被同一条否定意见覆盖。

## 4. 拟议协议（三臂，hurdle，100k，3 seeds，配对同 seed）

```
臂 A  恒定 50% 剂量          —— 已有数据(hurdle_speedup_v1)，不重跑
臂 B  固定 schedule 衰减     —— 基线：剂量 0.5 线性衰减到 0（PTF 原文式）
臂 C  competence-gated       —— 剂量由 student/源的实测相对能力决定
```

**臂 C 的测量方式（拟）**：在 128 个并行 env 中划出 8 个**探测 env**，
4 个纯 student、4 个纯源，不参与混合采样，只用于估计各自的 target 回报。
剂量 `d_t = clip(0.5 · g(J_src / J_stu), 0, 0.5)`，`g` 在 student 超过源后单调递减。
成本 8/128 = 6.25% 的 env。

**已知的设计漏洞（自列，请重点攻击）**：

1. **探测 env 的回报估计方差**。episode 长达 1000 步，4 个 env 给出的回报估计
   SE 可能远大于 student 与源的差距，导致门控在噪声上抖动。
   本项目已有 `SE_32 = 2·σ_panel(128)` 的标签可测性判据，但 4 个 env 更少。
2. **臂 C 相对臂 B 可能无增量**。若 student 超越源的时点在不同 seed 上高度一致
   （本实验三 seed 都在 ~15k），那固定 schedule 就已经足够，C 的自适应无处发挥。
   **这是本设计最可能的失败方式。**
3. **探测 env 改变了数据分布**。8 个 env 的数据是否进 replay？
   进则改变分布（与 A/B 不可比），不进则浪费 6.25% 的交互且需在剂量核算中扣除。
4. **"相对能力"仍是行为量**。本项目已反复确认"行为表现 ≠ 学习价值"
   （door 上 run 行为 +58% 却 harmful；今日探针中 zero-shot 排序 run>stand>walk
   与真实 U 排序 run>walk>stand 在 stand/walk 上互换）。
   我主张退出判断不受此约束（因为不外推），**但这正是需要被攻击的地方**。
5. **单任务**。hurdle 一个 target、run 一个源。方向依赖是本项目反复确认的事实。

## 5. 请 review 回答的问题

1. §3 的区分是否成立，还是"行为量预测学习价值"的换皮？特别是漏洞 4。
2. 若臂 B 就足够（漏洞 2），本方向是否应直接关闭？判据该怎么定？
3. 探测 env 的方差问题（漏洞 1）有没有更省的估计方式？
4. 该主张若成立，够不够作为论文的一个贡献点，还是只算工程修正？
5. 是否有更值得优先做的方向——例如直接做并行 racing 选源
   （`impossibility_characterization` §5 提到但尚未预注册）？

## 6. 相关出处

```
加速裁决        docs/experiments/hurdle_speedup_v1_results_20260730.md   (20f1e11)
源天花板探针     docs/data/hurdle_speedup_v1/source_ceiling_probe.json     (469c1fb)
十一族不可能性   docs/impossibility_characterization_of_transfer_prediction_20260730.md
force-PTF 旧证   记忆 project_force_ptf_run.md (2026-05-20)
行为≠学习价值    docs/experiments/door_gate_*  (M19)
```
