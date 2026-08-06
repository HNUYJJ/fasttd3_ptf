# 预注册：自动选源的最小测量代价 K*（RACING_K v1）

> 2026-07-30。**本文必须在任何臂被评估之前提交。**
> 立项门（`RESEARCH_EXECUTION_GUARDRAILS_20260721.md` §3）五问答复见 §1。

## 1. 强制立项门

1. **对应哪个核心问题**：§2 核心问题 1——"当前阶段应选择哪个 source"。
2. **唯一主要假设**：`U` 在**短** horizon 上的估计 `U(K_small)` 能正确识别最佳源，
   即 `argmax_i U_i(K_small) = argmax_i U_i(K=30k)`，且 `K_small ≪ 30k`。
3. **正负各自的后果**：
   - 正 → 自动选源有了可执行方案，且代价被定量刻画（`3·K*` 步交互）；
     十一族的不可能性刻画从"只有负结果"变为"负结果 + 最小代价上界"。
   - 负 → 连直接测量都需要 `K ≈ 30k`，此时 racing 的成本接近直接训练，
     **自动选源在本设定下彻底关闭**，不可能性刻画升级为完整结论。
   两个方向都改变论文结论。
4. **是否重复已有实验**：不重复。十一族测的都是**代理量** `X`，假设 `X ~ U`；
   本实验直接测 `U` 本身，只缩小 `K`。`EQD30K` 已给出 `K=30k` 的答案（即本实验的
   ground truth），**从未测过 K 能压到多小**。
5. **最小成本可否证方案**：见 §4，12 条 10k 训练 + 36 点评估，约 1.5 机时。

## 2. 为什么这不是第十二次换皮

今日刚因"estimand 未变"关闭了 Competence-Gated Transfer（`76dcd16`）。
本设计与十一族的区别必须写清楚：

| | 测什么 | 需要的因果跳跃 |
|---|---|---|
| 族 1–11 | 代理量 `X`（行为回报 / 即时 reward / 梯度内积 / critic 优势 / reward 结构 / 静态规格） | `X → U`：**跨量类外推**，从未被证实 |
| 本实验 | `U(K_small)` = `J_sf(源臂 at K) − J_sf(student 臂 at K)` | `U(K_small) → U(K_large)`：**同一量的 horizon 一致性**，可直接测量 |

`U` 的定义与全项目一致：`J_sf` = source-free student 的确定性评估，配对到 learner seed。
本实验**不引入任何新信号**，只是把已有的测量协议缩短。

## 3. 辨别设计：把 racing 与"zero-shot 行为排序"分开

**风险**：`K` 很小时，`U(K)` 可能退化为源的 zero-shot 行为质量——那就是已被否定的族 1。

**辨别依据**（今日探针 `469c1fb` 与已有 A 级标签恰好给出一个反向对）：

```
zero-shot 行为排序（hurdle，32 ep，确定性）:  run 169.21 > stand 146.94 > walk  96.35
真实 U 排序（EQD30K，K=30k，endpoint）    :  run 379.66 > walk  104.89 > stand  51.28
                                              ^^^^^^^^^^ walk 与 stand 排序相反
```

因此：**若 racing 在某个 `K` 上排出 `walk > stand`，即证明它测到了 zero-shot 测不到的量。**
这是本实验区分"真测量"与"行为代理"的关键判据，独立于主判据。

**ground truth 强度分层（必须如实承认）**：
- `run` 为 top-1：**强**。run 3 seeds CI90 `[271.5, 487.9]`，walk 3 seeds CI90 `[75.5, 134.3]`，
  两区间不重叠。
- `walk > stand`：**弱**。`EQD30K.hurdle.stand` 是 `single_seed: true`（仅 seed 1，51.278），
  无 CI。故 `walk > stand` 只作**次判据**，不参与主裁决。

## 4. 协议（冻结）

```
target        h1hand-hurdle-v0
起点          t = 0（从头训练，不用 anchor）—— 与 EQD30K 同 protocol family
臂            4 条: source=run / source=walk / source=stand / student-only(对照)
seeds         1, 2, 3（四臂配对同 seed）
训练长度      10k（单条训练内在 2k / 5k / 10k 存 checkpoint）
K 取值        2000, 5000, 10000      （ground truth 为已有的 K=30000）
剂量          behavior 0.5 / replay 0.5，与 EQD30K 逐项相同
评估          source-free student, deterministic, 128 episodes
其余          NUM_ENVS=128 BATCH=32768 BUFFER=51200 NUM_UPDATES=2 COMPILE=0 AMP=1
              —— 与 hurdle_speedup_v1 相同
```

`U_i(K) = J_sf(源臂 i at K) − J_sf(student-only 臂 at K)`，**per-seed 配对**。

**剂量验收**：三条源臂的 behavior share 必须落在 `[0.48, 0.52]`，否则该 seed 作废重跑。

**为什么必须自跑 student-only 对照而不复用 `hurdle_speedup_v1` 的 scratch 臂**：
后者无 2k/5k checkpoint。虽然 10k 点可交叉核对（scratch@10k = 3.4/12.9/4.0），
但 2k/5k 无从复用，故四臂同批跑，保证 same checkout。

## 5. 判据（冻结）

**主判据 —— top-1 命中**：

```
K* = 最小的 K ∈ {2000, 5000, 10000} 使得 3/3 seed 满足 argmax_i U_i(K) = run
```

| 条件 | 裁决 |
|---|---|
| `K* ≤ 5000` | `RACING_CHEAP` —— 选源代价 ≤ 3×5k = 15k 步 |
| `K* = 10000` | `RACING_VIABLE` —— 代价 30k 步，仍低于 EQD30K 的 90k |
| 三个 K 都做不到 3/3 | `RACING_REFUTED` —— 直接测量在短 K 亦不可行，自动选源关闭 |

**次判据 —— 超越 zero-shot（不参与主裁决）**：
记录每个 `K` 上 `U_walk(K) > U_stand(K)` 的 seed 数。≥2/3 即视为"排出了 zero-shot 排反的那一对"。

**成本—收益核算（公式冻结，跑前定死）**：

```
racing 成本 = N_sources × K*                     （N=3；并行则墙钟 = K*）
选源收益   = steps_scratch(θ) − steps_source(θ)   （取自 hurdle_speedup_v1 已测曲线）
净收益     = 选源收益 − racing 成本
```

以 `hurdle_speedup_v1` 的 θ=300 为例，per-seed `steps_scratch` 中位数 91828、
`steps_source` 中位数约 25566，收益约 66k 步。若 `K*=5000`，成本 15k，净收益约 +51k 步。
**该核算只在主判据非 `RACING_REFUTED` 时才有意义。**

## 6. 预先声明的边界

1. **单 target、单一源集合**。hurdle + {run, walk, stand}。方向依赖是本项目反复确认的事实
   （door 上三个 loco 源一致有害 9/9），本实验**不能**推广到其他 target。
2. **ground truth 自身有限**。`K=30k` 的 `U` 是本项目自己测的，不是外部真值；
   且 stand 仅单 seed（§3）。
3. **`K*` 是本设定下的上界估计，不是理论最小值**。只测了三个 `K`，
   真实的最小可用 horizon 可能落在 `2000` 以下或 `K` 取值之间。
4. **不声称解决了跨任务选源**。即便 `RACING_CHEAP`，得到的也只是
   "在一个 target 上、用真实交互、以 15k 步代价选对了源"。

## 7. 不得做的事（防止本线重蹈覆辙）

- 裁决后**不得**调 `K` 的取值、阈值或 seed 数来抢救结论。
- **不得**在裁决前查看任何臂的评估结果。
- 若 `RACING_REFUTED`，**不得**改用代理量"补救"——那会退回十一族。
- 剂量不达标的 seed 一律作废重跑，不得带病纳入。
