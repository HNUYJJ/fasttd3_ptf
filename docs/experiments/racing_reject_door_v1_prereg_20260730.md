# 【已作废，未揭盲】预注册：racing 能否**正确拒绝**——door 负迁移场地（RACING_REJECT v1）

> **2026-07-30 作废。作废时我未查看任何 U 数值**（评估已完成 36/36，只做了不涉及数值的
> 结构校验）。作废依据为 Codex review 指出的设计缺陷，五条我已逐条独立核实为真。
>
> ## 作废原因（核实记录）
>
> **致命：主假设事实上不可证伪。** §4 的 sanity check 要求 K=10000 时 9/9 per-seed 全负，
> 否则整体作废；而 9/9 全负 ⇒ 每个 seed 的 `max_i U_i(10000) < 0` ⇒ `K*_reject ≤ 10000`
> **必然成立**；K=10000 不全负时又走 `VOID_SANITY_FAILED` 而非 `REJECT_REFUTED`。
> 因此 `REJECT_REFUTED` 分支在 sanity 通过时**逻辑上不可达**——
> 本设计只能回答"能否把拒绝提前到 K≤5000"，**不能**回答"racing 能否拒绝"。
> 这是设计缺陷，写预注册时就该发现。
>
> 另有四处已核实的实现缺陷：
> 1. 预注册 §4 要求的剂量验收**完全没有写进裁决脚本**（`grep` 零匹配），
>    脚本可在剂量不合格时给出正式 verdict（剂量我另行手工验收过，PASS，但这不是脚本保证）；
> 2. `VOID_SANITY_FAILED` 时仍无条件写出并打印 `per_K`/`K_star_reject`，
>    违反 §7"整体作废、不予裁决、不得只保留通过部分"；
> 3. 评估脚本 `set -euo pipefail` + `CKPT=$(ls ... | head -1)` 使 `[MISSING]` 分支不可达
>    （实测退出码 2），缺 checkpoint 时静默终止而非报告；
> 4. "配对面板 SE"用了 `sqrt(se_src² + se_stu²)`，忽略协方差——
>    而 `p0_evaluator.py` 的面板是逐位配对的（注释：「(seed, rank) → 唯一 reset seed；
>    面板冻结，分支间逐位相同」），正确做法是取逐 episode 差值的 SE。
>
> ## 已产出数据的地位
>
> 12 条训练 + 36 点评估**全部保留且未揭盲**：剂量 0.4980–0.5006（PASS）、
> 12/12 anchor 恢复、36/36 结构校验通过、sha256 两两不同。
> 因判据的重新设计发生在**查看任何结果之前**，该批数据可作为
> `RACING_REJECT v2` 的**第一批**，但 v2 必须另跑独立重复（M24）。
>
> ---
>
> 以下为原始 v1 全文，仅供追溯，**协议部分仍有效，判据部分不得执行**。


> 2026-07-30。**本文必须在任何臂被评估之前提交。**
> 前置：`RACING_K v1` 已裁决 `RACING_VIABLE`（K\*=10000，hurdle，`a744adb`）。
> 本实验回答它最大的边界：**只在一个 target、且那个 target 上存在好源**。

## 1. 强制立项门（`RESEARCH_EXECUTION_GUARDRAILS_20260721.md` §3）

1. **对应哪个核心问题**：§2 核心问题 1 的后半——"何时严格退化为 100% student"。
2. **唯一主要假设**：在**所有源都有害**的 target 上，racing 用 K 步交互能
   **正确拒绝全部源**（即测得 `max_i U_i(K) < 0`），且 K ≪ 完整训练长度。
3. **正负各自的后果**：
   - 正 → racing 不只是"选源"，还能"弃源"，构成完整的负迁移免疫机制；
     `RACING_K` 的适用范围从"有好源的任务"扩展到"任意任务"。
   - 负 → racing 只在已知有好源时可用，是**半个**机制；
     必须在论文中明确声明它不提供负迁移保护。
   两个方向都改变论文能声称的范围。
4. **是否重复已有实验**：不重复。`door_at10k_gate_v1` 测的是 **K=10000 的完整 U**
   （已发表：9/9 per-seed 全负）。本实验问的是**更短的 K 能否得出同样的拒绝决定**，
   并把 K=10000 作为内建 sanity check。
5. **最小成本可否证方案**：复用现成 anchor，12 条 10k 训练 + 36 点评估，约 1.5 机时。

## 2. 为什么这不是换皮

与 `RACING_K` 同理：**estimand 未变**。测的是
`U_i(K) = J_sf(源臂 i) − J_sf(student 臂)`，即目标量本身，只缩短 `K`。
不引入任何代理量，不做跨量类外推。

## 3. ground truth（已发表，`door_at10k_gate_v1_results_20260727.md`）

```
door@10k→20k, K=10000, per-seed U (s1/s2/s3):
  stand  −23.43 / −42.83 / −31.66    mean −32.64   [−49.05, −16.23]  harmful
  walk    −7.06 / −38.58 / −20.96    mean −22.20   [−48.83,  +4.43]  uncertain（最不负）
  run    −17.78 / −41.04 / −33.08    mean −30.63   [−50.56, −10.71]  harmful

|U| / 配对面板 SE 中位 ≈ 9（最小 1.92）—— 测量干净
walk 在 3/3 seed 上都优于 run 与 stand
```

**正确答案：拒绝全部三个源。** 若必须排序，则 `walk > run ≈ stand`。

### 3.1 辨别依据（比 hurdle 更强：完全反向）

```
行为层排序(Transfer Map v1):  run 101 (+58%) ≫ stand 59 (−8%) > walk 25 (−61%, 62% 摔)
学习效用排序(U, K=10k)     :  walk (−22.20) > run (−30.63) ≈ stand (−32.64)
```

**行为层最好的 run 是 harmful；行为层最差的 walk 最不负。** 二者排序完全相反。
因此若 racing 选出 walk 为"最不坏"，即再次证明它测的不是行为质量。

## 4. 协议（冻结）

**逐项复用 `scripts/run_door_at10k_gate_v1.sh`，唯一改动是增加中间 checkpoint。**

```
target        h1hand-door-v0
anchor        artifacts/door_at10k_gate_v1/anchors/s{1,2,3}  （现成，10k exact-abstention 纯 student）
臂            student / stand / walk / run   × seeds 1,2,3   （四臂配对同 seed）
noise 重采样  PTF_RESUME_NOISE_SEED = 91000 + seed           （与 gate 逐位一致）
剂量          behavior 0.5 / replay 0.5，h=25，bootstrap_only
K 取值        2000, 5000, 10000   → checkpoint 于 global_step 12000 / 15000 / 20000
唯一改动      PTF_EVAL_CHECKPOINT_STEPS: 20000 → 12000,15000,20000
评估          source-free student, deterministic, 128 episodes
其余          TOTAL_TIMESTEPS=100000 + PTF_RUN_STOP_STEP=20000，NUM_ENVS=128 等全部不变
```

`U_i(K) = J_sf(源臂 i at 10000+K) − J_sf(student 臂 at 10000+K)`，**per-seed 配对**。

**剂量验收**：源臂 behavior share ∈ `[0.48, 0.52]`（gate 实测 0.4978–0.5005），否则该 seed 作废重跑。

**内建 sanity check（必须先过）**：K=10000 的 per-seed U 应复现 §3 的 ground truth。
判据：9/9 per-seed 同号（全负），且三源的 mean U 与已发表值之差在
**配对面板 SE 的 3 倍以内**。若不通过，说明实现或环境有偏移，**本实验作废，不予裁决**。

## 5. 判据（冻结）

**主判据 —— 正确拒绝**：

```
K*_reject = 最小的 K ∈ {2000, 5000, 10000} 使 3/3 seed 满足  max_i U_i(K) < 0
```

| 条件 | 裁决 |
|---|---|
| `K*_reject ≤ 5000` | `REJECT_CHEAP` —— 拒绝代价 ≤ 3×5k = 15k 步 |
| `K*_reject = 10000` | `REJECT_VIABLE` —— 与 `RACING_K` 的选源代价同量级 |
| 三个 K 都做不到 | `REJECT_REFUTED` —— racing 不提供负迁移保护，只能用于已知有好源的任务 |

**次判据 —— 负迁移下的排序（不参与主裁决）**：
记录每个 K 上 `argmax_i U_i(K) = walk` 的 seed 数。≥2/3 视为"在全负场地仍排对最不坏的源"。

**避损核算（公式冻结）**：

```
拒绝收益 = |mean U|（避免的损失）≈ 22–33         （door，K=10000 口径）
拒绝成本 = 3 源 × K*_reject
```

与 hurdle 的加速收益不同口径，**不得合并比较**。

## 6. 预先声明的边界

1. **单 target、单源集合**。door + {stand, walk, run}。
2. **ground truth 是本项目自测的**，非外部真值；walk 的 CI 跨 0（uncertain）。
3. **`K*_reject` 是本设定下的上界**，只测三个 K。
4. **本实验不检验"有好源时选对源"**——那是 `RACING_K` 的职责。
   两者合起来才构成完整机制；单独任一个都不足以声称通用自动选源。
5. **不得与 hurdle 的成本—收益合并核算**（避损 vs 加速，口径不同）。

## 7. 不得做的事

- 裁决后不得调 K 取值、阈值或 seed 数抢救结论。
- 不得在裁决前查看任何臂的评估结果。
- 内建 sanity check 不通过则**整体作废**，不得只保留通过的部分。
- 若 `REJECT_REFUTED`，不得改用代理量补救——那会退回十一族。
