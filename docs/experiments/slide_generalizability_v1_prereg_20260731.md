# 预注册：slide 的标签可推广性审计（`M31` 的首次实操）

> 2026-07-31。**本文必须在 seeds 4–6 的任何评估结果产出之前提交。**
> 直接由 `M31` 导出：door 上 `U` 的符号跨 learner 显著反转，
> 故任何 target 在投入选源实验之前，必须先验证其标签的**可推广性**。

## 1. 强制立项门

1. **对应哪个核心问题**：§2 核心问题 1 的前置条件——若 `U` 的符号本身不跨 learner 稳定，
   则"该选哪个 source"在该 target 上无良定义答案。
2. **唯一主要假设**：slide 上 `argmax_i U_i` 在**新 learner** 上仍为 `walk`。
3. **正负各自的后果**：
   - 正 → slide 是合格的第二个 target，且与 hurdle 构成**真正的 crossover**（§3）；
     可据此投入完整 racing 实验。
   - 负 → slide 与 door 同类（标签不可推广），**不得**投入 racing；
     并且这将是第二个证据，说明可推广性缺陷不是 door 独有——
     那会显著提高 `M31` 的分量。
   两个方向都改变下一步投入。
4. **是否重复已有实验**：不重复。`slide_bac_gate_v1` 在 seeds 1–3 上测得
   `U_walk/U_run/U_stand = +56.95/+16.90/−1.21`；本实验问的是**换一批 learner 是否还成立**。
5. **最小成本方案**：复用 BAC gate 协议，新建 3 个 anchor + 9 条 10k 训练 + 12 点评估。

## 2. `M31` 为什么要求这一步

```
door：gate s1–3 负 9/9、holdout s4–6 负 9/9  →  newbatch s7–9 正 2/9
      其中 s9 的 run = +36.32 ± 3.95 显著正（gate 结论为 −30.63）
```

**18 个 per-seed 效应全负，仍不足以保证第 19 个为负。**
现有全部 A 级标签（`EQD30K`/`sibling gate`/`BAC gate`）都只在 3 个 learner 上测过。

## 3. 为什么选 slide：它与 hurdle 构成真正的 crossover

`RACING_MULTI` 被判 FATAL 的首要原因是"两个 target 的候选集合不同，
一个全局固定排序即可解释"。**slide 与 hurdle 没有这个问题**：

```
候选集合完全相同：{stand, walk, run}

hurdle (EQD30K)：run(+379.66) > walk(+104.89) > stand(+51.28)     argmax = run
slide  (BAC)   ：walk(+56.95) > run(+16.90)  > stand(−1.21)       argmax = walk
```

**walk 与 run 在同一候选集合上换位。** 任何"全局固定源排序"都无法同时满足两者。
这是本项目首次具备该条件——但**它的前提是两个 target 的标签都可推广**，
hurdle 已有 6 个 learner 的间接支持（`RACING_K` 两批各 3/3 选中 run），slide 尚无。

## 4. 协议（冻结）

逐项复用 `scripts/run_slide_bac_gate_v1.sh`，只换 seeds。

```
target        h1hand-slide-v0
既有          seeds 1–3（BAC gate 已发表，作为参照，不重跑）
本轮          seeds 4–6，anchor 新建（10k exact-abstention 纯 student，协议同 s1–3）
臂            student / stand / walk / run（四臂配对同 seed）
noise 重采样  PTF_RESUME_NOISE_SEED = 91000 + seed
剂量          behavior 0.5 / replay 0.5，h=25，bootstrap_only
K             10000（→ global_step 20000）；**不需要中间 checkpoint**
评估          source-free student, deterministic, 128 episodes（16 eval seeds × 8 ranks）
```

**剂量验收**：源臂 behavior share ∈ `[0.45, 0.55]`（沿用 BAC gate 的带）。
**同 target 内各源臂的 share 差须 ≤ 5 个百分点**（`M26`：剂量与源身份不得共变）。

## 5. 判据（冻结）

`U_i = J_sf(源臂 i at 20000) − J_sf(student 臂 at 20000)`，per-seed 配对。

**主判据**：

```
GEN_OK       ⟺  seeds 4–6 中 3/3 满足 argmax_i U_i = walk
GEN_PARTIAL  ⟺  2/3 满足
GEN_FAILED   ⟺  ≤1/3 满足
```

**次判据（记录，不参与主裁决）**：`walk` 与 `run` 的 `U` 是否仍 3/3 为正；
`stand` 的符号（s1–3 已是 2 负 1 正，本就 `uncertain`，**不作要求**）。

**偶然通过率**：三选一，零效应下单 seed 为 `1/3`，3/3 为 `(1/3)³ = 1/27 ≈ 3.7%`。
本实验只有一个 K、一个判据，**无 look-elsewhere 修正**。

**前置门**：层1 工程硬检查（剂量带 + 臂间 share 差 + `source_names` + student 盲化 +
anchor 恢复行 + `episode_count==128` + `identity_checked` + 冻结面板 + sha256 + 有限性）。
任一不过 → `VOID_ENGINEERING`，不输出任何 `U`。

## 6. `CLAUDE.md` §8 设计层自查

- **8.1 辨别力**：平凡解释="walk 在所有 target 上都最好"。**已被 hurdle 排除**——
  hurdle 上 argmax 是 run 而非 walk（`RACING_K` 6/6）。另一平凡解释="总选 U 最大者"，
  在本实验中 argmax 就是判据本身，不构成额外捷径。
- **8.2 混淆**：剂量带 + **臂间 share 差 ≤5pp** 的显式检查（`M26`）。
- **8.3 独立重复**：本实验**本身就是**对 s1–3 的独立重复（新 seeds、新 anchor）。
- **8.4 前提蕴含**：前置门只检查工程量，不含任何 `U`；`GEN_FAILED` 真实可达。
- **8.5 site selection**：slide 是已知 walk 胜出后选的，故本实验只能回答
  "该已知结论是否推广"，**不能**声称对任意 target 成立。
- **8.6 是否重演本轮教训**：M25（候选集合相同 ✓）、M26（share 差已查）、
  M27（不做跨批数值比对，只比符号与 argmax）、M28（判据按 slide 实际输出定）、
  M29（无事后归纳）、M30（判据先于新数据冻结，不复用旧数据改门）。
- **8.7 判据切换红线**：本实验是**新数据 + 新判据**，不涉及在旧数据上换门。

## 7. 能与不能声称

**能**（若 `GEN_OK`）：slide 上 `argmax = walk` 这一结论在 6 个 learner 上成立；
slide 可作为 racing 的候选 target；与 hurdle 合起来具备 crossover 条件。

**不得**：不得称"slide 的标签完全可推广"（`stand` 的符号本就不稳）；
不得称 6 个 learner 足以保证推广（door 用了 18 个仍翻转）；
不得据此直接声称 racing 在 slide 上有效——那需要单独的 racing 实验。

## 8. 不得做的事

- 裁决后不得调判据、seed 数或剂量带。
- 若 `GEN_FAILED`，**不得**改用其他判据抢救；应如实报告并将 slide 一并排除。
- 在本文冻结前不得查看 seeds 4–6 的任何 `U`。
