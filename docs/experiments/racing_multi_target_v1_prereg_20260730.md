# 【已作废，未执行】预注册：RACING_MULTI —— racing 的选源正确性能否跨 target 成立

> **2026-07-30 作废，未跑任何臂。** Codex 决策前 review 判定
> "Do not run; redesign is needed"，三条 FATAL 我逐条核实成立。
>
> ## 作废原因
>
> **F1（致命）· 我的"辨别设计"根本不成立。** 我以为"两个 target 的正确答案不同"
> 能排除平凡解释，但**候选集合本身就不同**：stair 是 `{walk, slide源}`，
> slide 是 `{walk, stair源}`。一个**全局固定的源质量排序** `slide > walk > stair`
> 就能同时通过两边——在 `{walk,slide}` 中选 slide，在 `{walk,stair}` 中选 walk。
> 证据只支持"stair 上 slide>walk"与"slide 上 walk>stair"，
> 这与单一全局排序完全兼容，**不是排序反转**。
> 我把"答案不同"误当成了"crossover"。Codex 另指出"选 |U| 更大者"这一捷径同样能通过。
>
> **F2（致命）· 剂量混淆我提了却没解决。** sibling 臂的 behavior share 系统性高
> 2.4–3.3 个百分点（stair `0.4956–0.4983` vs walk `0.4651–0.4733`），
> 而 stair 上 sibling 的优势（`+11.05`~`+21.91`）与剂量优势**同向**，
> 无法区分"源更有用"与"源被用得更多"。我在 §4 只是沿用了宽容差，
> 这等于把混淆写进协议。正确做法是按步/配额强制**匹配实际剂量**，
> 并在该控制器下重建 K=10000 参照。
>
> **F3（致命）· 无独立 learner 重复。** 只用一批 seeds `{1,2,3}`——
> 而这正是我自己当天写进 `M24` 的教训（RACING_K 批1 3/3、批2 1/3），
> `RACING_REJECT v2` 已据此改用新 seeds `{4,5,6}`，本设计又忘了。
>
> 另有两条：K=10000 的复制检查未显式要求**排序**复现（各臂容差都过而源间排序反转仍可能）；
> `argmax` 的**并列**未定义处理规则。
>
> ## 唯一通过的一项
>
> Q1（可证伪性）**OK**：把 K=10000 移出主终点确实切断了 v1 式的"前提蕴含结论"，
> `MULTI_REFUTED` 真实可达。这说明 v1 的教训被正确吸收了，问题出在别处。
>
> ## 由此得到的正确方向（记录，尚未预注册）
>
> 真正的 crossover 需要**同一候选集合**在两个 target 上**赢家反转**。
> 本项目已有这样一对：候选集合 `{stand, walk, run}`——
>
> ```
> hurdle：run(+379.66) > walk(+104.89) > stand(+51.28)      argmax = run
> door  ：walk(−22.20) > run(−30.63) > stand(−32.64)        argmax = walk
> ```
>
> 同一集合、argmax 反转（run ↔ walk），**全局固定排序无法解释**；
> 且两者剂量带都是严格的 `[0.48,0.52]`，无 sibling gate 那样的系统性偏差。
>
> **但不得直接把它当作预注册检验**：hurdle 一侧的结果我已经看过
> （`RACING_K` 已裁决），只有 door 一侧仍是盲的。故在 `RACING_REJECT v2` 的结果中，
> 「door 的 argmax 是否为 walk」只能作为**探索性观察**报告，
> 并显式标注其非预注册地位。严格的 crossover 检验需要新预注册 + 新 learner seeds +
> 匹配剂量控制器。
>
> ---
>
> 以下为原始草案全文，仅供追溯，**不得执行**。


> 2026-07-30。**本文必须在任何臂被评估之前提交。**
> 直接回应 `RACING_K` 最大的边界：**只在一个 target（hurdle）、且那个 target 上恰好存在好源**。

## 1. 强制立项门（`RESEARCH_EXECUTION_GUARDRAILS_20260721.md` §3）

1. **对应哪个核心问题**：§2 核心问题 1——"当前阶段应选择哪个 source"。
2. **唯一主要假设**：racing 用 `K ≤ 5000` 步交互测得的 `argmax_i U_i(K)`，
   在**两个正确答案互不相同**的 target 上都与 ground truth 一致。
3. **正负各自的后果**：
   - 正 → racing 的选源正确性不是 hurdle 特有；连同 `RACING_REJECT`（door，全负场地）
     构成"选优 / 弃源"双向覆盖的跨任务证据。
   - 负 → racing 的选源正确性**依赖 target**，`RACING_K` 的结论不可外推，
     必须在论文中降级为单任务案例。
   两个方向都改变论文能声称的范围。
4. **是否重复已有实验**：不重复。`sibling_source_gate_v1` 测的是 **K=10000 的完整 U**
   （已发表，见 §3）；本实验问的是**更短的 K 能否得出同样的选择**。
5. **最小成本可否证方案**：复用现成 10k anchor，18 条 10k 训练 + 54 点评估。

## 2. 为什么这不是换皮

与 `RACING_K` / `RACING_REJECT` 同理：**estimand 未变**。测的是
`U_i(K) = J_sf(源臂 i) − J_sf(student 臂)` 本身，只缩短 `K`，不引入代理量。

## 3. ground truth（已发表，`sibling_source_gate_v1_results_20260729.md`）

per-seed 回报（K=10000，128 ep source-free 面板）：

```
stair target                                   slide target
seed  student   walk   slide源                 seed  student   walk    stair源
 1     44.77   45.11    67.02                   1     39.38   105.04    90.68
 2     45.86   42.69    55.92                   2     63.61   111.22    84.02
 3     42.28   45.67    56.72                   3     50.54   108.11    87.30

U_slide源 = +22.25/+10.06/+14.44                U_walk    = +65.66/+47.61/+57.57
U_walk    =  +0.34/ −3.17/ +3.39                U_stair源 = +51.30/+20.41/+36.76
D_sib = +15.40  CI90 [+5.72,+25.08]  3/3        D_sib = −20.79  CI90 [−31.61,−9.97]  0/3
```

| target | 候选源 | **正确答案** | ground truth 强度 |
|---|---|---|---|
| `h1hand-stair-v0` | walk, slide源 | **slide源** | 强：3/3 per-seed，CI 不跨 0 |
| `h1hand-slide-v0` | walk, stair源 | **walk** | 强：3/3 per-seed，CI 不跨 0 |

**两个 target 的正确答案互不相同**。这是本设计的核心辨别力：
"总是选 walk"会在 stair 上失败，"总是选 sibling 源"会在 slide 上失败。
且 stair 上 walk 的 U 仅 `+0.34/−3.17/+3.39`（近零），要求 racing 具备真实分辨力。

**与 door 互补**：door 三源全负（正确决策 = REJECT），此处两源全正（正确决策 = USE 某一个）。
两者合起来覆盖 racing 的双向决策。

## 4. 协议（冻结）

逐项复用 `sibling_source_gate_v1` 的 BAC-gate 协议，**唯一改动是增加中间 checkpoint**。

```
targets       h1hand-stair-v0        /  h1hand-slide-v0
anchor        artifacts/stair_bac_gate_v1/anchors/s{1,2,3}  （现成，10k）
              artifacts/slide_bac_gate_v1/anchors/s{1,2,3}  （现成，10k）
臂            stair: student / walk / slide源     （3 臂）
              slide: student / walk / stair源     （3 臂）
seeds         1, 2, 3（各臂配对同 seed）
noise 重采样  PTF_RESUME_NOISE_SEED = 91000 + seed
剂量          behavior 0.5 / replay 0.5，h=25，bootstrap_only
K 取值        2000, 5000, 10000  → checkpoint 于 global_step 12000 / 15000 / 20000
评估          source-free student, deterministic, 128 episodes（16 eval seeds × 8 ranks）
冻结源        checkpoints/terrain_sources/h1hand_{slide,stair}/manifest.json
```

**剂量验收**：源臂 behavior share ∈ `[0.45, 0.55]`（沿用 BAC gate 的带，
该带比 door/hurdle 的 `[0.48,0.52]` 宽，因 sibling gate 实测 walk 臂为 0.4651–0.4789）。
**同 target 内 sibling 臂与 walk 臂的 share 差须 ≤ 5 个百分点**——
sibling gate 已报告 sibling 臂系统性高 2.4–3.3%，此处如实沿用其容差。

## 5. 判据（冻结）

**主终点（可证伪）**：

```
correct(K, target, seed)  ⟺  argmax_i U_i(K) == ground_truth_argmax(target)
H：存在 K ∈ {2000, 5000}，使 correct 在 stair 3/3 且 slide 3/3 上成立
```

**K=10000 不参与主终点**，只作复制检查（§6）。

| 结果 | 裁决 |
|---|---|
| K=5000 在两个 target 各 3/3 | `MULTI_CONFIRMED` —— 选源正确性跨 target 成立，代价 ≤ 2 源 × 5k |
| 仅 K=2000 两 target 各 3/3，K=5000 失败 | `MULTI_NONMONOTONIC` —— 不得称"稳定"，须报为随 horizon 非单调 |
| 二者均未达标，且 K=10000 两 target 各 3/3 | `MULTI_REFUTED` —— **有意义的负结果**：短 K 的选源正确性不跨 target |
| K=10000 也未达标 | `REPLICATION_DIVERGED`（§6） |

**分 target 报告（不参与主裁决）**：若只有一个 target 达标，须明确报出是哪一个，
并按 §7 措辞约束表述——**不得**以"1/2 target 成立"包装成正结果。

**偶然通过率**：每个 (target, seed) 二选一，零效应下 `1/2`；
6 个 (target, seed) 全中 = `(1/2)⁶ = 1/64 ≈ 1.56%`；
计入 K ∈ {2000,5000} 的 look-elsewhere（×2）→ **保守上界 ≈ 3.1%**。
与 `RACING_REJECT v2` 同口径。

## 6. 三层验收（与 `RACING_REJECT v2` 同构）

**层1 · 工程硬检查**（任一不过 → `VOID_ENGINEERING`，不输出任何主结果）：
剂量带、`source_names` 精确匹配、student 臂盲化、训练日志 anchor 恢复行、
eval json 的 `sha256` 与实际 checkpoint 一致、`identity_checked is True`、
`len(episodes)==128`、episode seed 序列 == 冻结面板、`protocol.deterministic`、
`env_name`、全部 return 有限。

**层2 · 复制检查**（K=10000 与 sibling gate 已发表 per-seed 值比对）：
逐 seed，`|U_new − U_gate| ≤ 3 × paired_se`（paired_se = 逐 episode 差值序列的 SE）
且符号一致。12/12 通过 → `REPLICATION_OK`，否则 `REPLICATION_DIVERGED`。

**异常分类**：缺产物 → `INCOMPLETE`（优先）；产物存在但无效 → `VOID_ENGINEERING`；
未预期异常 → `VOID_ENGINEERING`，均 `exit 2` 且不输出主结果。

**优先级**：`VOID_ENGINEERING > REPLICATION_DIVERGED > 主终点`。

## 7. 盲态封闭

两个 target 的训练、评估、层1 全部完成前，不得读取或输出任何 U / return / 排序 / 命中数。
**允许**对 return 的有效性判定（类型、有限性，只提取 1 bit）；
**禁止**控制流依赖其数值大小、排序、聚合或阈值比较。
人类不得查看单臂评估 `.log`（内含 `p0_evaluator` 打印的 `return_mean`）。

## 8. 能与不能声称

**通过后能声称的最强措辞**：

> 在 stair 与 slide 两个 target、各自的二元候选源集合、t=10k、50% dose、
> 3 个 learner seeds、128-episode source-free 面板上，直接测量 U 的 racing 在 K≤5000 时
> 选出的源与 K=10000 的 ground truth 一致，且两个 target 的正确答案互不相同。

**不得声称**：

1. 不得称"通用自动选源"——三个 target（hurdle/stair/slide）、每个 target 的候选集合
   都是**已知 ground truth 后选定**的（outcome-informed site selection）。
2. 不得称统计显著——主判据是点决策一致性，不是 U 的显著性检验。
3. 不得把"1/2 target 成立"报为正结果（§5）。
4. 不得与 door 的避损核算或 hurdle 的加速核算合并（口径不同）。
5. 每个 target 仅 2 个候选源，**不等价于**真实部署时的多源选择。

## 9. 不得做的事

- 裁决后不得调 K、阈值、seed 数或容差抢救结论。
- 违反 §7 盲态封闭即整体作废。
- 前置检查失败时不得只保留通过的部分。
- 若 `MULTI_REFUTED`，不得改用代理量补救——那会退回十一族。
