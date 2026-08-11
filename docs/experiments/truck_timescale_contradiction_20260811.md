# truck 的时间尺度矛盾：同一 bank，20k 显著负、95k 显著正

日期：2026-08-11 · 性质：**文献内核实记录**（无新实验，全部引自已冻结的历史结果）
影响：三方 review（Claude / 5.6-sol / 5.6-Pro）当前共同的推理前提需要修正。

## 1. 事实

`configs/source_banks/h1hand_hurdle4_wfix_truck.yaml`（stand/walk/run/hurdle）在 truck 上：

| 实验 | warmup | 评估点 | 相对 scratch | 出处 |
|---|---|---|---|---|
| hurdle 增量 | — | 95k | **+229.9（t=+3.47，1421 全场新高）** | `EXPERIMENT_LOG.md:97`、`科研课题完整工作总结_20260727.md:509` |
| admission handoff v1 | 30k | 95k | **+227.8**（1590.804 − 1362.977） | `docs/archive/admission_handoff_v1_results.md:91` |
| PARE Gate A | 10k→20k | 20k | −227.6（t=−5.77） | `pare_gate_a_results_20260808.md` |
| PARE Gate A | 10k→20k | 100k | 转正，不显著 | 同上 |
| T2 / T4-R / N1 | 10k→20k | 20k | −93.5 ~ −227.6 | 各自结果文档 |

**口径可比性已核实**：`truck_admission_h4_fix` 为 3 seeds × 100k、128 env、
`bootstrap_only`、`admission_replay_handoff=physical_after_authority`、
bank = stand/walk/run/hurdle——**与 N1 逐项一致**，差异只有 warmup 时长
（30k vs 10k）与评估时点（95k vs 20k）。且 warmup 在 30k 结束，
95k 评估时 source 早已无 behavior authority，**天然 source-free**。

## 2. 这意味着什么

**"truck 负迁移"这个被反复引用的前提，只在 20k 尺度上成立。**
同一 bank 在长尺度上给出的是显著**正**迁移，且是"全场新高"。

Gate A → T2 → T3 → T4-R → N1 这条链**全部只测到 20k**。它们各自的裁决
在其自身判据下都是诚实的，但**用它们推断"固定 policy 载体不适合迁移"
是超出测量窗口的推广**。

## 3. 对 "Assimilation Gap" 的直接影响

5.6-Pro 提出：source 的 behavior competence `B_i` 高，而 learning utility
`U_i < 0`，两者之差为 Assimilation Gap，并据此把"off-policy 吸收"列为
一直被忽略的中间瓶颈。

但按上表，同一 bank 的 `U_i` 在 95k 是 **+229.9**。因此更可能的表述是：

> **Assimilation Lag（延迟），而非 Gap（缺口）。**
> student 最终吸收了 source 经验并超过 scratch，只是在 20k 时尚未兑现。

这个区分直接决定下一步：

- 若是 **Gap**：需要新机制改善 off-policy 吸收；
- 若是 **Lag**：机制本身没问题，**需要的只是足够长的测量窗口**——
  过去几轮的"负迁移"可能大部分是窗口太短的产物。

两者不能靠 20k 数据分辨。

## 4. 可零训练回答的下一步

`models/` 内仍保留 `truck_admission_handoff_v1_all_s1-3` 的
**30k / 60k / 90k / final 共 27 个 checkpoint**，以及 `truck_br_scr_s1-3`
与 `truck_h4_wfix_s1-3` 的 final。

因此"迁移收益随训练步数如何演化、转折点在哪"是一个**纯前向评估**问题，
用现有 `p0_evaluator` 即可，不需要任何新训练。这应先于 Reach interface pilot、
skill library、BFM 等任何新方向。

**限制**：scratch 侧只有 final checkpoint，没有 30k/60k/90k 中间点，
故严格配对差只能在 final 处取；中间点只能看 handoff 臂自身的轨迹形状。

## 5. 我自己的流程失误

CLAUDE.md §1 要求提新方案前 grep **失败**先例。我在 N1 审计中照做了，
但**没有 grep 成功先例**——`EXPERIMENT_LOG.md:97` 就在主日志里，不在 archive 深处。

三方 review 连续多轮讨论"truck 负迁移"，无一方引用这两条正迁移记录。
建议把"同时检索同 target 的正、负先例"补进 §1。
