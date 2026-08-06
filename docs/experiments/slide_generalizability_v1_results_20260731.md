# 结果：slide 标签可推广性审计 —— `GEN_OK`

## 并首次具备真正的 crossover 条件

> 2026-07-31。预注册 `docs/experiments/slide_generalizability_v1_prereg_20260731.md`（`2c1804f`），
> 裁决脚本 `scripts/analysis/analyze_slide_generalizability_v1.py`（`16fccf1`），
> 均在 seeds 4–6 的任何评估结果产出之前冻结。

## 1. 裁决

```
VERDICT: GEN_OK        seeds 4–6 中 3/3 满足 argmax_i U_i = walk

层1 工程检查 PASS（12 臂：剂量 / 臂间 share 差 / 臂身份 / 冻结面板 / sha256 / 协议）

per-seed U ± paired SE（K=10000，逐 episode 配对，128 ep）：
  s4:  stand −56.08±5.14   walk  +6.85±4.70   run −22.65±5.16    argmax = walk
  s5:  stand  −5.82±1.89   walk +49.05±2.60   run  +7.82±2.43    argmax = walk
  s6:  stand +20.24±2.00   walk +77.04±2.06   run +23.41±2.24    argmax = walk
```

连同 BAC gate 的 seeds 1–3（3/3 argmax = walk），**walk 在 6 个独立 learner 上一致为最优源**。
偶然通过率 `(1/3)³ = 3.7%`（三选一，单 K 单判据，无 look-elsewhere）。

## 2. 首次具备真正的 crossover

`RACING_MULTI` 被判 FATAL 的首要原因是"两个 target 的候选集合不同，
一个全局固定排序即可解释"。hurdle 与 slide **没有这个问题**：

```
候选集合完全相同：{stand, walk, run}

hurdle：run(+379.66) > walk(+104.89) > stand(+51.28)      argmax = run
slide ：walk(+44.31) > run(+2.86)    > stand(−13.89)      argmax = walk   （本轮 s4–6）
        walk(+56.95) > run(+16.90)   > stand(−1.21)                        （gate s1–3）
```

**walk 与 run 在同一候选集合上换位，且两侧各有 6 个 / 3 个 learner 支撑。**
任何"全局固定源质量排序"都无法同时满足两者——
这直接排除了"迁移只是源质量排序"这一朴素解释。

## 3. 机制发现：源间差消去了 student 基线的漂移

| 源 | gate `s1–3` mean | 新批 `s4–6` mean | 漂移 |
|---|---:|---:|---:|
| stand | −1.21 | −13.89 | **−12.68** |
| walk | +56.95 | +44.31 | **−12.64** |
| run | +16.90 | +2.86 | **−14.04** |

**三个源的漂移几乎相同**（−12.6 ~ −14.0）。因为

```
U_i = J_i − J_student        student 基线对所有 i 相同
```

当新批的 `J_student` 系统性偏高时，全部 `U_i` 同步下移；
而 racing 的决策量是**源间差**，基线在相减时**相消**：

```
argmax 间隔（walk − 次优）：  gate 40.05   →   新批 41.45      差仅 1.4
绝对 U 的漂移：                              ~13
```

这正是 Codex 在 v3 review 中的判断（"比较二者时 student 漂移会相消，
真正应估计的是 `D = J_walk − J_stair` 的稳定性"）——**本轮数据直接验证了它**。

**推论**：racing 的 `argmax` 天然比绝对 `U` 稳健，因为它只依赖源间差。

## 4. 但个别源的符号仍不可推广

```
walk ：gate 3/3 正 → 新批 3/3 正       稳定
run  ：gate 3/3 正 → 新批 2/3 正       s4 = −22.65 ± 5.16（显著负）  ← 翻转
stand：gate 1/3 正 → 新批 1/3 正       本就 `uncertain`
```

`run` 在 `s4` 上从 gate 的 `+16.90`（3/3 正）翻到 `−22.65`（显著负）。
故 `M31` 的现象在 slide 上**同样存在**，只是没有影响 argmax。

**这加强了 §3 的区分**：可推广的是**决策**（argmax），不是**每个源的绝对符号**。
任何把 per-source `U` 当作可复用标签的做法（本项目此前的做法）都仍然不成立。

## 5. 一个探索性观察（**不作为结论**）

三个 target 的 argmax 间隔与其稳定性：

| target | argmax 与次优的间隔 | run-to-run 漂移 | argmax 稳定性 |
|---|---:|---:|---|
| hurdle | ~275（run +379.66 vs walk +104.89） | ~15 | 6/6 稳定 |
| slide | ~40 | ~13 | 6/6 稳定 |
| door | ~8.4（walk −22.20 vs run −30.63） | ~24 | 不稳（v4 中 2/3 未拒绝） |

方向一致：间隔 ≫ 漂移时 argmax 稳定，间隔 < 漂移时不稳。
且 §3 给出了**机制解释**（基线相消，故间隔才是决策的有效信噪比）。

**但这仍是 3 个事后数据点。** 按 `M29`/`M30`，本文**不**把它写成判据或规律，
只记录为待前瞻检验的假设。与 `M29` 那次的区别是：本次三个"间隔"是同一个量
（argmax 与次优的 `U` 差），不存在当时的量纲混用问题。

## 6. 能与不能声称

**能**：

- slide 上"walk 是最优源"这一决策在 **6 个独立 learner** 上成立；
- slide 与 hurdle 在**同一候选集合**上 argmax 反转，构成真正的 crossover；
- racing 的决策量（源间差）不受 student 基线漂移影响——有数据与机制双重支持。

**不得**：

1. 不得称"slide 的标签完全可推广"——`run` 的符号在 `s4` 上翻转，`stand` 本就 `uncertain`；
2. 不得称 6 个 learner 足以保证推广——door 用了 18 个 per-seed 效应仍被第 19 个推翻（`M31`）；
3. 不得据此直接声称 racing 在 slide 上有效——本实验只验证了**标签**可推广，
   **未**验证 racing 在短 K 能否复现该 argmax（那是另一个实验）；
4. 不得把 §5 的观察当作判据或规律。

## 7. 数据

```
预注册 / 脚本   2c1804f / 16fccf1（均先于 seeds 4–6 评估冻结）
seeds 1–3       BAC gate 已发表（docs/experiments/slide_bac_gate_v1_results_20260728.md）
seeds 4–6       docs/data/slide_generalizability_v1/source_free_eval/   12 点（本轮）
anchor          artifacts/slide_bac_gate_v1/anchors/s{4,5,6}（本轮新建，协议同 s1–3）
裁决输出        docs/data/slide_generalizability_v1/results.json   run_id=597a7cce77a4
剂量            0.4771–0.4845，带 [0.45,0.55]
臂间 share 差   s4 0.0018 / s5 0.0046 / s6 0.0062  —— 远低于 M26 的 5pp 上限，
                故"某源更好只因它被用得更多"这一解释被排除
anchor 恢复     12/12 臂日志含 "Resumed core learner ... at step 10000"
```
