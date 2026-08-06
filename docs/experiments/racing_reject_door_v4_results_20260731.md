# 结果：RACING_REJECT v4 —— `PARTICIPANT_DIVERGED`

## door 的"三源一致有害"不推广到新 learner

> 2026-07-31。预注册 `docs/experiments/racing_reject_door_v4_prereg_20260731.md`（`38311fb`），
> 裁决脚本 `scripts/analysis/analyze_racing_reject_door_v4.py`（`50e6409`），
> 均在 seeds 7–9 的任何数据产出之前冻结。

## 1. 裁决

```
VERDICT: PARTICIPANT_DIVERGED          （主终点按 §5 优先级不予裁决）

层1 工程硬检查   PASS   seeds 4–9 共 24 臂 × 3 K：剂量 / 臂身份 / 面板 / sha256 / 协议
层2 批内自洽     FAIL   newbatch 的 s7 与 s9 在 K=10000 未拒绝
```

脚本已验证**未输出任何 `K≤5000` 结果**。

## 2. 核心发现：符号本身跨 learner 反转

`U ± paired SE`（K=10000，逐 episode 配对，128 ep）：

```
                 stand              walk               run          max
hold_s4    −3.62 ± 5.42     −2.75 ± 4.71     −2.27 ± 5.52     run  (跨零)
hold_s5   −32.47 ± 2.33    −27.35 ± 1.96    −41.42 ± 3.17     walk (显著负)
hold_s6   −44.67 ± 3.39    −55.32 ± 5.44   −104.56 ± 5.59     stand(显著负)
newb_s7    +3.51 ± 4.01    −54.03 ± 4.72    −25.18 ± 6.90     stand(跨零)   ← 未拒绝
newb_s8    −2.47 ± 2.29    −17.72 ± 2.31    −41.58 ± 4.72     stand(跨零)
newb_s9    −0.20 ± 5.49    −36.51 ± 7.24    **+36.32 ± 3.95** run  (显著正) ← 未拒绝
```

| 批 | per-seed 效应符号 |
|---|---|
| gate `s1–3`（已发表） | 负 **9/9** |
| holdout `s4–6` | 负 **9/9** |
| newbatch `s7–9` | 负 7/9，**正 2/9** |

**`s9` 的 `run = +36.32 ± 3.95` 是显著正迁移**（单侧 90% 下界 `+29.8 > 0`），
而 gate 对 run 的结论是 `−30.63`（harmful）。同一 `(source, target, stage, dose,
协议)` 下，**跨度 67**。

## 3. 这推翻了什么、没推翻什么

**没推翻**：gate 在 `s1–3` 上的测量本身。holdout `s4–6` 的符号 9/9 复现了它，
共 18/18 per-seed 为负。gate 的数值与结论在其自身 learner 上是可靠的。

**推翻的是推广性**：`door_at10k_gate_v1` 的核心表述

> "三个 locomotion 源在 door@10k→20k 上**一致有害**：9/9 个 per-seed 效应全为负"

必须限制为 **"在 seeds 1–6 上一致有害"**。它不是 door 这个 target 的性质，
而是**该 learner 子总体**的性质。

**这是 `M17`（learner-path dependence）迄今最强的证据。** 此前的证据是
通道归因跨 seed 反向（`M17`）与数值漂移（`M27`）；本轮首次出现
**符号本身的显著反转**——不是"效应大小不稳"，而是"有害/有益的方向不稳"。

## 4. 对本项目所有 U 标签的直接后果

`M18` 建立了**标签可测性**审计（该 stage 的标签能否被分辨）。
本轮显示这还不够，缺一层：

> **标签可推广性**：在 `n` 个 learner 上测得的符号，能否推广到新 learner？

现有全部 A 级标签（`EQD30K`、`sibling gate`、`door gate`）**都只在 3 个 learner
上测过**，且都未做过跨 learner 的符号稳定性检验。本轮是首次做，
结果是**符号不稳定**（door）。

**hurdle 是对照**：`run` 的 `U = +379.66`，CI90 `[+271.5, +487.9]`，
远离零点；`RACING_K` 两批 6 个 learner 全部选中 run。
所以符号稳定性**因 target 而异**，不能默认。

## 5. 对 racing 方向的含义（探索性，非预注册）

racing 的决策依赖 `U` 的符号与排序。因此：

- door 上 `U` 的符号跨 learner 不稳定 ⇒ **任何**基于 `U` 的选源/拒绝方法
  在 door 上都不可靠，这不是 racing 特有的缺陷；
- hurdle 上 `U` 的符号稳定（run 强正、远离零点）⇒ racing 可靠（已实证 6/6）。

由此**推测**（须前瞻检验，不得作为结论）：racing 的适用条件是
"该 target 上 `U` 的符号在 learner 总体上稳定"，而这本身可以用少量 learner 检验。

> **按 `M29` 自查**：本推测是从本轮数据事后得出的，因此标注为探索性。
> 与 M29 那次不同之处在于——`PARTICIPANT_DIVERGED` 的**含义**是预先声明的
> （v2 预注册 §5.3：「含义是 door 的"三源全负"结论不推广到新 learner，
> 这本身是一个值得单独报告的发现」），故 §3 的推翻性结论属预注册解读；
> 只有 §5 的"适用条件"推测是新的。

## 6. 主终点仍未裁决，且不得再改判据

door 的"racing 能否提前拒绝"经 v1（不可证伪）、v2（容差量纲错配）、
v3（判据切换，撤回）、v4（participant 分歧）四轮，**仍无结论**。

按 v4 预注册 §9 的红线：**v2/v3 已各改一次判据，不得再改。**

而且现在有了更根本的理由不再改：**door 的 ground truth 本身不跨 learner 稳定**
（§3），因此它**不是一个合适的判决场**——在一个 ground truth 会翻转的 target 上，
无法检验"短 K 能否复现长 K 的决策"。

**结论：door 作为 racing 拒绝能力的判决场，就此关闭。**
若要检验拒绝能力，需要另找一个 `U` 符号跨 learner 稳定的全负 target，
而这需要先做 §4 的可推广性审计。

## 7. 数据

```
预注册 / 脚本   38311fb / 50e6409（均先于 seeds 7–9 数据冻结）
seeds 4–6       docs/data/racing_reject_door_v2/source_free_eval/    36 点（v2 产出，本轮首次揭盲）
seeds 7–9       docs/data/racing_reject_door_v4/source_free_eval/    36 点（本轮新跑）
seeds 1–3       design data，**不参与本裁决**（脚本中不存在）
裁决输出        docs/data/racing_reject_door_v4/results.json   run_id=9e0aad8303a3
剂量            seeds 7–9 为 0.4982–0.5057；带 [0.48,0.52]
anchor          24/24 臂日志含 "Resumed core learner ... at step 10000"
```

seeds 7–9 的 anchor 为本轮新建，`completed_vector_steps=10000`、
`environment_transitions=1280000`、`num_envs=128`，与 s1–6 逐项相同。
