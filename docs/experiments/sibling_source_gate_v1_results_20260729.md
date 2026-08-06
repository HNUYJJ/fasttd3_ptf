# Slide ↔ Stair sibling-source gate 结果：`SIBLING_DIRECTION_DEPENDENT`

> 2026-07-29。预注册与裁决脚本提交于 `22090fb`，**先于任何 sibling 臂被评估**。
> 结论：**同 reward 实现族不构成稳健的正向迁移先验**；按预注册**不启动** Balance 确认场。

## 1. 裁决

```
VERDICT: SIBLING_DIRECTION_DEPENDENT
```

### 方向 1：slide 源 → stair target（sibling **胜出**）

| seed | student | walk | **slide 源** | D_sib |
|---|---:|---:|---:|---:|
| 1 | 44.77 | 45.11 | **67.02** | +21.91 |
| 2 | 45.86 | 42.69 | **55.92** | +13.23 |
| 3 | 42.28 | 45.67 | **56.72** | +11.05 |

```
D_sib = +15.40   90%CI [ +5.72, +25.08]   3/3 seed 为正   方向通过
次级：sibling − student = +15.59   90%CI [+5.18, +25.99]  显著正迁移
```

### 方向 2：stair 源 → slide target（sibling **输给 walk**）

| seed | student | walk | **stair 源** | D_sib |
|---|---:|---:|---:|---:|
| 1 | 39.38 | 105.04 | 90.68 | −14.36 |
| 2 | 63.61 | 111.22 | 84.02 | −27.20 |
| 3 | 50.54 | 108.11 | 87.30 | −20.81 |

```
D_sib = −20.79   90%CI [−31.61, −9.97]   0/3 seed 为正   方向未通过
次级：sibling − student = +36.16   90%CI [+10.10, +62.22]  仍是显著正迁移，但不及 walk
```

## 2. 假设的裁决

预注册的假设是：

> 与 target 共用同一 reward 实现（`ClimbingUpwards`）的 sibling source，
> 其 RBO 学习效用应**稳定**高于通用 walk source。

**该假设被否定。** 一个方向 +15.40（显著优于 walk），另一个方向 −20.79（显著劣于 walk），
符号相反且两者区间均不跨零。共用 reward 实现这一结构性质**无法**决定谁更好。

按预注册 §5，**不启动** `balance_simple → balance_hard` 确认场。
taxonomy 的**预测**路线到此为止，仅保留其问题刻画与 benchmark 划分用途。

## 3. 剂量混淆的分析（两个方向结论的稳健性不同）

sibling 臂的 behavior source share 系统性高于 walk 臂：

| target | sibling 臂 | walk 臂 | 差 |
|---|---|---|---|
| stair | 0.4956 – 0.4983 | 0.4651 – 0.4733 | +2.5 ~ +3.3% |
| slide | 0.5006 – 0.5037 | 0.4766 – 0.4789 | +2.4 ~ +2.5% |

成因：sibling 源在同族地形上不摔，episode 更长，latch 到期重抽的分布因而不同。
critic share 在所有臂上均为 0.4988–0.4992，无差异。

这使两个方向的结论**稳健性不对称**：

- **方向 2（负结论）更稳健**：剂量优势在 sibling 一侧，**它仍然输了 20.79**。
  剂量无法解释该负结果。
- **方向 1（正结论）需要打折**：剂量优势与胜出同向，二者无法完全分离。
  参照系是 BAC gate 中 0.8% 的剂量差对应了 +58 的效应差，故 2.5–3.3% 的剂量差
  不太可能解释全部 +15.40，但**本实验不能排除它贡献了一部分**。

## 4. 观察到的不对称（**记录，不作主张**）

按 J@20k 排列：

```
stair target:   slide 源 67.02  >  walk 45.11  ≳  student 44.77
slide target:   walk 108.12     >  stair 源 87.30  >  student 51.18
```

**slide 源在两个地形上都有用；stair 源只在自己的地形上最好，换到连续斜坡即不及通用 walk。**

一个可能的解释是：连续斜坡上学到的步态更接近平地行走的推广，能应付离散台阶；
而针对离散台阶的特化步态在连续斜面上反而不如通用步态。

**但这是 n=1 的方向对，且只有这一对任务。** 不得外推为
"从简单/连续地形学到的技能更易迁移"这类一般定律。列此仅为记录现象。

## 5. 本轮在整条线中的位置

这是**第三次**从不同角度得到同一结论：

| 轮次 | 检验的相似性 | 结果 |
|---|---|---|
| 任务分类学 | reward 代数签名同构 | slide/stair 同族，U 为 +56.95 与 +0.19 —— 不蕴含可迁移 |
| BAC 预测器比较 | reward 分量覆盖 | 相对 per-step / 主进度分量**无增量** |
| **本轮 sibling gate** | **完全相同的 reward 实现** | **方向依赖，不稳健优于通用源** |

三者共同支持一个收敛的否定性结论：

> **任务定义层面的相似性——无论是 reward 代数、reward 分量覆盖，还是完全相同的
> reward 实现——都不能预测冻结源的迁移效用。**

这条结论现在有前瞻实验支持（本轮预注册在先），不再只是回溯观察。

## 6. 限制

1. 只有一对任务（slide/stair）、3 个 learner seed、df=2。
2. 剂量混淆见 §3，方向 1 的正结论未能与剂量完全分离。
3. 两个源均为单 seed 训练的冻结策略（`tp_scr_s1`），源自身的质量差异
   （slide 源是否本就比 stair 源更强）未被独立测量，可能与"地形不对称"混淆。
4. 本轮**不**支持任何 source-selection 规则；`SIBLING_DIRECTION_DEPENDENT`
   的后果按预注册是停止预测路线，不是"部分成立"。

## 7. 数据

```
预注册与裁决脚本   docs/experiments/sibling_source_gate_v1_prereg_20260729.md  (22090fb)
                   scripts/analysis/analyze_sibling_source_gate_v1.py
面板（各 128 ep）  docs/data/stair_bac_gate_v1/source_free_eval/slidesrc_s{1,2,3}_step20000.json
                   docs/data/slide_bac_gate_v1/source_free_eval/stairsrc_s{1,2,3}_step20000.json
裁决输出           docs/data/sibling_source_gate_v1/sibling_source_gate_v1_results.json
冻结源             checkpoints/terrain_sources/h1hand_{slide,stair}/manifest.json
```
