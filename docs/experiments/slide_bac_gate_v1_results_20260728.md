# Slide BAC 判决场结果：BAC_SUPPORTED（前瞻预测命中，return 被证伪）

> 2026-07-28。预注册见 `bottleneck_aligned_coverage_v1_prereg_20260728.md`（提交 33d4c92 / 75772e8），
> 裁决脚本 `analyze_slide_bac_gate_v1.py` 与事前预测均在任何 slide 臂被评估之前提交。

## 1. 裁决

```
VERDICT: BAC_SUPPORTED
```

| seed | student | stand | walk | run | D_walk | D_run |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 39.38 | 37.54 | 105.04 | 65.67 | +67.50 | +28.12 |
| 2 | 63.61 | 59.13 | 111.22 | 70.88 | +52.09 | +11.75 |
| 3 | 50.54 | 53.23 | 108.11 | 67.67 | +54.88 | +14.44 |

主判据（learner-seed 配对 90% t 区间，df=2）：

```
D_walk = J(walk) − J(stand)   mean = +58.16   90%CI [+44.32, +72.00]   显著
D_run  = J(run)  − J(stand)   mean = +18.11   90%CI [ +3.31, +32.91]   显著
```

两者均 3/3 seed 为正，满足预注册的 `BAC_SUPPORTED` 条件。

## 2. 两个指标的正面对决

| | stand | walk | run | 与实测的关系 |
|---|---:|---:|---:|---|
| zero-shot return（事前） | **88.5** | 45.7 | 27.8 | **完全反向**（Spearman ρ = −0.5） |
| NET / BAC（事前） | **0.0129** | 0.5153 | 0.5086 | 排序命中 |
| **实测 J@20k** | **49.97** | **108.12** | 68.07 | — |
| 实测 U（相对 student） | −1.21 `uncertain` | +56.95 `helpful` | +16.90 `helpful` | — |

zero-shot return 认定最好的 stand，是三者中**唯一没有产生正迁移**的源；
return 认定最差的 run 实测正迁移 +16.90；return 排第二的 walk 实测最好 +56.95。

评估可靠性：episode-level SE 在 1.00–2.47 之间，而效应量 +18 ~ +58，
信噪比远高于 cabinet 那轮（该轮无任何 per-seed 效应超过 1.74 SE）。
但按教训 M16，episode SE 仅作评价可靠性诊断，裁决一律用 learner-seed 区间。

## 3. 必须同时记录的两处不支持

裁决通过不等于指标的每一部分都成立。以下两点与 `BAC_SUPPORTED` 同等重要：

### 3.1 NET 的定量刻度不成立，只有主判别稳健

```
NET  walk/run = 0.5153 / 0.5086 = 1.013   （相差 1.3%，指标判为"≈相等"）
U    walk/run = 56.95  / 16.90  = 3.371   （相差 237%）
```

NET 把 walk 与 run 判为几乎相同，实测两者相差 3.4 倍。
walk > run 这一位的排序虽然命中，但 1.3% 的 NET 差异不具备分辨 237% 效用差的能力，
**该位应视为未被有效预测，而非命中**。

真正稳健的是 **stand vs {walk, run}** 这一主判别：NET 差 40 倍，
实测 3/3 seed 一致且区间显著。这也正是 return 预测的反面。

**结论的正确表述**：BAC 能识别"哪个源不覆盖瓶颈"，
但不能定量预测覆盖瓶颈的诸源之间的效用差。

### 3.2 C(dose) 机会成本假说未获支持

door 的结果曾促使我提出 `U_i ≈ α·NET_i − C(dose)`，理由是 door 三源 NET ≈ 0
而 U 全为 −22 ~ −33，看起来存在一个与源身份无关的固定负项。

slide 直接否证了这个形式：stand 的 NET = 0.0129（≈0），
实测 U = −1.21，90%CI [−7.33, +4.91]，**跨零**。
若 C(dose) 是通用的固定机会成本，stand 应显著为负。

因此 door 的"三源全负"需要另一个解释，不能归因于通用剂量机会成本。
该假说在此撤回，不进入论文。

α 在两个 target 内部也不自洽（door 内 α 从 844 到 38300，slide 内从 33 到 110），
进一步说明当前形式只支持序关系判别，不支持定量建模。

## 4. 机制回读

slide 的 reward 是纯乘性 `stand_reward × small_control × move`：

| source | move | stand_reward | 每步 reward | episode return |
|---|---:|---:|---:|---:|
| zero | 0.169 | 0.850 | 0.140 | 9.3 |
| stand | 0.188 | 0.997 | 0.178 | **88.5** |
| walk | 0.821 | 0.733 | 0.589 | 45.7 |
| run | 0.835 | 0.626 | 0.469 | 27.8 |

stand 几乎不移动（move 0.188，仅比 zero 高 0.019），但因姿态稳、不摔而跑满 episode，
于是刷出全场最高的 return。walk/run 每步 reward 是 stand 的 3 倍以上，却因摔倒
导致 episode 缩短，return 反而更低。

这就是四种 return 失效机制中的**生存时长**一种，在本实验中得到直接确认：
**return 混入 episode 长度，使一个几乎不推进任务的源看起来最优。**

## 5. 边界

- 本轮**只裁决 slide 一个判决场**。结论适用范围是"纯乘性结构、且存在靠生存时长
  刷 return 的源"的任务，不能外推为一般定律。
- 按预注册，升级为一般性主张需在 **stair** 上重复（其预测已与 slide 一同冻结：
  NET 0.0046 / 0.4617 / 0.4573，return 排序 stand>walk>run，同样反向）。
  该预测在本结果揭盲之前即已写死，不存在事后挑选靶子。
- walk/run 之间的分辨能力未获支持（§3.1），任何依赖 NET 定量值的下游用法
  （如按 NET 加权 replay）目前没有证据基础。

## 6. 数据

- 12 份冻结评估（各 128 episodes）：`docs/data/slide_bac_gate_v1/source_free_eval/*.json`
- 裁决输出：`docs/data/slide_bac_gate_v1/slide_bac_gate_v1_results.json`
- 剂量验收：三源 behavior share 0.4766–0.4846（跨源最大差 0.8%），
  critic share 0.4988–0.4991（差 0.03%）——排序差异不可由剂量失配解释。
