# 预注册：零训练的 task-progress 粗筛（能否整体拒绝无用源）

> 2026-08-04。**本文与裁决脚本必须在任何探针数据产出之前提交 git。**
> 定位：检验一个**零训练**的 zero-shot 信号能否承担"绝对准入"——
> 即判断"这个 target 上是不是所有源都该拒绝"。
> **不做**源之间的精细排序（那是 racing 的职责，本实验不涉及）。

## 1. 立项门

1. **核心问题**：能否不付出训练代价就排除有害源？
2. **唯一主要假设**：源在 target 上的 zero-shot **前进进度**（不是 return）
   能把"全部源都无用"的 target 与"存在可用源"的 target 分开。
3. **正负后果**：正 → 粗筛可用，racing 只需在粗筛存活的候选上做精选，
   甚至在 racing 不可用时单独承担准入；负 → 零训练准入这条路按十一族先例关闭，
   准入只能靠实测（racing）。
4. **是否重复**：**不重复**，理由见 §2。
5. **最小成本**：12 组 zero-shot rollout（无训练），约 1 GPU·小时。

## 2. 为什么这不是十一族的换皮

族 1（zero-shot 行为 return / 位移）与族 11（静态规格）都已失败。本实验与它们的
差别必须写清楚，否则就是换皮：

| | 已失败的做法 | 本实验 |
|---|---|---|
| 测的量 | 源在 target 上的 **reward / return** | 源在 target 上的**前进进度**（位移）|
| 用途 | 给源**排序**，选最好的 | 只做**单向排除**：进度≈0 → 拒绝 |
| 方向性 | 双向（既选又拒） | **单向**（只拒，不选）|

**关键事实（`crawl_equal_dose_source_calibration_v1_results_20260723.md:44`）**：

> `probe_transfer_map_v2.py` 没有为 crawl 配置 task progress keys，因此
> `progress_gain` **恒为 0**，旧 score 实际退化成 scalar prefix reward。

即 crawl 上那次失败是 **`reward_gain` 的失败**，`progress_gain` 因配置缺失
**从未参与**。而在 hurdle 上（`hurdle_probe_rbo_label_alignment_v1_20260723.md`）：

```
v2 h=25 reward_gain    : walk > run > stand    top 错
v2 h=25 progress_gain  : run > walk > stand    对齐真值
v2 h=50 progress_gain  : run > walk > stand    对齐真值
```

同一份文档的 §5 明确写下"下一步必须加入 source-specific 负标签，检验这些候选
能否拒绝有害 source"，并选定 crawl 作负例。**该步骤从未完成**，本实验补上它。

**机制上为什么 return 会骗而进度不会（crawl）**：crawl 的 reward 是加性的
`0.1·small_control + 0.25·min(crawling,crawling_head) + 0.4·move + 0.25·xquat`，
再乘 `in_tunnel`。站立的 walk/run 源**在隧道外面走得很好**，`move` 分量给出正
reward（v2 实测 h=25 为 `+6.35/+5.43`）；但隧道顶高仅 1.35 m，站立源
**进不去**，任务进度为零。return 看不见这个阻断，位移看得见。

## 3. 适用边界（**必须随结论一起引用**）

本信号只对 **locomotion 类 target** 有定义——进度 = 前进位移。
door / cabinet / window 等 manipulation target 的进度不是位移
（族 1 在 door 上的失败正是用位移当进度的后果），**本实验不覆盖它们**，
结论不得外推。

## 4. 协议（冻结）

```
源（3）        stand / walk / run，checkpoints/official_sources/*/manifest.json
环境（4）      h1hand-crawl-v0   h1hand-hurdle-v0   h1hand-slide-v0        ← 判据用
               h1hand-walk-v0                                              ← 仅辅助解释，不进判据
rollout        zero-shot，deterministic（无探索噪声），不训练
并行           NUM_ENVS=32，EPISODE_STEPS=1000，ENV_SEED=7（沿用既有探针常量）
主测量         P(i,T) = 每 episode max_t(x_t − x_0) 的均值，x = qpos[0]（= obs[0]）
辅助测量       return / ep_len / fall_rate / 各 reward 分量均值（仅描述，不进判据）
```

`obs = concat(qpos, qvel)`（`humanoid_bench/tasks.py:32`），故 `obs[0] == qpos[0]`，
位移可零开销地从观测读出，不需要跨进程访问 MuJoCo。

## 5. 判据（冻结，先于数据）

**指标用绝对位移，不用比值**：三个 target 同为 locomotion、episode 同为 1000 步
（20 s 仿真），尺度可比；且 `stand` 源本身不前进，任何以"源在原任务上的位移"
为分母的比值都会除零。

**split-sample（§8.7）**：

```
design data  crawl（期望全部拒绝）+ hurdle（期望不拒绝 run）
holdout      slide（不参与阈值设定）

阈值取法（先于数据冻结，数值由 design data 决定）：
    lo = max_i P(i, crawl)          # 最"能走"的那个无用源
    hi = P(run, hurdle)             # 已知有用源的进度
    θ  = sqrt(lo × hi)              # 几何平均，落在两者之间
```

| 条件 | 裁决 |
|---|---|
| `lo < hi` 且 `P(walk, slide) > θ` | `PROGRESS_SCREEN_VIABLE` |
| `lo ≥ hi` | `SEPARATION_FAILED`（design data 上就分不开，holdout 不再检验）|
| `lo < hi` 但 `P(walk, slide) ≤ θ` | `HOLDOUT_FAILED`（误杀已知有用源）|
| 任一组合数据缺失 | `INCOMPLETE`（非零退出）|

## 6. `CLAUDE.md` §8 设计层自查

- **8.1 辨别力**：两个平凡解释各被一侧排除——"永远拒绝"会在 hurdle 上拒掉
  `run`（`hi` 落进 crawl 区间）→ `SEPARATION_FAILED`；"永远接受"会使
  `lo ≥ hi` 不成立但 crawl 三源全部高于 θ，同样落进 `SEPARATION_FAILED`。
  第三个捷径"选位移最大者"不适用——本判据是阈值型单向排除，不做排序。
- **8.2 混淆**：12 组共用同一 `ENV_SEED`、同一 episode 长度、同一 deterministic
  设置；源之间唯一差别是 checkpoint。无剂量概念（不训练）。
- **8.3 独立重复**：zero-shot 探针无 learner seed；不确定性来自 32 个 env 的
  episode 面板，报 SE。**因此本实验不产生 learner 级结论**，只筛选。
- **8.4 前提蕴含结论**：三个裁决分支均可达——`SEPARATION_FAILED` 在
  hurdle 的 run 因 62% 摔倒率而位移很小时发生；`HOLDOUT_FAILED` 在 slide 的
  walk 位移低于 θ 时发生。二者都不是逻辑上不可能。
- **8.5 site selection**：crawl / hurdle / slide 的真值**均已知**，故这是
  **校准 + 单次 holdout 检验**，不是确认性实验。结论只能声称"在这三个
  locomotion target 上成立"，跨任务推广须用新 target 重测。
- **8.6 本轮教训对照**：M28（源 bank 配置逐个核实）；M31（本实验不产生跨
  learner 标签，故不受符号可推广性问题影响）；M16/M24（不产生 learner 级
  统计结论，故不涉 learner 间方差）。
- **8.7 判据切换红线**：slide 为 holdout，全程不参与阈值设定；裁决后不得改 θ 取法。

## 7. 能与不能声称

**能**（若 `PROGRESS_SCREEN_VIABLE`）：在这三个 locomotion target 上，
zero-shot 前进进度可以**单向排除**无用源，成本为零训练。

**不得**：
1. 不得用它给源**排序**或选 top-1（判据只支持单向排除）；
2. 不得外推到 manipulation target（§3）；
3. 不得声称它替代 racing 的精选能力；
4. 不得因结果不佳而调 θ 取法或换 holdout（§8.7）。

## 8. 不得做的事

- 裁决后不得改阈值取法、不得把 slide 挪出 holdout、不得增删环境或源。
- 若 `SEPARATION_FAILED`，如实报告并按十一族先例关闭"零训练准入"这条路，
  不得通过换指标（速度/关节幅度/姿态…）在同一批 target 上抢救。
