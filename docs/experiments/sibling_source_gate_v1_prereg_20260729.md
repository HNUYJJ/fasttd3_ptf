# Slide ↔ Stair 双向 sibling-source gate（预注册）

> 2026-07-29。**本文件与裁决脚本在任何 sibling 臂被评估之前提交。**
> 简短预注册，一次 smoke 后执行，不再开设计审查循环。

## 1. 唯一要回答的问题

任务分类学 v1 的阶段二暴露：既有 14 个 U 单元**全部**取自"跨 reward 实现族且跨地形族"
这一种配置，从未有过同族迁移的对照点，因此任务图在现有数据上不可检验。

本 gate 补上这个缺失配置，检验一个可证伪的假设：

> **在地形仍不相同的条件下，与 target 共用同一 reward 实现（`ClimbingUpwards`）的
> sibling source，其 RBO 学习效用应稳定高于通用 walk source。**

slide 与 stair 满足：同一 `ClimbingUpwards.get_reward`、同机器人、同 obs/action 规格；
地形不同（`continuous_mesh` vs `discrete_slabs`）。

## 2. 复用与新增

**不训练任何新 source。** 已核验可直接使用的冻结源：

| source | checkpoint | global_step | obs_dim |
|---|---|---|---|
| slide | `models/h1hand-slide-v0__h1hand_slide_tp_scr_s1_20260615T044012Z__1_final.pt` | 100000 | 151（identity） |
| stair | `models/h1hand-stair-v0__h1hand_stair_tp_scr_s1_20260615T044012Z__1_final.pt` | 100000 | 151（identity） |

| 条件 | 状态 |
|---|---|
| slide 的 student / walk / 10k anchor | 已有（`slide_bac_gate_v1`） |
| stair 的 student / walk / 10k anchor | 已有（`stair_bac_gate_v1`） |
| **stair target ← slide source** | **本轮新跑，3 seeds** |
| **slide target ← stair source** | **本轮新跑，3 seeds** |

共 6 条新臂。

## 3. 冻结协议（与 BAC gate 逐项一致，只换 bank）

`t=10k, K=10k`、`bootstrap_only`、`admission_mode=all`、`student_logit=0.0`、
`expected_source_mass=0.5`、`h=25`、warmup 30000、训练到 20k、
`resume_noise_seed=91000+seed`、复用同一批 10k anchor；
评估为冻结 source-free 面板 128 deterministic episodes（16 eval seeds × 8 ranks）。

## 4. 主比较与裁决

```
主比较（同 seed 配对）：
    D_sib(dir) = J(sibling source) − J(walk source)     dir ∈ {slide→stair, stair→slide}

裁决（PI/审核冻结）：
  SIBLING_PRIOR_SUPPORTED   两个方向的配对均值都 > 0，
                            且每个方向至少 2/3 seed 同向为正
  SIBLING_DIRECTION_DEPENDENT  仅一个方向成立
                            → 不支持一般结构先验，记为方向依赖
  SIBLING_PRIOR_REFUTED     两个方向均不成立
                            → 停止 taxonomy 的预测路线，
                              仅保留其问题刻画与 benchmark 划分用途
```

**不加 seeds、不调剂量、不换 horizon、不改评估面板来抢救。**

次级（单独报告，不参与裁决）：sibling 相对 student 的 U；跨 seed 方差（描述性）。
episode-level SE 仅作评价可靠性诊断，不得代替 learner-seed 不确定性（教训 M16）。

## 5. 后续（预先指定，防事后挑靶）

**仅当** `SIBLING_PRIOR_SUPPORTED` 时，才启动预先指定的独立确认场
`balance_simple → balance_hard`——二者不仅 reward 实现相同，
`composition`（multiplicative）与 `terrain`（few_slabs）也相同。
通过该第二场后，才允许把 taxonomy 特征纳入迁移指标或 source selector。

若本 gate 失败，**不启动 Balance**。

## 6. 工程验收（只查这五项）

1. 10k anchor 正确恢复（日志 `Resumed core learner ... at step 10000`）
2. bank 加载为单一 sibling 源（日志 `Loaded source bank options: ['slide']` / `['stair']`）
3. behavior source share ∈ [0.45, 0.55]，且与同 target 的 walk 臂无实质差异
4. critic source share ≈ 0.5
5. 其余训练参数与 walk 臂逐项相同
