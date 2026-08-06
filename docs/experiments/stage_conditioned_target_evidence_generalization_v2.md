# Stage-conditioned target evidence v2

> 冻结日期：2026-07-26  
> 触发原因：PI 要求机制不得针对 Crawl/Hurdle 硬编码；v1 的 all-component veto
> 一般化压力测试也在 Hurdle 上产生错误的过度保守拒绝。

## 1. 一般化接口

核心算法只接受 target MDP 定义的三类证据：

```text
target_return
target_achievement_progress
optional_hard_constraints
```

任务语义通过 `TargetEvidenceContract` YAML 提供，作用等同于 obs/action adapter：

- progress 可来自模拟器状态差或任意 target `info` 字段；
- progress 可由若干 target components 门控，表示“只有满足任务语义时才算进度”；
- target components 全部保留为诊断量；
- 只有任务定义中真正不可交易的 hard constraint 才拥有 veto 权。

核心实现不得出现 `if task == crawl/hurdle/...`，也不得包含 source-specific 配置。

## 2. 冻结 admission rule

对同一 student occupancy 状态上的 source/student 25 步 matched branches：

```text
admit(source) iff
    LCB90(Δ target_return) > 0
and LCB90(Δ target_achievement_progress) > 0
and every explicitly declared hard constraint has LCB90(ΔC_k) >= 0
```

多个接纳 source 按 `LCB90(Δ target_achievement_progress)` 排序；无人通过则 exact
abstention。没有数值 epsilon、任务权重或 source return prior。

## 3. 为什么不是“针对 Crawl”

Crawl adapter 声明：

```text
achievement progress =
    root-x displacement
    × mean(min(crawling, crawling_head))
    × mean(in_tunnel)
```

因此“直立向前走”不会获得完整任务进度。该定义来自 target reward semantics，不来自
某个 source 的实验结果。

Hurdle adapter 使用：

```text
achievement progress =
    root-x displacement
    × mean(stand_reward)
    × mean(wall_collision_discount)
```

操纵任务可以把 progress 换成物体位移、门开启量、成功子任务数等；核心公式不变。

## 4. 一般化压力测试与一次结构性修正

第一次配置化尝试把 `stand_reward`、`wall_collision_discount` 等所有 component 都当作
hard veto。它在 Hurdle@10k 把 run/walk 全部拒绝：少量碰撞折扣的 LCB 略低于 0，
但 target return、progress 与 3-seed RBO 学习效果均为正。

这不是数值阈值问题，而是语义错误：reward components 可以存在由 target reward
定义的合法权衡，不能自动升级为硬约束。因此 v2 冻结为：

- components 默认用于 progress gating 与诊断；
- hard veto 必须由 target MDP 明确声明；
- 本轮 Hurdle/Crawl 均没有额外 hard constraint。

该修正冻结后不再根据正反例修改。

## 5. 最小验证门

- Hurdle@10k：接纳 run、walk，排序 run > walk；
- Crawl@10k：stand/walk/run 全部拒绝。

两项同时通过才允许进入在线低频 feasibility gate；否则停止。
