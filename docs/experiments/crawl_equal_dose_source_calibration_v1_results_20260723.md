# Crawl 等剂量单源 RBO 标定：seed-1 负例

日期：2026-07-23  
任务：`h1hand-crawl-v0`  
裁决：`NEGATIVE_SOURCE_FOUND`

## 研究问题

历史实验只证明 stand/walk/run 混合 bank 在 crawl 上低于 scratch，不能定位哪个
source 有害。本实验固定 0.5 source / 0.5 student、25 步锁存、0–30k
`bootstrap_only` 与相同 replay 规则，仅替换 source 身份。

## 结果

| arm | 5k–25k nAUC | ΔAUC vs scratch | 30k source-free return | Δendpoint | 最大前进位移 |
|---|---:|---:|---:|---:|---:|
| scratch | 517.821 | — | 798.671 | — | 12.339 |
| stand | 401.637 | −116.183 | 350.191 | −448.480 | 0.577 |
| walk | 401.846 | −115.975 | 582.067 | −216.604 | 2.173 |
| run | 434.582 | −83.239 | 590.601 | −208.070 | 2.469 |

三个 source 的 behavior source share 与 critic source share 均约为 0.500，且
execution/sample counts 在相同 seed 下相同，差异不是剂量造成的。

冻结 32 episode 面板中，stand、walk、run 相对 scratch 的 return 和最大前进位移
均为 0/32 获胜。负 endpoint 不是评估面板抽样偶然性。

## 与旧 probe 的冲突

Transfer Map v2 对 crawl 的记录：

| source | h=25 reward gain | h=50 reward gain | 实际 ΔAUC | 实际 Δendpoint |
|---|---:|---:|---:|---:|
| stand | −0.645 | −1.612 | −116.183 | −448.480 |
| walk | +6.345 | +13.907 | −115.975 | −216.604 |
| run | +5.428 | +13.662 | −83.239 | −208.070 |

所以 short-prefix reward 的正符号不是 RBO learning utility 的充分条件。

## 为什么旧 probe 会误判

1. crawl 的 v2 记录明确使用 `baseline="zero"`，不是 current student；
   当时没有找到 crawl 的 scratch-early checkpoint。
2. `probe_transfer_map_v2.py` 没有为 crawl 配置 task progress keys，因此
   `progress_gain` 恒为 0，旧 score 实际退化成 scalar prefix reward。
3. Crawl 官方 reward 是加法：

   `0.1 small_control + 0.25 crawl_posture + 0.4 move + 0.25 pelvis_orientation`

   再乘 `in_tunnel`。upright walk/run 可以通过 `move` 在短 prefix 中获得正 reward，
   即使它们没有形成正确的 crawl posture/occupancy。
4. 因而旧 probe 测到的是“source 相对 zero 能否暂时移动拿分”，不是“source 相对
   当前 student 是否把学习带向目标任务所需状态”。

## 对迁移性指标的约束

现有正负标签为：

- Hurdle：`G_run > G_walk > G_scratch`（3 seeds，双视角）；
- Crawl：`G_scratch > G_run > G_walk ≈ G_stand`（seed-1，双视角）。

一个可接受的指标必须同时完成：

1. Hurdle 内部排序；
2. Crawl 对三个 locomotion source 的 absolute rejection；
3. 使用 current student 而不是 zero/action-free baseline；
4. 避免单个易获得的 reward component 掩盖 target bottleneck。

因此，旧 `T0`/prefix reward 可保留为行为描述或 source 内部排序特征，但不能单独
决定 admission。

## 下一候选：stage-conditioned component-dominance probe

从当前 student occupancy 的同一批环境状态分叉 source 与 student 的短 horizon，
比较 target reward-component 向量，而不是只比较 scalar return：

- source 必须相对 student 改善 target progress；
- termination、任务必要姿态/接触/约束分量不得恶化；
- student 作为一等候选；无 source 满足 component dominance 时 exact abstention；
- 只先离线检查该规则能否同时判对 Hurdle 与 Crawl，不立即接在线 controller。

该候选仍需与历史 adaptive immediate-reward 失败结果区分：新增信息必须来自
matched student baseline 与 target-component non-degradation，而不是重新调 scalar
reward 阈值。

原始结果：
`logs/train/crawl_equal_dose_source_calibration_v1_20260723T172024Z/`

