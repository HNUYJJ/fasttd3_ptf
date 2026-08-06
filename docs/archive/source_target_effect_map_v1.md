# Source–Target–Effect Map v1（RBO-PTF 贡献 1，2026-06-15）

承接 ChatGPT v4 对 Transfer Map 的重定位:**不预测跨任务 ROI（已证 ill-posed），
而是揭示"哪些 source option 在哪个 target 上产生哪类 reward-bearing effect、能安全
执行多久"**，据此在每个 target 内配置 source 权重/horizon，并解释迁移为何有用/无用。

## 方法

[scripts/probe_transfer_map_v2.py](../scripts/probe_transfer_map_v2.py):对每个
(source, target) 执行教师整动作 prefix(max_h=50),逐步记录 prefix reward / fall /
task-progress。reward-bearing score = max_h[reward_gain(vs zero) − fall_prob];
safe_horizon = time-to-fall<0.5 的最长前缀。source=已训好的 loco options
(stand/walk/run,obs 151);target 覆盖 terrain/locomotion/manipulation 多类
(零/小 obs adapter,布局见 logs/probe/hb_task_layouts.json)。

## 核心结果:loco source 的 effect 覆盖（vs-zero reward-bearing score / fall / safe_h）

| target | 类型 | best source | score | fall | 解读 |
|---|---|---|---|---|---|
| slide | terrain | walk | **25.5** | 0% | locomotion/terrain traversal 强 |
| stair | terrain | walk | **22.0** | 16% | 同上 |
| pole | terrain | walk | **19.3** | 0% | 障碍前进强 |
| crawl | terrain | walk | **13.9** | 0% | 低姿前进强 |
| bookshelf_simple | manip | walk | 6.4 | 0% | approach/stability |
| hurdle | terrain | run | (训练 safe 538>rand 466) | 0% | 越障(已闭环验证) |
| maze | navigation | walk | (pilot +26%) | — | 导航(已闭环) |
| basketball | manip | stand | 2.3 | **95%** | 仅有限 stability,需 reach |
| room | nav+transport | stand | 1.7 | 0% | 有限 stability |
| cabinet/powerlift | manip | (训练强正迁移) | — | 0% | stability bootstrap(Day1) |
| insert/sit_hard/sit_simple/balance_simple | manip/posture | — | <1.3 或负 | — | loco 弱,瓶颈非 locomotion |

## 三类 reward-bearing infrastructure effect（论文核心结构）

1. **Locomotion / terrain traversal**(walk/run 主导):slide/stair/pole/crawl/hurdle/
   maze。loco source 产生强 forward/terrain-traversal reward-bearing snippet
   (score 14-25、低 fall)。**关键:现有 walk/run 已覆盖 terrain,无需专门训 slide/stair
   source**——这本身是 embodiment infrastructure 可复用性的证据。
2. **Stability / survival**(stand 主导):cabinet/powerlift/window/balance(Day1 audit
   已证增益=stabilization)+ basketball/room(有限)。loco source 让 critic 学到"站稳/
   不摔"的长期价值;在 scratch 卡于 stabilization 的任务上对价大。
3. **Reach / contact**(需 reach/object source):basketball/bookshelf/cabinet handle/
   insert。loco source 弱或高 fall(basketball stand fall 95%)——需要 reach/contact
   source + semantic obs adapter(下一步)。

## 与"迁移对价"的关系（升级版核心 insight）

迁移对价 ≠ 教师质量(teacher return),而是:

> **target 的 embodiment bottleneck 与 source 的 reward-bearing infrastructure effect
> 的匹配。** source o 是否在 target T 早期产生 scratch 缺乏的 reward-bearing
> infrastructure 状态(stability / locomotion / terrain traversal / approach /
> contact),决定迁移是否有对价。

这统一解释了:stand→cabinet/powerlift(stability);walk/run→stair/slide/pole/hurdle
(terrain);walk→maze(navigation);reach→basketball/bookshelf(contact,待验证);并预测
loco→insert/sit 无对价(瓶颈非 locomotion/stability)。

## 验证状态

- **terrain effect**:expanded map 给出强 score(slide 25/stair 22/pole 19/crawl 14);
  闭环 pilot 进行中(stair/slide/pole/crawl × {reward-weighted, uniform, scratch}),
  验证 reward-weighted bootstrap(优先抽 walk/run)是否优于 uniform(均匀含无用 stand)。
- **stability effect**:Day1 audit + 多任务三方已证(hurdle/cabinet safe≥rand 强正迁移)。
- **contact effect**:待 reach source + semantic adapter(阶段 B)在 basketball/bookshelf
  上验证。

## 方法论 caveat（贡献 1 的诚实边界）

不声称跨任务 ROI 预测(reward scale + scratch 分母 confound → ill-posed,见
[transfer_map_v2_analysis](transfer_map_v2_analysis.md))。Source–Target–Effect Map
的作用是 **within-task 配置 + 机制解释 + 定性 go/no-go**(score≤0 或高 fall 的 cell
跳过),不是 ROI ranker。这比"预测对价"更诚实,也更难被审稿人打穿。

数据:[logs/probe/transfer_map_v2_expanded.jsonl](../logs/probe/transfer_map_v2_expanded.jsonl)、
[transfer_map_v2.jsonl](../logs/probe/transfer_map_v2.jsonl)。
