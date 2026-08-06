# Cross-Task Transfer Map v1 分析（2026-06-12）

数据：[logs/probe/transfer_map_v1.jsonl](../logs/probe/transfer_map_v1.jsonl)（17 任务 × {zero, stand, walk, run} + reach/push_s1 × {push, package}，每 cell 16 episodes × 500 步，zero-shot 整动作执行）。

## 热图（return 相对 zero-action 的倍数 / fall 率）

| target | zero ret (fall) | stand | walk | run | 模式 |
|---|---|---|---|---|---|
| stair | 9 (100%) | **85 (0%)** | 32 (摔) | 26 (摔) | 生存型：stand 救命，步态 OOD |
| slide | 9 (100%) | **89 (0%)** | 46 (摔) | 28 (摔) | 同上 |
| hurdle | 13 (0%) | 75 | 94 | **157 (12%)** | 推进型：run 真在跨栏 |
| maze | 115 (0%) | 229 | **379** | 340 | 推进型：walk ×3.3 |
| pole | 256 (0%) | 286 | 280 (25%摔) | 115 (81%摔) | 低对价 + run 负迁移 |
| crawl | 149 (0%) | 152 | 142 | 132 | 零对价（全平） |
| sit_hard | 7 (100%) | 2 (0%摔) | 5 | 5 | 反直觉：stand 活着但没分 |
| balance_hard | 32 (100%) | 19 | 10 | 11 | **全员 OOD + stand 负迁移** |
| push | −192 (6%) | −210 | −425 | −486 | loco 负迁移；reach −184 微改善；push_s1 12 步 success（管道验证） |
| window | 2 (100%, 19步) | 4 (摔) | 4 (摔) | 3 (摔) | **全员 OOD**（safety 试金石） |
| spoon | 7 (100%) | **73 (0%)** | 8 | 14 | 生存型 ×10 |
| door | 64 (0%) | 59 | 25 (62%摔) | **101** | run +56%（前倾步态推进 approach）；walk 负迁移 |
| cabinet | 14 (0%) | **102 (0%)** | 41 | 91 | 生存型 ×7 |
| package | −3627 | −3507 | −3115 | −6594 | return 不可分辨（dense 惩罚 σ>1000） |
| powerlift | 19 (100%) | **171 (0%)** | 167 | 117 (25%摔) | 生存型 ×9 |
| truck | 295 (0%) | **601 (0%)** | 361 | 435 | 生存型 ×2 |
| room | 9 (100%) | **90 (19%摔)** | 24 (75%摔) | 88 | 生存型 ×10 |

## 四类模式（论文 motivation 的骨架）

1. **生存型对价**（stand 即大幅增益，机理=HB manipulation reward 的 stand 乘子结构）：
   spoon ×10、cabinet ×7、powerlift ×9、truck ×2、room ×10、stair/slide（loco 侧）。
2. **推进型对价**（行走/跑步技能直接推进任务进度）：maze ×3.3（walk）、hurdle（run 157±80，部分 episode 真跨栏）、door（run +56%）。
3. **全员 OOD（安全试金石）**：window（19-36 步全摔，初始姿态特殊）、balance_hard（stand 比 zero 还差）。
   迁移机制在这两个任务上的唯一正确行为是**全关**——SC-MCG 的 null-gate 安全性主张的天然测试场。
4. **return 不可分辨**：package（dense 惩罚淹没信号）——必须用 info 分量/事件级指标。

## 负迁移实证（map 直接给出反例：“源越强越通用”是错的）

- balance_hard：stand 19 < zero 32（标准站姿在平衡板上有害）；
- door：walk 25 (62% 摔) < zero 64，但 run 101 > zero——同为步态源方向相反；
- pole：run 81% 摔 vs zero 0%；push：walk/run 把 −192 拖到 −425/−486。

## info 分量层（比 return 高一档分辨率——RIC Relevance 的实证雏形）

对 return 不可分辨/可疑的任务，info 分量直接给出 (source, target, component) 三元相关：

- **package**（return 全员不可分辨）：**reach 是唯一把手带向箱子的源**
  （dist_hand_package_left 1.90→1.22，↓36%）；walk/run 把手距拉到 8.9/25
  （径直走离箱子——细粒度负迁移证据）；walk 的 dist_package_destination
  2.42→1.91 是撞运气伪影（身体撞箱顶过去，同 approach to_dest 教训）。
  ⇒ 重要修正：现成 reach 源就有 package 接近能力，当时手工训 approach 前
  没在 package 上测过 reach 的手距分量。
- **door**：run 的 passage_reward 0→0.21、door_hatch_openness 0.10→0.12
  （run 真在推进通过段）；stand 把 hand_hatch_proximity 0.69→0.33（站着不动
  手不靠近 hatch——"生存源"在推进型任务上的机会成本）。
- **hurdle**：run 的 move 0.17→0.56 + wall_collision_discount 0.17→0.98
  （真在跨栏前进，不只是活着）。
- **truck**：stand 的 reward_robot_package_truck 0.036→0.30（×8）。
- **cabinet**：walk/run 的 door_openness_reward ×4-5（身体擦碰开柜门）。

## 警示（下一步必须验证的 confound）

zero-shot 增益 ≠ 训练时迁移有对价：FastTD3 scratch 可能几千步就自学会站稳
（v1 pilot 教训：教师增量 < scratch 自学速度时迁移无对价）。

## 第二翼结果（scratch 50k 短跑，stamp 20260612T161202Z，2026-06-12）

教师 zero-shot（500 步）按稳态 reward ×2 换算到 eval 口径（1000 步）后：

| 任务 | scratch@5k | @30k | 教师起点（换算） | 对价 |
|---|---|---|---|---|
| hurdle | 5 | 31 | run ≈314 | **强**（scratch 50k 远不及教师起点） |
| cabinet | 26 | 53 | stand ≈204 | **强** |
| powerlift | 88 | 136 | stand ≈342 | **强** |
| maze | 139 | 352 | walk ≈758 | **强**（5k 差 5 倍） |
| truck | 758 | 1164 | stand ≈1202 | 临界 |
| spoon | 227 | 356 | stand ≈146 | **无**（自学站立极快——confound 实锤） |

**主战场任务集（宽 pilot）**：hurdle、cabinet、powerlift、maze（强对价四任务）
+ truck/door（临界/边际）+ window（全员 OOD 安全对照）+ package（hard case 诊断）。
注意：50k 曲线未收敛，100k pilot 中 scratch 可能后程发力——pilot 本身回答此问题。
