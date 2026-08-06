# 求审查：迁移性可能是 **元数据**（任务规格）的函数，而不是**数据**的函数

> 2026-07-30。发 ChatGPT 审查。**本文不含任何新实验结论**，只有源码事实 + 回溯一致性 + 一个待检验的假设。
> 背景：九个信号族全部失败，最近一次是 `QMP_FIDELITY_PARTIAL`（per-state critic 选源退化为 student）。

## 1. 触发点：一个我一直没查的源码事实

九次失败的信号族**全部**试图从**交互数据**中估计迁移性
（zero-shot 行为 / T⁰ / SIV / SHU / adaptive revocation / P0 lease /
update-space influence / T^critic / BAC / per-state QMP）。

没有一次去读 **target reward 函数的数值规格**。读了之后：

```python
# humanoid_bench/envs/basic_locomotion_envs.py
_WALK_SPEED = 1 ;  _RUN_SPEED = 5 ;  _STAND_HEIGHT = 1.65 ;  _CRAWL_HEIGHT = 0.8

class Walk(Task):            _move_speed = _WALK_SPEED   # 1
class Stand(Walk):           _move_speed = 0
class Run(Walk):             _move_speed = _RUN_SPEED    # 5
class Hurdle(Walk):          _move_speed = _RUN_SPEED    # 5
class ClimbingUpwards(Walk): move 用 _WALK_SPEED 硬编码   # 1
class Stair(ClimbingUpwards): pass
class Slide(ClimbingUpwards): pass
# humanoid_bench/envs/pole.py
_WALK_SPEED = 0.5            # 重定义！
class Pole:                  _move_speed = _WALK_SPEED   # 0.5
```

**三个源的训练规格**：`stand=0`、`walk=1`、`run=5`。这是它们各自 target 的 `_move_speed`。

## 2. 与已有 U 标签的回溯对照

已有的任务内 U 排序（全部来自本项目的等剂量因果实验）：

| target | reward 的关键门控 | 预测 | **真实 U 排序** | 一致 |
|---|---|---|---|:--:|
| hurdle | `move` 单边 `bounds=(5,∞)` | 只有 run 达标 | **run(+380) > walk(+105) > stand(+51)** | ✓ |
| slide / stair | `move` 单边 `bounds=(1,∞)`，另有 `standing = (head−foot ≥ 1.2)` 门控 | walk/run 达标 move，run 高速不利于 standing | **walk(+56.95) > run(+16.90) > stand(−1.21)** | ✓ |
| crawl | `crawling_head` **双边** `bounds=(0.6, 1.0)` | 所有直立源（head≈1.65）超上界→乘性归零 | **全负**（run −208 > walk −217 > stand −449） | ✓ |
| door | 无速度分量；有 `hand_hatch_proximity` / `door_openness` | loco 源无手部技能 | **全负**（walk −22.2 > run −30.6 > stand −32.6） | ✓ |

框架：HB 的 reward 是**乘性**的 `Π_c tolerance(f_c, bounds_c)`；
每个分量声明一个可接受区间（单边或双边）；任一门控归零则整体归零。
⇒ **源的效用 ≈ 它能同时满足多少个 target 门控分量。**

它还顺带解释了 M19 的反例：door-run 的 zero-shot 位移 +58% 却 harmful——
因为 door 的 reward **没有速度项**，位移只经 `passage_reward` 起作用，而 passage 需要先开门。

## 3. 与三个**已被否定**方案的区别（这是最需要审查的部分）

我知道 M15：新方案撞旧负结果是最伤公信力的错误。逐个论证：

| 已否定方案 | 它用的量 | 本假设用的量 | 是否同一个东西 |
|---|---|---|---|
| **BAC**（无增量） | rollout 实测的 reward 分量值 `x_c`，与 zero/student baseline 比差值 | reward 函数的**超参数**（bounds 的位置、单双边）+ 源的**训练规格常量** | **不同**：BAC 必须交互采样；本假设**零交互**，纯读源码常量 |
| **sibling gate**（方向依赖） | 两个 target 是否共用同一 reward **实现** | target 要求的**数值目标** vs 源的**训练目标** | **不同**：slide/stair 共用实现（sibling 的对象），本假设说的是二者都要求 speed=1 故 walk 源最优 |
| **taxonomy**（不可测） | reward 的**代数结构**同构 | reward 的**数值超参数** | **不同**：结构 vs 数值 |

**我自己看到的最大破绽**：BAC 的 `x_c` 在 rollout 里已经隐含了 bounds 信息
（`tolerance` 的输出值本身就编码了达标程度）。所以"零交互"是唯一实质区别。
若审查认为这不构成实质区别，本假设应当直接判死。

## 4. 我自己的风险自查

1. **n=4 回溯，且我是在看过全部答案之后才发现这个规律的。** 这是最严重的问题。
   任何回溯一致性在这里都近乎无价值，必须前瞻检验。
2. **多分量的权衡我并没有真正建模**。slide 上"walk > run"我给的解释是
   "run 高速不利于 standing 门控"——这是**事后编的**，没有测量支撑。
   若只看 move 分量，walk 和 run 都达标，应当相近，而真实差 40。
3. **单边 tolerance 的性质使预测退化**：`bounds=(x,∞)` 下"能力越强越好"，
   于是 run 在多数 locomotion target 上都该达标——但 crawl/door 上它并不好。
   所以真正起作用的可能是**门控分量**而不是 move 分量，而门控分量是否可从
   源的规格预测，我**没有**证据。
4. **可能存在更平凡的解释**：例如"源与 target 的 episode 存活时长匹配"，
   或"源在 target 上不摔倒的概率"。我没有排除。

## 5. 提议的前瞻检验（求审查设计）

前瞻性是唯一能救这个假设的东西。候选设计：

**方案 A：pole（`_move_speed = 0.5`）**
- 预测：`bounds=(0.5,∞)` 单边 → walk(1) 与 run(5) 都达标、stand(0) 不达标 → 预测 `{walk, run} > stand`；
- 需先确认本项目**没有** pole 作为 target 的等剂量 U 标签（初查 inventory 里出现过 pole，需核实）。

**方案 B：找一个 `_move_speed = 0` 的 target（如 balance/sit 类）**
- 预测 **stand > walk > run** —— 这是**最强的**前瞻预测，因为 stand 在已测的
  4 个 target 上**全部垫底**，规则若能预测出"stand 反而最好"且成立，几乎无法用平凡解释掩盖。

**我倾向方案 B**，因为它的预测方向与全部历史数据相反，证伪力最强。

## 5b. 补充事实（写于送审后、Codex 返回前，供综合裁定）

查完源码后，方案 B 的靶子基本确定为 **balance**：

```python
# humanoid_bench/envs/balance.py  BalanceBase.get_reward
return small_control * stand_reward * dont_move, {...}
#   dont_move = tolerance(horizontal_velocity, margin=2).mean()   ← 直接惩罚水平速度

# humanoid_bench/envs/basic_locomotion_envs.py  Walk.get_reward
if self._move_speed == 0:          # 即 Stand 源的训练目标
    dont_move = tolerance(horizontal_velocity, margin=2).mean()
    return small_control * stand_reward * dont_move, {...}
```

**balance 的 reward 与 stand 源的训练 reward 逐项同构**（唯一差异：balance 的
`standing` bounds 是 `_STAND_HEIGHT + 0.37`，因为站在板上）。
⇒ 规则给出的预测是 **stand > walk > run**，而 stand 在已测 4 个 target 上**全部垫底**。

三点必须同时披露：

1. **标签可测性**：`label_identifiability_audit_20260727.md` 全表中，
   balance_hard 的 `U/trend = 1.05`、`seedCV = 0.044` 是**非归档候选里最优的**
   （door 是 1.42 / 0.101）。统计上它是最好的判决场。
2. **它没有 RBO 学习效用标签**——此前只有一条 **zero-shot 行为**先验
   （bank 注释："所有 loco 源 zero-shot 全摔"）。该审计文档自己已更正：
   *"不能由 zero-shot 全负推出 RBO delayed learning utility 全负"*（正是 M19）。
   所以学习效用层面它仍是盲的。
3. **但我已经读到过那条 zero-shot 先验**，且该文档当时以"避免结果导向筛选"
   为由排除了它。缓解论据是：**我的规则预测方向（stand 最好）与那条先验
   （全员 OOD/全负）相反**，因此不构成朝有利方向的筛选。是否足以缓解，请裁定。

**必须预先处理的平凡解释**：stand 源近乎 no-op，在一个"要求静止"的 target 上，
"提供了平衡技能"与"只是不干扰"无法区分。
拟用 **zero-action 对照臂**（本项目 package coverage matrix 用过同款方法）来分离：
仅当 `stand > zero-action` 时才可声称 stand 提供了真实技能。

## 6. 具体想请你回答

1. §3 的"零交互"是否构成相对 BAC 的实质区别？若否，本线应当直接关闭吗？
2. §4.2/§4.3 的破绽（多分量权衡是事后编的、单边 tolerance 使 run 恒达标）
   是否已经足以判定该假设不成立？
3. 若值得继续，前瞻检验选 A 还是 B？B 的 target 该选哪个（需满足：
   `_move_speed=0` 或极低、本项目无 U 标签、标签可测性可能通过）？
4. 是否存在我没想到的平凡解释（§4.4）需要在预注册里先排除？
5. 更根本的：**"迁移性是元数据的函数而非数据的函数"**这个主张，
   如果前瞻检验成立，够不够作为一篇论文的核心贡献？还是说它太简单/太任务特定？
