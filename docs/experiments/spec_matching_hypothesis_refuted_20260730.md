# 规格匹配假设：`REFUTED`（未进入实验即被源码事实证伪）

> 2026-07-30。假设见 `docs/agent_collab/spec_matching_hypothesis_review_20260730.md`。
> 外部对抗性审查（Codex/GPT，`task-ms6yyaf7-f3e30i`）指出多处源码事实错误，
> 我逐条核实**全部成立**。**该假设终止，不进入任何实验。**
> 本文同时记录我在本轮犯的两个严重方法错误。

## 1. 裁决

**假设**：迁移性可从 target reward 的数值规格（bounds 位置、单双边、乘性门控）
与源的训练规格常量中读出，无需交互。

**裁决：`REFUTED`。** 在它自己声称的最佳案例上就被证伪：

```
slide 与 stair 共用同一个 reward 函数 ClimbingUpwards.get_reward,
数值常量完全相同(move 用 _WALK_SPEED=1, standing bounds=1.2, upright bounds=0.5)。
walk 源的 U:   slide  +56.95 [3/3 seed 正]
               stair   +0.19 [−5.35, +5.72]  跨零 uncertain
```

**同一份 spec，两个完全不同的效用。** 规格匹配无法产生这个差异，
因为它读到的输入在两个 target 上逐字节相同。

## 2. 我在假设中犯的源码事实错误（逐条核实成立）

### 2.1 "HB 的 reward 是乘性门控" —— 错

只有 `ClimbingUpwards`（slide/stair）与 `Walk/Stand` 是纯乘积。我据此外推到全部任务，错。

```python
# crawl —— 加权和,再乘 in_tunnel(basic_locomotion_envs.py:154)
reward = (0.1*small_control + 0.25*min(crawling, crawling_head)
          + 0.4*move + 0.25*reward_xquat) * in_tunnel

# door —— 五项加权和(door.py:117)
reward = (0.1*stand_reward*small_control + 0.45*door_openness_reward
          + 0.05*door_hatch_openness_reward + 0.05*hand_hatch_proximity_reward
          + 0.35*passage_reward)
```

**后果**：
- 我说"crawl 的 `crawling_head` 双边 bounds 使直立源乘性归零"——**错**。
  它只通过 `0.25 * min(crawling, crawling_head)` 进入加权和，而 `move` 独占 0.4。
- 我说"door 的位移只经 passage 起作用，而 passage 需要先开门"——**reward 代数上错**。
  `passage_reward = tolerance(imu_x, bounds=(1.2, ∞))` 是**独立加项**，占 0.35，
  不与 `door_openness_reward` 相乘。门是否挡路属 dynamics，需 MJCF 证据，
  不能记在 reward 代数账上。因此我对 **M19 反例的"解释"作废**。

### 2.2 忽略了 shaping 的下界 —— 错

```python
move = (5 * move + 1) / 6            # tolerance=0 时仍为 1/6,不归零
small_control = (4 + small_control) / 5   # 原值 0 时仍为 0.8
```

我在读 `ClimbingUpwards` 时看到过这两行却没意识到含义。
"只读 bounds 判达标"丢掉了 shaping 的主要数值结构。

### 2.3 单边 tolerance 使预测退化

`bounds=(x, ∞)` 下"能力越强越好"，walk 与 run 在多数 locomotion target 上都判"达标"，
规则**无法**预测二者次序——而次序恰恰是 slide(+56.95 vs +16.90) 的全部信息量所在。

## 3. 我犯的两个方法错误（比事实错误更严重）

### 3.1 没有查自己两天前的既有成果

`configs/reward_structure/humanoidbench_v1.py`（**我自己在 BAC 线写的**）里：

```python
"h1hand-crawl-v0": dict(kind=GATED,  min_groups=[(0.25, ["crawling","crawling_head"])], ...)
"h1hand-door-v0":  dict(kind=ADDITIVE, weights={"door_openness_reward": 0.45, ...})
```

**我两天前就正确分类过 crawl=GATED、door=ADDITIVE，这次却把它们当纯乘积。**
这是 M15（"方案设计前必须先检索本项目既有证据"）的直接违反，
而且违反对象是我自己的产出。

同理 `scripts/analysis/extract_task_taxonomy_v1.py`（也是我写的）
已经在提取 `_move_speed`（:74/:103）与 `bounds/margin/sigmoid/value_at_margin`（:130-137）。
所以我声称的"此前没人读过数值规格"**是假的**——新颖性主张不成立。

### 3.2 选择性省略了最不利的数据

送审文档 §2 的表格把 `slide / stair` 合并为一行，填入 `+56.95 / +16.90 / −1.21`
——**全部是 slide 的数字**。stair 的真实结果是 `−5.74 / +0.19 / −1.11`，三者全部跨零。

我在**上一轮的 sibling gate 里刚刚写过 stair 的数字**，因此这不是无知，
是呈现时的选择。合并同族任务看似合理，实际效果是让最不利的案例消失，
而那个案例恰恰是本假设的决定性反例（§1）。

**这是本轮最严重的问题。** 任何"任务族合并"今后必须逐 target 列出，不得聚合。

## 4. 接受的裁决与保留

- **作为迁移性预测器：终止。** 不启动方案 A(pole) 或 B(balance)。
- **保留的最大用途**：静态规格至多是一个**零成本的 compatibility pre-filter**
  （排除明显不兼容的源，例如无手部技能的 loco 源之于 door 的操作分量），
  **不能**预测"该源是否会改善后续 student"。
- **不得**以"survival 也是 spec 的隐含部分"之类的事后扩充来抢救。

## 5. 这轮真正留下的东西

审查同时确认了本项目已有的正确方向（`PAPER_CONTRIBUTION_RESTRUCTURE_20260728.md:197`）：
迁移效用必须写成条件分布

```
U ~ p( U | source, target, θ_t, D_t, occupancy_t, channel, dose, K )
```

而规格匹配是一个**只含 (source, target) 的静态量**，缺失 θ_t / occupancy / channel /
dose / horizon 全部条件变量。slide-vs-stair 的同 spec 异效用正是"缺失 transition kernel
与 occupancy"这一条的直接实证。

**因此本轮的净产出是：为"迁移效用不是 (source,target) 的函数"这一核心论断
补上了一个此前没有的、极强的反例——同一 reward 实现、同一数值常量、
同一源，效用 +56.95 vs +0.19。**

## 6. 数据

```
审查记录   docs/agent_collab/spec_matching_hypothesis_review_20260730.md（含我的原始错误，保留不改）
Codex 会话 019fb11e-fc4d-70b1-90e9-11e9e87f0b86
源码事实   humanoid_bench/envs/basic_locomotion_envs.py:129-160, door.py:109-125, pole.py:65-90
自有成果   configs/reward_structure/humanoidbench_v1.py, scripts/analysis/extract_task_taxonomy_v1.py
反例数据   docs/experiments/{slide_bac_gate_v1,stair_bac_gate_v1}_results_2026072{8,9}.md
```
