# HumanoidBench 任务分类学 v1（阶段一：静态提取）

> 2026-07-29。严格执行 `task_taxonomy_v1_prereg_20260729.md` 冻结的 F1–F6 schema。
> **零训练、零 rollout、未读取任何 U 标签。** U 的外部投影是阶段二，在本文提交后单独进行。

## 1. 提取方法与可复核性

| 特征 | 来源 | 方法 |
|---|---|---|
| F1 reward 代数 + 分量定义指纹 | `envs/*.py` | **AST 解析**（非正则）：追踪 return 变量的全部 `Assign`/`AugAssign`，提取每个 `rewards.tolerance` 的物理量表达式、`bounds`、`margin`、`sigmoid`、`value_at_margin`，并解析类常量（`_move_speed` 等）为数值 |
| F2 地形 / F3 自由物体 | `assets/tasks/*.xml` 递归 include | XML 解析，排除 `class="visual"` 的非碰撞几何；**排除机器人本体**（对所有任务相同） |
| F4 目标变量类型 | 由 F1 的物理量指纹按冻结规则映射 | 正则映射表，见脚本 `F4_RULES` |
| F5 termination | `envs/*.py` 沿 MRO 找 `get_terminated` | AST |
| F6 机器人与观测 | `assets/envs/h1hand_pos_*.xml` | XML，记录 include 链与 `qpos0` 维度/哈希 |

脚本：`extract_task_taxonomy_v1.py`、`extract_task_scene_v1.py`、`build_task_taxonomy_v1.py`。

### 1.1 执行中修正的四处提取器缺陷（必须记录）

初版提取器的判定与我此前手工核准的结果冲突，逐条查源码后确认**四处均为提取器 bug**：

| 任务 | 初版误判 | 根因（已查源码） |
|---|---|---|
| maze | additive | reward 形如 `(加权和) * gate + bonus`，顶层是 `Add` 而非 `Mult`，门控判据漏掉此形式 |
| cabinet | additive | 有 `reward += 100 * subtask` 的 **AugAssign**，初版完全未处理 |
| truck | unknown | `reward = 0` 后全靠多条 `AugAssign`（`+=1000/+=100/-=100`） |
| kitchen | unknown | `return bonus, ...`，返回变量名不是 `reward`；且 `bonus = float(len(completions))` |

修正后新增了 `event_dominated` / `event_count` / `penalty_unbounded` 三个更细的类别，
把此前手工版笼统的 `UNBOUNDED` 拆开。

### 1.2 地形分类的几何语义判据

初版把场景中**所有**几何当作地形，导致 slide 被判为 `structured_obstacles`。
查 XML 后发现 slide 的 3 个 box 是**边界墙**（`size="53 0.15 4"`，薄在 y、高在 z），
地形本体是 9 个 mesh 斜坡；而 stair 的 box 是**水平踏板**（`size="2.7 5.0 0.09"`，薄在 z）。

据此加入纯几何判据（与任何实验结果无关）：

```
最小半长在 z 且 sz < 0.5·max(sx,sy)  →  horizontal_slab（可行走表面）
最小半长在 x 或 y 且 sz > 2·min      →  vertical_wall（边界/障碍）
其余                                  →  object_or_structure
```

## 2. 任务特征表（32 任务，`*` 为现有 source）

| task | reward owner | composition | terrain | free | 目标变量 |
|---|---|---|---|---:|---|
| stand `*` / walk `*` / run `*` | `Walk` | multiplicative | flat_floor_only | 0 | posture, velocity |
| stair | `ClimbingUpwards` | multiplicative | **discrete_slabs** | 0 | posture, velocity |
| slide | `ClimbingUpwards` | multiplicative | **continuous_mesh** | 0 | posture, velocity |
| hurdle | `Hurdle` | multiplicative | flat_floor_with_objects | 0 | posture, velocity |
| crawl | `Crawl` | gated (min) | few_slabs | 0 | posture, velocity |
| maze | `MazeBase` | gated | flat_floor_with_objects | 0 | distance, posture, velocity |
| pole | `Pole` | gated | flat_floor_with_objects | 0 | posture, velocity |
| sit_simple / sit_hard | `Sit` | multiplicative | flat_floor_with_objects | 0 / 1 | object_state, posture, velocity |
| balance_simple / hard | `BalanceBase` | multiplicative | few_slabs | 1 / 2 | posture, velocity |
| highbar_simple / hard | `HighBarBase` | multiplicative | flat_floor_with_objects | 0 | posture |
| door | `Door` | additive | flat_floor_with_objects | 0 | object_state, posture |
| window / spoon | `Window` / `Spoon` | additive | few_slabs | 1 | distance, object_state / posture |
| cube / room / powerlift | 各自 | additive | 各自 | 2 / 6 / 1 | — |
| cabinet / truck / package / basketball / bookshelf×2 | 各自 | **event_dominated** | — | 4 / 5 / 1 / 1 / 12 | sparse_completion_event |
| kitchen | `Kitchen` | **event_count** | mixed_mesh_and_slabs | 1 | sparse_completion_event |
| push / reach | `Push` / `Reach` | **penalty_unbounded** | few_slabs / flat_floor_only | 1 / 0 | **unknown** |

完整表（含每个分量的 bounds/margin/sigmoid 指纹与证据行号）见 `docs/data/task_taxonomy_v1.json`。

**记为 unknown 而未补值的字段**：push 与 reach 的目标变量类型——二者的 reward
不经由 `rewards.tolerance` 构造（直接用距离与成功标志），F4 的映射规则无法机械判定，
按预注册记 `unknown`。

## 3. 族划分（不使用 U 标签）

**by_exact_reward_implementation（21 族）**（更名说明：这是"**完全相同的 reward 实现**"，
不是语义任务相似度；同族只意味着共用同一个 `get_reward` 函数体）：仅 7 族含 ≥2 个任务——
`Walk`{stand,walk,run}、`ClimbingUpwards`{**stair,slide**}、`Sit`{sit_simple,sit_hard}、
`BalanceBase`{balance_simple,balance_hard}、`HighBarBase`、`BookshelfBase`、`Insert`。

**by_terrain（6 族）**：flat_floor_with_objects(11)、few_slabs(9)、discrete_slabs(6)、
flat_floor_only(4)、mixed_mesh_and_slabs(1)、**continuous_mesh(1)**。

**by_composition（6 族）**：multiplicative(12)、additive(6)、event_dominated(6)、
gated(5)、penalty_unbounded(2)、event_count(1)。

**by_object_or_articulation_present**：has_object_or_articulation(19) vs no_free_object(13)。
（更名说明：该维度的实际判据是"场景中存在 freejoint 或 articulated joint"，
**不等于**"任务需要 manipulation 能力"——balance 只因平衡板有 freejoint 即被归入，
原名 `by_manipulation` 有误导性。）

## 4. 三项价值的裁决（预注册 §2 的停止条件）

### 4.1 改变 held-out 实验划分 —— **是**

分类图直接给出一条此前被违反的约束：

> **stair 与 slide 属于同一个 `reward_owner` 族（`ClimbingUpwards`），
> 却分属不同 terrain 族（discrete_slabs vs continuous_mesh）。**

我先前把 stair 当作 slide 的"重复验证场"，在 reward 维度上二者是**同族**，
不构成独立重复；它们真正构成的是**同 reward family 下的跨地形 robustness test**。
这与外部审核的判断一致，并由本图机械地给出，不依赖事后解释。

三种 held-out 划分已生成（`held_out_splits`），规则：同族任务不得跨 train/test 分割。

### 4.2 揭示 source bank 的结构性覆盖缺口 —— **是**（范围限定见下）

> **范围更正（2026-07-29 晚）**：本节的 `SOURCE_TASKS` 只含 `stand/walk/run`，
> 即**本轮评估的 loco bank**。项目另有可直接复用的非 loco 冻结源：
> `checkpoints/terrain_sources/h1hand_{slide,stair,crawl,hurdle,pole}/`
> （已核验 slide/stair 的 checkpoint 存在、18 MB、`global_step=100000`、
> `obs_dim=151` identity adapter，与 loco 源同构）。
> **因此"覆盖为零"仅对 loco bank 成立，不得外推为"项目没有非 loco 源"。**

| 维度 | 源所在族 | 同族 target | 异族 target |
|---|---|---:|---:|
| exact reward implementation | `Walk` | **0** | **29** |
| by_terrain | flat_floor_only | 1（reach） | 28 |
| by_composition | multiplicative | 9 | 20 |
| object/articulation present | no_free_object | 10 | **19** |

**现有 stand/walk/run source bank 与全部 29 个 target 在 reward 实现上无一同族；
在地形上只与 reach 同族；而 19/29 的 target 场景中存在自由物体或铰接关节，loco 源库不含任何操作类源。**

`by_composition` 是唯一存在实质同族覆盖的维度（9 个 target：hurdle、highbar×2、
sit×2、balance×2、stair、slide）——这也正是既往实验中出现过正迁移的区域。

### 4.3 形成可进入论文的问题刻画 —— **是**

可写入 problem characterization 的结构性事实：

> HumanoidBench 的 32 个任务分布在 21 个互不相同的 reward 实现、6 类地形、
> 6 类 reward 组合代数上；而常用的 locomotion source bank（stand/walk/run）
> 只占据其中**一个** reward 族、**一类**地形（flat_floor_only）。
> 因此"跨任务迁移"在该基准上默认就是**跨 reward 族且跨地形族**的迁移，
> 而非同族内的泛化。

三项均有产出，按预注册**不触发停止条件**，可进入阶段二。

## 5. 限制

1. 本文全部为**任务定义级**的静态先验，**不蕴含可迁移性**。
   族划分相同不代表可迁移（slide/stair 同 reward 族而迁移结果不同即为反例），
   族划分不同也不代表不可迁移。
2. F4 对 push / reach 记为 unknown，未补值。
3. 地形分类依赖 §1.2 的几何判据，其阈值（0.5、2×）是几何语义选择，
   非拟合结果；但对介于两者之间的形状可能产生误判，`maze` 的 30 个墙块
   即被归入 `object_or_structure` 而非 `vertical_wall`。
4. 仅 h1hand 变体；其他机器人变体未提取。

## 6. 产物

```
docs/data/task_taxonomy_v1_f1f5.json     F1 分量指纹 + F5 termination（32 任务）
docs/data/task_taxonomy_v1_f2f3f6.json   F2/F3 场景几何 + F6 机器人
docs/data/task_taxonomy_v1.json          合成表 + 族划分 + coverage + held-out splits
```


---

# 阶段二：已有 U 标签的外部投影（唯一一次，纯描述性）

> 在阶段一提交（`77a567a`）之后进行。按预注册 §5：只做描述性检验，
> **不回调 schema、族划分或任何权重**；若结果与图不一致，如实记为图的局限。

## 7. 投影结果

现有 14 个配对 U 单元（3 个 loco 源 × 5 个 target：slide/stair/door/crawl/hurdle）
在四个族维度上的同族/异族分布：

| 维度 | 同族 | 异族 |
|---|---|---|
| reward 族 | **n = 0** | n=14，中位 −3.48，正 5/14 |
| terrain 族 | **n = 0** | n=14，中位 −3.48，正 5/14 |
| composition 族 | n=8，中位 **+8.54**，正 **5/8** | n=6，中位 **−120.35**，正 **0/6** |
| manipulation 族 | n=11，中位 −1.11，正 5/11 | n=3，中位 −30.63，正 0/3 |

## 8. 裁决：**任务图无法用现有 U 标签检验**

### 8.1 reward 族与 terrain 族：无法检验

现有 U 标签中**同族样本数为 0**。三个 loco 源全部属于 `Walk` reward 族与
`flat_floor_only` 地形族，而五个被测 target 无一属于这两族。
因此这两个维度上的任务图**既未被支持也未被否证**——样本结构根本不允许检验。

这本身是对既有实验覆盖的一个诊断：**我们所有的 U 标签都取自"跨 reward 族且跨地形族"
这一种情形**，从未测过同族迁移。

### 8.2 composition 族：表面分离，但与 target 身份完全共线，**不构成证据**

同族 5/8 为正、异族 0/6 为正，看起来分离干净。但：

- 14 个 U 单元只来自 **5 个 target**；
- 同族的 8 个全部来自 slide / stair / hurdle（均为 `multiplicative`）；
- 异族的 6 个全部来自 door（`additive`）与 crawl（`gated`）。

即"同族 vs 异族"与"哪个 target"几乎完全重合。该分离可以被等价地重述为
"loco 源在 slide/stair/hurdle 上有时有正迁移，在 door/crawl 上没有"，
**与 composition 代数是否为解释变量无关**。样本结构不足以分离这两种解释。

### 8.3 因此

**本图在现有数据上不可检验，其价值仅限于阶段一已裁决的三项**
（改变实验划分、揭示 source-bank 覆盖缺口、问题刻画）。
不得据此宣称任务图预测迁移，也不得据此调整任何族划分。

要真正检验它，必须取得**同 reward 族或同 terrain 族**的 U 标签——
例如引入非 loco 源，或在同族任务对之间测迁移。这需要新实验，
超出本次批准范围，且须另行申请。

## 9. 阶段二的方法学副产品

投影暴露的样本结构问题本身应写入论文的实验设计说明：

> 既有全部迁移效用标签（14 个 source×target 单元）仅覆盖 5 个 target，
> 且全部落在"跨 reward 族且跨地形族"这一种配置上；
> 三个源共享同一 reward 实现与同一地形类别。
> 因此现有证据不足以区分"迁移失败源于跨族"与"迁移失败源于特定任务"。
