# 任务间关联在哪一层：reward 代数签名同构不蕴含可迁移

> **2026-07-29 晚，外部审核后修订。** 初稿有两处事实错误、一处把猜想写成了理论，
> 均已逐条查源码核实并修正，见 §0。修正后核心反例更干净、但适用范围更窄。

## 0. 三处修正（外部审核指出，我已查源码逐条确认）

### 0.1 【事实错误】"源与目标的 reward 函数逐项完全相同"—— 不成立

我的 signature 函数只提取了分量**名称**与组合算子，**没有比较分量的定义**。
这是代码缺陷，不只是表述问题。查 `basic_locomotion_envs.py` 实际差异：

| | Walk / Run | ClimbingUpwards（slide/stair） |
|---|---|---|
| `standing` | `tolerance(head_height, bounds=(1.65, ∞), margin=0.4125)`，**绝对高度** | `tolerance(head−left_foot, (1.2,∞), 0.45) × tolerance(head−right_foot, (1.2,∞), 0.45)`，**相对高度、两项乘积** |
| `upright` bounds | 0.9 | **0.5** |
| `move` 目标速度 | Walk `self._move_speed`=1；**Run=5** | 硬编码 `_WALK_SPEED`=1 |

（核实：`_STAND_HEIGHT=1.65`、`_WALK_SPEED=1`、`_RUN_SPEED=5`；
`Run(Walk)._move_speed=_RUN_SPEED`；`ClimbingUpwards.get_reward` 硬编码 `_WALK_SPEED`。）

正确表述只能是 **reward algebraic signature 同构**（顶层代数形式相同），
**不是** reward function 相同——尤其 Run 与 slide/stair 的目标速度差 5 倍。

**但真正的反例反而更干净，且不受此影响：**

> **slide 与 stair 使用完全相同的 target reward class（同一个 `ClimbingUpwards.get_reward`、
> 同样的常量）、相同机器人、相同源策略，却因 terrain dynamics 不同而产生
> 强正（U=+56.95）与近零（U=+0.19）的迁移效果。**

这足以证明：**target reward 结构本身不足以决定迁移效用。**
原先那个"源-目标 reward 同构"的说法既错误又不必要。

### 0.2 【事实错误】"九个信号族没有一个看过环境的物理结构"—— 不成立

zero-shot rollout、T⁰、SHU、matched-state probe、source transition、critic、influence
**全部是在真实目标物理环境中产生的数据**，它们已经隐式经历了地形动力学。

准确表述：**既有信号没有显式建模 source–target dynamics mismatch 与 contact topology，
而是把物理作用压缩成 return / Q / 行为统计 / 更新结果。**
这是"物理信息被聚合丢失"，不是"从未看过物理"。
（此表述与本线一贯的主题——标量聚合丢失结构——一致。）

### 0.3 【猜想被写成了理论】几何单调性

初稿称几何层"在方向上单调而非歧义"。这是**未经检验的假设**，不能作为方向理论：

- 几何差异没有统一的标量顺序；
- 连续/离散不是大小关系；
- 较大的地形变化可能有害，也可能产生有价值的新状态；
- 几何影响依赖具体 source policy 的步幅、抬脚高度与反馈能力。

slide/stair 支持的只是：**离散台阶可能破坏平地 walk policy 的接触与步态有效性**，
**不支持**"几何距离越大越不可迁移"这一般命题。

---

（以下为初稿正文，保留原始记录；凡与 §0 冲突处一律以 §0 为准。）

# 初稿：任务间关联在哪一层

> 2026-07-29。零训练、零 rollout，只读 reward 源码与 MJCF 场景定义。
> 起因：PI 提出的新问题——"有什么办法能挖掘出 HumanoidBench 这 30 个任务之间
> 隐藏的关联"。本文给出第一个可证伪的答案，并明确它**不是**一个迁移性预测器。

## 1. 这个问题与已封存路线的区别

此前九个失败的信号族做的都是**点预测**：给定 (source, target) 预测学习效用 U。
本文做的是**结构发现**：30 个任务构成什么样的关联景观。后者不看 U 标签即可构造，
其定位是 **problem characterization**，不是 uptake estimand，
因此不属于"不得复活的代理量搜索"。

区别的操作性判据：**构造过程中是否使用了 U 标签**。本文全部结论只用
reward 源码与 MJCF 场景文件，U 标签仅在最后用作外部检验。

## 2. 第一层：reward 代数的关联（零成本，已完成）

把 17 个有界 target 与三个源任务写成同一格式的签名
`(组合算子, 通用因子集合, 任务专属分量集合)`，得到**完全同构**（逐项相同，
不是"相似"）的三对：

| 源任务 | 目标任务 | 共同的 reward 代数 |
|---|---|---|
| `walk` / `run` | **slide** | `small_control × stand_reward × move` |
| `walk` / `run` | **stair** | `small_control × stand_reward × move` |
| `stand` | **balance_hard** | `small_control × stand_reward × dont_move` |

实测迁移结果：

| 对 | 学习效用 |
|---|---|
| walk → slide | **U = +56.95**，90%CI [+41.69, +72.20]，强正迁移 |
| walk → stair | **U = +0.19**，90%CI [−5.35, +5.72]，无效 |
| stand → balance_hard | **无 U 标签**（未做过学习效用实验）；zero-shot 行为为负（19 < zero 的 32） |

**结论（可证伪）：源任务与目标任务的 reward 函数逐项完全相同，
同一个源，学习效用可以是强正、也可以是零。**

这是最强的同构条件。连"reward 函数完全一致"都不足以蕴含可迁移，
那么任何更弱的 reward 相似度（分量重叠、权重接近、per-step reward 接近）
都不可能是充分条件。

**这一条同时解释了此前九个信号族的共同盲区**：它们全部建立在
reward / value / 行为观测之上，**没有任何一个看过环境的物理结构**。

## 3. 第二层：几何与动力学的关联（本文新增）

HumanoidBench 的场景定义把差异隔离得非常干净。
`assets/envs/h1hand_pos_{slide,stair}.xml` 中，机器人文件、初始姿态 `qpos0`
（75 维逐位相同）、solver 与 timestep 全部一致，**唯一的差异是 `tasks/*.xml` 一个 include**。

| | slide | stair |
|---|---|---|
| 几何类型 | `mesh` 连续斜面 | `box` 离散台阶 |
| 定义 | 顶点 (0.3, 0) → (5.3, 1.75) | 台阶高 0.18 m，踏面深 0.6 m |
| **等效坡度** | 1.75 / 5.0 = **0.35** | 0.18 / 0.6 = **0.30** |
| 接触模式 | 连续，高度沿步幅渐变 | 离散，单步需抬高 0.18 m |

**平均坡度几乎相同（0.35 vs 0.30）。二者的差异不是"更陡"，而是"连续 vs 离散"。**

对一个在平地上训练出来的 `walk` 源：

- 斜坡上，每步的高度变化连续摊在步幅内，平地步态基本仍然可用；
- 台阶上，必须离散地抬腿 0.18 m（约 20% 腿长）才能上一级，
  平地步态的抬腿高度不足，动作直接撞在台阶立面上。

这给出了 §2 那个反例对的**物理解释**，且该解释所依据的信息
（几何类型、尺度、离散性）**完全来自 MJCF，零 rollout 可提取**。

## 4. 这一层为什么有可能满足"方向理论"的要求

外部审核对重启 uptake 指标提出四个前置条件，其中最难的一条是
"对为什么正值代表有益、负值代表有害有明确的方向理论"。

几何层至少在方向上是单调而非歧义的：

```
目标几何相对源训练几何的扰动幅度  ↑   ⟹   源策略的动作有效性  ↓
```

这与 effect-space 距离的困境不同——后者"距离近可能是易吸收也可能是冗余，
距离远可能是有价值探索也可能是有害 OOD"，符号不定。
而几何扰动对**一个冻结策略的动作有效性**而言方向是确定的。

**但必须说清楚：这不足以使它成为 uptake estimand。**
几何是静态的，不满足 student-relative 与 stage-conditioned 两条。
它回答的是"哪些任务对之间**原则上**可迁移"，
不回答"当前这个 student 在这个阶段能否吸收"。二者是不同层次的问题。

## 5. 因此三层图景更新为

```
Transfer Utility ≈ Target Relevance × Student Uptake − Harm/Interference

  Target Relevance   reward 代数 / BAC / P3     —— 已有部分结果，但§2 证明它不充分
  【新增】Dynamics Compatibility  几何与接触结构  —— 本文提出，零 rollout 可提取
  Student Uptake     stage-conditioned          —— 仍是开放问题
  Harm/Interference  exact abstention / 生命周期 —— 已有机制
```

§2 的贡献是**否定性的且强**：它证明第一层单独不充分，
并指出所有既往信号族共享同一个盲区。

## 6. 限制（必须与结论同等醒目）

1. **只有一个决定性反例对**（slide vs stair）。balance_hard 缺 U 标签，
   目前只能作为待检验的第三个案例，不能计入证据。
2. **几何层尚未被系统提取，也未被检验过。** §3 是对单一任务对的人工阅读，
   不是一个跨 30 任务的自动化分析，更没有前瞻验证。
3. **不得将几何特征直接拟合 U。** 那会使本文退化为第十个代理量搜索。
   若要推进，正确的做法是先无监督地构造任务关联图（不看 U），
   再用已有的十余个 U 标签单元做**外部检验**，且检验规则须预注册。
4. 本文**不构成**重启 uptake 指标实验的申请。按已冻结的裁决，
   本线不再追加训练实验。

## 7. 数据与复核

```
reward 代数签名   configs/reward_structure/humanoidbench_v1.py（17 target，逐个从源码核准）
源任务 reward     humanoid_bench/envs/basic_locomotion_envs.py::Walk
场景定义          humanoid_bench/assets/envs/h1hand_pos_{slide,stair}.xml
                  humanoid_bench/assets/tasks/slide.xml
                  humanoid_bench/assets/locomotion/generated_xml_stairs.xml
U 标签            docs/data/{slide,stair}_bac_gate_v1/*_results.json
```
