# HumanoidBench 任务分类学 v1：特征 schema 预注册

> 2026-07-29。**本文件在提取任何特征、读取任何新 U 标签之前提交。**
> 外部审核批准一次严格限界的静态分析；本文冻结其定位、schema 与停止条件。

## 1. 定位（冻结，不得升级）

本分析产出的是 **HumanoidBench task taxonomy / task-structure prior**，用途仅限：

1. 防止把同 reward family 的任务（如 slide/stair）误当作彼此独立的验证；
2. 构造 reward-family-held-out / terrain-family-held-out / manipulation-family-held-out
   的评估划分；
3. 判断现有 `stand/walk/run` source bank 实际覆盖了哪些能力与动力学类型；
4. 为后续选择真正不同的正迁移、负迁移任务提供结构依据。

**它不是**：迁移性指标、uptake estimand、自动 admission controller，
也不是任何形式的 U 预测器。

## 2. 执行边界（冻结）

- 不训练、不 rollout、不产生任何新的环境交互；
- **不拟合 U**，不搜索"最能解释现有结果"的特征或权重；
- 特征 schema 由本文冻结，**在继续读取其他 U 标签前不得修改**；
- slide/stair 已被看过，**只能作为 hypothesis-generating case，不得再作独立验证**；
- **停止条件**：若最终任务图无法实际改变实验划分、source bank 设计
  或论文的 problem characterization，则该方向停止，不继续扩建。

## 3. 冻结的特征 schema

全部特征只来自两类源文件：`humanoid_bench/envs/*.py`（reward 与 termination）
与 `humanoid_bench/assets/**/*.xml`（场景与机器人）。

### F1. reward 代数（含分量定义，不只是名称）

初版 signature 只比较分量**名称**与组合算子，导致把 Walk 与 ClimbingUpwards
误判为同构（实际 `standing` 一个用绝对 head height、一个用 head 减双脚高度的乘积）。
v1 必须记录：

- 顶层组合算子：`multiplicative` / `additive` / `gated` / `unbounded`
- 每个分量的：名称、权重（加性时）、是否被门控、是否参与 `min()` 组
- 每个分量的**定义指纹**：所依赖的物理量（如 `head_height`、`head−foot_height`、
  `com_velocity`、物体位姿）、`bounds`、`margin`、`sigmoid` 类型
- 目标速度等任务常量的实际取值（如 `_move_speed` 解析后的数值）

### F2. terrain / contact topology

- 地面类型：`plane` / `continuous_mesh` / `discrete_boxes` / `movable_support`
- 若存在高度变化：等效坡度、单步高度跳变、踏面深度
- 是否存在会惩罚接触的障碍（`*_collision_discount` 类因子）

### F3. 自由物体与关节

- 场景中 free joint 物体数量与类型
- 铰接物体的关节类型（hinge / slide）与限位
- 物体质量量级

### F4. 目标变量类型

`velocity` / `displacement` / `posture` / `object_state` / `sparse_completion_event`
（可多选；用于区分"连续进度型"与"稀疏事件型"任务）

### F5. termination / 失败机制

- 是否有提前终止；终止判据依赖的物理量与阈值
- 是否存在稀疏成功终止

### F6. 机器人与观测

- 机器人变体（h1 / h1hand / h1touch / g1）与 obs 维度
- 源策略接入所需的 obs adapter 类型（identity / 切片 / 不兼容）

## 4. 输出物

1. 每个任务一行的特征表（机器可读 JSON + 可读 Markdown）；
2. 由 F1–F6 导出的任务族划分（**不使用 U 标签**）；
3. 一份"现有 source bank 覆盖了哪些族、未覆盖哪些族"的对照；
4. 三种 held-out 评估划分建议。

## 5. 允许的事后检验（必须在 schema 冻结之后）

只允许一次**外部检验**：把已有的十余个 `(source, target)` U 标签单元投影到
任务图上，报告"同族 vs 跨族"的 U 分布差异，作为描述性统计。

- 不得据此调整 schema、族划分或任何权重；
- 不得从中导出预测器；
- 若结果与图不一致，如实记录为图的局限。

## 6. 已知的、必须写进结论的限制

- slide/stair 是**生成假设的案例**，不是证据；
- 几何扰动与源动作有效性的单调关系是**待检验假设**，
  本分析不假定、也不检验它；
- 任务族划分是**基于任务定义的先验**，不蕴含可迁移性。
