# 预注册：迁移后自主性（post-transfer autonomy）判决场普查

> 2026-08-06。**本文件必须在生成任何汇总结果之前提交 git。**
> 提交后只允许修改路径参数，不得修改任何判据、阈值、分类规则或 dev/holdout 划分规则（CLAUDE.md §4）。
>
> 触发来源：PI 于 08-06 指出十二族 + racing 的 estimand 恒为 `U` 本身，
> 从未试图改变 `U` 的可达上界；新方向应指向"源退出后 target 能否突破源诱导的起点平台"。
> 外部 review（ChatGPT）追加修正：不得只用理论奖励上限筛场地。

---

## 0. 本普查**不是**第十三个信号族

族 12（`progress_screen_v1`）刚以 `HOLDOUT_FAILED` 关闭，M33 记录了教训。
必须先说清本普查与它的区别，否则就是换皮（CLAUDE.md §1）：

| | 族 1–12 | 本普查 |
|---|---|---|
| estimand | `U(source, target)` —— 哪个源对这个 target 有用 | **场地的可测量性与剩余空间** —— 哪个 target 能检验某个机制 |
| 输出用途 | 直接做选源/准入决策 | 只决定**在哪里做实验**，不进入任何科学结论 |
| 失败后果 | 决策错误 | 选错场地 → 浪费算力，但不产生错误结论 |

先例：`project_label_measurability`（标签可测性）同样是用先验量做前置筛选，
不预测 `U`，已被接受为合法。**本普查的任何输出都不得被引用为迁移性证据。**

---

## 1. 候选池（冻结）

全部 HumanoidBench h1hand 任务，排除已被明令关闭或已作源的：

```
排除  stand / walk / run / reach          —— 本项目的源，非 target
排除  door                                —— 判决场已关闭（M31：U 符号跨 learner 反转）
排除  kitchen                             —— success_bar=4，语义与其余任务不可比
候选  hurdle slide stair crawl sit sit_hard
      truck cabinet bookshelf_simple bookshelf_hard package maze basketball
      balance_simple balance_hard highbar_simple highbar_hard
      pole powerlift push window room insert spoon cube
```

`hurdle` 与 `slide` **必须参与普查**（作为已知的对照点，用于验证普查规则本身
能正确把它们判为 `SATURATED`）；但按 §5 的划分规则，它们**不得**被选为 dev 或 holdout。

---

## 2. 三种 headroom 的定义（冻结）

ChatGPT 的核心修正：理论上限只能用于排除，不能证明存在可利用空间。故分三层，
**必须分别报告，不得合并成单一分数**。

### 2.1 raw headroom（仅用于快速排除）

```
H_raw(T) = J_theory_max(T) − J_best_known(T)
```

`J_theory_max` 的取法（静态推导，不看实验数据）：

- reward 为若干 `[0,1]` 项相乘且无稀疏加项 → `max_episode_steps × 1.0`
- reward 含稀疏成功项（如 `package` 的 `reward_success * 1000`）→ 标 `UNBOUNDED_ANALYTIC`，
  **不估算**，直接依赖 §2.2 与 §2.3

`J_best_known` = 本仓库该 target 上任何 source-free 评估的最大值；无数据则 `UNKNOWN`。

### 2.2 operational headroom（主用）

用 HumanoidBench **官方** `success_bar` 作为"任务被认为解决"的门槛：

```
H_op(T) = success_bar(T) − J_best_known(T)
```

`H_op ≤ 0` ⟹ 该 target 已被解决，**判为 `SATURATED`**，无论 raw headroom 还有多少。

> 这是本普查最重要的一条。已核实：`Walk.success_bar = 700`，而 `Slide`/`Hurdle`
> 均继承 `Walk`。slide hard-exit 已达 929.1、hurdle 端到端 C 已达 840.4，
> **两者按官方标准都已解决**。这个判定先于任何新数据成立，不可事后调整。

### 2.3 milestone headroom（决定性）

只用任务**自己**在 `info` 里给出的字段，不用 evaluator 的聚合量：

```
可用字段   info["success"]            布尔，任务自定义
           info["success_subtasks"]   整数，已完成子阶段数
```

**CLAUDE.md §6 强制**：`p0_evaluator.py` 的 `aggregate.success_count` 读自
`terminated`，在 locomotion 上是**摔倒早停**计数而非成功率。本普查**禁止**使用它。
缺少上述两个字段的 target 一律标 `NO_MILESTONE_SIGNAL`。

```
H_ms(T) = max_subtasks(T) − best_observed_subtasks(T)
```

`H_ms > 0` 且该 milestone 从未被任何臂触发 ⟹ 存在真实能力缺口，即使 return 已不低。

---

## 3. 四道 gate（冻结，全部通过才成为 CANDIDATE）

| Gate | 判据 | 缺数据时 |
|---|---|---|
| **G1 前期 scaffold 有效** | 存在源使 0–30k 的 **source-free** 表现优于 scratch，且 3/3 seed 方向一致；**禁止**使用源自身执行时的 reward（族 12 已证其与 `U` 可反向 7.9×） | `UNKNOWN_G1` |
| **G2 退出后仍有缺口** | `H_op > 0` **或** `H_ms > 0`（干净 hard-exit 之后仍未达 `success_bar` 或仍有未触发的 milestone） | `UNKNOWN_G2` |
| **G3 源本身解不了终点瓶颈** | 源策略 zero-shot 在该 target 上不触发 `success` / 不推进 `success_subtasks` | `UNKNOWN_G3` |
| **G4 可测量** | 效应可被 3 个 learner seed 区分：milestone 事件率不是极罕见二值（`>5%` 的 episode 触发），或 return 的 learner 间 sd 小于待测效应的预期尺度 | `UNKNOWN_G4` |

**G1 与 G3 不矛盾**：G1 要求源能帮上前期（如 locomotion 前置能力），
G3 要求源帮不到终点（如 object interaction）。同时满足才是"跳板"场景。

---

## 4. 输出分类（互斥、穷尽）

```
SATURATED              H_op ≤ 0 且 H_ms ≤ 0            —— 已解决，无展示空间
NO_SCAFFOLD            G1 失败                          —— 源前期无帮助，非迁移问题
NO_POST_EXIT_DEFICIT   G2 失败                          —— hard exit 已解决，不需新机制
SOURCE_SOLVES_IT       G3 失败                          —— 源直接解决，无"超越"故事
UNMEASURABLE           G4 失败                          —— 信号不可辨识（cabinet@10k 的教训）
CANDIDATE              四门全过
UNKNOWN                任一门缺必要数据
```

**`UNKNOWN` 不得被推断补齐**（CLAUDE.md §4）。脚本遇到缺数据必须输出 `UNKNOWN`
并在汇总末尾列出"要把它变成已知需要跑什么"，**不得**落入任何实质裁决分支。

---

## 5. dev / holdout 划分规则（先于数据冻结，无自由度）

这是防 outcome-informed site selection（§8.5）的关键。规则是确定性的：

```
1. 取全部 CANDIDATE
2. 按 success_bar 降序排序；success_bar 相同则按任务名字典序
3. 排名第 1  → development target（机制、超参、阈值只在此确定）
4. 排名第 2  → holdout target（冻结，只在机制定稿后运行一次正式评估）
5. CANDIDATE 少于 2 个 → 输出 INSUFFICIENT_SITES 并停止，不得降低门槛凑数
```

**排序键 `success_bar` 是任务的静态属性，与本项目任何实验结果无关**，
因此排序在看到数据之前就已确定，不存在挑选自由度。

`hurdle`、`slide` 即使意外通过四门也不得入选（§1），保留其原有角色：
slide = 生命周期机制对照，hurdle = 早期加速与稳定性对照。

---

## 6. 设计层自查（CLAUDE.md §8）

**8.1 辨别力 —— 一个平凡解释**：
"`H_op > 0` 只是因为该任务难，与迁移毫无关系。" 这个解释能通过 G2 单门。
排除方式：G1 强制要求源在该 target 上**已被验证有前期收益**（source-free 口径），
G3 强制要求源**解不了**终点。单纯的"难任务"会在 G1 上失败（源帮不上任何忙）。
三门联合才刻画"跳板"，任一单门都不充分。

**8.2 混淆变量**：`success_bar` 与任务难度共变，故高 `success_bar` 的任务
天然更可能通过 G2。这不构成混淆，因为 `success_bar` 只用于**排序**（§5），
而入选资格由四门决定，与排序键无关。

**8.3 独立重复**：本普查不产生科学结论，不需要独立重复；
但由它选出的 dev target 上的后续机制实验必须用新 learner seeds（M24）。

**8.4 前提是否蕴含结论**：验算 —— G2（`H_op > 0`）是否蕴含 CANDIDATE？
不蕴含，因为 G1/G3/G4 独立。反向验算 —— 是否存在任务必然全过四门？
不存在：G1 要求源有用、G3 要求源没用（在不同阶段），构成实质约束。
**否定分支可达**：若全部候选在 G1 失败，输出 `INSUFFICIENT_SITES`。

**8.5 site selection**：本普查**就是** site selection，故用 §5 的确定性规则
消除自由度，并强制预留 holdout。所有机制结论最终必须在 holdout 上复现，
否则只能声称"在 development target 上成立"。

**8.6 是否重演本轮教训**：
- M33（降低用法强度救不了失败信号空间）—— 本普查不预测 `U`，见 §0，不是族 13；
- M33（采了数据却用推理代替查询）—— 本预注册所有静态规格数字均已 `grep` 原文核实，
  `success_bar` 全表见 §2.2 的核实记录；
- M32（3 个事后点不得写成判据）—— 本普查的产出是**场地分类**，不是判据；
- M30（outcome-contingent gate switching）—— 四门与分类规则本文件冻结，
  数据出来后不得增删任何一门。

---

## 7. 执行与产物

```
脚本      scripts/analysis/screen_post_transfer_sites.py   （本文件提交后才允许编写）
输出      docs/data/post_transfer_site_screen_v1/screen.json
汇总      docs/experiments/post_transfer_autonomy_site_screen_results_20260806.md
```

脚本要求：

- 数据不全 → 输出 `UNKNOWN` 并**非零退出**（CLAUDE.md §4）
- 缺失统计**独立扫描全部 (target × gate) 组合**，不得在前置门失败时 `continue`
- 每个数字必须给出来源文件路径，便于逐条核实

---

## 8. 本普查**不能**回答的问题（防止越界引用）

1. 不能说明任何源对任何 target 是否有用 —— 那是 `U`，本普查不测；
2. 不能说明 hard exit 是否足够 —— 那要 §9 的三臂 gate；
3. 不能说明机制是否有效 —— 那要机制实验本身；
4. `success_bar` 是 HumanoidBench 作者设定的门槛，**不是**本项目的发现，
   引用时必须注明出处为上游 benchmark。

---

## 9. 通过之后的下一步（此处只登记，不在本文件冻结判据）

对 development target 做三臂 gate：`scratch` / `continuous source` / `fixed clean hard-exit`，
共用同一 anchor。只有当 **fixed hard-exit 之后仍有清楚可测的缺口**时，
才批准实现 autonomy 机制。届时另行预注册。

已核实的实现约束：`admission_replay_mode` 当前只有 `shared` 与 `student_only`
两个合法值（`train_ptf.py` 显式 `raise`），**没有**"关行为、保留 source replay"的
`B⁻R⁺` 模式。若后续需要 2×2 通道分解，必须先补独立的 behavior/replay authority 接口
并配单元测试与 smoke audit。外部 review 称该臂可由 `fixed_quota` 配置实现，
经核实不成立 —— `fixed_quota` 只是 `analyze_admission_handoff.py` 的分析分组名。
