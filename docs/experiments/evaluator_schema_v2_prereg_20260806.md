# 预注册：evaluator schema v2 + checkpoint inventory + 场地普查 v2 口径

> 2026-08-06。**本文件必须在任何代码实现之前提交 git**，提交后停止等待 review。
> 本轮**不实现代码、不批量重评、不启动训练**。
>
> 触发：site screen v1 暴露三个缺陷 —— evaluator 把摔倒记为 success、
> milestone 从未采集、`J_best_known` 跨 method/seed/step 取 max（winner's curse）。
> 前者见 `post_transfer_autonomy_site_screen_results_20260806.md` §4.1 与 §9。

---

## 0. 本次要解决的三个问题

| # | 问题 | 现状证据 |
|---|---|---|
| P-1 | evaluator 把环境终止等同于任务成功 | `p0_evaluator.py:93` `if terminated: success = True`；而 `Walk` 系 `get_terminated` 是**摔倒** |
| P-2 | 任务自定义 milestone 从未采集 | 全部 25 个 target 的 G4 均为 `UNKNOWN` |
| P-3 | 场地比较用了不可比的数值 | v1 的 `J_best` 横跨 20k/75k/100k 三种预算、四种 method |

**这三个是测量系统缺陷，不是科学结论问题。** 修好之前，任何场地结论都不可信。

---

## 1. Episode schema v2（冻结）

```yaml
schema_version: 2

episode:
  # ---- v1 已有，必须逐位兼容 ----
  seed: int
  return: float
  progress_max_dx: float

  # ---- v2 新增：环境事实与任务语义分离 ----
  episode_length: int
  terminated: bool                 # 环境是否发出 terminated
  truncated: bool                  # 是否因 max_episode_steps 截断
  termination_semantics: enum      # failure | success | neutral | unknown
  task_success: bool | null        # null ≠ false，见 §1.1
  milestones: dict                 # 由 registry 的 adapter 产出，缺则 {}
  info_diagnostics: dict           # 通用标量汇总，仅作诊断，见 §1.3
  metric_status: enum              # OK | UNREGISTERED | ADAPTER_ERROR
```

> **2026-08-06 修订说明**：本文件在 `c0caf98` 首次冻结后，按 PI 下发的目标做了三处修订
> （字段更名 `info_summary`→`info_diagnostics`、§1.4 非标量分级处理、§2 增加两级候选口径）。
> **修订发生在任何 evaluator v2 数据产生之前**，属 review 反馈驱动，
> 不是 outcome-contingent gate switching（M30）——彼时无任何结果可供观察。

### 1.1 `task_success = null` 与 `false` 必须区分

```
false   该任务有经过审计的成功定义，且本 episode 确定未达成
null    该任务尚无经审计的成功定义 → 不可判定
```

**禁止**在缺 adapter 时默认 `terminated == success`，也**禁止**默认 `false`。
缺 adapter 一律 `null` + `metric_status = UNREGISTERED`（fail closed）。

`terminated_success` 字段**在 v2 中删除**，不保留别名。任何读取它的下游脚本
必须显式迁移，不得静默继承旧语义（CLAUDE.md §6 的陷阱正是由此而来）。

### 1.2 task metric registry（不得由 evaluator 猜语义）

```python
TASK_METRIC_REGISTRY: dict[str, TaskMetrics]
```

每个 adapter 必须显式声明：

```yaml
termination_is: failure | success | neutral   # get_terminated 的语义
milestone_names: [...]                        # 人工定义并审计
milestone_fn: 如何从 info / MuJoCo state 计算
needs_mujoco_state: bool
on_missing_field: raise | null                # 默认 raise（fail closed）
```

**首批只注册已逐条读过 `get_terminated` 与 `get_reward` 源码的任务。**
未注册任务照常评估 return，但 `task_success = null`、`metric_status = UNREGISTERED`。

首批注册目标（按现有 checkpoint 数量与场地价值排序，**此排序不是场地选择**）：

```
必须首批   stair  truck  cabinet
其次       bookshelf_simple  package  basketball
locomotion 对照  slide  hurdle  crawl        （用于验证"摔倒 ≠ success"确实被修正）
```

### 1.3 通用 `info_diagnostics` 只是诊断量，不是 milestone

对每个**有限标量** info 字段记录：

```
mean / max / final / nonzero_fraction / first_positive_step
```

**但 `info["reward_xxx"]` 通常只是 reward 分量，不代表任务阶段。**
`info_diagnostics` 的任何字段都**不得**自动升级为 milestone；
milestone 只能来自 §1.2 的人工 adapter。

### 1.4 非标量 info 按**是否为注册必需字段**分级处理（2026-08-06 修订）

首版规定"任何非标量无法序列化时一律报错"。该规则过严：HumanoidBench 会返回
数组、调试量以及与当前任务无关的结构，一个未知数组就终止整场评估会使 evaluator
极其脆弱。改为三级，**级别由字段是否被 registry 声明为必需决定**：

| 字段类别 | 处理 | 理由 |
|---|---|---|
| registry 声明的**必需** milestone 字段 | **硬报错**（fail closed），给出 key + type + shape | 它是判据输入，错了会污染结论 |
| 未注册的诊断字段，可转标量 | 进 `info_diagnostics` | 有用且无风险 |
| 未注册的诊断字段，**不可**转标量 | 记入 `info_diagnostics_unsupported`：`{key: {type, shape}}` | 显式留痕，便于日后升级为 milestone |

**绝对禁止**：静默跳过必需字段，或让缺失字段以默认值进入判据。
需覆盖的类型：Python 标量 / NumPy 标量 / ndarray / list / tuple / dict /
tensor / NaN / Inf / 时变 key。

**无法序列化时必须报错并给出 key 与 type，禁止静默跳过**——
v1 的 milestone 缺失正是"静默跳过"造成的。

---

## 2. 三个分离的 return 量（替代 v1 的单一 `J_best`）

```
best_observed_return
    跨 method/seed/step 的最大值。
    **仅用于排除"从未有任何策略达到过某水平"**。
    禁止跨 target 比较，禁止用于 headroom 计算。

fixed_budget_mean_return(method, step, n_seeds)
    同 method、同 global_step、多 learner seed 的均值 + learner 间 sd。
    这是**场地比较的唯一合法量**。n_seeds < 3 时标 INSUFFICIENT_SEEDS。

hard_exit_mean_return(step, n_seeds)
    hard-exit 臂在固定 step 的多 seed 均值。
    这是判断 post-exit residual deficit 的**唯一**合法指标。
```

每个数值必须携带完整身份，缺任一项即 `UNKNOWN`：

```
task / method / learner_seed / global_step / checkpoint_sha256 /
code_commit / source_bank_digest / bootstrap_budget / exit_policy / evaluation_panel
```

**禁止跨不同 `global_step` 比较。** v1 拿 `stair@20k` 对 `slide@75k` 是错误的，
已在结果文档 §9 撤回。

### 2.1 两级候选：消除 v1 设计中的循环依赖（2026-08-06 修订）

v1 的普查把"是否存在 hard-exit residual deficit"直接写进候选条件，构成死循环：

```
没有 hard-exit 数据  →  不能成为 CANDIDATE  →  不批准跑 hard-exit 实验  →  永远 UNKNOWN
```

所有从未做过 hard-exit 实验的新任务（truck / bookshelf / package …）都会被永久卡住。
按目标拆成两级，**各自使用不同的 return 量**：

| 级别 | 判据 | 允许使用的量 | 批准什么 |
|---|---|---|---|
| `SITE_CANDIDATE`（P3A） | milestone 可测 + 有 headroom + 有 early-scaffold 迹象 + source 解不了终点瓶颈 + 数据身份完整 | `best_observed_return`（仅排除用）、`fixed_budget_mean_return` | 批准运行三臂 gate |
| `AUTONOMY_CANDIDATE`（P3B） | early scaffold gain + continuous-source deficit + **hard-exit residual deficit** + measurable headroom | 追加 `hard_exit_mean_return` | 批准设计新机制 |

**关键**：`hard_exit_mean_return` 只在 P3B 使用。P3A **不得**要求它存在，
否则循环依赖复现。三臂 gate（P4）正是产生该量的地方。

---

## 3. 理论上限审计表（显式 registry，禁止 AST 自动推断）

```yaml
h1hand-slide-v0:
  theoretical_return_upper_bound: 1000
  proof_status: audited
  proof_source: "ClimbingUpwards.get_reward — [0,1] 项相乘，无稀疏加项 × 1000 步"

h1hand-cabinet-v0:
  theoretical_return_upper_bound: null
  proof_status: not_audited
```

`proof_status != audited` 时 `H_raw = UNKNOWN`，**且不得计算任何百分比**。

已审计（v1 已逐条读过 `get_reward` 全文）：`Walk` / `ClimbingUpwards` / `Hurdle` 系。
`success_bar = 2500` 的 cabinet 等**明确未审计**——其 `success_bar` 远超 1000，
说明 reward 未必单步 ≤ 1，可能含加性分量。

---

## 4. Checkpoint inventory manifest（v2 的前置）

**checkpoint 多 ≠ 证据完整。** 批量重评前必须先建 manifest：

```yaml
checkpoint_sha256:
env_name:
learner_seed:
global_step:
training_commit:
method_family:            # scratch | continuous_source | hard_exit | racing_arm | smoke | unknown
source_bank_digest:
bootstrap_steps:
admission_mode:
hard_exit_step:
is_complete:              # 训练是否正常结束
eligible_for_site_screen: bool
exclusion_reason:         # debug_smoke | incomplete_run | unknown_code_version | non_source_free | ...
run_group_id:             # 同一 run 的多个 step 共享此 id
```

**同一训练 run 的 10k/20k/30k/50k/100k 不是五个独立样本**——
统计时按 `run_group_id` 去重，`n_seeds` 只数不同 learner seed。

本地已探明的 `*.pt` 计数（仅供估算工作量，不代表可用样本数）：

```
truck 132   basketball 105   cabinet 95   stair 69   powerlift 58   package 15   bookshelf 9
```

---

## 4.5 模块契约（实现前冻结；测试按此编写）

判据逻辑必须与环境交互解耦，否则 T5–T10 无法在不启动 MuJoCo 的情况下测试。
三个新模块，全部为**纯函数 / 纯数据**：

```python
# fasttd3_ptf/evaluation/task_metrics.py ── 任务语义 registry
@dataclass(frozen=True)
class TaskMetrics:
    termination_is: str | Callable       # "failure"|"success"|"neutral"，或条件判定函数
                                         # （2026-08-06 实现前扩展，理由见 §4.6）
    milestone_names: tuple[str, ...]     # 人工审计过的 milestone
    required_info_keys: tuple[str, ...]  # 必需字段；解析失败 → 硬报错
    milestone_fn: Callable[[dict, dict | None], dict]   # (info, mj_state) -> milestones
    needs_mujoco_state: bool = False
    source: str = ""                     # get_terminated/get_reward 的核实出处

TASK_METRIC_REGISTRY: dict[str, TaskMetrics]

def resolve_task_outcome(env_name, terminated, truncated, info, mj_state=None
                         ) -> tuple[bool | None, str, str, dict]:
    """→ (task_success, termination_semantics, metric_status, milestones)

    未注册任务必须返回 (None, "unknown", "UNREGISTERED", {})。
    禁止任何形式的 `terminated → success` 默认推断。
    """

# fasttd3_ptf/evaluation/schema_v2.py ── episode 记录构造与 info 分级
SCHEMA_VERSION = 2

def summarize_info(info_history, required_keys) -> tuple[dict, dict]:
    """→ (info_diagnostics, info_diagnostics_unsupported)

    必需字段不可解析 → raise RequiredFieldError（fail closed，§1.4）。
    未注册非标量 → 进 unsupported，记 {key: {type, shape}}，不静默丢弃。
    """

# fasttd3_ptf/evaluation/site_rules.py ── 场地判定的 fail-closed 规则
class IncomparableError(Exception): ...

def classify_headroom(h_op, h_ms) -> str
    """H_ms 为 None 时**绝不**返回 SATURATED（T5）。"""

def pct_of_ceiling(value, ceiling) -> float | None
    """ceiling 为 None/未审计 → 返回 None，不计算百分比（T6）。"""

def require_comparable(a: dict, b: dict) -> None
    """global_step 不同 → raise IncomparableError（T7）。"""

def is_robustly_solved(returns_by_seed: dict, bar) -> bool | None
    """n_seeds < 3 → 返回 None（INSUFFICIENT_SEEDS），不得为 True（T8）。"""

def has_post_exit_deficit(hard_exit_stats, ceiling_stats) -> bool | None
    """hard_exit_stats 为空 → 返回 None，不得判定 deficit（T9）。"""
```

`site_rules` 与 `task_metrics` **不 import gymnasium / mujoco / torch**，
使 T1–T3、T5–T10 可在无 GPU、无 MuJoCo 的环境下秒级运行。
只有 T4（v1/v2 逐位兼容）需要真实 checkpoint 与环境，标记为集成测试。

## 4.6 逐任务核实的终止语义（实现前读源码得出）

`termination_is` 从 `str` 放宽为 `str | Callable` 的理由——读完源码发现
**终止语义不都是静态的**，静态枚举无法表达 bookshelf 与 basketball：

| 任务 | `get_terminated` 实际条件 | 出处 | 语义 |
|---|---|---|---|
| Walk / Run / Stand / Hurdle | `qpos[2] < 0.2` | `basic_locomotion_envs.py:96` | `failure`（摔倒） |
| **Crawl** | **恒 `return False`** | `basic_locomotion_envs.py:168` | `neutral`（**永不终止**） |
| Slide / Stair | `torso_upright < 0.1` | `basic_locomotion_envs.py:216` | `failure` |
| Sit / SitHard | `qpos[2] < 0.5` | `basic_locomotion_envs.py:356` | `failure` |
| Powerlift | `qpos[2] < 0.2` | `powerlift.py:99` | `failure` |
| Truck | 全部 package 上桌 | `truck.py:207` | `success` |
| Cabinet | `current_subtask == 5` | `cabinet.py:244` | `success` |
| Package | `dist_package_destination < 0.1` | `package.py:147` | `success` |
| **Bookshelf** | reason 0 摔倒 / reason 1 完成 / reason 2 物体掉落 | `bookshelf.py:190` | **条件**：读 `info["terminated_reason"]` |
| **Basketball** | 球掉 / 人摔 / 进筐，**三者都 `return True, {}`** | `basketball.py:143` | **条件**：info 无区分字段，须读 MuJoCo state |

两条由此确定的事实：

1. **v1 的 `if terminated: success = True` 在 crawl 上不触发**——crawl 恒不终止。
   此前笼统说"全部 locomotion 把摔倒记为 success"不精确，
   受影响的是 walk/run/stand/hurdle/slide/stair/sit/powerlift，**不含 crawl**。
2. **basketball 无法仅由 `(terminated, info)` 判定成败**。缺 MuJoCo state 时
   必须返回 `task_success=None` + `metric_status="INSUFFICIENT_STATE"`，
   不得猜测——这正是 `needs_mujoco_state` 字段的用途。

## 5. Fail-closed golden tests（实现前冻结，必须全部通过）

v1 恰好在这些点上出错，故全部转为自动测试：

| # | 测试 | 期望 |
|---|---|---|
| T1 | locomotion 摔倒 episode | `terminated=true`, `termination_semantics=failure`, `task_success=false` |
| T2 | manipulation 真成功（按该任务实际成功条件构造或用已知成功轨迹） | `task_success=true` |
| T3 | 未注册任务 | `task_success=null`, `metric_status=UNREGISTERED`，**不得猜测** |
| T4 | v1/v2 同 checkpoint 同 32-episode 面板 | `return` / `progress_max_dx` / `episode_length` / reset seed 顺序**逐位一致**；只允许 success 字段变化 |
| T5 | 非标量 info（list / ndarray / nested dict / NaN / Inf / 时变 key） | 行为由 schema 冻结，无法序列化时**报错并给出 key+type** |
| T6 | `H_ms = None` | **绝不得**输出 `SATURATED` |
| T7 | `H_raw = None` | **绝不得**计算百分比 |
| T8 | 两个数值 `global_step` 不同 | 比较函数**必须拒绝**并返回 `INCOMPARABLE` |
| T9 | `n_seeds < 3` | **不得**标记 `ROBUSTLY_SOLVED` |
| T10 | 缺 hard-exit 臂 | **不得**判定 `post_exit_deficit` |

---

## 6. 设计层自查（CLAUDE.md §8）

**8.1 辨别力**：平凡解释——"evaluator 改完后数字会变，是因为改了口径而非修了错误"。
排除方式：T4 要求 return/progress/length 逐位一致，**只有 success 字段允许变化**。
若 return 也变了，说明引入了新 bug，测试直接失败。

**8.2 混淆变量**：本轮不产生科学结论，无处理变量。但 §2 的三个量正是为了
消除 method/step/seed 与 target 的共变——v1 的失败就是没做这个分离。

**8.3 独立重复**：`fixed_budget_mean_return` 强制 `n_seeds ≥ 3`（M24），
低于则标 `INSUFFICIENT_SEEDS` 而非给点估计。

**8.4 前提是否蕴含结论**：验算 T6–T10 是否会使某个分支逻辑上不可达。
不会——它们只禁止在**数据缺失**时做实质裁决，数据齐备时全部分支仍可达。

**8.5 site selection**：本轮**不选场地**。首批 adapter 注册顺序（§1.2）
按现有 checkpoint 数量排序，是工作量考虑，**不构成场地选择**；
场地选择由 site screen v2 的确定性规则决定，届时另行预注册。

**8.6 是否重演本轮教训**：
- v1 的 `SATURATED` 误判 → T6；
- v1 的 `H_raw` 未实现 → §3 显式 registry + T7；
- v1 的跨预算比较 → §2 完整身份 + T8；
- v1 的 milestone 静默丢失 → §1.4 fail closed + T5；
- M33（用推理代替查询）→ §3 `proof_status` 必须逐条读源码，禁止 AST 推断。

**8.7 判据切换红线**：本文件冻结后，schema、registry 契约与 T1–T10 不得修改。
若实现中发现某测试无法通过，**须记录并 review，不得删除或放宽该测试**。

---

## 7. 本轮交付物与停止点

```
交付   本预注册（schema / adapter 契约 / 理论上限表 / inventory schema / T1-T10）
停止   提交后立即停止，等待 review
```

**本轮明确不做**：evaluator 代码实现、批量重评、checkpoint inventory 生成、
site screen v2、新训练、`B⁻R⁺` 接口、autonomy 机制、新 selector。

## 8. review 通过后的顺序（此处只登记）

```
1. 实现 evaluator v2（单独 commit，不含任何结果文件）
2. 运行 T1–T10（完整输出入 review packet）
3. 每个 task family 各跑一次 smoke
4. 生成 checkpoint inventory manifest（单独 commit）
5. 只重评 eligible_for_site_screen = true 的 checkpoint
6. site screen v2 预注册（新状态分类在此引入，不在 v1 内改）
7. 才决定 stair / truck / 其他任务的角色
```

## 9. 流程改进（本轮 review packet 自查发现）

commit `3235581` 把脚本修改与新生成的 `screen.json` 放在同一 commit，
违反"实现与结果必须分离"。**今后强制三段式**：

```
预注册 commit  →  实现 commit（不含结果文件）  →  结果 commit（不含代码改动）
```
