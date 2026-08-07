# Post-Diagnostic Reverification Protocol v2：Evaluator v2.1c（A3–A6 / A9）

> 2026-08-07。**必须在任何实现之前提交 git。** 冻结后只允许改路径参数，
> 不得修改字段定义、校验规则或判据。
>
> **本文件同样不是 blind pre-registration。** 与 v21b 一样，它写于看过
> P1.1/P1.1b 结果之后，是 post-diagnostic protocol：能保证"实现不回头迁就结果"，
> 不能保证"判据独立于已见结果"（`CLAUDE.md §8.7`）。
> 它**扩展**而不推翻 `evaluator_v21b_prereg_20260807.md` 的任何判据。

---

## 0. 要补的是什么

外部 review 的第二轮核查指出 6 项仍未落实，我逐条 `grep` 核实**全部成立**：

| 条 | 缺口 | 核实 |
|---|---|---|
| A3 | 无 identity manifest，不核对 checkpoint SHA256，不核对 protocol digest | `grep manifest\|expect_sha scripts/p0_evaluator_v2.py` → 空 |
| A4 | formal 模式仍可被 `--allow-overwrite` 放行；原子提交用 `os.link` 而非 `os.replace` | `p0_evaluator_v2.py:232,415` |
| A5 | 无 `evaluation_semantics_digest` | 全仓库 `grep` → 空 |
| A6 | 统一聚合而非 registry 冻结 reducer；无 `first_hit_step` | `task_metrics.aggregate_milestones` 对所有 key 一视同仁 |
| A9 | `paired_by_seed` 仅凭 `env+seed+step` 推断匹配；无 `training_protocol_digest`、无 `match_group` | `site_rules.py:33-34` |

A5 与 A9 是同一个问题的两面：**当前的 `panel_digest` 只覆盖 seeds/ranks/steps，
两份用不同版本 `task_metrics.py` 产出的结果会被判为可比**——
而任务语义映射一变，`task_success` 的含义就变了。这比跨预算比较更隐蔽。

---

## 1. Identity manifest（A3，冻结）

formal 模式**不再**从散落的 `--expect-*` 拼身份，改为读一份显式 manifest：

```yaml
# identity manifest（JSON）
checkpoint_sha256:        # 必需，evaluator 自行计算后比对
env_name:                 # 必需，与 checkpoint 内 args["env_name"] 比对
learner_seed:             # 必需，与 args["seed"] 比对
global_step:              # 必需，与 checkpoint 内 global_step 比对
training_protocol_digest: # 可选；checkpoint 内有 ptf_cfg / protocol 声明时**必需**
```

规则：

1. **formal 模式必须传 `--identity-manifest <path>`**；缺该参数即非零退出；
2. manifest 中缺 `checkpoint_sha256` / `env_name` / `learner_seed` / `global_step`
   任一项 → 非零退出，**不产出任何字节**；
3. checkpoint 内**存在** `ptf_cfg` 或 protocol 声明，而 manifest 未给
   `training_protocol_digest` → 非零退出。反之 checkpoint 无此类声明时该项可省
   （scratch 基线不应被挡在门外）；
4. 任一项与 checkpoint 实际值不符 → 非零退出；
5. checkpoint 内缺 `args["seed"]` 或 `global_step` → 非零退出
   （**无法核对 ≠ 核对通过**）；
6. `--expect-*` 系列参数**保留但仅 debug 模式可用**，formal 模式下传入即报错——
   避免两套身份来源并存导致"看起来声明了但走的是弱路径"。

### 1.1 debug 模式的产物不得进入正式分析

```yaml
identity_mode: "debug"
identity_checked: false
scientific_use_permitted: false
```

新增硬性要求：**debug 产物的文件名必须带 `.debug.` 段**
（由 evaluator 强制，不符则拒绝写出）。理由：`scientific_use_permitted` 字段
要打开文件才看得到，而批量分析脚本常常按 glob 收文件；文件名级别的标记
是唯一在"收集阶段"就能生效的护栏。

## 2. formal 模式禁止 overwrite（A4，冻结）

```
formal 模式：--allow-overwrite **无效**，目标存在即非零退出。
             不是"默认拒绝可放行"，是"无法放行"。
debug 模式：允许 --allow-overwrite。
```

原子提交统一改为 **tempfile → flush → `os.fsync` → `os.replace`**：

```
1. 同目录写 <out>.tmp.<pid>
2. fh.flush(); os.fsync(fh.fileno())
3. 若不允许覆盖：先 os.link 抢占目标（内核保证 fail-if-exists），成功后
   os.replace(tmp, out) 完成提交；link 失败即 FileExistsError
4. 若允许覆盖：os.replace(tmp, out)
5. finally 清理 tmp —— 任何路径都不得留下半成品
```

> 说明：review 要求"tempfile→fsync→os.replace"。单用 `os.replace` 无法表达
> fail-if-exists（它会静默覆盖），故不允许覆盖时先用 `os.link` 做原子抢占、
> 再 `os.replace` 完成提交。最终提交动作仍是 `os.replace`，且无 TOCTOU 窗口。

## 3. `evaluation_semantics_digest`（A5，冻结）

`panel_digest` 的语义**收窄并固定**为：仅 `eval_seeds` + `ranks` + `episode_steps`
+ `deterministic`。不再承担任何其他含义。

新增：

```
evaluation_semantics_digest = sha256(
    schema_version                                  # 输出结构
    + source_free_mode                              # 结构性 source-free 声明
    + sha256(scripts/p0_evaluator_v2.py)            # 评估主程序
    + sha256(fasttd3_ptf/evaluation/task_metrics.py)# 任务语义映射
    + sha256(fasttd3_ptf/evaluation/schema_v2.py)   # 记录构造
)
```

**正式可比性必须要求该 digest 相同**（见 §5）。三个文件的内容摘要都进去，
因为它们任何一个改变都会改变 `task_success` / `milestones` 的含义——
而这类改变恰恰不会体现在 `panel_digest` 上。

digest 计算读取的是**磁盘上的文件内容**，不是 import 后的模块对象，
以便同一进程内也能检测到文件被替换。

## 4. Registry 冻结 reducer（A6，冻结）

`TaskMetrics` 新增字段 `milestone_reducers: dict[str, tuple[str, ...]]`，
**逐 milestone 声明**要输出哪些 reducer。支持的 reducer 冻结为四个：

| reducer | 定义 |
|---|---|
| `final` | 最后一步的值；该步缺此 key 则 `null` |
| `max` | trajectory 上最大值；非数值类型 `null` |
| `ever_true` | `bool(v)` 为真是否至少出现过一次 |
| `first_hit_step` | **首次 `bool(v)` 为真的步索引**；从未为真则 `null` |

`first_hit_step` 与既有的 `first_step`（该 key 首次出现的步）**是两回事**，
两者都保留：前者是"第一次达成"，后者是"第一次有这个字段"。
既有的 `max_step` / `n_steps_present` / `first_step` 作为诊断量继续无条件输出。

强制映射（冻结）：

```
truck.success_subtasks      max, final      truck.success      ever_true, first_hit_step
cabinet.success_subtasks    max, final      cabinet.success    ever_true, first_hit_step
bookshelf.success_subtasks  max, final      bookshelf.success  ever_true, first_hit_step
package.success                             ever_true, first_hit_step
basketball.success_subtasks max, final
```

**fail closed**：某 milestone 声明了 reducer，但整条 trajectory 里该 key
**一次都没出现** → 该 milestone 记
`{"status": "MISSING_TRAJECTORY_FIELD", ...}` 且 episode 的
`metric_status` 置为 `MISSING_MILESTONE_FIELD`。
不得静默给出全 `null` 的聚合结构——那看上去像"测了但都是空"。

## 5. Comparability（A9，冻结）

`COMPARISON_REQUIREMENTS` 更新：

```python
_BASE = {"env_name", "global_step", "panel_digest", "schema_version",
         "evaluation_semantics_digest"}          # ← 新增第 5 项，所有 purpose 必需

"across_methods":  _BASE
"across_seeds":    _BASE | {"method_family"}
"paired_by_seed":  _BASE | {"learner_seed", "match_group", "training_protocol_digest"}
"same_checkpoint": _BASE | {"learner_seed", "checkpoint_sha256"}
```

三条新规则：

1. **`evaluation_semantics_digest` 对所有 purpose 必需**——语义映射不同的两份
   结果，数字长得一样也不可比；
2. **`paired_by_seed` 必须有 `match_group`**，且它只能来自预注册的
   experiment manifest，**不得由 `env+seed+step` 现场推断**。
   理由：同一 `(env, seed, step)` 可能来自完全不同的实验臂（不同 source、
   不同剂量、不同退出策略），推断出来的"配对"是假配对，而配对差值统计
   恰恰全靠配对正确；
3. **`checkpoint_sha256` 只在 `same_checkpoint` 下要求相等**。其余 purpose
   下不同实验臂的 SHA 本来就不同，要求相等会把所有真实比较都挡掉。
   SHA 的作用是**身份**（这个文件是不是我以为的那个），不是可比性。

## 6. smoke 判据增量（在 v21b 的 S1–S6 之上）

| # | 验证什么 | 通过条件 |
|---|---|---|
| S7 | identity manifest | formal + 完整 manifest 通过；manifest 缺任一必需字段 / SHA 不符 / 有 ptf_cfg 却无 protocol digest → 全部非零退出 |
| S8 | formal 禁 overwrite | formal + 目标已存在 + `--allow-overwrite` → 仍非零退出，且原文件字节不变 |
| S9 | semantics digest | 篡改 `task_metrics.py` 一个字节后 digest 改变；`require_comparable` 在 digest 不同时拒绝 |
| S10 | reducer + fail closed | truck 的 `success_subtasks` 出 `max`+`final`、`success` 出 `ever_true`+`first_hit_step`；声明了 reducer 但字段全程缺失 → `MISSING_MILESTONE_FIELD` |
| S11 | debug 文件名护栏 | debug 模式写非 `.debug.` 路径 → 拒绝 |

S7–S9、S11 用单元测试实现（需要构造损坏的 manifest 与被篡改的文件，
在真实 MuJoCo 里做既慢又不可控）；S10 的 fail-closed 分支用单元测试，
正常分支由真实 smoke 覆盖。

## 7. 停止条件

```
S1–S11 任一 FAILED             → 不得进入 P2
formal 模式仍能不带 manifest 运行 → 不得进入 P2
evaluation_semantics_digest 未进 require_comparable → 不得进入 P2
VACUOUS 不阻塞 P2（P2 只读 checkpoint 元数据）
```

## 8. 设计层自查（CLAUDE.md §8）

**8.1 辨别力**——平凡解释："digest 加了但没人校验"。排除：S9 要求
篡改文件后 `require_comparable` **实际拒绝**，不是只看 digest 值变了。
第二个平凡解释："manifest 只是把 `--expect-*` 换个写法"。排除：manifest
强制包含 `checkpoint_sha256`（旧路径完全没有），且 formal 下禁用 `--expect-*`，
不存在"看起来声明了但走弱路径"的可能。

**8.2 混淆变量**——`evaluation_semantics_digest` 把三个文件的 SHA 混在一起，
无法区分"哪个文件变了"。这是有意的：可比性是全有全无的判断，
细分来源会诱导"只是 schema 变了应该还能比"这类放宽。诊断需要时，
三个文件的独立 SHA 已单独记录在输出中。

**8.4 前提是否蕴含结论**——验算 S7–S11 能否全部真空通过。不能：
S7/S8/S11 全是**否定式**判据（要求某操作失败），前提恒成立；
S9 要求 digest 实际改变且比较实际被拒；S10 的正常分支无条件。

**8.6 是否重演本轮教训**——
D6（结果 commit 改代码）→ §9 写死自查命令；
D4（用更弱代理替换判据）→ §6 每条都指明**具体断言对象**，不留"等价物"空间；
A2（把 post-hoc 文档叫 prereg）→ 本文件头部已声明其定性。

## 9. 提交结构

```
protocol   本文件                                    ← 先于实现
实现       identity manifest / 原子提交 / semantics digest /
           reducer / comparability + 单元测试         ← 无任何运行产物
结果       smoke + 单元测试完整输出 + 结论文档          ← git show --stat 不得有 .py/.sh
```

**若结果暴露 bug**：按 `CLAUDE.md §4.1` 走——标 DIAGNOSTIC → hotfix commit →
重新冻结 → 独立重跑。**不得**在结果 commit 里修。
