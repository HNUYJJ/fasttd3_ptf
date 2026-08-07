# 预注册：Evaluator v2.1b（P1.1b 整改）

> 2026-08-07。**必须在任何实现之前提交 git。**
> 提交后只允许改路径参数，不得修改字段定义、校验规则或 smoke 判据。
>
> 本文件**替代** `evaluator_v21_hardening_prereg_20260806.md` 作为 P1.1 的有效判据。
> 旧预注册本身没有错（它写的是对的），错在实现没有逐条落实它，
> 且结果 commit 里改了代码。旧结果已降级为 DIAGNOSTIC。

---

## 0. 为什么要重做

外部 review 指出 6 项问题，我逐条核实**全部成立**。前 5 项是实现缺陷，
第 6 项是流程违规且最严重：

| # | 问题 | 核实出处 |
|---|---|---|
| D1 | `identity_checked` 只要传**任意一个** `--expect-*` 即为 true，不要求正式身份全部声明；无 formal/debug 模式 | `p0_evaluator_v2.py:262` `len(explicit) >= 1` |
| D2 | 输出是裸 `write_text()`，非临时文件 + 原子 rename | `p0_evaluator_v2.py:389` |
| D3 | milestone **只从最后一步 `info` 提取**，尽管 `info_history` 已保存 | `_run_episode_v2` 传 `info=info` |
| D4 | S5 判据未按预注册实现，改用了更弱的代理 | `smoke_evaluator_v21.py:121-126` |
| D5 | `verdict` 取 `ALL_PASS`，与同份输出里 S4 的 `VACUOUS` 自相矛盾 | `smoke.json` |
| **D6** | **结果 commit `19948c4` 同时改了两个 `.py`**，三段式不成立 | `git show --stat 19948c4` |

D6 的处置见 `CLAUDE.md §4.1`（本轮新增）与
`evaluator_v21_hardening_results_20260806.md` 顶部的降级声明。
**不重写 git 历史。**

### 0.1 D3 不是"加固"，是修 bug

HumanoidBench 源码里这些量真的会回落，逐条实测：

```
truck.py:113-115     for package in self.packages_on_table: ... .remove(package)
truck.py:199         reward_dict["success_subtasks"] = len(self.packages_on_table)
basketball.py:139    "success_subtasks": 1 if self.stage == "throw" else 0
basketball.py:140    "success": ball_hoop_distance < 0.05        ← 瞬时判定
cabinet.py:111,116   current_subtask += 1（单调，但 reset 回 1）
```

中途装上车又掉下来的 package、中途穿过篮筐的球，**只读最后一步全部丢失**。
`success` 尤其危险：它是瞬时距离判定，球穿筐后飞走，最后一步就是 `False`——
即"成功了但记成没成功"。

---

## 1. 身份校验：formal / debug 双模式（冻结）

新增 `--identity-mode {formal,debug}`，**默认 `formal`**。

### 1.1 formal（正式评估，唯一可用于科学裁决的模式）

必须**全部**显式声明并匹配，缺任一项即硬失败、非零退出、**不产出任何字节**：

```
--env-name              （已有）且必须等于 checkpoint 内 args["env_name"]
--expect-global-step    必须等于 checkpoint 内 global_step
--expect-seed           必须等于 checkpoint 内 args["seed"]
```

`--expect-admission-mode` **不列入强制项**：scratch checkpoint 没有 `ptf_cfg`，
强制它会把整条 scratch 基线挡在门外。但**若显式传入则必须匹配**。

checkpoint 内缺少 `args["seed"]` 或 `global_step` 字段时，formal 模式**同样硬失败**
（无法核对 ≠ 核对通过）。这类文件由 P2 deep scan 统计，不在此放行。

### 1.2 debug（冒烟 / 探查，产物带毒性标记）

允许不声明 `--expect-*`。强制 env 交叉核对**仍然执行**（不可关闭）。
输出中：

```yaml
identity_mode: "debug"
identity_checked: false
scientific_use_permitted: false     # 硬编码 false，不可通过参数改成 true
```

`scientific_use_permitted` 是给下游读的显式毒性标记：
`false` 的产物**不得**进入任何科学裁决。此前无法从输出中判断一份 JSON
是不是随手跑出来的。

### 1.3 `identity_checked` 的新定义（比旧实现严）

```python
identity_checked = (
    identity_mode == "formal"
    and env_matches_checkpoint          # 强制项
    and expect_global_step is not None and matched
    and expect_seed is not None and matched
    and (expect_admission_mode is None or matched)
)
```

即：**debug 模式下恒为 false**；formal 模式下若能走到写输出这一步则恒为 true
（不匹配已经抛错了）。旧实现的 `len(explicit) >= 1` 移除。

## 2. 原子写（冻结）

```
1. 写入 <out>.tmp.<pid>
2. flush + os.fsync
3. 不允许覆盖时：os.link(tmp, out)   ← 原子，且 out 已存在时抛 FileExistsError
   允许覆盖时  ：os.replace(tmp, out) ← 原子替换
4. finally: 无论成败都删除 tmp
```

用 `os.link` 而非"先 exists() 再 replace"：后者是 TOCTOU，两个进程可同时通过检查。
`os.link` 的 fail-if-exists 由内核保证。同文件系统内成立，跨文件系统会 `EXDEV`——
tmp 与 out 同目录，故不会发生。

启动时的 `out_path.exists()` 预检查**保留**（避免跑几十分钟才发现要覆盖），
但它是便利性检查，正确性由第 3 步保证。

## 3. milestone 沿 trajectory 聚合（冻结）

`resolve_task_outcome` 新增 `info_history: list[dict]` 参数。
milestone 提取改为对**每一步**调用 `milestone_fn`，再聚合。

输出结构（`schema_version` 升至 `2.2`）：

```yaml
milestones:
  <key>:
    final:          # 最后一步的值；该步缺此 key 则 null
    max:            # trajectory 上最大值；非数值类型则 null
    max_step:       # 首次达到 max 的步索引（0-based）；max 为 null 则 null
    first_step:     # 该 key 首次出现的步索引
    n_steps_present: # 该 key 出现过的步数
    ever_true:      # bool(v) 为真是否至少出现一次
```

规则：

1. **`max` 只对 `int` / `float`（含 `bool`，Python 中 `bool` 是 `int`）计算**。
   其余类型 `max = null`、`max_step = null`，但 `ever_true` 仍计算。
   `numpy` 标量按数值处理（`np.integer` / `np.floating`）；`NaN` 不参与 max。
2. `milestone_fn` 在**任何一步**抛异常 → 整个 episode 判 `ADAPTER_ERROR`，
   `milestones` 返回 `{}`。不做部分容错——半截的 milestone 比没有更危险。
3. `info_history` 为空（0 步）→ `milestones = {}`，不报错。
4. **旧的扁平格式 `{key: value}` 移除**，不留兼容路径。理由同
   `require_comparable`：保留弱路径就会有人继续用，而扁平格式恰好是丢数据的那个。

### 3.1 为什么不只存 max

`final` 与 `max` 都要存：`max` 回答"最好到过哪里"，`final` 回答"最后落在哪里"，
两者之差正是"中途达到又回落"的量——它本身就是一个值得看的信号。
只存 max 会把 truck 的"装上车又掉下来"和"稳稳装好"混为一谈。

## 4. `mujoco_state` 进入 episode 记录（冻结）

`_basketball_state` 的提取结果写入 episode 记录：

```yaml
mujoco_state:            # 不需要 state 的任务为 null
  ball_to_hoop_dist:     # float
mujoco_state_error:      # 提取失败时的原因字符串，成功为 null
```

这是 §5 的 S5 判据能够按预注册原文实现的前提——
旧实现之所以退化成弱代理，正因为 `ball_to_hoop_dist` 根本没进输出，
smoke 无从检查"是否提取到有限数值"，只能去看 `metric_status` 这个代理。

## 5. smoke 判据（冻结，逐条对齐、不得用代理替换）

真实 checkpoint + 真实 MuJoCo，每项 8 episodes。**判据必须逐字实现下表，
不得改用"更容易通过的等价物"**（D4 即此错误）。

| # | 任务 | 通过条件（全部为合取） |
|---|---|---|
| S1 | crawl | 8/8 `terminated == false`；`semantics == "neutral"`；`task_success == false`；`metric_status == "OK"` |
| S2 | slide | 存在 `terminated == true` 的 episode；**其全部** `task_success == false` 且 `semantics == "failure"` |
| S3 | truck | `metric_status` 全 `OK`；`milestones` 含 `success_subtasks` 且其 `n_steps_present > 0`；**若**有终止 episode 则其 `task_success == true` |
| S4 | bookshelf_simple | `metric_status` 全 `OK`；**若**有终止 episode，其 `semantics ∈ {success, failure}` |
| S5 | basketball | **至少一个 episode 的 `mujoco_state.ball_to_hoop_dist` 是有限数值**（`math.isfinite`）；且全部终止 episode 的 `semantics ∈ {success, failure}` |
| **S6** | truck（新增） | `milestones.success_subtasks` 的 `max`、`final`、`max_step`、`n_steps_present` 四个字段**全部存在且非 null**——验证 §3 的 trajectory 聚合真的在跑，而不是只把最后一步包了一层 |

**S5 的措辞是硬性的**：判据是 `isfinite(ball_to_hoop_dist)`，
**不是** `metric_status != INSUFFICIENT_STATE`。后者在 0 终止时真空通过，
前者不会——不论是否终止，reset 之后球与筐的距离总是可测的。

**S3 的降级条款保留**：truck 极难，8 episodes 内可能无成功，
故硬性要求是 milestone 通路可用；"若终止则 success" 是条件式。
S6 是无条件的，它验证的是聚合结构而非任务成败。

### 5.1 VACUOUS 与科学用途阻断

任一项的判据在**前提不成立**时真空为真（S3/S4 的条件式子句、S2 若无终止），
该项记 `VACUOUS`，**不计失败**，但：

```
1. 写入 verdict 的 unverified_paths 列表；
2. 该任务写入 scientific_use_blocked 列表。
```

`site_rules` 新增 `require_runtime_verified(env_name, *, purpose)`：
任务不在 `task_metrics.RUNTIME_VERIFIED_TERMINATION` 中即 raise。

**该集合的初值冻结为空 `frozenset()`。** 只能由 P1.1b 结果**之后的独立 commit**
依据实测填入——不得在实现 commit 里凭 diagnostic 轮的印象预填，
那正是"用结果改代码"的另一种形式。当前无科学裁决在跑（P3A 未启动），
故 fail-closed 到空集合不阻塞任何工作。

## 6. verdict 取值（冻结）

```
CORE_PATHS_VERIFIED_WITH_GAPS   无失败项，但存在 VACUOUS
ALL_PATHS_EXERCISED             无失败项且无 VACUOUS（本轮预期达不到，见下）
FAILED_<n>                      有 n 项失败
```

**`ALL_PASS` 及任何全称词禁止使用。** 旧值与 S4 的 VACUOUS 并存自相矛盾。

预期本轮 S4 仍为 VACUOUS：本地 bookshelf 只有 9 个 final checkpoint、
无早期 checkpoint，而 final 策略已学会不摔倒。**这一点写在看到结果之前**——
若结果与预期不符（S4 真的跑出终止），那是好事，按实测记录。

## 7. 停止条件

```
S1–S6 任一 FAILED            → 不得进入 P2 inventory v2
§1 formal 模式未实现          → 不得进入 P2
§2 原子写未实现               → 不得进入 P2
§3 trajectory 聚合未实现      → 不得进入 P2
VACUOUS 不阻塞 P2（P2 只读 checkpoint 元数据，不涉及 milestone 语义）
```

## 8. 设计层自查（CLAUDE.md §8）

**8.1 辨别力**——平凡解释："smoke 通过只是因为环境能跑起来"。
排除：S1/S2 要求**具体语义值**（neutral vs failure）；
S5 要求**具体数值有限**；S6 要求**四个聚合字段同时非 null**，
只把最后一步包一层壳会让 `max_step`/`n_steps_present` 露馅。
第二个平凡解释："改了判据让它更容易过"。排除：S5 由弱代理**改回**预注册原文，
S6 是净新增，无一项被放宽。

**8.2 混淆变量**——S2 换用 20k checkpoint 与"判据是否被满足"共变吗？
共变，且方向对我有利，故必须声明：**S2 的 checkpoint 选择在本预注册中冻结**为
`slide_bac_walk_s1__1_20000.pt`，理由是验证 failure 语义**必须有终止 episode**，
而 final scratch 已学会不摔倒（diagnostic 轮实测 0/8）。
本轮不得再更换；若它这次跑出 0/8 终止，S2 记 `VACUOUS`，**不许再换第三个**。

**8.4 前提是否蕴含结论**——验算 S1–S6 能否全部真空通过。
不能：S1 要求 8/8 具体值；S5 要求至少一个有限数值（无条件）；
S6 要求四字段非 null（无条件）。否定分支可达。
反向验算：`RUNTIME_VERIFIED_TERMINATION` 初值为空是否使 §5.1 的门恒失败？
是——这是有意的 fail-closed，且不构成"判据通过"，因为它不是 smoke 判据。

**8.6 是否重演本轮教训**——
D6（结果 commit 改代码）→ §9 的提交结构写死"结果 commit 内出现任何 `.py`/`.sh` 即违规"；
D4（用更弱代理替换判据）→ §5 明令"逐字实现、不得等价替换"，并靠 §4 让原判据可实现；
E-1（凭推理不实测）→ §0.1 的源码行号全部实际 grep 过，写在正文里可复核。

## 9. 提交结构（三段式，本轮必须真正做到）

```
预注册   本文件                                   ← 先于实现
实现     p0_evaluator_v2.py / task_metrics.py /
         schema_v2.py / site_rules.py /
         smoke_evaluator_v21b.py / 单元测试        ← 无任何运行产物
结果     smoke JSON + 结论文档                     ← git show --stat 内不得有 .py/.sh
后续     RUNTIME_VERIFIED_TERMINATION 填值         ← 依据结果，独立 commit
```

**若结果暴露 bug**：按 `CLAUDE.md §4.1` 走——标 DIAGNOSTIC → hotfix commit →
重新冻结 → 独立重跑。**不得**在结果 commit 里修。
