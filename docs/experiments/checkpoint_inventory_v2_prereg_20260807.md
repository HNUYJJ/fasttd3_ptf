# 预注册：Checkpoint Inventory v2（P2）

> 2026-08-07。**必须在任何实现之前提交 git。**
> 冻结后只允许改路径参数，不得修改身份定义、判据或退出条件。

## 0. 盲性声明（先说清楚，免得又被当成 blind prereg）

与 `evaluator_v21b/c` 不同，**本文件对 v2 的主终点是盲的**：
`run_instance` 唯一性冲突数、身份冲突数、`experiment_role` 的 UNKNOWN 比例、
协议感知 canonical 的实际命中——这些我都还没看过。

**不盲的部分**（如实列出）：v1 inventory 已披露的聚合量我已知晓——
`1661` 个 `.pt`、`135` 排除（`ANCHOR_BUNDLE` 99 / `SMOKE_OR_DEBUG` 24 /
`UNPARSEABLE_NAME` 12）、`964` canonical、canonical 步集为
`[20000, 50000, 100000, FINAL]`。v2 恰恰要改这个步集，故这一项的先验值
**不构成对 v2 判据的污染**——它是被修的对象，不是被验的结论。

---

## 1. 身份六元组（冻结）

v1 的 `run_group_id` 把多个概念混在一起。v2 拆开：

| 字段 | 唯一来源 | 无法确定时 |
|---|---|---|
| `checkpoint_id` | 文件 `sha256`（deep scan 阶段计算） | 不可能未知；读不出即 `EXCLUDED_UNREADABLE` |
| `run_instance_id` | checkpoint 内 `args["exp_name"]` + `args["seed"]` | `UNKNOWN_RUN_INSTANCE` |
| `learner_replication_id` | checkpoint 内 `args["seed"]` | `UNKNOWN_SEED` |
| `mechanism_family` | checkpoint 内 `ptf_cfg` / `source_names` / `agent_type` **实际保存的配置** | `UNKNOWN_MECHANISM` |
| `experiment_role` | **仅**冻结 run card / experiment manifest | `UNKNOWN_ROLE` |
| `training_protocol_digest` | `sha256(ptf_cfg)`；无 `ptf_cfg` 则 `NO_PROTOCOL` | — |

### 1.1 `run_instance_id`：本仓库没有 run UUID，且不得靠 mtime 猜

已核实：checkpoint 顶层键为
`[actor_state_dict, admission_audit, agent_type, args, …, global_step, ptf_cfg, source_names]`，
`args` 的 57 个键中与身份相关的只有 `exp_name` 与 `use_wandb`——
**没有 run UUID，没有 W&B run id**。

故 `run_instance_id = f"{exp_name}#{seed}"`。它读自 **checkpoint 内部**
（不是文件名解析），这一点与 v1 不同。但它**不保证唯一**：
同一 `exp_name` 被重跑时，两次 run 会得到相同的 id。

**唯一性冲突判据（冻结）**：同一 `run_instance_id` 下，若出现
**同一 `global_step` 的两个不同 `checkpoint_sha256`**，
则该 `run_instance_id` 整组标记 `AMBIGUOUS_RUN_INSTANCE` 并**非零退出**。

**明令禁止**：用 `file_mtime`、文件名中的时间戳、或路径顺序去区分 attempt。
时间戳能排序但不能证明归属——两次 run 的产物可以交错落盘。
宁可 `AMBIGUOUS_RUN_INSTANCE` 停下来，也不要猜一个看起来合理的分组。

### 1.2 `experiment_role`：当前**没有** run card，故一律 UNKNOWN

已核实：仓库内的 `runtime_manifest*.json`（`slide_hard_exit_v1` /
`slide_speedup_v1` / `critic_first_bridge_v1`）记录的是 **实验级**信息
（git head、源码快照 sha、文件哈希、环境），
**不含 checkpoint → 实验臂的映射**。

因此 v2 的 `experiment_role` 预期**大面积 `UNKNOWN_ROLE`**。
这不是缺陷，是正确行为：从文件名里的 `hard_exit` / `scratch` 字样推断角色，
正是"静默猜值"。`exit_policy`、`bootstrap_budget` 同理——
v1 从文件名推断它们，v2 一律改为 `UNKNOWN_*` 除非 run card 提供。

> **后续工作（不在本轮）**：从各实验的预注册文档回溯构建
> `docs/data/run_cards/*.json`，每条须注明依据的文档与行号，可审计。
> 在此之前，任何需要 `experiment_role` 的分析都不得进行。

### 1.3 `mechanism_family`：只反映 checkpoint 实际保存了什么

由 `ptf_cfg` / `source_names` / `agent_type` 判定，映射真值表**运行前冻结**：

```
ptf_cfg 缺失或为空                              → NO_PTF
ptf_cfg 存在 且 source_names 为 ['null'] 或空    → PTF_NULL_BANK
ptf_cfg 存在 且 source_names 非空                → PTF_WITH_SOURCES
agent_type 不在已知集合                          → UNKNOWN_MECHANISM
```

它回答"这个文件里存了什么"，**不回答**"这次实验想干什么"——后者是
`experiment_role` 的事，二者不得互相推断。

## 2. 身份冲突（冻结）

文件名解析出的 `(env, seed, global_step)` 与 checkpoint 内部值不一致
→ 该文件 `EXCLUDED_IDENTITY_MISMATCH`，记录双方的值，**不做任何修正猜测**。

三项**逐一**比对，任一不符即排除。理由：文件名是人和脚本写的，
checkpoint 内部是训练进程写的，不一致意味着至少一方错了——
此时选哪一方都是猜。

## 3. Canonical 步集改为协议感知（冻结）

v1 用固定的 `[20000, 50000, 100000, FINAL]`。v2 改为**逐 run 计算**：

```
10000                                   racing / 短 horizon 判决点
20000                                   标准早期点
bootstrap_end                            = ptf_cfg["bootstrap_budget"]（若有）
hard_exit_step                           = ptf_cfg 中的 hard exit 步（若有）
50000, 100000                            标准中/终点
configured_total_timesteps               = args["total_timesteps"]
```

规则：

1. 逐 run 取上述值的**并集**，缺失项跳过（不填默认值）；
2. `ptf_cfg` 里 bootstrap / hard-exit 的**具体键名在实现时确定并写入结果文档**，
   找不到对应键 → 该项记 `UNKNOWN`，不猜；
3. **`FINAL` 不再是一个"步"**——见 §3.1。

### 3.1 `FINAL` 必须解析成内部 `global_step` 并去重

v1 把 `*_final.pt` 当作独立的 canonical 步，于是
`run__1_100000.pt` 与 `run__1_final.pt` 可能是**同一个 checkpoint 被数了两次**。

v2：`FINAL` 文件读取内部 `global_step`，与同 run 的数字步文件比对。

```
内部 global_step 与某个数字步文件相同 且 sha256 相同  → FINAL_DUPLICATE_OF_<step>，不计入 canonical
内部 global_step 与某个数字步文件相同 但 sha256 不同  → AMBIGUOUS_RUN_INSTANCE（§1.1），非零退出
内部 global_step 无对应数字步文件                     → 按该 global_step 计入 canonical
```

## 4. `completion_status` 在 **run 层**判断（冻结）

v1 的问题：把正常的中间 checkpoint 判成 `interrupted`。
中间点本来就不是终点，它"未完成"是**定义使然**，不是异常。

v2 规则：

```
先按 run_instance_id 分组，取该组的 max(global_step) = observed_end
observed_end >= configured_total_timesteps            → COMPLETED
observed_end <  configured_total_timesteps            → TRUNCATED_RUN
configured_total_timesteps 未知                        → UNKNOWN_COMPLETION
```

**单个 checkpoint 不再有 `completion_status`**——该字段挂在 run 上。
每个 checkpoint 只记 `is_run_endpoint: bool`。

## 5. Sentinel（约 20 个，冻结选取规则）

**选取规则先于运行冻结**，不得看到结果后调整名单。
sentinel **只验证 metadata，不跑任何 episode 评估**。

| # | 覆盖点 | 选取规则 |
|---|---|---|
| 1–3 | scratch | `exp_name` 含 `scr` / `scratch` 且 `ptf_cfg` 为空的前 3 个（按路径字典序） |
| 4–6 | PTF with sources | `source_names` 非空的前 3 个 |
| 7–8 | PTF null bank | `source_names == ['null']` 的前 2 个 |
| 9–11 | 三种典型 target | crawl / slide / truck 各取 1 |
| 12–14 | FINAL 文件 | `*_final.pt` 取 3 个，其中至少 1 个所在 run 同时有数字步文件（用于验 §3.1 去重） |
| 15–16 | 重复 step | 同一 `run_instance_id` 下同一 `global_step` 有多个文件的，取 2 组 |
| 17–18 | UNKNOWN 字段 | 预期产生 `UNKNOWN_MECHANISM` 或 `UNKNOWN_ROLE` 的各取 1 |
| 19 | 人为身份冲突 | **构造**：把某 checkpoint 复制并改名成错误的 seed，验证 `EXCLUDED_IDENTITY_MISMATCH` |
| 20 | 人为 step 冲突 | **构造**：复制并改名成错误的 global_step，同上 |

第 19/20 项是**注入的**，放在临时目录，不污染 `models/`。
若 5、15–18 的规则在实际数据中命中不足，**如实记为 `SENTINEL_UNAVAILABLE` 并说明**，
不得替换成别的文件凑数。

## 6. 退出条件（冻结）

```
任一 sentinel 出现身份冲突而未被识别               → 非零退出
任一字段被静默猜值（应 UNKNOWN 却给了具体值）      → 非零退出
run_instance_id 映射不唯一（AMBIGUOUS_RUN_INSTANCE）→ 非零退出
数据不全                                          → INCOMPLETE + 非零退出
```

**sentinel 全部通过后停下等 review，不得自行全量 deep scan。**
全量扫描需要单独批准。

## 7. 设计层自查（CLAUDE.md §8）

**8.1 辨别力**——平凡解释："sentinel 通过只是因为解析器没崩"。
排除：第 19/20 项是**注入的错误**，解析器不崩但必须**报错**才算通过；
§3.1 的 FINAL 去重要求给出具体的 `FINAL_DUPLICATE_OF_<step>` 判定，
不是"文件能读"。

**8.2 混淆变量**——`exp_name` 同时承载了"实验意图"与"run 标识"两种含义，
这是数据本身的缺陷（无 UUID）。处理方式是**只取其标识作用**
（拼成 `run_instance_id` 并检测冲突），**不从中解析角色**（§1.2）。

**8.4 前提是否蕴含结论**——验算退出条件能否恒不触发。
不能：第 19/20 项注入的冲突**必然**存在，若解析器不报错则必然非零退出。
反向：若解析器过度敏感把正常文件也判冲突，第 1–18 项会失败。两侧都可达。

**8.5 site selection**——sentinel 是**按规则选**的不是挑出来的，规则先冻结。
但仍须声明：本轮结论只覆盖这 20 个文件，**不得**外推到全部 1661 个。

**8.6 是否重演本轮教训**——
"结果 commit 改代码"→ §8 写死提交结构；
"用更弱的代理替换判据"→ §5 每项写明具体选取规则，不留"等价物"空间；
"把 post-hoc 文档叫 prereg"→ §0 已声明盲性边界。

## 8. 提交结构

```
预注册   本文件                                  ← 先于实现
实现     inventory v2 脚本 + 单元测试             ← 无任何运行产物
结果     sentinel 输出 + 结论文档                 ← git show --stat 不得有 .py/.sh
```

**若结果暴露 bug**：按 `CLAUDE.md §4.1` 走。
