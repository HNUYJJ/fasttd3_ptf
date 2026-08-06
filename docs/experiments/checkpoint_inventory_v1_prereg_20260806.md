# 预注册：Checkpoint Inventory v1

> 2026-08-06。目标 P2 的判据冻结文件，**必须先于任何 manifest 生成提交**。
> 提交后只允许改路径参数，不得修改字段定义、eligibility 判据或 canonical 选取规则。

---

## 0. 本阶段要防的三件事

| # | 风险 | 来源 |
|---|---|---|
| R-1 | 把同一 run 的多个 step 当成独立样本 | 目标 P2 明令；site screen v1 已犯 |
| R-2 | 跨 method/seed/step 取最大值代表任务表现（winner's curse） | v1 的 `J_best_known` 缺陷，已撤回 |
| R-3 | smoke / 失败 / 身份不明的 run 混入正式统计 | 本地 1696 个 `.pt` 来源混杂 |

**本阶段不产生任何评估数据**，只建立 checkpoint 的身份清单。
重评是 P2 的第二步，须在推送完成后进行（P0 条款）。

---

## 1. 扫描范围（冻结）

```
纳入   models/**/*.pt          评估用 checkpoint（1552 个）
纳入   artifacts/**/*.pt       分支产物（109 个）
排除   checkpoints/p0_anchors/**  anchor bundle（learner/rng/replay 三件套，非评估对象）
排除   任何 replay.pt / rng.pt    同上
```

## 2. Manifest 字段（冻结）

每条记录必须包含以下全部字段；任一无法确定即填 `UNKNOWN`，**不得推断补齐**：

```yaml
path:                # 仓库相对路径
checkpoint_sha256:   # 文件内容哈希（延迟计算，见 §5）
env_name:            # 从文件名第 1 段解析
run_name:            # 从文件名第 2 段解析
run_group_id:        # == run_name；同一 run 的所有 step 共享
learner_seed:        # 从文件名第 3 段解析
global_step:         # 从文件名第 4 段解析；"final" → FINAL
file_size_bytes:
file_mtime_utc:

# ── 以下需读 checkpoint 内容才能确定，第一遍一律 UNKNOWN_NEEDS_DEEP_SCAN ──
method_family:       # scratch | continuous_source | hard_exit | racing_arm | smoke | UNKNOWN
training_commit:
source_bank_digest:
bootstrap_budget:
exit_policy:
completion_status:   # COMPLETE | INTERRUPTED | UNKNOWN

# ── 判定结果 ──
eligibility:         # ELIGIBLE | EXCLUDED | PENDING_DEEP_SCAN
exclusion_reason:    # 见 §4
is_canonical:        # 是否被选为重评对象，见 §3
```

## 3. Canonical checkpoint 的选取规则（防 R-2 的关键）

**canonical 必须是预先指定的固定步数点，绝不是"表现最好的那个"。**

```
canonical_steps = {20000, 50000, 100000, FINAL}
```

对每个 `(env_name, run_group_id)`，其 canonical 集合 =
该 run 中 `global_step` 落在 `canonical_steps` 内的全部 checkpoint。

规则要点：

1. **不看任何 return 值**——选取只依赖 `global_step`，故不可能产生 winner's curse；
2. 同一 `run_group_id` 的多个 canonical step **不是独立样本**，
   统计时按 `run_group_id` 去重，`n_seeds` 只数不同的 `learner_seed`；
3. 某 run 没有任何 canonical step → 该 run 整体 `is_canonical=false`，
   但仍留在 manifest 中并注明，**不得静默丢弃**。

## 4. `exclusion_reason` 取值（冻结，互斥）

```
ANCHOR_BUNDLE          learner.pt / replay.pt / rng.pt —— 非评估对象
UNPARSEABLE_NAME       文件名不符合 {env}__{run}__{seed}_{step}.pt
SMOKE_OR_DEBUG         run_name 含 smoke / toy / debug / dbg / test / trial
UNKNOWN_ENV            env_name 不在 HumanoidBench 注册表中
INCOMPLETE_RUN         completion_status == INTERRUPTED（需 deep scan 确定）
UNKNOWN_IDENTITY       deep scan 后仍无法确定 method_family
```

`SMOKE_OR_DEBUG` 的关键词匹配**大小写不敏感**，且只匹配 `run_name` 段，
避免误伤 env 名或路径。

## 5. 两遍扫描（工程约束，非判据放宽）

本地 `.pt` 合计 **126 GB**，全量反序列化不可行。故：

```
第一遍（本次）  只读文件名 + 文件系统元数据（size / mtime）
                → 产出完整 manifest，深度字段一律 UNKNOWN_NEEDS_DEEP_SCAN
                → 计算 canonical 集合（只依赖 global_step，不需读内容）

第二遍（后续）  只对 is_canonical == true 的文件读 checkpoint 内容，
                填充 method_family / training_commit / source_bank_digest /
                bootstrap_budget / exit_policy / completion_status，
                并计算 checkpoint_sha256
```

**第一遍的 manifest 不得用于任何统计或场地判断**——它只回答"有哪些文件、
哪些值得深扫"。所有需要 `method_family` 的判断必须等第二遍。

## 6. 设计层自查（CLAUDE.md §8）

**8.1 辨别力**：平凡解释——"canonical 选取只是换个方式挑好看的 checkpoint"。
排除方式：canonical 只依赖 `global_step` 这一个与性能无关的量，
选取代码不接触任何 return / eval 数据（可由代码审查直接验证）。

**8.2 混淆变量**：`run_name` 中的时间戳与代码版本共变，故 `training_commit`
必须从 checkpoint 内部读取，**不得**从时间戳推断。

**8.4 前提是否蕴含结论**：验算 `ELIGIBLE` 是否必然非空——不必然。
若全部 run 都被 `SMOKE_OR_DEBUG` 或 `UNPARSEABLE_NAME` 排除，
则输出 `NO_ELIGIBLE_CHECKPOINTS` 并非零退出，该分支可达。

**8.5 site selection**：本阶段**不选场地**。manifest 覆盖全部 env，
不按任务价值排序、不做任何取舍。

**8.6 是否重演本轮教训**：
- v1 的跨 step 取 max → §3 明令 canonical 只依赖固定步数；
- v1 的数据缺失落进裁决分支 → §2 全部 UNKNOWN 显式化，§5 第一遍禁止用于统计；
- M33（推理代替查询）→ `method_family` 必须读文件内容，禁止从文件名猜。

## 7. 产物与停止条件

```
脚本    scripts/analysis/build_checkpoint_inventory.py  （本文件提交后才允许编写）
输出    docs/data/checkpoint_inventory_v1/manifest.json
汇总    docs/experiments/checkpoint_inventory_v1_results_20260806.md
```

脚本要求：

- 任一 checkpoint 无法解析 → 该条 `eligibility=EXCLUDED` + 具体 `exclusion_reason`，
  **不得跳过不记**；
- 存在 `UNKNOWN_NEEDS_DEEP_SCAN` 时**非零退出**（提示 manifest 尚不完整）；
- 输出必须列出按 `run_group_id` 去重后的 `(env, method, n_seeds)` 三元组统计，
  使"同一 run 多 step 不是独立样本"可被直接核对。

**停止条件**：第一遍完成后不得直接进入重评——重评需要
(a) 推送完成（P0 条款），(b) 第二遍 deep scan 完成。
