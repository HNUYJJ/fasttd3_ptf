# 结果：Checkpoint Inventory 第一遍 —— `INCOMPLETE`（按设计）

> 2026-08-06。预注册 `checkpoint_inventory_v1_prereg_20260806.md`（`76fa2e2`），
> 脚本 `scripts/analysis/build_checkpoint_inventory.py`（先于运行冻结）。
> 输出 `docs/data/checkpoint_inventory_v1/manifest.json`，退出码 `2`。

## 0. 本文件**不能**用来做什么

预注册 §5 明令：第一遍 manifest **不得用于任何统计或场地判断**。
深度字段（`method_family` / `training_commit` / `source_bank_digest` /
`bootstrap_budget` / `exit_policy` / `completion_status`）全部为
`UNKNOWN_NEEDS_DEEP_SCAN`，因此：

- **不得**据此说某任务有 N 个 scratch 基线——method 尚未确定；
- **不得**据此排序候选场地——那是 P3A 的事，且需要第二遍数据；
- 下表的 `runs` / `seeds` 只说明**文件存在性**，不说明可用性。

## 1. 总量

```
扫描        1661 个 .pt
排除         135    ANCHOR_BUNDLE 99 / SMOKE_OR_DEBUG 24 / UNPARSEABLE_NAME 12
待深扫      1526
canonical    964    （落在 {20000, 50000, 100000, FINAL} 的文件）
```

`canonical` 的选取**只依赖 `global_step`**，脚本不 import torch、
不读取任何 return / eval 数据——这是对 v1 `J_best_known` winner's curse 的
结构性根治，可由代码审查直接验证。

## 2. 按 env 的文件清单（已按 `run_group_id` 去重）

`seeds` 列是**不同 `learner_seed` 的个数**，不是文件数——
同一 run 的多个 step 不是独立样本（预注册 §3）。

| env | run groups | distinct seeds | canonical files | excluded |
|---|---:|---:|---:|---:|
| `h1hand-balance_hard-v0` | 22 | 3 | 22 | 0 |
| `h1hand-balance_simple-v0` | 1 | 1 | 1 | 0 |
| `h1hand-basketball-v0` | 60 | 14 | 63 | 2 |
| `h1hand-bookshelf_simple-v0` | 9 | 3 | 9 | 0 |
| `h1hand-cabinet-v0` | 58 | 3 | 70 | 0 |
| `h1hand-crawl-v0` | 73 | 3 | 93 | 9 |
| `h1hand-door-v0` | 85 | 9 | 142 | 1 |
| `h1hand-hurdle-v0` | 99 | 3 | 144 | 9 |
| `h1hand-maze-v0` | 26 | 3 | 26 | 0 |
| `h1hand-package-approach-v0` | 2 | 1 | 2 | 0 |
| `h1hand-package-contact-v0` | 2 | 1 | 2 | 0 |
| `h1hand-package-nearcarry-v0` | 3 | 1 | 3 | 0 |
| `h1hand-package-v0` | 8 | 1 | 8 | 0 |
| `h1hand-pole-v0` | 27 | 3 | 27 | 0 |
| `h1hand-powerlift-v0` | 29 | 3 | 29 | 2 |
| `h1hand-push-v0` | 21 | 3 | 21 | 0 |
| `h1hand-reach-v0` | 3 | 1 | 3 | 0 |
| `h1hand-run-v0` | 1 | 1 | 1 | 0 |
| `h1hand-slide-v0` | 89 | 7 | 152 | 3 |
| `h1hand-spoon-v0` | 12 | 3 | 12 | 0 |
| `h1hand-stair-v0` | 39 | 3 | 54 | 0 |
| `h1hand-stand-v0` | 4 | 1 | 4 | 0 |
| `h1hand-truck-v0` | 35 | 3 | 49 | 0 |
| `h1hand-walk-v0` | 1 | 1 | 1 | 0 |
| `h1hand-window-v0` | 26 | 3 | 26 | 0 |

## 3. 与场地普查 v1 的关系（重要）

场地普查 v1 报告22 个 target 无任何 source-free 评估数据，据此判 `UNKNOWN`。
本 inventory 显示：**其中多数任务是有 checkpoint 的，只是从未用冻结面板评估过。**

普查扫描的是 `docs/data/**/source_free_eval/`（评估产物），
本 inventory 扫描的是 `models/` 与 `artifacts/`（训练产物）。二者是不同的东西。

**这印证了路线 B 的判断**：补齐 `UNKNOWN` 主要不需要新训练，
而需要用 evaluator v2 重评已有 checkpoint。但具体哪些可用，
**必须等第二遍确定 `method_family` 与 `completion_status` 之后才能说**。

## 4. 已知限制

1. **深度字段全部未知**。第一遍只读文件名与 size/mtime。
2. **`UNPARSEABLE_NAME` 12 个未逐一核查**。它们可能是有价值的产物，
   也可能是临时文件；已如实计入排除并保留在 manifest 中，不得静默丢弃。
3. **`SMOKE_OR_DEBUG` 靠 run_name 关键词匹配**，可能误伤命名含 `test` 的正式 run。
   第二遍读到 `args` 后应复核这 24 个。
4. **`basketball` 有 14 个 distinct seed**，明显多于其他任务的 3 个，
   提示该任务的 seed 编号跨实验不一致。第二遍须确认它们是否真属不同 learner。
5. **不含 `checkpoints/p0_anchors/`**（99 个 anchor bundle 文件），
   它们是分叉点快照而非评估对象，按预注册 §1 排除。

## 5. 下一步的两个前置条件

```
第二遍 deep scan   只对 964 个 canonical 文件读 checkpoint 内容
                   填 method_family / training_commit / source_bank_digest /
                   bootstrap_budget / exit_policy / completion_status + sha256
                   —— 读元数据，不产生评估结果，不受 P0 推送条款约束

重评               必须先满足 P0「推送到远程审计分支」——产生正式评估结果
```
