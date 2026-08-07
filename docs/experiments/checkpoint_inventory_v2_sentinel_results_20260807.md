# 结果：Checkpoint Inventory v2 Sentinel —— `SENTINEL_FAILED`（退出码 1）

> 2026-08-07。预注册 `checkpoint_inventory_v2_prereg_20260807.md`（`bf14f20`，先于实现）。
> **冻结实现 `2835633`**，本文件所报结果全部在该实现上产出，期间零代码改动。
> 原始输出 `docs/data/checkpoint_inventory_v2/sentinel.json`，**退出码 `1`**。

## 1. 裁决

```
VERDICT: SENTINEL_FAILED           退出码 1
  处理 20 | ELIGIBLE 16 | 文件名↔内部身份冲突 0 | 注入冲突 2（全部正确识别）
  ! 2 组 AMBIGUOUS_RUN_INSTANCE（映射不唯一）
  SENTINEL_UNAVAILABLE: racing 规则 0/2
```

**这不是脚本 bug，是判据按设计工作。** 预注册 §6 写明"`run_instance_id`
映射不唯一 → 非零退出"，它触发了，且触发原因是**真实的数据歧义**（§3）。

按预注册 §6，**到此停下等 review，不自行全量 deep scan**。

## 2. 先确认检测能力本身有效

"0 个文件名↔内部身份冲突"这个结果本身没有信息量——它既可能是数据干净，
也可能是检测器坏了。故预注册 §5 第 19/20 项要求**注入**冲突：

```
注入 1  复制某 checkpoint，文件名 seed 改为 +77   → 期望 EXCLUDED_IDENTITY_MISMATCH
注入 2  复制同一 checkpoint，文件名 step 改为 +12345 → 期望 EXCLUDED_IDENTITY_MISMATCH
实测    2/2 均被正确识别
```

两个注入件写在临时目录、跑完即删，未污染 `models/`。
**先确认检测器会响，"没响"才是证据。**

## 3. 真实发现：`p0_dup_archive` 造成 run_instance 映射不唯一

```
run_instance_id = p0_crawl_abstain#1
  @10750  2 个不同 sha256
          models/h1hand-crawl-v0__p0_crawl_abstain__1_10750.pt
          models/p0_dup_archive/crawl_B/h1hand-crawl-v0__p0_crawl_abstain__1_10750.pt
  @11500  同上
```

即：**同一 `(exp_name, seed, global_step)` 存在两个内容不同的模型**。
这正是预注册 §1.1 预见的情形——本仓库没有 run UUID，
`exp_name#seed` 在同一实验被重跑时不唯一。它从预见变成了实测。

### 3.1 顺带查出 v1 inventory 的一个缺陷

```
models/p0_dup_archive/  下共 20 个 .pt，分 crawl_A / crawl_B / truck_A / truck_B
v1 inventory 收录了全部 20 个（PENDING_DEEP_SCAN），其中 4 个被判为 is_canonical
```

即 **v1 的 `964` canonical 里混入了 4 个来自重复归档的文件**。
`docs/` 下 `grep` 不到任何记录该归档的文档（只有 inventory 自身的输出提到它），
**归档理由不可考**，A/B 两组孰为"正版"无从判定。

### 3.2 我**没有**做的事，以及为什么

一个很自然的"修法"是加一条 `EXCLUDED_ARCHIVED` 规则把 `p0_dup_archive/`
整个排除，这样 sentinel 立刻转绿。**我没有这样做**：
那是在同一数据上、看到失败之后加一个已知能通过的门
（`CLAUDE.md §8.7` 的 outcome-contingent gate switching 红线），
即使理由听起来正当、即使如实披露，也不能恢复确认性地位。

这条规则**该不该加是 review 的决定**，不是我的。若决定加，正确做法是：
新写一份预注册、说明排除理由与依据、重新跑一遍，而不是在本轮补丁。

## 4. 另外三项发现

### 4.1 `mechanism_family` 无一为 `NO_PTF`——文件名确实不能推断机制

sentinel 里有 3 个是按 `scr|scratch` 命名规则选的，但实测：

```
PTF_WITH_SOURCES  16
PTF_NULL_BANK      4
NO_PTF             0
```

**命名含 `scr` 的文件，checkpoint 内实际都存有 `ptf_cfg`。**
若按 v1 的做法从文件名推断 `method_family`，它们会被标成 scratch，
而实际保存的是 PTF 配置（多半是 null-bank 的 scratch 对照臂）。
这是预注册 §1.3"机制只反映文件里存了什么、不回答实验想干什么"的实测支持。

### 4.2 `FINAL` 去重逻辑**未被触发**（VACUOUS）

3 个 `*_final.pt` sentinel 所在的 run 都没有同 `global_step` 的数字步文件，
故 `final_dedup_notes` 为空——**§3.1 的去重路径这次一次都没执行**。
与 evaluator 那边的处理一致：条件式判据在前提不成立时真空为真，
**不得计为已验证**。该路径目前只有单元测试 `T5` 覆盖。

### 4.3 racing 规则 `SENTINEL_UNAVAILABLE`（0/2）

按 `racing` 命名的 checkpoint 在 eligible 集合中一个都没有。
如实记录，**未替换成别的文件凑数**（预注册 §5 明令）。
后果：racing 臂的元数据解析这次没有被覆盖。

## 5. 预注册自身的一个缺陷（如实报告，本轮未改）

预注册 §4 用 `configured_total_timesteps`（`args["total_timesteps"]`）判 completion。
但实现时核实到 `ptf_cfg` 另有 `run_stop_step`（`train_ptf.py:344/486`），
它是**训练退出**控制，与 `total_timesteps` 可以不同——
实测某 run 为 `run_stop_step=20000` 而 `total_timesteps=100000`。

这类 run 会被判成 `TRUNCATED_RUN`，而它其实是**按配置正常结束**的。
本轮实测 `TRUNCATED_RUN` 3 个 / `COMPLETED` 12 个，其中有多少属于这种误判
需要 run 层信息才能定，而这又依赖 run card。

**判据已冻结，本轮如实实现未改**，仅把 `run_stop_step` 作为诊断量输出。
修正建议留给下一版预注册：`completion` 应以
`min(total_timesteps, run_stop_step)`（若后者存在）为准。

## 6. 完整命令与产物

```bash
PY=/home/yjj/miniconda3/envs/FastTD3/bin/python
$PY tests/test_checkpoint_inventory_v2.py
#   → 24 项全部通过
PYTHONPATH=. $PY scripts/analysis/build_checkpoint_inventory_v2.py
#   → SENTINEL_FAILED, exit 1
```

```
5e7eae0bc6ed9d90f13ffc581ccfdaee821ac32636b9e1fab47516766652f60b  scripts/analysis/build_checkpoint_inventory_v2.py
9b73eb0d3ab0df8d6d73b0aedb369b2fc3894e20a89c81653c7212be42fdc995  tests/test_checkpoint_inventory_v2.py
7f16d6ac0b8cc11fbcd580dfbb2ed56375e4120b3026ff614d041a7bf8988333  docs/data/checkpoint_inventory_v2/sentinel.json
```

## 7. 需要 review 决定的问题（我不自行决定）

1. **`p0_dup_archive/` 是否应整体排除？** 若是，需新预注册说明理由与依据，
   而不是本轮补丁（§3.2）。同时 v1 的 `964` canonical 需相应订正。
2. **A/B 两组孰为正版？** 当前无任何文档可依。若无法判定，
   建议两组都标 `AMBIGUOUS` 并排除出科学裁决，而不是二选一。
3. **是否先建 run card registry？** `experiment_role` 当前 100% `UNKNOWN_ROLE`，
   inventory 的下游用途（区分 scratch / hard-exit / racing 臂）全部阻塞在这里。
   建议从各实验预注册文档回溯构建、每条注明依据文档与行号。
4. **completion 判据是否改用 `min(total_timesteps, run_stop_step)`？**（§5）

## 8. 已知限制

1. **本轮结论只覆盖这 20 个文件**，不得外推到全部 1661 个（预注册 §8.5）；
2. **FINAL 去重路径未被真实触发**（§4.2），只有单元测试覆盖；
3. **racing 臂未被覆盖**（§4.3）；
4. **`experiment_role` 100% UNKNOWN**，这是正确的 fail-closed 行为，
   但意味着 inventory 目前**不能**回答"哪些是 hard-exit 臂"这类问题；
5. **`hard_exit_step` 记为 `UNKNOWN_NO_DEDICATED_KEY`**——已 `grep` 全仓库确认
   `ptf_cfg` 内无该专用键，它在语义上等于 `mcg_warmup_steps` 但那是设计层解释；
6. **全量 deep scan 未执行**，需单独批准（脚本要求 `--full --approved-by`）。
