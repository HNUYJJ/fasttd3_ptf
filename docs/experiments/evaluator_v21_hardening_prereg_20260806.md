# 预注册：Evaluator v2.1 Hardening

> 2026-08-06。**必须在任何实现之前提交 git。** 提交后只允许改路径参数，
> 不得修改字段定义、校验规则或 smoke 判据。

---

## 0. 要修的是什么

外部 review 指出、经核实成立：**evaluator v2 丢掉了 v1 的两项安全能力**，
因此我此前写的"v2 与 v1 唯一差别是任务语义层"**不成立**。

| 能力 | v1 | v2（当前） |
|---|---|---|
| checkpoint 身份校验 | `p0_evaluator.py:156-175`：`--expect-global-step` / `--expect-seed` / `--expect-admission-mode`，并输出 `identity_checked` | **无**。只接受用户给的 `--env-name`，不核对 checkpoint 内容 |
| 拒绝覆盖已有结果 | `p0_evaluator.py:228`：`if out_path.exists()` 硬拒 | **无**。直接 `write_text()` |

后果：批量重评时，喂错 checkpoint、错 env、错预算或重复覆盖，
**仍会生成外观合法的 JSON**。这在 P2 的 964 文件规模上是不可接受的风险。

第三项缺陷：`site_rules.require_comparable()` 只检查 `global_step`，
测试甚至明确允许 `crawl@100k` 与 `hurdle@100k` 通过。**"同为 100k"远不等于可比。**

---

## 1. 身份校验（冻结）

`p0_evaluator_v2.py` 恢复并强化 v1 的校验：

```
--expect-env             checkpoint 内 args["env_name"] 必须匹配
--expect-global-step     checkpoint 内 global_step 必须匹配
--expect-seed            checkpoint 内 args["seed"] 必须匹配
--expect-admission-mode  checkpoint 内 ptf_cfg["admission_mode"] 必须匹配
```

**强制 env 交叉核对**：即使不传 `--expect-env`，也必须用 checkpoint 内的
`args["env_name"]` 与命令行 `--env-name` 核对，不一致即硬失败。

> **2026-08-06 描述更正**：本条初稿写作"新增 v1 没有的一条"，**这是错的**——
> 读 `p0_evaluator.py:152-155` 后确认 v1 已有该强制核对。v2.1 只是把它**恢复**
> 回来，不是新增。判据本身不变（仍要求强制核对），仅更正对 v1 的事实描述。

`identity_checked` 的定义（比 v1 更严）：

```python
identity_checked = (
    env_matches_checkpoint        # 强制项，恒为真否则已抛错
    and all(expect_* 中显式传入的项都已匹配)
    and n_explicit_expectations >= 1   # 至少显式声明一项，否则为 False
)
```

即：**只做强制 env 核对而不传任何 `--expect-*` 时，`identity_checked=false`**，
输出仍然产生但明确标记为未经完整身份声明。

## 2. 输出字段（冻结，在 v2 基础上新增）

```yaml
schema_version: 2.1
checkpoint:
  path:
  sha256:
  global_step:            # 读自 checkpoint 内部，非文件名
  learner_seed:           # 读自 args["seed"]
  env_name_in_checkpoint: # 读自 args["env_name"]
  training_commit:        # 读自 args 中的 git 信息（无则 UNKNOWN）
  args_digest:            # args 全量的 sha256（可比对同配置）
  ptf_cfg_digest:         # ptf_cfg 全量的 sha256
  ptf_cfg_summary:        # admission_mode / source_names / bootstrap 相关字段
identity_checked: bool
identity_expectations:    # 实际传入了哪些 --expect-*
panel_digest:             # sha256(eval_seeds + ranks + episode_steps + deterministic)
```

`panel_digest` 是 `require_comparable` 的关键输入——不同面板产出的数字不可比，
而此前无法从输出中判断两份结果是否用了同一面板。

## 3. 拒绝覆盖（冻结）

```
输出路径已存在 → 硬失败，非零退出，不写任何字节
--allow-overwrite 显式传入时才允许，且在输出中记 overwrote_existing: true
```

## 4. `require_comparable` 按比较目的分级（冻结）

不再是单一规则，而是按**比较目的**要求不同的一致性集合：

```python
COMPARISON_REQUIREMENTS = {
    # 同一 target 上比不同方法（如 scratch vs hard-exit）
    "across_methods":  {"env_name", "global_step", "panel_digest", "schema_version"},

    # 同一方法比不同 learner seed
    "across_seeds":    {"env_name", "method_family", "global_step",
                        "panel_digest", "schema_version"},

    # 逐 seed 配对（最严，用于配对差值统计）
    "paired_by_seed":  {"env_name", "global_step", "panel_digest",
                        "schema_version", "learner_seed"},

    # 同一 checkpoint 的重复评估（用于验证可复现性）
    "same_checkpoint": {"env_name", "global_step", "panel_digest",
                        "schema_version", "learner_seed", "checkpoint_sha256"},
}
```

规则：

1. **必须显式传 `purpose`**，无默认值——防止"随手比一下"；
2. 要求集合中任一字段在任一侧**缺失**，即 `IncomparableError`
   （身份不完整不等于身份相同）；
3. 跨 `env_name` 的比较在所有 purpose 下均被拒绝。若确需跨 target 陈述，
   必须先在各自 target 内算出配对差值，再比较差值——由调用方负责，
   本函数不提供跨 target 通道；
4. 旧的单字段 `require_comparable(a, b)` 签名**移除**，不保留兼容别名——
   保留别名就会有人继续用弱检查。

## 5. 真实 runtime smoke（冻结判据）

必须用**真实 checkpoint + 真实 MuJoCo 环境**跑，不接受 mock。
每项的通过条件：

| # | 任务 | 验证什么 | 通过条件 |
|---|---|---|---|
| S1 | crawl | neutral 语义 | `terminated` 恒为 false；`termination_semantics == "neutral"`；`task_success == false`；`metric_status == "OK"` |
| S2 | slide | failure 语义 | 存在 `terminated=true` 的 episode，且其 `task_success == false`、`semantics == "failure"` |
| S3 | truck | success 语义与中间 milestone | `metric_status == "OK"`；`milestones` 含 `success_subtasks`；**若** 有 episode `terminated=true` 则其 `task_success == true` |
| S4 | bookshelf_simple | 条件终止 | `metric_status == "OK"`；若有 `terminated=true` 的 episode，其 `info["terminated_reason"]` 必须存在且被正确映射 |
| S5 | basketball | MuJoCo state 提取 | **`ball_to_hoop_dist` 不得恒为 None**——至少一个 episode 成功提取到有限数值 |

**S3 的降级条款**：truck 极难，8 episodes 内可能无任何成功。故 S3 的硬性要求是
**milestone 提取通路可用**（`success_subtasks` 出现在 `milestones` 中）；
"若终止则 `task_success=true`" 是条件式要求，无终止时该子句真空成立。
**不得**因为跑不出成功就跳过 S3。

**S5 不设降级**：`ball_to_hoop_dist` 恒 None 意味着 `_basketball_state` 的
提取路径是坏的，那么 basketball 的 `task_success` 将永远是
`INSUFFICIENT_STATE`——该任务实质上不可评估。必须修好。

### 5.1 smoke 的规模

每项 **8 episodes**（`EVAL_SEEDS[0] × RANKS`），CPU 即可。
目的是验证通路而非获得科学结论，**其数值不得用于任何科学判断**。

## 6. 设计层自查（CLAUDE.md §8）

**8.1 辨别力**：平凡解释——"smoke 通过只是因为环境能跑起来"。
排除方式：S1/S2 要求**具体的语义值**（neutral vs failure），
S5 要求**具体的数值可提取**，都不是"没崩溃"就能通过的。

**8.4 前提是否蕴含结论**：验算 S1–S5 是否可能全部真空通过。
不能：S1 要求 crawl 的 `terminated` 恒 false（若 crawl 实际会终止则失败）；
S5 要求至少一个有限数值（无降级条款）。否定分支可达。

**8.6 是否重演本轮教训**：
- E-1（凭推理不实测）→ 本阶段全部判据要求**真实运行**，不接受 mock；
- E-9（未查签名就调用）→ 实现前须读 `p0_evaluator.py` 的 `--expect-*` 完整实现；
- v2 丢失 v1 能力 → §1/§3 逐条对照 v1 的行号，确保不再遗漏。

## 7. 停止条件

```
S1–S5 任一失败 → 不得进入 P2 inventory v2
require_comparable 的新签名未覆盖 §4 全部 purpose → 不得进入 P2
```

## 8. 提交结构（三段式）

```
预注册   本文件（先于实现）
实现     p0_evaluator_v2.py 的身份校验/防覆盖 + site_rules.require_comparable 重写
         + 单元测试（不含 smoke 结果）
结果     smoke 完整输出 + 结论文档（不含代码改动）
```
