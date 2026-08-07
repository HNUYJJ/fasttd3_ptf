# 结果：Evaluator v2.1c（A3–A6 / A9）—— `CORE_PATHS_VERIFIED_WITH_GAPS`

> 2026-08-07。判据来自 **post-diagnostic reverification protocol**
> `evaluator_v21c_reverification_protocol_20260807.md`（`5d92eb5`，先于实现冻结）
> 与 `evaluator_v21b_prereg_20260807.md`（`91fc0ef`）。
> **该二者都不是 blind prereg**——写它们时我已看过前几轮结果，定性见各自头部声明。
>
> **冻结实现 `0bd8640`**。本文件所报结果全部在该实现上产出，期间未修改任何一行代码。
> 这一点由 `evaluation_semantics_digest` **自证**（见 §2），不是我的口头保证。
>
> 原始输出 `docs/data/evaluator_v21c_smoke/smoke.json`，退出码 `0`。

## 1. 裁决

```
VERDICT: CORE_PATHS_VERIFIED_WITH_GAPS      失败 0 项，VACUOUS 2 项

  S1  crawl        PASS      0/8 终止，语义全 neutral
  S2  slide        PASS      5/8 终止，全部判为 failure
  S3  truck        VACUOUS   milestone 已验证；0/8 终止 → success 语义未执行
  S4  bookshelf    VACUOUS   milestone 已验证；0/8 终止 → reason 映射未执行
  S5  basketball   PASS      8/8 终止全 failure；ball_to_hoop_dist 实测有限值
  S6  truck 聚合   PASS      聚合覆盖整条 trajectory（n_steps_present = 1000）
  S10 reducer      PASS      truck / bookshelf / basketball 三者的冻结 reducer 全部真实输出

单元测试：94 passed, 1 skipped（S7/S8/S9/S11 在此，见 §5）
claim linter：OK（2 活跃 + 1 历史），exit 0
sync 回归测试：4 组全过，exit 0
```

## 2. 实现未被污染，这一点是自证的

上一轮我只能声明"期间没改代码"。本轮 `evaluation_semantics_digest` 让它可验证：

```
smoke.json 记录的 digest   80619b7e4f33a7c7…
结果提交时重新计算         80619b7e4f33a7c7…      一致
git diff 0bd8640 -- <三个语义文件>   空
```

digest 覆盖 `schema_version` + `source_free_mode` + `p0_evaluator_v2.py` /
`task_metrics.py` / `schema_v2.py` 三个文件的内容摘要。
**任何一个字节的改动都会让它对不上**——这正是 A5 想要的性质，
顺带把"我保证没改"变成了可核对的事实。

## 3. S10 给出了一个上一轮拿不到的 runtime 证据

`first_hit_step`（首次为真）与 `first_step`（首次出现该字段）是两个不同的量。
上一轮只有单元测试 `T19b` 覆盖，本轮 basketball 给出了**真实数据上的分离**：

```json
basketball.success_subtasks = {
  "first_step": 0,          ← 该字段从第 0 步就存在
  "first_hit_step": 9,      ← 但第一次为真是第 9 步
  "max": 1, "final": 1, "max_step": 9,
  "n_steps_present": 125, "ever_true": true
}
```

`0 ≠ 9`。若把 `first_hit_step` 实现成 `first_step` 的别名（一个很容易犯的错），
这里会露馅。对照 truck / bookshelf：它们的 `first_hit_step` 全是 `null`
（策略一个 package 都没装上 / 一本书都没上架），`ever_true = false`——
这是**合法观测**而非缺陷，故 S10 明确允许 `first_hit_step` 为 null。

三个任务的冻结 reducer 全部真实输出，无一 `MISSING_MILESTONE_FIELD`：

| 任务 | 声明 | runtime 实测 |
|---|---|---|
| truck | `success_subtasks: max+final`；`success: ever_true+first_hit_step` | 全部非 null（`first_hit_step` 除外，从未达成） |
| bookshelf | 同上 | 同上；`success` 的值是 `False`（bool），`max` 存原值 |
| basketball | `success_subtasks: max+final` | `max=1, final=1, max_step=9` |

## 4. 仍未在 runtime 观察到的：`max > final` 的回落

**这条限制从上一轮延续下来，本轮没有补上。**

D3 修复针对的具体故障是"中途装上车又掉下来的 package 被丢失"
（`truck.py:113-115` 的 `.remove()` 分支）。要观察到它，需要
`max > final` 的 episode。本轮实测：

```
truck        max=0, final=0      策略一个 package 都没装上
bookshelf    max=0, final=0      一本书都没上架
basketball   max=1, final=1      相等，无回落
```

即**回落捕获仍只有单元测试 `T11` / `T11b` / `T11c` 覆盖**。
补齐它需要一个真能装上 package 的 truck 策略——本地没有。
不得把 S6 / S10 的 PASS 读成"已证明能抓住回落"。

## 5. S7–S9 / S11 由单元测试覆盖，理由与结论

protocol §6 规定这四项用单元测试实现：它们需要构造损坏的 manifest 与
被篡改的文件，在真实 MuJoCo 里做既慢又不可控。

| # | 覆盖测试 | 关键断言 |
|---|---|---|
| S7 identity manifest | `T16`–`T16g`（8 条） | 无 manifest 拒绝；四个必需字段**逐一**缺失都拒绝（参数化，不是只查一两个）；SHA 不符拒绝；有 `ptf_cfg` 却无 protocol digest 拒绝；**scratch 无 `ptf_cfg` 时不得因此被拒**（`T16e`，防过度收紧把整条基线挡在门外）；manifest 文件损坏/非对象/不存在各有独立报错 |
| S8 formal 禁 overwrite | `T20` + CLI 分支 | 目标存在时 `FileExistsError`，原文件**逐字节不变**；无 tmp 残留；`allow_overwrite` 才替换 |
| S9 semantics digest | `T17`–`T17e`（5 条） | **实际篡改 `task_metrics.py` 一个字节**后 digest 改变、恢复后回到原值；四个 purpose 在 digest 不同时**全部**拒绝；旧产物缺该字段 → 不可比而非当作相同放行 |
| S11 debug 文件名护栏 | CLI 分支 | debug 输出名不含 `.debug.` 段即拒绝 |

`T18` 另外锁住 A9 的两条语义：`paired_by_seed` 缺 `match_group` /
`training_protocol_digest` 即不可比；而 `checkpoint_sha256` **只在**
`same_checkpoint` 下要求相等（`T18d` 显式验证其余 purpose 不因 SHA 不同被拒——
SHA 的作用是身份，不是可比性）。

## 6. VACUOUS 两项与科学裁决阻断（与上一轮一致）

```
scientific_use_blocked:
  h1hand-truck-v0             termination_semantics   0/8 终止
  h1hand-bookshelf_simple-v0  termination_semantics   0/8 终止
```

`site_rules.require_runtime_verified(env, purpose=...)` 是强制点。
bookshelf 在真实 termination path 被验证之前**不得进入 P3 科学裁决**。
它不阻塞纯 checkpoint 元数据 inventory（P2 不涉及任务语义）。

## 7. 完整命令与产物

```bash
PY=/home/yjj/miniconda3/envs/FastTD3/bin/python
PYTHONPATH=. $PY -m pytest tests/test_evaluator_schema_v2.py -q
#   → 94 passed, 1 skipped（T4 需 EVAL_V2_INTEGRATION_CKPT）
PYTHONUNBUFFERED=1 PYTHONPATH=. $PY scripts/analysis/smoke_evaluator_v21c.py
#   → CORE_PATHS_VERIFIED_WITH_GAPS, exit 0
PYTHONPATH=. $PY scripts/analysis/claim_linter.py
#   → OK（2 个活跃文件 + 1 个历史文件）, exit 0
bash tests/test_sync_to_publish.sh
#   → 全部通过, exit 0
```

```
12c10fb2e2c51cae00dec753b25176a33486d3550c4aaaaaefa8ed9e3b3eb181  scripts/p0_evaluator_v2.py
c8bfe43a1c01fd97db41be279f44cab7e7c6e8b9ce3dd199768bd3281868434f  fasttd3_ptf/evaluation/task_metrics.py
06719f491bed02372f8be07720c359f748e348c12e46570a7f100555c904b4b3  fasttd3_ptf/evaluation/schema_v2.py
1eb3e577c4b21cce6a990d52329d7de2874c5e4a5017495df3153bd62119d6d3  fasttd3_ptf/evaluation/site_rules.py
aaaeae7965adbdf144e650e57597733c8ec0256a9aef3d1e73edc422c1ea5c83  scripts/analysis/smoke_evaluator_v21c.py
fa2093ab9d848f4be8dedcb425107aeaa37b989540ed54fdb6c6ba74b774c0ff  scripts/analysis/claim_linter.py
79268dfc0ff4dda6ce0d1a915e2ee12c96948d2e8829bb5e0370efdb56756fbc  docs/data/evaluator_v21c_smoke/smoke.json
```

## 8. 已知限制

1. **`max > final` 的回落捕获未在 runtime 验证**（§4）——只有单元测试覆盖，
   需要一个真能装上 package 的 truck 策略；
2. **S3 / S4 的终止语义未验证**，已被 `scientific_use_blocked` 拦截；
3. **bookshelf 的 milestone 只验证了低值区间**——其成功事件与终止 reason 1
   是同一事件，终止既未发生，高值区间必然也未被观察到；
4. **formal 模式从未在真实 checkpoint 上端到端跑过**。smoke 全部走 debug
   （smoke 产物本就不得用于科学裁决），formal 的强制性由 `T16` 系列
   在构造的 checkpoint 上验证。P2 的 sentinel 阶段仍不涉及 formal 评估
   （只读元数据），故**第一次真实 formal 评估将发生在 P3 之前**，届时须
   单独确认一次；
5. **`identity manifest` 目前没有生成器**——须手工写或由 P2 inventory 产出。
   P2 预注册应把它纳入交付物，否则 formal 模式在实践中会因为"太麻烦"被绕开；
6. **claim linter 是行级正则**，能挡住原措辞回流，挡不住语义等价的改写
   （例如把 impossibility 写成 "cannot in principle"）。它是护栏不是判官；
7. **smoke 每项仅 8 episodes**，只验证通路，数值不得用于任何科学判断。
