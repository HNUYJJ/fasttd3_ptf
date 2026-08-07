# 结果：Evaluator v2.1b（P1.1b）—— `CORE_PATHS_VERIFIED_WITH_GAPS`

> 2026-08-07。预注册 `evaluator_v21b_prereg_20260807.md`（`91fc0ef`，先于实现）。
> **冻结实现 `8042314`**，本文件所报结果全部在该实现上产出，
> 期间未修改任何一行代码——这正是上一轮（`19948c4`）没做到的事。
> 原始输出 `docs/data/evaluator_v21b_smoke/smoke.json`，退出码 `0`。

## 1. 裁决

```
VERDICT: CORE_PATHS_VERIFIED_WITH_GAPS      失败 0 项，VACUOUS 2 项

  S1 crawl        PASS      0/8 终止，语义全 neutral（Crawl 恒不终止，符合源码）
  S2 slide        PASS      5/8 终止，全部判为 failure
  S3 truck        VACUOUS   milestone 通路已验证；0/8 终止 → success 语义未执行
  S4 bookshelf    VACUOUS   milestone 通路已验证；0/8 终止 → reason 映射未执行
  S5 basketball   PASS      8/8 终止全 failure；ball_to_hoop_dist 实测有限值
  S6 truck 聚合   PASS      四个聚合字段全非 null，n_steps_present = 1000
```

**`ALL_PASS` 及任何全称词已按预注册 §6 禁用。** 上一轮的 `ALL_PASS` 与
同一份输出里 S4 的 `VACUOUS` 自相矛盾；本轮的 verdict 名字本身就带着 gap。

按预注册 §7：S1–S6 无失败项、formal 模式 / 原子写 / trajectory 聚合均已实现，
**可进入 P2**。VACUOUS 不阻塞 P2——P2 只读 checkpoint 元数据，不涉及任务语义。

## 2. 与上一轮（DIAGNOSTIC）的关键差别

| | 上一轮 `19948c4` | 本轮 |
|---|---|---|
| 三段式 | **不成立**（结果 commit 里改了两个 `.py`） | 预注册 `91fc0ef` → 实现 `8042314` → 本结果，实现冻结后零代码改动 |
| S5 判据 | `metric_status != INSUFFICIENT_STATE`（弱代理，0 终止时真空通过） | `isfinite(ball_to_hoop_dist)`，预注册原文，无条件 |
| S3 | `PASS` | `VACUOUS`——终止语义路径确实没被执行过 |
| verdict | `ALL_PASS` | `CORE_PATHS_VERIFIED_WITH_GAPS` |
| 未验证路径 | 只在文档里提了一句 | 进 `scientific_use_blocked`，并由 `require_runtime_verified` 强制 |

## 3. S5：判据改回原文之后，证据强度实质提高

上一轮只能证明"终止 episode 的 `metric_status` 不全是 `INSUFFICIENT_STATE`"。
本轮直接检查数值本身：

```
ball_to_hoop_dist = 3.673, 4.050, 5.650, ...      （逐 episode 不同）
mujoco_state_error = None                          （8/8 提取成功）
```

**逐 episode 取值不同**这一点很重要——它排除了"读到某个常量或缓存值"的
平凡解释。若提取路径坏掉、或读到 reset 时的固定初值，8 个 episode 应当相同。

这条判据之所以此前无法实现，是因为 `ball_to_hoop_dist` 用完即弃、不进输出。
本轮把 `mujoco_state` / `mujoco_state_error` 写入 episode 记录（预注册 §4）后
才谈得上"按原文检查"。**判据要能实现，它依赖的量就必须可见**。

## 4. S6：聚合结构已验证，但"回落捕获"在 runtime 上**未被观察到**

S6 通过，四个字段全部非 null：

```json
{"final": 0, "max": 0, "max_step": 0, "first_step": 0,
 "n_steps_present": 1000, "ever_true": false}
```

`n_steps_present = 1000` 证明聚合覆盖了整条 trajectory，而不是把最后一步
包了一层壳（包壳的话这个值只能是 1）。

**但必须如实声明一处限制**：该 truck checkpoint 的策略一个 package 都没装上，
`success_subtasks` 全程恒为 0，因此 **`max > final` 的回落场景在真实 runtime 上
一次都没出现**。D3 修复所针对的那个具体故障（中途装上车又掉下来的 package 被
丢失）目前只有单元测试 `T11` / `T11b` / `T11c` 覆盖。

即：**聚合机制已验证，回落捕获未在 runtime 验证**。
不得把 S6 PASS 读成"已证明能抓住回落"。要补齐它，需要一个真能装上 package 的
truck 策略——本地没有（见 §7）。

## 5. VACUOUS 两项与科学裁决阻断

S3 / S4 都是 0/8 终止，条件式判据**真空成立**。真空成立不是验证。

```
scientific_use_blocked:
  h1hand-truck-v0             termination_semantics   0/8 终止，success 语义未执行
  h1hand-bookshelf_simple-v0  termination_semantics   0/8 终止，reason 0/1/2 映射未执行
```

S4 的结果与预注册 §6 **写在看到结果之前**的预期一致：
"预期本轮 S4 仍为 VACUOUS，因本地 bookshelf 只有 final checkpoint 且已学会不摔倒。"

`site_rules.require_runtime_verified(env, purpose=...)` 是这条阻断的强制点，
两个清单 `RUNTIME_VERIFIED_TERMINATION` / `_MILESTONE` 当前均为空集合
（fail-closed），填值见 §6。

### 5.1 为什么 termination 与 milestone 要分开记

ChatGPT 的意见是"阻止 bookshelf 被用于 milestone / site 科学裁决"。
实测表明这两条通路的证据强度不同，合并会损失信息：

```
truck       milestone 已验证（success_subtasks 提取到，1000 步）；termination 未验证
bookshelf   milestone 已验证（success / success_subtasks 均提取到）；termination 未验证
```

故实现里分成两个清单。**但对 bookshelf 需要一条额外的保留意见**：
它的成功事件（`task_index == 5`）与终止 reason 1 是同一事件，
既然终止从未发生，那么 milestone 的**高值区间**同样从未被观察到——
已验证的只是"低值区间的提取通路可用"。引用 bookshelf 的 milestone 时须知此界。

## 6. `RUNTIME_VERIFIED_*` 的建议填值（**本 commit 不填**）

smoke 给出的实测依据：

```json
"runtime_verified": {
  "termination_semantics": ["h1hand-basketball-v0", "h1hand-crawl-v0", "h1hand-slide-v0"],
  "milestone": ["h1hand-bookshelf_simple-v0", "h1hand-truck-v0"]
}
```

按预注册 §9，填值必须是**本结果之后的独立 commit**——在结果 commit 里改代码
正是上一轮的错误。此处只记录依据。

一条说明：crawl 的 `termination_semantics` 计为已验证，验证的是
"**未终止 → neutral**"这条路径被 8/8 执行。Crawl 恒不终止
（`basic_locomotion_envs.py:168 return False, {}`），这就是它语义的全部，
不存在未覆盖的终止分支。

## 7. 完整命令与产物

```bash
PY=/home/yjj/miniconda3/envs/FastTD3/bin/python
PYTHONPATH=. $PY -m pytest tests/test_evaluator_schema_v2.py -q
#   → 65 passed, 1 skipped（T4 需 EVAL_V2_INTEGRATION_CKPT）
PYTHONUNBUFFERED=1 PYTHONPATH=. $PY scripts/analysis/smoke_evaluator_v21b.py
#   → CORE_PATHS_VERIFIED_WITH_GAPS, exit 0
```

```
68f98b6802ff6e622e93afff352b73433c7e94027ad89718d35708023b21b073  scripts/p0_evaluator_v2.py
56a191d0eaca41fd1c2409d91806835328dcd86231020b4775d5ba7efd56df52  fasttd3_ptf/evaluation/site_rules.py
a92d216df9d7902f54be53b08b8ecf7506e0583b0aafc61280e0bb4dd0774628  fasttd3_ptf/evaluation/task_metrics.py
06719f491bed02372f8be07720c359f748e348c12e46570a7f100555c904b4b3  fasttd3_ptf/evaluation/schema_v2.py
f0328bd73c672d53a263bedbd4d28f048e23684b649052dfe23a3a8de8d33b29  scripts/analysis/smoke_evaluator_v21b.py
ced079adcd11a309a18bd4e53c7f1ac04e0b27162c1e391db0f609f2a987ea90  docs/data/evaluator_v21b_smoke/smoke.json
```

## 8. 已知限制

1. **回落捕获未在 runtime 验证**（§4）——D3 所修的具体故障只有单元测试覆盖，
   需要一个能真正装上 package 的 truck 策略才能补齐；
2. **S3 / S4 的终止语义未验证**（§5），两者已被 `scientific_use_blocked` 拦截；
3. **bookshelf 的 milestone 只验证了低值区间**（§5.1）；
4. **formal 模式与原子写只有单元测试覆盖**（T12 六条 / T13 四条）——
   smoke 用 debug 模式跑，因为 smoke 产物本就不得用于科学裁决；
   formal 的强制性需要构造缺字段的 checkpoint，那在单元测试里做更干净；
5. **smoke 每项仅 8 episodes**，只验证通路，数值不得用于任何科学判断；
6. **`schema_version` 升到 2.2 是破坏性变更**——旧的扁平 milestone 格式已移除。
   目前 `resolve_task_outcome` 全仓库只有一个调用点（已 grep 确认），
   但 P2 之后若有脚本读旧格式评估产物，须显式迁移；
7. **`training_commit` 多数仍为 UNKNOWN**，覆盖率待 P2 统计。
