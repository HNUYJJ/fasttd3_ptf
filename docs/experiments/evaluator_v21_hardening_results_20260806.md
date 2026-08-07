# 【已降级为 DIAGNOSTIC】Evaluator v2.1 Hardening 首轮

> **2026-08-07 降级声明（外部 review 指出，我核实成立）。本文件不再是一份合格的结果文档。**
>
> 产出它的 commit `19948c4`（publish `691dcff`）**名义是"结果提交"，实际同时修改了
> `scripts/analysis/smoke_evaluator_v21.py`（+34/−12）与 `scripts/p0_evaluator_v2.py`（+29）**。
> 即：basketball 修复、S2 更换 checkpoint、`VACUOUS` 判定，全都是**看到首轮运行结果之后**
> 在这个结果 commit 里加入的。因此本轮声称的
> "预注册 → 实现 → 纯结果"三段式**不成立**，实现与结果之间没有边界。
>
> 这比 S4 是否 VACUOUS 严重得多：三段式的全部意义就是"判据与实现先冻结、结果不能回头改它们"，
> 而在结果 commit 里改代码恰好取消了这个保证。**运行结果暴露实现 bug 时可以修，
> 但必须把原结果标成 diagnostic，另开 hotfix commit 重新冻结，再独立重跑。**
>
> **本文件的效力**：以下内容一律降级为 **diagnostic 记录**——
> 可用于说明"发现了什么 bug、为什么要改"，
> **不得**被引用为"evaluator v2.1 已通过验证"。`docs/data/evaluator_v21_smoke/smoke.json`
> 同样降级，其 `"verdict": "ALL_PASS"` 字段**作废**。
>
> 替代它的是 P1.1b：预注册 `evaluator_v21b_prereg_20260807.md`，
> 结果 `evaluator_v21b_results_20260807.md`。**引用请引 P1.1b，不要引本文件。**
>
> 不重写 git 历史——`19948c4` 原样保留，本声明是它的如实标注。

> 2026-08-06 原始记录。预注册 `evaluator_v21_hardening_prereg_20260806.md`（本地 `063c88a`），
> 实现 `bebdc5e`，smoke 脚本冻结 `ff3b73f`，结果 `19948c4`（**含代码改动，见上**）。
> 原始输出 `docs/data/evaluator_v21_smoke/smoke.json`，退出码 `0`。

## 1. 裁决【作废】

```
VERDICT: ALL_PASS      ← 作废：措辞过强，且产出它的 commit 违反三段式
  S1 crawl      PASS      0/8 终止，语义全 neutral
  S2 slide      PASS      5/8 终止，全部判为 failure
  S3 truck      PASS      milestone 通路可用
  S4 bookshelf  VACUOUS   0/8 终止 → 条件判定路径**未被真实验证**
  S5 basketball PASS      8/8 终止，全部判为 failure（修复后）
```

`ALL_PASS` 有两处问题：

1. **与 S4 的 VACUOUS 并存本身自相矛盾**。正确表述是
   "P1.1 核心路径基本可用，bookshelf runtime termination path 未验证"；
2. 它是在**违反三段式的 commit** 里产出的，无论内容对错都不具备验证效力。

本轮之后 `verdict` 字段不再允许取 `ALL_PASS`（见 P1.1b 预注册 §6）。

## 2. S5 抓到一个真实 bug（本轮最有价值的产出）

首轮 smoke：basketball **8/8 终止但全部 `INSUFFICIENT_STATE`**。
诊断发现 `_basketball_state` 的访问路径写错，且被 `except` 静默吞掉：

```python
# 错误实现
named = env.unwrapped._env.named.data     # HumanoidEnv 没有 _env 属性
except (AttributeError, KeyError): return None   # ← 静默吞掉，恒返回 None

# 实测 env.unwrapped 的类型与属性
unwrapped type: HumanoidEnv   has _env: False   has named: True

# 修复
named = env.unwrapped.named.data          # HumanoidEnv 直接持有 named
```

后果曾是：**basketball 的 `task_success` 永远是 `INSUFFICIENT_STATE`，
该任务实质不可评估**，而从输出上看只是数据不足，不像 bug。
预注册对 S5 明令无降级条款，正是为了逼出这类静默失败。

修复后：`ball_to_hoop_dist = 5.76`（reset 时球离篮筐 5.76 m），
8/8 终止 episode 全部正确判为 `failure`（球未进筐）。

**连带修正**：异常不再静默——失败原因写入 `_last_error` 并进 smoke 输出，
使提取失败与该任务不需要 state可区分。

## 3. S4 是 VACUOUS，不是 PASS

bookshelf 的判据是条件式的（**若**终止，则语义必须是 success/failure）。
8 个 episode 无一终止，该判据**真空成立**——但条件判定路径根本没被执行。

首版 smoke 会把这种情况报成 `PASS`，这会误导。已改为独立的 `VACUOUS` 状态：
不计入失败（不阻塞 P2），但也**不得计为已验证**。

缺口的具体内容：`_bookshelf_termination` 对 `terminated_reason` 0/1/2 的映射
**只在单元测试 T1d 中用 mock 验证过**，未经真实 runtime 验证。
本地 bookshelf 只有 9 个 final checkpoint、无早期 checkpoint，
而 final 策略已学会不摔倒，故当前数据条件下无法补齐。

## 4. S2 更换 checkpoint 的理由（必须说明，避免被读成挑数据）

```
首轮   h1hand_slide_tp_scr_s1_..._final.pt      0/8 终止 → 无法验证 failure 语义
改用   slide_bac_walk_s1__1_20000.pt            5/8 终止 → 可验证
```

final scratch 已学会不摔倒，而验证 failure 语义**必须有 terminated 的 episode**。
这与挑数据让测试通过的区别在于：判据没有变（仍要求终止 episode 的
`task_success=False` 且语义为 `failure`），变的是**能触发该代码路径的输入**——
如同测试异常处理必须先构造异常。5/8 这个比例与 T4 独立观察到的一致。

## 5. 恢复的两项安全能力

| 能力 | v1 出处 | v2 初版 | v2.1 |
|---|---|---|---|
| checkpoint 身份校验 | `p0_evaluator.py:152-175` | 无 | 已恢复，并在跑 episode **之前**执行 |
| 拒绝覆盖 | `p0_evaluator.py:228` | 无 | 已恢复，且提前到跑 episode 之前，避免白跑 |

新增 v1 没有的 `panel_digest`——不同评估面板产出的数字不可比，
而此前无法从输出中判断两份结果是否用了同一面板。

## 6. `require_comparable` 的四级分类

```
across_methods   env_name + global_step + panel_digest + schema_version
across_seeds     追加 method_family
paired_by_seed   追加 learner_seed
same_checkpoint  追加 learner_seed + checkpoint_sha256
```

`purpose` 为强制关键字参数、无默认值；跨 `env_name` 在所有 purpose 下拒绝；
字段缺失即 `IncomparableError`（身份不完整 ≠ 身份相同）。
旧的单字段签名**已移除**，不留兼容别名。

测试 T7b 显式验证 `crawl@100k` vs `hurdle@100k` 在**所有** purpose 下被拒——
这正是旧实现放行的情形。

## 7. 完整命令与产物

```bash
PY=/home/yjj/miniconda3/envs/FastTD3/bin/python
PYTHONPATH=. $PY -m pytest tests/test_evaluator_schema_v2.py -q     # 39 passed, 1 skipped
PYTHONPATH=. $PY scripts/analysis/smoke_evaluator_v21.py            # ALL_PASS, exit 0
```

```
73f30d383b8896fc2df7a1c8ec8e320f8ace01f390831e97fa0b28446e84d121  scripts/p0_evaluator_v2.py
9c6c817a5479e36c95ced4cdf69c528320cbe3481f1b0e551d259eaff1866983  fasttd3_ptf/evaluation/site_rules.py
6152181f85ff214c88bdb39c04d6b5dca2128970a046dc6909b989016a11be53  scripts/analysis/smoke_evaluator_v21.py
6ca4cacb7e18f61ecaa23ac59339b86f81dec8aed71da9fe538a831f38202745  docs/data/evaluator_v21_smoke/smoke.json
```

## 8. 已知限制

1. **S4 未真实验证**（VACUOUS）——bookshelf 条件终止只有 mock 测试覆盖；
2. **S3 未观察到 truck 成功**，按预注册降级条款以 milestone 通路可用通过；
3. **smoke 每项仅 8 episodes**，只验证通路，数值不得用于任何科学判断；
4. **身份校验依赖 checkpoint 内 `args`**，若某些旧 checkpoint 缺 `args["env_name"]`，
   强制核对会失败——P2 deep scan 时须统计这类文件；
5. **`training_commit` 多数为 UNKNOWN**——`args` 中未必记录 git 信息，
   实际覆盖率待 P2 统计。

## 9. 【2026-08-07 追加】本轮遗留的实现缺陷（外部 review 指出，逐条已核实）

这五条在写本文件时**没有被发现**，全部留到 P1.1b 修复：

| # | 缺陷 | 核实出处 |
|---|---|---|
| D1 | `identity_checked` 只要显式传**任意一个** `--expect-*` 即为 true，不要求 seed / global_step 等正式身份全部声明；也没有 formal / debug 两种模式 | `p0_evaluator_v2.py:262` `len(explicit) >= 1` |
| D2 | 输出仍是裸 `write_text()`，不是临时文件 + 原子 `rename`；中途失败会留下截断的 JSON | `p0_evaluator_v2.py:389` |
| D3 | milestone **只从最后一步 `info` 提取**，尽管 `info_history` 已经保存 | `_run_episode_v2` 传 `info=info`；`info_history` 只喂 `summarize_info` |
| D4 | S5 的 smoke 判据**没有实现预注册**。预注册要求"至少提取到一个有限 `ball_to_hoop_dist`"，实现只检查"若有终止 episode 则不全为 `INSUFFICIENT_STATE`"——更弱的代理，且 0 终止时真空通过 | `smoke_evaluator_v21.py:121-126` |
| D5 | `verdict` 取 `ALL_PASS` 与 S4 的 VACUOUS 并存 | 见 §1 |

**D3 的证据比"理论担忧"强**——HumanoidBench 源码里这些量真的会回落：

```
truck.py:113-115       for package in self.packages_on_table: ... .remove(package)
truck.py:199           reward_dict["success_subtasks"] = len(self.packages_on_table)
basketball.py:139      "success_subtasks": 1 if self.stage == "throw" else 0
basketball.py:140      "success": ball_hoop_distance < 0.05      ← 瞬时判定，球穿筐飞走即回 False
```

即：中途装上车又掉下来的 package、中途穿过篮筐的球，**只读最后一步全部丢失**。
故 milestone 必须沿 trajectory 聚合（first / max / final），这不是加固而是修 bug。

**S4 的正确后果**（本文件原先漏写）：bookshelf 的 runtime termination path 未验证，
因此 **bookshelf 不得进入任何 milestone / site 科学裁决**，
直到它被真实终止 episode 验证过。它不阻塞纯 checkpoint 元数据 inventory。
