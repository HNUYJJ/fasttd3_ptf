# 结果：Evaluator v2.1 Hardening —— `ALL_PASS`（S4 为 VACUOUS）

> 2026-08-06。预注册 `evaluator_v21_hardening_prereg_20260806.md`（`3261cb4`），
> 实现 `f78ddb2`，smoke 脚本先于运行冻结。
> 原始输出 `docs/data/evaluator_v21_smoke/smoke.json`，退出码 `0`。

## 1. 裁决

```
VERDICT: ALL_PASS      单元测试 39 passed + smoke 5 项无失败
  S1 crawl      PASS      0/8 终止，语义全 neutral
  S2 slide      PASS      5/8 终止，全部判为 failure
  S3 truck      PASS      milestone 通路可用
  S4 bookshelf  VACUOUS   0/8 终止 → 条件判定路径**未被真实验证**
  S5 basketball PASS      8/8 终止，全部判为 failure（修复后）
```

按预注册 §7，S1–S5 无失败项，**可进入 P2**。但 S4 的 VACUOUS 必须并列声明。

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
