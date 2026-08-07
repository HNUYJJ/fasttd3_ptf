# 结果：Checkpoint Inventory v2.1 Sentinel —— `SENTINEL_PASS`（退出码 0）

> 2026-08-07。blind prereg `checkpoint_inventory_v21_prereg_20260807.md`（`af3cacb`）。
> **冻结实现 `42c1d67`**（首轮 `b3cea7e` 暴露缺口 → 标 DIAGNOSTIC `7252acb` →
> hotfix `42c1d67` → 本次独立重跑）。期间零代码改动。
> 原始输出 `docs/data/checkpoint_inventory_v21/sentinel.json`，退出码 `0`。

## 1. 裁决

```
VERDICT: SENTINEL_PASS                          退出码 0
  独立发现 filesystem universe   1587 个 .pt
  处理 19 | ELIGIBLE 19 | 角色已解析 10 | 不可解析 0
  注入件 5 个：全部按预期分类
  真实数据的 AMBIGUOUS_EXECUTION：0
  FORMAL_ALIAS_INTEGRITY_FAILURE：0
  SENTINEL_UNAVAILABLE：无
```

## 2. 上一轮的错误定性被正面推翻

上一轮把 `p0_dup_archive` 判成"来源不可考的重复归档"、报 2 组
`AMBIGUOUS_RUN_INSTANCE`。按冻结协议实现 execution identity 之后：

```
alias_notes:
  h1hand-crawl-v0|p0_crawl_abstain|1|FORMAL@13000
      "2 个文件 SHA 相同，已 alias 去重"
```

即**正式路径与 archive A 的 SHA 逐字节相同**——这正是
`p0_orchestrator` 协议要求的"正式路径恢复为 A 内容"。
`FORMAL_ALIAS_INTEGRITY_FAILURE` 为 0，协议未被破坏。
archive B 落入独立的 `execution_instance`，与 A 不冲突。

**数据一直是干净的；上一轮的失败是身份模型表达不了"同一 CLI 的两次执行"。**

## 3. 检测能力先被证明，"0 冲突"才成为证据

真实数据报 0 个 `AMBIGUOUS_EXECUTION`。这本身无信息量——
可能是数据干净，也可能是检测器坏了。故预注册 §9 第 3 项要求**注入**一个
真正无法区分的执行歧义：

```
构造   复制 donor 一份；再用 torch 重存一份带额外标记的
       （SHA 不同，但 args / global_step 原样保留，故身份三项仍一致，
        不会被 EXCLUDED_IDENTITY_MISMATCH 提前拦掉）
实测   injected_exec_conflict_a → AMBIGUOUS_EXECUTION
       injected_exec_conflict_b → AMBIGUOUS_EXECUTION
       reason: "同一 execution 内出现不同 SHA，且无冻结证据可区分"
```

五个注入件全部按预期分类：

| 注入件 | 期望 | 实测 |
|---|---|---|
| 改错 seed | `EXCLUDED_IDENTITY_MISMATCH` | ✓ |
| 改错 global_step | `EXCLUDED_IDENTITY_MISMATCH` | ✓ |
| 不可解析文件名 | `EXCLUDED_UNPARSEABLE_NAME` | ✓ |
| execution 歧义 a | `AMBIGUOUS_EXECUTION` | ✓ |
| execution 歧义 b | `AMBIGUOUS_EXECUTION` | ✓ |

### 3.1 首轮曾漏掉这一项，我主动降级了自己的 PASS

首轮（`b3cea7e`）同样报 `SENTINEL_PASS`，但 `make_injected()` 只造了三件，
**第 3 项从未被执行**——`ambiguous_executions` 为空是因为没有输入能触发它，
不是数据干净。那是真空成立，与 evaluator 的 VACUOUS 同类。
处置见 `7252acb`（标 DIAGNOSTIC，原始输出保留为
`sentinel_diagnostic_round1.json`）→ `42c1d67`（纯代码 hotfix）→ 本次重跑。

## 4. 角色解析：10/19，且全部有冻结证据

```
ABSTAIN          FORMAL                   p0_lease_oracle_crawl
RACING_ARM_run   FORMAL                   racing_min_horizon_v1_s1
PREFIX           FORMAL                   slide_hard_exit_v1_s1
CONTINUOUS       FORMAL                   slide_hard_exit_v1_s1
HARD_EXIT        FORMAL                   slide_hard_exit_v1_s1
```

`slide` 三臂同 `match_group`（脚本 `:52`：Both continuations consume this
immutable bundle），故 cont 与 exit 可配对。

registry 9 条 entry 由 `validate_run_card_registry.py` 逐条校验：
`evidence_excerpt` 与该行**实际内容**比对一致，`source_commit` 均早于预注册。
行号写错会被抓出——registry 不是"我说了算"。

**上一轮 racing `SENTINEL_UNAVAILABLE` 0/2 是我的规则错误，不是数据缺失**：
实际前缀是 `rck`（96 个）/ `rad`（96 个），我用了 `racing` 这个词去匹配。

## 5. Universe diff：双向都有发现

```
v1_total 1661 | v2_total 1587 | common 1552
v1_only  109  | v2_only   35
```

**`v2_only = 35`——v1 真的漏了**，全部是 source bank 的 checkpoint：

```
checkpoints/formal_20260515T055812Z/sources/h1hand_{reach,run,stand,walk}/final.pt
checkpoints/formal_20260515T055812Z/targets/h1hand_push_{fasttd3_scratch,ptf}/final.pt
```

这正是"以 v1 manifest 为数据宇宙就永远发现不了"的实证——
上一轮 `--full` 正是那样做的。

**`v1_only = 109`——是我的 `SCAN_ROOTS` 缺陷，不是文件消失**：
它们全在 `artifacts/` 下（anchor bundle 的 `learner.pt` / `replay.pt` / `rng.pt`）。
预注册 §7 把 roots 冻结为 `(models, checkpoints)`，漏了 `artifacts/`。
v1 曾扫到并把它们标为 `ANCHOR_BUNDLE` 排除（99 个）。
**判据已冻结，本轮如实实现未改**；修正建议：下一版 roots 加入 `artifacts/`
并保留 `ANCHOR_BUNDLE` 分类（它们是 anchor bundle 而非策略 checkpoint，
本就不应进入评估集合，但应当被**发现并显式排除**，而不是根本没扫到）。

## 6. 完整命令与产物

```bash
PY=/home/yjj/miniconda3/envs/FastTD3/bin/python
$PY tests/test_inventory_identity_v21.py
#   → 38 项全部通过, exit 0
PYTHONPATH=. $PY scripts/analysis/validate_run_card_registry.py
#   → 9 条 entry 全部通过, exit 0
PYTHONPATH=. $PY scripts/analysis/build_checkpoint_inventory_v21.py
#   → SENTINEL_PASS, exit 0
```

```
19f25b88671b4f8007873bb07fe81ebeeb3011f37db858dba774b0f7b12bd3aa  scripts/analysis/build_checkpoint_inventory_v21.py
1ffafcaf934e22262f3bc16e277916f85bf2a89f7e1579c625c7a2efd77668f0  fasttd3_ptf/evaluation/inventory_identity.py
f638050bc38b65ba7e94808c69dca4a44f3969e587e49eb6d88c1a4f0e35c474  docs/data/run_cards/run_card_registry_v1.json
162e3d0e476fa923acf3353071ea2bb7ec3cc720b5c0f3159b5d5ff9e79b5bea  docs/data/checkpoint_inventory_v21/sentinel.json
```

## 7. 预注册 §9 十项覆盖对照

| # | 要求 | 状态 |
|---|---|---|
| 1 | P0 正式路径 + archive A → `EXACT_ALIAS`，不报 AMBIGUOUS | ✓ 实测 |
| 2 | archive B → `REPEATABILITY_DUPLICATE`，不计新 replication | ✓ 实测 |
| 3 | 构造真未知冲突 → `AMBIGUOUS_EXECUTION` | ✓ 注入件（首轮漏，已补） |
| 4 | underscore env 解析成功 | ✓ 实测 2 个 |
| 5 | 不可解析文件名 → `EXCLUDED_UNPARSEABLE_NAME` | ✓ 注入件 |
| 6 | `run_stop=30k,total=100k,observed=30k` → `COMPLETED` | ✓ 单元测试 T4（构造用例） |
| 7 | `observed < run_stop` → `TRUNCATED_RUN` | ✓ 单元测试 T4（构造用例） |
| 8 | `endpoint < bootstrap_end` → bootstrap_end 不进 canonical | ✓ 单元测试 T3（构造用例） |
| 9 | ≥2 个 Racing 角色由 registry 命中 | ✓ 实测 4 个（rck×2 + rad×2） |
| 10 | 两个人为身份冲突 2/2 被抓 | ✓ 注入件 |

第 6、7、8 项按预注册 §9 末句以构造的 metadata 验证判定函数，**标明是构造用例**——
真实数据中恰好没有 `run_stop_step` 且 `observed == run_stop` 的 sentinel 命中。

## 8. 已知限制

1. **`SCAN_ROOTS` 漏了 `artifacts/`**（§5），109 个文件未被发现。
   判据已冻结故本轮未改，修正留给下一版。
2. **本轮 sentinel 只覆盖 19 个文件**，不得外推到全部 1587 个；
   全量结论待 `--full` 阶段。
3. **第 6/7/8 项是构造用例**，不是真实 checkpoint 上的观察。
4. **`experiment_role` 覆盖率 10/19**——registry 只覆盖已核实证据的三组
   （P0 / Slide / Racing）。其余仍 `UNKNOWN_ROLE`，这是正确的 fail-closed。
5. **`pairing_invariant` 白名单是我按预注册 §5.1 冻结的默认集**，
   尚未有任何 run card 覆盖它。真正做配对分析前应逐 match_group 复核。
6. **registry 的 `expected_endpoint` 多为 null**——只有 slide prefix 有
   （30000，来自脚本注释）。其余需要读各实验的启动参数才能填。
