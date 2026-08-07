# 【DIAGNOSTIC_FULL_SCAN_FAILED】Inventory v2.1 全量 metadata deep scan

> ## 2026-08-07 定性：本次全量扫描降级为 DIAGNOSTIC，**不是最终 inventory**
>
> 三类失败**全部来自 inventory 口径问题，没有一条证明训练数据损坏**。
> 尤其两点必须在引用前读到：
>
> **① 2 组 `FORMAL_ALIAS_INTEGRITY_FAILURE` 是实现误报，不得引用为 P0 协议失败。**
> 逐对比对同名文件 4/4 全部一致（见 §2）。P0 duplicate 协议完好。
>
> **② 263 组 `AMBIGUOUS_EXECUTION` 的根因已查明：跨文件名比较 raw SHA 本身无效。**
> PyTorch 的 zip 序列化把**文件名 stem 写进 zip 内部 entry 根目录**——
> 我已独立复现：同一个 Python 对象保存成 `foo_13000.pt` 与 `foo_final.pt`，
> zip 内 entry 分别是 `foo_13000/data.pkl` 与 `foo_final/data.pkl`，
> SHA256 必然不同；连等长文件名（`same_a` / `same_b`）也不同。
> `train_ptf.py:880` 确实用 `_use_new_zipfile_serialization=True`。
>
> 因此 `_100000.pt` 与 `_final.pt` 的 SHA 不同**不能推出状态不同**，
> 263/263 的系统性模式由此完全解释。**raw file SHA 只是物理文件身份，
> 不是 checkpoint 状态身份。**
>
> 顺带解释了我的 `EXACT_ALIAS` 判定为何碰巧成立：正式路径与 archive A
> **文件名相同**（只是目录不同），zip entry 名一致，故 SHA 可比。
> 这是运气，不是设计。
>
> 修正见 P2.2（`checkpoint_inventory_v22_protocol_20260807.md`）：
> 引入 `evaluation_state_digest` 与 `full_state_digest`，
> 按 tensor dtype/shape/raw bytes 递归计算，**不经 torch.save**。
>
> 原始输出 `full.json` 保留不改。§5 的有效产出仍可引用，
> 但须同时声明本降级声明。


> 2026-08-07。blind prereg `checkpoint_inventory_v21_prereg_20260807.md`（`af3cacb`）。
> **冻结实现 `42c1d67`**，零代码改动。无环境 rollout（纯元数据）。
> 原始输出 `docs/data/checkpoint_inventory_v21/full.json`。
>
> 执行门（目标第 8 项）：新版 sentinel `SENTINEL_PASS`（`0c1e002`）+
> formal smoke 通过（`e0cdd44`）—— 二者均已满足。
>
> **按目标要求，扫完到此停下等 review，不进入 P3A 或批量 evaluator。**

## 1. 裁决

```
VERDICT: FULL_SCAN_FAILED                       退出码 1
  独立发现 universe   1587 个 .pt（roots = models, checkpoints）
  处理 1587 | ELIGIBLE 1016 | 角色已解析 254 | 身份冲突 0

  ! 2 组   FORMAL_ALIAS_INTEGRITY_FAILURE
  ! 263 组 AMBIGUOUS_EXECUTION
  ! 37 个  EXCLUDED_UNPARSEABLE_NAME
```

**三类失败的性质完全不同，必须分开读。** 其中两类是我这一侧的问题
（一个误报、一个预注册遗漏），只有第三类是判据按设计工作。

## 2. 2 组 `FORMAL_ALIAS_INTEGRITY_FAILURE` 是**误报**，P0 协议完好

这条报警的字面含义是"正式路径与 archive A 的内容不一致 → duplicate 协议被破坏"。
**那是错的。** 逐对比对同名文件：

```
crawl  _13000.pt   models/… f2038024ceaf852c  ==  archive_A/… f2038024ceaf852c   ✓
crawl  _final.pt   models/… 5d99c20e984f8184  ==  archive_A/… 5d99c20e984f8184   ✓
truck  _13000.pt   models/… 2ebff6974d5f998e  ==  archive_A/… 2ebff6974d5f998e   ✓
truck  _final.pt   models/… 118316e1865ddb4d  ==  archive_A/… 118316e1865ddb4d   ✓
```

**4/4 逐对一致。协议没有被破坏，正式路径确实是 A 的内容。**

`n_sha=2` 的真实来源是同一 run 内 `_13000.pt` 与 `_final.pt` **本身内容不同**：

```
crawl  f2038024ceaf852c (_13000)  vs  5d99c20e984f8184 (_final)
truck  2ebff6974d5f998e (_13000)  vs  118316e1865ddb4d (_final)
```

我的 `resolve_aliases()` 按 `(execution_instance_id, global_step)` 分组，
把这两类文件放进了同一组（内部 `global_step` 都是 13000），
于是组内出现 2 个 SHA；又因组内含 `alias_of_formal_path=True` 的成员，
被判成 integrity failure。**两个正交维度（文件角色 vs 归档路径）被混在一次判定里。**

> 这条如果被照字面引用，会变成对 P0 duplicate 协议的错误指控——
> 与上一轮我把 archive 说成"来源不可考"是同一类错误的镜像。故单列澄清。

## 3. 263 组 `AMBIGUOUS_EXECUTION` 全部来自**预注册未规定 FINAL 语义**

```
263 组中：final-vs-数字步 pattern = 263；其他 = 0
```

**263/263，零例外**。典型形态：

```
xxx__1_100000.pt   内部 global_step = 100000
xxx__1_final.pt    内部 global_step = 100000，但 SHA 不同
```

即 `_final.pt` 与同 step 的数字步文件内容不同。这在本仓库是**系统性的**，
不是个例——涉及 slide(27) / rad(24) / stage(18) / door(15) / stair(15) /
cabinet(12) / e2e(12) / p0(12) 等几乎所有实验族。

### 3.1 根因：v2.1 预注册漏了继承 v2 的 FINAL 规则

v2 的预注册（`bf14f20`）§3.1 有明确规定：

```
内部 global_step 与某数字步文件相同 且 SHA 相同  → FINAL_DUPLICATE_OF_<step>，不计 canonical
内部 global_step 与某数字步文件相同 但 SHA 不同  → 非零退出
```

**v2.1 的 blind prereg（`af3cacb`）没有把这条继承下来**——全文只在文件名正则里
出现过 `final` 一词，没有规定 FINAL 文件的去重与冲突语义。实现随之也没有。

于是所有 `_final.pt` 都被当成普通文件参与 alias 判定，
与同 step 的数字步文件撞在一起。

### 3.2 我**没有**在本轮补这条规则

补上它 263 组立刻消失、full scan 转绿。**我没有这样做**：
预注册没规定的东西，不能在看到结果之后补进本轮
（`CLAUDE.md §8.7`，outcome-contingent gate switching）。
即使规则本身正确、即使 v2 里写过、即使如实披露。

正确做法是新写一份预注册，明确 FINAL 的语义，然后重跑。
**该规则应当怎么写，是 review 的决定**——尤其是这一条：

> `_final.pt` 与同 step 的数字步文件**内容不同**在本仓库是常态。
> 需要先弄清它是保存时机差异（optimizer/replay 状态在最后一次保存时又变了）
> 还是别的原因，再决定"同 step 不同 SHA"该判 duplicate 还是 ambiguous。
> 在弄清之前，两种判法都是猜。

## 4. 37 个 `EXCLUDED_UNPARSEABLE_NAME` 是判据按设计工作

```
final     14   （checkpoints/formal_*/sources/<name>/final.pt 这类）
learner    7   ┐
replay     7   ├ artifacts 风格的 anchor bundle 组件
rng        7   ┘
其他       2   （h1hand-crawl-v0__p0eq_… 前缀）
```

这些**结构上就不匹配** `{env}__{run}__{seed}_{step}.pt` 命名规则——
它们是 source bank 的 final 权重与 anchor bundle 的组件，本来就不是
"某 run 某 seed 某步的策略 checkpoint"。

fail-closed 排除它们是**正确的**（v2 曾把这类文件 `fname_parsed=False` 却仍标
`ELIGIBLE`，是 fail-open）。但预注册 §8 把
"`EXCLUDED_UNPARSEABLE_NAME` 出现 → 非零退出"写成了硬失败，
故它们把整个 full scan 拖成 FAILED。

**判据是我自己冻结的，本轮如实执行。** 建议下一版区分两种情形：
"命名错误的 checkpoint"（应硬失败）与"结构上不适用该命名规则的非 checkpoint 文件"
（应归入一个显式的排除类别，如 `NOT_A_POLICY_CHECKPOINT`）。

## 5. 有效产出（这部分不受上述三类影响）

```
universe               1587 个 .pt（独立发现，不依赖 v1 manifest）
ELIGIBLE               1016
角色已解析              254   （registry 覆盖 P0 / Slide / Racing 三组）
身份冲突                0     （文件名 ↔ checkpoint 内部）

experiment_role         UNKNOWN_ROLE 1296 | ABSTAIN 50 | RACING_ARM_{run,stand,
                        student,walk} 各 36 | LEASE 30 | …
execution_role          UNKNOWN 1296 | FORMAL 244 | REPEATABILITY_DUPLICATE 10
mechanism_family        PTF_WITH_SOURCES 1167 | PTF_NULL_BANK 361 | NO_PTF 22
run_completion          648 个 execution：COMPLETED 498 / TRUNCATED_RUN 148 /
                        UNKNOWN_COMPLETION 2
```

**`REPEATABILITY_DUPLICATE = 10`**——P0 duplicate 的 B 组被正确识别，
且**不计入独立 learner replication**。这是本轮身份模型修正的直接产出。

**`UNKNOWN_ROLE = 1296 / 1587 = 82%`**：registry 目前只覆盖三组已核实证据的实验。
这是正确的 fail-closed（无冻结证据不猜角色），但也说明
**inventory 现在还不能回答"哪些是 treatment arms"这个问题的大部分**。

## 6. 一个方法学缺陷：我的 tmux + tee 日志掩盖了退出码

本次 full scan 的日志末尾记 `EXIT=0`，而实际裁决是 `FULL_SCAN_FAILED`。
原因是我的启动命令写作 `... | tee $LOG; echo EXIT=$?` ——
`$?` 取的是 **`tee` 的退出码**，不是 python 的。

**影响范围**：本轮所有用 tmux + tee 跑的任务（v21b / v21c smoke、formal smoke）
的日志 `EXIT=` 行都不可信。所幸它们的 `verdict` 字段独立记录在 JSON 里、
且都是通过态，故**结论不受影响**；但方法有缺陷，须改用 `${PIPESTATUS[0]}`。
本文件所报的退出码一律以 `verdict` / `failures` 字段为准，不以日志行为准。

## 7. 完整命令与产物

```bash
PY=/home/yjj/miniconda3/envs/FastTD3/bin/python

# 门（均已通过）
PYTHONPATH=. $PY scripts/analysis/build_checkpoint_inventory_v21.py     # SENTINEL_PASS, 0
EVAL_V2_INTEGRATION_CKPT=models/h1hand-slide-v0__slide_bac_walk_s1__1_20000.pt \
EVAL_V2_INTEGRATION_ENV=h1hand-slide-v0 \
  PYTHONPATH=. $PY -m pytest tests/test_evaluator_schema_v2.py -q -k T4  # 1 passed, 0
PYTHONPATH=. $PY scripts/p0_evaluator_v2.py --checkpoint … \
  --identity-manifest docs/data/formal_pipeline_smoke/identity_manifest.json … # 0

# 本次
PYTHONPATH=. $PY scripts/analysis/build_checkpoint_inventory_v21.py \
  --full --approved-by "PI 目标第 8 项"                                 # FULL_SCAN_FAILED, 1
```

```
1724ee41df6e8143df2016ec6c313b1fa810c28e9f5c3675e386dc680049cc12  docs/data/checkpoint_inventory_v21/full.json
162e3d0e476fa923acf3353071ea2bb7ec3cc720b5c0f3159b5d5ff9e79b5bea  docs/data/checkpoint_inventory_v21/sentinel.json
617b575a8110bebad14bd152ee8166079605fe1c3b005c012cf8028776fa8622  docs/data/formal_pipeline_smoke/pipeline_smoke_eval.json
68ad7a80d13ce29dcdd91166885044f70ac21fc5b885e0fa615ce0c6991c7c79  docs/data/formal_pipeline_smoke/identity_manifest.json
f638050bc38b65ba7e94808c69dca4a44f3969e587e49eb6d88c1a4f0e35c474  docs/data/run_cards/run_card_registry_v1.json
```

## 8. 需要 review 决定的问题

1. **FINAL 语义怎么定？**（§3）——先弄清 `_final.pt` 与同 step 数字步文件
   内容不同的**原因**，再决定判 duplicate 还是 ambiguous。在弄清之前两种判法都是猜。
2. **`resolve_aliases` 的分组键**（§2）——是否应先按"文件角色"（数字步 / final）
   分层，再判归档 alias？当前把两个正交维度混在一次判定里，产生了误报。
3. **`EXCLUDED_UNPARSEABLE_NAME` 是否应区分两类**（§4）——
   "命名错误的 checkpoint" vs "结构上不适用该命名规则的非 checkpoint 文件"。
4. **`SCAN_ROOTS` 是否加入 `artifacts/`**（sentinel 结果 §5）——
   109 个 anchor bundle 文件当前未被发现。
5. **run card registry 是否继续扩充？**——当前 82% `UNKNOWN_ROLE`。

## 9. Known limitations

1. **本次 verdict 是 FAILED，故全量结果不得作为"已完成的 inventory"引用**。
   §5 的有效产出可以引用，但须同时声明 §2–§4 的三类问题。
2. **`SCAN_ROOTS` 漏 `artifacts/`**，109 个文件未被发现（判据已冻结，未改）。
3. **`UNKNOWN_ROLE` 82%**——inventory 目前不能回答大部分"哪些是 treatment arms"。
4. **`pairing_invariant` 白名单未经任何 run card 复核**，做配对分析前须逐组确认。
5. **`expected_endpoint` 在 registry 中多为 null**，只有 slide prefix 有值。
6. **无环境 rollout**：本次是纯元数据扫描，不涉及任何 episode 评估。
7. **日志 `EXIT=` 行不可信**（§6），一律以 JSON 的 `verdict` / `failures` 为准。
