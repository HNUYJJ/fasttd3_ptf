# Post-Diagnostic Protocol：Checkpoint Inventory v2.2

> 2026-08-07。**必须在任何实现之前提交 git。**
> 冻结后只允许改路径参数，不得修改 digest 定义、分类规则或判据。
>
> **本文件不是 blind pre-registration。** 它写于看过 P2.1 full scan 结果
> （263 组 AMBIGUOUS、2 组 integrity failure、37 个 unparseable）之后，
> 是 post-diagnostic protocol：能保证"实现不回头迁就结果"，
> 不能保证"判据独立于已见结果"（`CLAUDE.md §8.7`）。

## 0. 根因：跨文件名比较 raw SHA 本身无效

已独立复现（`CLAUDE.md §0` 要求逐条核实）：

```python
obj = {...}                                    # 同一个 Python 对象
torch.save(obj, "foo_13000.pt", _use_new_zipfile_serialization=True)
torch.save(obj, "foo_final.pt", _use_new_zipfile_serialization=True)
# zip 内 entry：foo_13000/data.pkl   vs   foo_final/data.pkl
# SHA256 必然不同；连等长文件名 same_a / same_b 也不同
```

`train_ptf.py:880` 确实用 `_use_new_zipfile_serialization=True`；
`:3733` 在训练循环外、无新 learner update 的情况下写 `_final.pt`。

**PyTorch 把文件名 stem 写进 zip 内部 entry 根目录**，因此
`_100000.pt` 与 `_final.pt` 的 SHA 不同**不能推出状态不同**。
P2.1 的 263/263 系统性模式由此完全解释。

顺带解释了 P2.1 的 `EXACT_ALIAS` 为何碰巧成立：正式路径与 archive A
**文件名相同**（只是目录不同），entry 名一致故 SHA 可比。那是运气不是设计。

---

## 1. 三层 digest（冻结）

| digest | 覆盖 | 用途 |
|---|---|---|
| `file_sha256` | 文件原始字节 | **物理文件身份**。只在**文件名相同**时可跨路径比较 |
| `evaluation_state_digest` | `actor_state_dict` + `obs_normalizer_state` | evaluator 实际消费的状态。它相同 ⇒ 评估结果必然相同 |
| `full_state_digest` | 加载后**全部**逻辑内容 | 完整状态身份（含 optimizer / qnet / 计数器等） |

### 1.1 计算规则（冻结）

**禁止重新 `torch.save` 后取 SHA**——那会再次引入文件名依赖，
正是本轮要修的问题本身。改为递归规范化编码：

```
tensor        b"T" + dtype 名 + shape + C-contiguous raw bytes
              （先 .detach().cpu()；非连续先 .contiguous()）
dict          b"D" + 逐 key 排序后 (编码(key), 编码(value))
list / tuple  b"L"/"P" + 按顺序编码元素
str/bytes     b"S"/b"B" + utf-8 字节
int/float/bool/None  各自类型标记 + repr
其他          b"O" + repr（兜底，保证可编码不崩）
```

每一项都带**类型标记**，故 `1` 与 `"1"`、`[1,2]` 与 `(1,2)` 不会碰撞。
浮点用 `repr` 保证往返精度。

## 2. FINAL 规则（冻结）

同一 `execution_instance` + 同一内部 `global_step` 下，
数字步文件与 `_final.pt` 的关系按**语义**判定，**不再比较跨文件名 raw SHA**：

```
eval digest 同 且 full digest 同   → FINAL_LOGICAL_ALIAS
                                     final 不计入 canonical
eval digest 同 且 full digest 不同 → EVAL_EQUIVALENT_STATE_DIVERGENCE
                                     正式评估**优先数字步文件**；记录 resume-state 差异；
                                     不是失败（evaluator 只消费 actor + obs norm）
eval digest 不同                   → FINAL_POLICY_DIVERGENCE
                                     **硬失败、非零退出**
```

无同步数字步文件时，`_final.pt` 可作为 endpoint canonical。

### 2.1 全量执行前必须先做诊断

**先对 ≥20 对真实 final-numbered pair 做 field 级与 digest 级诊断**，
覆盖 slide / racing / stair / truck / cabinet / P0 六族，产出：

```
每对的 file_sha256 是否不同（预期：全部不同，因 §0）
每对的 evaluation_state_digest 是否相同
每对的 full_state_digest 是否相同
若 full 不同，逐 top-level key 列出差异项
```

诊断结果决定三态的实际分布。**诊断先行，再全量。**

## 3. P0 alias 与 FINAL 完全分离（冻结）

P2.1 的误报源于把 `(execution_instance_id, global_step)` 当分组键，
于是 `main_13000` / `A_13000` / `main_final` / `A_final` 四个文件混进同一组。

v2.2 分组键加入 `file_role`：

```
file_role ∈ {NUMBERED, FINAL}          由文件名的 step 段决定
alias 分组键 = (execution_instance_id, global_step, file_role)

按 file_role 分别配对：
    main NUMBERED ↔ archive_A NUMBERED     要求 raw SHA 相同（文件名相同，可比）
    main FINAL    ↔ archive_A FINAL        要求 raw SHA 相同（同上）
禁止 NUMBERED 与 FINAL 互比 raw SHA。
```

archive B 始终 `REPEATABILITY_DUPLICATE`，与 A 的差异是**度量对象**，不报错。

## 4. file_kind 分类（冻结）—— 先分类，再解析

P2.1 把"文件名不符合 learner checkpoint 格式"本身当错误，
于是 source bank 的 `final.pt` 与 anchor bundle 的 `learner.pt` 都成了失败项。
**它们本就不共享 target-run 的命名契约。**

```
TARGET_POLICY_CHECKPOINT   匹配 {env}__{run}__{seed}_{step}.pt   → 执行全套 learner 规则
SOURCE_POLICY_ASSET        checkpoints/**/sources/**/*.pt        → 支持资产，不进 learner 统计
ANCHOR_BUNDLE_COMPONENT    文件名 ∈ {learner,replay,rng}.pt 或位于 anchors 目录
                                                                 → 优先核验 bundle manifest/checksum，
                                                                   **不反序列化巨大 replay**
UNKNOWN_PT_FILE            其余                                   → **非零退出**
```

只有 `TARGET_POLICY_CHECKPOINT` 执行文件名解析、身份核对、canonical 规则。

输出四个 universe：`physical_universe` / `policy_universe` /
`source_asset_universe` / `anchor_asset_universe`。

### 4.1 discovery roots 扩展

```
models/  checkpoints/  artifacts/
```

`artifacts/` 加入**物理资产发现**，但其内容按 §4 分类后
**不进入 policy checkpoint 统计**。

## 5. UNKNOWN role 不得默认计入 seed 计数（冻结）

P2.1 的 `counts_as_new_learner_replication` 在无 run card 时默认 `True`。
**那是 fail-open**：82% 的 `UNKNOWN_ROLE` 会被当成独立 learner 复制计入 n。

```
run card 明确声明          → 按声明（True / False）
无 run card / UNKNOWN_ROLE → UNKNOWN，**不可用于 seed 计数**
```

只有 experiment manifest / run card 明确的 replication 才进入正式 n。

## 6. Comparison contract（冻结）

P2.1 的 `pairing_invariant` 白名单只有 14 个字段，
而训练代码自己的 anchor-resume 检查用的是 `_RESUME_MATCH_KEYS`
（`train_ptf.py:2372`，**41 个键**），缺 `num_steps`、`policy_frequency`、
`learning_starts`、`use_cdq`、`obs_normalization`、learner architecture
（`*_hidden_dim` / `*_num_blocks`）、LR schedule（`*_learning_rate_end`）等。
**当前 digest 可能把并不真正 matched 的两臂判为 matched。**

v2.2：每个 `match_group` 在 registry 中**显式冻结**三组字段：

```yaml
base_invariant_fields:  默认 = train_ptf.py 的 _RESUME_MATCH_KEYS 全集
treatment_fields:       允许不同（该 match_group 的处理维度）
anchor_lineage_fields:  anchor_dir / anchor_step / anchor_provenance_groups /
                        anchor_resume / resume_noise_seed / run_stop_step
```

anchor-resume 实验**必须**核对 anchor bundle digest、`resume_noise_seed`、
`run_stop`/budget 与基础 FastTD3 超参。

### 6.1 `site_rules` 的比较等级

```
paired_by_seed   要求 match_group + comparison_contract_id + pairing_invariant_digest
across_methods   **只能标 DESCRIPTIVE**，不得进入因果 / 配对统计
```

`across_methods` 当前只要求 env/step/panel/schema，不要求 matched protocol——
故它产出的差值不具备配对效力，必须在输出中标明 `DESCRIPTIVE`。

## 7. Registry 扩充策略（冻结目标口径）

**不追求全库 100% 文件覆盖。** 先产出 unresolved family 频次表，
只扩充**即将重评**与 `PAPER_CLAIMS_20260804.md` 涉及的实验族。

```
目标：正式 re-evaluation shortlist 的 role / match_group coverage = 100%
非目标：物理文件覆盖 100%
```

大量 diagnostic、旧 probe、source asset 不必为降低 UNKNOWN 百分比而人工标注。

## 8. 退出码传播（冻结）

所有 tmux / 管道启动统一 `set -o pipefail` 并用 `rc=${PIPESTATUS[0]}`。
加一条**回归测试**：故意让管道首个命令失败，验证记录的退出码非零。

## 9. Sentinel 覆盖（冻结，规则先于运行）

| # | 覆盖点 | 期望 |
|---|---|---|
| 1 | 真实 final-numbered pair，状态相同 | `FINAL_LOGICAL_ALIAS` |
| 2 | **构造** final policy divergence（改 actor 权重） | `FINAL_POLICY_DIVERGENCE`，非零退出 |
| 3 | P0 main/A 的 NUMBERED 配对 | raw SHA 相同，`EXACT_ALIAS` |
| 4 | P0 main/A 的 FINAL 配对 | 同上，且**不与 NUMBERED 混组** |
| 5 | P0 archive B | `REPEATABILITY_DUPLICATE` |
| 6 | source asset | `SOURCE_POLICY_ASSET`，不进 learner 统计 |
| 7 | anchor component | `ANCHOR_BUNDLE_COMPONENT`，不反序列化 replay |
| 8 | **构造** unknown `.pt` | `UNKNOWN_PT_FILE`，非零退出 |
| 9 | 无 run card 的 checkpoint | `counts_as_new_learner_replication = UNKNOWN` |
| 10 | **构造** contract mismatch（`num_steps` 不同） | `pairing_invariant_digest` 不同，配对被拒 |

第 2、8、10 项为构造件，写临时目录，跑完即删。

## 10. 退出条件

```
FINAL_POLICY_DIVERGENCE            → 非零退出
UNKNOWN_PT_FILE                    → 非零退出
AMBIGUOUS_EXECUTION（真实无法区分）  → 非零退出
FORMAL_ALIAS_INTEGRITY_FAILURE     → 非零退出（现在按 file_role 分别判，不再误报）
任一 sentinel 未按预期分类           → 非零退出
```

`EVAL_EQUIVALENT_STATE_DIVERGENCE` **不是失败**——记录并优先用数字步文件。

**sentinel 全通过后重跑一次 full metadata scan；
只有 verdict PASS 且 `UNKNOWN_PT_FILE = 0` 才停下等 review。**

## 11. 设计层自查（CLAUDE.md §8）

**8.1 辨别力**——平凡解释："digest 相同只是因为都算成了空"。排除：§9 第 2 项
**构造** policy divergence（真改 actor 权重），必须报 `FINAL_POLICY_DIVERGENCE`；
若 digest 恒相同，该项会失败。第二个平凡解释："把 FINAL 一律判 alias 就转绿"。
排除：那样第 2 项同样失败。

**8.2 混淆变量**——`evaluation_state_digest` 只覆盖 actor + obs norm。
若 evaluator 将来消费更多状态（如 critic），该 digest 会**低估**差异。
故在实现中把"evaluator 实际读取的键"与 digest 覆盖的键写在同一处常量，
并加断言：`p0_evaluator_v2` 的 `load_student` 只用 actor 与 obs_normalizer。

**8.4 前提是否蕴含结论**——验算退出条件能否恒不触发。不能：第 2、8、10 项
注入的错误必然存在，检测器不报即非零退出。反向：检测器过敏会让第 1、3、4 项失败。

**8.6 是否重演本轮教训**——
"跨文件名比 raw SHA"→ §1 明确 `file_sha256` 只在文件名相同时可比；
"把不该解析的文件当错误"→ §4 先分类再解析；
"UNKNOWN 默认 True"→ §5 改 fail-closed；
"白名单太短"→ §6 以 `_RESUME_MATCH_KEYS` 全集为基础。

## 12. 提交结构

```
protocol   本文件                                    ← 先于实现
诊断       ≥20 对 pair 的 digest 诊断（§2.1）         ← 先于全量，产物属结果
实现       digest / file_kind / contract + 测试       ← 无运行产物
结果       sentinel + full scan + 报告                ← 不得含 .py/.sh
```

**本轮不运行任何环境 rollout 或训练。**
