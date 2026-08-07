# 预注册（blind）：Checkpoint Inventory v2.1 —— provenance / identity 修正

> 2026-08-07。**必须在任何实现之前提交 git。**
> 冻结后只允许改路径参数，不得修改身份定义、digest 语义、判据或退出条件。

## 0. 盲性声明

**本文件对 P2.1 的主终点是盲的**：新身份模型下的 alias 去重结果、
`FORMAL_ALIAS_INTEGRITY_FAILURE` 是否发生、真正的 `AMBIGUOUS` 数量、
run card 的角色命中率、full scan 的 v1↔v2 universe diff——这些我都还没看过。

**不盲的部分**（如实列出，来自上一轮 `e37403f`）：20 处理 / 16 ELIGIBLE /
2 组 `AMBIGUOUS_RUN_INSTANCE`（`p0_crawl_abstain#1` @10750 与 @11500）/
6 行 `fname_parsed=False` / racing 规则 0 命中 / `mechanism_family` 无 `NO_PTF`。
v2.1 要修的正是产生这些数字的模型本身，故它们不构成对新判据的污染。

---

## 1. 上一轮的错误定性（已更正）

`e37403f` 把 `models/p0_dup_archive/` 当成"来源不可考的重复归档"。**这是错的。**
逐条核实的证据（全部在 `scripts/` 与 `tests/`，我上轮只 grep 了 `docs/`）：

```
tests/test_p0_orchestrator.py:118   归档语义:A=第一次产物(不可变),B=第二次,
                                    正式路径恢复为 A 内容
tests/test_p0_orchestrator.py:146   assert archive_a/name == "FIRST"
tests/test_p0_orchestrator.py:147   assert archive_b/name == "SECOND"
tests/test_p0_orchestrator.py:148   # 正式路径 = A(第一次运行)。
scripts/p0_orchestrator.py:12       duplicate 语义:与对应 abstain 分支 CLI 逐位一致
scripts/p0_orchestrator.py:186      duplicate 度量=A vs B 的 primary eval
scripts/p0_orchestrator.py:244      第一次产物归档成功后才允许第二次启动
git log e03efdc                     2026-07-17「duplicate 归档定稿」
```

**真正的缺陷是 inventory 的 execution identity 模型**：
`run_instance_id = exp_name#seed` 无法表达"同一 CLI 的两次独立执行"。
数据没脏，模型不够。

---

## 2. 身份模型：三层（冻结）

```
run_family_id          = env_name + exp_name + learner_seed
                         「同一实验配置的同一 learner 复制」
execution_instance_id  = run_family_id + execution_role
                         「该 family 的第几次独立执行」
learner_replication_id = env_name + exp_name + learner_seed
                         「统计上的独立 learner 单位」——与 execution 分离
```

关键区分：**同一 `learner_replication_id` 可以有多个 `execution_instance_id`**
（重复性 duplicate），但它们**不计作多个独立 learner seed**。
上一轮把二者混为一谈，才把合法的 A/B 判成映射不唯一。

### 2.1 `execution_role`（冻结取值）

| 取值 | 含义 | 唯一来源 |
|---|---|---|
| `FORMAL` | 正式执行 | run card registry |
| `REPEATABILITY_DUPLICATE` | 独立重启的重复性副本 | run card registry |
| `UNKNOWN_EXECUTION_ROLE` | 无法由冻结证据确定 | 默认 |

### 2.2 alias 去重与完整性（冻结）

同一 `execution_instance_id` 下、同一 `global_step` 的多个文件：

```
SHA 全部相同        → EXACT_ALIAS，保留一条 canonical，其余记 alias_of
SHA 不同 且 能由冻结 external provenance 区分 execution
                    → 分属不同 execution_instance，各自独立，**不是** AMBIGUOUS
SHA 不同 且 **无法**由冻结证据区分
                    → AMBIGUOUS_EXECUTION（非零退出）
```

**P0 archive 的专门规则**（由 §1 的冻结协议确定）：

```
models/<run_name>_<step>.pt                    execution_role = FORMAL
models/p0_dup_archive/<task>_A/<run_name>_<step>.pt   同上 FORMAL —— 与正式路径同一 execution
models/p0_dup_archive/<task>_B/<run_name>_<step>.pt   REPEATABILITY_DUPLICATE

正式路径 SHA == A 的 SHA   → EXACT_ALIAS（预期）
正式路径 SHA != A 的 SHA   → FORMAL_ALIAS_INTEGRITY_FAILURE（非零退出）
                              协议要求正式路径恢复 A 的内容，不符即协议被破坏
B 与 A 的 SHA 不同          → **正常**，那正是重复性度量的对象，不得报 AMBIGUOUS
B 不计入 learner_replication 计数
```

---

## 3. 文件名解析（冻结）—— UNPARSED 必须 fail closed

**上一轮的真实 bug**：正则的 env 段写作 `[^_]+(?:-[^_]+)*`，吃不下含下划线的
env（`h1hand-balance_hard-v0`、`h1hand-bookshelf_simple-v0`）。
实测 20 个 sentinel 中 **6 个 `fname_parsed=False`，却全部 `eligibility=ELIGIBLE`、
`identity_conflicts=[]`** —— 解析失败被当成"身份核对通过"，是 fail-open。

修正：

```
正则改为   ^(?P<env>.+?)__(?P<run>.+)__(?P<seed>\d+)_(?P<step>\d+|final)\.pt$
           env 用非贪婪匹配到第一个 `__`；分隔符是双下划线，env 内只含单下划线
解析失败   → eligibility = EXCLUDED_UNPARSEABLE_NAME（**不是** ELIGIBLE）
           并计入 §7 的非零退出条件
```

**"无法核对" ≠ "核对通过"**——这条在 evaluator 的 formal 模式里已经立过，
inventory 这里漏了。

---

## 4. `effective_endpoint` 与 canonical 裁剪（冻结）

已核实 `train_ptf.py` 的原文语义：

```
:339   run_stop_step 独立控制训练退出——total_timesteps 保持不变以维持 LR 余弦日程
:2280  run_stop_step = int(ptf_cfg.get("run_stop_step") or args.total_timesteps)
:2286  if run_stop_step <= 0 or run_stop_step > args.total_timesteps: → 报错
```

故：

```
effective_endpoint = run_stop_step   若 ptf_cfg 显式设置（非 None）
                   = total_timesteps 否则

校验 0 < run_stop_step <= total_timesteps，不满足 → INVALID_ENDPOINT_CONFIG（非零退出）
```

**明令禁止 `min(total_timesteps, run_stop_step)`**：那会把非法配置静默"修好"，
而代码本身把它当错误。inventory 的职责是如实记录，不是替训练脚本纠错。

### 4.1 canonical 只保留 `step <= effective_endpoint`

上一轮的 bug：`total_timesteps=13` 的 diagnostic run 列出
canonical `[13, 30, 10000, 20000, 50000, 100000]`。

```
canonical_steps  = {固定点 10k/20k/50k/100k} ∪ {bootstrap_end} ∪ {effective_endpoint}
                   然后**只保留 <= effective_endpoint 的**
超出者           → 记入 out_of_scope_steps，不进 canonical
```

`bootstrap_end`（`ptf_cfg["mcg_warmup_steps"]`）若 > `effective_endpoint`，
同样进 `out_of_scope_steps`——该 run 根本没跑到 warmup 结束。

### 4.2 `completion_status`（run 层，冻结）

```
observed_end >= effective_endpoint  → COMPLETED
observed_end <  effective_endpoint  → TRUNCATED_RUN
effective_endpoint 不可确定          → UNKNOWN_COMPLETION
```

---

## 5. Digest 语义拆分（冻结）

上一轮把 `sha256(ptf_cfg)` 直接当 `training_protocol_digest`，并要求
`paired_by_seed` 时相同。**这会拒绝所有真实的 matched comparison**——
scratch / continuous / hard-exit 的 `ptf_cfg` 本来就必须不同，那正是 treatment。

拆成四项：

| 字段 | 定义 | 配对时 |
|---|---|---|
| `ptf_cfg_digest` | `sha256(整个 ptf_cfg)` | 仅作身份，**不要求**相同 |
| `treatment_digest` | `sha256(ptf_cfg 中不在 pairing-invariant 白名单的字段 + source_names)` | **允许不同**（这就是处理） |
| `pairing_invariant_digest` | `sha256(白名单字段)` | **必须相同** |
| `match_group` | 来自 run card registry | **必须相同** |

### 5.1 pairing-invariant 白名单（冻结）

只含"两臂之间本就应当一致的 nuisance 配置"：

```
args:     total_timesteps, num_envs, batch_size, buffer_size, gamma, tau,
          actor_learning_rate, critic_learning_rate, policy_noise, noise_clip,
          num_updates
ptf_cfg:  anchor_dir, anchor_step, anchor_provenance_groups
```

缺失字段记 `MISSING`（参与 digest，使"缺"与"有"可区分）。
run card registry 可**逐 match_group 覆盖**该白名单；覆盖须在 registry 中
写明理由与 evidence。

### 5.2 `source_bank_digest`

source bank 有冻结的 checkpoint / config hash 时记录之，否则 `UNKNOWN`。
不得用 bank 文件路径代替 hash。

### 5.3 `site_rules.paired_by_seed` 同步修改

```
旧：COMPARISON_BASE | {learner_seed, match_group, training_protocol_digest}
新：COMPARISON_BASE | {learner_seed, match_group, pairing_invariant_digest}
```

`training_protocol_digest` 从要求集合中**移除**（它是 treatment，本就该不同）。

---

## 6. Run card registry（冻结其来源规则）

`docs/data/run_cards/run_card_registry_v1.json`，schema `run_card_registry.v1`。

**唯一合法来源**：冻结的 run card / orchestrator / training script / experiment
manifest。每条 entry **必须**带：

```yaml
match:              run_name 的匹配规则（正则），或显式路径前缀
experiment_role:    实验臂角色
execution_role:     FORMAL / REPEATABILITY_DUPLICATE
match_group:        配对组
expected_endpoint:  该臂的预期 endpoint（可为 null）
canonical_points:   该协议的判决点（可为空）
evidence_path:      证据文件路径
evidence_line:      证据行号
source_commit:      该证据所在的 commit
mapping_rule:       从证据到本条映射的推理，一句话
```

**禁止仅根据文件名词义猜角色**（如见到 `scr` 就判 scratch）。
无法由冻结证据证明的一律 `UNKNOWN_ROLE`。

至少覆盖三组（已核实证据存在）：

```
P0      scripts/p0_orchestrator.py:177  for arm in ("lease", "abstain")
        run_name = {env}__p0_{task}_{arm}__{seed}   （:80 _run_name）
        duplicate A/B 见 §2.2
Slide   scripts/run_slide_hard_exit_v1.sh:18  for arm in prefix cont exit
        name = shev1_${arm}_s${SEED}
Racing  scripts/run_racing_min_horizon_v1.sh:11  ARM: student|run|walk|stand
        NAME = ${EXP_PREFIX}_${ARM}_s${SEED}
```

---

## 7. Full scan 的数据宇宙（冻结）

上一轮 `--full` 用 `rows_v1` 过滤——**v1 漏掉或错误排除的文件，v2 永远发现不了**。

```
v2.1 独立扫描冻结的 filesystem roots：
    models/**/*.pt            （含 p0_dup_archive 等子目录）
    checkpoints/**/*.pt
v1 manifest 只作**差异对照**，不作数据来源。
输出 universe_diff：v1_only / v2_only / common 三张表及计数。
```

## 8. 退出条件（冻结）

```
任一 sentinel 未按预期分类                      → 非零退出
FORMAL_ALIAS_INTEGRITY_FAILURE                 → 非零退出
AMBIGUOUS_EXECUTION（无法由冻结证据区分）        → 非零退出
EXCLUDED_UNPARSEABLE_NAME 出现                  → 非零退出（fail-closed）
INVALID_ENDPOINT_CONFIG                        → 非零退出
任一字段被静默猜值                              → 非零退出
```

## 9. Sentinel 覆盖（冻结，选取规则先于运行）

| # | 覆盖点 | 期望 |
|---|---|---|
| 1 | P0 正式路径 + archive A（同 step） | `EXACT_ALIAS`，**不报** AMBIGUOUS |
| 2 | P0 archive B（同 step） | `REPEATABILITY_DUPLICATE`，不计新 learner replication |
| 3 | 构造：同 family 同 step 不同 SHA、且无冻结证据可区分 | `AMBIGUOUS_EXECUTION` |
| 4 | underscore env（`balance_hard` / `bookshelf_simple`） | 解析成功，env 正确 |
| 5 | 构造：不可解析文件名 | `EXCLUDED_UNPARSEABLE_NAME`（**不是** ELIGIBLE） |
| 6 | `run_stop=30k, total=100k, observed=30k` | `COMPLETED` |
| 7 | `observed < run_stop` | `TRUNCATED_RUN` |
| 8 | `effective_endpoint < bootstrap_end` | bootstrap_end 进 `out_of_scope_steps`，不进 canonical |
| 9 | ≥2 个 Racing 角色由 registry 命中 | `experiment_role != UNKNOWN_ROLE` |
| 10 | 两个人为身份冲突（改错 seed / 改错 step） | 2/2 `EXCLUDED_IDENTITY_MISMATCH` |

第 3、5、10 项为**构造件**，写临时目录，跑完即删，不污染 `models/`。
第 6、7、8 项若真实数据中不存在，用构造的 metadata（不落盘 checkpoint）验证判定函数，
并在结果中标明是构造用例。

## 10. 设计层自查（CLAUDE.md §8）

**8.1 辨别力**——平凡解释一："sentinel 通过只是因为解析器没崩"。排除：
第 3、5、10 项是**注入的错误**，不崩不算通过，必须**报对应的错**。
平凡解释二："把 AMBIGUOUS 判据放宽就能转绿"。排除：第 3 项要求
**真正无证据可区分的冲突仍必须 AMBIGUOUS**——放宽会让它失败。

**8.2 混淆变量**——`execution_role` 由 registry 给出，而 registry 由我编写，
存在"为了让 sentinel 过而写 registry"的风险。约束：registry 每条必须带
`evidence_path` + `evidence_line` + `source_commit`，且证据 commit 必须**早于**
本预注册。无法满足者一律 `UNKNOWN_ROLE`。

**8.4 前提是否蕴含结论**——验算退出条件能否恒不触发。不能：
第 3、5、10 项注入的错误必然存在，检测器不报即非零退出。
反向：检测器过度敏感会让第 1、4、6 项失败。两侧可达。

**8.5 site selection**——sentinel 按规则选，规则先冻结；结论只覆盖被检文件。
full scan 阶段的结论覆盖冻结的 filesystem roots 全体。

**8.6 是否重演本轮教训**——
"只 grep docs 就下结论"→ §6 要求每条 registry entry 附 `evidence_path`，
逼迫我指出具体文件与行号；
"UNPARSED 当成无冲突"→ §3 明确 fail-closed 并进退出条件；
"结果 commit 改代码"→ §11 写死提交结构。

## 11. 提交结构

```
预注册    本文件                                  ← 先于实现
registry  run_card_registry_v1.json + 校验器      ← 属实现，先于运行
实现      inventory v2.1 + site_rules 修改 + 测试  ← 无运行产物
结果      sentinel 输出 + 结论文档                 ← git show --stat 不得有 .py/.sh/.json(registry)
full      全量 metadata 扫描 + universe diff       ← 仅当上一步全通过
```

**若结果暴露 bug**：按 `CLAUDE.md §4.1` 走——标 DIAGNOSTIC → hotfix →
重新冻结 → 独立重跑。不得在结果 commit 里修。
