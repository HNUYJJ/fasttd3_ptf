# 预注册：RACING_REJECT v2 —— racing 能否**提前**做出正确的拒绝决策

> 2026-07-30。v1 已作废（`da32b13`，作废时未揭盲）。
> 本版按 Codex 两轮 review 重做，`APPROVE_WITH_FIXES` 的 6 项修复全部写入本文。
> **提交本文时，我仍未查看任何 U / return / CI 数值。**

## 1. v1 为什么死（一句话）

`sanity(K=10000 时 9/9 全负)` ⇒ `∀seed: max_i U_i(10000) < 0` ⇒ `K*_reject ≤ 10000` 必然成立
⇒ `REJECT_REFUTED` 不可达。**主假设不可证伪。**

## 2. 主终点（可证伪）

racing 的实用输出是一个决策：

```
decide(K) = argmax_i U_i(K)      若 max_i U_i(K) > 0     （使用该源）
          = REJECT               否则                     （不使用任何源）
```

door 的参照决策是 `REJECT`（gate 已发表：三源 K=10000 上 9/9 per-seed 全负）。

```
H：存在 K ∈ {2000, 5000}，使 decide(K) = REJECT 在 批1 3/3 且 批2 3/3 上成立
```

**K=10000 不参与主终点**（v1 正是死在这里），只作复制检查（§5）。

| 结果 | 裁决 |
|---|---|
| K=5000 两批各 3/3（K=2000 是否通过不影响） | `EARLY_REJECT_CONFIRMED` |
| 仅 K=2000 两批各 3/3，K=5000 失败 | `EARLY_REJECT_NONMONOTONIC` —— **不得称"稳定安全网"**，须报为方向随 horizon 非单调 |
| K=2000 与 K=5000 均未达标，K=10000 决策为 REJECT | `EARLY_REJECT_REFUTED` —— 有意义的负结果：**拒绝与选源代价同量级，不存在便宜的安全网** |
| K=10000 的决策也不是 REJECT | `PARTICIPANT_DIVERGED`（§5.3） |

**非单调条款的理由**：`RACING_K` 已实证早期信号可方向错误（hurdle 上 K=2000 时 run 系统性排最后，
5.3–14.3 个 episode-SE，3/3 一致）。允许"K=2000 或 5000 任一通过即确认"会挑到孤立通过点，
故把 `EARLY_REJECT_CONFIRMED` 锚定在 K=5000。

## 3. 独立重复：批2 使用**全新 seeds**（Codex 修复 1）

`RACING_K` 的"两批"实为同 seeds `{1,2,3}` 仅改 `EXP_PREFIX`，差异来自 CUDA 非确定性
（E10 / E15：同进程顺序两次运行亦非逐位确定，逐位等价只在 CPU 可达）。
它是有效的**重跑稳定性**证据，但**不是 learner 总体的独立抽样**。

| | seeds | anchor | 数据状态 |
|---|---|---|---|
| 批 1 | 1, 2, 3 | `artifacts/door_at10k_gate_v1/anchors/s{1,2,3}`（现成） | v1 已产出，**未揭盲** |
| 批 2 | **4, 5, 6** | **需新建**（同协议：10k exact-abstention 纯 student） | 待跑 |

批2 与批1 的 seed 不相交，构成 6 个独立 learner。

**批1 复用的合法性**：v2 判据在查看任何 U 数值之前冻结（本文提交时间早于任何揭盲），
故判据选择未受数据影响。结果文档须原样记录这一事实。

## 4. 错误率：如实降级，不夸大（Codex 修复 2）

主判据用点估计 `max_i U_i(K) < 0`，与 racing 的实用形式一致。其偶然通过率**取决于零假设的写法**：

| 零假设 | 单 learner | 6 learner | 说明 |
|---|---|---|---|
| 尖锐可交换零（4 臂联合可交换、无并列） | 1/4 | `(1/4)⁶ = 1/4096 ≈ 0.024%` | **示意值**：需要四臂可交换，而 stand/walk/run 的训练动力学未必同分布 |
| 复合零 `H₀: max_i E[U_i] ≥ 0`，边界情形（一源恰为零） | ≈ 1/2 | `(1/2)⁶ = 1/64 ≈ 1.56%` | **与"错误拒绝一个无害源"最相关** |

再计入 K ∈ {2000,5000} 的 look-elsewhere（最多 ×2）：

> **保守上界：约 1/32 ≈ 3.1%**。本实验报告该值，不使用 0.024%。

同时**必须报告但不参与裁决**：每批每源的 learner 层
`mean ± t₀.₉₅,₂ · SE_learner`（两侧 90%，`t₀.₉₅,₂ = 2.920`；**每批 n=3 分别报告**，
不合并 df）（Codex 修复 5）。据此读者可自行区分"点估计负"与"统计显著负"。

## 5. 三层验收（Codex 修复 3）

### 5.1 层 1 · 工程硬检查（任一不过 → `VOID_ENGINEERING`，不输出任何主结果）

- 源臂 behavior share ∈ `[0.48, 0.52]`（**必须由裁决脚本读 checkpoint 校验**，v1 缺此项）
- 源臂 `source_names` == 臂名；student 臂 `source_names == ['null']` 且无 `admission_audit`
  （防 stand↔run 臂对调——v1 的 mean 比对检测不到，两源已发表 mean 仅差 2.01）
- 全部臂日志含 `Resumed core learner ... at step 10000`
- 每个 eval json：`episode_count == 128`、`identity_checked == True`、
  `checkpoint.global_step` 匹配、`checkpoint.path` 含正确臂名与 seed、sha256 两两不同

### 5.2 层 2 · 复制检查（**仅批1**，与 gate 已发表值比对）

`p0_evaluator` 的面板逐位配对（"(seed, rank) → 唯一 reset seed；面板冻结，分支间逐位相同"），
故 **paired SE 取逐 episode 差值序列的 SE**，不用 `sqrt(se₁²+se₂²)`（v1 之误）。

```
对每个源 i、每个 seed s ∈ {1,2,3}:
    U_new  = U_i,s(K=10000)  本次
    U_gate = gate 已发表 per-seed 值
    paired_se = SE( [r_source,e − r_student,e]_{e=1..128} )      本次，逐 episode 配对
    逐 seed 通过 ⟺  sign(U_new) == sign(U_gate)  且  |U_new − U_gate| ≤ 3 × paired_se
9/9 全通过 → REPLICATION_OK；否则 REPLICATION_DIVERGED
```

**局限（预先声明）**：gate 只发表了 per-seed 点值，未发表其 paired SE，
故容差只用本次的 paired SE，是近似；且 paired SE 是**评估噪声**，
不含 run-to-run 训练漂移（E15），故本检查对 CUDA 漂移的误杀率无法预先标定。
`REPLICATION_DIVERGED` **不自动等同于实现 bug**。

### 5.3 层 3 · 批2 的批内参照（无外部 ground truth）

批2 的 seeds 4–6 没有已发表 ground truth，故**不做外部复制检查**。
改用批内自洽：批2 的 `decide(K=10000)` 必须为 `REJECT`。
若否 → `PARTICIPANT_DIVERGED`，含义是 **door 的"三源全负"结论不推广到新 learner**——
这本身是一个值得单独报告的发现，但**不得**据此裁决主终点。

### 5.4 优先级（冻结）

```
VOID_ENGINEERING  >  REPLICATION_DIVERGED(批1)  >  PARTICIPANT_DIVERGED(批2)  >  主终点裁决
```

任一前置失败时，裁决脚本**不得输出、不得写入、不得打印**任何 K≤5000 的
per-seed U、排序或命中数（v1 违反此项）。

## 6. 盲态封闭（Codex 修复 4）

**在两批的训练、评估、层1 工程检查全部完成之前，不得读取或输出任一批的
U、return、CI、排序、命中数。** 允许接触的仅限无结果字段：
`global_step` / `seed` / `sha256` / `identity_checked` / `episode_count` /
`source_names` / `behavior share` / 日志中的 anchor 恢复行。

> 已知的暴露路径：eval json 内含 `episodes[].return`。因此结构校验脚本
> **只允许读取上列字段**，禁止聚合或打印任何 return。

## 7. 协议（冻结）

```
target        h1hand-door-v0
批1 seeds     1,2,3   anchor = artifacts/door_at10k_gate_v1/anchors/s{1,2,3}（现成）
批2 seeds     4,5,6   anchor = 新建，协议与批1 anchor 逐项相同（10k exact-abstention 纯 student）
臂            student / stand / walk / run   （四臂配对同 seed）
noise 重采样  PTF_RESUME_NOISE_SEED = 91000 + seed
剂量          behavior 0.5 / replay 0.5，h=25，bootstrap_only
K 取值        2000, 5000, 10000  → checkpoint 于 global_step 12000 / 15000 / 20000
评估          source-free student, deterministic, 128 episodes
其余          TOTAL_TIMESTEPS=100000 + PTF_RUN_STOP_STEP=20000, NUM_ENVS=128, ...
```

与 `run_door_at10k_gate_v1.sh` 的差异：`PTF_EVAL_CHECKPOINT_STEPS`、
`EXP_NAME`、`PROJECT`、日志目录。**不称"唯一改动"**——`RACING_K` 已报告在仅剩
`exp_name/project/eval_checkpoint/run_stop` 差异时仍存在未解释的 U 水平差异
（§6 of `racing_min_horizon_v1_results`），故如实列出全部差异。

## 8. 能与不能声称（Codex 修复 5 的措辞约束）

**通过后能声称的最强措辞**：

> 在 door、固定源集合 `{stand, walk, run}`、t=10k、50% dose、6 个独立 learner seeds、
> 128-episode source-free 面板上，直接测量 U 的 racing 在 K=5000 时给出的
> **点决策**与 K=10000 的参照决策一致（均为 REJECT）。
> 这是"在一个已知全负案例上提前拒绝可行"的证据。

**不得声称**：

1. 不得称"统计显著地证明所有源有害"——主判据是点决策一致性；
   door 的 ground truth 自身对 walk 也只给出 `uncertain`（90% CI 跨 0）。
2. 不得称"通用负迁移免疫机制"——单 target、单源集合、且该场地是**已知全负**后选定的
   （outcome-informed site selection）。
3. 不得与 hurdle 的加速收益合并核算（避损 vs 加速，口径不同）。
4. `EARLY_REJECT_NONMONOTONIC` 时不得称"稳定安全网"。

## 9. 不得做的事

- 裁决后不得调 K 取值、阈值、seed 数或容差抢救结论。
- 违反 §6 盲态封闭即整体作废。
- 前置检查失败时不得只保留通过的部分（v1 违反过）。
- 若 `EARLY_REJECT_REFUTED`，不得改用代理量补救——那会退回十一族。

---

# 修订 v2.1（2026-07-30，仍未揭盲）

> Codex 实现后 review 判定"实现未忠实通过冻结预注册，裁决脚本不能用于正式揭盲"。
> 我核实其指控成立，据此修订。**修订时仍未查看任何 U / return 数值。**
> 主终点、阈值、seeds、协议、成本核算**均未改动**；改的是判据文本的歧义与工程验收的严密性。

## R1. 零点消歧（唯一的实质判据澄清）

冻结文本内部不一致：§2 主定义为 `decide = argmax_i U_i 若 max_i U_i > 0，否则 REJECT`
（即 `REJECT ⟺ max_i U_i ≤ 0`），而 §4 与首版脚本写作 `< 0`。

**以 §2 的主定义为准：`REJECT ⟺ max_i U_i(K) ≤ 0`。**
U 为连续量，恰好等于 0 的概率为零，实践上无差别；此处消歧只为杜绝
"看到零附近结果后再选择"的可能。

## R2. §6 盲态封闭改为可执行的表述

原文"结构校验脚本只允许读取上列字段"在工程上不可实现——`json.loads` 必然把整份
文件（含 `episodes[].return`）读入内存。首版脚本因此**技术性违反**了 §6。

**修订后的封闭条件**（约束的是信息去向，不是内存驻留）：

1. 层1 的**输出**（stdout、`results.json`、报错信息）不得包含任何 return 派生量；
2. 层1 的**控制流**不得依赖任何 return 派生量；
3. **人类在层1 通过前不得查看**：单臂评估日志
   `docs/data/racing_reject_*/source_free_eval/*.log`（内含 `p0_evaluator` 打印的
   `return_mean`，属揭盲材料）、任何 eval json 的 `episodes`/`aggregate` 字段。

> 已核实：批1（door v1）的单臂评估 `.log` 从未被查看，只查看过 driver 汇总日志
> （仅含 "eval ... DONE" 行）。该泄露路径此前未被封堵，现予明文列出。

## R3. 层1 必须补齐的验收项（首版脚本缺失，Codex 逐条指出）

- **训练日志**含 `Resumed core learner ... at step 10000`（§7 已要求，首版未实现）
- `source_names` **精确等于** `[arm, "null"]`（首版只查 `names[0]`，
  故 `["stand","walk","null"]` 也能通过 stand 臂）
- eval json 的 `checkpoint.sha256` **等于**当前 glob 命中的 checkpoint 实际 sha256
  （首版只查 json 内自报路径子串，旧评估 + 同名新 checkpoint 可静默通过）
- `identity_checked is True`（首版用 truthiness，字符串 `"yes"` 也过）
- `len(episodes) == 128`（首版只信 `aggregate.episode_count`）
- episode `seed` 序列等于冻结面板（16 eval seeds × 8 ranks，128 个互不相同）
- 协议字段：`critic_sample_counts` share ∈ 同带、`protocol.deterministic is True`、
  `protocol.source_free` 非空、`env_name == "h1hand-door-v0"`
- `execution_counts` 长度为 2、元素非负、和大于 0
- 全部 return **有限**（拒绝 `NaN` / `Inf`）

## R4. 异常分类（首版未覆盖）

```
缺产物（checkpoint / eval json / 训练日志缺失）        -> INCOMPLETE，exit 2
产物存在但无效（json 损坏、torch.load 失败、字段缺失、
                类型错误、非有限值、面板不符、sha 不符）  -> VOID_ENGINEERING，exit 2
缺产物 与 缺陷 同时存在                                -> INCOMPLETE 优先
```

最后一条与 `CLAUDE.md` §4"数据不全必须 INCOMPLETE"对齐（首版在混合情形下判 VOID，冲突）。
所有 present-but-invalid 异常必须被捕获并归入 `VOID_ENGINEERING`，
不得以未捕获异常终止（首版的 `json.loads` / `torch.load` / 配对断言均会裸抛）。

## R5. 输出防陈旧（首版可把上一次的结果当本次裁决）

`results.json` 必须：运行开始时**先删除**旧文件；写入用临时文件 + 原子替换；
内容包含本次 `run_id` 与全部输入 checkpoint 的 sha256 摘要。
否则一次成功裁决后若再次运行并中途崩溃，磁盘上的旧 `per_K` 会被误读为本次输出。

## R6. 不变项（明确声明，防止被误认为事后调整）

**未改动**：主终点（K=5000 锚定）、`EARLY_REJECT_*` 三分支、批1/批2 seeds、
K 取值 {2000,5000,10000}、剂量带 `[0.48,0.52]`、复制检查的 `3×paired_se`、
`t₀.₉₅,₂ = 2.919986`、偶然通过率上界 ≈1/32、§8 的措辞约束、§9 的禁止事项。

## R2'. 盲态封闭的精确边界（2026-07-30 二次修订，仍未揭盲）

Codex 二轮 review 指出：层1 的有限性检查 `math.isfinite(e["return"])` **按 R2 字面
构成"控制流依赖 return"**。此处澄清边界（与 R2 修订 §6 同理——把工程上不可满足的
绝对表述改为可执行且仍严格的表述）：

> **允许**：对 return 的**有效性判定**（类型、有限性）。它只提取 1 bit（有效/无效），
> 不泄露数值大小、排序或聚合。
> **禁止**：控制流依赖 return 的**数值大小、排序、聚合或与阈值的比较**。

**不做有限性检查的后果**：`NaN` 的一切比较恒为 `False`，会污染 `max_i U_i ≤ 0`
与 `max()` 的结果，导致**静默改判**——这正是 R4 要杜绝的"无效产物静默通过"。
故该检查是必需的验收项，不是盲态漏洞。

## R7. 未预期异常一律归入 VOID_ENGINEERING（二次修订）

静态自查（AST 扫描）发现 12 处可能裸抛且不在 `try` 内的 IO/解析调用
（`open` / `read_text` / `glob` / `unlink` / `replace` / `stdev` 等）。
按 R4"所有 present-but-invalid 必须捕获，不得裸抛"，`main()` 增加最外层兜底：
任何未预期异常 → 写出仅含 `verdict` / `unexpected_exception` / `note` 的
`results.json` 并 `exit 2`，**不输出任何主结果**。

已实测：注入 `RuntimeError` 后正确输出 `VOID_ENGINEERING`，`results.json` 仅 3 个键，
`per_K` / `layer2` / `layer3` 零泄漏。

另删除死代码 `class Absent`（0 处 `raise`、1 处 `except`，分支永不可达）。

## R8. 层1 的 `source_names` 期望值写错（揭盲时发现并修正）

首轮裁决输出 `VOID_ENGINEERING`，54 条缺陷全部是
`source_names=['stand'](应为['stand','null'])` 一类。**这是我的验收项写错，不是数据问题。**

根因：door 的 bank 配置是 `null_option: false` → `source_names=['stand']`；
hurdle 的是 `null_option: true` → `['run','null']`。R3 写死 `[arm,"null"]`
是把 hurdle 的模式套到了 door 上。

**更难看的是**：我在数小时前的剂量验收里**自己打印过** `names=['stand']`，
写 R3 时却没有回看。另外，Codex 建议此项时引用了 `source_bank.py` 的保存逻辑，
我核实了那段代码，却没核实**它在 door 场景下的实际输出**——
核实必须到"本场景实际值"这一层。

修正为与 `null_option` 无关的等价判据：`[n for n in names if n != "null"] == [arm]`。
它仍精确防臂对调（Codex 提该项的本意），且对两种 bank 配置都正确。
**主终点未变**，此项属层1 实现错误的修正，不是判据放宽。

## 揭盲结果：`REPLICATION_DIVERGED`（主终点不予裁决）

层1 全通过（24 臂 × 3 K：剂量 / 臂身份 / 面板 / sha256 / 协议）。层2 复制检查 6/9 FAIL：

```
符号：9/9 与 gate 一致（全负）—— door "三源全负" 的方向复现
数值：|本次 − gate| 中位 24.23，最大 43.78；而容差中位仅 11.31
```

**根因是容差量纲错配（我的设计错误）**：容差用 `3×paired_se`，即**评估噪声**（±3–5），
却要去卡 **run-to-run 训练漂移**（实测 ±3.3–43.8）。这与 `M16` 同类——
用 episode 尺度代替 learner 尺度。该失效模式已在 §5.2 预先声明
（"paired SE 是评估噪声，不含 run-to-run 训练漂移（E15）…
`REPLICATION_DIVERGED` 不自动等同于实现 bug"），故按预注册如实裁决，**不调容差抢救**。

**独立佐证漂移属正常量级**：`RACING_K` 在 hurdle 上同 seed 两批的 K=10000 漂移为
`−12.4 / −20.1 / −13.0`（约 15），与 door 此处的中位 24 同量级。

按 §5.4 优先级，主终点与层3 均不执行，脚本未输出任何 `K≤5000` 结果（已验证）。
