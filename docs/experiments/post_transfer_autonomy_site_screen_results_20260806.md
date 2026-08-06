# 结果：迁移后自主性判决场普查 —— `INSUFFICIENT_SITES`（测量不足，非方向证伪）

> 2026-08-06。预注册 `post_transfer_autonomy_site_screen_prereg_20260806.md`（提交 `3070c78`）。
> 脚本 `scripts/analysis/screen_post_transfer_sites.py`，原始输出
> `docs/data/post_transfer_site_screen_v1/screen.json`，退出码 `2`（INCOMPLETE）。
>
> **本文件记录的是修正后的第二次运行。** 首次运行的脚本有两个实现缺陷（见 §6），
> 判据本身未改动。

## 1. 裁决

```
CANDIDATE = 0   →  INSUFFICIENT_SITES
UNKNOWN   = 24
NO_SCAFFOLD = 1 （crawl）
SATURATED = 0   （修正后：milestone 未测，按预注册不得判定饱和）
```

按预注册 §5，`CANDIDATE < 2` 时停止，**不得降低门槛凑数**。
故本轮不产生 development / holdout 划分，autonomy 机制**不获批准实现**。

### 1.1 这个结论的确切含义

```
成立      在当前测量条件下，无法识别出合格场地
不成立    HumanoidBench 内不存在合格场地
```

24 个 `UNKNOWN` **全部**源于数据缺失，无一源于判据模糊。
这是 **measurement verdict，不是 scientific refutation**。补齐测量后须重新裁决。

## 2. 有数据的六个 target

`success_bar` 与 `max_episode_steps` 由 `ast` 沿类继承链解析自上游 HumanoidBench：

| target | success_bar | J_best_known | H_op | % of theory max | 判决 |
|---|---:|---:|---:|---:|---|
| crawl | 700 | 984.9 | −284.9 | **98.5%** | `NO_SCAFFOLD` |
| slide | 700 | 951.5 | −251.5 | **95.1%** | `UNKNOWN`（G3/G4 缺） |
| hurdle | 700 | 851.1 | −151.1 | **85.1%** | `UNKNOWN`（G3/G4 缺） |
| stair | 700 | 67.0 | **+633.0** | **6.7%** | `UNKNOWN`（G1/G3/G4 缺） |
| cabinet | 2500 | 100.6 | **+2399.4** | NOT_AUDITED | `UNKNOWN`（G1/G3/G4 缺） |
| door | — | — | — | — | 已排除（M31） |

`J_best_known` 取该 target 上**任何** source-free 评估的最大值（预注册 §2.1），
故 hurdle 取 `851.1`（s2 端到端 C 臂）而非常引的 3-seed 均值 `840.4`。

### 2.1 `success_bar` 不是性能上限——两个口径必须分开

`success_bar` 是上游 benchmark 设定的**成功阈值**，超过它只意味着
`ABOVE_SUCCESS_THRESHOLD`，不等于饱和。真正约束展示空间的是**理论回报上限**：

```
Walk / ClimbingUpwards / Hurdle 的 reward 均为 [0,1] tolerance 项相乘、无稀疏加项
（已逐个读 get_reward 全文核实，出处记录在脚本的 REWARD_STRUCTURE_AUDITED）
max_episode_steps = 1000  →  理论 episode 上限 = 1000

crawl  984.9 / 1000 = 98.5%   几乎到理论天花板
slide  951.5 / 1000 = 95.1%
hurdle 851.1 / 1000 = 85.1%   名义剩 15%，但需同时满足全程满速 + 零控制代价 + 零墙碰撞
```

**这个论证不依赖 `success_bar`**，只依赖 reward 结构与 episode 长度。
三者都不适合承担"后期巨大上限突破"的最终结果，但理由是**理论回报结构**，
不是"超过了 700"。

`cabinet` 等含稀疏成功项的任务标 `NOT_AUDITED`——其 reward 结构未逐条核实，
按预注册 §2.1 不估算理论上限（M33：不得凭"看起来像"推断）。

## 3. crawl 的正确表述（双重，缺一不可）

首版本文件把 crawl 的负迁移归因改写成"任务已解决，故任何源注入必然是净损失"。
**这个表述过强，已更正。**"最终没有 headroom"不蕴含"迁移必然无价值"——
源仍可能更早达阈、降低方差、提高有限预算 AUC。

crawl 上这些可能性被**实测数据**排除，而不是被 headroom 论证排除：

```
早期（racing K=10000）   三源 9/9 全负，且全部显著
                        s1  stand −44.06   walk −258.77   run −79.17
                        s2  stand −186.76  walk −196.22   run −181.86
                        s3  stand −162.98  walk −60.31    run −70.35
                        出处 docs/data/racing_admission_v1/results.json

终点（100k）             盲目用源 809.2  vs  scratch 960.2   （−151.0）
                        出处 docs/data/endtoend_v1/results.json
```

**早期与终点同时为负**，故正确表述是：

> crawl 已能被 scratch 高质量解决（98.5% of theory max），因此**不适合**
> 展示 final-ceiling improvement；但在匹配预算下现有 source bank 仍造成了
> 可复现的负迁移（早期 9/9 显著负、终点 −151），因此它**依然是**
> source mismatch 与安全退让的有效反例。

**不得**把 source mismatch 的解释删除或替换成 headroom 解释。

## 4. 22 个零数据 target 的成因

**没有一个是判据模糊，全部是数据缺失。**

| 缺口 | 影响 | 补齐需要 |
|---|---|---|
| 无任何 source-free 评估 | truck / bookshelf(2) / package / maze / basketball / balance(2) / highbar(2) / pole / powerlift / push / room / insert / spoon / cube / sit(2) | **不必新训练**，见 §5 |
| 有数据缺 G1 | cabinet / stair | 源臂 vs scratch 的 0–30k 配对 |
| **全体缺 G4** | 25/25 | evaluator schema v2 |

### 4.1 G4 的缺口是 evaluator 的功能缺陷，不只是"少采了字段"

```python
# scripts/p0_evaluator.py:93
if terminated:
    success = True          # → terminated_success → aggregate.success_count
```

而 `Walk` 系的 `get_terminated` 是**摔倒**（`torso_upright < 0.1` 或头部过低）。
因此当前 evaluator 在全部 locomotion 任务上**把摔倒记为 success**。
文件头注释（`p0_evaluator.py:17`）自陈该语义仅对 truck 成立——
这与 CLAUDE.md §6 记录的陷阱是同一件事，但影响比"少采 milestone"更严重。

## 5. checkpoint 清点：不需要新训练即可补齐大部分 UNKNOWN

本地已有大量 checkpoint，此前未被计入普查（普查只扫 `docs/data/**/source_free_eval/`）：

```
truck 132    basketball 105    cabinet 95    stair 69
powerlift 58    package 15    bookshelf 9        （*.pt 计数）
```

因此**不应**为补 UNKNOWN 而新跑 100k scratch——先用升级后的 evaluator
重评已有 checkpoint，榨干其信息价值，只对确实缺失的少数任务补训练。

## 6. 首次运行的两个实现缺陷（已修，判据未动）

自查发现脚本未正确实现自己冻结的预注册：

1. **`SATURATED` 误判**：预注册 §4 要求 `H_op ≤ 0` **且** `H_ms ≤ 0`，
   脚本只看 `H_op` 就判定。而 `H_ms` 全为 `None`（milestone 从未采集），
   **数据缺失落进了实质裁决分支**，同时违反 CLAUDE.md §4 与预注册本身。
   修正后 `H_op ≤ 0` 但 milestone 未测时判 `UNKNOWN_G2`。
   后果：`SATURATED` 3 → 0。
2. **`H_raw` 未实现**：预注册 §2.1 定义了理论上限口径，脚本没算。补上后
   才有了 §2.1 中独立于 `success_bar` 的论证。

两处都是**实现与已冻结判据不符**，修正方向是让实现服从预注册，未修改任何判据。
首次运行的输出保留在 scratchpad 以供对比；主结论 `INSUFFICIENT_SITES` 不变。

## 7. 本普查**不能**声称的

1. 不能说明任何源对任何 target 是否有用；
2. 不能说明 hard exit 是否足够；
3. 不能说明 autonomy 机制是否有效；
4. `success_bar` 与 `max_episode_steps` 均为上游 HumanoidBench 设定，非本项目发现；
5. `INSUFFICIENT_SITES` 是**测量不足**的裁决，不得写成"该方向不可行"。

## 8. 下一步：路线 B（evaluator schema v2），不启动新训练

```
P1  evaluator schema v2：区分 terminated / truncated / task_success /
    success_subtasks / termination_reason；通用保存标量 info 的
    mean/max/final/ever_positive/nonzero_fraction；
    保证旧 return_mean 与 progress_max_dx 在同 checkpoint 上逐位兼容。
P2  清点并重评本地已有 checkpoint（stair / truck / bookshelf / cabinet /
    powerlift / basketball / package），不新跑 100k scratch。
P3  重跑普查（v2 预注册，届时才可引入新的状态分类）。
P4  仅当 v2 产出 CANDIDATE，才批准三臂 gate；再据其结果决定是否做 2×2 通道分解。
```

**stair 只能作为优先 screening target，不得直接定为 development target**：
其 20k 的正迁移证据（`+15.40`）存在剂量混淆（sibling 臂 behavior share 高 2.5–3.3pp，
且优势方向与剂量同向，见 `PAPER_CLAIMS` I-5），完整训练历史又显示 horizon 敏感性。

`truck` 暂留作潜在 holdout，**不得**因其 `H_op` 看起来最大而提前定为开发场地——
那是 outcome-informed site selection 的前身（CLAUDE.md §8.5）。

`B⁻R⁺` 接口**暂不实现**。只有当重新普查产出 `CANDIDATE`、且三臂 gate 确认
hard exit 后仍有残余缺口时才实现，届时应正式拆成两个独立控制量
（`behavior_authority` / `replay_authority`），而非再加一个特殊模式。

---

## 9. 撤回：`J_best_known` 口径缺陷导致跨 target 对照无效（2026-08-06 追加）

外部 review 指出 stair 与 slide 的数据身份不匹配。核实脚本记录的来源，**其批评成立，
且范围比它指出的更大**：

| target | J_best | 来源文件 | method | seed | step |
|---|---:|---|---|---|---:|
| crawl | 984.9 | `endtoend_v1/crawl/.../scratch_s3_step100000.json` | scratch | s3 | 100k |
| hurdle | 851.1 | `endtoend_v1/hurdle/.../exit_s2_step100000.json` | hard-exit | s2 | 100k |
| slide | 951.5 | `slide_hard_exit_v1/.../exit_s1_step75000.json` | hard-exit | s1 | **75k** |
| stair | 67.0 | `stair_bac_gate_v1/.../slidesrc_s1_step20000.json` | slide-src | s1 | **20k** |
| cabinet | 100.6 | `cabinet_at10k_gate_v1/.../walk_s1_step20000.json` | walk-src | s1 | **20k** |

**五个 target 横跨三种训练预算与四种 method。**

### 9.1 明确撤回的内容

1. **`stair 6.7%` vs `slide 95.1%` 的对照作废**。stair 仅训练到 20k、slide 到 75k，
   差异被预算完全混淆。此前据此声称"同一份 reward，一个已解决一个只完成 6.7%"
   **是错误的**，不得引用。
2. §2.1 表中的 `% of theory max` 列**不得用于跨 target 比较**，只在同 step、
   同 method、同 seed 数的前提下才有意义。

### 9.2 根源是判据设计缺陷，不是实现偏差

预注册 §2.1 定义 `J_best_known = 该 target 上任何 source-free 评估的最大值`。
脚本忠实实现了它。但该定义同时对 method、seed、global_step 取 max，
构成 winner's curse，并使不同 target 的数字失去可比性。

按 CLAUDE.md §4，**判据一经冻结不得修改**，故不在 v1 内补救；
本节记录缺陷，由 v2 预注册重新定义（见 §9.4）。

### 9.3 主结论不受影响

`INSUFFICIENT_SITES` 由 `CANDIDATE = 0` 推出，而 `CANDIDATE = 0` 的直接原因是
**G4 在全部 25 个 target 上均为 `UNKNOWN`**（milestone 从未被采集）。
这与 `J_best_known` 的取法无关，故主裁决维持。

用可比口径（同为 100k、3-seed 均值）重算，方向不变：

```
crawl   960.2 / 1000 = 96.0%      （endtoend_v1 A 臂 3-seed 均值）
slide   929.1 / 1000 = 92.9%      （endtoend_v1 C 臂 3-seed 均值）
hurdle  840.4 / 1000 = 84.0%      （endtoend_v1 C 臂 3-seed 均值）
stair   无 100k 数据 → 不可比，UNKNOWN
```

### 9.4 v2 必须分开的三个量（不得再用单一 `J_best`）

```
best_observed_return      跨 method/seed/step 的最大值。
                          **仅用于排除"从未有任何策略达到过某水平"**，不得跨 target 比较。
fixed_budget_mean_return  同 method、同预算、多 seed 的均值。用于场地比较。
hard_exit_mean_return     hard-exit 臂在固定 step 的多 seed 均值。
                          **这才是判断 post-exit residual deficit 的唯一合法指标。**
```

每个数值必须携带完整身份，缺任一项即 `UNKNOWN`：

```
task / method / learner_seed / global_step / checkpoint_sha256 /
code_commit / source_bank / bootstrap_budget / exit_policy / evaluation_panel
```

### 9.5 连带修正：`9/9 显著负` 的措辞

crawl 的 `9/9` 是 **3 source × 3 learner seeds = 9 个 source–learner cells**，
它们共享同一 student 基线，**不是 9 个独立实验**。
正确写法是"9 个 source–learner cells 方向均为负"，
**不得**写成"9 个独立实验均显著"。统计单位是 learner seed，不是 evaluation episode（M16）。
