# 预注册：端到端迁移系统（准入 + 选源 + 剂量退出）

> 2026-08-04。**本文、决策文件与裁决脚本必须在任何新主训练启动之前提交 git。**
> 定位：首次把三个已各自裁决的零件组合成一条决策链，检验**组合后的整体效果**。

## 1. 立项门

1. **核心问题**：三个零件组合后，能否在不同 target 上**自动**做出正确决定——
   该用源时用对源并及时退出，不该用时整体拒绝？
2. **唯一主要假设**：端到端臂在 hurdle/slide 上优于 scratch，在全部三个 target 上
   优于"用零成本信号选源并全程注入"。
3. **正负后果**：正 → 论文的第二支柱由零件升级为系统；
   负 → 论文如实退化为"不可能性刻画 + 各零件单独结果"，不声称系统级贡献。
4. **是否重复**：不重复——三个零件从未在同一条决策链上运行过。
5. **最小成本**：9 条 100k 训练（slide 完全复用，hurdle 复用 2/3 臂）。

## 2. 系统定义与决策链

```
阶段 1  racing        4 臂(3 源 + student) × K=10000        成本 40k 步
阶段 2  决策（自动）   admit = ∃i U_i > 2·SE_i ; source = argmax_i U_i
                     ——由 scripts/analysis/decide_racing_admission_v1.py 从
                       P1 冻结输出导出，无人工介入
阶段 3  主训练        admit → 带选中源训练，30k 硬退出，至 100k
                     reject → 纯 student 训练至 100k
```

**决策已于本预注册之前冻结**（`docs/data/endtoend_v1/decisions.json`）：

| target | seeds | 决策 | 选中源 |
|---|---|---|---|
| crawl | 1,2,3 | **REJECT** ×3 | — |
| hurdle | 1,2,3 | **ADMIT** ×3 | run |
| slide | 1,2,3 | **ADMIT** ×3 | walk |

各零件的单独证据：准入 `ADMISSION_VIABLE`、选源 `SELECTION_VALUABLE`（hurdle 4.95×）、
退出 `HARD_EXIT_SUPPORTED`（slide +631.8）。**本实验只检验组合，不重测零件。**

## 3. 三臂（冻结）

| 臂 | 定义 | 数据来源 |
|---|---|---|
| **A scratch** | 空 bank，纯 RL，100k | hurdle `hspd_scratch`✓ / slide `sspd_scratch`✓ / crawl **新跑** |
| **B 盲目用源** | 用**零成本信号**选源并**全程注入**（代表没有准入与退出的做法）| hurdle `hspd_source`(run)✓ / slide `sspd_walk`(walk)✓ / crawl **新跑**(run) |
| **C 端到端** | 按 §2 决策链执行 | hurdle **新跑**(run+30k 退出) / slide `shev1_exit`✓ / crawl = 纯 student（同 A 配置）|

**臂 B 的选源准则（先于数据冻结）**：取 `progress_screen_v1` 已测的 zero-shot
前进位移 argmax——族 12 证明该信号不可用于准入，但它正是"没有 racing 时人们会用的
零成本信号"，故用它定义盲目基线是可辩护的：

```
crawl   stand 0.221  walk 3.664  run 14.302  → run
hurdle  stand 0.188  walk 8.717  run 22.521  → run
slide   stand 0.183  walk 1.814  run  1.753  → walk
```

**必须承认的后果**：在 hurdle 与 slide 上，该信号**碰巧与 racing 选出同一个源**，
故这两个 target 上 `C vs B` 实际检验的是**退出机制**而非选源；
只有 crawl 上 `C vs B` 检验的是**准入**。选源的单独价值由
`hurdle_selection_value_v1` 支撑，本实验不重复检验。

## 4. 成本口径（冻结）

端到端臂比另两臂多花 **40k 步** 的 racing 成本。本实验报告两个口径：

```
口径 1（主判据，learner 步数对齐）  三臂均在主训练 100k 处比较；
                                   端到端的 40k racing 成本**显式列出**，不摊入
口径 2（描述性，总交互对齐）        端到端在总交互 100k（= 主训练 60k）处的性能，
                                   因缺 60k 评估点，用已有的 50k 点作**保守下界**
```

口径 1 对端到端有利，口径 2 对端到端不利；**两者都报，不得只报其一**。

## 5. 协议（冻结）

```
新跑臂（9 条）
  hurdle C:  SOURCE_BANK=calibration/h1hand_hurdle_rbo_run.yaml，30k 后 ADMISSION_MODE=none
             实现方式与 slide_hard_exit_v1 同构：prefix 0→30k 存 branch anchor，
             再从该 anchor 以 MODE=none / MASS=0.0 续训至 100k
  crawl  A:  SOURCE_BANK=empty.yaml, ADMISSION_MODE=legacy, 0→100k
  crawl  B:  SOURCE_BANK=calibration/h1hand_crawl_rbo_run.yaml, MODE=all, MASS=0.5, 0→100k
seeds        1,2,3（与 racing 决策逐 seed 对应）
其余参数      NUM_ENVS=128 BATCH=32768 BUFFER=51200 LEARNING_STARTS=10 NUM_UPDATES=2
             COMPILE=0 AMP=1 TOTAL_TIMESTEPS=100000
评估          scripts/p0_evaluator.py，source-free，deterministic，panel128
评估点        30000, 50000, 75000, 100000
```

**crawl 的 C 臂 = A 臂**（决策为 REJECT，主训练即纯 student），二者共用同一批数据；
因此 crawl 上**不检验** `C vs A`（那会自动成立，违反 §8.4），只检验 `C vs B`。

## 6. 判据（冻结）

```
逐 (target, seed) 配对比较 100k 的 source-free return（panel128）

(a)  hurdle 与 slide：  C > A     要求每个 target 3/3 seed
(b)  三个 target 全部：  C > B     要求每个 target 3/3 seed

(a) ∧ (b)                     → ENDTOEND_SUPPORTED
(b) 成立但 (a) 不成立          → ENDTOEND_PARTIAL（系统安全但未带来增益）
(b) 不成立                     → ENDTOEND_REFUTED
任一组合缺失                   → INCOMPLETE（非零退出）
工程验收失败                   → ENGINEERING_INVALID
```

跨 learner 的汇总一律用 **learner 间**离散度（3 seeds 的 sd），
不用 episode 面板 SE（`M16`）；per-seed 比较用配对面板。

## 7. `CLAUDE.md` §8 设计层自查

- **8.1 辨别力**：两个平凡策略各被一侧排除——"**永不用源**"使 C≡A，(a) 失败；
  "**永远用源且不退出**"使 C≡B，(b) 失败。第三个捷径"总是选 |U| 最大者"
  不适用，因为 crawl 上全部 U 为负而系统给出 REJECT。
- **8.2 混淆**：三臂同 seed 配对、同评估面板、除 bank/退出外逐项同参数；
  剂量逐 checkpoint 验收；crawl 的 A 与 C 共用数据故无臂间差异可混淆。
- **8.3 独立重复**：每 target 3 个独立 learner seed；按 `M24`，为正须注明待独立重复。
- **8.4 前提蕴含结论**：`ENDTOEND_REFUTED` 与 `ENDTOEND_PARTIAL` 均可达——
  (a) 会在"30k 退出后仍不及 scratch"时失败（hurdle 的 100k 倍率已知只有 1.24×，
  这是真实风险）；(b) 会在盲目用源碰巧不差时失败。
  crawl 的 `C vs A` 已被排除出判据正是为了避免自动成立。
- **8.5 site selection**：三个 target 的源标签真值均已知，故本实验检验的是
  **系统在已知场地上的行为**，不是发现新事实。跨任务推广须在真值未知的新 target 上前瞻验证。
- **8.6 本轮教训对照**：`M33`（先验算目标区间非空：hurdle 的 C 臂在 30k 退出后
  参照 slide 的 +631.8 有大幅上行空间，A 臂 100k 已知约 644，区间非空）；
  `M28`（bank 已核实）；`M16`（跨 learner 用 sd 不用面板 SE）；
  `M24`（单批 3 seeds）；`M26`（剂量验收 + 臂间 share 差）。
- **8.7 判据切换红线**：决策文件与判据均先于新训练冻结；裁决后不得改臂定义、
  评估点或比较口径。

## 8. 能与不能声称

**能**（若 `ENDTOEND_SUPPORTED`）：三个零件组合成的自动决策链，在这三个 target 上
无需人工指定源或退出时机即可优于 scratch 与盲目用源。

**不得**：

1. 不得声称跨任务普适（§8.5）；
2. 不得省略 40k racing 成本，也不得只报口径 1（§4）；
3. **不得**把 hurdle/slide 上的 `C vs B` 解释为选源价值——那两处盲目信号碰巧选对，
   实际检验的是退出机制（§3）；
4. 不得把剂量退出本身称为贡献（工程基线）；
5. 不得省略 `M24` 的单批限制。

## 9. 不得做的事

- 裁决后不得改决策文件、臂定义、评估点或判据。
- 若为 `ENDTOEND_REFUTED` 或 `PARTIAL`，如实报告并按目标退化为
  "不可能性刻画 + 各零件单独结果"，不声称系统级贡献。
