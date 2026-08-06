# 预注册：racing 的准入能力（要不要用源，而非用哪个）

> 2026-08-04。**本文与裁决脚本必须在 crawl / slide 的任何评估结果产出之前提交 git。**
> 定位：补齐 `M31(d)` 明确指出、但因判决场选错而四轮未果的唯一机制缺口。

## 1. 立项门

1. **核心问题**：racing 能否判断"**这个 target 上是不是所有源都该拒绝**"，
   而不仅仅是"在候选里选哪个"。
2. **唯一主要假设**：在 `K = 10000` 的 racing 测量下，per-seed 准入决策
   `admit = ∃i: U_i > 2·SE_i` 在**全负 target 上判拒**、在**有正源的 target 上判纳**。
3. **正负后果**：正 → 准入与选源可由同一次 racing 测量同时给出，
   端到端系统的最后一块补齐；负 → 准入能力不成立，论文只保留"选源"这一半，
   并如实记录 racing 无法承担绝对准入。
4. **是否重复**：不重复，见 §2。
5. **最小成本**：24 条 10k 训练 + 24 点评估（hurdle 复用既有同协议数据）。

## 2. 与 `racing_reject_door_v1–v4` 的区别（关键，否则就是第五次重复）

door 系列四轮全部未能裁决主终点，**根因是场地而非机制**。`M31(d)` 原文：

> door 因此**不是合适的判决场**，racing 拒绝能力的检验须另找 `U` 符号稳定的全负 target。

| | door v1–v4（已失败） | 本实验 |
|---|---|---|
| 负例 target | door，`U` 符号**跨 learner 反转**（18/18 负 → 新批 2/9 正）| **crawl**，效应量 −208 ~ −448（比 door 的 −7~−43 大一个数量级），32-episode 面板 **0/32 获胜** |
| 正例 | **无**（只有全负场地）| **hurdle + slide**，排除"永远拒绝"这一平凡解 |
| 判据形态 | 依赖 door 自身的 ground truth 稳定性 | per-seed 自含：`U_i > 2·SE_i`，不依赖任何跨 learner 标签 |

**"混源稀释"这一替代解释已被排除**：`crawl_equal_dose_source_calibration_v1`
用单源等剂量证明 crawl 上 stand/walk/run **每一个单独用都是负的**
（−448.480 / −216.604 / −208.070），故"全部拒绝"确为正确答案。

## 3. estimand 与判据（冻结）

```
U_i,s(K) = J(源 i 臂, seed s, K 步)  −  J(student 臂, seed s, K 步)
           逐 episode 配对（同一 128-episode 冻结面板，(seed,rank) → seed*1000+rank 逐位相同）
SE_i,s   = 该 128 条配对差值序列的标准误

per-seed 准入决策（racing 的真实用法：在哪个 learner 上测，就在哪个 learner 上用）
    admit(T, s) = ∃i ∈ {stand, walk, run} :  U_i,s > 2 · SE_i,s
```

**为什么这里用 episode 面板 SE 而不是 learner 间方差（与 M16 不冲突）**：
M16 针对的是"跨 learner 的**结论**"。本判据不做任何跨 learner 推广——
它在每个 learner 上独立给出一个决策，再统计这些决策的正确率。
`transfer_utility_is_not_a_property_20260731.md` §4.1 已论证：racing 的估计与
应用在同一 learner，故 `M31` 的符号不可推广性对它不构成威胁。

**为什么 δ = 2·SE 而不是一个校准出来的常数**：门槛相对**每个测量自身的噪声**定义，
不依赖任何 design data，因而不存在阈值自证（§8.4）与 site selection 污染（§8.5）。

## 4. 判决场（冻结）

| 角色 | target | 期望 | 真值来源 |
|---|---|---|---|
| **负例** | crawl | **0/3 seed admit** | 单源等剂量 −448.480 / −216.604 / −208.070（32-ep 面板 0/32 获胜）|
| 正例 | hurdle | **3/3 seed admit** | `K=10000` 实测 9/9 全正（run +102.19/+110.51/+81.16）|
| 正例 | slide | **3/3 seed admit** | walk 为最优源，`GEN_OK` 6 learner 一致（K=30000 时 +56.95）|

## 5. 协议（冻结，与 `run_racing_min_horizon_v1.sh` 逐项同构）

```
臂（4）        student(空 bank, ADMISSION_MODE=legacy) / stand / walk / run
源臂配置        SOURCE_BANK=configs/source_banks/calibration/h1hand_{target}_rbo_{src}.yaml
               PTF_MCG=1  GROUPS=legs_torso,arms,hands  WARMUP_MODE=admission_bootstrap
               ABLATION=bootstrap_only  ADMISSION_MODE=all  STUDENT_LOGIT=0.0
               EXPECTED_SOURCE_MASS=0.5  WARMUP_MIN_STEPS=25
               REPLAY: recency=0 uniform_mix=1 priority_alpha=0 handoff=physical_after_authority
target（2 新跑） h1hand-crawl-v0   h1hand-slide-v0        seeds 1,2,3
target（1 复用） h1hand-hurdle-v0  ← docs/data/racing_min_horizon_v1/correct_lr/
起点            t=0（从头训练，不用 anchor）
K               10000（PTF_RUN_STOP_STEP=10000；checkpoint 另存 2000/5000 备用，**不进判据**）
LR 日程         TOTAL_TIMESTEPS=100000（= 部署时长训练长度，与 racing_min_horizon 一致）
其余            NUM_ENVS=128 BATCH=32768 BUFFER=51200 LEARNING_STARTS=10 NUM_UPDATES=2
               COMPILE=0 AMP=1
评估            scripts/p0_evaluator.py，source-free，deterministic，--eval-seeds panel128
```

**hurdle 复用的合法性**：`racing_min_horizon_v1` 的 `correct_lr` 批使用**同一份**
`run_racing_min_horizon_v1.sh`、同一 K、同一 128-episode 面板、同一四臂结构，
仅 target 不同。本实验只从其 per-episode 评估重算 `U` 与 `SE`，不引用其裁决结论。

**剂量验收**：源臂 behavior share ∈ `[0.45, 0.55]`；同 target 内臂间 share 差 < 5pp（M26）。

**臂身份验收（M28）**：crawl/slide 的 bank 为 `null_option: false`，
hurdle 为 `true`，故 `source_names` 格式不同。判据一律用
**"非 `null` 部分 == `[arm]`"**，不得硬编码列表。

## 6. 裁决（冻结）

```
false_admit  = crawl 上 admit 的 seed 数              期望 0
false_reject = hurdle 与 slide 上 **not** admit 的 seed 数   期望 0

false_admit > 0                              → ADMISSION_FALSE_ADMIT     （优先，错更严重）
false_admit == 0 且 false_reject > 0         → ADMISSION_FALSE_REJECT
false_admit == 0 且 false_reject == 0        → ADMISSION_VIABLE
任一 (target, arm, seed) 组合缺失             → INCOMPLETE（非零退出）
工程验收失败（剂量/臂身份/面板/协议）          → ENGINEERING_INVALID
```

## 7. `CLAUDE.md` §8 设计层自查

- **8.1 辨别力**：两个平凡解各被一侧排除——"**永远拒绝**"在 hurdle/slide 上产生
  `false_reject = 6` → `ADMISSION_FALSE_REJECT`；"**永远接受**"在 crawl 上产生
  `false_admit = 3` → `ADMISSION_FALSE_ADMIT`。第三个捷径"选 |U| 最大者"不适用——
  本判据不排序，只判存在性。
- **8.2 混淆**：同 target 内四臂除 source bank 外逐项同参数；剂量逐 checkpoint 验收；
  `U` 由**同 seed 配对差值**定义，故 learner 初始化与环境种子在相减时抵消。
- **8.3 独立重复**：三个 target 各 3 个独立 learner seed；**crawl 由本实验从 1 seed
  补到 3 seeds**（M31 要求的符号稳定性检验一并完成）。
- **8.4 前提蕴含结论**：三个实质分支均可达。`ADMISSION_FALSE_REJECT` 的最大风险
  是 **slide 在 K=10000 时 walk 的 `U` 尚未超过 `2·SE`**（其真值 +56.95 测于 K=30000），
  这是真实可能的结果而非设计缺陷；`ADMISSION_FALSE_ADMIT` 在 crawl 某源短期为正时发生。
- **8.5 site selection**：三个 target 的真值**均已知**，故本实验是
  **在已知真值的场地上检验判据**，不是发现新事实。结论只能声称
  "在这三个 target 上成立"，跨任务推广须用真值未知的新 target 前瞻验证。
  判据 `δ = 2·SE` 不含任何由这三个 target 校准出的量，故不存在阈值自证。
- **8.6 本轮教训对照**：
  - `M28`——bank 配置已逐个核实（6 个文件存在，`null_option: false`，identity/151）；
  - `M31`——本判据不做跨 learner 推广，估计与应用同 learner；
  - `M33`——已先验算目标区间非空：hurdle 的 `U = +81~+110` 远超 `2·SE`（面板 SE 量级 2–5），
    crawl 的真值 −208~−448 远低于零，故"crawl 全拒 ∧ hurdle 全纳"在数据上可同时成立；
    唯一不确定项是 slide（见 8.4），已如实标注为风险而非通过条件。
  - `M24`——单批 3 seeds，若为正须注明"待独立重复"。
- **8.7 判据切换红线**：crawl/slide 为全新数据；hurdle 仅复用原始 per-episode 评估
  重算，不引用其裁决。裁决后不得改 `δ`、不得改 target 集合、不得改 K。

## 8. 能与不能声称

**能**（若 `ADMISSION_VIABLE`）：在这三个 target 上，一次 `K=10000` 的 racing
测量可同时给出**准入**（要不要用源）与**选源**（用哪个）两个决策。

**不得**：

1. 不得声称跨任务普适——三个 target 的真值均已知（§8.5）；
2. 不得省略 `M24` 的单批限制；
3. 不得把 `δ = 2·SE` 解释为"统计显著性检验"——它是 per-seed 决策门槛，
   不控制多重比较（每个 target 3 个源同时检验）；
4. 若为 `ADMISSION_FALSE_REJECT` 且原因是 slide 的 K 不足，
   **不得**在本实验内加大 K 抢救——那须作为新的预注册实验。

## 9. 不得做的事

- 裁决后不得调 `δ`、K、target 集合或剂量带。
- 若为负，如实报告并按 §8.4 记录是哪一类错误；论文相应只保留"选源"这一半。
