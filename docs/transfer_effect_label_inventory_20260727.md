# Stage-conditioned downstream transfer label audit — inventory v1.1

> 日期：2026-07-27（v1.1 结构性修订，一次性完成）  
> 性质：零训练。t=0 特征重建为只读环境 rollout，无梯度更新、无训练状态写入。  
> **裁决：`READY_FOR_CABINET_GATE`**  
> 机器可读：`docs/data/transfer_effect_label_inventory_20260727.json`（v1.1）  
> t=0 特征矩阵：`docs/data/transfer_effect_label_inventory_t0_features_20260727.json`

## 0. 本次修订采纳的三点意见

1. **撤回"t=0 不存在合法决策时点特征"的绝对结论**——已实际重建并给出特征矩阵（§2）。
2. **"证据有效"与"可进入同一模型"分离**——A 级保留，新增
   `model_eligible / protocol_family / label_horizon_K / independent_anchor_id /
   multi_seed_supported / feature_materialized` 六个字段，并把 STAGE10K 主标签
   horizon 统一为 \(K=10\text{k}\)（§3）。
3. **按独立 anchor 计数**——不再把同 anchor 下的三条 source 臂当作三个独立样本（§4）。

---

## 1. 标签口径修订

STAGE10K 第一版主标签统一为

\[
U_i(t{=}10\text{k},d,K{=}10\text{k})=J^{sf}_{20\text{k}}(\theta^{source_i})-J^{sf}_{20\text{k}}(\theta^{student})
\]

取在线 source-free deterministic eval 在 step 20000 的读数。这样标签窗口**恰好终止于
20k 决策边界**，20k 的换源不可能污染标签——这直接把 hurdle s1 从"仅窗口"升级为完整
A 级 cell。30k 冻结面板降为 secondary persistence 指标，不作第一版监督标签。

| cell | anchor | \(U(K{=}10\text{k})\) | 30k 终点（secondary） |
|---|---|---:|---:|
| STAGE10K.hurdle.s1.walk | hurdle t10k s1 | **+11.173** | +89.71（含 20k 换源，污染） |
| STAGE10K.hurdle.s2.run | hurdle t10k s2 | **+86.678** | +199.49 |
| STAGE10K.hurdle.s3.walk | hurdle t10k s3 | **+42.332** | +67.96 |
| STAGE10K.crawl.s3.run | crawl t10k s3 | **−104.036** | −142.91 |

EQD30K（\(t=0,K=30\text{k}\)）保持原口径，但**属不同 protocol family / 不同 horizon，
不与 STAGE10K 混入同一训练集**。

---

## 2. t=0 特征可重建性与实际特征矩阵

### 2.1 可重建性：成立

- 重建方式：`torch.manual_seed(训练 seed)` → 按训练构建顺序重建 Actor/Critic；
- 审计：vendored FastTD3 未改动；工作树在 `manual_seed` 与 actor 构建之间无新增 RNG 消耗；
- 实测：同 seed 两次重建 **逐位一致**，不同 seed **确有差异**；
- 唯一保留：无 t=0 checkpoint 可作端到端对照，有效性依赖构建顺序等价性（已记录在 JSON）。

因此 v1.0 中"t=0 无合法特征"的说法**撤回**。

### 2.2 实际特征矩阵（面板与在线机制完全相同：4 reset seeds × 8 ages = 32 状态，h=25）

| unit | ‖student action‖ | source | ΔR mean | ΔR LCB90 | ΔP mean | ΔP LCB90 | 动作距离 | 状态覆盖(NN) |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| hurdle s1 | 0.0583 | stand | +2.442 | +1.844 | +0.1732 | +0.1523 | 6.882 | 42.73 |
| hurdle s1 | | walk | +3.557 | +3.143 | +0.3557 | +0.3247 | 6.808 | 52.81 |
| hurdle s1 | | run | +3.357 | +2.808 | +0.3628 | +0.3300 | 7.144 | 52.90 |
| hurdle s2 | 0.0573 | stand | +2.240 | +1.655 | +0.1512 | +0.1345 | 6.885 | 41.57 |
| hurdle s2 | | walk | +3.932 | +3.513 | +0.3729 | +0.3481 | 6.811 | 52.06 |
| hurdle s2 | | run | +3.784 | +3.168 | +0.3864 | +0.3575 | 7.153 | 53.32 |
| hurdle s3 | 0.0475 | stand | +2.296 | +1.720 | +0.1557 | +0.1402 | 6.879 | 42.87 |
| hurdle s3 | | walk | +3.886 | +3.449 | +0.3796 | +0.3542 | 6.837 | 52.06 |
| hurdle s3 | | run | +4.271 | +3.551 | +0.4352 | +0.4016 | 7.173 | 52.50 |
| crawl s1 | 0.0607 | stand | +0.733 | +0.159 | +0.0729 | +0.0417 | 7.031 | 44.93 |
| crawl s1 | | walk | +2.524 | +1.919 | +0.0946 | +0.0520 | 7.077 | 54.23 |
| crawl s1 | | run | +1.965 | +1.358 | +0.0793 | +0.0369 | 7.259 | 55.25 |

t=0 critic 统计已记录但标记 `degenerate=True`（critic 随机初始化），不得单独作指标。

### 2.3 三个可直接读出的结论

1. **t=0 的 student 实测接近 no-op**：动作 L2 范数 0.0475–0.0607，而 source-student
   动作距离约 6.8–7.3。因此"t=0 的 matched student baseline"在数值上非常接近当年误判
   Crawl 的 zero baseline——这一点现在是**实测量**而不是推断。
2. **现行 feasibility 层在 t=0 会接纳全部 6 个 cell**（ΔR、ΔP 的 LCB 全为正），
   而其中 3 个（crawl）标签强负。这三行正是应当保留的 hard negative 监督，
   与 PI 判断一致。
3. **任务内排序信息真实存在**：t=0 的 ΔP 在 **3/3 seeds** 上把 hurdle 排成
   `run > walk > stand`，与 RBO 标签顺序完全一致；crawl 上正确识别出 stand 最有害，
   但 walk/run 顺序颠倒。这独立复现了项目既有判断——**相对排序有信息、绝对符号不可靠**。

**必须同时声明的警告**：ΔP 的 LCB 在两任务间确实存在间隔（hurdle [0.134, 0.402] vs
crawl [0.037, 0.052]），但当前只有两个任务且符号与任务身份完全共线，该间隔
**不能**被读作已发现的阈值。

---

## 3. 分级与建模资格分离（v1.1 新字段）

| 等级 | cell 数 | 说明 |
|---|---:|---|
| A | 10 | source-level 有效因果结果（保留） |
| B | 7 | bank / allocation-rule / lifecycle-rule / 多源序列 |
| C | 4 | Cabinet 单源（旧协议）、classic PTF 蒸馏、Transfer Map probe、MCG |
| D | 5 | adaptive revocation、SIV、SHU、P0、influence gate |

A 级按 `model_eligible` 拆为两个互不混用的池：

| pool | cells | \(t\) | \(K\) | 特征已物化 | 多种子 |
|---|---:|---:|---:|---|---|
| `eqd30k_pool` | 6 | 0 | 30k | 是（本次重建） | hurdle walk/run 是；其余否 |
| `stage10k_pool` | 4 | 10k | 10k | 是（quarantine JSON） | 否（每 cell 单 seed） |

---

## 4. 按独立实验单元重算的数据规模

计数规则：一个独立单元 = 一个 `(target task, training seed, stage anchor)`；
同 anchor 下的多条 source 臂共享 learner 与环境条件，**不是**独立样本。

| 池 | 独立 anchor | source 臂运行数 | source-cell 数 |
|---|---:|---:|---:|
| t=0 EQD30K | 4（hurdle s1/s2/s3、crawl s1） | 10 | 6 |
| t=10k STAGE10K | 4（hurdle s1/s2/s3、crawl s3） | 4 | 4 |
| **合计** | **8** | **14** | **10** |

**任务内异质性**：没有任何一个 target task 同时包含 helpful 与 harmful source。
hurdle 三源全正（幅度异质：stand +51 < walk +105 < run +380），
crawl 三源全负（stand −449 < walk −217 < run −208）。

**标签符号与任务身份共线性**：两个池均为**完全共线**——符号可由任务身份 100% 预测。
因此当前数据上任何"成功分离"都无法与"记住任务语境"区分。这正是必须先做 Cabinet
而不是补 Stair/Basketball 的原因。

---

## 5. Cabinet@10k 标准化等剂量标定：最小实验卡

**唯一目的**：在同一 target 内产生 run/stand/walk 的**异质标签**，打破符号与任务身份的
共线性，使模型有可能学习 source identity 而非 task context。

### 5.1 冻结协议（单因素：只改 source 身份）

| 项 | 取值 |
|---|---|
| target | `h1hand-cabinet-v0` |
| anchor | 同一 10k exact-abstention 纯 student anchor（每 seed 一个） |
| 臂 | stand / walk / run / student-only，共 4 臂 |
| behavior / replay dose | 0.5 / 0.5（单 source 与 student 同一 categorical，logits 均 0） |
| horizon | h=25 锁存 |
| 执行语义 | full-action，`bootstrap_only`（无蒸馏、无 MCG） |
| 干预窗口 | 10k → 20k |
| 主标签 | \(U(t{=}10\text{k},K{=}10\text{k})\)，20k 冻结 source-free evaluator |
| seeds | 1 / 2 / 3 |
| 附加产出 | 10k 处输出全部预注入特征（与 §2 同协议） |

### 5.2 两项必须先完成的配置前置（不是核心算法改动）

1. **`configs/target_evidence/humanoidbench_cabinet_v1.yaml` 尚不存在**。cabinet 在 info 中
   暴露 `door_openness_reward` 与 `subtask_complete`，需据此声明 achievement progress
   与必要约束。属 target MDP contract，允许；须在开跑前冻结。
2. **source bank 需换 obs adapter**。cabinet 观测维度实测 **213**，calibration bank 的
   `identity/output_dim=151` 不适用；历史已有成熟范式
   `h1hand_loco_sources_cabinet.yaml` 使用 `hb_robot_qpos_qvel(qpos_dim=109,
   output_dim=151)`。这是**必要的观测适配差异**，须逐 cell 记录，不是干预协议差异。

### 5.3 GPU 预算（基于本轮实测速率）

实测：10k anchor（纯 student，128 env）≈ 8.1–8.4 min；带 source 的 10k 步 ≈ 11–12 min
（源前向使吞吐从约 20 降到约 15–16 sps）；32-episode 冻结评估 ≈ 3.5 min/臂。

| 项 | 数量 | 单位耗时 | 小计 |
|---|---:|---:|---:|
| 10k anchor | 3 | 8.5 min | 26 min |
| 10k→20k 臂 | 12（4 臂 × 3 seeds） | 12 min | 144 min |
| 20k 冻结 source-free 评估 | 12 | 3.5 min | 42 min |
| 10k 预注入特征提取 | 3 | 2 min | 6 min |
| **实测基准合计** | | | **≈ 3.6 GPU h** |
| **保守（×1.5 余量）** | | | **≈ 5.5 GPU h** |

双路并行墙钟约 2–3 h。相比 v1.0 提出的 ~10 GPU h 三任务矩阵，成本更低、识别力更强。

### 5.4 预注册的判读规则（防止事后解释）

- **若 Cabinet 出现符号异质**（例如 run 正、stand 负）→ 共线性被打破，可进入
  leave-Cabinet-out gate；
- **若 Cabinet 三源同号**（全正或全负）→ 裁决转为 `STILL_TASK_CONFOUNDED`，
  **停止**，不得改为再加任务、改剂量或改 horizon 抢救；
- Basketball 全程**不参与**本阶段数据收集与模型选择，保留为完全未见的外部
  abstention 测试；只有 leave-Cabinet-out gate 通过、模型冻结后才允许使用。

---

## 6. 修订后的可识别性裁决：`READY_FOR_CABINET_GATE`

理由：

1. 特征重建**未被阻塞**（§2 已给出实际矩阵），故不是 `FEATURE_RECONSTRUCTION_BLOCKED`；
2. 当前数据确实**符号层面完全任务共线**（§4），若就此拟合模型必然落入
   `STILL_TASK_CONFOUNDED`；
3. 但该共线性有一个已识别、最小、且可证伪的破解手段——Cabinet@10k 标准化标定，
   其协议、前置配置、预算与判读规则均已冻结（§5）。

因此当前状态是"**已准备好执行 Cabinet gate**"，而不是"已可建模"或"不可识别"。

## 7. 本轮未做

不拟合 SCTU、不启动 Cabinet、不补 Crawl seeds、不运行 Stair/Basketball、
不改训练代码、未提交 git。等待批准。
