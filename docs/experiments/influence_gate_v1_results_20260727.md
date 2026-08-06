# Update-space influence proxy 离线双向 gate v1：结果与裁决

> 日期：2026-07-27  
> 授权范围：PI 于 2026-07-27 对已封存 gradient-influence 家族的**一次性临时解封**，
> 仅用于本次离线双向可行性 gate。  
> **裁决结果：FAIL（含排序反转）→ 按预注册规则重新封存该信号族。**

## 1. 冻结设置（揭盲前预注册，运行后未改动）

**操作性校准标签**（10k–20k 段内 paired nAUC，online−scratch，同 seed；
非无保留 source-level ground truth）：

| Cell | 角色 | 段内证据 |
|---|---|---:|
| crawl_s3 · run@10k | harmful | −211.06 |
| hurdle_s2 · run@10k | helpful | +19.79 |
| hurdle_s3 · walk@10k | helpful | +9.92 |
| hurdle_s1 · walk@10k | helpful | +8.48 |

**Estimand**：`I_critic = L_val(U(θ; 50%D_stu+50%D_src)) − L_val(U(θ; 100%D_stu))`。
control 臂严格 100% student，未混入任何 source 数据。`U` = 一个完整 FastTD3
outer update unit（2×critic AdamW + 第 2 次后 1×actor AdamW + 每次 critic 后
soft target τ=0.1），使用 anchor 恢复的真实 optimizer/scheduler 状态，无虚拟学习率。
两臂共享 student batch、替换位置与全部随机数，唯一差异是被替换半批的内容。

**执行记录**（解释用，揭盲后未调整）：4 个 10k anchor 均由 exact-abstention
纯 student 重放生成（env/seed 校验通过，replay ptr=10000）；held-out val 为
64 个整列 slot（8192 条），与训练 batch 抽样池严格不相交；每 cell 64 次配对重抽，
5000 次 bootstrap 求 90% 区间。

## 2. 结果

| Cell | 角色 | I_mean | LCB90 | UCB90 | std | 正号率 | unique src trans | 名义 src share |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| crawl_s3_run | **harmful** | +0.07080 | +0.06980 | +0.07178 | 0.0048 | 100% | 6400 | 0.5 |
| hurdle_s1_walk | helpful | +0.03061 | +0.03047 | +0.03075 | 0.0007 | 100% | 6127 | 0.5 |
| hurdle_s2_run | helpful | **+0.13947** | +0.13834 | +0.14062 | 0.0056 | 100% | 6357 | 0.5 |
| hurdle_s3_walk | helpful | +0.06327 | +0.06117 | +0.06544 | 0.0104 | 100% | 6296 | 0.5 |

区间为**给定各自 10k learner anchor 的条件 batch 不确定性**，不是跨 seed 置信区间，
也不代表该指标具备跨任务泛化能力。source transitions 由 probe 面板采集（≤6400 条），
在 16384 行的半批中被有放回重采样约 2.6×。

## 3. 三级裁决

- **ABSOLUTE_PASS**：需 harmful LCB>0（成立）**且**全部 helpful UCB<0（**不成立**，
  三个 helpful cells 的 UCB 均显著为正）→ 未达成。
- **RANKING_ONLY**：需 harmful LCB(+0.0698) > max helpful UCB(+0.1406) → **不成立**。
- **判定：FAIL**。

失败形态比"区间重叠"更严重：**排序反转**。已知正迁移最强的 cell
（hurdle_s2_run，段内 +19.79）在该指标下被判为"最有害"（+0.1395），
高于真正有害的 crawl_s3_run（+0.0708）近一倍；四个 cell 的排序与冻结标签
基本无关。64 次重抽的 std 极小（0.0007–0.0104），区间极窄，因此
**FAIL 不是功效不足或噪声，而是指标本身指向错误**。

## 4. 辅助诊断（不参与裁决）

| Cell | 角色 | critic grad cos | disagreement ratio |
|---|---|---:|---:|
| crawl_s3_run | harmful | +0.194 | 4.07 |
| hurdle_s1_walk | helpful | +0.354 | 3.89 |
| hurdle_s2_run | helpful | −0.359 | 2.92 |
| hurdle_s3_walk | helpful | −0.259 | 5.12 |

辅助量**不支持**主结果、也不构成任何救回路径：梯度余弦在 helpful 组内部即正负分裂
（+0.354 vs −0.359），harmful 反而居中为正；candidate 2 的 critic disagreement 比
（harmful 4.07）完全落在 helpful 区间 2.92–5.12 之内，无分离度。按裁决，
candidate 2 仅为描述性读数，不得在 candidate 1 失败后用于救回结论。

## 5. 失败机制（记录为负结果解释，不作为修复提案）

四个 cell 的 I 全部为正，且量级与"source 数据离 student 分布多远"同向、
与"是否有益于后续学习"无关。这符合该 estimand 的机械性质：用任何非 student 分布的
数据替换半个 batch，都会降低 critic 对 student 分布 held-out 数据的拟合度。
hurdle_s2_run 恰是分布最远、也最有益的一个，于是出现反转。

因此该指标实际度量的是**即时分布错配**，而非**延迟学习价值**——与项目已有的
T⁰、T^critic、SHU 等"即时量"失败属于同一类错误，只是换到了参数更新空间。
本文档不提出该家族的任何修复变体（归一化、换 val 分布、换 horizon 等均不在授权内）。

## 6. 对迁移性指标路线的含义

- **重新封存 update-space influence 家族**：不调学习率、batch 数、阈值、标签或统计口径，
  不做变体抢救。该家族现与 SIV/SHU/P0/adaptive revocation 并列为已裁决失败线。
- **不进入在线外部验证**：预注册要求 gate 通过才允许 Crawl s3 + Hurdle 正例的最小在线验证；
  现无资格。
- **正面价值**：本轮以约 1 GPU 小时成本，在"一步更新几何"这一维度上排除了一个
  看似最直接的候选，并给出了明确机制解释——**即时分布错配 ≠ 延迟学习价值**。
  这条负结果与课题核心论点一致，可直接进入论文的 negative-results/动机部分。
- **剩余空白未变**：第二层 student-relative learning value 仍未解决，且现有证据表明
  它不能由"注入前的任何即时量"（行为空间、值空间、单步更新空间）预测。

## 7. 产物与校验

| 工件 | 路径 | sha256(前16) |
|---|---|---|
| 结果 JSON（含 64 次逐样本） | `docs/data/influence_gate_v1/influence_gate_v1_results.json` | `b038daae23a2a993` |
| 探针脚本（只读，参数副本上更新） | `scripts/analysis/influence_gate_v1_probe.py` | `7c08d6549aab83d7` |

Anchor bundles（各 1.9G，`artifacts/influence_gate_v1/anchors/`）：

| anchor | env / seed | completed_steps | manifest sha256(前16) | learner sha256(前16) |
|---|---|---:|---|---|
| hurdle_s1 | h1hand-hurdle-v0 / 1 | 10000 | `a863ec86ef1d84b1` | `35358370080357ae` |
| hurdle_s2 | h1hand-hurdle-v0 / 2 | 10000 | `5cb6f9b81b4b4f30` | `b0dde97602feeff1` |
| hurdle_s3 | h1hand-hurdle-v0 / 3 | 10000 | `2df7af5cbf09f417` | `190318663b899faf` |
| crawl_s3 | h1hand-crawl-v0 / 3 | 10000 | `4afff385bd23f8d5` | `8aa7c724b8bfc56c` |

四个 anchor 均由 exact-abstention 纯 student 重放生成（空 bank、`run_stop_step=10000`、
`total_timesteps=30000` 保持 LR 日程、eval/render=0、torch_deterministic），
replay `ptr=10000` 与 `completed_vector_steps` 一致。日志：
`logs/train/influence_gate_v1/replay_{hurdle_s1,hurdle_s2,hurdle_s3,crawl_s3}.log`
与探针日志 `probe_run.log`。运行结束无残留训练/探针进程；训练代码未改动，未提交 git。
