# Phase-1 统一机制规格：Bounded Bank Lease（v0.4）

> 日期：2026-07-19
> 状态：**兼容性审计完成（十八次复核裁定）；v0.4 锁定 9 条档：basketball
> 复用历史 retention，truck 两臂全新跑，均含 HEAD+schedule 设计。
> 不是 run card，不授权任何训练。**
> v0.3→v0.4：truck REUSE_FAIL（两臂新跑）；预算定案 9 条；新增 §6.1 truck
> 两臂 schedule 设计（0–30k 代码分支一致）与 §6.2 basketball 新臂 schedule
> vs 历史 all 的 CPU 语义等价测试前置门。
> v0.1→v0.2 变更：①单比特 A(t) 概念错误→behavior/replay 双通道 B(t)/R(t)
> （v0.1 无法表示 retention 对照臂）；②基线考古补漏——basketball_static
> retention 臂已存在（我漏审 adaptive_admission_v1 族的 static 对照）；
> ③新跑预算 9→条件性 6 条；④新增冻结兼容性审计（run card 前置）；
> ⑤主效应窗口改 35k–80k（buffer 覆盖算术）；⑥estimand 明确为 T2 硬退出
> 边际效果，判序 3→7 分支。
> 核心定位（冻结措辞）：在没有可靠迁移性 oracle 时，通过可弃权、有限
> 剂量和经验生命周期控制限制负迁移；不声称能够准确识别有害 source。

## 1. 机制定义（双通道状态）

**bounded bank lease** 的状态是 bank 级**双通道**：

    B(t) ∈ {0,1}：source behavior authority（bank 是否控制行为）
    R(t) ∈ {0,1}：source replay eligibility（bank 历史数据是否在 active replay）

| 阶段/条件 | B(t) | R(t) |
|---|---|---|
| [0, 30k) 两臂相同 | 1 | 1（authority 配额采样） |
| 30k 后 retention 臂（现状） | 0 | 1（物理占比，随 buffer 覆盖衰减） |
| 30k 后 hard-exit 臂（lease 机制） | 0 | **0**（立即退出，provenance 排除） |
| exact abstention | 0 | 0 |

- lease 语义 = **B 与 R 同步终止**（t_off 时双通道归零）；retention 现状
  = B 先停、R 拖尾。
- `t_off − t_on` = 预注册风险预算，不由在线 return 调整。
- 无回滚声明：learner 参数不回滚，主张是"切断后续暴露"，不是"保证恢复
  scratch"。
- 第一版全 source 同时到期（bank 级共享；错开到期会被 softmax 重归一化
  抬高剩余源相对概率，破坏剂量解释）。

## 2. 实验的真正命题（T2 边际，非 T1+T2 整体）

两臂的 B(t) 完全相同（都在 30k 结束行为），**唯一操纵变量是 30k 后的
R(t)**。estimand：

    Δ_exit = J(B_{>30k}=0, R_{>30k}=0) − J(B_{>30k}=0, R_{>30k}=physical tail)

即：**在行为 authority 已于 30k 结束的条件下，replay eligibility 应继续
保留还是与 behavior authority 同步终止？** 本实验只隔离 T2 硬退出的
边际因果效果；T1（行为窗口本身）的价值由既有 warmup/retention/scratch
证据链叙述，不由本实验裁决。

## 3. 实现载体（零新代码）

| 臂 | 载体 | 备注 |
|---|---|---|
| retention | `mcg_warmup_steps=30000` + `admission_replay_handoff=physical_after_authority` | 历史正式 runs 即此配置（§5） |
| hard exit | `AdmissionSchedule`：(0, admit-bank@冻结 masses) → (30000, exact abstention) | 撤销即时生效（mcg.py:440-446）；schedule 模式不进静态 fast path |

hard-exit 臂同样设 `physical_after_authority`（排除逻辑优先，参数一致保
单因素）。同 seed 整程对比（历史范式），不主张逐位配对。

审计要求（E16：验收带先推导后冻结）：
- hard-exit 臂：30k 后 execution 与 critic sample 的源计数增量**严格为
  零**（checkpoint 差分）；
- retention 臂：30k 后源 critic 占比 = 物理存量占比的机制预期（随覆盖
  线性衰减至 81.2k 归零），验收带由 buffer 覆盖推导；
- 机制中介变量：每 checkpoint 的 source physical share 与 source critic
  share（两臂）。

## 4. 分析窗口与指标（buffer 覆盖算术）

buffer=51.2k 步/env，source 只在 [0,30k) 写入 → retention 臂旧 source
数据至 **30k+51.2k=81.2k** 被完全覆盖；此后两臂无 treatment 差异，只看
100k 终点会把效应冲淡。

- **主指标**：35k–80k post-lease AUC（treatment 差异存续段；35k 起点
  避开 30k 切换瞬态）；
- 次指标：80k–95k persistence AUC（差异消失后的遗留）；
- 终点：100k source-free 性能；
- 机制中介：§3 的双通道 share 轨迹。

评估格点沿用历史 5k 间隔 `[eval]` 曲线（10k–95k 18 点），与既有基线
直接可比。

## 5. 基线复用审计（v0.2 修正版）

| 资产 | 配置要点 | 角色 |
|---|---|---|
| `admission_handoff_v1_20260713THANDOFFV1Z/truck_admission_h4_fix_s1-3` | 100k、admission-all、hurdle4 bank、warmup 30k、physical_after_authority、seeds 1-3 | **truck retention 臂，复用** |
| `adaptive_admission_v1_20260714T110054Z` 的 `basketball_static_s1-3` | 100k、admission-all（adaptive=off 对照）、std9_wfix_basketball（9 源）、student_logit=3.5892126423877646、warmup 30k、physical_after_authority、认证 json 21 项 checks 全 PASS | **basketball retention 臂，复用**（v0.1 考古漏审此族；其 10k–95k 相对 scratch 均值差 s1 −64.2/s2 −213.4/s3 −17.6，3/3 负，负迁移场地成立） |
| 同族 `crawl_static_s1-3` | 同款、crawl bank | 备选负任务（basketball 负迁移更强，仍以 basketball 为主） |
| `admission_handoff_v1/powerlift_admission_all_fix_s1-3` | 同款、std9_wfix_powerlift | 可选暂态伤害轴（powerlift headroom 95k 耗尽，不承载正任务 retention 裁决） |
| `admission_core_v1_FINALV2/powerlift_retain_all_s1-3` | fixed_quota 旧行为 | 已被取代，仅历史对照 |

主任务对：**truck（正）+ basketball（负）**；truck 替代 powerlift 的
修正已获十六次复核确认。

剂量说明：该实验族 warmup 内源 mass≈0.5（basketball static 认证 json 的
execution_counts 自洽验证：全程源占比 0.147≈0.5×30k/100k），由各任务
冻结 student_logit 决定；hard-exit 臂逐任务沿用同一 logit（非 P0 的
0.10——P0 是低剂量 lease 设计，本实验对齐历史 retention 臂剂量）。

## 6. 新跑预算（定案）

**审计已完成（十八次复核裁定）：basketball REUSE_PASS、truck REUSE_FAIL
→ 9 × 100k**：

| 任务 | retention 臂 | hard-exit 臂 |
|---|---|---|
| basketball | 复用历史 `basketball_static_s1-3`（3 条，0 新跑） | 新跑 3（HEAD+schedule） |
| truck | 新跑 3（HEAD+schedule，历史臂降为背景） | 新跑 3（HEAD+schedule） |

新跑合计 **9 × 100k ≈ 13–14 GPU-hours**（可选 powerlift hard-exit +3）。

### 6.1 truck 两臂 schedule 设计（0–30k 代码分支完全一致）

两臂都用 HEAD 的 `AdmissionSchedule` 路径，0–30k 走同一代码分支，唯一
差异是 30k 的 R 是否归零：

- retention 臂：`schedule = (0, admit-all)`，之后无撤销决策；30k 由
  `mcg_warmup_steps` 结束触发 `physical_after_authority`（R 拖尾）；
- hard-exit 臂：`schedule = (0, admit-all) → (30000, exact abstention)`
  （30k 时 R 归零）。

此设计消除 truck 因跨实现/跨代码分支引入的混杂——正是 REUSE_FAIL 的
补救：不复用历史臂，改为两臂同 HEAD 同分支新跑。

### 6.2 basketball 新臂前置门：schedule vs 历史 all 的 CPU 语义等价测试

basketball retention 臂=历史 `admission_mode=all`（a5cec9d 语义，§5 已
锚定）；hard-exit 新臂走 HEAD 的 schedule。两臂 0–30k 代码路径不同
（历史 all vs 新 schedule），故复用成立还需一个**训练前 CPU 语义等价
测试**（无长程训练）：30k 前逐步比较 `admission_mode=all` 与
`schedule step0 admit-all`，验证以下语义一致——candidate 概率（softmax
masses）、source 选择（segment 边界 categorical）、MCG 锁存（current/
steps_left）、transition provenance（8 字段）、replay 采样语义（allowed
slots + 配额权重）。

通过 = basketball 两臂 0–30k 可比（等价链：历史 all @ a5cec9d ≡ HEAD
all〔§2 语义中性〕≡ HEAD schedule-step0-admit-all〔本测试〕）；不通过
= basketball 也退化为两臂全新跑。测试完整设计写入 run card 作为训练
前置门。

## 7. 冻结兼容性审计（run card 前置，必做；逐任务输出 REUSE_PASS/FAIL）

复用历史 retention 臂的前提，逐项核对并写入审计文档：

1. 历史 runs 的 implementation commit/SHA（meta.txt 与 artifacts 记录）；
2. 当前 HEAD 训练核心对**非 resume 普通 100k run** 的语义一致性——
   `train_ptf.py`、`mcg.py`、`ptf_replay.py`、`admission_control.py` 自
   历史 commit 以来的相关路径差异逐项定性（触发条件不满足=中性须论证）；
3. bank YAML、source manifest 及**源权重哈希**逐项相等；
4. student_logit、MCG groups、warmup、batch、buffer、更新次数、学习率、
   scheduler 及普通非 resume 路径逐项相等；
5. **eval 协议**（35k–80k AUC 的数据来源是历史训练日志的 5k 网格
   `[eval]` 行——历史 checkpoint 通常只有 30k/60k/90k/final，无法离线
   重建 AUC）：built-in evaluator 代码一致性、task horizon、env/global
   NumPy 播种、eval env 数量与 seed、reward/return 口径、eval 间隔、
   **35k–80k 日志完整性**（每 5k 一点全覆盖）。训练语义兼容但 eval
   协议不兼容 → 该任务不能拿历史 AUC 作正式对照（可能须重跑两臂，
   而非只重评少数 checkpoint）；
6. **hard-exit 臂用哪个实现启动**：优先=历史 retention 运行对应的冻结
   实现（消除跨实现混杂）；若用当前 HEAD，须给出逐项语义等价审计。

## 8. 预写结果判序（层级制，防事后挪标）

量化 gate（改善/保留/恶化的 δ 与检验）在 run card 预注册；层级先冻结
（逐层判定，非并列互斥分支）：

1. **工程有效性**：审计计数（hard-exit 臂 30k 后源增量严格零等）不过
   → ENGINEERING_INVALID，止；
2. **可测性/seed 异质性**：CI 过宽或 seed 反号超阈 → UNCERTAIN（不得
   归入"无差异"），止；
3. **basketball 的 hard-exit − retention 方向**：改善/恶化/无差异；
4. **truck 的 hard-exit − retention 方向**：保留/损失/无差异/改善；
5. **与 scratch 比较后的追加限定标签**（不与 3/4 并列，是结果修饰）：
   `MITIGATION_ONLY`（负任务改善但仍显著低于 scratch——只称"缓解"）/
   `NEGATIVE_TRANSFER_AVOIDED` / `POSITIVE_GAIN_RETAINED` /
   `POSITIVE_GAIN_LOST`。

方向组合的机制解读（预写，不作分类用）：负改善∧正保留=T2 成立；
负改善∧正损失=风险-收益权衡存在；负恶化=旧 source replay 可能有补偿
价值或 hard exit 造成分布突变；正改善=正迁移 bank 的退役数据亦可阶段
失效；**双任务真无差异（CI 排除有意义效应）→ 只能结论"30k 后的
source replay 物理拖尾不是该配置下的主要可干预因果通道"——残余负迁移
可能位于 0–30k 已形成的 critic/actor/optimizer 状态、source 诱导
occupancy、早期数据覆盖或其他未分离通道，不得收窄归因到 learner
state**。

## 9. 下一步

1. 本 v0.2 交 ChatGPT 复核；
2. 通过后执行 §7 兼容性审计（零训练）→ 结果决定 6 条 or 12 条；
3. 起草最小 run card（预注册 gate/δ/评估协议/冻结矩阵）→ PI 批准；
4. 此前不写新代码、不启动训练。
