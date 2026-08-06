# Run Card：Phase-1 Bounded Bank Lease — T2 硬退出边际因果实验

> 版本：**v0.6（草案；二十三次复核窄范围修订后——finalize 全 payload 比较/
> 门 B 同配置验证补全/反例单测 52 项）**
> 日期：2026-07-19
> 状态：**待 ChatGPT 复核 + PI 批准。批准前不训练、不冻结 δ。**
> 上游：机制审计 `phase1_mechanism_audit_20260719.md`（v2）；机制规格
> `phase1_bounded_bank_lease_spec_20260719.md`（v0.4）；兼容性审计
> `phase1_reuse_compatibility_audit_20260719.md`（truck + scratch REUSE_FAIL）。
> 定位（冻结）：无可靠迁移性 oracle 时，通过可弃权、有限剂量与经验生命
> 周期控制**限制负迁移**；不声称能准确识别有害 source。
> **任务选择边界**：basketball（负）/truck（正）是按历史结果选定的
> **purposeful 机制压力测试场地**，非 HumanoidBench 随机任务抽样；结论
> 不得外推为"总体有效性"。
> v0.2→v0.3（二十次复核）：①**更正**——`execution_counts_at_apply` 确
> 实存在于 `decision_history`（我 v0.2 只查 `policy_events` 即断言"不
> 存在"，错误），门 B 改混合取证；②冻结 `save_interval=0` 避免与显式
> checkpoint 同名覆盖，checkpoint steps 改逗号分隔；③门 B 数学化
> （q_k/r_{a:b}/ε 包络，90k/100k 不再要求 main source>0）；④门 A 定为
> 有限 harness（80–100 步 + 静态 `snapshot_at` 证明，不真跑 30k）；
> ⑤scratch 标签降描述性；`POSITIVE_GAIN_LOST` 收紧 + 新增
> `SCRATCH_COMPARISON_UNCERTAIN`。
> v0.3→v0.4（二十一次复核）：scratch 双层裁定（`CAUSAL_COMPARATOR_REUSE_FAIL`
> + `METRIC_SCALE_REUSE_PASS`——**更正**我"scratch 无 provenance"的错判，
> W&B 档案完整）；δ 定位 = externally anchored SESOI；δ 候选与门 B 容差生成。
> v0.4→v0.5（二十二次复核四阻塞）：①ε 由硬编码 0.01 改 Hoeffding 统计式
> （§4）、剂量带诚实标注为工程风险预算；②两脚本补输入身份验证 + generator
> SHA + git HEAD/dirty；③δ 脚本补自动断言（wandb run 唯一匹配、patch 剔除
> 离线 probe 后逐位一致、历史 PTF 实为 scratch）；④补 `--finalize` 安全
> 冻结流程 + 反例单测 26 项；⑤80k 口径分离（历史 NA / 新跑臂全链）。
> v0.5→v0.6（二十三次复核）：①finalize 改**完整科学 payload 逐位比较**
> （只排除 status/git_head/git_dirty/finalized_from_candidate——旧版只比
> 汇总数，输入 checkpoint/generator 被换而汇总恰好不变仍会通过）；②门 B
> 补全"同配置"验证：mcg / warmup_mode / ablation / warmup_min_steps /
> replay 三参数 / task-specific student_logit / num_envs·batch·buffer·
> num_updates·learning_starts·total_timesteps·eval_interval，且**完整
> candidate masses 向量须跨全部 ckpt+seed 一致**（旧版只比长度与 source
> 总和）；`admission_adaptive` 等 5 项仅 7-14 版存在 → 采用"存在则验证、
> 缺失如实记录"，不让脚本声称超出实际的验证范围；③反例单测 26→**52 项**。

## 0. 一句话

source behavior authority 已于 30k 结束的条件下，检验 **source replay
eligibility 应继续保留（physical tail）还是与 behavior 同步硬退出**，对
已知负迁移任务（basketball）与已知正迁移任务（truck）的 35k–80k 性能
有无因果差异。

## 1. estimand（T2 边际）

    Δ_exit(task) = J(B_{>30k}=0, R_{>30k}=0) − J(B_{>30k}=0, R_{>30k}=physical tail)

J = 35k–80k normalized trapezoidal AUC（§5.1）。两臂 B(t) 相同（都在
30k 结束行为），唯一操纵变量 = 30k 后 R(t)。只隔离 T2 硬退出边际效果。

## 2. 实验矩阵（9 × 100k）

| # | 任务 | 臂 | 来源 | seeds |
|---|---|---|---|---|
| 1 | basketball | retention | 复用 `adaptive_admission_v1/basketball_static_s1-3` | 1,2,3 |
| 2 | basketball | hard-exit | 新跑（HEAD+schedule） | 1,2,3 |
| 3 | truck | retention | 新跑（HEAD+schedule） | 1,2,3 |
| 4 | truck | hard-exit | 新跑（HEAD+schedule） | 1,2,3 |

新跑 9 条（basketball hard-exit 3 + truck 两臂 6）；basketball retention
复用（REUSE_PASS）。门 A 失败则 basketball 也两臂新跑 → 12 条。

## 3. 臂设计（schedule 路径）

- **retention**：`schedule=(0, admit-all)`，无后续撤销；30k 由
  `mcg_warmup_steps` 结束触发 `physical_after_authority`（R 拖尾，随
  buffer 覆盖至 81.2k 归零）；
- **hard-exit**：`schedule=(0, admit-all) → (30000, exact abstention)`
  （30k 时 R 归零，provenance 排除，物理轨迹保留）。

truck 两臂均如此 → 0–30k 逐位同分支；basketball hard-exit 如此，其
retention 是历史 all（0–30k 可比性由门 A 保证）。

## 4. 训练前置门（CPU，无长程训练）

### 门 A：basketball 0–30k CPU 语义等价（有限 harness，不真跑 30k）

basketball retention=历史 all、hard-exit=新 schedule，0–30k 路径不同。
因 30k 前 schedule 无新事件（唯一事件在 step 0），等价证明 = **短
trace 动态比较 + 静态 `snapshot_at` 证明**，不需真跑 30k：

- **静态**：`schedule.snapshot_at(0) / snapshot_at(25) / snapshot_at(29999)`
  与历史 all 的 admission snapshot 逐位相同（admitted mask / source_logits
  / student_logit / candidate masses），证明 30k 前配额恒定不变；
- **动态 harness**（80–100 步、少量 env、同进程顺序、CPU、fp32）：复用
  P0/no-trigger 等价范式，跨 **≥3 个 25-step segment 边界 + 强制 ≥1 次
  done/reset**，逐步断言 `admission_mode=all` 与 `schedule step0 admit-all`
  全部一致——candidate 概率、source 选择、MCG 锁存（current/steps_left/
  current_arm）、**student 与最终组合 action**、**reward/done/transition
  checksum**、8 字段 provenance、**critic/actor replay indices**、replay
  采样语义、**全局 RNG 与 MCG named generator 终态**、**actor/qnet/
  optimizer CPU 逐位终态**、admission history 除 `mode/decision_id`
  元数据外无语义差异。

全通过 = basketball 复用成立（等价链：历史 all @ a5cec9d ≡ HEAD all
〔审计 §2〕≡ HEAD schedule-step0-admit-all〔门 A〕）；任一项不过 →
basketball 退化两臂全新跑（12 条），报 PI/复核。

### 门 B：双臂 treatment audit（数学化；退出**与**保留都要证）

**取证来源更正**（我 v0.2 的对抗发现有误）：`execution_counts_at_apply`
**确实存在**，只是在 `decision_history`（train_ptf 侧）而非 `policy_events`
（replay 侧，那里是 `sample_counts_at_apply`）——两个计数位于不同审计
对象。故门 B 用**混合取证**：

- hard-exit execution：30k schedule decision 的 `execution_counts_at_apply`；
- hard-exit critic：30k replay policy event 的 `sample_counts_at_apply`；
  二者与 final audit 作差；
- retention：显式 completed-step checkpoint（§6）的 `admission_audit` 差分；
- basketball 历史 retention：仅有 30k/60k/90k/final checkpoint，用这四点
  并**披露其旧 save_interval 保存语义**（不对称在结果声明）。

定义（`C^main`=main_buffer_counts，`C^critic`=critic_sample_counts；
下标 source=源列之和 / all=全列之和；k=checkpoint 步）：

    物理占比       q_k    = ΣC^main_{source,k} / ΣC^main_{all,k}
    区间 critic 占比 r_{a:b} = ΔC^critic_source / ΔC^critic_all（a→b 差分）

**冻结断言**：

**容差已预冻结**（`scripts/analysis/p1_gate_b_tolerance.py` →
`gate_b_tolerance_candidate.json`；历史 retention 同配置臂 3 seeds ×
30k/60k/90k/final 实测 + 统计推导，E16 合规——**在任何新跑结果之前**确定）：

**ε（统计式，非工程余量）**——二十二次复核更正：旧版
`观测越界×1.5 + 0.01` 的 0.01 是硬编码常数，却被误称"采样地板"，已废弃。
现按 Hoeffding 两侧界推导：

    ε(N) = sqrt( ln(2M/α) / (2N) )
    M=24（2 task × 3 seed × 4 区间）、α=0.001、
    N=655,360,000（**新跑臂**最小区间 10k 步 × num_updates 2 × batch 32768）
    → ε_raw = 9.068e-05  →  **ε_frozen = 0.001**（向上取整）

（N 必须取新跑臂最小区间——ε 用于判定新跑臂；历史臂最小区间 1.97e9 仅
作参照记录。）

**剂量带 = 历史 min/max ± 0.02，性质为"预注册工程风险预算"**（诚实标注，
**不是**机制仿真推导）：

| 任务 | behavior share@30k | critic share@30k |
|---|---|---|
| basketball | [0.4696, 0.5110] | [0.4773, 0.5174] |
| truck | [0.4795, 0.5210] | [0.4791, 0.5192] |

0–30k 剂量匹配（脚本 `load_verified` 逐 checkpoint 强制）：
- **完整 candidate_masses 向量**跨全部 checkpoint + seed 逐位一致（不只
  比长度与 source 总和）；两臂之间亦须逐位相同；
- 机制配置逐项相等：`mcg=True` / `mcg_warmup_mode=admission_bootstrap` /
  `mcg_ablation=bootstrap_only` / `mcg_warmup_min_steps=25` / replay
  `recency=0, uniform_mix=1.0, priority=0` / task-specific
  `admission_student_logit`（basketball 3.5892126423877646、truck
  14.216676716804526）；
- 训练规模逐项相等：`num_envs=128 / batch=32768 / buffer=51200 /
  num_updates=2 / learning_starts=10 / total_timesteps=100000 /
  eval_interval=5000`；
- **conditional 字段**（`admission_adaptive` 等 5 项仅 2026-07-14 版实现
  存在，truck 的 7-13 版无此字段）：存在则必须相等，缺失则在
  `input_identities.conditional_cfg` 中如实记为 `absent (field did not
  exist in this implementation vintage)`——**不得假装验证过**；
- behavior / critic source share@30k 落在上表任务级容差内。

retention（历史实测已验证全部成立，见下表）：
- 每 checkpoint `active_buffer_counts == main_buffer_counts`（无逻辑排除）；
- q_30 ≥ q_60 ≥ q_80 ≥ q_90 = q_100 = 0（单调衰减，81.2k 后归零）；
- 区间 critic 占比落在物理占比端点包络内：`q_b − ε ≤ r_{a:b} ≤ q_a + ε`。

历史 retention 实测参照（包络越界观测 = **0.0**，即区间 critic 占比完全
落在物理占比端点包络内）。**注意 80k 口径**：历史臂只有 30k/60k/90k/final
checkpoint，**无 80k**，故历史侧 80k 记 `NA`，不构造该数据点；q 单调链
在历史侧按 q_30 ≥ q_60 ≥ q_90 = q_final = 0 验证，新跑臂才按
30/60/80/90/100k 全链断言：

| 任务 | q_30 | q_60 | q_80 | q_90 | q_final | r_{30:60} | r_{60:90} | active==main |
|---|---|---|---|---|---|---|---|---|
| basketball | 0.490 | 0.202 | NA | 0.000 | 0.000 | 0.334 | 0.071 | ✓ |
| truck | 0.500 | 0.207 | NA | 0.000 | 0.000 | 0.341 | 0.073 | ✓ |

hard-exit：
- 30k 后 source execution 与 critic 增量**严格为零**；
- 60k/80k：允许 `main source > 0` 但 `active source = 0`（逻辑退出 ≠
  物理删除）；
- **90k/100k：source 物理数据本已覆盖为零，不再要求 `main source > 0`**。

任一断言不满足 → 该 run ENGINEERING_INVALID，不进裁决。

## 5. 评估协议与指标

### 5.1 主/次指标（训练内 eval 曲线，两臂协议一致）

eval 由 train_ptf 内置 `evaluate()` 产生，`eval_interval=5000`，网格
**5k–95k 共 19 点**（复用臂与新跑臂同协议，审计 §3 确认零改动）。
buffer=51.2k、source 只在 0–30k 写入 → retention 旧 source 数据至 81.2k
完全覆盖，此后两臂无 treatment 差异。

- **主指标** J_s = normalized trapezoidal AUC over 35k–80k：
  `J_s = trapz(R_{35k:80k}, t) / (80000 − 35000)`（10 点，单位=return）；
- **次指标**：80k–95k persistence nAUC（同式，4 点：80/85/90/95k）；
- **终点**：**95k** source-free 性能（两臂在 81.2k 后均 source-free，95k
  是训练内 eval 最后一点，协议一致）。**删除 100k endpoint**——内置
  `evaluate()` 最后只产生 95k（循环 `while step<100000`，95k 后不再
  eval），100k 需离线 evaluator+额外兼容性审计，对 35k–80k T2 estimand
  非关键，故不设。

### 5.2 机制中介（门 B 用，非性能）

显式 completed-step checkpoint 30k/60k/80k/90k/100k 的 admission_audit：
source physical share、source critic share（两臂衰减曲线）。

## 6. 冻结参数（写入每 run meta；E17 合规）

| 项 | 值 |
|---|---|
| total_timesteps | 100000 |
| num_envs / batch_size / buffer_size | 128 / 32768 / 51200 |
| learning_starts / num_updates | 10 / 2 |
| mcg_warmup_steps / mode / ablation | 30000 / admission_bootstrap / bootstrap_only |
| mcg_groups | [legs_torso, arms] |
| admission_replay_handoff | physical_after_authority |
| replay recency / uniform_mix / priority | 0.0 / 1.0 / 0.0 |
| eval_interval | 5000 |
| **save_interval** | **0**（禁用普通保存；否则其 `global_step+=1` 前的 off-by-one 文件会与显式 checkpoint 同名覆盖） |
| ptf_eval_checkpoint_steps（新跑臂机制快照，**逗号分隔**） | `"30000,60000,80000,90000,100000"` |
| seeds | [1, 2, 3] |
| truck bank / student_logit | `h1hand_hurdle4_wfix_truck.yaml`（stand/walk/run/hurdle）/ 14.216676716804526 |
| basketball bank / student_logit | `h1hand_std9_wfix_basketball.yaml`（9 源）/ 3.5892126423877646 |

剂量 = warmup 内源 aggregate mass ≈0.5（由 student_logit 决定，对齐历史
retention 臂）。每 run meta 记：per-file implementation SHA 清单 +
`base_git_head + dirty 状态`（E17）；**在干净 commit 上执行**。

## 7. 预注册判序（精确公式）

### 7.1 per-task Δ_exit 分级（paired，df=2）

每 seed d_s = J_{hard,s} − J_{retention,s}；均值 d̄，SD(d_s)；90% paired CI：

    CI = d̄ ± 1.8856 · SD(d_s) / √3

冻结 δ_task（§7.3）后分级（判定顺序）：

1. `HETEROGENEOUS`：同时存在 d_s > δ 与 d_s < −δ；
2. `IMPROVEMENT`：LCB > δ；
3. `HARM`：UCB < −δ；
4. `EQUIVALENT`：整个 CI ⊂ [−δ, δ]；
5. 其余 → `UNCERTAIN`。

**无 duplicate 数值地板**，故 `UNCERTAIN` 只能表述为"3 seeds 下统计
不确定"，**不得**称"数值不可测"（与 P0 的 UNCERTAIN_NUMERIC 区分）。

### 7.2 层级联合裁决

1. 工程有效性：门 A（basketball）+ 门 B（双臂）不过 → ENGINEERING_INVALID；
2. 可测性：任一任务 UNCERTAIN/HETEROGENEOUS → 该任务不进方向结论；
3. basketball 方向（§7.1）；4. truck 方向（§7.1）；
5. 与 scratch 比较的限定标签（§7.4）。

**双任务 EQUIVALENT 的归因边界**：只能结论"30k 后 source replay 物理
拖尾不是该配置下主要可干预因果通道"——残余负迁移可能位于 0–30k 已形成
的 critic/actor/optimizer 状态、source 诱导 occupancy、早期数据覆盖或
其他未分离通道，**不得**收窄归因到 learner state。

### 7.3 δ 的定位与候选值

**定位（二十一次复核，务必照此表述）**：δ_task 是从历史、同任务、同
return 口径的 scratch 曲线得到的 **externally anchored SESOI /
practical margin**（最小关切效应量）——**不是**当前实现的数值噪声地板，
**也不是**新实验的方差估计。依据=scratch 的
`METRIC_SCALE_REUSE_PASS`（兼容性审计 §8.3）。

δ_task = 0.5 × (scratch 35k–80k nAUC 的跨 3-seed SD)，两任务分别。

**候选已生成**（`scripts/analysis/p1_freeze_delta.py` →
`docs/data/p1_bounded_bank_lease/delta_candidate.json`，`status=candidate`）：

| 任务 | 跨 seed SD | **δ (SESOI)** |
|---|---|---|
| basketball | 91.573 | **45.786** |
| truck | 73.037 | **36.518** |

（与 ChatGPT 独立复算一致。）JSON 内绑定的 provenance：每条 scratch 的
训练日志 SHA256 + W&B `config.yaml` / `wandb-metadata.json` / 历史入口
`train_ptf.py` / `diff.patch` / `output.log` 的 SHA256，并断言"六份入口
代码 SHA 全同、git base 全同、CLI 纯 scratch、W&B 与 logs 曲线逐条
相同"。**正式冻结（candidate → frozen）待 PI 批准**。

### 7.4 scratch 限定标签（**描述性——不进正式判序**）

**scratch 兼容性分层裁定（兼容性审计 §8.3）**：
`CAUSAL_COMPARATOR_REUSE_FAIL`（scratch @ b183f40 ≠ hard-exit @ HEAD，
不能承担严格配对因果标签）+ `METRIC_SCALE_REUSE_PASS`（evaluator/任务/
return 口径/5k 网格/入口代码全可考，SD 可作外部尺度→§7.3 的 δ）。故层-5
scratch 比较**保持描述性，不进正式判序**；δ 的外部尺度用途不受影响。
核心 Δ_exit=hard-exit − retention 是配对差，不依赖 scratch，主判序
（层 3/4）自洽。

描述性标签公式（仅供背景解读，e_s = J_{hard,s} − J_{scratch,s}，90% CI
同式；因 scratch 不可考，任何标签均加"描述性"前缀）：

- `POSITIVE_GAIN_RETAINED`：LCB(e) > δ；
- `NEGATIVE_TRANSFER_AVOIDED`：CI(e) ⊂ [−δ, δ]；
- `MITIGATION_ONLY`：Δ_exit 显著 IMPROVEMENT，但 UCB(e) < −δ；
- `POSITIVE_GAIN_LOST`（**收紧**）：须同时满足 (a) retention 显著优于
  scratch（LCB of J_ret−J_scr > δ）**且** (b) hard-exit 相对 scratch
  明确为 EQUIVALENT 或 HARM（CI(e) ⊂ [−δ,δ] 或 UCB(e) < −δ）；
- `SCRATCH_COMPARISON_UNCERTAIN`（**新增**）：hard-exit 相对 scratch 落
  UNCERTAIN 时的兜底，不得标为 `POSITIVE_GAIN_LOST`。

## 8. 预算与执行

- 新跑 9 × 100k，单卡 SPS≈21 → ≈1.4h/run + 启动 ≈13–14 GPU-hours；
- 执行前 throughput smoke 确认 SPS/RSS（双并发需 <节点 RAM 上限，E2）；
  设计目标 ≤24 GPU-hours，>48 重批；
- tmux + PYTHONUNBUFFERED + tee + wandb 在线 + `WANDB_INIT_TIMEOUT=300`；
  负载看门同 P0（load<165 连续 2 次 + 空闲 GPU）；失败即停、保留现场、
  不自动重跑。

## 9. 待决

**已完成（零训练）**：
- ~~scratch 兼容性审计~~ → 双层裁定（审计 §8）；
- ~~门 B 的 ε 与 0–30k 剂量容差~~ → 已由历史 retention 臂实测预冻结
  （§4 门 B 表；ε_raw=9.068e-05 → **ε_frozen=0.001**，包络越界观测 0.0）；
- ~~δ 冻结脚本 + 候选值~~ → basketball 45.786 / truck 36.518，含 W&B
  provenance 绑定（§7.3，`status=candidate`）。

**二十三次复核修订（v0.6）**：finalize 改完整科学 payload 逐位比较；门 B
补全同配置验证（机制配置/训练规模/student_logit/masses 向量跨 ckpt+seed
一致；conditional 字段"存在则验、缺失如实记"）；反例单测 52 项。

**二十二次复核四阻塞已修（v0.5）**：
- ε 改统计式（Hoeffding，见 §4 门 B）；剂量带诚实标注为工程风险预算；
- 两脚本补输入身份验证（24 checkpoint 逐项核对 env/seed/step/bank/
  admission_mode/warmup/handoff/masses + SHA256 入 JSON）、generator SHA、
  git HEAD + dirty；
- δ 脚本补自动断言：wandb run **恰好匹配一个**、六份 patch 剔除离线
  probe 段后逐位一致（不再是硬编码文本）、历史 PTF 配置实为
  `mcg=False/execute_sources=False/bank 空`；
- 两脚本补 `--finalize --candidate --expected-candidate-sha256 --out`
  安全冻结流程（重算比对 + 干净树 + 拒绝覆盖）；
- 反例单测 `tests/test_p1_freeze.py`（26 项，全仓 184 passed）；
- 80k 口径分离（历史 NA / 新跑臂全链）。

**已完成（PI 批准后执行）**：
1. δ 与门 B candidate→frozen；
2. 门 A CPU 等价测试 PASS；
3. 200-step GPU smoke PASS；
4. 正式矩阵 `20260719T130500Z`：9/9 × 100k 全部退出码 0；
5. Gate B：9/9 新跑工程有效；
6. 冻结主判序：basketball=`HETEROGENEOUS`，truck=`HETEROGENEOUS`。

## 10. 最终裁决（2026-07-20）

主结果见 `docs/phase1_bounded_bank_lease_results_20260720.md`，机器可读数据见
`docs/data/p1_bounded_bank_lease/final_result.json`。T2 hard-exit 的 replay
生命周期语义得到完整验证，但两个任务均未形成方向稳定的性能效果。该机制
保留为风险控制/正确性组件，不升级为性能贡献；本轮关闭，不做事后调参补救。
