# Run Card：P0 — SIV-v2 Counterfactual Lease Oracle 最终可行性审计

> 版本：**v2.1.3**（2026-07-17；九次代码审查后的证据链与执行安全收口；
> **estimand/任务/L/dose/P0 阈值一律不变**）
> 状态：**实现与本地审查已收口 → throughput smoke（仍须另行授权）→
> 冻结 δ/commit/plan → PI 最终批准 → 正式 P0**
> 授权依据：PI 2026-07-16 裁决（编写 run card）+ 2026-07-17 转发采纳的实现
> 阶段授权（实现+本地等价性测试；**不包括 throughput smoke 与正式 P0 训练**）。
> 当前未生成可执行的正式冻结 plan、未跑 smoke、未启动任何 P0 训练。

## v2.1.1 → v2.1.2 变更记录（四次复核意见，全部核实后采纳）

| # | v2.1.1 缺陷（已核实） | v2.1.2 修正 |
|---|---|---|
| 1 | d_dup 判序漏洞：d_dup≥δ 藏在第 4 步 UNCERTAIN"其余"里，前面命中 POSITIVE/NEGATIVE/NULL 即短路——数值分辨率门覆盖不到已被分类的结果 | per-task 判序改为 **d_dup≥δ 前置**：(1) d_dup≥δ → UNCERTAIN_NUMERIC；(2) HETEROGENEOUS；(3) POSITIVE/NEGATIVE；(4) NULL；(5) 其余 → UNCERTAIN |
| 2 | §11 行为占比误用 `admission_sample_counts`（那是 replay 采样计数） | 行为执行占比读 `admission_execution_counts`（train_ptf.py:1959 逐步 bincount）；critic replay 占比读 `critic_sample_counts`（ptf_replay.py:580） |
| 3 | noise_scales 重采样的 RNG 来源未定义（直接 manual_seed 会污染刚恢复的 anchor 全局 RNG）；"先重绑 actor_detach、后重采样"顺序会让双 actor 的 buffer 再次不一致 | 重采样必须用**独立 torch.Generator**（配对 noise seed 播种，不触碰全局 RNG）；同一份采样结果**同时写入 actor 与 actor_detach** 的 noise_scales，随后断言逐位一致 |
| 4 | "同进程天然保证逐位确定性"表述过强 | 改为：同进程消除进程级差异源，逐位对照仍依赖**冻结的 determinism 配置 + checksum 判定**；checksum 分叉即测试 FAIL，须诊断，不得静默放宽为统计对照 |

## v2.1 → v2.1.1 变更记录（ChatGPT 三次复核意见，全部代码级核实后采纳）

| # | v2.1 缺陷（已核实） | v2.1.1 修正 |
|---|---|---|
| 1 | §7 联合裁决表逻辑矛盾：第 2 层定义双 NULL 通过 measurement gate，第 3 层表格却把双 NULL 写在"measurement 不过"行 | **六步固定判序**（§7 重写）：工程无效 → HETEROGENEOUS(F-d) → UNCERTAIN(F-a) → 双 NULL(F-b/LOCAL_NULL，语义="测得效应≈0"而非"测不出"，属 measurement 通过) → crawl=NEG∧truck=POS(PASS) → 其余已确定组合(SURROGATE_FAIL) |
| 2 | 等价性测试 4 参照对象错误：abstain 分支 reset 环境后 occupancy 与"不间断 scratch"（10k 时 episode 中间）必然不同，strata 逐位对照字面执行必 FAIL；no-trigger smoke 只证控制器未触发的局部一致性，不证 anchor 恢复等价 | 测试 4 重写（A.3）：**同一 anchor、双 reset、bank+admission none vs 空 bank**，同进程顺序执行（E10：跨进程 CUDA 非确定性下逐位对照只在同进程内成立），核对动作 checksum/transition checksum/provenance/replay 抽样索引/终态 RNG；另增 round-trip 加载测试 |
| 3 | reset-start 下 RNG 与 episode 状态定义不完整：anchor 保存含 python RNG+named generators（anchor_io.py:90），白名单漏写；`noise_scales` 是 episode 级状态（Actor.explore 只在 done 时重采样，fast_td3.py:202），reset 后沿用 anchor 中上一 episode 的值语义错误 | A.1/A.2 补全：**加载顺序清单**（核心组件→重建 option/bank/admission→重绑 actor_detach→**最后**恢复全局 RNG，与 anchor_io restore 语义一致）；named option-selector generator **不从 anchor 恢复**、按分支配对 seed 面板重新播种；fresh reset 后按配对 noise seed **显式重采样 noise_scales**；强制冻结 reward_normalization=False（h1hand 历史默认即 False，normalizer=nn.Identity 无状态）；删除"两分支保持相同 exploration noise 序列"声称——序列分叉属 estimand |
| 4 | provenance 全量重写为 student 会破坏合法元数据（segment_id/env_rank/learner_step，ptf_replay.py:62） | 优先方案：**anchor 训练按任务目标组数配置 provenance schema**（crawl=3/truck=2，空 bank 下仅影响列数，分支零重建原样导入）；fallback：仅重建组维度字段（behavior_source/source_by_group/executed_group_mask），标量元数据字段原样保留 |
| 5 | δ_progress_truck 无可追溯历史数据（wandb 无 per-component reward 历史曲线） | **truck progress 降级为描述性指标**；confirmatory 同向性检查仅对 crawl 执行（0.5 m 绝对阈有物理意义）；truck 的 concordance 主张由 primary return 独立承载（handoff +227.8 本就是 return 尺度） |
| 6 | student_logit 运行时为 float32（admission_control.py:89），"运行时 float64 精确相等"声称不成立 | float64 解析值保留为配置输入；新增**启动时运行时断言 \|source_mass − 0.10\| ≤ 1e-6**（float32 下理论偏差 ~2e-7，门限富余） |
| 7 | σ_dup 只有一次配对差，不是标准差；固定 save_interval 无法覆盖 10750（train_ptf.py:2201，10750 不整除 750） | 改名 **d_dup**（repeatability discrepancy）；新增 `--ptf-eval-checkpoint-steps` 显式列表机制 |
| 8 | 训练参数与 treatment 保真缺审计 | §11 新增冻结参数表（AMP/compile/determinism/obs+reward normalization）与 treatment 运行时审计（lease 分支 source 行为占比≈0.10、critic batch source 配额占比、abstain 分支双零、eval source-free 断言） |

## v2 → v2.1 变更记录（ChatGPT 二次复核意见，全部代码级核实后采纳）

| # | v2 缺陷（已核实） | v2.1 修正 |
|---|---|---|
| 1 | anchor-resume 非"薄接线"：anchor（空 bank）保存 option 族组件，与挂 bank 分支的组件集/维度不匹配（anchor_io 严格加载要求集合精确一致）；T_max=args.total_timesteps 会把 LR 日程压成 13k；actor_detach 需与加载后权重重新绑定；provenance 按 3 组保存 vs truck 2 组 schema 冲突 | **core-only resume 设计**（附录 A）：只加载核心 learner 组件；option 族按分支配置重新初始化（bootstrap_only+gate 关下 option 不参与任何行为/loss，已验证语义）；保留 total_timesteps=100000（LR 日程不变）+ 新增独立 `run_stop_step=13000`；resume 后 actor_detach 重绑定 + 六项等价性测试；replay provenance 按运行时 group 数重建（anchor 数据全 student 常量填充） |
| 2 | treatment 不只是"10% source 行为"：admission 启用后 replay 走 provenance 配额，control 空 bank 走 uniform——被测的是完整 lease 机制（behavior+replay quota+occupancy） | **estimand 据实改写**（§1）：U 定义为完整 lease 机制效应；control 改为**同 bank + admission_mode=none**（exact abstention，已验证逐位回 scratch 分布），两分支组件对称；replay 参数冻结中性化：recency_half_life=0、uniform_mix=1.0、priority_alpha=0、physical_after_authority |
| 3 | fork 从新 reset 开始，非当前 occupancy 条件化（anchor 不存 simulator state） | 采用低成本选项：estimand 改名 **U^reset**（reset-start, learner-stage-conditioned lease utility），论文措辞同步限定；不重建 simulator checkpoint |
| 4 | action authority 与历史标签不匹配：truck +227.8 用 [legs_torso, arms]（hands=student），crawl 旧 WFix 用 3 组全 61 维 | **按任务复现各自历史 composition**（保持 matched label）：crawl=3 组、truck=[legs_torso, arms] |
| 5 | 内置 evaluate() 复用训练环境（eval_envs=envs）、无固定 eval seed、会打断训练 episode；crawl 是 1000 步（v2 误写 500） | **独立冻结 evaluator**（§6）：分支训练 eval_interval=0 不打断；存中间 checkpoint，事后离线评估（单环境顺序逐 episode——实现定稿见 §6、4 eval seeds×8 ranks 同面板、deterministic actor、1000 步） |
| 6 | \|mean U\|>2σ_dup 误置工程层（真零效应合法）；duplicate 换 noise seed 测的是算法随机性非数值地板；五分类可重叠、阈值边界未定义 | σ_dup 条件移入 measurement gate；**duplicate=完全相同 seeds 仅重启进程**（测 CUDA/进程 repeatability；算法随机性由 3 seeds 捕获）；分类优先级冻结 + 边界归 UNCERTAIN + t₀.₉₀,₂=1.8856 写死 |
| 7 | 历史方向被当作局部效应的 ground truth | **拆为 measurement gate（内部有效性）与 historical-concordance gate（外部一致性）**；方向不一致的正确结论="局部 lease utility 不是完整训练收益的可靠 surrogate"（SURROGATE_FAIL），非"oracle 测错" |

（v1→v2 变更记录见 git 历史 9858263；§0 限界条款自 v1 起未变。）

---

## 0. 路线重启声明与限界条款（不变）

U 是旧 SIV 总干预效应 T 的 lease-scale 延伸；执行即一次性推翻 v3 停止决定；
只测 counterfactual oracle；D3/gradient influence 封存；无在线 controller；
**失败后不得更换 task/L/dose/threshold 挽救**；Phase-1 不依赖本结果。

## 1. Estimand（据实改写）

```
U^reset_bank(t=10k, L=3000) = E_seeds[ J_sf(lease branch) − J_sf(abstain branch) ]
```

- **被测 treatment = 完整 lease 机制**：source behavior authority（mass 0.10）
  + admission-consistent replay 配额 + source-induced occupancy——正是 Phase-2
  一次 lease 决策的全部效应；不声称"纯行为差异"；
- **reset-start 限定**：分支从相同 reset seed 面板重新开始（anchor 不含
  simulator state），estimand 是 learner-stage-conditioned、非
  occupancy-conditioned——论文措辞受此约束；
- K=6000 critic updates 为 3000 步在线训练的自然产物。

## 2. Lease 长度（不变）

L=3000 = 预先固定的工程决策周期（30k warmup 容纳 ~10 次决策）。

## 3. 分支协议

```
anchor（每任务×seed，共 6）:
  空 bank scratch 训练至 10000 步，--ptf-anchor-step 10000 存 bundle
  （learner+optimizer+scheduler+replay+RNG）

lease branch（source）:
  core-only resume（附录 A）→ 挂任务 bank，admission_mode=all，
  admission_bootstrap，student_logit 精确校准（τ=1，float64）:
    crawl: 16.6823567039（源总 mass=0.10，已独立验算）
    truck: 16.4139012941（同上）
  mcg_groups: crawl=[legs_torso, arms, hands]（=历史 WFix 全 61 维）,
              truck=[legs_torso, arms]（=handoff +227.8 的历史配置）
  h=25 锁存，bootstrap_only，gate 关，无蒸馏
  replay 冻结: recency_half_life=0 / uniform_mix=1.0 / priority_alpha=0 /
              physical_after_authority
  reward_normalization=False 强制冻结（h1hand 历史默认即 False）
  训练 10000→13000（total_timesteps 保持 100000；run_stop_step=13000）

abstain branch（control）:
  同一 bundle core-only resume → 挂同一 bank，admission_mode=none
  （exact abstention：100% student、零 source RNG 消耗、已验证逐位回
  scratch 分布），其余与 lease branch 逐项一致

配对: 两分支相同 env reset seed 面板；resume+reset 后按相同配对 noise seed
     显式重采样 noise_scales（初始 exploration 状态逐位配对）。
     **不声称后续 noise 序列相同**——done 时序与 RNG 消耗路径在 source
     执行后必然分叉，分叉属 estimand 本体。

checkpoints: 10750/11500/12250/13000 四点，经新增
     `--ptf-eval-checkpoint-steps` 显式列表保存（固定 save_interval 无法
     覆盖 10750；训练中 eval_interval=0，不打断）。
```

## 4. 实验矩阵与标签

| cell | 任务 | bank | 先验方向假设（static 证据；语义=0-30k 完整 exposure 总效应） | seeds | 分支 |
|---|---|---|---|---|---|
| C1 | crawl | loco | negative（terrain 三方翻转 + wfix 3-seed 静态对照） | s1-s3 | lease+abstain |
| C2 | truck | h4 | positive（handoff fix−scr +227.8，t=4.74，matched 2-group authority） | s1-s3 | lease+abstain |
| N1/N2 | 各任务 s1 | — | duplicate=abstain branch **完全相同 seeds 重启进程**（CUDA/进程 repeatability 地板） | s1 | duplicate |

14 分支 + 6 anchor。标签为**外部一致性参照**（historical-concordance gate 用），
不是局部效应的 ground truth（§7）。

## 5. Gate 统计（冻结形式）

- 配对效应 U_s = J_lease,s − J_abstain,s（s=1,2,3）；
- one-sided Student-t，df=2，**t₀.₉₀,₂ = 1.8856**：
  bound = mean(U_s) ± 1.8856·SD(U_s)/√3；
- **d_dup**（repeatability discrepancy）= duplicate 配对差 |Δ|——单次重复
  差值，**不是标准差**，故不称 σ；进入 measurement gate 的分辨率条件，
  **不在工程有效性层**——真零效应是合法科学结果；
- **student_logit 运行时验证**：float64 解析值（§3）仅为配置输入；启动时以
  实际 float32 路径（admission_control.candidate_probabilities）计算源总
  mass 并断言 **|mass − 0.10| ≤ 1e-6**（float32 下理论偏差 ~2e-7），
  不声称运行时精确相等；
- δ 按任务×指标执行前冻结（数据源=既有 scr/wfix 历史 eval 曲线 10k-15k 窗口，
  非 P0 数据；精确数值写死进冻结的分析脚本）：
  δ_return_task = 0.5×历史 scratch 跨 seed SD；δ_progress_crawl = 0.5 m
  （root-x 位移绝对阈）；
- primary = source-free mean return；progress confirmatory **仅 crawl**：
  **同向性 = paired mean progress delta 与 return 同号且 |delta| > 0.5 m**；
  **truck progress 为描述性指标**（`reward_robot_package_truck` 无可追溯
  历史曲线，无法预冻结 δ；truck 的 concordance 主张由 primary return 独立
  承载——历史标签 +227.8 本就是 return 尺度）。

## 6. 独立 evaluator（冻结协议）

- 独立脚本（不触碰训练环境）；**执行形式=单环境顺序逐 episode**（v2.1.2
  实现定稿，六次复核确认可复现；早期草案写 SubprocVecEnv，已按实际实现
  统一）；每 checkpoint：**4 eval seeds × 8 ranks = 32 episodes**，
  lease/abstain 分支使用完全相同的 (seed, rank) 面板（reset seed =
  eval_seed×1000+rank，双播种，E11 语义）；
  deterministic source-free actor（结构性 source-free：evaluator 不构建
  bank/option/admission 组件）；episode 1000 步（crawl/truck 均为标准长度）；
  **身份验证**：正式评估必须传 --expect-global-step/--expect-seed/
  --expect-admission-mode，与 checkpoint 内容不符即拒绝；
- 指标（精确定义）：
  - return：episode return 均值；
  - crawl progress（confirmatory）：`max_t(x_t − x_0)` 的 episode 均值
    （x=root x 坐标）；
  - crawl posture（描述性）：crawling∧crawling_head tolerance 的 episode mean；
  - truck progress（**描述性**，§5）：`reward_robot_package_truck` 的
    episode **mean**（reducer 冻结）；
  - truck termination（描述性）：**语义=任务成功**（packages 全上桌，非 fall），
    单独报告，预期恒 0；
- primary endpoint 固定 checkpoint@13000；中间点仅作显形曲线描述。

## 7. 层级裁决（优先级冻结，互斥完备）

**第 0 层 ENGINEERING_VALID**：14/14 exit=0、resume 等价性测试通过（附录 A）、
配对 seed 面板核对一致。失败 → ENGINEERING_INVALID（修复后重跑同配置，
不算改参）。

**第 1 层 per-task 分类**（primary return @13000；优先级自上而下，先命中先判；
所有恰好等于阈值的边界情形一律归 UNCERTAIN）：

1. **UNCERTAIN_NUMERIC：d_dup ≥ δ（本任务的 duplicate 重复性差异不低于
   效应阈——数值分辨率门前置，任何后续分类在该条件下都不可信）**；
2. HETEROGENEOUS：存在 |U_s|>δ 且符号相反的 seed 对；
3. POSITIVE：LCB > +δ 且（crawl 时）progress 同向；NEGATIVE：UCB < −δ 且
   （crawl 时）progress 同向（truck 无 confirmatory progress，§5）；
4. NULL：CI ⊂ (−δ, +δ)；
5. UNCERTAIN：其余（含 CI 跨界、crawl progress 与 return 反向）。

UNCERTAIN_NUMERIC 是 UNCERTAIN 的子类：进入第 2 层步 3（F-a），
标注原因=数值地板。

**第 2 层 joint 判序（六步固定顺序，自上而下先命中先判，互斥完备）**：

| 步 | 条件 | 裁决 |
|---|---|---|
| 1 | 第 0 层失败 | ENGINEERING_INVALID（修复后重跑同配置，不算改参） |
| 2 | 任一任务 HETEROGENEOUS | **F-d 不稳定**："无可部署统一判据；封存" |
| 3 | 任一任务 UNCERTAIN | **F-a 统计不可测**："本预算分辨率不足；判据封存"（标注具体原因：d_dup 主导 / CI 过宽 / progress 冲突） |
| 4 | 双 NULL | **F-b / LOCAL_NULL**："实验**可测**（measurement 通过），(η=0.1, L=3000) 局部效应≈0；封存，提示长时程累积效应"——语义是"测得效应接近零"，**不是**"测不出来" |
| 5 | crawl=NEGATIVE 且 truck=POSITIVE | **P0 PASS** → 进入 P1' 讨论 |
| 6 | 其余已确定组合（含单 NULL 单方向、双方向错向） | **SURROGATE_FAIL**："局部 lease utility 可测，但不是完整训练收益的可靠 surrogate；lease 判据封存（Phase-2 依赖该 surrogate 性），此为 estimand 分离的强科学结果，入双通道证据链" |

与双 gate 语义的映射（叙事用，判决以上表字面为准）：measurement gate=
步 1-3 全通过（oracle 可测）；historical-concordance gate=步 5 命中；
步 4/6 均属"measurement 过、concordance 不过"，区别在效应为零（LOCAL_NULL）
还是方向/组合与历史不一致（SURROGATE_FAIL）。

任何非 PASS 组合 → Phase-2 不上线；输出组合码（如 `crawl_NEGATIVE__truck_NULL`）。

## 8. Deployment diagnostics（报告项，不变）

latency（一次完整决策 wallclock vs 3000 步在线时长）/ cost（占 100k 训练比例
+ (S+1) 外推）/ scalability（S=3/9 资源表）。

## 9. 预算（smoke 实测；≤24 GPU-hours 目标，>48 重批）

草估（自查）：anchor 6×1h≈6h；分支 14×(3000 步≈20min+启动)≈6h；离线 eval
**52 次**（12 正式分支×4 ckpt=48 + 2 duplicate×2 份 primary=4；duplicate 的
中间 checkpoint 不评——d_dup 只在 primary endpoint 度量）×32 episodes×
1000 步 ≈2-3h；**合计 ≈14-15h**。
smoke 协议（无结果窥视）：1 anchor 段+1 分支 200 步段+1 次 eval 面板，只读
吞吐/VRAM/RAM。预算外推采用“每作业实测启动开销 + 目标步数/SPS”，而非只用
目标步数/SPS（否则会漏计 6 个 anchor 与 14 个短分支重复支付的初始化成本）。
削减预案（duplicate 不可削）：(1) 中间 checkpoint 4→2；
(2) eval 面板 4×8→3×8；(3) 仍超 24h 报 PI；>48h 不启动。
artifact：anchor 6 bundle（数 GB/个实测）+ 分支 checkpoint/eval json，预计 <60GB。

## 10. 已声明的局限

reset-start（非 occupancy-conditioned）、单 stage/dose/L、bank 级归因、
treatment=完整 lease 机制（不可归因到行为/replay 单通道——那是旧 SIV 分解的
职责，本实验不做）、分支级 CUDA 非确定性由 duplicate 显式计量。

## 11. 冻结与审计

执行前冻结：run card + 实现 SHA + δ 文件及其 3 份历史日志 SHA + student_logit
float64 精算值 + 四组 seed 面板（anchor/branch/noise/eval）+ 分析脚本。
冻结 plan 必须同时绑定 bank YAML、source manifest、manifest 实际引用的 source
checkpoint 权重、两任务 δ 文件；任一缺失或哈希漂移均拒绝执行。每分支 meta：
anchor digest/bank SHA/配置/git SHA。所有预期产物为 one-shot evidence，启动前若
路径已存在则拒绝，防止新进程误认证旧产物。裁决按 §7 字面执行。

**冻结训练参数表**（anchor 与全部分支逐项一致，写入每 run meta）：

| 参数 | 冻结值 |
|---|---|
| reward_normalization | **False**（h1hand 历史默认；normalizer=nn.Identity 无状态） |
| obs normalization | 启用（normalizer 状态从 anchor 恢复） |
| AMP/bf16 | 按 anchor 训练配置原样冻结（GradScaler 状态在白名单内） |
| torch.compile | PTF 激活时自动关闭（E9），显式记录 |
| determinism flags | 记录 cudnn/cublas 设置；分支级非确定性由 d_dup 计量 |
| num_updates / batch / lr / γ 等 | 全部继承 anchor 配置，禁止分支间差异 |

**Treatment 运行时审计**（每分支训练结束写入 meta，作第 0 层核对项）：

1. lease 分支：source 行为步占比 ≈ 0.10（读 `admission_execution_counts`，
   train_ptf.py:1959 的逐步 bincount；公差 ±0.02——mass 是 per-step
   categorical 期望）；critic batch 的 source-provenance 配额占比 ≈ 0.10
   （读 `critic_sample_counts`，ptf_replay.py:580）；
2. abstain 分支：`admission_execution_counts` 的 source 位**严格 =0** 且
   `critic_sample_counts` 的 source 位**严格 =0**；
3. 全部离线 eval：deterministic source-free actor，断言 eval 路径不触碰
   bank/option 组件。

---

## 附录 A：anchor-resume 实现设计（core-only）

### A.1 加载集合（严格白名单）

| 加载 | 不加载（按分支配置重新初始化） |
|---|---|
| actor、qnet、qnet_target、obs/critic-obs normalizer、actor/critic optimizer、q/actor scheduler、GradScaler、replay（数据+ptr）、**global RNG 全集（python/numpy/torch_cpu/torch_cuda，anchor_io.py:90 保存的全部四类）** | option_module、option_target、option/beta optimizer（bootstrap_only+gate 关下不参与任何行为与 loss——resume 前以断言验证该语义）、MCG gating 状态、admission 状态（由分支配置全新构建）、**named option-selector generator（不从 anchor 恢复，按分支配对 seed 面板重新播种——anchor 空 bank 阶段从未消耗、无语义；abstain 分支零消耗是 exact abstention 既验证语义；lease 分支消耗路径本就与 abstain 不同）**、reward normalizer（冻结 reward_normalization=False，nn.Identity 无状态，无需恢复） |

**加载顺序（实现定稿，与代码一致——v2.1.2 六次复核后对齐）**：
1. 构建全部模块（含按分支配置新建的 option/bank/admission 组件；
   named option-selector generator 在构建时按分支 option_seed 播种）；
2. env reset（发生在训练脚本的自然位置，早于 resume 块；env 随机性由
   env seed 面板控制，与全局 RNG 隔离——E11 wrapper 双播种）；
3. `load_anchor_core()` 白名单加载核心 learner（模型/optimizer/scheduler/
   GradScaler/replay 数据+ptr），**其内部最后恢复 global RNG 四类全集**
   （`restore_global_rng_state`，"Restore RNG last" 语义）；
4. anchor/branch 冻结参数一致性断言（41 键白名单，不一致即拒绝）；
5. 重绑定 actor_detach（A.2.2）；
6. noise_scales 配对重采样（A.2.5：独立 generator，不触碰已恢复的全局
   RNG；同一份采样同时写 actor 与 actor_detach，随后逐位断言）。

实现：`--ptf-anchor-resume <bundle>` 走 `load_anchor_core()` **严格白名单
路径**：调用方必须恰好传入白名单集合（防漏加载/防越权加载），组件本身以
strict=True 加载；bundle 中白名单以外的组件（option 族等）被忽略；named
generator 一律不从 anchor 恢复；带状态 reward_normalizer 直接拒绝；
scheduler last_epoch==anchor 步数断言。global_step 恢复为 10000。
（早期草案的 "strict=False 白名单路径" 表述已废弃——实际语义是
"白名单严格、白名单外忽略"。）

### A.2 关键语义修正

1. **停止点**：`args.total_timesteps` 保持 100000（scheduler T_max 依赖它，
   :728/:733），新增 `--ptf-run-stop-step 13000` 独立控制训练循环退出；
2. **actor_detach 重绑定**：加载 actor 权重后重新执行
   `from_module(actor).data.to_module(actor_detach)`，并断言含 noise_scales
   在内的全部 buffer 逐位一致（断言时点=A.2.5 重采样**之后**）；
3. **replay provenance**（优先方案）：**anchor 训练即按任务目标组数配置
   provenance schema**（crawl=3/truck=2；空 bank 下 group 数仅影响 provenance
   列数，不影响行为）→ 分支 resume 零重建、原样导入，segment_id/env_rank/
   learner_step 等标量元数据（ptf_replay.py:62）天然保留；
   **fallback**（若空 bank 训练无法配置组数）：仅重建组维度字段
   （behavior_source/source_by_group/executed_group_mask，anchor 段全 student
   填充），**标量元数据字段原样保留，禁止常量覆盖**；
4. **scheduler/optimizer step 恢复**：从 bundle 恢复后断言 LR 与 anchor 保存
   时刻逐位一致；
5. **noise_scales 配对重采样**：`noise_scales` 是 episode 级状态
   （Actor.explore 只在 done 时重采样，fast_td3.py:202）；分支 fresh reset 后
   **不得沿用 anchor 中上一 episode 的值**——resume+reset 后以配对 noise seed
   显式重采样全部 env 的 noise_scales（lease/abstain 两分支逐位相同的初始
   exploration 状态）；后续序列分叉属 estimand，不作声称。
   **RNG 隔离**：重采样必须用独立 `torch.Generator`（配对 noise seed 播种），
   禁止 `torch.manual_seed` 触碰刚恢复的 anchor 全局 RNG；
   **双 actor 同步**：同一份采样结果同时写入 actor 与 actor_detach 的
   noise_scales buffer，随后执行 A.2.2 的逐位一致断言；
6. **checkpoint 显式列表**：新增 `--ptf-eval-checkpoint-steps
   10750,11500,12250,13000`（固定 save_interval 的整除机制无法覆盖 10750，
   train_ptf.py:2201）。

### A.3 等价性测试（实现完成后、smoke 前必须全过）

> **执行状态（2026-07-17，toy 段：crawl，16 envs，anchor@120→分支 320）**：
> 七项完成。1/2/3/7=PASS（tests/test_anchor_core_resume.py 单测 +
> resume 块运行时断言）；4=语义层（CPU 逐位）PASS+执行层（GPU 幅度，
> 诊断性质）通过——正式 phase report 比值 actor 1.10/qnet 1.253/
> target 1.10，均 ≤3× 地板；5=REPORTED（跨进程 d_dup 预演：分叉幅度与
> 同进程噪声地板同量级，qnet 0.091 vs 0.092@320 步）；6=PASS
> （option/beta optimizer 零 Adam 状态）。测试套件全绿。
> 产物：scripts/p0_equivalence_tests.py + logs/p0_equivalence/run_<UTC>/
> report.json（不可变审计目录，含 git SHA/CLI/device/anchor checksum）。
> 五次复核后修复项：checkpoint completed-step 语义（off-by-one+缺
> run_stop_step 文件）、anchor/branch 冻结参数一致性断言、segment 命名
> 空间续接、provenance 2/3-group 全字段 round-trip 测试——修复后测试
> 需重跑刷新本状态。

1. resume 后 state digest 对照：actor/critic/target/normalizer/optimizer/
   scheduler/replay ptr 与 bundle 记录一致；
2. LR 断言：resume@10000 的 LR == anchor 保存时 LR（total_timesteps 语义）；
3. actor_detach 一致性：权重+noise_scales 逐位（noise_scales 配对重采样后）；
4. **control-arm 等价测试（v2.1.2 实测后修订；不看 outcome）**：
   参照对象修正（v2.1.1）——abstain 分支 reset 环境后的 occupancy 与
   "不间断 scratch"必然不同，不应被要求等价。对照双方：
   分支 A=挂任务 bank + `admission_mode=none`；分支 B=空 bank 纯 student；
   相同 reset seed、相同初始 learner RNG、相同 noise_scales 重采样种子。

   **实测诊断记录（2026-07-17 toy 段，触发判定口径修订）**：
   GPU 同进程顺序执行的逐位对照首跑 FAIL（qnet/qnet_target 分叉），
   随后的 B-B 对照（同进程顺序跑两次**完全相同**的空 bank 分支）
   **逐位复现了同一失败模式**（qnet 分叉、actor/normalizer/RNG 终态一致）
   ——分叉与 resume 语义、bank、admission 全部无关；
   `CUBLAS_WORKSPACE_CONFIG=:4096:8` 与 fp32（--no-amp）均不消除。
   根因=同进程第二次运行复用第一次建立的 CUDA 库内部状态
   （cublasLt 启发式/分配器布局），critic 大 GEMM（1024 hidden、101 atoms、
   CDQ）选择不同归约顺序；此噪声在同进程两次运行间**原理性不可消除**
   （四次复核对"同进程≠天然逐位"的警告字面应验）。

   **修订后判定口径（双层）**：
   - **语义层（决定性，逐位）**：CPU 上（无 cuBLAS 执行历史噪声）同进程
     A-B 对照，全部核心组件 checksum 逐位一致 + 硬断言集（终态全局 RNG
     digest 一致、option/beta optimizer 无任何 Adam 状态、
     execution_counts source 位严格 0、global_step 一致）。
     CPU 逐位一致 ⇒ 两代码路径语义等价的充分证明；
   - **执行层（GPU 幅度对照，诊断性质）**：同进程顺序跑 A、B、B′
     （B′=B 的完全重复），每组件 max-abs 满足 |A−B| ≤ 3×|B−B′|
     （纯执行噪声地板）。**判据定位（五次复核收窄）**：单一 A-B-B′ 顺序、
     3× 阈值为工程启发式、运行顺序与 CUDA 库状态混杂——通过只表示
     "**未发现超出重复运行噪声量级的额外差异**"，不构成 GPU 上分布等价的
     充分证明；充分性主张由 CPU 语义层独立承载。超出 3× 即存在语义差异
     嫌疑，回到诊断。
     数字口径（同次采样内自洽）：正式判定值=正式 phase 运行的 report.json
     （320 步 fp32：actor 1.10 / qnet 1.253 / target 1.10）；诊断期另一次
     独立采样曾录得 0.73/1.79/1.71/1.53——两组各为独立三段采样，比值本身
     有跑间波动，判定只认正式 phase 的当次 report。
   与正式 P0 的关系（措辞收窄）：该执行噪声**不自动使 P0 无效**——其对
   结果的影响由 duplicate 分支与 d_dup gate 在 measurement 层显式评估。
   证明目标不变：control 分支挂 bank 无任何副作用（exact abstention 在
   resume 语境下成立）。anchor 恢复自身的正确性由测试 1-3、7 覆盖；
5. duplicate 语义验证：完全同 seed 重启进程 ×2，报告分叉幅度（即 d_dup 的
   生成机制预演）；
6. option 族不参与断言：两分支各跑 200 步，断言 option_module 参数梯度恒零、
   option 前向未被行为路径调用；
7. **round-trip 测试（新增）**：anchor load → 立即 re-save，两个 bundle 的
   全部白名单组件 digest 逐位一致（加载路径无损）。

### A.4 实现工作量与风险

- 触碰面：train_ptf.py（CLI+resume 分支+run_stop_step）、anchor_io.py
  （load_anchor_core）、ptf_replay.py（provenance 重建接口）——估计 <300 行；
- 风险清单：normalizer 统计的 dtype/device 迁移、replay import 的 strict 校验
  与 provenance 列数、RNG 恢复顺序（先 RNG 后构建会被构建消耗——沿用
  rng_isolation 的既有顺序）；
- 全部实现经 ChatGPT 代码复核后才进入 smoke。

## 11.5 执行包实现指针（五次复核阻塞问题 2 的回应）

| run card 组件 | 实现 |
|---|---|
| §6 独立 evaluator | `scripts/p0_evaluator.py`（结构性 source-free：不构建 bank/option/admission；双播种 reset(E11)；4×8×1000 步冻结面板；crawl progress=qpos[0] 位移 max；truck=info["reward_robot_package_truck"]；输出拒绝覆盖） |
| §5 δ 冻结 | `scripts/p0_freeze_delta.py`（10k-15k 窗口 [eval] 行提取；0.5×跨 seed SD；输出拒绝覆盖；两任务 dry-run 已验证：crawl≈33.6 / truck≈28.8，正式冻结在执行批准前） |
| §5/§7 裁决器 | `scripts/p0_adjudicate.py`（t=1.8856 写死；d_dup 前置判序；六步 joint；treatment 审计=lease 0.10±0.02/abstain 双零，未过→ENGINEERING_INVALID；角例单测 tests/test_p0_adjudicate.py） |
| §4 实验矩阵 | `configs/experiments/p0_siv_v2_lease_oracle.yaml`（14 分支+6 anchor 全 cell、noise seed 面板、δ 数据源、执行前 SHA256 冻结） |
| §4/§9 执行与证据链 | `scripts/p0_orchestrator.py`（冻结 plan、source 权重/δ 指纹、显式 GPU 池、进程组清理、真实 execution record、duplicate 事务回滚、smoke 路径隔离与含启动成本预算） |
| §8 deployment diagnostics | launcher yaml `deployment_diagnostics` 段（报告项） |

## 12. 复核状态与遗留给代码复核的检查项

三次复核（2026-07-17）已回应 v2.1 检查单的 #1（白名单遗漏→RNG 全集/named
generator/noise_scales，已修）、#3（student_logit→float32 运行时断言，已修）、
#4（判序矛盾→六步固定顺序，已修）、#5（测试 4 参照错误→control-arm 等价
测试重写，已修）。裁决："科学设计条件通过，v2.1.1 定点修订后直接进入代码
实现；estimand/任务/L/dose/P0 阈值不变。"

**遗留给实现代码复核的检查项**：

1. §3 replay 冻结参数组合（uniform_mix=1.0 + physical_after_authority）在
   admission=all 语义下的实际采样行为是否如预期中性（代码路径核查）；
2. load_anchor_core 白名单与加载顺序的实现是否与 A.1 清单逐项一致
   （含 GradScaler growth tracker）；
3. provenance 优先方案（anchor 按任务组数配 schema）在空 bank 训练路径上的
   可行性核查，不可行则落 fallback；
4. treatment 审计（§11）的统计口径与 admission_sample_counts 现有字段对齐。

**五次复核（2026-07-17）处理记录**：裁决=Needs revision（暂不批准 smoke）。
六项全部核实采纳，零反驳：
1. checkpoint off-by-one+缺 run_stop_step 文件 → 保存移至递增与 scheduler
   step 之后按已完成步数判断；等价性测试 branches phase 断言全部显式
   checkpoint（含 stop 步本身）生成且 global_step 正确；
2. anchor/branch 冻结参数一致性 → resume 块 41 键白名单断言（不一致即
   ValueError；实测首跑即抓到旧 anchor 的 amp 不一致，按设计工作）；
3. provenance 审计 → 2/3-group 全字段 round-trip 单测（含 segment_id/
   env_rank/learner_step 标量字段+组数不匹配拒绝）；segment 命名空间续接
   （resume 后 counter 从 anchor max_segment_id 续编，防碰撞）；
4. P0 执行包 → §11.5 五组件全部落地；
5. 等价性报告 → 不可变 run_<UTC>/ 目录+manifest（git SHA/CLI/device/stop/
   anchor checksums）；--device 用法谎言已删（用 CUDA_VISIBLE_DEVICES）；
   run card 比值数字矛盾已修（正式判定只认当次 report，两组采样标注来源）；
6. GPU 双层判据收窄 → "未发现超过重复运行噪声量级的额外差异"（非充分证明，
   充分性由 CPU 语义层承载）；"P0 不受影响"改为"噪声不自动使 P0 无效，
   影响由 duplicate 与 d_dup gate 评估"。
修复后全链重跑：CPU 语义层 PASS、GPU 执行层 PASS（ratio ≤1.08，
9 个显式 checkpoint 全部生成验证）、evaluator 32-episode 面板跑通
（crawl return_mean=265.55@toy step320，指标提取正常）、
δ dry-run 两任务通过。

**七次复核（2026-07-17）处理记录**：裁决=Needs revision（转移到 orchestrator
执行安全与证据闭环）。全部核实采纳：
1. GPU 队列重写：显式空闲 GPU 池（len(active) 索引法在先完成场景会把新作业
   派到已占用卡——实锤采纳）；失败/中断终止并 wait 全部子进程；并发硬上限
   2（E2）；log handle 显式关闭；配 5 项队列单测（含 GPU 归还与孤儿清理）；
2. 裁决器三类绕过封死：eval 补 deterministic/source_free/identity_checked/
   episodes 真实 reset-seed 面板验证；duplicate 走与正式臂完全相同的 eval
   身份验证（manifest 增 checkpoint_a/b 供 SHA 交叉验证）；checkpoint
   treatment 全配置校验（bank/mcg_groups/student_logit/noise seed=77000+s/
   warmup/ablation/replay 冻结参数/run_stop_step/anchor 溯源 bundle）；
   配 4 项新反例（28 项裁决器测试）；
3. duplicate 归档语义定稿：**正式 s1=第一次运行(A)**——第一次产物 mv 至
   不可变归档 A → 独立日志重启第二次 → 产物 mv 至归档 B → A copy 回正式
   路径；manifest 的 abstain_s1 证据链=A；流程提为可注入函数并单测；
4. eval 作业带 expected_artifacts（执行器自动确认 JSON 产出），完整性
   清单逐条标注强制执行者；
5. `--smoke` 隔离模式：1 anchor(500 步)+1 lease 分支(200 步)+1 完整 eval
   面板，p0smoke_* 前缀+独立目录，与正式路径零交集（单测断言）；
6. 文档 52/56 残留与尾随空格修正。
待八次（定点）复核。

**八次复核（2026-07-17）处理记录**：裁决=Needs revision。八项全部核实采纳：
1. smoke 三阶段严格顺序执行（anchor→branch→eval 逐作业入队，单测断言
   不批量）；
2. 进程组清理（start_new_session+killpg SIGTERM→SIGKILL）——只 terminate
   顶层 Popen 会留 SubprocVecEnv 孙进程（实锤采纳）；配"父进程再启孙进程"
   测试；
3. smoke 资源采集：队列 metrics 采样进程组 RSS 峰值（/proc 全组求和）+
   该卡 GPU 显存峰值（nvidia-smi，共卡注明）+SPS（log 提取）+wall-clock，
   smoke report 含 6×10000/14×3000/52×eval 的 GPU-hours 外推与 24/48 阈值；
   --gpus 拒绝重复 ID；
4. duplicate 事务化：preflight 全部前置（零副作用）→staging 暂存→成功后
   原子 rename 提交 A/B→失败 finally 恢复 A 回正式路径并清理 staging；
   配回滚/重试/preflight 三场景测试；
5. 裁决器五绕过封死：duplicate checkpoint+execution record 必填且走全套
   _validate_checkpoint；episodes 恰好 32 行且每冻结 seed 恰好一次；
   source_free 精确白名单（字符串"false"拦下）；δ 文件验证（task/definition/
   window/3×sha256/有限正数/crawl progress 阈）；source_names 逐位等于冻结
   列表+计数列数=num_sources+1（truck 4 源=5 列）+非负整数；8 项新反例
   （裁决器 36 tests）；
6. "SHA 必须不同"删除（方法论修正采纳：独立重启可产出逐位相同 checkpoint
   =理想 d_dup=0）；独立性改由 execution_record（execution_id/日志路径
   不同）证明，orchestrator 归档时写入；配"SHA 相同仍合法"阳性测试；
7. 冻结 plan 执行：--execute-plan+--expected-plan-sha256，核对文件 SHA/
   当前 HEAD/工作树干净/bank yaml SHA/source manifest SHA（plan 增
   frozen_inputs 指纹）；正式矩阵禁止即时 build 执行；
8. 预算段 52 口径修正（12×4+2×2）。
九次复核结果见下；smoke 仍须节点空闲且获得单独授权后执行。

**九次复核与直接修复（2026-07-17）处理记录**：科学 estimand、任务、L、dose、
阈值均未改变；本轮只收口执行安全和证据可验证性。

1. 修复失败作业自身的孤儿进程：队列在 leader 已退出/被 reap 后仍主动清理其
   process group，而不只清理 `active` 中的其他作业；忽略 zombie 以避免每次
   清理误等满 30 秒。加入“失败 leader 先退出但 child 仍存活”的实锤反例；
2. duplicate 回滚覆盖“部分 B 已写入”场景：失败时先删除所有 formal B，再从
   staging A 无条件恢复全部产物，同时删除失败 B 的 execution record，杜绝
   `B/B/A/A/A` 混合态并允许干净重试；
3. execution record 改为由实际 `Popen` 成功路径生成，绑定 execution_id、PID、
   GPU、完整 CLI+SHA、起止时间、git SHA、真实 log+SHA、全部预期产物 SHA；A/B
   必须 CLI 与 git 相同、execution_id 与日志不同，且 primary checkpoint 必须
   能回链到各自记录；
4. 裁决器不再信任“看起来像 SHA”的 δ 元数据：要求三份历史日志真实存在、
   SHA 相符，重算 10k/15k 窗口均值、跨 seed SD 与 `δ=0.5×SD`；eval aggregate
   必须由 32 条 episode 明细重算一致，`source_free` 使用精确冻结声明；duplicate
   A/B 均再次审计 exact abstention 双零；
5. frozen plan 新增 source checkpoint 权重和两任务 δ 文件指纹；δ 未正式冻结时
   可生成审阅 plan，但 `verify_frozen_plan` 必须拒绝执行。正式顺序因此固定为：
   smoke → 冻结并提交 δ → 在干净 commit 上重新生成 plan+SHA → PI 批准；
6. smoke 的 execution record 改入 `logs/p0_smoke/`，不再触碰正式目录；完整 GPU
   ID 列表在取单卡前校验；预算计入每作业实测启动成本；
7. 所有 planned output 在进程启动前必须不存在，防 exit=0 进程误认证旧产物；
8. 验证结果：P0 定点测试 **60 passed**；全仓测试在 `PYTHONPATH=.`、FastTD3
   环境下 **158 passed, 11 warnings**。首次未设置 `PYTHONPATH` 的全仓命令在
   collection 阶段因 `ModuleNotFoundError: fasttd3_ptf` 中止，补齐仓库根路径后
   完整通过；该次中止不是测试失败。`git diff --check` 通过。

**当前裁决**：实现可进入 throughput smoke 申请阶段，但本轮未获/未使用 smoke
或正式训练授权；不得把“执行基础设施已通过”写成“counterfactual oracle 已经
有效”，后者只能由正式 P0 结果决定。

## 13. 正式 P0 执行与最终结果（2026-07-18）

**执行记录**：throughput smoke 通过（退出码 0；GPU-hours 外推 3.65h，24h 目标
6.6 倍余量；首次触发因 wandb.init 90s 网络超时失败，加 `WANDB_INIT_TIMEOUT=300`
重试通过）→ δ 正式冻结并提交（crawl 33.556 / truck 28.847，与 dry-run 逐位
一致，commit 93fc72f）→ 正式 plan 冻结（`plan_20260718T044727Z.json`，
sha256=a700fe56…，HEAD=633eeca）→ ChatGPT 第十次复核有条件批准（单 GPU、
分阶段、失败即停、显式裁决）→ 正式执行 2026-07-18 05:56–09:30Z（3h33min，
GPU 5/1，anchors 56min → branches+duplicates 60min → 52 evals 97min，
20 训练作业+52 eval 零失败，每阶段独立通过 verify_frozen_plan）→ 裁决器
显式运行。

**预注册正式裁决（不可覆盖）**：`ENGINEERING_INVALID`
（`logs/p0_lease_oracle/p0_adjudication_result.json`）。触发原因：truck 三个
lease seed 的累计 critic 源采样占比 0.0787/0.0793/0.0773 低于冻结审计带
`[0.08,0.12]` 下限（差 0.001–0.003）；exec 通道与 crawl 全部达标。

**三层最终记录（ChatGPT 十一次意见的两层结论架构）**：
1. 预注册结论：`ENGINEERING_INVALID`（如上，保持字面效力）；
2. 机制诊断（post hoc）：critic 累计占比下偏=per-env 桶可用性×配额归一化的
   冷启动结构效应（分段占比末 750 步收敛 0.096–0.100 到名义值；解析
   q≈η−N_eff·h/L 与 CPU 微仿真复现实测量级与任务排序）——验收带把 exec
   名义剂量带统一套到累计 critic 观测上属校准失配，非注入实施失败。完整
   分析=`docs/p0_posthoc_engineering_sensitivity_20260718.md`（标注
   post hoc，不具裁决地位）；
3. 条件科学结论（接受机制解释的条件下，裁决器纯函数程序化计算）：
   crawl=`UNCERTAIN_NUMERIC`（d_dup=50.50≥δ=33.56）、truck=`UNCERTAIN`
   （CI=[−149.07,+16.13] 跨零）、联合=**`F-a`：统计不可测（数值地板），
   判据封存**。即使接受修正也非 PASS、非 SURROGATE_FAIL。

**处置**：不重训、不改参挽救、不修改原裁决器/原结果；Phase-2
counterfactual lease oracle 作为在线算法贡献封存；回 Phase-1（exact
abstention/provenance/有限剂量+TTL/撤销后退出 active replay/注入前移），
其边界=限制伤害与清除数据，不解决"判断 source 当前有益/有害"。truck U
均值 −66.47 的负号（vs 历史 matched handoff ≈+227.8）记为描述性趋势
（estimand 不同：完整 bootstrap 累计收益 vs 10k 后局部边际效用），支持
"注入窗口靠前"假设，不得写成"已证明有害"。
