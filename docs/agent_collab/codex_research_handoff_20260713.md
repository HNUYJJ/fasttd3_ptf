# Codex 接手后科研工作完整交接：RBO、双通道机制与 Admission Core v1

> 交接时间：2026-07-13T07:43:59Z  
> 交接方：ChatGPT（Codex 执行环境）  
> 接收方：Claude Code、ChatGPT/Codex、Human PI  
> 仓库：`/home/yjj/fasttd3_ptf`  
> 当前基线：`main @ b183f40bcfe6b04fdefdab922371394caaba828f`  
> 重要状态：工作树很脏，大量研究代码、配置、测试、文档和 artifacts 尚未提交；**不得 reset、checkout 覆盖或批量清理未知改动**。  
> 本文范围：记录本轮对话中 Codex 接手科研方向审计后，完成的思路讨论、失败路线、代码实现、实验、结果、运维和当前下一步。早于本轮的 Step-2 z-native/anchor-xattn 工作只作为仓库历史，不是当前活跃路线。

---

## 0. Claude 最先需要知道的五件事

1. **现在不是“一个新万能 selector 已完成”。** 已完成的是 metric-agnostic Admission Core：student-inclusive categorical、exact abstention、quarantine、provenance replay、runtime revocation、actor/critic coherence 和 MCG 后置接口。自动迁移性估计仍未解决。
2. **当前代码路线和论文证据路线不完全相同。** 代码中已有 Admission Core v1；论文当前证据最扎实的默认算法仍是静态 RBO/WFix。Admission Core 的 basketball 负迁移安全门通过，但 powerlift 100k 正迁移 retention 门失败，因此尚不能无条件替代 RBO。
3. **最重要的机制新发现是 post-warmup replay handoff 不完整。** powerlift 到60k时 source transition 只占物理 buffer 的约20.44%，但30k→60k新增 critic samples 中 source 仍约50%。固定跨来源配额放大了正在老化的 source 数据，可能解释30k加速明显而100k收益衰减。
4. **两条自动 selector 路线已被正式负结果叫停。** Cabinet SIV 2×2 得到 `B0=-0.0511, R0=-0.0304, I=+0.0331, T=-0.0484`，未过实践阈值；SHU 又将同一 downstream-negative source 错误判为 eligible。不要重启调阈值或小任务搜索。
5. **建议下一步不是再造一个 transferability 分数。** 先用现有 checkpoint 完成 replay exposure audit，再修 aggregate source replay mass 的 authority-aligned decay；只新增 powerlift 3-seed 一个条件。若仍不能保留100k收益，则 Admission Core 降为安全/机制贡献，论文回到静态 RBO + 条件规律 + 双通道诊断。

---

## 1. 权威阅读顺序与证据层级

### 1.1 建议 Claude 按此顺序阅读

1. 本文：完整交接与当前判断。
2. `docs/admission_core_v1_results.md`：Admission Core 最终实现和 FINALV2 结果。
3. `docs/admission_core_v1_completion_audit.md`：逐条 requirement→evidence→verdict。
4. `artifacts/admission_core_v1/final_completion_audit.json`：机器可读最终审计。
5. `docs/paper_core_contribution_reconstruction_v3.md`：当前论文默认主路线。
6. `docs/core_mechanism_polishing_v4_plan.md`：论文冻结门和不应继续浪费算力的事项。
7. `docs/dual_channel_transfer_evidence_matrix_v1.md`：execution/replay 双通道证据总账。
8. `configs/experiments/rbo_core_result_registry_v1.yaml`：headline 数字注册表。
9. `docs/source_intervention_mechanism_gate_v1.md`：SIV 2×2 正式负结果。
10. `docs/stage_conditioned_source_admission_gate_v1.md`：SHU mandatory contradiction。

### 1.2 证据优先级

发生冲突时使用以下顺序：

1. 当前代码、冻结 checkpoint、原始 episode trace、机器可读 artifact；
2. `admission_core_v1_results/completion_audit` 和 `rbo_core_result_registry_v1.yaml`；
3. `paper_core_contribution_reconstruction_v3.md`；
4. `advisor_feedback_analysis_20260702.md` 的较新章节；
5. 历史 v1/v2 重构和早期计划文档。

历史文档中大量“将要”“可能”“建议”已经被后续实验否决。不得把计划描述成已完成结果。

---

## 2. 接手时的问题与最初科研担忧

Human PI 的核心担忧是：

- 当前贡献看起来主要只是 bootstrap；
- transferability/source selection 指标缺少 insight 和 solid 证据；
- 多数任务只有前期加速，后期上限提升有限；
- locomotion source 的收益可能只是站立、平衡和存活，而不是学到了目标技能；
- 多源 WFix 30k hard progress 低于 run-only，怀疑次优源稀释高价值数据；
- crawl 等任务即使 source 有害，旧固定0.5 teacher 机制仍强迫 source 暴露，是设计缺陷；
- 希望保留“迁移性分数→选教师→bootstrap→replay→MCG”的迁移强化学习主线，但不能用未经验证的标量包装。

围绕这些担忧，先完成了旧机制语义澄清：

- warmup 不是每25步把12/10.5/2.5步切给不同教师；每个并行 env 在 segment 边界只选一个 arm，并连续执行默认25步；
- 旧 safe-bootstrap 先以0.5选 teacher/student；进入 teacher 分支后才在 source 内按静态权重选择一个 source；
- cabinet 示例中 run/stand/walk 的约48%/42%/10%是 teacher 分支内部份额，乘0.5后约为全部环境步的24%/21%/5%，student约50%；
- FastTD3 是 off-policy，但这不意味着应贪心只采最新最高 return 轨迹；trajectory return 混合初始状态、长度、随机性和策略质量，top-return replay 会缩窄支持并造成相关样本过拟合；
- source 的价值必须至少拆成 behavior/occupancy 与 replay/update 两类后果，不能用一个即时 return 同时决定执行和历史数据采样。

---

## 3. 完整时间线

## 3.1 2026-07-02：student-as-arm、在线控制与 replay 通道

### 假设演化

导师提出三点：replay 应更关注当前有用来源；student 应成为候选 arm；需要迁移性指标。初始方案把 `T^0/T^online/T^critic` 设想为一个统一标量，用于选源、student-as-arm 和 replay weighting。

### 实现

- `fasttd3_ptf/ptf/mcg.py`：新增 `online_bootstrap`，source+student arm value/count、在线 reward EMA、先验→在线混合、epsilon exploration、arm 日志。
- `fasttd3_ptf/official_fasttd3_ptf/train_ptf.py`：环境 step 后回传 arm reward；接入 replay weights 和日志。
- `fasttd3_ptf/official_fasttd3_ptf/ptf_replay.py`：按 option/source 权重重采样，student 权重恒1，坏 source 只降不升，支持 actor/critic/both/split。
- 分析脚本：`analyze_onlineb.py`、`analyze_tcritic_offline.py`、`analyze_wfix.py`、`analyze_terrain.py`。

### 结果

- crawl onlineb：student share 从约50%升到76%，但 AUC 仍约 `634.8±77.5`（完整3-seed终值），弱于 scratch `812.0`；说明只减少 source execution 不足。
- crawl OBRW（在线执行控制+对称 replay attenuation）：`729.5±27.6`，相对 onlineb `+94.7`，3/3 seed 正；说明 replay exposure 是持久通道。
- pole：onlineb/OBRW 与 WFix 基本同档，表明局部 replay 控制不一定破坏正迁移。
- slide：OBRW `614.8±16`，高于 onlineb `551.3±5` 和 WFix `522.7±20`，是 OBRW 相对静态 RBO 唯一决定性大胜场。

### 被实验修正的观点

“student share 上升即可自动回到 scratch”被 crawl 否定。student-as-arm 是必要的控制接口，但不是 exact fallback。

## 3.2 2026-07-02 至 07-03：T-gated、split replay 与多 horizon 失败

### T-gated/hard abstain

尝试用阈值将 arm value 变成 hard abstain。结果暴露阈值脆弱、早期 signal 不可靠，不具备跨任务稳定性。该路线封存。

### actor/critic replay 归因矩阵

比较 actor-only、critic-only、split 和 both：

- crawl actor-only约676.2；
- critic-only约629.6；
- split约715.9；
- both约767.5（该组对照口径来自同一归因实验总账）。

结论：actor/critic 接触不同 source-state 分布会造成 AC mismatch；后续 admission 主路径强制 actor 复用 critic batch。

### multi-horizon arm

实现 source×horizon arms，希望自动学 temporal extent。1-seed 和3-seed结果在 slide/crawl 显著恶化，arm 数增加带来探索税、方差和毒性扩散。全局 multi-horizon 不进入主方法，只作为 limitation。

### negative controls：door/spoon

- door OBRW约 `295.8±8.5` vs scratch `295.0±5.4`，基本零伤害，student share升到86–88%；
- spoon OBRW `354.2±1.4` vs scratch `315.4±13.9`，3/3正。

当时支持“soft control 局部有效”，但后续 basketball 3/3 负直接否定其 universal safety。

## 3.3 2026-07-03 至 07-05：terrain、breadth、扩源与第二批任务

### Terrain 因果分解

stair/slide/pole/crawl ×3 seeds：

- RBO/WFix − uniform random bootstrap：12个 task×seed pair 中11个正，平均 `+77.9`，paired `t=3.08`；
- safe-h50 − RBO：平均 `−11.4, t=-0.46`。

结论：在 source bank 强分化时，静态 source weighting 的边际价值显著；horizon 不是总体主增益，但存在任务依赖。

### Breadth 第一批

目标扩到 maze/truck/cabinet/door/spoon。大体结果：

- maze：scratch 267.7，rand 346.1，RBO 351.0，OBRW 341.7；
- truck：scratch 1156.3，rand 1296.6，RBO 1191.4，OBRW 1304.8；
- cabinet：scratch 121.0，rand 210.9，RBO 235.3，OBRW 219.6；
- door：scratch 295.0，RBO 285.9，OBRW 295.8；
- spoon：scratch 315.4，RBO 346.9，OBRW 354.2。

RBO/OBRW 与 uniform 在多数弱分化 bank 中打平，说明主要收益常来自 reward-bearing source data，而不是精细 weighting。

### Source library 扩展

在三 locomotion source 上加入 hurdle：

- truck：RBO `1191.4→1421.3`，`+229.9, t=3.47`；
- maze：`351.0→351.3`，`+0.3, t=0.16`。

形成条件规律：扩源收益约受 `source complementarity × remaining target headroom` 联合约束。probe 增量不是充分条件。

### 第二批 standard-9

任务：bookshelf_simple、basketball、window、powerlift、balance_hard；bank 为 stand/walk/run/reach + hurdle/stair/slide/crawl/pole。

- bookshelf_simple：RBO−scratch `+32.1, t=1.13`，正趋势但不显著；
- basketball：RBO−scratch `−101.5, t=-2.58`，OBRW−scratch `−74.0`，两者均3/3负；
- window：RBO−scratch `−31.9, t=-0.99`，高方差负趋势；
- powerlift：RBO−scratch `+77.6, t=14.78`，3/3正；
- balance_hard：RBO−scratch `+6.1, t=0.75`，RBO≈rand，标准9源全部 probe 近零/全零时 weighting 无优势。

balance_hard rand-s3 和 WFix-s2 首次因 replay 分配 CUDA OOM，2026-07-12 按原配置补齐；失败尝试保留 provenance，最终逻辑 run slot 48/48 有效。

关键结论：`T^0` 低分绝对值没有方向判别力——powerlift在全部短probe<1时是强正例，basketball弱正probe却大负迁移。

## 3.4 2026-07-11：针对“只是站立稳定”的 hard-progress 审计

### P0：四任务 stability-deconfounded audit

覆盖 hurdle/maze/powerlift/basketball，scratch/WFix、10k/30k/100k、3 training seeds、每 checkpoint 32 eval episodes。

- hurdle：50k move progress约 `0.731 vs 0.356`，100k约 `0.922 vs 0.923`；支持真实早期跨障加速，不支持 ceiling。
- maze：存在与 survival 方向相反的 early progress，不能只用“活得更久”解释。
- powerlift：return/AUC强正，但现有 hard-skill 指标没有证明真实举重技能完成；只能写 sample efficiency/return 正例。
- basketball：姿态或 viability 信号不能挽救任务表现，确认负迁移。

后来发现旧 evaluator 用 `env.unwrapped.seed(seed)` 未正确播种 Gymnasium `np_random`，所以旧 P0/P1/P2 的条件均值和跨训练种子方向保留为描述性证据，但不再称 exact same-state episode pairing。

### P1：cabinet run-only vs stand-only vs WFix

run 在30k/100k hard progress 上总体优于 stand，3 seed方向支持 source identity matters；所有条件评估 episode length 都为1000，但 cabinet 跌倒不终止，所以“等长 episode 排除稳定性”这条旧表述后来被收窄。

### P2：run-dose-matched `run24`

构造 `warmup_exec_prob=0.24021406`，使 run 注入量与 WFix 中 run realized share匹配：

- WFix realized run share约23.894%；
- run24 与其相差约0.065个百分点；
- run24−WFix hard progress约 `+0.031±0.113`，seed方向 `−/0/+`。

结论：不支持“stand/walk 主动有毒，因此应该 winner-take-all”；更像高价值 run 的剂量/机会成本候选，但没有稳定单调 dose-response。

新增 `scripts/analyze_warmup_source_dose.py`、protocol/result docs 和测试。

## 3.5 2026-07-11：Source Intervention Value 2×2 机制门

### 为什么做

即时 return、稳定性和 hard progress 经常错位，需要直接区分：

- source 改变后续可达状态/occupancy；
- source transition 进入 replay 后的 update effect；
- 二者交互。

### 主要基础设施

- `anchor_io.py`：完整 learner anchor，网络/optimizer/scheduler/scaler/normalizer/replay/RNG/hash；
- `humanoid_bench_env.py`：seed plumbing 和 branch-state RPC；
- `hb_branch_state.py`：MuJoCo/task state捕获恢复和诊断；
- `factorial_data.py`：共享 potential trajectory bank 与固定采样；
- `learner_factory.py`、`update_kernel.py`：可复现 FastTD3 分支更新；
- `rng_isolation.py`：Python/NumPy/Torch/CUDA RNG 隔离；
- `probe_source_intervention_2x2.py` 和 `run_source_intervention_gate_v1.sh`；
- 对应 anchor/factorial/RNG/update tests。

### 正式实验

cabinet scratch seed1 at10k，512 anchors；student/run-composite prefix 25步 + student follow-up 25步；共享 D0；六分支 `00/10/01/11/d0_only/duplicate00`；400 critic/200 actor updates；64个 source-free eval seeds。

首次 CUDA rerun 因 distributional projection 约1e-9非确定性导致 duplicate00 分叉，定位后启用 deterministic algorithms、CUBLAS workspace、关闭 TF32；正式 rerun duplicate 完全一致。失败首轮被封存，未进入结论。

### 结果

- treatment 确实改变状态：endpoint L2 mean约62.81；head height `+0.2371`、upright `+0.0978`、hand-to-door distance `−0.1211`，但 door fraction仅 `+0.00036`；
- K=400：`B0=-0.051065`，`R0=-0.030366`，`I=+0.033073`，`T=-0.048357`；
- duplicate noise=0，实践阈值0.10；所有机制量均未过门；
- 姿态改善与 door hard progress 方向分离。

裁决：`Engineering Go=true, Feasibility Go=false, STOP_COMPLEX_ESTIMATOR`。不扩3/5 learner seeds，不训练 `hat(SIV)/DV`，不启动闭环 allocation。

## 3.6 2026-07-11：外部审计意见与论文核心重构

Human PI 将 ChatGPT-5.5-Pro 和5.6-Pro审计复制到：

- `docs/ChatGPT-5.5-Pro_review_20260711.md`；
- `docs/ChatGPT-5.6-Pro_review_20260711.md`。

讨论后认为5.5意见更完整，但没有盲从。先后形成：

- v1：Temporary Behavioral Scaffolding/EPS方向；
- v2：恢复 transfer RL 主线，提出 stage-conditioned admission、exact abstention、quarantine、consistent replay、MCG后置；
- v3：在SHU失败和全结果总账后，回到静态RBO主方法 + source-bank条件规律 + 双通道负迁移 + broad regime map；
- v4 plan：在完整论文前先通过算法、novelty、metric、causal、skill、boundary、reproducibility冻结门。

PI 明确希望保留迁移强化学习主线，而不是只做行为脚手架描述；同时不允许用更多无关小实验堆贡献。

## 3.7 2026-07-12：SHU stage-conditioned admission gate

### 设计

在相同 student occupancy anchor 上比较 `student→student` 与 `source→student` 的25+25短干预；direct、handoff和risk做保守置信门；probe数据仅进quarantine。

### 工程实现

- `source_admission.py`：schema、duplicate/provenance校验、robust scale、bootstrap bounds、non-compensatory rule；
- `probe_stage_conditioned_source_admission.py`：preflight/smoke/collect/analyze/report；
- `hb_branch_state.py` 扩展通用HumanoidBench diagnostics；
- `test_source_admission.py`、`test_hb_branch_state.py`。

### Mandatory cabinet 结果

- 512/512 anchors，duplicate exact，treatment非零，quarantine-only；
- direct raw mean `+0.0253`；
- handoff raw mean `+0.1728`，one-sided LCB为正；
- risk差 `−0.0156`；
- SHU 判定 `eligible=true`。

但同 source/stage/horizon/operator 的 downstream 2×2 为 `T=-0.04836`、AUC `-0.02288`，要求拒绝。SHU产生 mandatory label contradiction；independent student-null scale还塌到 `1e-6` floor。

裁决：`STOP_CLOSED_LOOP`。不补跑 hurdle/crawl，不调阈值，不把 behavior/handoff score当 replay data utility。

## 3.8 2026-07-12 至 07-13：Admission Core v1 实现

Human PI 随后明确要求：暂不实现迁移性指标，但至少准确实现 student-inclusive exact abstention 等安全/生命周期机制并用实验验证。

### 实现文件

#### `admission_control.py`

- immutable `AdmissionSnapshot`；
- `all/none/static/manifest/schedule`；
- admitted source mask、source logits、student logit；
- source+student categorical；
- exact-empty one-hot student；
- quarantine artifact SHA256绑定。

#### `mcg.py`

- 新 `admission_bootstrap`；
- 同一 categorical 直接选择 sources+student，不再有 admission 外层固定0.5 teacher；
- source mask与动态logits；
- source撤销时立即释放 latch；
- admission后才允许body-group source authority。

#### `ptf_replay.py`

- transition provenance：`behavior_source/source_by_group/executed_group_mask/decision_id/segment_id/segment_step`；
- student与每个source stratum；
- rejected source active mass=0；
- contributing group中任一source撤销，混合transition退出active replay；
-来源配额 + 来源内 recency/TD priority/uniform；
- policy events和critic/actor实际采样计数；
- snapshot/restore。

#### `train_ptf.py`

- admission config/CLI接线；
- static/manifest/schedule决策；
- behavior与replay原子更新；
- 写入main replay前拒绝source断言；
- target-only exact-abstention fast path；
- source/option/MCG/transfer update短路；
- checkpoint admission audit；
- actor在admission主路径复用critic batch。

#### 其他代码

- `humanoid_bench_env.py`：Gymnasium `np_random` + worker global `np.random` 双播种；
- evaluator：task-specific horizon，basketball=500、powerlift=1000；
- analyzer/adjudicator/finalizer/verifier/orchestrator脚本；
- checkpoint audit、training recovery、watcher；
- admission/replay/RNG/evaluator/anchor/update tests。

### 没改的核心语义

- source bank与obs/action adapters保留；
- source轨迹仍在target env中获得真实target reward；
- segment horizon仍默认25；
- warmup仍30k；
-最终评估仍source-free student；
-正式性能实验仍`bootstrap_only`，没有把MCG蒸馏混入主归因。

## 3.9 2026-07-12 至 07-13：Admission Core 正式实验

### 冻结协议

- stamp：`20260712TFINALV2Z`；
- 两队列，每次最多两个训练，避免CPU争用；
- 关键实现/protocol/source bank SHA256固定；
- 六个100k runs：
  - basketball exact-none，seed1/2/3；
  - powerlift admission-all，seed1/2/3。

### 工程结果

- 57项 admission/MCG/replay/RNG/evaluator核心测试全过；
- quarantine真实artifact：512 anchors，0 learner updates，0 main replay writes；
- runtime revoke：step20后 source execution=0、source critic samples=0；320 source transitions物理保留、active=0；
- MCG-full provenance smoke跨过warmup并完成actor/critic更新；
- six final checkpoints、30k audits、implementation hashes全部认证。

### Basketball exact-none

30k/60k/100k三个seed均：

- source candidate mass/execution/main replay/critic samples全部0；
- student mass精确1；
- actor independent samples=0；
-100k每个run有12,800,000 student executions。

FINALV2 fixed paired结果：

- progress delta vs scratch `+0.03125±0.08268, t=0.655`；
- legacy WFix progress delta `−0.09375±0.11267`；
- exact-none相对legacy恢复 `+0.125`；
- return delta vs scratch `+87.18±88.68`。

负迁移安全gate PASS。解释只能是“正确弃权后回到自主RL统计分布”，不是source正迁移，也不证明系统自动识别了basketball。

### Powerlift admission-all

student logit设为九个source logits的logsumexp，使student aggregate mass=0.5并保留旧WFix source相对权重；因此行为剂量近似旧0.5 student/0.5 teacher，主要隔离scheduler/replay语义。

30k：

- student execution share约0.505；
- student critic share约0.504；
-九个source全部有非零execution和critic exposure；
- progress delta vs scratch `+0.0003879±0.0001556, t=4.318`；
- return delta `+48.57±17.30`。

100k：

- progress delta `+0.0001017±0.0002937, t=0.600`；
- return delta `+4.74±33.00`；
-相对legacy WFix retention `0.1996<0.5`。

正迁移 retention gate FAIL，联合性能gate FAIL。只能声称早期加速，不能声称100k ceiling或稳定保留。

## 3.10 2026-07-13：W&B 回填、磁盘清理与 replay 新诊断

### W&B 回填

正式run启动时 `WANDB=0`，原始stdout保留但没有W&B run。应PI要求实现：

- `scripts/backfill_admission_core_v1_wandb.py`；
-每run回填999个speed点、19个eval点、frame/LR和30k/60k/90k/100k admission counters/shares；
-原始log/meta作为artifact上传；
-明确标记backfilled和不可恢复指标；
-六个run逐点远端API验证，每run1000 history rows与本地完全一致。

项目：`https://wandb.ai/yujiajie-nju/fasttd3_ptf`  
group：`admission_core_v1_20260712TFINALV2Z_backfill`  
manifest：`artifacts/admission_core_v1/wandb_backfill_manifest.json`  
verification：`artifacts/admission_core_v1/wandb_backfill_verification.json`

无法真实恢复：actor/q loss、grad norm、env/buffer reward、逐update MCG tensors；没有伪造。

一次 powerlift-s3 首传缺16行，缺陷run被删除，以 `acv1bf-24764abfc667-r1`重建并通过完整验证。

### 权重清理

按PI明确授权删除不必要 `.pt`：

- 删除2,941个文件；
-释放47.52 GiB；
- `.pt`占用58.20→10.68 GiB；
-保留412个正式final、当前Admission六个final+18个阶段checkpoint、被配置引用checkpoint，以及replay/trajectory/quarantine/demo科研证据；
-删除旧周期checkpoint、无引用smoke final、被final覆盖的latest、失败/中止/smoke权重。

Claude 不应期待大部分旧中间权重仍存在；需要使用结果artifact或final。当前 Admission `20260712TFINALV2Z` 24个权重仍保留。

### 最新 replay exposure 审计

使用现存30k/60k/90k/final checkpoint重新计算：

| stage | mean source physical buffer share | source critic cumulative share | source critic share in stage increment |
|---:|---:|---:|---:|
| 30k | 0.49478 | 0.49626 | 0.49626 |
| 60k | 0.20437 | 0.49813 | 0.50000（30k→60k） |
| 90k | 0 | 0.44832 | 0.34871（60k→90k） |
| 100k | 0 | 0.40349 | 0（90k→100k） |

source execution在30k后停止；但当前replay先固定source/student stratum总质量，再在来源内部应用recency/priority/uniform。正式配置又关闭recency/priority。因此到60k，只有20.4%的物理source数据仍获得50%的新增critic exposure，直到约81.2k被circular buffer完全覆盖。

这是当前最值得修复的机制缺口：**admission-consistent不等于authority-aligned lifecycle**。来源内部recency不能降低跨source/student总quota。

---

## 4. 原机制与当前机制对照

| 方面 | 原始 RBO/WFix | Admission Core v1 |
|---|---|---|
| source准入 | 所有bank source默认可用 | 外部显式admitted/rejected snapshot |
| teacher/student | 外层固定0.5 teacher；teacher内softmax source | sources+student一次categorical |
| 全源有害 | 仍强制0.5 teacher暴露 | 全拒绝时精确100% student |
| probe | 静态T0相对权重 | 可绑定quarantine artifact，但当前不自动推断utility |
| probe数据 | 无统一强隔离契约 | quarantine-only，0 update，0 main replay write |
| main replay写入 | warmup数据全部进入 | student和admitted source；写入前断言 |
| replay | 标准历史uniform；OBRW为可选扩展 | provenance strata + admission mass + 来源内quality |
| revoke | 行为停后旧数据自然驻留 | active mass可立即为0，物理证据保留 |
| actor/critic | 标准共享或历史split消融 | admission主方法强制共享batch |
| MCG | 曾被当作主组件 | admission后的可选body-group执行器 |
|最终策略 | source-free | source-free，保持不变 |

重要：powerlift all的student mass仍为0.5，因此当前失败不能简单归因于“学生比例改变”。

---

## 5. 当前代码变更总览

### 5.1 核心已修改文件

- `fasttd3_ptf/official_fasttd3_ptf/train_ptf.py`：在线arm、replay控制、anchor、admission、target-only、provenance、audit、seed/eval接线。
- `fasttd3_ptf/official_fasttd3_ptf/ptf_replay.py`：source weighting、actor/critic roles、snapshot、provenance、admission strata、recency/priority、revocation、audit。
- `fasttd3_ptf/ptf/mcg.py`：online_bootstrap、multi-horizon历史变体、admission_bootstrap、mask/latch/revocation/logging。
- `scripts/official_fasttd3_train_target_ptf.sh`：相关env/CLI参数传递。
- `scripts/probe_transfer_map_v2.py`：probe/诊断修订。

### 5.2 新核心模块

- `admission_control.py`
- `source_admission.py`
- `anchor_io.py`
- `factorial_data.py`
- `hb_branch_state.py`
- `humanoid_bench_env.py`
- `learner_factory.py`
- `rng_isolation.py`
- `update_kernel.py`

### 5.3 新/重要脚本

- `probe_source_intervention_2x2.py`
- `probe_stage_conditioned_source_admission.py`
- `stability_deconfounded_audit.py`
- `analyze_warmup_source_dose.py`
- `analyze_breadth_batch2_local.py`
- `analyze_admission_core_v1.py`
- `adjudicate_admission_core_v1.py`
- `audit_admission_checkpoint.py`
- `verify_admission_training.py`
- `run_admission_core_v1.sh`
- `complete_admission_core_v1.sh`
- `finalize_admission_core_v1.sh`
- `watch_admission_30k_final_v2.sh`
- `backfill_admission_core_v1_wandb.py`
- source-bank builder scripts。

### 5.4 新测试

- `test_admission_control.py`
- `test_source_admission.py`
- `test_ptf_replay_snapshot.py`
- `test_anchor_io.py`
- `test_factorial_data.py`
- `test_hb_branch_state.py`
- `test_humanoid_bench_seed.py`
- `test_rng_isolation.py`
- `test_update_kernel.py`
- `test_adjudicate_admission_core_v1.py`
- `test_analyze_warmup_source_dose.py`
- `test_stability_deconfounded_audit.py`
- `test_mcg.py`大幅扩展。

### 5.5 配置与文档

- `configs/experiments/` 下新增SIV、SHU、Admission、评估spec、result registry等；
- standard-9、big、hurdle4、stability audit等source bank；
- 结果/审计/重构文档均位于`docs/`。

注意：`git diff --stat`只显示已跟踪文件；大量新模块仍是untracked。任何交接代理必须先读`git status --short`，不能把untracked当可删除垃圾。

---

## 6. 已支持、未支持与禁止主张

### 6.1 已支持

1. 当前主要性能通道是 reward-bearing warmup bootstrap，而不是已证独立的MCG蒸馏。
2. source暂时改变target-MDP的数据/状态分布，最终student source-free学习；这是迁移RL而非参数初始化。
3. source weighting在bank强分化时优于uniform；弱分化时边际价值小。
4. 扩源收益受complementarity与target headroom联合约束。
5. execution/occupancy与replay/update是不同持续时间的暴露通道。
6. actor/critic replay干预需要一致分布。
7. exact abstention在给定正确决策时可严格关闭source四类暴露，并通过basketball安全门。
8. hurdle/cabinet/maze存在不能全部用站立/存活解释的早期hard progress，但powerlift没有当前hard-skill证据。

### 6.2 未支持

1. 自动、校准的source-to-target transferability/ROI estimator。
2. `T0/Tonline/Tcritic/SHU`任一可通用判断长期learning utility。
3. Admission Core自动发现basketball应全部拒绝。
4. Admission Core稳定保留100k正迁移。
5. OBRW/student-as-arm保证回到scratch。
6. MCG蒸馏是与bootstrap同等扎实的headline贡献。
7. 普遍提高asymptotic ceiling或学会完整目标技能。
8. winner-take-all/阶段最优教师优于mixture；run24没有支持主动多源毒性。

### 6.3 明确禁止

- 隐藏basketball/window/stair/crawl/balance_hard负例/null；
- 把source posture或episode length当目标技能完成；
- 继续调SHU阈值、换task/source/horizon挽救mandatory contradiction；
- 用单learner-seed 2×2声称通用科学显著性；
- 把30k powerlift加速写成100k ceiling提升；
- 直接把current admission all称为paper主方法而不说明retention gate失败；
- 未经PI批准启动大矩阵或高成本训练。

---

## 7. 当前论文路线

当前 `paper_core_contribution_reconstruction_v3.md` 的默认论文结构：

1. **Reward-bearing Option Bootstrap（RBO）**：source-conditioned target-data acquisition；
2. **Source-bank-conditioned allocation law**：bank分化决定weighting价值，complementarity×headroom决定扩源价值；
3. **Dual-channel negative-transfer diagnosis**：execution与replay通道、AC coherence、OBRW局部控制；
4. **Broad HumanoidBench transfer regime map**：positive/null/negative/horizon-sensitive/saturation与hard-progress边界。

Admission Core v1 应暂时定位为：

- 已完成且solid的安全/生命周期substrate；
- exact fallback enforcement和replay provenance贡献候选；
- 尚未通过长期正迁移门的候选算法升级；
- 不能声称自动source selection。

MCG保留在supporting/appendix，主方法`bootstrap_only`关闭其性能干扰。

---

## 8. 推荐下一步：最小高信息量路线

### 8.1 先做零训练的 replay exposure audit

利用现存Admission 30k/60k/90k/final checkpoint，生成统一表/图：

- behavior authority；
- physical source occupancy；
- critic/actor source exposure；
- source sample reuse ratio；
- source撤出到完全覆盖的时间。

把当前新发现正式写入结果文档和method design。无需新GPU。

### 8.2 修 aggregate replay handoff，而不是只修来源内部recency

当前代码先固定candidate stratum mass：

`masses = admission_candidate_masses × available`

然后才在每个stratum内部应用priority/recency/uniform。因此recency不能把总source quota转给student。

建议定义：

`m_i^replay(t) ∝ admitted_i(t) × m_i^admission × g_i(t)`

其中source authority结束后`g_i(t)`下降，被移除质量全部转给student。必须保持：

- source执行期不破坏30k weighted bootstrap；
- exact revoke仍为0；
- actor/critic共享；
-不按top-return贪心；
-不引入新transferability estimator。

具体`g_i`应预注册，优先选择由buffer turnover/last execution定义、无需task调参的形式。需警惕仅做within-stratum recency的伪修复。

### 8.3 唯一建议新增训练

只加一个条件：

`powerlift admission-all + authority-aligned replay handoff` × seeds1/2/3 ×100k。

复用已有scratch、legacy WFix、fixed-quota admission-all对照；同时看30k和100k。

建议gate：

1. 保留30k显著早期加速；
2. 100k progress delta为正且达到原预登记统计门；
3. retention至少达到legacy WFix progress delta的0.5；
4. source critic exposure在authority停止后随lifecycle衰减，不再在物理20%时固定采50%；
5. 不重跑basketball exact-none。

### 8.4 结果分叉

- **PASS**：Admission Core可升级为论文主框架，再讨论真正source utility/admission estimator；
- **FAIL**：停止性能升级，Admission Core作为安全/机制appendix；论文主算法固定为静态RBO，进入v4四件套和claim audit。

### 8.5 为什么现在不先做selector

powerlift all已经近似“oracle允许所有已知正向source”，但100k仍未保留。selector只解决“选谁”，不能修复“source停止后旧数据如何交权”。先修明显lifecycle mismatch比再造未验证分数信息量更高。

---

## 9. 运维、日志与外部可视化

### 9.1 Admission FINALV2 本地日志

- `logs/train/admission_core_v1_20260712TFINALV2Z/`
- `logs/probe/admission_core_v1/`
- `artifacts/admission_core_v1/`

### 9.2 W&B

Project：`https://wandb.ai/yujiajie-nju/fasttd3_ptf`  
Group：`admission_core_v1_20260712TFINALV2Z_backfill`

Runs：

- basketball s1：`acv1bf-39a9404f3d88`
- basketball s2：`acv1bf-75dd2763bf97`
- basketball s3：`acv1bf-09d05479763d`
- powerlift s1：`acv1bf-e1b742c23c35`
- powerlift s2：`acv1bf-ec708abb67b8`
- powerlift s3：`acv1bf-24764abfc667-r1`

这些是事后回填run，不是原始live W&B；scientific values/global steps精确，event timestamps按原起止时间插值。

### 9.3 实验监视

此前正式实验使用tmux/orchestrator/finalizer，W&B被显式关闭，导致PI不能在线查看。未来默认：

- `WANDB=1`并保留本地log；
- tmux/orchestrator负责进程；
- run card写清监控、结束条件和finalizer；
- Codex/Claude跨会话不会被本地进程自动“唤醒”，需通过共享文件/外部调度继续交接。

### 9.4 当前磁盘

已清理约47.52 GiB旧权重；不要因为旧intermediate缺失而重新跑大实验。需要历史曲线先查W&B、日志、JSON/Markdown artifacts和final checkpoint。

---

## 10. 当前已知文档/状态不一致

1. `configs/experiments/admission_core_v1.yaml` 顶部仍写 `status: preregistered_not_run`，但FINALV2已完成；权威状态是result/completion audit。应做一次小文档修复。
2. `docs/agent_collab/research_tasks.md` 主要是旧Step-2 anchor-xattn路线，不能当当前RBO/Admission任务板；本次新协作以新共享对话文件和本文为准。
3. 工作树包含大量untracked研究产物；尚未形成干净commit。Claude在建议提交前必须先按模块审计scope，不能把所有dirty文件一锅提交。
4. SIV失败首轮的`.pt`在磁盘清理中已删除，但正式gate report/results仍在；文档中的failed artifact路径是历史审计引用，不能假设权重仍存在。

---

## 11. 给 Claude 的第一轮审查任务

请不要立即修改代码或启动训练。先以独立审稿人和算法设计者身份回答：

1. 是否同意“fixed source replay stratum mass造成post-warmup stale-source amplification”是powerlift retention失败的首要可检验机制？请审查代码和checkpoint计数，寻找替代解释。
2. authority-aligned aggregate replay handoff应采用什么无task调参公式？比较：
   - 与active physical occupancy封顶；
   - 按last-source-execution指数衰减；
   - 按buffer turnover解析衰减；
   - 直接在bootstrap结束时回归uniform physical sampling。
3. 最小3-seed powerlift实验是否足以裁决？还缺哪个不需新训练的对照？
4. 如果该修复成功，它是否足以作为论文核心机制，还是仍只是engineering lifecycle correction？
5. 如果失败，静态RBO四贡献结构是否足以成稿？其中最薄弱的claim是什么？

Claude应把回复追加到新共享对话文件，不改写本文或既有轮次。

---

## 12. 协作规则

- Human PI决定高成本实验、方向变化和对外claims；
- Claude默认独立批判审查、替代解释、方法新颖性和实验设计；
- ChatGPT/Codex默认代码核验、实现、实验编排、统计复算和证据整合；
-每轮先读共享对话最后完整turn，写前刷新文末，只追加自己的真实回复；
-高成本训练前必须有run card：命令、假设、预算、成功/失败/停止条件；
-任何结论必须区分fact、inference、proposal。

---

## 13. 最终一句话状态

**本轮已经把“有害source无法退出”的机制缺陷修成了可严格弃权、可隔离probe、可追踪并可撤销的数据生命周期底座；实验同时证明它能消除basketball的强制有害暴露、保留powerlift的30k加速，但固定source replay配额使旧source数据在行为authority结束后仍被过采样，100k正迁移未保住。下一步只应修这一个aggregate replay handoff缺口并用powerlift三种子裁决，不能再回到未经验证的万能迁移性标量。**
