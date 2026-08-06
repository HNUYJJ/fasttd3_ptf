# 原始 Git 提交历史（时序证据）

> 本仓库是 2026-08-06 重建的干净历史（原仓库因体积问题重建）。
> 原始开发历史共 215 个提交，完整时序记录于此。
> 本项目方法论要求**判据必须先于数据冻结并提交**（见 CLAUDE.md §4），
> 该时序可由下表逐条核对。

```
2026-08-06 06:10  6af07f2  撤回 stair/slide 跨 target 对照：J_best_known 口径缺陷（winner's curse）
2026-08-06 06:01  839d014  修正 crawl 归因口径 + 记录普查为 measurement verdict
2026-08-06 05:58  3235581  修正普查脚本的两个实现缺陷（使其符合已冻结预注册，未改判据）
2026-08-06 05:47  495b311  场地普查 INSUFFICIENT_SITES：三个正面 target 全部已超官方 success_bar
2026-08-06 05:45  ba790cc  冻结场地普查脚本（先于生成任何结果）
2026-08-06 05:43  3070c78  方向重估 + 场地普查预注册（判据先于任何数据冻结）
2026-08-06 01:02  5b993c2  P3 成文：论文正文 v1（双支柱 + 逐条边界）
2026-08-06 00:57  f794097  证据总表同步 P1/P2：新增两条正面结果，缺口清单与 crawl 覆盖状态更新
2026-08-06 00:56  4f53b0d  证据总表与 CLAIMS 同步 P1/P2 结果
2026-08-06 00:55  7ffcf06  ENDTOEND_SUPPORTED，但净收益高度依赖成本口径（两个口径都报）
2026-08-05 21:25  aec8b84  P3：论文章节结构与图表清单
2026-08-05 20:16  d334e21  自查修正 I-5：sibling gate 两个方向的稳健性不对称，必须分别陈述
2026-08-05 20:02  45775ed  自查修正：racing 成本的两个口径不一致，补 §2.1 澄清
2026-08-05 19:55  0842159  P3 基础件：主张—证据—边界的逐条映射
2026-08-05 18:38  a7a3893  Freeze end-to-end eval + adjudication scripts before any new data
2026-08-05 18:35  f5a669a  Pre-register end-to-end system + freeze automatic decisions before any training
2026-08-05 18:28  aa25440  ADMISSION_VIABLE: racing 一次 K=10000 测量可同时决定准入与选源
2026-08-05 16:34  8479ba5  EXPERIMENT_LOG: 补 Phase 8（07-29 → 08-04），此前总表停在 Phase 7
2026-08-05 16:09  55d1423  Pre-register racing admission gate (criteria frozen before crawl/slide data)
2026-08-05 16:02  c491db7  P0 收尾：补齐三份结论文档，族 12 并入不可能性刻画，更新证据总表
2026-08-05 09:15  9119836  gitignore: 补 papers/ 规则的落盘
2026-08-05 09:15  1ee5099  gitignore: 只跟踪代码与配置，权重/检查点/论文一律排除（本地文件保留）
2026-08-05 08:23  1f01705  Archive docs, agent-collab records, eval data and anchor audit json
2026-08-05 08:23  23d41f2  Archive experiment configs and eval-script changes
2026-08-05 08:23  a9e9741  Archive pending mechanism changes: classic-PTF option/termination path
2026-08-05 08:23  268f5dc  gitignore: exclude anchor bundle tensors (artifacts/**/*.pt)
2026-08-05 08:02  5ae3d3b  Correct the mechanism claim: run did not cross the crawl tunnel
2026-08-05 07:22  661d167  Record progress-screen HOLDOUT_FAILED: behavior and learning value invert 7.9x
2026-08-05 07:14  788afa0  Pre-register zero-training progress screen (criteria frozen before any data)
2026-08-01 13:19  cab7c50  Freeze Slide hard-exit scientific gate
2026-08-01 13:08  62a6199  Add complete branch-anchor hard-exit path
2026-08-01 04:50  de9f366  Record bounded Hurdle selection-value result
2026-08-01 04:43  e7f11c4  Freeze Slide speedup execution and adjudication chain
2026-07-31 05:33  d90d9b1  证据总表:加入 slide GEN_OK 与 target 覆盖的诚实盘点
2026-07-31 05:31  a5fcd9b  Pre-register slide speedup:把 hurdle 的加速口径搬到第二个 target
2026-07-31 05:27  efcc882  slide 可推广性审计 GEN_OK:首次具备真正的 crossover;并发现基线相消机制(M32)
2026-07-31 04:17  be36115  Freeze hurdle 选源价值裁决脚本(先于 stand 臂任何评估)
2026-07-31 04:08  16fccf1  Freeze slide 可推广性审计裁决脚本(先于 seeds 4-6 任何评估)
2026-07-31 04:04  60b5159  CLAUDE.md §0.1: Codex 审查必须 --resume,禁止 --fresh(PI 要求)
2026-07-31 03:46  06f1adf  Pre-register hurdle 选源价值实验:补齐端到端链条的最后一环
2026-07-31 03:44  66b0136  补 §4.1:L4 不是 racing 的威胁而是它的理由
2026-07-31 03:42  bfa97f3  把 L1-L4 组织为一个重新刻画:迁移效用不是 (source,target) 的属性
2026-07-31 03:39  2c1804f  Pre-register slide 标签可推广性审计(M31 的首次实操)
2026-07-31 03:33  1483942  把 M31 回写到受影响的既有结论与证据状态总表
2026-07-31 03:32  0d594ad  RACING_REJECT v4 揭盲: PARTICIPANT_DIVERGED —— door 的"一致有害"不推广到新 learner
2026-07-31 01:34  50e6409  Freeze RACING_REJECT v4 裁决脚本(split-sample,先于任何新臂被评估)
2026-07-31 01:18  38311fb  Pre-register RACING_REJECT v4: split-sample 重做(Codex 裁定的唯一合法路径)
2026-07-31 01:14  b66e6cb  撤回 RACING_REJECT v3(未执行):outcome-contingent gate switching;加 M30 与 CLAUDE.md §8.6/8.7
2026-07-31 00:53  01dd93e  Pre-register RACING_REJECT v3: 前置门改用符号判据,裁决 door 拒绝主终点
2026-07-31 00:51  dd63187  M29:自查抓住一个假模式(未遂,未投入实验)
2026-07-31 00:49  7b2c03c  作废 RACE-then-RUN(未执行):argmax U 与 argmax J 恒等等五条致命缺陷
2026-07-31 00:23  35c3bb4  证据状态总表:一页看清已裁决/未裁决/缺什么
2026-07-31 00:21  a1f8961  RACE-then-RUN 设计草案(待 review,未实现)
2026-07-31 00:18  0ea1adf  docs: RACING_REJECT v2 收尾
2026-07-31 00:17  05c9002  RACING_REJECT v2 结果文档 + M27/M28
2026-07-31 00:16  67d139c  RACING_REJECT v2 揭盲: REPLICATION_DIVERGED,主终点不予裁决
2026-07-30 23:14  5546ce2  CLAUDE.md 增设计层检查 §8;追加 M25-M26
2026-07-30 23:13  dbed275  作废 RACING_MULTI(未执行):辨别设计不成立 + 剂量混淆 + 无独立重复
2026-07-30 22:51  76a19f6  Pre-register RACING_MULTI: racing 的选源正确性能否跨 target 成立
2026-07-30 22:47  06d37dc  按 Codex 二轮 review 收口:R2' 盲态边界 + R7 异常兜底 + 删死代码
2026-07-30 22:07  806cfa8  重写 RACING_REJECT v2 裁决脚本:按修订 v2.1 的 R1-R5 全部落实
2026-07-30 22:05  971b6af  预注册修订 v2.1(仍未揭盲):按 Codex 实现后 review 消歧并补严工程验收
2026-07-30 21:42  46566af  Implement RACING_REJECT v2 裁决脚本(三层验收),待 Codex 实现后 review
2026-07-30 21:34  ac1bd26  Pre-register RACING_REJECT v2: 含 Codex APPROVE_WITH_FIXES 的全部 6 项修复
2026-07-30 19:05  f3c708a  RACING_REJECT v2 设计草案(待 review,未实现)
2026-07-30 19:01  da32b13  作废 RACING_REJECT v1(未揭盲):主假设逻辑上不可证伪
2026-07-30 18:40  67071e1  建立强制执行点 CLAUDE.md;追加 M20-M24
2026-07-30 16:47  82c9844  Freeze RACING_REJECT v1 训练/评估/裁决脚本(先于任何臂被评估)
2026-07-30 16:46  18a20cb  Pre-register RACING_REJECT v1: racing 能否在全负场地正确拒绝(door)
2026-07-30 16:41  3adda86  不可能性刻画补上另一半:RACING_K 已完成,零成本预测不可行但最小成本测量可行
2026-07-30 16:40  a744adb  RACING_K v1 结果: RACING_VIABLE K*=10000 -- 独立重复推翻了单批的 K*=5000
2026-07-30 15:16  a1c2829  裁决脚本仅参数化数据目录(判据逻辑不变,重跑 compressed_lr 结果一致)
2026-07-30 15:13  11ff656  更正 437bd50 的归因错误:LR 日程并未被压缩;rck2 改作独立重复
2026-07-30 14:54  437bd50  RACING_K v1 (compressed-LR 版): RACING_CHEAP K*=5000 -- 但发现 LR 日程缺陷
2026-07-30 11:49  ecadb75  Add idempotent RACING_K training driver (并行度硬上限 3)
2026-07-30 11:38  88a5e26  Freeze RACING_K v1 训练/评估/裁决脚本(先于任何臂被评估)
2026-07-30 11:34  6776c03  Pre-register RACING_K v1: 自动选源的最小测量代价 K*
2026-07-30 11:28  76dcd16  Close Competence-Gated Transfer before implementation -- 第四次行为代理换皮
2026-07-30 11:13  469c1fb  Probe: hurdle 源天花板 -- student 在 ~15k 就超过源,但剂量全程恒定 50%
2026-07-30 11:06  20f1e11  hurdle speedup v1: SPEEDUP_CONFIRMED -- 3.5-4.4x early sample efficiency
2026-07-30 06:26  d66cee1  Add dose audit for the hurdle speedup experiment
2026-07-30 04:53  2e09246  Anchor the negative result in the transferability-estimation literature
2026-07-30 04:05  1730377  Characterize the failure of transfer-utility prediction across eleven signal families
2026-07-30 04:02  1fcf136  Freeze hurdle speedup evaluation and adjudication before any long-run arm is evaluated
2026-07-30 04:00  ba7a7de  Pre-register hurdle sample-efficiency speedup measurement
2026-07-30 03:55  a4adbb7  Spec-matching hypothesis REFUTED before any experiment; record two method failures of mine
2026-07-29 18:00  f80f483  Record critic-first bridge feasibility failure
2026-07-29 16:43  72b947a  Add preregistered critic-first bridge bootstrap gate
2026-07-29 15:24  eb0f6b8  QMP-fidelity v1: PARTIAL -- per-state Q-switch degenerates to the student
2026-07-29 14:28  0d93e6d  Freeze QMP-fidelity evaluation and adjudication before any arm is evaluated
2026-07-29 14:25  be7cedf  Implement QMP-fidelity behavior-only mode with verified classic-PTF isolation
2026-07-29 14:07  1648b9c  Revise QMP-fidelity run card per CONDITIONAL_APPROVE
2026-07-29 13:47  f6d0647  Correct probe verdict to INCONCLUSIVE; draft minimal QMP-fidelity run card
2026-07-29 12:45  dbff013  per-state Q-switch probe: PROBE_REFUTED by the frozen criteria, but the probe has no decisive power
2026-07-29 12:37  afdc001  Pre-register per-state x per-body-group Q-switch probe; pivot to QMP-style non-aggregated selection
2026-07-29 07:47  db719e8  Close sibling-source gate: DIRECTION_DEPENDENT — shared reward implementation is not a robust prior
2026-07-29 06:55  22090fb  Pre-register Slide<->Stair sibling-source gate; correct taxonomy scope and naming
2026-07-29 06:23  b014db4  Stage 2: project existing U labels onto the task graph — not testable with current data
2026-07-29 06:22  77a567a  Build HumanoidBench task taxonomy v1 (stage 1: static extraction only)
2026-07-29 06:08  d2fd97d  Pre-register the task-taxonomy feature schema before any extraction
2026-07-29 06:07  9ca9c77  Correct two factual errors and one overstated hypothesis in the task-relatedness note
2026-07-29 05:56  91c04c1  Find where task relatedness actually lives: reward-algebra isomorphism does not imply transferability
2026-07-29 05:25  05474a6  Retract the BTE-as-mediator claim; narrow three overstated readings
2026-07-29 05:08  07d0d20  Locate the ceiling of reward-side predictors: behavioral transfer efficacy
2026-07-29 04:38  60711bc  Stop BAC as a transferability metric: no increment over simple baselines
2026-07-29 03:57  3b450a6  Close stair BAC replication: PARTIAL by rule, but the gate had no decisive power
2026-07-29 02:47  a060d80  Clarify BAC's applicability boundary: additive tasks are structurally unseparable
2026-07-29 02:35  6408532  Add self-contained BAC review packet with a sensitivity analysis
2026-07-29 02:33  32f66d0  Pre-register stair BAC replication (adjudication frozen before any arm runs)
2026-07-28 19:38  e0f07a1  Close slide BAC gate: BAC_SUPPORTED; retract the C(dose) hypothesis
2026-07-28 18:17  45e5821  Add slide BAC gate source-free evaluation script
2026-07-28 18:14  969f998  Add focused unit tests for bottleneck-aligned coverage
2026-07-28 18:11  75772e8  Pre-register slide BAC decision gate (adjudication frozen before any arm runs)
2026-07-28 17:54  33d4c92  Pre-register bottleneck-aligned coverage; freeze forward predictions
2026-07-28 12:26  27c1643  Add episode-prefix option handoff; pre-register Door placement ablation
2026-07-28 05:01  995bc2a  Close the transferability-metric line; restructure the paper contributions
2026-07-28 03:48  e04d540  Close Door channel decomposition (UNRESOLVED: cross-seed mechanism heterogeneity)
2026-07-28 02:55  1175833  Decouple replay eligibility from behavior authority; pre-register Door decomposition
2026-07-27 17:29  3c66b47  Close Door@10k equal-dose calibration gate (DOOR_ALL_SAME_SIGN)
2026-07-27 14:50  5944792  Pre-register Door@10k equal-dose gate (adjudication frozen before unblinding)
2026-07-27 13:51  e415ffe  Add zero-training label-identifiability audit (CANDIDATE_FOUND: door)
2026-07-27 11:35  0b7b5ae  Close Cabinet@10k equal-dose calibration gate (CABINET_UNCERTAIN)
2026-07-20 04:32  a5de1a3  Close Phase-1 bounded bank lease experiment
2026-07-19 12:59  e0c9d4b  Freeze Phase-1 bounded bank lease execution matrix
2026-07-19 12:55  c5eb228  Record passing Phase-1 Gate A
2026-07-19 12:54  5a60e98  Implement bounded-lease Gate A equivalence harness
2026-07-19 12:45  a6e7a3b  Freeze Phase-1 replay lifecycle audit tolerances
2026-07-19 12:45  b56cfe6  Freeze Phase-1 SESOI margins after PI approval
2026-07-19 12:33  61070c7  二十三次复核修订(run card v0.6):finalize 改完整科学 payload 逐位比较(只排除 status/git_head/git_dirty/finalized_from_candidate;旧版只比汇总数→输入ckpt/generator被换而汇总不变仍会通过);门B补全同配置验证(mcg/warmup_mode/ablation/warmup_min_steps/replay三参数/task-specific student_logit/6项训练规模+eval_interval,完整 candidate masses 向量跨全部ckpt+seed一致);conditional 字段(admission_adaptive 等5项仅7-14版存在,truck 7-13版无)采用'存在则验缺失如实记'不夸大验证范围;反例单测26→52项(全仓210);修文档残留 ε=0.01→ε_frozen=0.001
2026-07-19 12:06  9b45f11  二十二次复核四阻塞修订(run card v0.5):ε 由硬编码0.01改 Hoeffding 统计式(M=24/α=0.001/N=6.55e8 新跑臂最小区间→ε_raw 9.068e-05→冻结0.001,与ChatGPT核算一致)+剂量带诚实标注为工程风险预算;两脚本补输入身份验证(24ckpt 逐项 env/seed/step/bank/mode/warmup/handoff/masses+SHA256)+generator SHA+HEAD/dirty;δ补自动断言(wandb run 唯一匹配/patch 剔除离线probe后逐位一致/历史PTF实为 mcg=False execute_sources=False bank空);补 --finalize 安全冻结流程;反例单测26项(全仓184);80k口径分离(历史NA/新跑臂全链)
2026-07-19 11:38  459c24f  二十一次复核:scratch 双层裁定(更正我的'无 provenance'错判——W&B 档案完整:六份入口代码 SHA 全同/git base 全同/CLI 纯 scratch/曲线与 logs 逐条相同;truck s1 diff 差异仅 probe 脚本不触训练路径)+δ 候选(SESOI 定位,basketball 45.786/truck 36.518,绑定 W&B provenance)+门B容差ε预冻结(历史 retention 实测,ε观测越界0.0→冻结0.01)+run card v0.4+E18(否定断言须穷尽检索)
2026-07-19 08:01  20e8058  run card v0.3(二十次复核)+scratch 兼容性审计:更正 execution_counts_at_apply 确在 decision_history(我 v0.2 只查 policy_events 误判)→门B混合取证;门B数学化(q_k/r_a:b/ε包络,90k/100k不要求main source>0);save_interval=0 防同名覆盖+checkpoint steps 逗号;门A有限harness(snapshot_at静态+80-100步不真跑30k);scratch REUSE_FAIL无provenance→层5标签降描述性+POSITIVE_GAIN_LOST收紧+SCRATCH_COMPARISON_UNCERTAIN;task selection purposeful边界
2026-07-19 07:34  8b952ad  run card v0.2(十九次复核五修):判序精确公式(nAUC/paired CI/δ/五级分级)+门B扩双臂treatment audit(对抗发现 execution_counts_at_apply 不存在→改 checkpoint admission_audit 差分)+门A加强end-to-end CPU等价+评估口径修正(5k-95k 19点/删100k endpoint)+scratch标签公式化;δ交叉核算 basketball 45.786/truck 36.518 待脚本冻结
2026-07-19 07:00  008521b  十八次复核:truck REUSE_FAIL 接受→9 条档定案(basketball 复用+3新跑/truck 两臂6新跑,均 HEAD+schedule);规格 v0.4(truck 0-30k 同分支+basketball CPU 等价前置门);E17 强化(脏树须存源码快照非仅哈希);Phase-1 run card v0.1 草案
2026-07-19 06:35  bd0d284  规格 v0.3(十七次四修:预算三档/归因不越界/层级判序/eval协议审计项)+基线复用兼容性审计(零训练):basketball REUSE_PASS强(a5cec9d 逐位锚定+3commit中性),truck REUSE_PASS限定(replay通道逐位等价,行为通道靠34项认证间接);hard-exit须用HEAD(历史无AdmissionSchedule)
2026-07-19 05:58  fda359a  规格 v0.2:十六次复核六点修订(双通道 B/R 状态+basketball_static 基线补漏+条件性 6 条+兼容性审计前置+主窗口 35k-80k+T2 边际 estimand+判序 7 分支)
2026-07-19 05:35  b242198  审计 v2.1 两处小修+bounded bank lease 规格 v0.1(T1+T2 bank 级/零新代码载体/单因素验证/基线考古:truck+powerlift retention 臂可复用,basketball 两臂需新跑,正任务修正为 truck)
2026-07-19 04:34  fdfced9  机制审计 v2:修正 v1 中心结论错误(schedule 已可表达 T1/T2 与延迟窗口)+三种 TTL 语义区分+证据两列拆分+critic 措辞收窄
2026-07-19 03:49  7627271  Phase-1 机制现状审计:六项逐条(四强证据/TTL 半缺/窗口粗粒度)+统一规格前待决清单
2026-07-18 18:11  8ea719b  十二次复核 Minor revision 修复:仿真输入改冻结配置重算(ckpt仅断言)+脚本入库 scripts/analysis+结果入 docs/data+措辞收窄;posthoc 文档升级 FINAL v1.0
2026-07-18 16:32  3012b1f  P0 终局:预注册 ENGINEERING_INVALID 保持+posthoc 冷启动机制分析(DRAFT 待复核)+条件结论 F-a+E16
2026-07-18 04:47  633eeca  gitignore 补 P0 smoke/anchor bundle 目录(manifest/checksums 非 pt 文件;冻结 plan 执行要求干净树)
2026-07-18 04:45  93fc72f  P0 δ 正式冻结:crawl 33.556/truck 28.847(预注册 scratch 3-seed 日志,10k-15k 窗口)
2026-07-17 15:57  48cb1db  Harden P0 evidence and execution pipeline
2026-07-17 15:17  4e9dc57  八次复核修复:smoke 顺序/进程组清理/资源采集/duplicate 事务化/裁决器五绕过/冻结 plan 执行
2026-07-17 13:51  e03efdc  七次复核修复:GPU 队列重写+裁决器证据闭环+duplicate 归档定稿+smoke 隔离模式
2026-07-17 10:12  9570a5b  P0 执行包收口:五次+六次复核修复(checkpoint completed-step/配置断言/segment 续接/裁决器入口验证+反例/orchestrator/evaluator 身份验证/δ 冻结强化/manifest 时机)
2026-07-17 05:50  a49e423  P0 core-only anchor-resume 实现+run card v2.1.2+七项等价性测试全过
2026-07-17 00:13  3b0f349  Run card v2.1.1: 三次复核定点修订(判序矛盾/等价测试参照/RNG与noise_scales语义/provenance保留/d_dup/checkpoint列表/冻结参数与treatment审计)
2026-07-16 18:17  717d928  P0 run card v2.1:core-only anchor-resume 设计+estimand 据实改写+双 gate 拆分
2026-07-16 17:49  9858263  P0 run card v2:改为 true online lease fork(方案B),修正 v1 全部复核缺陷
2026-07-16 17:33  8317c86  P0 run card v1 草案:SIV-v2 counterfactual oracle 最终可行性审计(待 ChatGPT 交叉复核)
2026-07-16 17:24  07aa2ea  按 PI 确认重新删除四个文件(PI 本人有意删除,2026-07-16 一手确认)
2026-07-16 17:20  9a74815  第三轮对抗审查正式收敛:接受失败类型分解+部署四gate+lease预先定长,联合建议定稿待 PI 授权
2026-07-16 16:30  7826bdc  第二轮对抗审查收敛:接受 8 项修正,坚持 2 点(P0=lease框架共同必要条件/K尺度失配),统一建议提交 PI
2026-07-16 16:27  81c8c9d  恢复 8a2d441 中未披露的四个文件删除(审计链修复)
2026-07-16 16:05  8a2d441  组件①方案对抗审查闭环:ChatGPT Major revision + Claude 反向审查 → v2 融合框架
2026-07-16 15:25  8ed6de2  方案 v1 增补 2025-26 最新文献:IIF(NeurIPS25 Oral)/WSRL(ICLR25)/ARB/QoQ + A+ 短驻留变体
2026-07-16 15:10  144b3ce  组件①重攻方案 v1:源注入判定/时机/事后无害化(四层防线+离线考古验证设计)
2026-07-16 09:33  00d6666  文档重组:整合三大主文档,历史文档归档,顶层清理
2026-07-16 09:20  04f4b0d  代码清理:删除全部已弃用路线代码,简化活跃模块分支,补关键中文注释
2026-07-16 05:53  a5cec9d  Pre-cleanup snapshot: 全量保存整理前状态(139 个变更/未跟踪文件,便于清理后可完整恢复)
2026-06-15 05:26  b183f40  Ablation: wfix banks (weighted source, fixed horizon=25) to decouple variables
2026-06-15 05:20  4ba7a68  Stop tracking MUJOCO_LOG.TXT (mujoco runtime log noise)
2026-06-15 05:20  c7af8df  Docs: RBO-PTF diagnosis, Source-Target-Effect Map, handoffs
2026-06-15 05:20  b4f0b73  Add multitask aggregation (safe/rand/scratch AUC, ROI, variance)
2026-06-15 05:20  c40c9db  Generate safe/uniform source banks for safe_bootstrap
2026-06-15 05:20  e0e391a  Add task-progress audit + Transfer Map v2 snippet probes
2026-06-15 05:20  561b340  RBO-PTF: safe_bootstrap warmup mode (reward-weighted source selection)
2026-06-14 04:21  5c7d6d9  Handoff RIC v2 for ChatGPT: ablation results (bootstrap≈full, gate=safety), 9-task table, window high-variance
2026-06-14 04:03  557f2ad  Safety 2-seed: window is high-variance (+76/-27), motivating safe-horizon bootstrap
2026-06-14 01:48  2f1977f  Boundary-task pilot: 9-task main table complete, Transfer Map predicts the full opportunity gradient
2026-06-13 21:33  4c6599e  Ablation 2-seed: bootstrap≈full, gate is safety not clean-task performance
2026-06-13 17:38  00825d0  Ablation: bootstrap_only captures 65-112% of full gain (mean ~90%)
2026-06-13 10:54  4ba9305  Ablation: bootstrap_only captures 65-112% of full gain (mean ~90%)
2026-06-13 09:15  a95bf48  Extend aggregate to multi-method comparison (scratch baseline + boot/nobo/full)
2026-06-13 07:27  3f6a862  MCG ablation switch: full / bootstrap_only / no_bootstrap
2026-06-13 07:12  d3df6ee  Handoff RIC v1 for ChatGPT: Transfer Map + wide pilot positive results + warmup-bootstrap finding
2026-06-13 06:54  d9d5241  Safety control yields positive transfer (window +76%, balance +20%); mechanism = warmup bootstrap
2026-06-13 06:50  9fd4bba  Wide pilot v1 results: SC-MCG positive transfer on four opportunity tasks
2026-06-13 02:35  e0ff261  Safety-control source banks: window/balance_hard loco banks for negative-transfer test
2026-06-12 18:23  721ab6f  Wide pilot infra: per-task loco source banks + opportunity-cost analysis
2026-06-12 16:18  6690b5e  Transfer Map: info-component layer analysis (reach reaches the package box; run drives door passage)
2026-06-12 14:08  137a0a6  Transfer Map v1 analysis: four transfer-opportunity patterns across 17 HB tasks
2026-06-12 13:59  bf57b05  RIC-PTF groundwork: HB task layout census + Cross-Task Transfer Map probe
2026-06-12 13:23  9af565e  Probe 1 verdict: BC fits demos (MSE 0.13 << var 0.81) but rollout fails on distribution shift
2026-06-12 13:15  b88ed1a  EODT day-1 probes: stage-memory oracle (Probe 3) + demo collector + BC-only trainer (Probe 1)
2026-06-12 12:44  e38e1a5  Handoff v2 for ChatGPT-5.5-Pro: SC-MCG results + package campaign + next-step candidates
2026-06-12 09:48  6258c8b  Nearcarry v3: long-range dest (0.25-2.0m) + dual-scale progress + 1000-step episodes
2026-06-12 07:44  1cc1299  Chain warmup: episode-level demo + initiation-as-scheduler (round-2 lesson)
2026-06-12 05:30  208682f  Bank v3: point nearcarry at v2 checkpoint (acceptance: to_dest 81% vs v1 6%)
2026-06-12 03:55  94acdf9  Initiation-aware warmup + nearcarry reward v2 + to_dest matrix column
2026-06-12 00:15  6618884  Near-carry source env: third interface link (move floor box to near dest)
2026-06-11 23:00  de4b02d  Contact v2 passes acceptance: 100% contact in its initiation set
2026-06-11 21:32  9e3882a  Contact reward v2: crouch-friendly (not_fallen replaces stand-upright)
2026-06-11 20:03  7ccf2ad  Interface-aware contact source env + training entry
2026-06-11 20:00  e578dbf  Package source coverage matrix probe (with zero-action control)
2026-06-11 17:47  7a43898  v1.2: distill weight = hard gate x confidence
2026-06-11 15:47  641df42  Teacher relevance pre-screen probe (warmup budget gate)
2026-06-11 15:45  2cc108e  MCG v1.1 safety patch: significance-calibrated gate (SC-MCG)
2026-06-11 11:56  eca5849  Door phase-funnel evaluator (approach/handle/open/passage from obs)
2026-06-11 11:40  3916a3e  Add empty source bank for paired scratch controls
2026-06-11 09:57  28015c7  MCG v1: modular critic-guided transfer (option -> (teacher, body-group))
2026-06-11 03:45  271f419  Pilot v3 lessons: execute FULL teacher action; unfreeze beta + epsilon floor for v3.1
2026-06-11 01:58  e9390bb  Add reusable checkpoint evaluator for package (probe-based success rate)
2026-06-11 01:43  b947d9e  Add behavior-level call-and-return: --ptf-execute-sources (+ option min-steps)
2026-06-10 23:16  914f561  Build package coverage source bank; launch PTF pilot v2 + scratch control
2026-06-10 17:51  86acaa2  Approach reward v2: ring-zone target + box-calm term (v1 chased and punted the box)
2026-06-10 16:15  c0a1b50  Add approach-source training pipeline (plan A: auxiliary-reward pretraining)
2026-06-10 16:07  a17995c  Near-box probe: manipulate-stage skills DO transfer zero-shot
2026-06-10 15:48  6914f8f  Zero-shot transferability probe: push->package skill overlap is zero
2026-06-10 12:53  96e5480  Add repo structure map (docs/REPO_MAP.md) and rewrite README
2026-06-10 12:50  2e7aded  Remove legacy my_fasttd3_ptf line and its private ecosystem
2026-06-10 12:47  d7451d4  Decouple official PTF path from legacy my_fasttd3_ptf package
2026-06-10 12:43  40b04cc  Baseline: full repo snapshot before structure cleanup
```
