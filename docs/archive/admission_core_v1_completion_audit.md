# Admission Core v1 completion audit

> 审计原则：只有当前代码、测试、checkpoint 与配对评估 artifact 才算证据。实验失败也必须按预注册阈值如实记录，不能把“完成验证”偷换为“两个性能 gate 都通过”。

| 原始要求 | 实现证据 | 工程验证 | 运行证据 | 当前裁定 |
|---|---|---|---|---|
| Student-inclusive source admission | `AdmissionSnapshot.candidate_probabilities`；`McgBehaviorController.admission_probabilities`；`admission_bootstrap` 直接在 sources+student 上做一次 categorical sampling | none one-hot、static mask、student/source sampling tests | all smoke 中 student mass=0.5；none smoke 中 student mass=1.0 | 已证明 |
| 无外层固定 0.5 teacher | `admission_bootstrap` 不读取 `warmup_exec_prob` 决定 teacher/student；student logit 显式进入同一 softmax | controller sampling tests | powerlift all checkpoint 的 candidate masses 显式保存 | 已证明 |
| Exact abstention | 静态空 admission 与 empty-bank scratch 共用 target-only fast path；source/option/MCG execution/update 全部跳过；全 student replay 使用 scratch `randint`；写 replay 前保留拒绝断言 | 57项核心测试全过；CPU 首次更新后 learner state 与 scratch 逐 bit 相同；CUDA 不主张长程逐 bit 等价 | FINALV2 basketball 三个 seed 的30k/60k/100k均为source execution/replay/critic=0；100k negative-safety gate PASS | 已证明 |
| Quarantine/main-replay 隔离 | probe config/bank validators 强制 quarantine-only、0 learner updates、0 main replay writes；admission manifest 只绑定 artifact digest | `test_source_admission.py` 与 manifest digest tests | 已有 quarantine artifact/report；训练入口不存在 quarantine transition import | 已证明 |
| Admission-consistent main replay | 仅 student/admitted provenance 有 active mass；来源配额不依赖历史数量；来源内 recency/TD priority/uniform | quota、recency、priority、snapshot tests | 三个 powerlift seed 的30k→60k lifecycle 均证明 warmup 后source execution冻结、旧源数据自然衰减且actor复用critic batch | 已证明 |
| 撤销后旧 source 退出 active replay | 显式 step schedule 同时更新 behavior 与 replay；释放 latch；per-group contributing source 全部参与 allowed check | runtime revocation 与 mixed-MCG revocation tests | step20 后 source execution=0、source critic sample=0；320条旧数据物理保留但 active=0 | 已证明 |
| Actor/critic 一致采样 | admission 模式禁止 legacy split/reweight；actor 复用 critic batch | 配置互斥与 replay role tests | checkpoint `actor_sampling=shared_critic_batch` | 已证明 |
| 可选 MCG 后置接口 | gating `source_mask`；动态 admission update；每组 source provenance；任一 contributing source 撤销即过滤 | masked gating、latch revoke、mixed provenance tests | MCG-full smoke 跨过 warmup 并完成 actor/critic 更新 | 已证明（性能非主张） |
| 最小负迁移实验 | basketball 9-source exact-none，3 seeds，30k/100k | finalizer + fixed paired eval spec | 100k progress Δ=+0.03125，旧WFix=−0.09375，recovery=+0.125；三项安全检查全过 | 完成，**PASS** |
| 最小正迁移实验 | powerlift 9-source admission-all，student aggregate mass=0.5，3 seeds，30k/100k | fixed paired eval + preregistered retention gate | 30k progress Δ=+0.0003879、t=4.318；100k Δ=+0.0001017、t=0.600，retention=0.1996 | 完成，100k gate **FAIL** |
| 最终科学裁决 | exact reset pairing、完整 factorial coverage、预登记性能阈值 | 1152 records、0 duplicates、每 condition 192/192、双RNG协议唯一且有效 | basketball safety PASS；powerlift retention FAIL；both=false | 已完成，部分成立 |

## 当前不作出的主张

- 不声称已有自动迁移性指标；`all/none/static/manifest/schedule` 是外部显式 admission 决策接口。
- 不把旧 T0 allocation weight 解释为跨任务 ROI 或 learning utility。
- 不声称 MCG 有独立性能贡献；本轮只证明它位于 admission 之后且生命周期语义正确。
- 不声称 CUDA 长程 exact-none 与 scratch 参数逐 bit 相同；主张的是行为/数据/更新通道的严格 source-free，以及最终固定协议下的统计非劣性。
- 不以 smoke、训练时 eval 曲线或两个 seed 代替最终 3-seed paired evaluation。
- 不把 powerlift 30k 的显著加速表述为100k上限提升；最终 retention gate 已明确否定该外推。

## 实现冻结与旧运行边界

- 最终训练 stamp：`20260712TFINALV2Z`。
- protocol、评估 spec、训练入口以及 admission/replay/MCG/seed 关键文件均由 `final_v2_implementation_sha256.txt` 固定；每个 run meta 另存 implementation digest。
- FINALV2 episode 尚未采集前修正了 evaluator 的 task horizon：basketball=500、powerlift=1000，与各自训练 MDP 一致；同时修正零方差 t 的符号。该预采集协议修订已更新 hash manifest，未改动训练实现或 checkpoint。
- finalizer 预演发现工程审计脚本从 `scripts/` 直接启动时缺少仓库根目录 import path；已在最终裁决前修复并更新 hash manifest，训练实现和统计协议不变。
- `TADMV1Z` 与 `TRNGV1Z` 在最终 target-only/双 RNG 修复前启动，只可作故障定位与工程诊断，不进入最终三种子裁决。
