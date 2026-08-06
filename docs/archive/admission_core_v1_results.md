# Admission Core v1：实现与验证记录

> 日期：2026-07-13  
> 状态：工程机制通过；冻结版 `20260712TFINALV2Z` 的 3-seed 正/负迁移实验与固定配对评估已完成。负迁移安全 gate 通过，正迁移 retention gate 未通过。  
> 边界：本轮不实现、也不声称已经解决自动迁移性估计。

## 1. 本轮要验证的机制不变量

1. student 是与所有 source 同层的一等候选，不存在 admission 路径之外的固定 teacher Bernoulli。
2. admission 空集必须是 exact abstention：source 行为、source replay exposure 和 source 蒸馏严格为零。
3. probe trajectory 只进入 quarantine，不更新 learner、不写 main replay。
4. main replay 只允许 student 和当前 admitted source；撤销后旧 source transition 可保留作审计，但 active sampling mass 必须严格为零。
5. actor/critic 不使用互相冲突的来源分布；主方法中 actor 复用 critic batch。
6. MCG 只能在 policy-level admission 之后选择 source；全部拒绝时 MCG 也是零作用。

## 2. 已完成实现

- `admission_control.py`：显式 `all/none/static/manifest/schedule` 决策，student-inclusive 概率与 quarantine artifact 哈希绑定。
- `mcg.py`：`admission_bootstrap` 单一 categorical 调度、source mask、立即释放被撤销 latch、动态 source/student logits。
- `ptf_replay.py`：按 provenance stratum 分配配额；stratum 内组合 recency、TD priority 和 uniform coverage；source 撤销后 active mass 为零；记录 policy event 和实际采样计数。
- `train_ptf.py`：训练期显式 admission schedule、写入前拒绝源断言、exact-abstain 源策略/蒸馏短路、checkpoint admission audit。

## 3. Quarantine / main replay 隔离证据

`source_admission.py::validate_gate_config` 和探针入口共同强制：

- `quarantine_only=true`；
- `learner_updates=0`；
- `main_replay_writes=0`。

`validate_quarantine_bank` 还检查每条 path 的完整 transition tensor、行为 provenance、source/group authority，以及独立重复 student 路径的一致性。训练入口只消费 admission decision 和绑定的 artifact SHA256；不存在把 quarantine transition 导入 `PTFReplayWrapper` 的代码路径。

真实 artifact 集成检查使用 `cabinet_s1_step10000/quarantine.pt`（208 MB、512 anchors）：内容 validator 通过，manifest 文件 SHA256 绑定通过，并得到只 admit `run` 的 immutable snapshot；artifact 元数据仍为 `quarantine_only=true`、`learner_updates=0`、`main_replay_writes=0`。对应证据为 `artifacts/admission_core_v1/quarantine_manifest_integration.json`。

## 4. Exact abstention 的早期 100k 诊断证据（非最终裁决）

篮球任务、9-source bank、admission=`none`：

| seed | source execution | source physical main buffer | source critic samples | student execution |
|---:|---:|---:|---:|---:|
| 1 | 0 | 0 | 0 | 12,800,000 |
| 2 | 0 | 0 | 0 | 12,800,000 |

两个 pre-freeze seed 的 student candidate mass 均为精确 1.0，证明了旧实现的 source 三通道隔离；但它们早于最终 target-only fast path 和 basketball 完整播种修复，不能进入最终性能裁决。正式证据以 `20260712TFINALV2Z` 为准。

## 5. 运行时撤销实验

配置：`admission_revocation_smoke.yaml`，step 0 只 admit `stand`，step 20 撤销全部 source，训练到 step 60。

| 量 | 撤销前/撤销时 | 撤销后新增 |
|---|---:|---:|
| source execution | 320 | **0** |
| source critic samples | 4,608 | **0** |
| physical source transitions retained for audit | 320 | — |
| active source transitions after revocation | — | **0** |

这验证的是立即退出 active replay，而不是物理删除历史证据。

## 6. 最终科学问题的裁决

- basketball exact-none 的负迁移安全 gate **PASS**：100k common-prefix progress delta 相对 scratch 为 `+0.03125±0.08268`，`t=0.655`；旧 WFix 为 `−0.09375±0.11267`。exact-none 相对旧 WFix 恢复 `+0.125`，且 return delta 为 `+87.18±88.68`。由于 exact-none 严格 source-free，这只能解释为“弃权后回到自主 RL 的统计分布”，不能解释为 source 带来的正迁移。
- powerlift admission-all 的正迁移 retention gate **FAIL**：30k 有明确前期加速，common-prefix progress delta=`+0.0003879±0.0001556`、`t=4.318`，return delta=`+48.57±17.30`、`t=4.864`；但100k progress delta 仅 `+0.0001017±0.0002937`、`t=0.600`，return delta=`+4.74±33.00`。相对旧 WFix 100k progress delta `+0.0005097`，retention 仅 `0.1996<0.5`。
- 因此联合性能 gate **FAIL**。当前实现证明了 exact abstention 能修复确定性的有害源暴露，也证明 admission-all bootstrap 能产生显著早期加速；但尚未证明它能稳定保留或提高100k性能上限。

这与机制诊断一致：源策略主要改变前期数据/状态分布；固定 warmup 后，收益可能衰减，且 all-source 注入并未解决阶段失效与次优源稀释。下一轮若继续追求论文级长期正迁移，应针对 stage-conditioned admission/handoff 与 replay 生命周期做改进，而不是把本轮30k加速外推成100k上限贡献。

## 7. Post-admission MCG provenance 补强

MCG 在 warmup 后可能只让 arms/hands 使用 source，因此不能再用身体组 0 的 source id 代表整条 transition。当前实现会：

- 为每条 admission transition 保存 `source_by_group` 和 `executed_group_mask`；
- 用所有 active group 中的 canonical source 进行来源配额分层，而不是固定读取 group 0；
- sampling 前检查全部 contributing source；只要其中任一 source 已被撤销，整条混合 transition 就退出 active replay；
- actor 继续复用 critic batch，避免 actor/critic 来源分布分叉。

`admission_mcg_full_provenance_smoke` 已跨过 warmup→MCG full 切换并完成 critic/actor 更新；混合 source 撤销的定向单元测试也已通过。当前 admission/MCG/replay/RNG/evaluator 核心测试总数为 57，全部通过；其中 MCG/admission 定向测试 6 项、replay lifecycle 定向测试 8 项。

## 8. Powerlift admission-all 的 pre-freeze 30k 配额诊断（s1/s2）

预登记 candidate mass 中 student=0.500。正式 checkpoint 的实际统计为：

| seed | student execution share | student physical-buffer share | student critic-sample share |
|---:|---:|---:|---:|
| 1 | 0.50596 | 0.50596 | 0.50374 |
| 2 | 0.50489 | 0.50489 | 0.50391 |

九个 source 的单独 share 同样接近各自 candidate mass；critic sample 与行为/驻留分布没有出现 actor/critic split 或历史数量支配配额的迹象。这里只作为机制诊断；最终表格必须由同一冻结实现的 `FINALV2` 三个 seed 重建。

## 9. 配对评估协议修正

首次 2-seed interim 汇总虽然达到 384/384 episode coverage 且无重复，但复用的旧 scratch/WFix JSONL 没有 `seed_protocol` 字段，汇总结果为 `gymnasium_vec_reset_v1 + legacy_unknown`，因此 `exact_reset_pairing_validated=false`。该结果不进入科学裁决。

随后发现 v1 evaluator 仍只播种 Gymnasium `env.np_random`，没有播种 basketball task 直接使用的 worker-global `np.random`；因此即使记录都标记为 `gymnasium_vec_reset_v1`，也不能证明 basketball 初始状态真正配对。该协议及其 interim 结果全部降级为故障定位证据。

旧 v1 interim 曾达到 384/384 coverage、0 duplicates 并给出以下数值，但现已明确不进入科学裁决：

- powerlift admission-all：common-prefix progress delta `+0.000661`，return delta `+52.42`；旧 WFix 分别为 `+0.000208`、`+36.23`，是正向早期信号；
- basketball admission-none：common-prefix progress delta `−0.015625`，return delta `−33.37`；没有显示优于旧 WFix 的早期恢复。

最终 evaluator 使用 `gymnasium_plus_global_numpy_vec_reset_v2`，在每个 worker 内同时播种两类 RNG；同一 basketball checkpoint/seed 的两次独立采集已达到 episode return、length、terminal 和完整 metric trace 零差异。finalizer 只重新采集 v2 数据，summary 也只在唯一协议为 v2 时设置 `exact_reset_pairing_validated=true`。

最终采集前的预检还发现：原 eval spec 曾用一个全局 1000-step horizon 包装所有任务，但正式训练 MDP 中 basketball 注册/训练 horizon 为 500，powerlift 为 1000。该设置虽对所有 condition 对称，却会把 basketball 评估成不同的目标 MDP。因此在任何 FINALV2 episode 采集前完成协议修订：eval spec 记录 task-level horizon（basketball=500、powerlift=1000），collector 按 task 构造 TimeLimit 和 rollout；CLI 显式 override 仍会同时覆盖所有 task。相同修订还修复了三 seed 完全零方差时 t statistic 的符号，并由新增回归测试覆盖。训练代码和已生成 checkpoint 不受影响。

## 10. Exact fallback 的 target-only fast path 与环境播种修复

当前实现让 immutable exact-empty admission 与 empty-bank scratch 共用同一个 target-only fast path：恢复 target actor/critic/replay 构建后的 learner RNG，rollout 只执行 student，跳过 option/source/MCG execution 与 transfer update；全 student replay 使用同一个 `torch.randint` primitive。schedule 模式即使初始为空也不进入静态 fast path，因此之后仍可 admit source。

此外，HumanoidBench basketball 的 task reset 使用进程全局 `np.random`，而 base env 使用 Gymnasium `env.np_random`。本地 wrapper 现在在每个 SubprocVecEnv worker 首次 seeded reset 时同时播种两者；basketball 同/异 seed 的定向回归测试已通过。旧 `TRNGV1Z` 因只覆盖前一层 RNG 而作废。

同一 GPU 的 FINALV2 短程复核给出了更精确的边界：step 1 时 scratch 与 exact-none 的 actor、critic、target critic 和 observation normalizer 全部逐 bit 相同，但首次 CUDA learner update 后开始数值分叉；与此同时 source action、source main-replay exposure 和 source critic exposure 仍全部严格为零。使用同一代码、seed、环境和 replay 参数的 CPU 控制实验在首次更新后仍保持上述四类状态逐 bit 相同。这支持“环境、数据、replay index 与 learner RNG 路径一致”，也说明 CUDA 下额外冻结教师常驻及 admission/provenance tensor 的显存/数值布局足以破坏长程逐 bit 等价。

因此 exact abstention 的主张严格限定为：source behavior、source main-replay exposure、source critic exposure、source distillation 全部为零，并在固定评估协议下统计回到 scratch 分布；**不主张 CUDA 长程参数逐 bit 相同**。完整诊断见 `artifacts/admission_core_v1/exact_none_scratch_equivalence_final_v2.json`。

## 11. 最终冻结协议

- run stamp：`20260712TFINALV2Z`；
- 两条队列始终只并行两个训练，避免 CPU 争用；
- 每个 meta 保存 source-bank、protocol 与 implementation SHA256；
- 关键文件还由 `artifacts/admission_core_v1/final_v2_implementation_sha256.txt` 固定；
- 六个 checkpoint 全部认证后，finalizer 才会运行工程审计、重新采集固定 reset 评估、汇总并执行预注册裁决。

## 12. FINALV2 正式 30k 机制审计

Powerlift admission-all 三个冻结版 checkpoint 均已通过自动审计：

| seed | student candidate | student execution | student critic sample | independent actor samples | result |
|---:|---:|---:|---:|---:|---|
| 1 | 0.500000 | 0.505549 | 0.503688 | 0 | PASS |
| 2 | 0.500000 | 0.504974 | 0.503715 | 0 | PASS |
| 3 | 0.500000 | 0.505134 | 0.503803 | 0 | PASS |

三个 seed 中九个 source 均有非零 execution 和 critic exposure；每个来源的 physical main-buffer count 与 execution count 完全一致。各来源 execution/critic share 相对预设 quota 的最大误差均低于预登记审计容差 0.015。s1/s2 训练内 30k eval return 分别为 201.278 和 182.080，仅作健康记录，不替代最终固定 v2 配对评估。

## 13. 30k→60k post-warmup replay 生命周期

三个 seed 的 60k checkpoint 均证明：source execution 在 30k warmup 结束后完全冻结，而 student execution 各增加 3,840,000；旧 source transition 随 circular buffer 覆盖从约 1.90M 降至约 1.34M，但在仍驻留且 source 仍 admitted 时继续按同一 critic 分布被采样。actor independent sample 始终为 0。该结果区分了“行为 authority 已结束”和“合法旧 source 数据仍在 off-policy replay 中学习”这两个生命周期。

训练内 60k eval return 为 s1=315.910、s2=329.570，仅作健康记录。详细计数见 `artifacts/admission_core_v1/powerlift_post_warmup_lifecycle.json`。

## 14. FINALV2 basketball exact-none 的 30k/60k 正式审计

basketball 三个 seed 的 30k 与 60k checkpoint 均通过全部 exact-none 不变量：九个 source 的 candidate mass、execution、physical main-buffer count 和 critic sample count 全部为 0；student candidate mass 精确为 1.0；actor independent sample count 为 0，actor 复用 critic batch。30k 时每个 seed 已写入 3,840,128 条 student transition、critic 累计采样 1,965,424,640 个 student 样本。

这证明 exact abstention 在 warmup 内外持续成立，而不是只在初始化或单个 smoke step 成立。30k 证据位于 `artifacts/admission_core_v1/checkpoint_30k_audit/basketball_none_s{1,2,3}.json`，60k 证据位于 `artifacts/admission_core_v1/checkpoint_60k_audit/basketball_none_s{1,2,3}.json`。性能是否统计回到 scratch 仍待三个 seed 的 100k 固定配对评估。

## 15. FINALV2 六个 final checkpoint 状态

powerlift admission-all 与 basketball exact-none 的 s1/s2/s3 均已完成100k，并通过独立 checkpoint/log 认证。basketball 每个 final checkpoint 都只有12,800,000次 student execution 和6,552,879,104个 student critic samples；九个 source 的 execution、physical main replay 与 critic sample 均为0，student candidate mass=1.0，actor independent sample=0。powerlift 三个 final checkpoint 均保存 student mass=0.5、所有 source 的非零 execution/critic exposure 与 shared actor/critic batch。checkpoint SHA256 位于 `artifacts/admission_core_v1/training_verification/`。

最终工程 analyzer 的六项 verdict 全为 true：actor/critic shared batch、exact abstention、implementation hashes、powerlift student mass、quarantine isolation、runtime revocation。完整 artifact 为 `artifacts/admission_core_v1/engineering_audit.json`。

## 16. 最终固定配对评估与预注册裁决

最终采集包含1152条 episode records，0 duplicate；每个 task/condition 都达到192/192 factorial cells。唯一 seed protocol 为 `gymnasium_plus_global_numpy_vec_reset_v2`，`exact_reset_pairing_validated=true`。统计单位是三个 train seeds，每个 seed 内先平均32个 `(eval_seed, env_rank)` 配对 episode。

| task / condition | step | common-prefix progress Δ vs scratch | t | return Δ vs scratch | gate |
|---|---:|---:|---:|---:|---|
| powerlift admission-all | 30k | +0.0003879±0.0001556 | 4.318 | +48.57±17.30 | 前期加速成立 |
| powerlift admission-all | 100k | +0.0001017±0.0002937 | 0.600 | +4.74±33.00 | retention **FAIL** |
| powerlift WFix | 100k | +0.0005097±0.0004716 | 1.872 | −17.26±28.77 | 对照 |
| basketball admission-none | 100k | +0.03125±0.08268 | 0.655 | +87.18±88.68 | negative-safety **PASS** |
| basketball WFix | 100k | −0.09375±0.11267 | −1.441 | −62.35±137.36 | 有害对照 |

预注册裁决见 `artifacts/admission_core_v1/performance_verdict.json`：basketball 三项 safety check 全过；powerlift 虽然 progress delta 为正，但 `t<2.92` 且 retention=`0.1996<0.5`，因此未通过。完整 paired traces 与汇总分别位于 `logs/probe/admission_core_v1/` 和 `artifacts/admission_core_v1/paired_evaluation.{json,md}`。
