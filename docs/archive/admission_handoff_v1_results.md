# Admission Handoff v1：正式结果与论文裁决

> 日期：2026-07-13  
> 状态：6/6 formal runs 完成，6/6 checkpoint verification 通过；powerlift 与 truck 的全部预注册 gate 通过。  
> 冻结 stamp：`20260713THANDOFFV1Z`  
> 结论边界：本轮验证 replay lifecycle 修复与 late retention，不实现或声称自动 transferability estimator。

## 1. 科学问题

Admission Core v1 在 warmup 后停止 source 行为，但 replay 仍按固定 provenance quota 抽样。只要 buffer 中还剩一条 source transition，该 source stratum 就继续获得完整 quota；随着 source 数据被 circular buffer 覆盖，单条旧 transition 的重复采样率会发散。

本轮检验以下设计原则：

\[
m_i^{\mathrm{replay}}(t)=\mathrm{admitted}_i(t)\,
\begin{cases}
m_i^{\mathrm{admission}}, & a_i(t)=1,\\
\rho_i^{\mathrm{physical}}(t), & a_i(t)=0,
\end{cases}
\]

其中 `a_i(t)` 表示 source 是否仍有 behavior authority；`ρ_i` 是 allowed replay 中的物理驻留比例。student 吸收释放的质量，rejected source 在两个阶段都严格为零。

形式化假设是：source authority 在 30k 结束后，replay exposure 也必须从固定 quota handoff 到物理驻留分布；否则 obsolete source data 会在退役期产生 repetition divergence。

## 2. 实验与完整性

两组实验均为 3 seeds × 100k，128 env，`bootstrap_only`，warmup=30k，actor 复用 critic batch，且与原 admission-all 配置只差 `admission_replay_handoff=physical_after_authority`。

| cell | 任务 | source bank | 作用 |
|---|---|---|---|
| `powerlift_admission_all_fix` | powerlift | std9 WFix | 裁决 30k–80k 暂态 replay 伤害是否消除 |
| `truck_admission_h4_fix` | truck | stand/walk/run/hurdle | 在仍有 95k headroom 的任务上裁决 late retention |

工程完整性：

- 六条训练 exit code 全为 0，final step 均为 100k；
- 六条 run 均在线同步 W&B；
- 每条 run 前后冻结 SHA256 校验通过；
- 60k、90k、final checkpoint 的 handoff config、authority event、sampling phase 和 physical mass audit 全部通过；
- 全仓测试 `252 passed`，真实 HumanoidBench 35-step smoke 跨过 handoff 边界。

## 3. Powerlift：暂态伤害被修复

### 3.1 预注册 gate

| 指标 | 结果 | gate | 裁决 |
|---|---:|---:|---|
| 5k–30k fix−fixed-quota | `+3.742` | `|mean|≤10` | PASS |
| 35k–80k fix−fixed-quota | `+20.075` | `mean≥10` | PASS |
| 上项 per seed | `+10.329/+13.451/+36.446` | 3/3 positive | PASS |
| 35k–80k fix−legacy WFix | `−1.459` | `|mean|≤10` | PASS（compatibility） |
| 80k repaired mean | `319.370` | 不低于预注册 collapse floor `300.031` | PASS |
| authority/physical audit | 6/6 checkpoint rows | 全通过 | PASS |

最清楚的因果闭环出现在 80k：

| 方法 | s1 | s2 | s3 | mean |
|---|---:|---:|---:|---:|
| scratch | 207.721 | 196.791 | 271.748 | 225.420 |
| fixed-quota admission | 245.652 | 170.293 | 160.657 | **192.201** |
| legacy WFix | 334.960 | 320.318 | 286.155 | 313.811 |
| handoff fix | 322.209 | 308.054 | 327.847 | **319.370** |

修复相对 fixed-quota 的 80k 增量为 `+76.557/+137.761/+167.190`，3/3 同向；修复后均值也与 legacy WFix 同档。使用 Claude 指出的更保守 secondary check——固定使用旧 fixed-quota 的 80k SD=`46.540`——新曲线的 collapse floor 为 `263.688`，新均值 `319.370` 仍大幅通过。逐 seed 相对自身 75k/85k 邻点均值为 `+11.685/−13.753/+29.494`：s2 有正常局部波动，但不存在旧版 3/3 的百点级同步崩塌。

### 3.2 独立采样暴露证据

下表不是由 `effective_replay_masses` 和同源 buffer count 互相比较，而是直接用不同 checkpoint 的累计 `critic_sample_counts` 做差：

| 阶段 | handoff 实际 source critic share | fixed-quota 对照 | T0004 见结果前预测 |
|---|---:|---:|---:|
| 30k→60k | **33.651%** | 50.000% | 33.700% |
| 60k→90k | **7.189%** | 34.871% | 7.200% |
| 90k→100k | **0.000%** | 0.000% | 0.000% |

三个 seed 分别为：

- 30k→60k：`33.648%/33.674%/33.632%`；
- 60k→90k：`7.181%/7.195%/7.192%`；
- 90k→100k：全部 `0%`。

实际采样与在结果揭晓前根据 circular turnover 推导的预测误差小于 `0.05` 个百分点，同时与 fixed-quota 形成大幅分离。这一证据独立闭合了“authority release → sampling exposure 衰减 → 80k collapse 消失”的机制链。

### 3.3 Powerlift 的诚实边界

- 本任务只支持暂态 replay lifecycle correctness；不能重新包装成 100k retention 结论。
- 95k fix=`311.830`、WFix=`311.646`、scratch=`305.107`，三者已收敛，说明 powerlift 后期 headroom 本来就耗尽。
- 35k–80k 的 seed-level 修复量虽然 3/3 为正，但 `n=3`；这里依赖预注册幅度门、80k 因果崩点和独立采样计数共同支撑，不声称仅凭一个 t-test 完成普适证明。

## 4. Truck：保留真正的 late transfer benefit

95k 结果：

| 方法 | s1 | s2 | s3 | mean |
|---|---:|---:|---:|---:|
| scratch | 1376.403 | 1380.068 | 1332.461 | 1362.977 |
| legacy WFix | 1589.255 | 1659.202 | 1651.708 | 1633.388 |
| handoff fix | 1625.789 | 1516.012 | 1630.612 | 1590.804 |

Handoff fix−scratch 为 `+249.386/+135.944/+298.151`，mean=`+227.827`，3/3 positive，paired `t=4.741`（df=2），通过预注册的 `mean≥150`。它保留了 legacy WFix 相对 scratch 差距的 `84.3%`。

但 fix−WFix 为 `+36.534/−143.190/−21.096`，mean=`−42.584`。因此正确主张是：

> authority-coupled handoff 在 truck 上保留了持续到 95k 的显著正迁移，并避免 replay authority 与 behavior authority 脱节；它没有证明优于 legacy uniform replay，也不应声称提高了 legacy WFix 的性能上限。

Truck bank 的行为质量中 student=50%，hurdle≈27.6%，walk≈12.3%，run≈10.1%，stand 仅约0.006%。95k 的 `+227.8` 因而不能归因于 stand teacher 的站立注入；它为“迁移收益不只是站稳”的担忧提供了强反例。不过，本实验没有单源训练归因，不能进一步断言收益全部来自 hurdle skill。

## 5. 对核心贡献的影响

本轮把 replay 通道的论述从经验性“旧 source 数据可能有害”推进为可推导、可预测、可复现的生命周期原则：

1. source 的价值不仅是 source/task-dependent，也具有 stage/lifecycle dependence；
2. behavior authority 与 replay authority 是两个必须共同管理的通道；
3. 固定 provenance quota 在 source retirement 阶段会产生解析可预测的 repetition divergence；
4. replay handoff 必须与当前 behavior authority 一致，同时保留 exact revoke；
5. 该修复既消除了预测时点的 3/3 collapse，也在有 headroom 的 truck 上保留了 late positive transfer。

论文中可把它与 exact abstention、quarantine、runtime revocation 合并为：

> **Provenance-consistent source data lifecycle**：source experience 只有在被 admit 且仍具行为权威时才享有 admission quota；权威结束后，其 replay exposure 回归物理驻留并随 turnover 自然退役；source 被拒绝时则严格退出 active replay。

这是一项扎实的 correctness-critical mechanism contribution，但不应单独包装成新算法 headline。

## 6. 仍未解决的问题

1. **自动迁移性指标仍未解决**：本轮使用 `all`/既定 bank，不产生新的 source utility estimator。
2. **Exact abstention 的决策来源仍是外部 snapshot/manifest**：基础设施已经能严格弃权，但“何时弃权”尚未由可信学习指标自动决定。
3. **Truck 不能分离 handoff 的单独收益**：没有 truck fixed-quota admission 对照；它验证 admission+handoff 组合能保留正迁移，handoff 的纯因果效应由 powerlift 裁决。
4. **MCG 未进入本轮主效应**：formal config 是 `bootstrap_only`；MCG 仍是 admission 后的可选 authority executor，而非此结果的性能来源。

## 7. 下一步裁决

不再追加同类小实验。下一步按以下顺序推进：

1. 交由 Claude 对本结果、补强统计和 claim boundary 做只读对抗审计；
2. 审计通过后，把 provenance-consistent lifecycle 写入论文方法公式、机制图和贡献列表；
3. 将 admission exact-none、quarantine/revocation、powerlift repetition-divergence 修复、truck late retention 组织为一条统一证据链；
4. 自动 transferability/admission estimator 仍单独标为 open mechanism，不用本轮结果冒充已经解决。

## 8. 证据索引

- 预注册：`configs/experiments/admission_handoff_v1.yaml`
- 自动裁决：`artifacts/admission_handoff_v1/20260713THANDOFFV1Z/analysis_20260713THANDOFFV1Z.{json,md}`
- 训练认证：`artifacts/admission_handoff_v1/20260713THANDOFFV1Z/training_verification/`
- 冻结哈希：`artifacts/admission_handoff_v1/20260713THANDOFFV1Z/frozen_implementation.sha256`
- 原始日志：`logs/train/admission_handoff_v1_20260713THANDOFFV1Z/`
- 实现：`fasttd3_ptf/official_fasttd3_ptf/ptf_replay.py`、`train_ptf.py`
