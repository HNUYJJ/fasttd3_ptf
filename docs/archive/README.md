# 归档文档索引

> 2026-07-16 文档重组：docs/ 顶层的全部历史文档移入本目录，**内容未改动**，
> 目录内文档之间的相对链接仍然有效。整合后的主文档在上一级：
> [`RESEARCH_ROADMAP.md`](../RESEARCH_ROADMAP.md)（科研路线）、
> [`EXPERIMENT_LOG.md`](../EXPERIMENT_LOG.md)（实验结果及分析）、
> [`ISSUES_AND_LESSONS.md`](../ISSUES_AND_LESSONS.md)（问题记录）。
>
> 注意：2026-07-16 之前写的文档（含 `agent_collab/` 协作记录）中凡引用
> `docs/X.md` 的，一律到本目录 `docs/archive/X.md` 查找。
> 这些文档是**科研诚信链的一部分**（预注册、负结果审计、独立复算记录），
> 只归档不删除。

## 论文战略（决策演化链）

- `paper_core_contribution_reconstruction_v1.md` — TBS 中心（SIV 失败后，07-11）
- `paper_core_contribution_reconstruction_v2.md` — 六组件统一框架 = **原始科研目标的正式表述**（07-12；SHU gate 失败记录在其 §8）
- `paper_core_contribution_reconstruction_v3.md` — 静态 RBO 保底路线（07-12；勿当终点）
- `core_mechanism_polishing_v4_plan.md` / `dual_channel_transfer_evidence_matrix_v1.md` / `eps_feasibility_gate_v1.md` — v3 配套
- `ChatGPT-5.5-Pro_review_20260711.md` / `ChatGPT-5.6-Pro_review_20260711.md` — 外部评审意见

## Admission 系列（lifecycle 主线，07-06 → 07-15）

- `admission_core_v1_results.md` / `admission_core_v1_completion_audit.md` — Core v1（exact-none 安全门 PASS；retention FAIL 引出诊断）
- `admission_handoff_v1_results.md` — **handoff 修复 6/6+4/4 全 PASS（贡献③定稿证据）**
- `run_card_adaptive_admission_v1.md` — adaptive v2.1 预注册 run card（Phase B baseline 设计也存于此）
- `adaptive_admission_v1_results.md` — **Phase A 正式裁决 FAIL（行为信号第三次否定）**
- `adaptive_admission_v1_codex_independent_audit.md` — ChatGPT 独立复算

## RBO 主线结果（06-13 → 07-05）

- `handoff_ric_v1_20260613.md` / `handoff_ric_v2_20260614.md` — 宽 pilot 与 bootstrap_only≈full 核心消融
- `handoff_rbo_v3_20260614.md` / `handoff_rbo_chain_20260615.md` / `handoff_discussion_20260616.md` — RBO 定调与全链接力
- `terrain_core_result_v1.md` / `result_wfix_fallrecovery_20260620.md` — terrain 三方与 **wfix 解耦定论（+77.9，t=3.08）**
- `advisor_feedback_analysis_20260702.md` — 导师意见 → transferability 统一框架（T^critic 公式在 §3）
- `breadth_expansion_20260704.md` — 广度三批 + wfix 裁决（主方法简化）
- `wide_pilot_v1_results.md` / `task_progress_audit_v1.md` / `cabinet_p2_warmup_source_dose.md` — pilot 与审计
- `source_target_effect_map_v1.md` — Source-Target-Effect Map（机制框架与边界声明）

## Stability-deconfounded audit（P0/P1/P2，14 份）

- 顶层：`stability_deconfounded_transfer_audit_v1.md`（协议）/ `..._findings.md`（P0 结论）/ `stability_deconfounded_audit_v1_results.md`
- P1（cabinet 源身份）：`..._p1_cabinet_s123_findings.md`（**核心：run>stand 3/3**）+ s1/s123/run_vs_stand/run_vs_wfix 结果表
- P2（run 剂量匹配）：`..._p2_cabinet_run24_findings.md`（**核心：return 不可作在线选源信号**）+ protocol 与 3 份结果表

## 已否证路线的预注册与审计记录

- `source_intervention_mechanism_gate_v1.md` — SIV 2×2 机制门（行为信号第一次否定）
- `stage_conditioned_source_admission_gate_v1.md` / `stage_conditioned_replay_data_value_probe_v1.md` — SHU gate（第二次否定）
- `handoff_mcg_v1_20260611.md` / `handoff_mcg_v2_20260612.md` — MCG 与 package 专项（"状态覆盖≠回报事件"）
- `step2_research_briefing.md` / `step2_framework_review.md` — entity encoder / z-native / anchored readout（表征路线全 null；对应代码已于 2026-07-16 删除）

## 工具与参考

- `transfer_map_v1_analysis.md` / `transfer_map_v2_analysis.md` — TransferMap probe 分析（v2 是 RBO 选源工具的依据）
- `transfer_rl_reading_list.md` — 文献清单
- `experiment_registry.md` — 原实验注册表（已被 `../EXPERIMENT_LOG.md` 吸收）
- `IMPLEMENTATION_NOTES.md` — 最早期实现笔记（引用路径已失效，仅历史价值）
- `design/` — 早期 pipeline 设计与实现审计
- `experiments/` — my_ 时代旧实验记录（h1hand_push_compare 等）
- `migration/` — official FastTD3 迁移记录（my_fasttd3_ptf 删除依据）
