# 首轮 formal smoke 标 DIAGNOSTIC —— 它抓到了一个真实 bug

2026-08-07。`pipeline_smoke_eval_diagnostic_round1.json` 是首轮 formal
evaluator 的**原始输出**，原样保留、退出码 0，但**其 `schema_version` 字段是错的**。

## 抓到的 bug

```
顶层 schema_version        "2.1"     ← 硬编码于 p0_evaluator_v2.py:560
episodes[*].schema_version 2.2       ← 来自 schema_v2.SCHEMA_VERSION
```

**后果**：`site_rules.require_comparable` 用的是**顶层**的 `schema_version`。
schema 从 2.1 升到 2.2 时（milestone 由扁平值改为 trajectory 聚合结构、
新增 mujoco_state，是破坏性变更）顶层字段没跟着变——
于是**两份用不同 schema 产出的结果会被判为"同 schema_version"而可比**。

这与 `evaluation_semantics_digest` 想防的是同一类问题，但漏在了另一个字段上：
digest 覆盖三个语义文件的内容，却不覆盖"顶层声明的 schema 号是否与实际一致"。

## 这正是本次验收的价值

`evaluator_v21c_results §8.4` 曾记："formal 模式从未在真实 checkpoint 上
端到端跑过……第一次真实 formal 评估将发生在 P3 之前，届时须单独确认一次。"
这次确认抓到了它。单元测试抓不到——它们构造 payload 时不走 `main()` 的
硬编码分支。

## 首轮仍然有效的观察

```
identity_mode              formal
identity_checked           true
scientific_use_permitted   true
manifest_checked_fields    checkpoint_sha256 / env_name / global_step /
                           learner_seed / training_protocol_digest   （5/5）
evaluation_semantics_digest 80619b7e…（与 v21c smoke 一致，语义文件未变）
原子写                      正常；formal + --allow-overwrite → 退出码 1（实测）
```

## 处置

按 `CLAUDE.md §4.1`：本文件标 DIAGNOSTIC → hotfix（只有代码）→ 重新冻结 →
独立重跑。**请引用重跑后的 `pipeline_smoke_eval.json`。**
两轮都标 `PIPELINE_SMOKE`：验证的是流水线可用性，**不用于任何科学结论**。
