# Stage-conditioned component probe v1 — result

**Decision: `BIDIRECTIONAL_FEASIBILITY_PASS`**

> 执行日期：2026-07-26  
> 历史说明：这是最初 task-specific feasibility 聚合结果；当前实现与后续在线实验以
> `stage_conditioned_target_evidence_generalization_v2_results_20260726.md` 为准。  
> Primary：Hurdle@10k + Crawl@10k  
> 协议：32 个 student occupancy 状态 × 25 步 matched branch × stand/walk/run  
> 设计冻结文档：`docs/experiments/stage_conditioned_component_probe_v1.md`

| Primary gate | Pass | Observed |
|---|---:|---|
| Hurdle admits run and walk, ranks run > walk | True | `['run', 'walk']` |
| Crawl rejects all sources | True | `NONE` |

## 1. 实验到底测了什么

这不是再跑一轮 RBO 训练，而是把已经获得的 RBO 因果标签作为真值，检验一个便宜的
候选迁移性 proxy：

1. 在当前 10k student 的真实 occupancy 上冻结 MuJoCo `FULLPHYSICS` 状态；
2. 从同一状态分别让 student/source 执行 25 步；
3. 计算 source 相对 student 的累计 target reward、root-x 进度和任务必要可行性；
4. 三个 90% bootstrap 下界同时不差于 student 才接纳，否则 exact abstention。

Hurdle 可行性为 `min(stand_reward, wall_collision_discount)`；Crawl 可行性为
`min(crawling, crawling_head, in_tunnel)`。因此指标不会把“直立向前走”自动当成
“会爬行”。

## 2. Primary 结果

### Conservative lower bounds

| Task | Source | Admitted | LCB90 ΔR | LCB90 ΔP | LCB90 ΔF |
|---|---|---:|---:|---:|---:|
| hurdle | stand | False | -0.2611 | -0.1427 | 0.0497 |
| hurdle | walk | True | 0.4804 | 0.0448 | 0.0465 |
| hurdle | run | True | 0.5474 | 0.1215 | 0.0219 |
| crawl | stand | False | -1.9366 | -0.0180 | -0.0970 |
| crawl | walk | False | -0.6555 | 0.0516 | -0.0108 |
| crawl | run | False | -1.4746 | 0.0363 | 0.0288 |

关键解读：

- **Hurdle**：walk/run 的 reward、progress、feasibility 下界均为正；run 的 progress
  下界 `0.1215` 高于 walk 的 `0.0448`，与 3-seed equal-dose RBO 标签
  `run > walk > scratch` 一致。stand 因 progress 下界为负被拒绝。
- **Crawl**：三源全部被拒绝。walk/run 虽有正位移，但 walk 的可行性下界为负，
  walk/run 的 target reward 下界也为负；因此系统输出 100% student。
- 这说明旧 probe 的核心问题不只是噪声，而是缺少 matched student baseline 与目标任务
  必要约束。单看 locomotion progress 会在 Crawl 上产生错误接纳。

## 3. 晚阶段描述性检查（不参与 Primary gate）

| Task/stage | Admitted order | 结果 |
|---|---|---|
| Hurdle@25k | `NONE` | walk/run 的 reward 与位移仍为正，但相对更强 student 的 feasibility LCB 略低于 0 |
| Crawl@30k | `NONE` | 三源继续被拒绝，且主要 target components 明显劣于 student |

Hurdle 从 10k 的 `run, walk` 变为 25k 的 abstention，与“source value 随 student 阶段
失效”的机制假设一致。但目前没有 25k 局部 RBO 干预真值，所以这里只能作为描述性证据，
不能声称已经证明阶段切换正确。

## 4. 工程与产物

- Hurdle 10k scratch anchor：
  `models/h1hand-hurdle-v0__stage_component_probe_hurdle_scratch_anchor10k_s1_20260726T091504Z__1_10000.pt`
- Anchor W&B run id：`urxq7zd0`，project：`ptf_fasttd3_transferability_probe`
- Raw probe：
  `logs/probe/stage_component_probe_v1_20260726T091504Z/`
- 自动裁决：
  `logs/probe/stage_component_probe_v1_20260726T091504Z/decision.json`
- 聚焦回归：`18 passed`（本 probe 3 项 + replay snapshot 15 项）。

## 5. 科学裁决与下一步

Passing only authorizes an online low-frequency feasibility test.

The full RBO training outcomes are causal intervention labels; this probe is only a cheap predictor.

准确裁决是：

- **已支持**：同一冻结规则第一次同时通过 Hurdle 正迁移选源/排序和 Crawl 负迁移弃权；
- **尚未支持**：它尚不是被验证的一般迁移性指标，也没有证明在线使用能改善训练；
- **下一步唯一合理实验**：把 probe 低频插入 warmup，但探测数据仍留在 quarantine，
  仅在同一规则通过时给 source 一个 25 步、有限剂量的 lease；先做 Hurdle/Crawl seed 1
  的在线 feasibility gate。若不能分别保持正迁移和恢复 Crawl scratch，则停止，不扩
  seeds、不调整本轮规则。
