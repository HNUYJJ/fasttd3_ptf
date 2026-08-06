# Stage-conditioned target evidence v2 — generalized result

**Decision: `BIDIRECTIONAL_FEASIBILITY_PASS`**

> Core implementation contains no task-name branch.  
> Hurdle/Crawl semantics are supplied by target-evidence YAML adapters.

| Primary gate | Pass | Observed |
|---|---:|---|
| Hurdle admits run and walk, ranks run > walk | True | `['run', 'walk']` |
| Crawl rejects all sources | True | `NONE` |

## Conservative lower bounds

`ΔP` is target-achievement progress, not unconstrained root-x displacement.
`ΔF` reports the worst diagnostic component lower bound; these components are not
hard vetoes unless the target contract explicitly declares them as such.

| Task | Source | Admitted | LCB90 ΔR | LCB90 ΔP | worst diagnostic LCB90 ΔF |
|---|---|---:|---:|---:|---:|
| hurdle | stand | False | -0.2611 | -0.0682 | -0.0079 |
| hurdle | walk | True | 0.4804 | 0.0312 | -0.0214 |
| hurdle | run | True | 0.5474 | 0.0544 | -0.0203 |
| crawl | stand | False | -1.9366 | -0.0045 | -0.0970 |
| crawl | walk | False | -0.6555 | 0.0189 | -0.0108 |
| crawl | run | False | -1.4746 | 0.0047 | 0.0000 |

## Claim boundary

Passing only authorizes an online low-frequency feasibility test.

The full RBO training outcomes are causal intervention labels; this probe is only a cheap predictor.

## Generalization verdict

- Hurdle：接纳 `run, walk`，按 target-achievement progress 得到 `run > walk`；
- Crawl：拒绝全部 source，exact abstention；
- 核心规则、bootstrap confidence、horizon 和零阈值在两任务间完全相同；
- task adapter 只描述 target MDP 的进度/约束语义，不包含 source 名称、历史 RBO
  排名或从当前结果拟合的参数。

Raw artifacts:
`logs/probe/stage_component_probe_generalized_v2_20260726/`.
