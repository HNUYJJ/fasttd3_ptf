# Stability-deconfounded transfer audit v1 results

> **2026-07-11 validity correction.** The collector's legacy
> `env.unwrapped.seed(seed)` call did not seed HumanoidBench reset noise. Thus
> run/stand condition means and their cross-training-seed direction remain
> descriptive, but these rows are not exact same-state counterfactual pairs and
> the paired statistics must not be used as such. Cabinet also does not terminate
> on a fall; equal episode length therefore does not by itself deconfound
> posture. See
> [`source_intervention_mechanism_gate_v1.md`](source_intervention_mechanism_gate_v1.md)
> for the corrected design.

> Generated from episode-paired evaluations. Statistical units are training seeds;
> environment episodes are averaged within each seed before descriptive t statistics.

The primary estimand is task-progress difference at the shorter survival prefix of
each condition/stand episode pair. Positive values therefore cannot be explained
only by the condition remaining alive for more steps.

| task | step | condition vs stand | pairs | common-prefix progress stand→condition (Δ) | raw progress stand→condition (Δ) | stability Δ | return stand→condition (Δ) | episode length Δ | early failure stand→condition (Δ) |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| cabinet | 10000 | run | 96 | +0.01042±0.018→+0.04167±0.0361 (+0.03125±0.0312) | +0.01042±0.018→+0.04167±0.0361 (+0.03125±0.0312) | -0.006734±0.0314 | +75.36±58.3→+118.1±46 (+42.78±102) | +0±0 | +0±0→+0±0 (+0±0) |
| cabinet | 30000 | run | 96 | +0.07292±0.0786→+0.5208±0.284 (+0.4479±0.253) | +0.07292±0.0786→+0.5208±0.284 (+0.4479±0.253) | +0.04427±0.0223 | +113.4±73.6→+219.7±32.4 (+106.4±69.3) | +0±0 | +0±0→+0±0 (+0±0) |
| cabinet | 100000 | run | 96 | +0.4479±0.0651→+0.9688±0.0541 (+0.5208±0.11) | +0.4479±0.0651→+0.9688±0.0541 (+0.5208±0.11) | +0.1009±0.0509 | +259.1±35.8→+262.5±38 (+3.434±17.5) | +0±0 | +0±0→+0±0 (+0±0) |

## Secondary task metrics at the common survival prefix

| task | step | condition | metric | stand→condition (Δ) |
|---|---:|---|---|---:|
| cabinet | 10000 | run | door_openness_reward | +0.1008±0.0786→+0.2039±0.0984 (+0.1032±0.169) |
| cabinet | 10000 | run | subtask_complete | +0.01042±0.018→+0.04167±0.0361 (+0.03125±0.0312) |
| cabinet | 30000 | run | door_openness_reward | +0.1971±0.141→+0.6785±0.232 (+0.4814±0.219) |
| cabinet | 30000 | run | subtask_complete | +0.0625±0.0625→+0.5104±0.269 (+0.4479±0.235) |
| cabinet | 100000 | run | door_openness_reward | +0.7237±0.0543→+0.9689±0.0302 (+0.2452±0.079) |
| cabinet | 100000 | run | subtask_complete | +0.4375±0.0625→+0.9375±0.0625 (+0.5±0.125) |

Interpretation rule: a positive raw-progress delta with a near-zero common-prefix
delta is stability/exposure-mediated; a positive common-prefix delta is evidence of
task-progress acceleration beyond merely surviving longer.
