# Stability-deconfounded transfer audit v1 results

> Generated from episode-paired evaluations. Statistical units are training seeds;
> environment episodes are averaged within each seed before descriptive t statistics.

The primary estimand is task-progress difference at the shorter survival prefix of
each condition/run24 episode pair. Positive values therefore cannot be explained
only by the condition remaining alive for more steps.

| task | step | condition vs run24 | pairs | common-prefix progress run24→condition (Δ) | raw progress run24→condition (Δ) | stability Δ | return run24→condition (Δ) | episode length Δ | early failure run24→condition (Δ) |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| cabinet | 10000 | run | 96 | +0±0→+0.04167±0.0361 (+0.04167±0.0361) | +0±0→+0.04167±0.0361 (+0.04167±0.0361) | +0.0114±0.00513 | +79.94±10.5→+118.1±46 (+38.19±35.5) | +0±0 | +0±0→+0±0 (+0±0) |
| cabinet | 30000 | run | 96 | +0.2917±0.172→+0.5208±0.284 (+0.2292±0.416) | +0.2917±0.172→+0.5208±0.284 (+0.2292±0.416) | +0.02713±0.0159 | +289.6±77.2→+219.7±32.4 (-69.84±90) | +0±0 | +0±0→+0±0 (+0±0) |
| cabinet | 100000 | run | 96 | +0.8958±0.141→+0.9688±0.0541 (+0.07292±0.0955) | +0.8958±0.141→+0.9688±0.0541 (+0.07292±0.0955) | +0.07752±0.0565 | +259±20.4→+262.5±38 (+3.495±54.6) | +0±0 | +0±0→+0±0 (+0±0) |

## Secondary task metrics at the common survival prefix

| task | step | condition | metric | run24→condition (Δ) |
|---|---:|---|---|---:|
| cabinet | 10000 | run | door_openness_reward | +0.1258±0.0291→+0.2039±0.0984 (+0.07818±0.0721) |
| cabinet | 10000 | run | subtask_complete | +0±0→+0.04167±0.0361 (+0.04167±0.0361) |
| cabinet | 30000 | run | door_openness_reward | +0.6139±0.104→+0.6785±0.232 (+0.06461±0.336) |
| cabinet | 30000 | run | subtask_complete | +0.2812±0.156→+0.5104±0.269 (+0.2292±0.388) |
| cabinet | 100000 | run | door_openness_reward | +0.9257±0.0252→+0.9689±0.0302 (+0.04326±0.0367) |
| cabinet | 100000 | run | subtask_complete | +0.8542±0.0955→+0.9375±0.0625 (+0.08333±0.0361) |

Interpretation rule: a positive raw-progress delta with a near-zero common-prefix
delta is stability/exposure-mediated; a positive common-prefix delta is evidence of
task-progress acceleration beyond merely surviving longer.
