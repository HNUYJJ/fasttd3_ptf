# Stability-deconfounded transfer audit v1 results

> Generated from episode-paired evaluations. Statistical units are training seeds;
> environment episodes are averaged within each seed before descriptive t statistics.

The primary estimand is task-progress difference at the shorter survival prefix of
each condition/wfix episode pair. Positive values therefore cannot be explained
only by the condition remaining alive for more steps.

| task | step | condition vs wfix | pairs | common-prefix progress wfix→condition (Δ) | raw progress wfix→condition (Δ) | stability Δ | return wfix→condition (Δ) | episode length Δ | early failure wfix→condition (Δ) |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| cabinet | 10000 | run | 96 | +0.02083±0.0361→+0.04167±0.0361 (+0.02083±0.0361) | +0.02083±0.0361→+0.04167±0.0361 (+0.02083±0.0361) | +0.01089±0.0135 | +74.84±30.3→+118.1±46 (+43.29±24.9) | +0±0 | +0±0→+0±0 (+0±0) |
| cabinet | 30000 | run | 96 | +0.2604±0.118→+0.5208±0.284 (+0.2604±0.307) | +0.2604±0.118→+0.5208±0.284 (+0.2604±0.307) | +0.0368±0.0114 | +210.9±59.7→+219.7±32.4 (+8.775±41.1) | +0±0 | +0±0→+0±0 (+0±0) |
| cabinet | 100000 | run | 96 | +0.9479±0.0477→+0.9688±0.0541 (+0.02083±0.1) | +0.9479±0.0477→+0.9688±0.0541 (+0.02083±0.1) | +0.0003635±0.0118 | +244.5±22→+262.5±38 (+18.03±59.8) | +0±0 | +0±0→+0±0 (+0±0) |

## Secondary task metrics at the common survival prefix

| task | step | condition | metric | wfix→condition (Δ) |
|---|---:|---|---|---:|
| cabinet | 10000 | run | door_openness_reward | +0.1264±0.0602→+0.2039±0.0984 (+0.07759±0.0819) |
| cabinet | 10000 | run | subtask_complete | +0.02083±0.0361→+0.04167±0.0361 (+0.02083±0.0361) |
| cabinet | 30000 | run | door_openness_reward | +0.5169±0.121→+0.6785±0.232 (+0.1616±0.197) |
| cabinet | 30000 | run | subtask_complete | +0.2604±0.118→+0.5104±0.269 (+0.25±0.298) |
| cabinet | 100000 | run | door_openness_reward | +0.9613±0.0133→+0.9689±0.0302 (+0.00767±0.0424) |
| cabinet | 100000 | run | subtask_complete | +0.9271±0.0477→+0.9375±0.0625 (+0.01042±0.11) |

Interpretation rule: a positive raw-progress delta with a near-zero common-prefix
delta is stability/exposure-mediated; a positive common-prefix delta is evidence of
task-progress acceleration beyond merely surviving longer.
