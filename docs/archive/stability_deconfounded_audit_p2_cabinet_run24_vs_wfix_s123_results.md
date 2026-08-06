# Stability-deconfounded transfer audit v1 results

> Generated from episode-paired evaluations. Statistical units are training seeds;
> environment episodes are averaged within each seed before descriptive t statistics.

The primary estimand is task-progress difference at the shorter survival prefix of
each condition/wfix episode pair. Positive values therefore cannot be explained
only by the condition remaining alive for more steps.

| task | step | condition vs wfix | pairs | common-prefix progress wfix→condition (Δ) | raw progress wfix→condition (Δ) | stability Δ | return wfix→condition (Δ) | episode length Δ | early failure wfix→condition (Δ) |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| cabinet | 10000 | run24 | 96 | +0.02083±0.0361→+0±0 (-0.02083±0.0361) | +0.02083±0.0361→+0±0 (-0.02083±0.0361) | -0.0005095±0.00951 | +74.84±30.3→+79.94±10.5 (+5.1±22.4) | +0±0 | +0±0→+0±0 (+0±0) |
| cabinet | 30000 | run24 | 96 | +0.2604±0.118→+0.2917±0.172 (+0.03125±0.113) | +0.2604±0.118→+0.2917±0.172 (+0.03125±0.113) | +0.009677±0.0146 | +210.9±59.7→+289.6±77.2 (+78.61±130) | +0±0 | +0±0→+0±0 (+0±0) |
| cabinet | 100000 | run24 | 96 | +0.9479±0.0477→+0.8958±0.141 (-0.05208±0.188) | +0.9479±0.0477→+0.8958±0.141 (-0.05208±0.188) | -0.07715±0.0593 | +244.5±22→+259±20.4 (+14.54±18.8) | +0±0 | +0±0→+0±0 (+0±0) |

## Secondary task metrics at the common survival prefix

| task | step | condition | metric | wfix→condition (Δ) |
|---|---:|---|---|---:|
| cabinet | 10000 | run24 | door_openness_reward | +0.1264±0.0602→+0.1258±0.0291 (-0.0005883±0.04) |
| cabinet | 10000 | run24 | subtask_complete | +0.02083±0.0361→+0±0 (-0.02083±0.0361) |
| cabinet | 30000 | run24 | door_openness_reward | +0.5169±0.121→+0.6139±0.104 (+0.097±0.191) |
| cabinet | 30000 | run24 | subtask_complete | +0.2604±0.118→+0.2812±0.156 (+0.02083±0.0955) |
| cabinet | 100000 | run24 | door_openness_reward | +0.9613±0.0133→+0.9257±0.0252 (-0.03559±0.0348) |
| cabinet | 100000 | run24 | subtask_complete | +0.9271±0.0477→+0.8542±0.0955 (-0.07292±0.141) |

Interpretation rule: a positive raw-progress delta with a near-zero common-prefix
delta is stability/exposure-mediated; a positive common-prefix delta is evidence of
task-progress acceleration beyond merely surviving longer.
