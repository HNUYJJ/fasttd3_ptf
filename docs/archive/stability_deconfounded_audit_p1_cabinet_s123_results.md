# Stability-deconfounded transfer audit v1 results

> Generated from episode-paired evaluations. Statistical units are training seeds;
> environment episodes are averaged within each seed before descriptive t statistics.

The primary estimand is task-progress difference at the shorter survival prefix of
each condition/scratch episode pair. Positive values therefore cannot be explained
only by the condition remaining alive for more steps.

| task | step | condition vs scratch | pairs | common-prefix progress scratch→condition (Δ) | raw progress scratch→condition (Δ) | stability Δ | return scratch→condition (Δ) | episode length Δ | early failure scratch→condition (Δ) |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| cabinet | 10000 | run | 96 | +0±0→+0.04167±0.0361 (+0.04167±0.0361) | +0±0→+0.04167±0.0361 (+0.04167±0.0361) | +0.002496±0.0139 | +32.54±4.77→+118.1±46 (+85.59±49.6) | +0±0 | +0±0→+0±0 (+0±0) |
| cabinet | 30000 | run | 96 | +0.02083±0.018→+0.5208±0.284 (+0.5±0.267) | +0.02083±0.018→+0.5208±0.284 (+0.5±0.267) | +0.06708±0.0312 | +89.49±30.6→+219.7±32.4 (+130.2±53.3) | +0±0 | +0±0→+0±0 (+0±0) |
| cabinet | 100000 | run | 96 | +0.5±0.36→+0.9688±0.0541 (+0.4688±0.368) | +0.5±0.36→+0.9688±0.0541 (+0.4688±0.368) | +0.1284±0.0124 | +181.1±72.6→+262.5±38 (+81.42±43.9) | +0±0 | +0±0→+0±0 (+0±0) |
| cabinet | 10000 | stand | 96 | +0±0→+0.01042±0.018 (+0.01042±0.018) | +0±0→+0.01042±0.018 (+0.01042±0.018) | +0.009229±0.0443 | +32.54±4.77→+75.36±58.3 (+42.82±56.8) | +0±0 | +0±0→+0±0 (+0±0) |
| cabinet | 30000 | stand | 96 | +0.02083±0.018→+0.07292±0.0786 (+0.05208±0.0651) | +0.02083±0.018→+0.07292±0.0786 (+0.05208±0.0651) | +0.02281±0.0124 | +89.49±30.6→+113.4±73.6 (+23.87±56.7) | +0±0 | +0±0→+0±0 (+0±0) |
| cabinet | 100000 | stand | 96 | +0.5±0.36→+0.4479±0.0651 (-0.05208±0.313) | +0.5±0.36→+0.4479±0.0651 (-0.05208±0.313) | +0.02753±0.0466 | +181.1±72.6→+259.1±35.8 (+77.99±36.8) | +0±0 | +0±0→+0±0 (+0±0) |
| cabinet | 10000 | wfix | 96 | +0±0→+0.02083±0.0361 (+0.02083±0.0361) | +0±0→+0.02083±0.0361 (+0.02083±0.0361) | -0.00839±0.0267 | +32.54±4.77→+74.84±30.3 (+42.3±35) | +0±0 | +0±0→+0±0 (+0±0) |
| cabinet | 30000 | wfix | 96 | +0.02083±0.018→+0.2604±0.118 (+0.2396±0.126) | +0.02083±0.018→+0.2604±0.118 (+0.2396±0.126) | +0.03028±0.0341 | +89.49±30.6→+210.9±59.7 (+121.5±88.6) | +0±0 | +0±0→+0±0 (+0±0) |
| cabinet | 100000 | wfix | 96 | +0.5±0.36→+0.9479±0.0477 (+0.4479±0.344) | +0.5±0.36→+0.9479±0.0477 (+0.4479±0.344) | +0.128±0.00301 | +181.1±72.6→+244.5±22 (+63.39±93.4) | +0±0 | +0±0→+0±0 (+0±0) |

## Secondary task metrics at the common survival prefix

| task | step | condition | metric | scratch→condition (Δ) |
|---|---:|---|---|---:|
| cabinet | 10000 | run | door_openness_reward | +0.03559±0.00306→+0.2039±0.0984 (+0.1684±0.101) |
| cabinet | 10000 | run | subtask_complete | +0±0→+0.04167±0.0361 (+0.04167±0.0361) |
| cabinet | 30000 | run | door_openness_reward | +0.138±0.0441→+0.6785±0.232 (+0.5405±0.246) |
| cabinet | 30000 | run | subtask_complete | +0.02083±0.018→+0.5104±0.269 (+0.4896±0.253) |
| cabinet | 100000 | run | door_openness_reward | +0.6444±0.335→+0.9689±0.0302 (+0.3245±0.339) |
| cabinet | 100000 | run | subtask_complete | +0.4896±0.346→+0.9375±0.0625 (+0.4479±0.386) |
| cabinet | 10000 | stand | door_openness_reward | +0.03559±0.00306→+0.1008±0.0786 (+0.06519±0.0777) |
| cabinet | 10000 | stand | subtask_complete | +0±0→+0.01042±0.018 (+0.01042±0.018) |
| cabinet | 30000 | stand | door_openness_reward | +0.138±0.0441→+0.1971±0.141 (+0.05906±0.109) |
| cabinet | 30000 | stand | subtask_complete | +0.02083±0.018→+0.0625±0.0625 (+0.04167±0.0477) |
| cabinet | 100000 | stand | door_openness_reward | +0.6444±0.335→+0.7237±0.0543 (+0.07931±0.296) |
| cabinet | 100000 | stand | subtask_complete | +0.4896±0.346→+0.4375±0.0625 (-0.05208±0.313) |
| cabinet | 10000 | wfix | door_openness_reward | +0.03559±0.00306→+0.1264±0.0602 (+0.09076±0.0631) |
| cabinet | 10000 | wfix | subtask_complete | +0±0→+0.02083±0.0361 (+0.02083±0.0361) |
| cabinet | 30000 | wfix | door_openness_reward | +0.138±0.0441→+0.5169±0.121 (+0.3789±0.163) |
| cabinet | 30000 | wfix | subtask_complete | +0.02083±0.018→+0.2604±0.118 (+0.2396±0.126) |
| cabinet | 100000 | wfix | door_openness_reward | +0.6444±0.335→+0.9613±0.0133 (+0.3169±0.328) |
| cabinet | 100000 | wfix | subtask_complete | +0.4896±0.346→+0.9271±0.0477 (+0.4375±0.312) |

Interpretation rule: a positive raw-progress delta with a near-zero common-prefix
delta is stability/exposure-mediated; a positive common-prefix delta is evidence of
task-progress acceleration beyond merely surviving longer.
