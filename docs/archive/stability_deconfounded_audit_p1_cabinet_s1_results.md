# Stability-deconfounded transfer audit v1 results

> Generated from episode-paired evaluations. Statistical units are training seeds;
> environment episodes are averaged within each seed before descriptive t statistics.

The primary estimand is task-progress difference at the shorter survival prefix of
each condition/scratch episode pair. Positive values therefore cannot be explained
only by the condition remaining alive for more steps.

| task | step | condition vs scratch | pairs | common-prefix progress scratch→condition (Δ) | raw progress scratch→condition (Δ) | stability Δ | return scratch→condition (Δ) | episode length Δ | early failure scratch→condition (Δ) |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| cabinet | 10000 | run | 32 | +0→+0.0625 (+0.0625) | +0→+0.0625 (+0.0625) | +0.01253 | +27.09→+152.3 (+125.2) | +0 | +0→+0 (+0) |
| cabinet | 30000 | run | 32 | +0.03125→+0.7812 (+0.75) | +0.03125→+0.7812 (+0.75) | +0.09012 | +54.95→+242.6 (+187.7) | +0 | +0→+0 (+0) |
| cabinet | 100000 | run | 32 | +0.125→+1 (+0.875) | +0.125→+1 (+0.875) | +0.1362 | +97.33→+224.8 (+127.5) | +0 | +0→+0 (+0) |
| cabinet | 10000 | stand | 32 | +0→+0.03125 (+0.03125) | +0→+0.03125 (+0.03125) | +0.05221 | +27.09→+59.69 (+32.6) | +0 | +0→+0 (+0) |
| cabinet | 30000 | stand | 32 | +0.03125→+0.0625 (+0.03125) | +0.03125→+0.0625 (+0.03125) | +0.02383 | +54.95→+68.5 (+13.56) | +0 | +0→+0 (+0) |
| cabinet | 100000 | stand | 32 | +0.125→+0.375 (+0.25) | +0.125→+0.375 (+0.25) | -0.02297 | +97.33→+217.8 (+120.5) | +0 | +0→+0 (+0) |
| cabinet | 10000 | walk | 32 | +0→+0 (+0) | +0→+0 (+0) | +0.001173 | +27.09→+26.49 (-0.603) | +0 | +0→+0 (+0) |
| cabinet | 30000 | walk | 32 | +0.03125→+0.0625 (+0.03125) | +0.03125→+0.0625 (+0.03125) | +0.01565 | +54.95→+107.7 (+52.79) | +0 | +0→+0 (+0) |
| cabinet | 100000 | walk | 32 | +0.125→+0.625 (+0.5) | +0.125→+0.625 (+0.5) | -0.01428 | +97.33→+231.5 (+134.1) | +0 | +0→+0 (+0) |
| cabinet | 10000 | wfix | 32 | +0→+0.0625 (+0.0625) | +0→+0.0625 (+0.0625) | +0.005133 | +27.09→+107.8 (+80.71) | +0 | +0→+0 (+0) |
| cabinet | 30000 | wfix | 32 | +0.03125→+0.3438 (+0.3125) | +0.03125→+0.3438 (+0.3125) | +0.04388 | +54.95→+278.5 (+223.6) | +0 | +0→+0 (+0) |
| cabinet | 100000 | wfix | 32 | +0.125→+0.9062 (+0.7812) | +0.125→+0.9062 (+0.7812) | +0.1259 | +97.33→+268.1 (+170.8) | +0 | +0→+0 (+0) |

## Secondary task metrics at the common survival prefix

| task | step | condition | metric | scratch→condition (Δ) |
|---|---:|---|---|---:|
| cabinet | 10000 | run | door_openness_reward | +0.03233→+0.2614 (+0.229) |
| cabinet | 10000 | run | subtask_complete | +0→+0.0625 (+0.0625) |
| cabinet | 30000 | run | door_openness_reward | +0.09408→+0.8643 (+0.7702) |
| cabinet | 30000 | run | subtask_complete | +0.03125→+0.75 (+0.7188) |
| cabinet | 100000 | run | door_openness_reward | +0.2764→+0.9827 (+0.7063) |
| cabinet | 100000 | run | subtask_complete | +0.125→+1 (+0.875) |
| cabinet | 10000 | stand | door_openness_reward | +0.03233→+0.1071 (+0.07478) |
| cabinet | 10000 | stand | subtask_complete | +0→+0.03125 (+0.03125) |
| cabinet | 30000 | stand | door_openness_reward | +0.09408→+0.1337 (+0.03963) |
| cabinet | 30000 | stand | subtask_complete | +0.03125→+0.0625 (+0.03125) |
| cabinet | 100000 | stand | door_openness_reward | +0.2764→+0.666 (+0.3895) |
| cabinet | 100000 | stand | subtask_complete | +0.125→+0.375 (+0.25) |
| cabinet | 10000 | walk | door_openness_reward | +0.03233→+0.04145 (+0.009119) |
| cabinet | 10000 | walk | subtask_complete | +0→+0 (+0) |
| cabinet | 30000 | walk | door_openness_reward | +0.09408→+0.2027 (+0.1086) |
| cabinet | 30000 | walk | subtask_complete | +0.03125→+0.0625 (+0.03125) |
| cabinet | 100000 | walk | door_openness_reward | +0.2764→+0.8208 (+0.5443) |
| cabinet | 100000 | walk | subtask_complete | +0.125→+0.625 (+0.5) |
| cabinet | 10000 | wfix | door_openness_reward | +0.03233→+0.1957 (+0.1634) |
| cabinet | 10000 | wfix | subtask_complete | +0→+0.0625 (+0.0625) |
| cabinet | 30000 | wfix | door_openness_reward | +0.09408→+0.6533 (+0.5593) |
| cabinet | 30000 | wfix | subtask_complete | +0.03125→+0.3438 (+0.3125) |
| cabinet | 100000 | wfix | door_openness_reward | +0.2764→+0.9491 (+0.6727) |
| cabinet | 100000 | wfix | subtask_complete | +0.125→+0.875 (+0.75) |

Interpretation rule: a positive raw-progress delta with a near-zero common-prefix
delta is stability/exposure-mediated; a positive common-prefix delta is evidence of
task-progress acceleration beyond merely surviving longer.
