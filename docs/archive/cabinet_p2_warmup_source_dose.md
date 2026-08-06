# Cabinet P2 warmup source-dose audit

> Dose values are the unweighted mean of periodic W&B history cross-sections 
> of vectorized environment assignments. They are not exact behavior-segment, 
> environment-step, replay-transition, or replay-buffer counts.

Strict history filter: `0 <= _step < 30000`. Source shares are absolute shares 
over all sampled environment assignments, not source-conditional teacher weights.

## Per-run cross-sectional estimates

| condition | seed | retained samples | retained steps | source absolute shares | teacher | student | max source-sum error |
|---|---:|---:|---:|---|---:|---:|---:|
| run24 | 1 | 299 | 100–29900 | run=23.850% | 23.850% | 76.150% | 0 |
| run24 | 2 | 299 | 100–29900 | run=23.824% | 23.824% | 76.176% | 0 |
| run24 | 3 | 299 | 100–29900 | run=23.814% | 23.814% | 76.186% | 0 |
| run50 | 1 | 299 | 100–29900 | run=49.927% | 49.927% | 50.073% | 0 |
| run50 | 2 | 299 | 100–29900 | run=49.741% | 49.741% | 50.259% | 0 |
| run50 | 3 | 299 | 100–29900 | run=49.605% | 49.605% | 50.395% | 0 |
| wfix | 1 | 299 | 100–29900 | stand=20.984%, walk=4.978%, run=23.965% | 49.927% | 50.073% | 0 |
| wfix | 2 | 299 | 100–29900 | stand=20.676%, walk=5.281%, run=23.785% | 49.741% | 50.259% | 0 |
| wfix | 3 | 299 | 100–29900 | stand=20.681%, walk=4.993%, run=23.931% | 49.605% | 50.395% | 0 |

## Across-seed estimates

Sample SD is computed across the per-run estimates, with one value per training seed.

| condition | share | mean ± SD across seeds | n seeds |
|---|---|---:|---:|
| run24 | source `run` (absolute) | 23.829% ± 0.019% | 3 |
| run24 | teacher | 23.829% ± 0.019% | 3 |
| run24 | student | 76.171% ± 0.019% | 3 |
| run50 | source `run` (absolute) | 49.758% ± 0.161% | 3 |
| run50 | teacher | 49.758% ± 0.161% | 3 |
| run50 | student | 50.242% ± 0.161% | 3 |
| wfix | source `stand` (absolute) | 20.780% ± 0.177% | 3 |
| wfix | source `walk` (absolute) | 5.084% ± 0.171% | 3 |
| wfix | source `run` (absolute) | 23.894% ± 0.096% | 3 |
| wfix | teacher | 49.758% ± 0.161% | 3 |
| wfix | student | 50.242% ± 0.161% | 3 |

Every retained cross-section passed the configured check that the sum of 
source absolute shares matches teacher share within tolerance.
