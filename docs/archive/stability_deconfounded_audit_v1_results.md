# Stability-deconfounded transfer audit v1 results

> **2026-07-11 validity correction.** The collector used
> `env.unwrapped.seed(seed)`, which does not seed Gymnasium's per-environment
> `np_random` used by HumanoidBench reset noise. Consequently, the condition
> means and the large cross-training-seed directions below remain descriptive
> evidence, but the episode rows are **not exact same-initial-state pairs** and
> paired uncertainty must not be given a counterfactual interpretation. In
> addition, cabinet does not terminate when the humanoid falls, so equal
> 1000-step episode lengths do not by themselves remove posture/stability as an
> explanation. Future causal evidence follows
> [`source_intervention_mechanism_gate_v1.md`](source_intervention_mechanism_gate_v1.md).

> Generated from episode-paired evaluations. Statistical units are training seeds;
> environment episodes are averaged within each seed before descriptive t statistics.

The primary estimand is task-progress difference at the shorter survival prefix of
each condition/scratch episode pair. Positive values therefore cannot be explained
only by the condition remaining alive for more steps.

| task | step | condition vs scratch | pairs | common-prefix progress scratch→condition (Δ) | raw progress scratch→condition (Δ) | stability Δ | return scratch→condition (Δ) | episode length Δ | early failure scratch→condition (Δ) |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| cabinet | 10000 | wfix | 96 | +0±0→+0.02083±0.0361 (+0.02083±0.0361) | +0±0→+0.02083±0.0361 (+0.02083±0.0361) | -0.00839±0.0267 | +32.54±4.77→+74.84±30.3 (+42.3±35) | +0±0 | +0±0→+0±0 (+0±0) |
| cabinet | 30000 | wfix | 96 | +0.02083±0.018→+0.2604±0.118 (+0.2396±0.126) | +0.02083±0.018→+0.2604±0.118 (+0.2396±0.126) | +0.03028±0.0341 | +89.49±30.6→+210.9±59.7 (+121.5±88.6) | +0±0 | +0±0→+0±0 (+0±0) |
| cabinet | 100000 | wfix | 96 | +0.5±0.36→+0.9479±0.0477 (+0.4479±0.344) | +0.5±0.36→+0.9479±0.0477 (+0.4479±0.344) | +0.128±0.00301 | +181.1±72.6→+244.5±22 (+63.39±93.4) | +0±0 | +0±0→+0±0 (+0±0) |
| maze | 10000 | wfix | 96 | +1±0→+1.542±0.203 (+0.5417±0.203) | +1±0→+1.583±0.191 (+0.5833±0.191) | -0.0658±0.0618 | +141.9±14→+264.8±37.1 (+123±51) | -168.9±141 | +0.4375±0.298→+0.6667±0.148 (+0.2292±0.172) |
| maze | 30000 | wfix | 96 | +1.271±0.469→+1.896±0.0955 (+0.625±0.496) | +1.302±0.523→+1.979±0.0361 (+0.6771±0.559) | +0.02258±0.068 | +213.6±119→+360±11.9 (+146.4±129) | +1.312±257 | +0.2917±0.191→+0.375±0.156 (+0.08333±0.346) |
| maze | 100000 | wfix | 96 | +1.865±0.0786→+2±0 (+0.1354±0.0786) | +1.938±0.0541→+2±0 (+0.0625±0.0541) | +0.08982±0.0638 | +355.4±24.4→+374.4±23.6 (+18.98±47.6) | -41.8±127 | +0.1875±0.108→+0.2188±0.113 (+0.03125±0.219) |
| powerlift | 10000 | wfix | 96 | +0.1904±0.000412→+0.1906±6.66e-05 (+0.000158±0.000366) | +0.1904±0.00041→+0.1906±6.86e-05 (+0.0001573±0.000361) | -0.005802±0.0277 | +122.9±9.9→+115.4±18.7 (-7.53±28.2) | -46.66±195 | +0.3125±0.0827→+0.4375±0.162 (+0.125±0.244) |
| powerlift | 30000 | wfix | 96 | +0.1909±0.000512→+0.1905±0.000255 (-0.0003667±0.000591) | +0.1909±0.000614→+0.1905±0.000255 (-0.0004432±0.000717) | +0.1612±0.0308 | +149.9±9.35→+179.5±26.5 (+29.61±19.4) | -77.98±62 | +0.1354±0.0786→+0.3646±0.203 (+0.2292±0.172) |
| powerlift | 100000 | wfix | 96 | +0.1904±0.000193→+0.1906±0.000143 (+0.0002429±0.000118) | +0.1904±0.000193→+0.1906±0.000143 (+0.0002429±0.000118) | +0.03481±0.0667 | +302.3±11.1→+316±19.4 (+13.79±30.5) | +3.99±50.9 | +0.1042±0.0477→+0.08333±0.0722 (-0.02083±0.0786) |
| basketball | 10000 | wfix | 96 | +0±0→+0±0 (+0±0) | +0±0→+0±0 (+0±0) | +0.04674±0.0784 | +19.97±2.13→+16.13±0.321 (-3.843±2) | -10.46±4.57 | +1±0→+1±0 (+0±0) |
| basketball | 30000 | wfix | 96 | +0.01042±0.018→+0±0 (-0.01042±0.018) | +0.01042±0.018→+0±0 (-0.01042±0.018) | +0.1655±0.187 | +57.84±17.5→+37.8±7.14 (-20.04±14.4) | -10.39±25.5 | +0.9896±0.018→+1±0 (+0.01042±0.018) |
| basketball | 100000 | wfix | 96 | +0.3229±0.172→+0.1354±0.0955 (-0.1875±0.0827) | +0.3438±0.174→+0.2708±0.148 (-0.07292±0.0786) | +0.1628±0.238 | +390±173→+326±147 (-64.02±84.5) | +12.14±8.49 | +0.6562±0.174→+0.7292±0.148 (+0.07292±0.0786) |

## Secondary task metrics at the common survival prefix

| task | step | condition | metric | scratch→condition (Δ) |
|---|---:|---|---|---:|
| cabinet | 10000 | wfix | door_openness_reward | +0.03559±0.00306→+0.1264±0.0602 (+0.09076±0.0631) |
| cabinet | 10000 | wfix | subtask_complete | +0±0→+0.02083±0.0361 (+0.02083±0.0361) |
| cabinet | 30000 | wfix | door_openness_reward | +0.138±0.0441→+0.5169±0.121 (+0.3789±0.163) |
| cabinet | 30000 | wfix | subtask_complete | +0.02083±0.018→+0.2604±0.118 (+0.2396±0.126) |
| cabinet | 100000 | wfix | door_openness_reward | +0.6444±0.335→+0.9613±0.0133 (+0.3169±0.328) |
| cabinet | 100000 | wfix | subtask_complete | +0.4896±0.346→+0.9271±0.0477 (+0.4375±0.312) |
| maze | 10000 | wfix | stage_convert_reward | +100±0→+208.3±40.7 (+108.3±40.7) |
| maze | 10000 | wfix | checkpoint_proximity_reward | +6.551e-06±1.13e-05→+0.4975±0.176 (+0.4975±0.176) |
| maze | 10000 | wfix | move | +0.1668±9.22e-05→+0.2246±0.0308 (+0.05777±0.0307) |
| maze | 30000 | wfix | stage_convert_reward | +154.2±93.8→+279.2±19.1 (+125±99.2) |
| maze | 30000 | wfix | checkpoint_proximity_reward | +0.2369±0.41→+0.7843±0.0977 (+0.5474±0.413) |
| maze | 30000 | wfix | move | +0.1814±0.0161→+0.2545±0.0385 (+0.07312±0.0497) |
| maze | 100000 | wfix | stage_convert_reward | +272.9±15.7→+300±0 (+27.08±15.7) |
| maze | 100000 | wfix | checkpoint_proximity_reward | +0.5834±0.157→+0.7923±0.0288 (+0.2088±0.181) |
| maze | 100000 | wfix | move | +0.1984±0.0124→+0.2035±0.0108 (+0.005101±0.0192) |
| basketball | 10000 | wfix | success_subtasks | +1±0→+1±0 (+0±0) |
| basketball | 10000 | wfix | reward_hand_proximity | +0.6702±0.0642→+0.6408±0.0831 (-0.02938±0.132) |
| basketball | 10000 | wfix | reward_ball_success | +0.4028±0.0684→+0.3639±0.0151 (-0.0389±0.0772) |
| basketball | 30000 | wfix | success_subtasks | +1±0→+1±0 (+0±0) |
| basketball | 30000 | wfix | reward_hand_proximity | +0.6599±0.119→+0.6269±0.0621 (-0.03301±0.0574) |
| basketball | 30000 | wfix | reward_ball_success | +0.8038±0.231→+0.5964±0.0846 (-0.2074±0.156) |
| basketball | 100000 | wfix | success_subtasks | +1±0→+1±0 (+0±0) |
| basketball | 100000 | wfix | reward_hand_proximity | +0.6288±0.106→+0.6864±0.0372 (+0.05751±0.124) |
| basketball | 100000 | wfix | reward_ball_success | +0.9681±0.0262→+0.8994±0.0368 (-0.06863±0.0627) |

Interpretation rule: a positive raw-progress delta with a near-zero common-prefix
delta is stability/exposure-mediated; a positive common-prefix delta is evidence of
task-progress acceleration beyond merely surviving longer.
