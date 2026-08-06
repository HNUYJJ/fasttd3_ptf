# h1hand-push FastTD3 vs PTF-FastTD3

## Goal

Test whether PTF-FastTD3 accelerates or improves `h1hand-push-v0` compared with FastTD3-only under aligned target-task hyperparameters.

## Target

- Target task: `h1hand-push-v0`
- Baseline config: `configs/target/humanoidbench/ablations/h1hand_push_fasttd3_scratch.yaml`
- PTF config: `configs/target/humanoidbench/h1hand_push_ptf.yaml`
- Both target runs use:
  - `seed: 10`
  - `num_envs: 128`
  - `batch_size: 32768`
  - `buffer_size_per_env: 50000`
  - `num_updates: 4`
  - `total_env_steps: 15000000`
  - `learning_starts: 8192`
  - `save_interval: 500000`

## Sources

The first source bank uses source policies whose action/observation semantics are clean for push:

- `h1hand-stand-v0`
- `h1hand-walk-v0`
- `h1hand-run-v0`
- `h1hand-reach-v0`

The push bank is `configs/source_banks/h1hand_push_sources.yaml`.

`reach` is included because `h1hand-push` observations begin with `robot 151 + left_hand 3 + target 3`, matching the `h1hand-reach` source observation prefix.

## Commands

```bash
conda run -n FastTD3 bash scripts/experiments/h1hand_push_compare.sh
conda run -n FastTD3 bash scripts/experiments/h1hand_push_eval.sh 20
```

W&B project: `fasttd3_ptf`.

## Primary Metrics

Compare learning curves from JSONL logs:

- `episode_return_mean`
- `speed`
- `frame`
- `actor_lr`
- `critic_lr`
- `env_rewards`
- `buffer_rewards`
- `critic_loss`
- `qf_loss`
- `qf_max`
- `qf_min`
- `rl_actor_loss`
- `actor_loss`
- `actor_grad_norm`
- `critic_grad_norm`

PTF diagnostics to inspect:

- `transfer_loss`
- `ptf_option_q_loss`
- `ptf_beta_loss`
- `ptf_null_option_ratio`
- `ptf_option_frac/*`
- `ptf_beta_mean/*`
- `ptf_source_compat/*`

## Interpretation

PTF is useful if it reaches the same return with fewer environment steps or obtains a higher final return without destabilizing critic/actor losses.

If PTF underperforms, first check whether `ptf_null_option_ratio` rises when sources are unhelpful. If null usage stays low while transfer loss remains high, the transfer pressure is probably too strong or option termination is not rejecting bad source actions quickly enough.
