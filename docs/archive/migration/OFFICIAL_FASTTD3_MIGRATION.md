# Official FastTD3 Migration

This project keeps the FastTD3 runtime source used by experiments at:

- `fasttd3_ptf/official_code/FastTD3/fast_td3/`: official_code project-local snapshot.

The official_code snapshot should stay source-compatible with the original FastTD3
files. Do not put PTF logic inside `fasttd3_ptf/official_code/FastTD3/fast_td3/train.py`
unless a change is explicitly part of an auditable patch.

## Intended Experiment Split

- FastTD3 source policies: run official FastTD3 `train.py`.
- FastTD3 target baseline: run official FastTD3 `train.py`.
- PTF-FastTD3: create a separate official-based entry point outside the
  official_code tree, reusing official actor, critic, replay buffer, normalizers,
  hyperparameters, HumanoidBench wrapper, logging, evaluation, and rendering.

## Current Official-Based PTF Entry

The first official-based PTF entry point is:

```bash
conda run -n FastTD3 python -m fasttd3_ptf.official_fasttd3_ptf.train_ptf \
  --env-name h1hand-push-v0 \
  --exp-name official_ptf_push \
  --project fasttd3_ptf \
  --ptf-source-bank configs/source_banks/h1hand_basic_sources.yaml
```

The entry point keeps `fasttd3_ptf/official_code/FastTD3/fast_td3` untouched
and adds PTF from the outside:

- official `Actor` / `Critic`;
- official `SimpleReplayBuffer`, wrapped by an external option-id buffer;
- official `EmpiricalNormalization` and reward normalizers;
- official optimizer, scheduler, AMP, eval, render, and W&B metric names;
- PTF `SourcePolicyBank`, `OptionModule`, option selection, beta termination,
  and beta-weighted masked action distillation.

Official HumanoidBench FastTD3 defaults to `num_steps=1`, and the current PTF
option replay wrapper intentionally supports only `num_steps=1`.

## Official Source Checkpoint Export

Convenience scripts are available:

```bash
conda run -n FastTD3 bash scripts/official_fasttd3_train_h1hand_sources.sh
conda run -n FastTD3 bash scripts/official_fasttd3_export_h1hand_sources.sh
```

Target comparison scripts:

```bash
conda run -n FastTD3 bash scripts/official_fasttd3_train_target_scratch.sh
conda run -n FastTD3 bash scripts/official_fasttd3_train_target_ptf.sh
```

Useful environment overrides include `ENV_NAME`, `SOURCE_BANK`, `SEED`,
`PROJECT`, `TOTAL_TIMESTEPS`, `NUM_ENVS`, `BATCH_SIZE`, `BUFFER_SIZE`,
`LEARNING_STARTS`, `NUM_UPDATES`, `POLICY_FREQUENCY`, `DEVICE_RANK`, `CUDA`,
`WANDB`, `COMPILE`, `AMP`, `EVAL_INTERVAL`, and `RENDER_INTERVAL`.

After training a source policy with official FastTD3, export a source manifest:

```bash
conda run -n FastTD3 python -m fasttd3_ptf.source_bank.exporter \
  --checkpoint models/h1hand-stand-v0__h1hand_stand_source__1_final.pt \
  --env-id h1hand-stand-v0 \
  --name stand \
  --output checkpoints/sources/h1hand_stand/manifest.json
```

Then build a source bank:

```bash
conda run -n FastTD3 python -m fasttd3_ptf.source_bank.builder \
  --sources \
    checkpoints/sources/h1hand_stand/manifest.json \
    checkpoints/sources/h1hand_walk/manifest.json \
    checkpoints/sources/h1hand_reach/manifest.json \
  --output configs/source_banks/h1hand_stand_walk_reach_sources.yaml
```

## Modular Implementation Location

The older modular implementation is now grouped under
`fasttd3_ptf/my_fasttd3_ptf/`. Keep it for toy smoke tests and debugging until
the official-source PTF path fully covers the same research workflow. Remove
only the self-implemented FastTD3 training core after:

1. Official FastTD3-only source training runs from the official_code copy.
2. Official FastTD3-only target training runs from the official_code copy.
3. Official-based PTF target training runs with source-bank loading, option ids,
   beta-weighted action distillation, and option diagnostics.
4. Toy tests and HumanoidBench smoke checks pass through the new path.

After that, remove only the self-implemented FastTD3 training core. Keep PTF
modules that are still used by the official-based PTF entry point.
