# Implementation notes

## Mapping from papers to code

- FastTD3 backbone: `fasttd3_ptf/agents/fasttd3_agent.py`, `fasttd3_ptf/models/actor.py`, `fasttd3_ptf/models/critic.py`, `fasttd3_ptf/buffers/replay_buffer.py`.
- PTF option module: `fasttd3_ptf/models/option.py`, `fasttd3_ptf/ptf/option_selector.py`, `fasttd3_ptf/agents/ptf_fasttd3_agent.py`.
- Source policy bank: `fasttd3_ptf/ptf/source_policy.py`, `fasttd3_ptf/ptf/source_bank.py`.
- Continuous-action transfer loss: `fasttd3_ptf/ptf/distillation.py`.
- Off-policy multi-option compatibility: `fasttd3_ptf/ptf/compatibility.py`.
- HumanoidBench-facing wrappers/configs: `fasttd3_ptf/envs/registry.py`, `configs/source/humanoidbench`, `configs/target/humanoidbench`.

## PTF-FastTD3 design choice

The target actor always outputs the environment action. Frozen source policies are not directly executed by default; they shape the target actor through a masked action-distillation loss:

`actor_loss = -Q(s, actor(s)) + lambda(t) * (1 - beta(s, option)) * D(actor(s), source_option(s))`

The null option has no source action and therefore disables transfer on those samples.

## HumanoidBench adapter warning

The included `h1hand_basic_sources.yaml` now uses provisional `h1hand_default` action groups: stand/balance supervise legs/torso plus proximal arm stabilizers, walk/run supervise legs/torso, and reach supervises upper body plus hands. Before serious experiments, inspect the action ordering in your HumanoidBench version and adjust `fasttd3_ptf/envs/action_schema.py` or the YAML masks. This is especially important for locomotion sources, because hand/finger actions from a walking policy can hurt manipulation.

Passthrough observation/action adapters require exact dimensions. Cross-task transfer should use explicit `robot_only`, `reach`, `slice`, or `action_pad` adapters rather than implicit truncation or zero padding.

## Validation performed in this environment

The code was syntax-checked with:

```bash
python -m compileall -q fasttd3_ptf tests
```

After installing `gymnasium`, a short toy source run and a short toy PTF target run completed successfully on CPU with tiny overrides. Full HumanoidBench training is not validated here because the HumanoidBench/MuJoCo task suite is not installed in this execution environment.
