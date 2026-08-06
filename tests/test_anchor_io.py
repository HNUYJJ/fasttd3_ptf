from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import torch
from tensordict import TensorDict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fasttd3_ptf.official_fasttd3_ptf.anchor_io import (
    load_anchor_bundle,
    save_anchor_bundle,
    verify_anchor_bundle,
)
from fasttd3_ptf.official_fasttd3_ptf.paths import ensure_fasttd3_import_path

ensure_fasttd3_import_path()
from fast_td3_utils import SimpleReplayBuffer  # type: ignore  # noqa: E402

from fasttd3_ptf.official_fasttd3_ptf.ptf_replay import PTFReplayWrapper  # noqa: E402


def _objects(seed: int):
    torch.manual_seed(seed)
    model = torch.nn.Linear(3, 2)
    target = torch.nn.Linear(3, 2)
    target.load_state_dict(model.state_dict())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
    scaler = torch.amp.GradScaler(enabled=False)
    replay = PTFReplayWrapper(
        SimpleReplayBuffer(
            n_env=2,
            buffer_size=8,
            n_obs=3,
            n_act=2,
            n_critic_obs=3,
            n_steps=1,
            device=torch.device("cpu"),
        )
    )
    generator = torch.Generator().manual_seed(seed + 99)
    return model, target, optimizer, scheduler, scaler, replay, generator


def _add_transition(replay: PTFReplayWrapper, value: int) -> None:
    obs = torch.full((2, 3), float(value))
    replay.extend(
        TensorDict(
            {
                "observations": obs,
                "actions": torch.full((2, 2), value / 10.0),
                "next": {
                    "rewards": torch.full((2,), float(value)),
                    "dones": torch.zeros(2, dtype=torch.long),
                    "truncations": torch.zeros(2, dtype=torch.long),
                    "observations": obs + 1,
                },
            },
            batch_size=2,
        ),
        torch.full((2,), -1),
    )


def _update(model, optimizer, scheduler, x, y):
    optimizer.zero_grad(set_to_none=True)
    loss = torch.nn.functional.mse_loss(model(x), y)
    loss.backward()
    optimizer.step()
    scheduler.step()


def test_anchor_roundtrip_reproduces_next_update_and_rng(tmp_path):
    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    model, target, optimizer, scheduler, scaler, replay, generator = _objects(7)
    for step in range(3):
        _add_transition(replay, step)
        _update(
            model,
            optimizer,
            scheduler,
            torch.full((4, 3), float(step)),
            torch.full((4, 2), 0.25),
        )
    target.load_state_dict(model.state_dict())
    bundle = save_anchor_bundle(
        tmp_path / "anchor",
        completed_vector_steps=3,
        num_envs=2,
        modules={"actor": model, "target": target},
        optimizers={"actor": optimizer},
        schedulers={"actor": scheduler},
        scaler=scaler,
        replay=replay,
        configuration={"env": "toy", "seed": 7},
        auxiliary_state={"update_count": 3},
        generators={"option": generator},
        provenance_default={"behavior_source": "student"},
        repo_root=Path(__file__).resolve().parents[1],
        code_paths=[Path(__file__).resolve()],
    )
    manifest = verify_anchor_bundle(bundle)
    assert manifest["replay_ptr"] == 3
    assert manifest["environment_transitions"] == 6

    baseline_random = (random.random(), np.random.rand(), torch.rand(3), torch.rand(3, generator=generator))
    x = torch.rand(5, 3)
    y = torch.rand(5, 2)
    _update(model, optimizer, scheduler, x, y)
    baseline_params = {name: value.detach().clone() for name, value in model.state_dict().items()}

    model_b, target_b, optimizer_b, scheduler_b, scaler_b, replay_b, generator_b = _objects(999)
    loaded = load_anchor_bundle(
        bundle,
        modules={"actor": model_b, "target": target_b},
        optimizers={"actor": optimizer_b},
        schedulers={"actor": scheduler_b},
        scaler=scaler_b,
        replay=replay_b,
        generators={"option": generator_b},
    )
    assert loaded["auxiliary_state"]["update_count"] == 3
    branch_random = (
        random.random(),
        np.random.rand(),
        torch.rand(3),
        torch.rand(3, generator=generator_b),
    )
    assert baseline_random[0] == branch_random[0]
    assert baseline_random[1] == branch_random[1]
    torch.testing.assert_close(baseline_random[2], branch_random[2], rtol=0.0, atol=0.0)
    torch.testing.assert_close(baseline_random[3], branch_random[3], rtol=0.0, atol=0.0)
    _update(model_b, optimizer_b, scheduler_b, x, y)
    for name, value in model_b.state_dict().items():
        torch.testing.assert_close(value, baseline_params[name], rtol=0.0, atol=0.0)

    original_batch = replay.gather(torch.tensor([[0, 2], [1, 0]]))
    restored_batch = replay_b.gather(torch.tensor([[0, 2], [1, 0]]))
    for key in original_batch.keys(True, True):
        assert torch.equal(original_batch[key], restored_batch[key]), key

