from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _runtime_available() -> bool:
    try:
        import mujoco  # noqa: F401
        import stable_baselines3  # noqa: F401
    except ImportError:
        return False
    return True


@pytest.mark.skipif(not _runtime_available(), reason="HumanoidBench runtime unavailable")
def test_humanoidbench_reset_and_action_prefix_are_seed_reproducible():
    from fasttd3_ptf.official_fasttd3_ptf.paths import (
        ensure_fasttd3_import_path,
        ensure_humanoidbench_import_path,
    )

    ensure_fasttd3_import_path()
    ensure_humanoidbench_import_path()
    from fasttd3_ptf.official_fasttd3_ptf.humanoid_bench_env import HumanoidBenchEnv

    env_a = HumanoidBenchEnv("h1hand-cabinet-v0", 1, device="cpu", seed=123)
    env_b = HumanoidBenchEnv("h1hand-cabinet-v0", 1, device="cpu", seed=123)
    env_c = HumanoidBenchEnv("h1hand-cabinet-v0", 1, device="cpu", seed=124)
    try:
        obs_a = env_a.reset()
        obs_b = env_b.reset()
        obs_c = env_c.reset()
        torch.testing.assert_close(obs_a, obs_b, rtol=0.0, atol=0.0)
        assert not torch.equal(obs_a, obs_c), "different reset seeds must be non-degenerate"

        action = torch.linspace(-0.15, 0.15, env_a.num_actions).view(1, -1)
        for _ in range(3):
            obs_a, reward_a, done_a, info_a = env_a.step(action)
            obs_b, reward_b, done_b, info_b = env_b.step(action)
            torch.testing.assert_close(obs_a, obs_b, rtol=0.0, atol=0.0)
            torch.testing.assert_close(reward_a, reward_b, rtol=0.0, atol=0.0)
            torch.testing.assert_close(done_a, done_b, rtol=0.0, atol=0.0)
            torch.testing.assert_close(
                info_a["observations"]["raw"]["obs"],
                info_b["observations"]["raw"]["obs"],
                rtol=0.0,
                atol=0.0,
            )
    finally:
        env_a.close()
        env_b.close()
        env_c.close()


@pytest.mark.skipif(not _runtime_available(), reason="HumanoidBench runtime unavailable")
def test_basketball_global_numpy_reset_is_seed_reproducible():
    """Basketball.reset_model uses np.random rather than env.np_random."""

    from fasttd3_ptf.official_fasttd3_ptf.paths import (
        ensure_fasttd3_import_path,
        ensure_humanoidbench_import_path,
    )

    ensure_fasttd3_import_path()
    ensure_humanoidbench_import_path()
    from fasttd3_ptf.official_fasttd3_ptf.humanoid_bench_env import HumanoidBenchEnv

    env_a = HumanoidBenchEnv("h1hand-basketball-v0", 1, device="cpu", seed=321)
    env_b = HumanoidBenchEnv("h1hand-basketball-v0", 1, device="cpu", seed=321)
    env_c = HumanoidBenchEnv("h1hand-basketball-v0", 1, device="cpu", seed=322)
    try:
        obs_a = env_a.reset()
        obs_b = env_b.reset()
        obs_c = env_c.reset()
        torch.testing.assert_close(obs_a, obs_b, rtol=0.0, atol=0.0)
        assert not torch.equal(obs_a, obs_c)
    finally:
        env_a.close()
        env_b.close()
        env_c.close()


def test_stability_collector_does_not_use_legacy_humanoidbench_seed():
    source = (Path(__file__).resolve().parents[1] / "scripts/stability_deconfounded_audit.py").read_text()
    assert "env.unwrapped.seed" not in source
    assert "envs.seed(int(eval_seed))" in source
    assert "GlobalNumpySeedOnReset" in source
    assert "gymnasium_plus_global_numpy_vec_reset_v2" in source
