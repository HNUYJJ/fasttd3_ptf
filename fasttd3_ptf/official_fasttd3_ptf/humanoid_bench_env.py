"""Project-local HumanoidBench vector wrapper with complete worker seeding.

The upstream FastTD3 wrapper calls ``env.unwrapped.seed``. HumanoidBench's
legacy ``seed`` method only seeds NumPy's process-global RNG, while base reset
noise is drawn from Gymnasium's per-environment ``np_random``.  Conversely,
some task resets (notably basketball) still call process-global ``np.random``.
Stable-Baselines3 supplies a seed to the next ``reset(seed=...)``; the local
wrapper below applies that seed to both RNGs inside each worker process.  This
keeps the upstream tensor interface without modifying vendored source.
"""

from __future__ import annotations

from collections.abc import Callable

import gymnasium as gym
import numpy as np
import torch
from gymnasium.wrappers import TimeLimit
from stable_baselines3.common.vec_env import SubprocVecEnv

import humanoid_bench  # noqa: F401  # register HumanoidBench environments


_SHORT_EPISODE_ENVS = {
    "h1hand-push-v0",
    "h1-push-v0",
    "h1hand-cube-v0",
    "h1cube-v0",
    "h1hand-basketball-v0",
    "h1-basketball-v0",
    "h1hand-kitchen-v0",
    "h1-kitchen-v0",
}


class GlobalNumpySeedOnReset(gym.Wrapper):
    """Seed legacy task-level NumPy randomness alongside Gymnasium's RNG."""

    def reset(self, *, seed: int | None = None, options=None):
        if seed is not None:
            # Each SubprocVecEnv worker is a separate process, so this does not
            # couple parallel environments. Subsequent automatic resets keep
            # advancing that worker's deterministic global stream.
            np.random.seed(int(seed))
        return self.env.reset(seed=seed, options=options)


def max_episode_steps(env_name: str) -> int:
    return 500 if env_name in _SHORT_EPISODE_ENVS else 1000


def make_env(
    env_name: str,
    rank: int,
    render_mode: str | None = None,
) -> Callable[[], gym.Env]:
    """Build one worker without using HumanoidBench's ineffective legacy seed."""

    del rank  # Retained in the public factory signature for call sites.
    episode_steps = max_episode_steps(env_name)

    def _init() -> gym.Env:
        import humanoid_bench  # noqa: F401

        env = gym.make(env_name, render_mode=render_mode)
        env = TimeLimit(env, max_episode_steps=episode_steps)
        env = GlobalNumpySeedOnReset(env)
        return env

    return _init


class HumanoidBenchEnv:
    """Tensor-facing parallel HumanoidBench environment used by FastTD3/PTF.

    ``seed`` is applied through ``SubprocVecEnv.seed`` and therefore consumed by
    the next reset as ``seed + rank``. Automatic resets then continue each
    worker's deterministic RNG stream normally.
    """

    def __init__(
        self,
        env_name: str,
        num_envs: int = 1,
        render_mode: str | None = None,
        device: torch.device | str | None = None,
        seed: int | None = 0,
    ) -> None:
        self.sim_device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.num_envs = int(num_envs)
        self.envs = SubprocVecEnv(
            [
                make_env(env_name, rank, render_mode=render_mode)
                for rank in range(self.num_envs)
            ]
        )
        if seed is not None:
            # SB3 stores per-worker seeds and supplies them to Gymnasium on the
            # next reset. This is intentionally not env.unwrapped.seed(...).
            self.envs.seed(int(seed))

        self.max_episode_steps = max_episode_steps(env_name)
        self.asymmetric_obs = False
        self.num_obs = int(self.envs.observation_space.shape[-1])
        self.num_actions = int(self.envs.action_space.shape[-1])

    def reset(self, *, seed: int | None = None) -> torch.Tensor:
        if seed is not None:
            self.envs.seed(int(seed))
        observations = self.envs.reset()
        return torch.as_tensor(
            observations,
            device=self.sim_device,
            dtype=torch.float32,
        )

    def render(self):
        if self.num_envs != 1:
            raise AssertionError("render currently supports one environment only")
        return self.envs.render()

    def step(self, actions: torch.Tensor):
        if not isinstance(actions, torch.Tensor):
            raise TypeError("HumanoidBenchEnv.step expects a torch.Tensor")
        observations, rewards, dones, raw_infos = self.envs.step(actions.detach().cpu().numpy())

        true_next = observations.copy()
        truncateds = np.zeros_like(dones)
        for index, info in enumerate(raw_infos):
            if info.get("TimeLimit.truncated", False):
                truncateds[index] = True
                true_next[index] = info["terminal_observation"]

        infos = {
            "observations": {
                "raw": {
                    "obs": torch.as_tensor(
                        true_next,
                        device=self.sim_device,
                        dtype=torch.float32,
                    )
                }
            },
            "time_outs": torch.as_tensor(truncateds, device=self.sim_device),
            "raw_infos": raw_infos,
        }
        return (
            torch.as_tensor(observations, device=self.sim_device, dtype=torch.float32),
            torch.as_tensor(rewards, device=self.sim_device, dtype=torch.float32),
            torch.as_tensor(dones, device=self.sim_device),
            infos,
        )

    def close(self) -> None:
        self.envs.close()
