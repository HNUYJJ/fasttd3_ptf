"""零训练的 task-progress 粗筛探针（预注册 docs/experiments/progress_screen_v1_prereg_20260804.md）。

测 3 个 loco 源在 4 个 locomotion 环境上的 zero-shot 前进进度。

主测量 P(i,T) = 每 episode `max_t(x_t − x_0)` 的均值，x = qpos[0]。
`Task.get_obs()` 返回 `concat(qpos, qvel)`（humanoid_bench/tasks.py:32），
故 `obs[0] == qpos[0]`，位移可零开销地从观测读出，无需跨进程访问 MuJoCo。

**done 那一步必须用 `info["terminal_observation"]`**：SB3 VecEnv 在 done 时
返回的已经是下一个 episode 的首帧观测，直接读会把位移算成新 episode 的起点差。

用法: CUDA_VISIBLE_DEVICES=1 python scripts/analysis/probe_source_progress_screen_v1.py
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch

from fasttd3_ptf.official_fasttd3_ptf.paths import (
    ensure_fasttd3_import_path,
    ensure_humanoidbench_import_path,
)
from fasttd3_ptf.ptf.source_policy import SourcePolicy

ACTION_DIM = 61
NUM_ENVS = 32
EPISODE_STEPS = 1000
ENV_SEED = 7                      # 沿用 probe_hurdle_source_ceiling.py 的常量
OUT = Path("docs/data/progress_screen_v1/probe.json")

# 判据用 crawl / hurdle / slide；walk 仅作辅助解释（预注册 §4）
ENVS = [
    "h1hand-crawl-v0",
    "h1hand-hurdle-v0",
    "h1hand-slide-v0",
    "h1hand-walk-v0",
]

SOURCES = {
    "stand": "checkpoints/official_sources/h1hand_stand/manifest.json",
    "walk": "checkpoints/official_sources/h1hand_walk/manifest.json",
    "run": "checkpoints/official_sources/h1hand_run/manifest.json",
}


def make_env_fn(env_id: str, rank: int, max_steps: int, seed: int):
    def _init():
        ensure_humanoidbench_import_path()
        import gymnasium as gym
        import humanoid_bench  # noqa: F401
        from gymnasium.wrappers import TimeLimit

        env = gym.make(env_id)
        env = TimeLimit(env, max_episode_steps=max_steps)
        env.unwrapped.seed(seed + rank)
        return env

    return _init


@torch.no_grad()
def rollout(envs, act_fn, num_envs: int, episode_steps: int) -> dict:
    obs = envs.reset()
    x0 = np.asarray(obs)[:, 0].astype(np.float64).copy()
    max_dx = np.zeros(num_envs, dtype=np.float64)
    ep_ret = np.zeros(num_envs, dtype=np.float64)
    ep_len = np.zeros(num_envs, dtype=np.int64)
    active = np.ones(num_envs, dtype=bool)

    rets: list[float] = []
    dxs: list[float] = []
    lens: list[int] = []
    falls: list[float] = []
    info_sums: dict[str, float] = defaultdict(float)
    info_steps = 0

    for _ in range(episode_steps + 5):
        act = act_fn(obs)
        obs, rew, dones, infos = envs.step(act)
        obs_arr = np.asarray(obs)
        for i in np.nonzero(active)[0]:
            # done 时 obs 已是下一 episode 首帧,必须读 terminal_observation
            if dones[i] and "terminal_observation" in infos[i]:
                xi = float(np.asarray(infos[i]["terminal_observation"])[0])
            else:
                xi = float(obs_arr[i, 0])
            max_dx[i] = max(max_dx[i], xi - x0[i])
            ep_ret[i] += float(rew[i])
            ep_len[i] += 1
            for k, v in infos[i].items():
                if isinstance(v, (bool, np.bool_, int, float, np.floating, np.integer)):
                    info_sums[k] += float(v)
            info_steps += 1
        for i in np.nonzero(dones & active)[0]:
            truncated = bool(infos[i].get("TimeLimit.truncated", False))
            rets.append(float(ep_ret[i]))
            dxs.append(float(max_dx[i]))
            lens.append(int(ep_len[i]))
            falls.append(0.0 if truncated else 1.0)
            active[i] = False
        if not active.any():
            break

    for i in np.nonzero(active)[0]:          # 跑满未 done 的
        rets.append(float(ep_ret[i]))
        dxs.append(float(max_dx[i]))
        lens.append(int(ep_len[i]))
        falls.append(0.0)

    n = max(len(dxs), 1)
    return dict(
        progress_max_dx_mean=float(np.mean(dxs)),
        progress_max_dx_std=float(np.std(dxs)),
        progress_max_dx_se=float(np.std(dxs) / np.sqrt(n)),
        progress_max_dx_per_episode=dxs,
        return_mean=float(np.mean(rets)),
        return_se=float(np.std(rets) / np.sqrt(n)),
        ep_len_mean=float(np.mean(lens)),
        fall_rate=float(np.mean(falls)),
        episodes=len(dxs),
        info_means={k: v / max(info_steps, 1) for k, v in info_sums.items()
                    if k != "TimeLimit.truncated"},
    )


def main() -> None:
    ensure_fasttd3_import_path()
    ensure_humanoidbench_import_path()
    from stable_baselines3.common.vec_env import SubprocVecEnv

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results: dict[str, dict] = {}

    for env_id in ENVS:
        print(f"\n=== {env_id} ===", flush=True)
        envs = SubprocVecEnv(
            [make_env_fn(env_id, i, EPISODE_STEPS, ENV_SEED) for i in range(NUM_ENVS)]
        )
        try:
            for name, manifest in SOURCES.items():
                src = SourcePolicy.from_spec(
                    {"name": name, "manifest": manifest}, device, ACTION_DIM
                )

                def act_fn(o, _src=src):
                    t = torch.as_tensor(o, device=device, dtype=torch.float32)
                    return _src.act(t).cpu().numpy()

                r = rollout(envs, act_fn, NUM_ENVS, EPISODE_STEPS)
                results[f"{env_id}|{name}"] = r
                print(
                    f"  [{name:5s}] dx={r['progress_max_dx_mean']:7.3f} "
                    f"± {r['progress_max_dx_se']:.3f}  "
                    f"return={r['return_mean']:8.2f}  "
                    f"ep_len={r['ep_len_mean']:6.1f}  fall={r['fall_rate']:.2f}",
                    flush=True,
                )
        finally:
            envs.close()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(
        {
            "prereg": "docs/experiments/progress_screen_v1_prereg_20260804.md",
            "protocol": {
                "envs": ENVS, "sources": list(SOURCES),
                "num_envs": NUM_ENVS, "episode_steps": EPISODE_STEPS,
                "env_seed": ENV_SEED, "deterministic": True, "trained": False,
                "measure": "max_t(x_t - x_0), x = qpos[0] = obs[0]",
            },
            "results": results,
        },
        indent=2, ensure_ascii=False))
    print(f"\nwritten {OUT}")


if __name__ == "__main__":
    main()
