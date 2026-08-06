"""hurdle 上"源的天花板"探针：源策略本身在 target 上能拿多少回报？

动机（2026-07-30，hurdle_speedup_v1 结果的后续）：
speedup 实验的 source 臂全程恒定 50% 剂量（`PTF_MCG_WARMUP_STEPS` = 总步数），
到 75k 时 student 已达 698，但源仍在以一半比例注入行为与 replay。
若源自身的 zero-shot 回报远低于 student 的当期水平，则恒定剂量在后期
构成**过度约束**——这与 2026-05-20 force-PTF 的结论（机制健康、恒定 λ 压制 actor）同因。

本探针只做 zero-shot rollout，不训练，用于判断上述前提是否成立：
读出 run / walk / stand 三个 loco 源在 h1hand-hurdle-v0 上的确定性回报，
与 speedup 实验已测的 student 曲线逐点对比。

用法: CUDA_VISIBLE_DEVICES=0 python scripts/probe_hurdle_source_ceiling.py
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from fasttd3_ptf.official_fasttd3_ptf.paths import (
    ensure_fasttd3_import_path,
    ensure_humanoidbench_import_path,
)
from fasttd3_ptf.ptf.source_policy import SourcePolicy

TARGET_ENV = "h1hand-hurdle-v0"
ACTION_DIM = 61
OBS_DIM = 151
NUM_ENVS = 32
EPISODE_STEPS = 1000
ENV_SEED = 7
OUT = Path("docs/data/hurdle_speedup_v1/source_ceiling_probe.json")

OFFICIAL_SOURCES = {
    "run": "checkpoints/official_sources/h1hand_run/manifest.json",
    "walk": "checkpoints/official_sources/h1hand_walk/manifest.json",
    "stand": "checkpoints/official_sources/h1hand_stand/manifest.json",
}

# speedup 实验已测的 source 臂 student 曲线（source-free，128 ep），用于对比
STUDENT_CURVE = {10000: 64.75, 20000: 241.72, 30000: 396.01,
                 50000: 435.51, 75000: 645.88, 100000: 479.07}


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
def rollout(envs, act_fn, num_envs: int, episode_steps: int):
    obs = envs.reset()
    ep_ret = np.zeros(num_envs)
    ep_len = np.zeros(num_envs, dtype=np.int64)
    active = np.ones(num_envs, dtype=bool)
    rets, lens, falls = [], [], []
    info_sums: dict[str, float] = defaultdict(float)
    info_steps = 0
    for _ in range(episode_steps + 5):
        act = act_fn(obs)
        obs, rew, dones, infos = envs.step(act)
        ep_ret[active] += rew[active]
        ep_len[active] += 1
        for i in np.nonzero(active)[0]:
            for k, v in infos[i].items():
                if isinstance(v, (bool, np.bool_, int, float, np.floating, np.integer)):
                    info_sums[k] += float(v)
            info_steps += 1
        for i in np.nonzero(dones & active)[0]:
            truncated = bool(infos[i].get("TimeLimit.truncated", False))
            rets.append(float(ep_ret[i]))
            lens.append(int(ep_len[i]))
            falls.append(0.0 if truncated else 1.0)
            active[i] = False
        if not active.any():
            break
    for i in np.nonzero(active)[0]:
        rets.append(float(ep_ret[i]))
        lens.append(int(ep_len[i]))
        falls.append(0.0)
    return dict(
        return_mean=float(np.mean(rets)), return_std=float(np.std(rets)),
        return_se=float(np.std(rets) / max(np.sqrt(len(rets)), 1.0)),
        ep_len_mean=float(np.mean(lens)), fall_rate=float(np.mean(falls)),
        episodes=len(rets),
        info_means={k: v / max(info_steps, 1) for k, v in info_sums.items()
                    if k != "TimeLimit.truncated"},
    )


def main() -> None:
    ensure_fasttd3_import_path()
    ensure_humanoidbench_import_path()
    from stable_baselines3.common.vec_env import SubprocVecEnv

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    envs = SubprocVecEnv(
        [make_env_fn(TARGET_ENV, i, EPISODE_STEPS, ENV_SEED) for i in range(NUM_ENVS)])

    results = {}
    try:
        for name, manifest in OFFICIAL_SOURCES.items():
            src = SourcePolicy.from_spec(
                {"name": name, "manifest": manifest}, device, ACTION_DIM)

            def act_fn(obs, _src=src):
                t = torch.as_tensor(obs, device=device, dtype=torch.float32)
                return _src.act(t).cpu().numpy()

            r = rollout(envs, act_fn, NUM_ENVS, EPISODE_STEPS)
            results[name] = r
            print(f"[{name:5s}] return={r['return_mean']:8.2f} ± {r['return_se']:.2f} (SE)  "
                  f"ep_len={r['ep_len_mean']:6.1f}  fall={r['fall_rate']:.2f}  "
                  f"n={r['episodes']}")
    finally:
        envs.close()

    print("\n=== 与 speedup 实验 source 臂的 student 曲线对比 ===")
    ceiling = results["run"]["return_mean"]
    print(f"run 源自身 zero-shot 回报 = {ceiling:.2f}")
    for step, sret in sorted(STUDENT_CURVE.items()):
        rel = "student 更强" if sret > ceiling else "源更强"
        print(f"  student@{step//1000:3d}k = {sret:7.2f}   "
              f"student/源 = {sret/max(ceiling,1e-9):6.2f}x   {rel}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(
        {"env": TARGET_ENV, "num_envs": NUM_ENVS, "episode_steps": EPISODE_STEPS,
         "env_seed": ENV_SEED, "deterministic": True,
         "sources": results, "student_curve_for_reference": STUDENT_CURVE},
        indent=2, ensure_ascii=False))
    print(f"\nwritten {OUT}")


if __name__ == "__main__":
    main()
