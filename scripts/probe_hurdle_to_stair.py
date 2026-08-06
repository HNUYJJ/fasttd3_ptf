"""hurdle→stair 抬腿迁移 zero-shot 探针(RBO-PTF 源库扩展想法验证)。

观察: stair 现有源只有 stand/walk/run(平地移动), h1 学不会抬腿、上不去第一级台阶。
假设: hurdle(跨栏)策略含"抬腿越障" motor primitive, 在 stair 上 zero-shot 能否
      抬腿、不被台阶绊倒、往上走。若 hurdle 明显优于 loco 源, 则证明跨技能源迁移,
      把贡献①从"loco 内部源选择"升级为"跨技能库源选择"(论文最亮正例)。

做法(不训练, 纯 zero-shot rollout, 复用 task_progress_audit 基础设施):
  五个 frozen policy 都在 h1hand-stair-v0 确定性 rollout, 对比:
    - fall%(被台阶绊倒摔=失败) / ep_len(站得久=没早摔) / return / stair info 分量。
  源: hurdle(实验组) vs walk/run/stand(现有 loco 对照) vs stair-scratch(in-domain 参照,
      用户实测它自己也上不去第一级台阶)。

用法: CUDA_VISIBLE_DEVICES=7 python scripts/probe_hurdle_to_stair.py
"""
from __future__ import annotations

import os
import sys
import time
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

ACTION_DIM = 61
TARGET_ENV = "h1hand-stair-v0"
OBS_DIM = 151
NUM_ENVS = 16
EPISODE_STEPS = 1000
ENV_SEED = 7

# 实验组 + in-domain 参照(直接构造 SourcePolicy 的训练 ckpt)
CKPT_SOURCES = {
    "hurdle": "models/h1hand-hurdle-v0__h1hand_hurdle_mt_safe_s1_20260614T093518Z__1_final.pt",
    "stair_scr": "models/h1hand-stair-v0__h1hand_stair_tp_scr_s1_20260615T044012Z__1_final.pt",
}
# 现有 loco 源(official, 走 from_spec 读 manifest)
OFFICIAL_SOURCES = {
    "walk": "checkpoints/official_sources/h1hand_walk/manifest.json",
    "run": "checkpoints/official_sources/h1hand_run/manifest.json",
    "stand": "checkpoints/official_sources/h1hand_stand/manifest.json",
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
                if isinstance(v, (bool, np.bool_)):
                    info_sums[k] += float(v)
                elif isinstance(v, (int, float, np.floating, np.integer)):
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
    info_means = {k: v / max(info_steps, 1) for k, v in info_sums.items()
                  if k != "TimeLimit.truncated"}
    return dict(
        return_mean=float(np.mean(rets)), return_std=float(np.std(rets)),
        ep_len_mean=float(np.mean(lens)), fall=float(np.mean(falls)),
        episodes=len(rets), info_means=info_means,
    )


def load_source(name: str, device: torch.device) -> SourcePolicy:
    if name in CKPT_SOURCES:
        return SourcePolicy(
            name, CKPT_SOURCES[name], device,
            target_action_dim=ACTION_DIM,
            source_obs_dim=OBS_DIM, source_action_dim=ACTION_DIM,
            obs_adapter_spec={"type": "identity", "output_dim": OBS_DIM},
        )
    return SourcePolicy.from_spec(
        {"name": name, "manifest": OFFICIAL_SOURCES[name]}, device, ACTION_DIM)


def main() -> None:
    ensure_fasttd3_import_path()
    ensure_humanoidbench_import_path()
    from stable_baselines3.common.vec_env import SubprocVecEnv

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    envs = SubprocVecEnv([make_env_fn(TARGET_ENV, i, EPISODE_STEPS, ENV_SEED)
                          for i in range(NUM_ENVS)])

    order = ["hurdle", "walk", "run", "stand", "stair_scr"]
    results = {}
    for name in order:
        sp = load_source(name, device)
        act_fn = lambda o, _sp=sp: _sp.act(
            torch.as_tensor(o, device=device, dtype=torch.float32)).cpu().numpy()
        t0 = time.time()
        res = rollout(envs, act_fn, NUM_ENVS, EPISODE_STEPS)
        results[name] = res
        print(f"[{name:10s}] ret={res['return_mean']:8.1f}±{res['return_std']:5.1f} "
              f"fall={res['fall']:4.0%} ep_len={res['ep_len_mean']:6.1f} "
              f"({res['episodes']} ep, {time.time()-t0:.0f}s)", flush=True)
    envs.close()

    print("\n" + "=" * 78)
    print("hurdle→stair zero-shot 探针: 抬腿是否迁移(fall低/ep_len长/return高=抬腿不被绊倒)")
    print("=" * 78)
    print(f"{'source':12s} {'return':>10s} {'fall%':>7s} {'ep_len':>8s}")
    for name in order:
        r = results[name]
        print(f"{name:12s} {r['return_mean']:10.1f} {r['fall']:7.0%} {r['ep_len_mean']:8.1f}")

    # stair info 分量(找代表"前进/登台阶"的 key): 列出所有源都非零的分量, hurdle 列优先
    print("\n--- stair info 分量(per-step 均值; 看哪个代表前进/登高) ---")
    all_keys = sorted({k for r in results.values() for k in r["info_means"]})
    print(f"{'component':32s} " + " ".join(f"{n:>9s}" for n in order))
    for k in all_keys:
        vals = [results[n]["info_means"].get(k, 0.0) for n in order]
        if all(abs(v) < 1e-9 for v in vals):
            continue
        print(f"{k:32s} " + " ".join(f"{v:9.3f}" for v in vals))


if __name__ == "__main__":
    main()
