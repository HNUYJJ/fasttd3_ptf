"""fall-recovery 离线探针(RBO-PTF Open Q2, ChatGPT Step 0; 不训练)。

PI 观察: safe/PTF 训出的 h1 摔倒能自己站起来, scr/rand 起不来; safe 恢复更快更稳。
硬数据侧证: rand 的 eval_avg_length 显著低(摔了早终止)。本探针把定性观察变定量。

做法(passive recovery, 复用 probe_hurdle_to_stair 框架):
  stair/slide/pole/crawl × {safe,rand,scr} 的 seed1 final policy, 在各自 env
  确定性 rollout, 记录 per-env upright(torso up-vector·z) 时序, 检测 near-fall→recovery:
    - fall 事件: upright 从站立(>u_stand)跌到 <u_fall(踉跄/倒);
    - recovery: fall 后 upright 回到 >u_stand 连续 K 步(站稳)且 env 未终止;
    - unrecovered: fall 后直到 episode 因摔倒终止都没站回。
  指标: ep_len / upright_mean / ended_by_fall% / fall 事件数 / recovery_rate / recovery_time。

用法: CUDA_VISIBLE_DEVICES=5 python scripts/probe_fall_recovery.py
"""
from __future__ import annotations

import glob
import os
import sys
import time
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
OBS_DIM = 151
NUM_ENVS = 32
EPISODE_STEPS = 1000
ENV_SEED = 11
TASKS = ["stair", "slide", "pole", "crawl"]
METHODS = ["safe", "rand", "scr"]

U_STAND = 0.75   # 站立: upright > 0.75
U_FALL = 0.50    # 跌倒/踉跄: upright < 0.50
K = 10           # 连续站稳 K 步算 recovery


def find_ckpt(task: str, method: str) -> str | None:
    pats = glob.glob(
        f"models/h1hand-{task}-v0__h1hand_{task}_tp_{method}_s1_*__1_final.pt")
    return sorted(pats)[-1] if pats else None


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
def rollout_recovery(envs, act_fn, num_envs: int, episode_steps: int):
    """每 env 跑一个 episode, 记录 upright 时序 + 终止原因。"""
    obs = envs.reset()
    up_seqs: list[list[float]] = [[] for _ in range(num_envs)]
    done_reason: list[str | None] = [None] * num_envs  # 'truncated'|'fall'
    for _ in range(episode_steps + 5):
        act = act_fn(obs)
        obs, rew, dones, infos = envs.step(act)
        for i in range(num_envs):
            if done_reason[i] is not None:
                continue
            up = infos[i].get("upright")
            if up is not None:
                up_seqs[i].append(float(up))
            if dones[i]:
                trunc = bool(infos[i].get("TimeLimit.truncated", False))
                done_reason[i] = "truncated" if trunc else "fall"
        if all(d is not None for d in done_reason):
            break
    for i in range(num_envs):
        if done_reason[i] is None:
            done_reason[i] = "truncated"
    return up_seqs, done_reason


def analyze_seq(up: list[float]):
    """state machine: 数 near-fall 事件 + 其中 recovery 数 + recovery 用时。"""
    n_fall = 0
    n_rec = 0
    rec_times: list[int] = []
    L = len(up)
    if L == 0:
        return 0, 0, [], 0.0, 0
    state = "stand" if up[0] > U_STAND else "low"
    fall_start = None
    i = 0
    while i < L:
        u = up[i]
        if state == "stand" and u < U_FALL:
            n_fall += 1
            fall_start = i
            state = "fallen"
        elif state == "fallen" and u > U_STAND:
            end = min(i + K, L)
            if end - i >= K and all(up[j] > U_STAND for j in range(i, end)):
                n_rec += 1
                rec_times.append(i - fall_start)
                state = "stand"
        i += 1
    upright_mean = float(np.mean(up))
    return n_fall, n_rec, rec_times, upright_mean, L


def main() -> None:
    ensure_fasttd3_import_path()
    ensure_humanoidbench_import_path()
    from stable_baselines3.common.vec_env import SubprocVecEnv

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for task in TASKS:
        envs = SubprocVecEnv([make_env_fn(f"h1hand-{task}-v0", i, EPISODE_STEPS, ENV_SEED)
                              for i in range(NUM_ENVS)])
        for method in METHODS:
            ck = find_ckpt(task, method)
            if ck is None:
                print(f"[{task} {method}] MISSING ckpt", flush=True)
                continue
            sp = SourcePolicy(
                f"{task}_{method}", ck, device,
                target_action_dim=ACTION_DIM,
                source_obs_dim=OBS_DIM, source_action_dim=ACTION_DIM,
                obs_adapter_spec={"type": "identity", "output_dim": OBS_DIM})
            act_fn = lambda o, _sp=sp: _sp.act(
                torch.as_tensor(o, device=device, dtype=torch.float32)).cpu().numpy()
            t0 = time.time()
            up_seqs, done_reason = rollout_recovery(envs, act_fn, NUM_ENVS, EPISODE_STEPS)
            tot_fall = tot_rec = 0
            all_rec_t: list[int] = []
            up_means, lens = [], []
            for up in up_seqs:
                nf, nr, rt, um, L = analyze_seq(up)
                tot_fall += nf
                tot_rec += nr
                all_rec_t += rt
                up_means.append(um)
                lens.append(L)
            ended_fall = sum(1 for d in done_reason if d == "fall")
            rec_rate = tot_rec / tot_fall if tot_fall > 0 else float("nan")
            rows.append(dict(
                task=task, method=method, ep=NUM_ENVS,
                ep_len=float(np.mean(lens)), upright=float(np.mean(up_means)),
                ended_fall=ended_fall / NUM_ENVS, n_fall=tot_fall,
                rec_rate=rec_rate, rec_time=float(np.mean(all_rec_t)) if all_rec_t else float("nan")))
            print(f"[{task:6s} {method:4s}] ep_len={np.mean(lens):6.1f} "
                  f"upright={np.mean(up_means):.3f} ended_by_fall={ended_fall/NUM_ENVS:4.0%} "
                  f"near-fall={tot_fall:3d} recovered={tot_rec:3d} "
                  f"rate={rec_rate:5.0%} ({time.time()-t0:.0f}s)", flush=True)
        envs.close()

    print("\n" + "=" * 92)
    print("fall-recovery 探针: ended_by_fall%低 + recovery_rate高 + upright高 = 更鲁棒/能恢复")
    print("=" * 92)
    print(f"{'task':6s} {'method':6s} {'ep_len':>7s} {'upright':>8s} "
          f"{'end_fall%':>9s} {'nearfall':>8s} {'rec_rate':>8s} {'rec_time':>8s}")
    for r in rows:
        rr = f"{r['rec_rate']:7.0%}" if not np.isnan(r["rec_rate"]) else "    n/a"
        rt = f"{r['rec_time']:8.1f}" if not np.isnan(r["rec_time"]) else "     n/a"
        print(f"{r['task']:6s} {r['method']:6s} {r['ep_len']:7.1f} {r['upright']:8.3f} "
              f"{r['ended_fall']:9.0%} {r['n_fall']:8d} {rr:>8s} {rt:>8s}")


if __name__ == "__main__":
    main()
