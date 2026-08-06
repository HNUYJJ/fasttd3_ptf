"""T^critic 离线 logging(ChatGPT 第二优先级, 2026-07-02): 不控制训练, 只验证判别力。

问题: 半交互 transferability T^critic(t) = E_{s~B_t}[minQ(s,π_i(s)) − minQ(s,π_stu(s))]
能否(以及从哪个训练时刻起)正确判别源的可迁移性?
  crawl 期望: 全源 T^critic < 0(critic 知道站立教师不如学生) → 支持判 abstain
  pole  期望: walk 的 T^critic ≥ 0(教师有用) → 支持留 transfer
对照: T^online(arm value EMA)在 crawl 上 ~14k 步才分化; T^critic 若更早分辨则
可提前接管, 若早期乱跳则按裁定推迟到 warmup 后半段(c(t) ramp t0=0.3·warmup)。

稳健化(按 ChatGPT 裁定):
  Δ_i(s) = min_j Qj(s,π_i(s)) − min_j Qj(s,π_stu(s))
  T_mean = mean(Δ); T_robust = 0.5·mean + 0.5·q25(Δ)   (不用 q10, 已证过保守)
  U_i(s) = |Q1(s,π_i(s)) − Q2(s,π_i(s))|; penalty = q75(U)
  T_final = T_robust − λ_u·q75(U), λ_u=0.5

做法: 对 task ∈ {crawl, pole} 的 onlineb run 中间 ckpt(10k..30k+final):
  加载学生 actor/critic/normalizer → 学生 rollout 收集状态(近似 B_t 的
  on-policy 部分) → 对 bank 每个源算 Δ/T → 打印时序表。

用法: CUDA_VISIBLE_DEVICES=5 python scripts/analyze_tcritic_offline.py
"""
from __future__ import annotations

import glob
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from scripts.probe_lib import (
    _FrozenNorm,  # noqa: F401 (load_student 内部用)
    collect_states,
    load_student,
    make_env_fn,
)
from fasttd3_ptf.config import load_yaml
from fasttd3_ptf.official_fasttd3_ptf.paths import (
    ensure_fasttd3_import_path,
    ensure_humanoidbench_import_path,
)
from fasttd3_ptf.ptf.source_policy import SourcePolicy

ACTION_DIM = 61
ONLINEB_STAMP = "20260702T025544Z"
CKPT_STEPS = [10000, 15000, 20000, 25000, 30000, "final"]
TASKS = ["crawl", "pole"]
LAMBDA_U = 0.5
NUM_ENVS = 16
ROLLOUT_STEPS = 150  # ×16 env = 2400 状态
NOISE = 0.1


@torch.no_grad()
def q_heads_mean(critic, obs_n, actions):
    """两 head 的期望 Q 分别返回(外部自行 min / 差)。"""
    support = critic.q_support
    vs = []
    for head in (critic.qnet1, critic.qnet2):
        logits = head(obs_n, actions)
        probs = torch.softmax(logits, dim=-1)
        vs.append((probs * support).sum(-1))
    return vs[0], vs[1]


def find_ckpt(task, step):
    tag = "final" if step == "final" else str(step)
    pats = glob.glob(
        f"models/h1hand-{task}-v0__h1hand_{task}_tp_onlineb_s1_{ONLINEB_STAMP}__1_{tag}.pt")
    return sorted(pats)[-1] if pats else None


def main():
    ensure_fasttd3_import_path()
    ensure_humanoidbench_import_path()
    from stable_baselines3.common.vec_env import SubprocVecEnv

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for task in TASKS:
        bank = load_yaml(f"configs/source_banks/h1hand_loco_wfix_{task}.yaml")
        teachers = [
            SourcePolicy.from_spec(sp, device=device, target_action_dim=ACTION_DIM)
            for sp in bank["sources"]
        ]
        envs = SubprocVecEnv(
            [make_env_fn(f"h1hand-{task}-v0", i, 1000, seed=7) for i in range(NUM_ENVS)]
        )
        print("=" * 100)
        print(f"[{task}] T^critic 时序(onlineb s1 {ONLINEB_STAMP}; "
              f"负值=源不如学生; 真实结论: "
              f"{'全源应为负(abstain)' if task == 'crawl' else 'walk 应≥0(transfer)'})")
        print("=" * 100)
        hdr = f"{'step':>7s} {'Qstu':>8s}"
        for tp in teachers:
            hdr += f" | {tp.name:>7s}: {'Tmean':>7s} {'Trob':>7s} {'Tfin':>7s} {'q75U':>6s}"
        print(hdr)
        for step in CKPT_STEPS:
            ck = find_ckpt(task, step)
            if ck is None:
                print(f"{str(step):>7s} MISSING")
                continue
            actor, critic, obs_norm, critic_norm, gstep = load_student(ck, device)
            states = collect_states(envs, actor, obs_norm, device, ROLLOUT_STEPS, NOISE, seed=7)
            obs_t = torch.as_tensor(states, device=device, dtype=torch.float32)
            obs_a, obs_c = obs_norm(obs_t), critic_norm(obs_t)
            a_stu = actor(obs_a)
            q1s, q2s = q_heads_mean(critic, obs_c, a_stu)
            q_stu = torch.minimum(q1s, q2s)
            row = f"{gstep:7d} {float(q_stu.mean()):8.1f}"
            for tp in teachers:
                a_i = tp.act(obs_t)
                q1, q2 = q_heads_mean(critic, obs_c, a_i)
                delta = (torch.minimum(q1, q2) - q_stu).cpu().numpy()
                u = (q1 - q2).abs().cpu().numpy()
                t_mean = float(delta.mean())
                t_rob = 0.5 * t_mean + 0.5 * float(np.percentile(delta, 25))
                q75u = float(np.percentile(u, 75))
                t_fin = t_rob - LAMBDA_U * q75u
                row += f" | {tp.name:>7s}: {t_mean:7.1f} {t_rob:7.1f} {t_fin:7.1f} {q75u:6.1f}"
            print(row, flush=True)
        envs.close()

    print("\n判读要点:")
    print("  1. crawl: T^critic 从哪个 step 起全源为负? 早于 T^online 的 ~14k 则可提前接管")
    print("  2. pole: walk 的 T 是否稳定 ≥0 / 明显高于 stand? 若被误判为负则 critic 信号不可用")
    print("  3. Tfin(带 head-disagreement penalty) vs Tmean 的排序是否一致(penalty 是否引入扭曲)")


if __name__ == "__main__":
    main()
