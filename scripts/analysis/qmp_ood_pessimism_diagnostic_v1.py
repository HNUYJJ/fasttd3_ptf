"""为什么 QMP 的 Q-switch 几乎从不选择冻结的跨任务源?——OOD 悲观性诊断。

**探索性分析,不参与 run card §6 的任何判据。** 事后提出,用于解释观察到的
`qmp/source_share ≈ 0` 现象,不构成对该现象的预注册检验。

假设 H_ood:
    QMP 原文的 mixture 是**同时训练**的多任务策略,其动作落在各自 replay 分布内;
    本项目的源是**冻结的跨任务**策略,其动作对 target critic 是分布外的。
    FastTD3 的 CDQ 取 min(Q1,Q2),使分布外动作被系统性低估,
    于是 argmax 几乎恒选 student——与源是否真的有用无关。

可证伪的预测(若 H_ood 成立):
    P1. 源动作上的双 head 分歧 |Q1−Q2| 显著大于 student 动作上的分歧
        (head 分歧是认识不确定性的常用代理);
    P2. 该现象在 slide(源确实有益,walk U=+56.95)与 door(源确实有害)上**同样**出现
        ——即它由动作的分布外性驱动,而非由源的真实效用驱动。

    P2 是关键:若 slide 上源动作并不显示更大的 head 分歧、且 Q 值不落后,
    则 H_ood 被削弱,QMP 不选源必须另寻解释。

打分口径与训练端 QMP 完全一致:score_i = min_h Q_h(s, π_i(s))。
注意这**不是** MCG 的 paired delta min_h[Q_h(cand)−Q_h(stu)]。

用法: CUDA_VISIBLE_DEVICES=0 python scripts/analysis/qmp_ood_pessimism_diagnostic_v1.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch

from scripts.probe_lib import _FrozenNorm, collect_states, load_student, make_env_fn  # noqa: F401
from fasttd3_ptf.config import load_yaml
from fasttd3_ptf.official_fasttd3_ptf.paths import (
    ensure_fasttd3_import_path,
    ensure_humanoidbench_import_path,
)
from fasttd3_ptf.ptf.source_policy import SourcePolicy

ACTION_DIM = 61
TASKS = ("door", "slide")
SEEDS = (1, 2, 3)
NUM_ENVS = 16
ROLLOUT_STEPS = 150  # ×16 = 2400 状态
NOISE = 0.1
PROBE_SEED = 7


@torch.no_grad()
def q_heads(critic, obs_n, actions):
    support = critic.q_support
    vs = []
    for head in (critic.qnet1, critic.qnet2):
        probs = torch.softmax(head(obs_n, actions), dim=-1)
        vs.append((probs * support).sum(-1))
    return vs[0], vs[1]


def run_task(task, device):
    bank = load_yaml(f"configs/source_banks/calibration/h1hand_{task}_qmp_loco3.yaml")
    teachers = [
        SourcePolicy.from_spec(sp, device=device, target_action_dim=ACTION_DIM)
        for sp in bank["sources"]
    ]
    ensure_humanoidbench_import_path()
    from stable_baselines3.common.vec_env import SubprocVecEnv

    envs = SubprocVecEnv(
        [make_env_fn(f"h1hand-{task}-v0", i, 1000, seed=PROBE_SEED) for i in range(NUM_ENVS)]
    )
    rows = []
    try:
        for seed in SEEDS:
            ck = f"models/h1hand-{task}-v0__qmp_{task}_s{seed}__{seed}_20000.pt"
            actor, critic, obs_norm, critic_norm, gstep = load_student(ck, device)
            states = collect_states(envs, actor, obs_norm, device, ROLLOUT_STEPS, NOISE, seed=PROBE_SEED)
            obs_t = torch.as_tensor(states, device=device, dtype=torch.float32)
            obs_a, obs_c = obs_norm(obs_t), critic_norm(obs_t)
            a_stu = actor(obs_a)

            q1s, q2s = q_heads(critic, obs_c, a_stu)
            score_stu = torch.minimum(q1s, q2s)
            disag_stu = (q1s - q2s).abs()

            per_src = {}
            for tp in teachers:
                a_i = tp.act(obs_t)
                q1, q2 = q_heads(critic, obs_c, a_i)
                score_i = torch.minimum(q1, q2)
                disag_i = (q1 - q2).abs()
                per_src[tp.name] = {
                    "score_mean": float(score_i.mean()),
                    "score_minus_student_mean": float((score_i - score_stu).mean()),
                    "win_rate_vs_student": float((score_i > score_stu).float().mean()),
                    "head_disagreement_mean": float(disag_i.mean()),
                    # 动作与 student 的 L2 距离:分布外程度的直接几何代理
                    "action_l2_to_student": float((a_i - a_stu).norm(dim=1).mean()),
                }
            rows.append({
                "task": task, "seed": seed, "global_step": gstep,
                "num_states": int(obs_t.shape[0]),
                "student": {
                    "score_mean": float(score_stu.mean()),
                    "head_disagreement_mean": float(disag_stu.mean()),
                },
                "sources": per_src,
            })
            print(f"[{task} s{seed}] student score={float(score_stu.mean()):8.3f} "
                  f"head_disag={float(disag_stu.mean()):.4f}")
            for n, v in per_src.items():
                print(f"    {n:6s} score={v['score_mean']:8.3f} "
                      f"(Δ={v['score_minus_student_mean']:+7.3f}) "
                      f"win={v['win_rate_vs_student']:.4f} "
                      f"head_disag={v['head_disagreement_mean']:.4f} "
                      f"(×{v['head_disagreement_mean']/max(disag_stu.mean().item(),1e-9):.2f}) "
                      f"a_l2={v['action_l2_to_student']:.3f}", flush=True)
    finally:
        envs.close()
    return rows


def main():
    ensure_fasttd3_import_path()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_rows = []
    for task in TASKS:
        print("=" * 92)
        print(f"[{task}] QMP 打分口径 score = min_h Q_h(s, π(s)) —— 与训练端一致")
        print("=" * 92)
        all_rows.extend(run_task(task, device))

    # P1/P2 汇总
    print("\n" + "=" * 92)
    print("H_ood 预测检验(探索性,不参与裁决)")
    print("=" * 92)
    summary = {}
    for task in TASKS:
        rs = [r for r in all_rows if r["task"] == task]
        ratios, wins = [], []
        for r in rs:
            ds = r["student"]["head_disagreement_mean"]
            for v in r["sources"].values():
                ratios.append(v["head_disagreement_mean"] / max(ds, 1e-9))
                wins.append(v["win_rate_vs_student"])
        summary[task] = {
            "head_disagreement_ratio_mean": float(np.mean(ratios)),
            "source_win_rate_mean": float(np.mean(wins)),
        }
        print(f"  {task:6s} 源/学生 head 分歧比 = {np.mean(ratios):.2f}× | "
              f"源胜过学生的状态比例 = {np.mean(wins):.4f}")
    print("\n  P1(源动作 head 分歧更大): "
          f"{'支持' if all(v['head_disagreement_ratio_mean'] > 1.0 for v in summary.values()) else '不支持'}")
    print("  P2(door 与 slide 同样出现,与源真实效用无关): "
          "看两行是否都表现出低 win_rate —— slide 的 walk 源真实效用为 +56.95")

    out = Path("docs/data/qmp_fidelity_v1/ood_pessimism_diagnostic.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"note": "探索性分析,不参与 run card §6 判据",
                               "per_seed": all_rows, "summary": summary},
                              indent=2, ensure_ascii=False))
    print(f"\nwritten {out}")


if __name__ == "__main__":
    main()
