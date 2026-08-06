"""per-state × per-body-group Q-switch 机制探针(door)。

预注册: docs/experiments/per_state_qswitch_probe_v1_prereg_20260729.md (afdc001)
方向依据: docs/direction_per_state_qswitch_20260729.md

问题(H2): 在 target critic 判定"整条源策略替换有害"的状态上,是否仍存在大量
身体组使得局部替换被判为有益?即"整体有害"是否蕴含"处处有害"。

这是离线探针,零训练成本。它检验机制前提,**不能**推断迁移效果——本项目已
多次证明 critic 判断与最终学习效用可以反向。

判据与分位数在预注册中冻结,本脚本只实现,不得事后调整。

用法: CUDA_VISIBLE_DEVICES=0 python scripts/analysis/per_state_qswitch_probe_v1.py
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
from fasttd3_ptf.ptf.mcg import DEFAULT_GROUPS, ModularGating
from fasttd3_ptf.ptf.source_policy import SourcePolicy

ACTION_DIM = 61
TASK = "door"
SOURCES = ("stand", "walk", "run")
SEEDS = (1, 2, 3)
CKPT_STEP = 20000
NUM_ENVS = 16
ROLLOUT_STEPS = 300  # ×16 env = 4800 状态,与 2026-06-11 push 探针同口径
NOISE = 0.1
PROBE_SEED = 7

# --- 多重比较校正(预注册 §4,冻结) ---
# group 侧 max 覆盖 3源×3组=9 个比较,full 侧覆盖 3 个比较。用 Šidák 把两侧的
# family-wise 名义假阳性率对齐,否则 group 仅凭比较次数多就会占优。
ALPHA_FW = 1.0 - 0.95**3                        # ≈ 0.142625
Q_FULL = 0.95                                   # full: 每比较 α=0.05
N_GROUP_COMPARISONS = len(SOURCES) * len(DEFAULT_GROUPS)
Q_GROUP = 1.0 - (1.0 - (1.0 - ALPHA_FW) ** (1.0 / N_GROUP_COMPARISONS))  # ≈ 0.98305

# --- 判据(预注册 §5,冻结) ---
FRAC_SUPPORTED_MIN = 0.30                       # ≈ 2 × α_fw
FRAC_REFUTED_MAX = ALPHA_FW                     # ≈ 0.1426
SEEDS_REQUIRED = 3


@torch.no_grad()
def q_heads_mean(critic, obs_n, actions):
    """两 head 的期望 Q 分别返回(distributional critic 的 support 加权)。"""
    support = critic.q_support
    vs = []
    for head in (critic.qnet1, critic.qnet2):
        probs = torch.softmax(head(obs_n, actions), dim=-1)
        vs.append((probs * support).sum(-1))
    return vs[0], vs[1]


@torch.no_grad()
def full_null_margin(qheads_fn, critic_obs, a_student, src_actions, quantile, generator):
    """full 动作替换的 null margin——与 mcg.null_margins 同构,mask 为全 1。

    null 候选 = batch 内其他状态下的源整动作,刻画 critic 对"与当前状态无关的
    整动作建议"的纯噪声响应。
    """
    batch, num_src = src_actions.shape[0], src_actions.shape[1]
    perm = torch.randperm(batch, generator=generator).to(a_student.device)
    q1_ref, q2_ref = qheads_fn(critic_obs, a_student)
    null_d = []
    for i in range(num_src):
        q1_c, q2_c = qheads_fn(critic_obs, src_actions[perm, i, :])
        null_d.append(torch.minimum(q1_c - q1_ref, q2_c - q2_ref))
    return float(torch.quantile(torch.cat(null_d).float(), quantile))


def load_sources(device):
    """逐个读 door@10k gate 的单源 calibration bank,与该 gate 逐位同源。"""
    out = []
    for name in SOURCES:
        bank = load_yaml(f"configs/source_banks/calibration/h1hand_{TASK}_rbo_{name}.yaml")
        spec = bank["sources"][0]
        assert spec["name"] == name, f"bank {name} 的 sources[0] 是 {spec['name']}"
        out.append(SourcePolicy.from_spec(spec, device=device, target_action_dim=ACTION_DIM))
    return out


def run_seed(seed, envs, teachers, gating, device):
    ckpt = f"models/h1hand-{TASK}-v0__{TASK}_at10k_student_s{seed}__{seed}_{CKPT_STEP}.pt"
    if not os.path.exists(ckpt):
        raise FileNotFoundError(ckpt)
    actor, critic, obs_norm, critic_norm, gstep = load_student(ckpt, device)

    states = collect_states(envs, actor, obs_norm, device, ROLLOUT_STEPS, NOISE, seed=PROBE_SEED)
    obs_t = torch.as_tensor(states, device=device, dtype=torch.float32)
    obs_a, obs_c = obs_norm(obs_t), critic_norm(obs_t)
    a_stu = actor(obs_a)

    def qheads_fn(cobs, acts):
        return q_heads_mean(critic, cobs, acts)

    # [B, S, A] 源动作
    src_actions = torch.stack([tp.act(obs_t) for tp in teachers], dim=1)

    # --- full: Δ_full,i(s) = min_h [Q_h(s,π_i) − Q_h(s,π_stu)] (paired head delta) ---
    q1_ref, q2_ref = qheads_fn(obs_c, a_stu)
    delta_full = torch.empty(obs_t.shape[0], len(teachers), device=device)
    for i in range(len(teachers)):
        q1_c, q2_c = qheads_fn(obs_c, src_actions[:, i, :])
        delta_full[:, i] = torch.minimum(q1_c - q1_ref, q2_c - q2_ref)

    # --- group: Δ_{i,g}(s) (复用 mcg 的 v1.1 口径) ---
    delta_group = gating.deltas(qheads_fn, obs_c, a_stu, src_actions)  # [B, S, G]

    # --- null margins ---
    gen = torch.Generator(device="cpu").manual_seed(PROBE_SEED)
    m_full = full_null_margin(qheads_fn, obs_c, a_stu, src_actions, Q_FULL, gen)
    gen_g = torch.Generator(device="cpu").manual_seed(PROBE_SEED)
    m_group = gating.null_margins(qheads_fn, obs_c, a_stu, src_actions, Q_GROUP, gen_g)  # [G]

    # --- 校正后的显著正比例 ---
    sig_full = (delta_full - m_full).max(dim=1).values                       # [B]
    sig_group = (delta_group - m_group.view(1, 1, -1)).flatten(1).max(dim=1).values  # [B]
    frac_sig_full = float((sig_full > 0).float().mean())
    frac_sig_group = float((sig_group > 0).float().mean())

    # --- 次级观察(不参与裁决) ---
    per_group_frac_sig = {
        g: float(((delta_group[:, :, gi] - m_group[gi]).max(dim=1).values > 0).float().mean())
        for gi, g in enumerate(gating.groups)
    }
    return {
        "seed": seed,
        "ckpt": ckpt,
        "global_step": gstep,
        "num_states": int(obs_t.shape[0]),
        "frac_sig_group": frac_sig_group,
        "frac_sig_full": frac_sig_full,
        "margin_full_q95": m_full,
        "margin_group_q9830": {g: float(m_group[gi]) for gi, g in enumerate(gating.groups)},
        # 次级: Δ_full 均值符号应为负(与 door gate 的 9/9 学习效用负同向)
        "delta_full_mean": {SOURCES[i]: float(delta_full[:, i].mean()) for i in range(len(teachers))},
        "q_student_mean": float(torch.minimum(q1_ref, q2_ref).mean()),
        # 次级: 未校正 sign 版,仅与 2026-06-11 push 探针可比
        "frac_pos_sign_full": float((delta_full.max(dim=1).values > 0).float().mean()),
        "frac_pos_sign_group": float((delta_group.flatten(1).max(dim=1).values > 0).float().mean()),
        "per_group_frac_sig": per_group_frac_sig,
    }


def adjudicate(rows):
    """预注册 §5 的判据,冻结。"""
    fracs = [r["frac_sig_group"] for r in rows]
    n_dir = sum(1 for r in rows if r["frac_sig_group"] > r["frac_sig_full"])
    mean_frac = float(np.mean(fracs))
    c1 = all(f >= FRAC_SUPPORTED_MIN for f in fracs)
    c2 = n_dir == SEEDS_REQUIRED
    if c1 and c2:
        verdict = "PROBE_SUPPORTED"
    elif any(f < FRAC_REFUTED_MAX for f in fracs) or n_dir <= 1:
        verdict = "PROBE_REFUTED"
    else:
        verdict = "PROBE_WEAK"
    return {
        "verdict": verdict,
        "frac_sig_group_per_seed": fracs,
        "frac_sig_group_mean": mean_frac,
        "frac_sig_full_per_seed": [r["frac_sig_full"] for r in rows],
        "direction_consistency": f"{n_dir}/{len(rows)}",
        "criterion_1_all_ge_0.30": c1,
        "criterion_2_direction_3of3": c2,
    }


def main():
    ensure_fasttd3_import_path()
    ensure_humanoidbench_import_path()
    from stable_baselines3.common.vec_env import SubprocVecEnv

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teachers = load_sources(device)
    gating = ModularGating(action_dim=ACTION_DIM, groups=DEFAULT_GROUPS, device=device)
    envs = SubprocVecEnv(
        [make_env_fn(f"h1hand-{TASK}-v0", i, 1000, seed=PROBE_SEED) for i in range(NUM_ENVS)]
    )

    print("=" * 96)
    print(f"per-state × per-body-group Q-switch 探针 | target={TASK} | 源={SOURCES}")
    print(f"Šidák: α_fw={ALPHA_FW:.6f} | full q={Q_FULL:.4f} | group q={Q_GROUP:.5f} "
          f"({N_GROUP_COMPARISONS} 比较)")
    print("=" * 96)

    rows = []
    try:
        for seed in SEEDS:
            r = run_seed(seed, envs, teachers, gating, device)
            rows.append(r)
            print(f"[s{seed}] step={r['global_step']} states={r['num_states']} "
                  f"Q_stu={r['q_student_mean']:.1f}")
            print(f"      frac_sig  group={r['frac_sig_group']:.4f}  full={r['frac_sig_full']:.4f}"
                  f"   (sign 版 group={r['frac_pos_sign_group']:.4f} full={r['frac_pos_sign_full']:.4f})")
            print(f"      Δ_full 均值 {r['delta_full_mean']}")
            print(f"      per-group frac_sig {r['per_group_frac_sig']}", flush=True)
    finally:
        envs.close()

    res = adjudicate(rows)
    print("=" * 96)
    print(f"VERDICT: {res['verdict']}")
    print(f"  frac_sig_group 每 seed: {[f'{f:.4f}' for f in res['frac_sig_group_per_seed']]}")
    print(f"  frac_sig_full  每 seed: {[f'{f:.4f}' for f in res['frac_sig_full_per_seed']]}")
    print(f"  判据1(全部≥{FRAC_SUPPORTED_MIN}): {res['criterion_1_all_ge_0.30']}")
    print(f"  判据2(方向 3/3): {res['criterion_2_direction_3of3']}  "
          f"实际 {res['direction_consistency']}")
    print("=" * 96)

    out_dir = Path("docs/data/per_state_qswitch_probe_v1")
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "prereg": "docs/experiments/per_state_qswitch_probe_v1_prereg_20260729.md",
        "prereg_commit": "afdc001",
        "config": {
            "task": TASK, "sources": list(SOURCES), "seeds": list(SEEDS),
            "ckpt_step": CKPT_STEP, "num_envs": NUM_ENVS, "rollout_steps": ROLLOUT_STEPS,
            "noise": NOISE, "probe_seed": PROBE_SEED, "groups": list(DEFAULT_GROUPS),
            "alpha_fw": ALPHA_FW, "q_full": Q_FULL, "q_group": Q_GROUP,
        },
        "per_seed": rows,
        "adjudication": res,
    }
    path = out_dir / "per_state_qswitch_probe_v1_results.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"written {path}")


if __name__ == "__main__":
    main()
