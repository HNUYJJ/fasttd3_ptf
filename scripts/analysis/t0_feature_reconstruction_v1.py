"""t=0 注入前特征零训练重建(inventory v1.1 用,只读)。

回答 PI 的修订点 1:EQD30K 的六个 source-level 标签(hurdle/crawl × stand/walk/run)
其 t=0 决策时点特征是否可重建、实际取值如何。

做法:
- 按原训练 seed 重建 t=0 student(torch.manual_seed(seed) → Actor(**同 kwargs));
  vendored FastTD3 未改动,且 manual_seed 与 actor 构建之间无新增 RNG 消耗,
  故 t=0 actor 是 seed 的确定性函数(脚本内验证重建确定性);
- obs normalizer 在 t=0 为恒等(mean=0,std=1);
- 用与在线机制完全相同的 matched-state 面板协议采 32 个 student occupancy 状态;
- 每个状态分别 fork student 与 stand/walk/run 各 25 步,用同一 target-evidence
  contract 计算 target return / achievement progress / feasibility / hard constraints;
- 附加描述子:source-vs-student 动作距离、状态覆盖(最近邻距离)、student 动作范数;
- critic/TD 统计在 t=0 由随机初始化 critic 给出,记录但显式标记 degenerate,
  不得单独作为指标。

不训练、不写训练状态、不改训练代码。
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from fasttd3_ptf.config import load_yaml  # noqa: E402
from fasttd3_ptf.official_fasttd3_ptf import (  # noqa: E402
    ensure_fasttd3_import_path,
    ensure_humanoidbench_import_path,
)
from fasttd3_ptf.official_fasttd3_ptf.target_evidence import TargetEvidenceContract  # noqa: E402
from fasttd3_ptf.official_fasttd3_ptf.target_evidence_probe import (  # noqa: E402
    _collect_snapshots,
    _restore,
    bootstrap_interval,
    make_target_env,
)
from fasttd3_ptf.ptf.source_policy import SourcePolicy  # noqa: E402

ensure_fasttd3_import_path()
from fast_td3 import Actor, Critic  # noqa: E402

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BANK_YAML = REPO / "configs/source_banks/calibration/h1hand_loco3_rbo_equal_h25.yaml"
CONTRACTS = {
    "h1hand-hurdle-v0": REPO / "configs/target_evidence/humanoidbench_hurdle_v1.yaml",
    "h1hand-crawl-v0": REPO / "configs/target_evidence/humanoidbench_crawl_v1.yaml",
}
OUT = REPO / "docs/data/transfer_effect_label_inventory_t0_features_20260727.json"

# 与在线机制冻结面板一致
RESET_SEEDS = [11001, 23001, 37001, 53001]
AGES = [0, 5, 10, 25, 50, 100, 150, 200]
HORIZON = 25
SOURCES = ["stand", "walk", "run"]
# EQD30K 标签实际存在的 (task, seed)
UNITS = [("h1hand-hurdle-v0", 1), ("h1hand-hurdle-v0", 2), ("h1hand-hurdle-v0", 3),
         ("h1hand-crawl-v0", 1)]
# HumanoidBench h1hand 冻结超参(与训练一致)
ACTOR_KW = dict(n_act=61, num_envs=128, init_scale=0.01, hidden_dim=512,
                std_min=0.001, std_max=0.4)
CRITIC_KW = dict(n_act=61, num_atoms=101, hidden_dim=1024)
VMIN_VMAX = {"h1hand-hurdle-v0": (-250.0, 250.0), "h1hand-crawl-v0": (-250.0, 250.0)}
N_OBS = 151


class _Protocol:
    def __init__(self):
        self.reset_seeds = list(RESET_SEEDS)
        self.occupancy_ages = sorted(AGES)
        self.horizon = HORIZON


def build_t0_learner(seed: int, env_name: str):
    """复现 train_ptf 的 t=0 构建顺序:manual_seed → actor → actor_detach → critic ×2。"""
    torch.manual_seed(seed)
    actor = Actor(n_obs=N_OBS, device=DEVICE, **ACTOR_KW)
    _actor_detach = Actor(n_obs=N_OBS, device=DEVICE, **ACTOR_KW)  # 顺序占位,消耗同样 RNG
    v_min, v_max = VMIN_VMAX[env_name]
    qnet = Critic(n_obs=N_OBS, v_min=v_min, v_max=v_max, device=DEVICE, **CRITIC_KW)
    actor.eval()
    qnet.eval()
    return actor, qnet


def rollout_branch(env, snapshot, act_fn, contract, horizon):
    """单分支 fork:返回 contract 聚合量 + 轨迹(状态/动作)供描述子使用。"""
    obs = _restore(env, snapshot)
    evidence = contract.new_accumulator(env)
    total_reward = 0.0
    states, actions = [], []
    terminated = truncated = False
    for _ in range(horizon):
        action = np.asarray(act_fn(obs), dtype=np.float32)
        states.append(np.asarray(obs, dtype=np.float32))
        actions.append(action)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        evidence.observe(info)
        if terminated or truncated:
            break
    result = evidence.finish()
    return dict(ret=total_reward, progress=result["progress"],
                feasibility=result["feasibility"], steps=len(states),
                terminated=bool(terminated),
                states=np.stack(states), actions=np.stack(actions))


def nn_coverage(src_states: np.ndarray, stu_states: np.ndarray) -> float:
    """source 轨迹状态到 student 状态云的平均最近邻 L2(状态覆盖/novelty 描述子)。"""
    d = np.linalg.norm(src_states[:, None, :] - stu_states[None, :, :], axis=-1)
    return float(d.min(axis=1).mean())


def run_unit(env_name: str, seed: int) -> dict:
    contract = TargetEvidenceContract.from_yaml(str(CONTRACTS[env_name]))
    actor, qnet = build_t0_learner(seed, env_name)

    # t=0 normalizer 为恒等(EmpiricalNormalization 初值 mean=0,std=1,eps=1e-2)
    @torch.no_grad()
    def student_act(obs_np):
        o = torch.as_tensor(obs_np, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        return actor(o / (1.0 + 1e-2)).squeeze(0).float().cpu().numpy()

    bank = load_yaml(str(BANK_YAML))
    src_policies = {}
    for name in SOURCES:
        spec = next(s for s in bank["sources"] if s["name"] == name)
        src_policies[name] = SourcePolicy.from_spec(spec, device=DEVICE, target_action_dim=61)

    def make_src_act(pol):
        @torch.no_grad()
        def _act(obs_np):
            o = torch.as_tensor(obs_np, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            return pol.act(o).squeeze(0).float().cpu().numpy()
        return _act

    src_acts = {n: make_src_act(p) for n, p in src_policies.items()}

    env = make_target_env(env_name)
    proto = _Protocol()
    snapshots = _collect_snapshots(env, student_act, proto)

    per_panel = {n: dict(dR=[], dP=[], dFeas={}, act_dist=[], coverage=[],
                         src_steps=[], stu_steps=[]) for n in SOURCES}
    stu_action_norms = []
    try:
        for snap in snapshots:
            stu = rollout_branch(env, snap, student_act, contract, HORIZON)
            stu_action_norms.append(float(np.linalg.norm(stu["actions"], axis=-1).mean()))
            for name in SOURCES:
                src = rollout_branch(env, snap, src_acts[name], contract, HORIZON)
                rec = per_panel[name]
                rec["dR"].append(src["ret"] - stu["ret"])
                rec["dP"].append(src["progress"] - stu["progress"])
                for fname, fval in src["feasibility"].items():
                    rec["dFeas"].setdefault(fname, []).append(
                        float(fval) - float(stu["feasibility"][fname]))
                k = min(len(src["actions"]), len(stu["actions"]))
                rec["act_dist"].append(float(
                    np.linalg.norm(src["actions"][:k] - stu["actions"][:k], axis=-1).mean()))
                rec["coverage"].append(nn_coverage(src["states"], stu["states"]))
                rec["src_steps"].append(src["steps"])
                rec["stu_steps"].append(stu["steps"])
    finally:
        env.close()

    # t=0 critic 统计:随机初始化 → degenerate,记录并标记
    with torch.no_grad():
        probe_obs = torch.as_tensor(
            np.stack([_ for _ in [snapshots[0].state[:N_OBS]]]), dtype=torch.float32,
            device=DEVICE)
        pa = actor(probe_obs / (1.0 + 1e-2))
        q1, q2 = qnet(probe_obs, pa)
        v1 = qnet.get_value(torch.softmax(q1, dim=1))
        v2 = qnet.get_value(torch.softmax(q2, dim=1))
        critic_disagreement_t0 = float((v1 - v2).abs().mean())

    out = dict(env_name=env_name, seed=seed, stage_t=0,
               panel_size=len(snapshots),
               student_action_l2_mean=float(np.mean(stu_action_norms)),
               critic_stats_t0=dict(twin_disagreement=critic_disagreement_t0,
                                    degenerate=True,
                                    reason="critic is randomly initialised at t=0; recorded for completeness, must not be used as a standalone admission signal"),
               sources={})
    for name in SOURCES:
        rec = per_panel[name]
        dR, dP = np.asarray(rec["dR"]), np.asarray(rec["dP"])
        out["sources"][name] = dict(
            dR_mean=float(dR.mean()),
            dR_lcb90=float(bootstrap_interval(dR)["lcb90"]),
            dP_mean=float(dP.mean()),
            dP_lcb90=float(bootstrap_interval(dP)["lcb90"]),
            dFeasibility_mean={k: float(np.mean(v)) for k, v in rec["dFeas"].items()},
            action_distance_mean=float(np.mean(rec["act_dist"])),
            state_coverage_nn_mean=float(np.mean(rec["coverage"])),
            source_survival_steps_mean=float(np.mean(rec["src_steps"])),
            student_survival_steps_mean=float(np.mean(rec["stu_steps"])),
        )
    return out


def verify_reconstruction_determinism() -> dict:
    a1, _ = build_t0_learner(1, "h1hand-hurdle-v0")
    a2, _ = build_t0_learner(1, "h1hand-hurdle-v0")
    same = all(torch.equal(p1, p2) for p1, p2 in zip(a1.state_dict().values(),
                                                     a2.state_dict().values()))
    b1, _ = build_t0_learner(2, "h1hand-hurdle-v0")
    differs = any(not torch.equal(p1, p2) for p1, p2 in zip(a1.state_dict().values(),
                                                            b1.state_dict().values()))
    return dict(same_seed_bitwise_identical=bool(same),
                different_seed_differs=bool(differs))


def main():
    ensure_humanoidbench_import_path()
    report = dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        purpose="t=0 pre-injection feature reconstruction for the six EQD30K source-level labels",
        reconstruction_basis=dict(
            method="torch.manual_seed(train seed) then rebuild Actor/Critic in the training construction order",
            audit="vendored FastTD3 unmodified; no new RNG consumption between manual_seed and actor construction in the working tree",
            determinism=verify_reconstruction_determinism(),
            caveat="not verified against a stored t=0 checkpoint (none exists); validity rests on construction-order equivalence",
        ),
        protocol=dict(reset_seeds=RESET_SEEDS, occupancy_ages=AGES, horizon=HORIZON,
                      panel_size=len(RESET_SEEDS) * len(AGES),
                      normalizer="identity at t=0 (EmpiricalNormalization initial mean=0 std=1)"),
        units=[],
    )
    print(json.dumps(report["reconstruction_basis"]["determinism"]), flush=True)
    for env_name, seed in UNITS:
        print(f"=== {env_name} seed={seed} ===", flush=True)
        unit = run_unit(env_name, seed)
        report["units"].append(unit)
        print(json.dumps(unit, indent=1), flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print(f"saved: {OUT}")


if __name__ == "__main__":
    main()
