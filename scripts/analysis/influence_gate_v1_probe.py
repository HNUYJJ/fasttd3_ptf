"""Update-space influence proxy 离线双向 gate(v1,预注册,只读探针)。

裁决(PI 2026-07-27)授权范围内的一次性可行性 gate:
- 主分数 I_critic = L_val(U(θ; 50%stu+50%src)) − L_val(U(θ; 100%stu));
  U = 一个完整 FastTD3 outer update unit(2 次 critic + 1 次 actor + 每次
  critic 后 soft target τ,真实 AdamW/scheduler 状态,无虚拟学习率);
- 两臂配对:共享 student batch、替换位置与全部随机数,唯一差异是被替换
  半批的内容(student 原行 vs source 行);
- 冻结标签:harmful = crawl_s3_run;helpful = hurdle_s1_walk / s2_run / s3_walk
  (10k–20k 段内 paired nAUC 证据,操作性校准标签,非无保留 ground truth);
- 三级裁决:ABSOLUTE_PASS / RANKING_ONLY / FAIL(FAIL 即重新封存该家族);
- 区间为"给定 10k anchor 的条件 batch 不确定性",非跨 seed 置信区间;
- 辅助量(梯度内积/cos、actor 目标变化、critic disagreement 比)仅解释,
  不参与裁决;揭盲后不得调整采样量或 gate。

所有更新在参数副本上进行;不写任何训练状态、checkpoint 或 replay。
"""
from __future__ import annotations

import copy
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from fasttd3_ptf.official_fasttd3_ptf import (  # noqa: E402
    ensure_fasttd3_import_path,
    ensure_humanoidbench_import_path,
)
from fasttd3_ptf.official_fasttd3_ptf.anchor_io import load_anchor_core  # noqa: E402
from fasttd3_ptf.official_fasttd3_ptf.ptf_replay import PTFReplayWrapper  # noqa: E402
from fasttd3_ptf.official_fasttd3_ptf.target_evidence_probe import (  # noqa: E402
    _capture,
    _collect_snapshots,
    _restore,
    make_target_env,
)
from fasttd3_ptf.ptf.source_policy import SourcePolicy  # noqa: E402
from fasttd3_ptf.config import load_yaml  # noqa: E402

ensure_fasttd3_import_path()
from fast_td3 import Actor, Critic  # noqa: E402
from fast_td3_utils import EmpiricalNormalization, SimpleReplayBuffer  # noqa: E402

# ---- 预注册常量(揭盲后不得改动)----
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
ANCHOR_ROOT = REPO / "artifacts/influence_gate_v1/anchors"
BANK_YAML = REPO / "configs/source_banks/calibration/h1hand_loco3_rbo_equal_h25.yaml"
OUT_DIR = REPO / "docs/data/influence_gate_v1"
RESET_SEEDS = [11001, 23001, 37001, 53001] + list(range(90001, 90029))  # 32
AGES = [0, 5, 10, 25, 50, 100, 150, 200]  # 8 → 256 states
HORIZON = 25
N_REPEATS = 64
VAL_SLOTS = 64          # 每 env 64 个 slot 整列作 held-out val(8192 条)
BOOT_SAMPLES = 5000
BOOT_SEED = 20260727
CONF = 0.90
BASE_SEED = 727001

CELLS = [
    dict(name="crawl_s3_run", env="h1hand-crawl-v0", anchor="crawl_s3", source="run", role="harmful"),
    dict(name="hurdle_s1_walk", env="h1hand-hurdle-v0", anchor="hurdle_s1", source="walk", role="helpful"),
    dict(name="hurdle_s2_run", env="h1hand-hurdle-v0", anchor="hurdle_s2", source="run", role="helpful"),
    dict(name="hurdle_s3_walk", env="h1hand-hurdle-v0", anchor="hurdle_s3", source="walk", role="helpful"),
]


class _Protocol:
    """最小 protocol 适配 _collect_snapshots 的接口。"""

    def __init__(self, reset_seeds, occupancy_ages):
        self.reset_seeds = list(reset_seeds)
        self.occupancy_ages = sorted(occupancy_ages)


def load_learner(anchor_dir: Path):
    """按 train_ptf 的构建方式重建 learner 并从 anchor 白名单恢复。"""
    pre = torch.load(anchor_dir / "learner.pt", map_location="cpu", weights_only=False)
    args = pre["configuration"]["args"]
    n_obs = int(torch.as_tensor(pre["modules"]["obs_normalizer"]["_mean"]).shape[-1])
    n_act = int(pre["modules"]["actor"]["fc_mu.0.weight"].shape[0])
    assert args["reward_normalization"] is False and args["use_cdq"] is True
    assert int(args["num_updates"]) == 2 and float(args["gamma"]) == 0.99
    actor = Actor(n_obs=n_obs, n_act=n_act, num_envs=int(args["num_envs"]), device=DEVICE,
                  init_scale=float(args["init_scale"]), hidden_dim=int(args["actor_hidden_dim"]),
                  std_min=float(args["std_min"]), std_max=float(args["std_max"]))
    ckw = dict(n_obs=n_obs, n_act=n_act, num_atoms=int(args["num_atoms"]),
               v_min=float(args["v_min"]), v_max=float(args["v_max"]),
               hidden_dim=int(args["critic_hidden_dim"]), device=DEVICE)
    qnet = Critic(**ckw)
    qnet_target = Critic(**ckw)
    obs_normalizer = EmpiricalNormalization(shape=n_obs, device=DEVICE)
    critic_obs_normalizer = EmpiricalNormalization(shape=n_obs, device=DEVICE)
    q_optimizer = optim.AdamW(list(qnet.parameters()), lr=torch.tensor(float(args["critic_learning_rate"]), device=DEVICE), weight_decay=float(args["weight_decay"]))
    actor_optimizer = optim.AdamW(list(actor.parameters()), lr=torch.tensor(float(args["actor_learning_rate"]), device=DEVICE), weight_decay=float(args["weight_decay"]))
    q_scheduler = optim.lr_scheduler.CosineAnnealingLR(q_optimizer, T_max=int(args["total_timesteps"]), eta_min=torch.tensor(float(args["critic_learning_rate_end"]), device=DEVICE))
    actor_scheduler = optim.lr_scheduler.CosineAnnealingLR(actor_optimizer, T_max=int(args["total_timesteps"]), eta_min=torch.tensor(float(args["actor_learning_rate_end"]), device=DEVICE))
    scaler = torch.amp.GradScaler(enabled=False)
    base_rb = SimpleReplayBuffer(n_env=int(args["num_envs"]), buffer_size=int(args["buffer_size"]), n_obs=n_obs, n_act=n_act,
                                 n_critic_obs=n_obs, asymmetric_obs=False, playground_mode=False,
                                 n_steps=1, gamma=float(args["gamma"]), device=DEVICE)
    rb = PTFReplayWrapper(base_rb)
    core = load_anchor_core(
        anchor_dir,
        modules={
            "actor": actor,
            "critic": qnet,
            "critic_target": qnet_target,
            "obs_normalizer": obs_normalizer,
            "critic_obs_normalizer": critic_obs_normalizer,
        },
        optimizers={"actor": actor_optimizer, "critic": q_optimizer},
        schedulers={"actor": actor_scheduler, "critic": q_scheduler},
        scaler=scaler,
        replay=rb,
        map_location=DEVICE,
    )
    assert int(core["completed_vector_steps"]) == 10000, core["completed_vector_steps"]
    actor.eval()
    qnet.train()
    return dict(actor=actor, qnet=qnet, qnet_target=qnet_target,
                obs_norm=obs_normalizer, rb=rb,
                q_opt=q_optimizer, actor_opt=actor_optimizer, core=core)


@torch.no_grad()
def collect_source_transitions(cell, learner):
    """matched-state 面板(_collect_snapshots 同款)+ source 25 步锁存执行。"""
    env = make_target_env(cell["env"])
    actor, obs_norm = learner["actor"], learner["obs_norm"]

    def student_act(obs_np):
        o = torch.as_tensor(obs_np, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        return actor(obs_norm(o, update=False)).squeeze(0).float().cpu().numpy()

    bank = load_yaml(str(BANK_YAML))
    spec = next(s for s in bank["sources"] if s["name"] == cell["source"])
    source = SourcePolicy.from_spec(spec, device=DEVICE, target_action_dim=61)

    snapshots = _collect_snapshots(env, student_act, _Protocol(RESET_SEEDS, AGES))
    obs_l, act_l, rew_l, nxt_l, done_l, trunc_l = [], [], [], [], [], []
    for snap in snapshots:
        obs = _restore(env, snap)
        for _ in range(HORIZON):
            o = torch.as_tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            a = source.act(o).squeeze(0).float().cpu().numpy()
            nobs, r, term, trunc, _info = env.step(a)
            obs_l.append(np.asarray(obs, dtype=np.float32))
            act_l.append(np.asarray(a, dtype=np.float32))
            rew_l.append(float(r))
            nxt_l.append(np.asarray(nobs, dtype=np.float32))
            done_l.append(bool(term))
            trunc_l.append(bool(trunc))
            obs = nobs
            if term or trunc:
                break
    env.close()
    out = dict(
        obs=torch.as_tensor(np.stack(obs_l), device=DEVICE),
        act=torch.as_tensor(np.stack(act_l), device=DEVICE),
        rew=torch.as_tensor(np.asarray(rew_l, dtype=np.float32), device=DEVICE),
        nxt=torch.as_tensor(np.stack(nxt_l), device=DEVICE),
        done=torch.as_tensor(np.asarray(done_l, dtype=np.int64), device=DEVICE),
        trunc=torch.as_tensor(np.asarray(trunc_l, dtype=np.int64), device=DEVICE),
    )
    return out, len(obs_l)


def flat_student_batch(rb, slot_idx):
    """按 [n_env, k] slot 索引 gather 并展平为逐字段张量 dict。"""
    td = rb.gather(slot_idx)
    return dict(
        obs=td["observations"], act=td["actions"], rew=td["next"]["rewards"],
        nxt=td["next"]["observations"], done=td["next"]["dones"], trunc=td["next"]["truncations"],
    )


def critic_loss(qnet, qnet_target, actor, obs_norm, batch, noise, gamma=0.99):
    """官方 update_main 的 C51 投影 + CDQ 交叉熵损失(fp32)。"""
    obs_n = obs_norm(batch["obs"], update=False)
    nxt_n = obs_norm(batch["nxt"], update=False)
    dones = batch["done"].bool()
    truncs = batch["trunc"].bool()
    bootstrap = (truncs | ~dones).float()
    next_actions = (actor(nxt_n) + noise).clamp(-1.0, 1.0)
    discount = torch.full_like(batch["rew"], gamma)
    with torch.no_grad():
        p1, p2 = qnet_target.projection(nxt_n, next_actions, batch["rew"], bootstrap, discount)
        v1 = qnet_target.get_value(p1)
        v2 = qnet_target.get_value(p2)
        dist = torch.where(v1.unsqueeze(1) < v2.unsqueeze(1), p1, p2)  # use_cdq
    qf1, qf2 = qnet(obs_n, batch["act"])
    l1 = -torch.sum(dist * F.log_softmax(qf1, dim=1), dim=1).mean()
    l2 = -torch.sum(dist * F.log_softmax(qf2, dim=1), dim=1).mean()
    return l1 + l2


@torch.no_grad()
def soft_update(src, tgt, tau=0.1):
    for p, tp in zip(src.parameters(), tgt.parameters()):
        tp.data.mul_(1.0 - tau).add_(p.data, alpha=tau)


def run_update_unit(learner, batches, noises, mix_spec=None):
    """一个完整 outer update unit:2×critic(+第 2 次后 actor)+每次 soft target。

    mix_spec: None → control(100% student);否则 (positions, src_rows_1, src_rows_2)
    把 batch_i 的 positions 行替换为 source 行。
    """
    actor, qnet, qnet_target = learner["actor"], learner["qnet"], learner["qnet_target"]
    obs_norm = learner["obs_norm"]
    q_opt, actor_opt = learner["q_opt"], learner["actor_opt"]
    for i in (0, 1):
        batch = {k: v.clone() for k, v in batches[i].items()}
        if mix_spec is not None:
            pos, src_rows = mix_spec["positions"][i], mix_spec["src_rows"][i]
            for k in batch:
                batch[k][pos] = src_rows[k]
        loss = critic_loss(qnet, qnet_target, actor, obs_norm, batch, noises[i])
        q_opt.zero_grad(set_to_none=True)
        loss.backward()
        q_opt.step()
        if i == 1:  # num_updates=2 时官方 i%2==1 步做 actor 更新
            obs_n = obs_norm(batch["obs"], update=False).detach()
            pi = actor(obs_n)
            a1, a2 = qnet(obs_n, pi)
            av = torch.minimum(qnet.get_value(F.softmax(a1, dim=1)), qnet.get_value(F.softmax(a2, dim=1)))
            a_loss = -av.mean()
            actor_opt.zero_grad(set_to_none=True)
            a_loss.backward()
            actor_opt.step()
        soft_update(qnet, qnet_target)


@torch.no_grad()
def val_loss(learner, val_batch, noise):
    return float(critic_loss(learner["qnet"], learner["qnet_target"], learner["actor"],
                             learner["obs_norm"], val_batch, noise).item())


def snapshot_states(learner):
    return dict(
        actor=copy.deepcopy(learner["actor"].state_dict()),
        qnet=copy.deepcopy(learner["qnet"].state_dict()),
        tgt=copy.deepcopy(learner["qnet_target"].state_dict()),
        q_opt=copy.deepcopy(learner["q_opt"].state_dict()),
        a_opt=copy.deepcopy(learner["actor_opt"].state_dict()),
    )


def restore_states(learner, snap):
    learner["actor"].load_state_dict(snap["actor"])
    learner["qnet"].load_state_dict(snap["qnet"])
    learner["qnet_target"].load_state_dict(snap["tgt"])
    learner["q_opt"].load_state_dict(copy.deepcopy(snap["q_opt"]))
    learner["actor_opt"].load_state_dict(copy.deepcopy(snap["a_opt"]))


def run_cell(cell):
    torch.manual_seed(BASE_SEED)
    anchor_dir = ANCHOR_ROOT / cell["anchor"]
    learner = load_learner(anchor_dir)
    src, n_src = collect_source_transitions(cell, learner)
    rb = learner["rb"]
    valid_n = rb.valid_size
    assert valid_n == 10000, valid_n

    g = torch.Generator(device="cpu")
    g.manual_seed(BASE_SEED + 1)
    val_slot = torch.randperm(valid_n, generator=g)[:VAL_SLOTS].to(DEVICE)
    train_slots = torch.tensor(sorted(set(range(valid_n)) - set(val_slot.tolist())), device=DEVICE)
    val_idx = val_slot.view(1, -1).expand(128, -1).contiguous()
    val_batch = flat_student_batch(rb, val_idx)

    base_snap = snapshot_states(learner)
    n_act = 61
    half = (128 * 256) // 2  # 16384

    results, aux = [], {}
    for r in range(N_REPEATS):
        gr = torch.Generator(device="cpu")
        gr.manual_seed(BASE_SEED + 100 + r)
        # 共享随机量(两臂完全一致)
        batches, noises, positions, src_rows = [], [], [], []
        for i in (0, 1):
            sel = train_slots[torch.randint(0, train_slots.numel(), (128, 256), generator=gr)]
            batches.append(flat_student_batch(rb, sel.to(DEVICE)))
            noises.append((torch.randn(128 * 256, n_act, generator=gr).to(DEVICE) * 0.001).clamp(-0.5, 0.5))
            positions.append(torch.randperm(128 * 256, generator=gr)[:half].to(DEVICE))
            pick = torch.randint(0, n_src, (half,), generator=gr).to(DEVICE)
            src_rows.append(dict(obs=src["obs"][pick], act=src["act"][pick], rew=src["rew"][pick],
                                 nxt=src["nxt"][pick], done=src["done"][pick], trunc=src["trunc"][pick]))
        vnoise = (torch.randn(val_idx.numel(), n_act, generator=gr).to(DEVICE) * 0.001).clamp(-0.5, 0.5)

        restore_states(learner, base_snap)
        run_update_unit(learner, batches, noises, mix_spec=None)
        l_ctrl = val_loss(learner, val_batch, vnoise)

        restore_states(learner, base_snap)
        run_update_unit(learner, batches, noises, mix_spec=dict(positions=positions, src_rows=src_rows))
        l_mix = val_loss(learner, val_batch, vnoise)

        results.append(l_mix - l_ctrl)
        if r == 0:
            aux = compute_aux(learner, base_snap, batches[0], positions[0], src_rows[0], src, val_batch)

    restore_states(learner, base_snap)
    arr = np.asarray(results, dtype=np.float64)
    rng = np.random.default_rng(BOOT_SEED)
    boots = np.array([rng.choice(arr, size=arr.size, replace=True).mean() for _ in range(BOOT_SAMPLES)])
    lo, hi = np.percentile(boots, [(1 - CONF) / 2 * 100, (1 + CONF) / 2 * 100])
    return dict(
        cell=cell["name"], role=cell["role"], source=cell["source"], env=cell["env"],
        anchor=str(anchor_dir), unique_source_transitions=n_src,
        nominal_source_share=0.5, val_train_disjoint=True,
        n_repeats=N_REPEATS,
        I_mean=float(arr.mean()), I_std=float(arr.std(ddof=1)),
        I_lcb90=float(lo), I_ucb90=float(hi),
        positive_sign_rate=float((arr > 0).mean()),
        samples=[float(x) for x in arr],
        aux=aux,
    )


def compute_aux(learner, base_snap, batch, pos, src_rows, src, val_batch):
    """解释性辅助量(不参与裁决):梯度内积/cos、actor 目标 Δ、disagreement 比。"""
    restore_states(learner, base_snap)
    qnet, tgt, actor, obs_norm = learner["qnet"], learner["qnet_target"], learner["actor"], learner["obs_norm"]
    zero_noise = torch.zeros(pos.numel(), 61, device=DEVICE)

    def grad_of(rows):
        loss = critic_loss(qnet, tgt, actor, obs_norm, rows, zero_noise[: rows["obs"].shape[0]])
        qnet.zero_grad(set_to_none=True)
        loss.backward()
        return torch.cat([p.grad.flatten() for p in qnet.parameters() if p.grad is not None]).detach().clone()

    stu_rows = {k: v[pos] for k, v in batch.items()}
    g_src = grad_of(src_rows)
    g_stu = grad_of(stu_rows)
    qnet.zero_grad(set_to_none=True)
    inner = float(torch.dot(g_src, g_stu))
    cos = float(inner / (g_src.norm() * g_stu.norm() + 1e-12))
    with torch.no_grad():
        def disagreement(obs, acts=None):
            on = obs_norm(obs, update=False)
            a = actor(on) if acts is None else acts
            q1, q2 = qnet(on, a)
            v1 = qnet.get_value(F.softmax(q1, dim=1))
            v2 = qnet.get_value(F.softmax(q2, dim=1))
            return float((v1 - v2).abs().mean())
        d_src = disagreement(src["nxt"])
        d_stu = disagreement(val_batch["nxt"])
    return dict(critic_grad_inner=inner, critic_grad_cos=cos,
                disagreement_src_nextstate=d_src, disagreement_stu_nextstate=d_stu,
                disagreement_ratio=float(d_src / max(d_stu, 1e-9)))


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ensure_humanoidbench_import_path()
    out = dict(
        protocol=dict(
            estimand="I_critic = L_val(U(theta;50%stu+50%src)) - L_val(U(theta;100%stu))",
            update_unit="2x critic AdamW + 1x actor AdamW (on 2nd) + soft target tau=0.1 each",
            reset_seeds=RESET_SEEDS, ages=AGES, horizon=HORIZON,
            n_repeats=N_REPEATS, val_slots=VAL_SLOTS,
            bootstrap=dict(samples=BOOT_SAMPLES, seed=BOOT_SEED, confidence=CONF),
            interval_scope="conditional batch uncertainty given the 10k learner anchor; not a cross-seed CI",
            labels="operational calibration labels from within-stage 10k-20k paired segment nAUC",
            precision="fp32 probe (training used bf16 AMP); normalizer frozen (update=False)",
        ),
        utc=datetime.now(timezone.utc).isoformat(),
        cells=[],
    )
    for cell in CELLS:
        print(f"=== {cell['name']} ===", flush=True)
        res = run_cell(cell)
        torch.cuda.empty_cache()
        out["cells"].append(res)
        print(json.dumps({k: v for k, v in res.items() if k != "samples"}, indent=1, ensure_ascii=False), flush=True)

    harmful = [c for c in out["cells"] if c["role"] == "harmful"]
    helpful = [c for c in out["cells"] if c["role"] == "helpful"]
    h = harmful[0]
    absolute = h["I_lcb90"] > 0 and all(c["I_ucb90"] < 0 for c in helpful)
    ranking = h["I_lcb90"] > max(c["I_ucb90"] for c in helpful)
    if absolute:
        verdict = "ABSOLUTE_PASS"
    elif ranking:
        verdict = "RANKING_ONLY"
    else:
        verdict = "FAIL"
    out["verdict"] = verdict
    path = OUT_DIR / "influence_gate_v1_results.json"
    path.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(f"\nVERDICT: {verdict}\nsaved: {path}")


if __name__ == "__main__":
    main()
