#!/usr/bin/env python3
"""T1 机制 probe：source-authority 状态与 student-authority 状态给 actor 的
policy-improvement 梯度是否冲突。**零环境交互，只读已有的 20k branch anchor。**

## 被检验的命题

deterministic actor-critic 的 actor 更新是在某个状态分布上求 action-value 梯度：

    ψ_θ(s) = ∇_θ π_θ(s) · ∇_a Q(s,a)|_{a=π_θ(s)},    g_d = E_{s∼d}[ψ_θ(s)]

scaffold 期的 replay 状态分布是混合 d_M = (1−α)·d_0 + α·d_S，故

    g_M − g_0 = α·(g_S − g_0)

若 ⟨g_0, g_S⟩ < 0，source-shaped 状态就在给 actor 一个与 student-authority
状态相冲突的更新方向——**而这并不要求 source transition 对 critic 无用**。
这正是要测的：source experience 对 value learning 与对 policy improvement
的作用可能不同甚至相反。

## 为什么必须有组内 split-half 对照

跨组 cosine 单独看无法解释：cos=−0.1 既可能是真冲突，也可能是梯度估计噪声。
故同时报告**同一组内**随机对半的 cosine。组内 cosine 高而跨组为负，才是冲突；
组内 cosine 本身就低，说明估计噪声主导，跨组结论不可信。

## 判据（先于查看任何 cosine 冻结）

PRIMARY = truck。进入 T2 需**同时**满足：
  (a) 至少 2/3 learner seed 的 cos(g_src, g_stu) < 0；
  (b) 这些 seed 的组内 split-half cosine 均 > 0.5（估计足够稳定）。
否则停止该假设，不发明算法。stair 仅作 diagnostic，不参与判定。

每个 learner seed 只产出**一个** cosine——batch 重复不是 learner replication。

用法：python scripts/analysis/probe_provenance_actor_gradient_v1.py [--tasks truck stair]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from fasttd3_ptf.official_fasttd3_ptf import ensure_fasttd3_import_path  # noqa: E402

ensure_fasttd3_import_path()

from fast_td3 import Actor, Critic  # type: ignore  # noqa: E402

ANCHOR_ROOT = REPO / "artifacts/pare_gate_a_v1/anchors"
SCAFFOLD_LO, SCAFFOLD_HI = 10_000, 20_000
CHUNK = 4096
SPLIT_HALF_MIN = 0.5
SEEDS = (1, 2, 3)


def flat_dot(a, b):
    return float(sum((x * y).sum() for x, y in zip(a, b)))


def flat_norm(a):
    return float(sum((x * x).sum() for x in a)) ** 0.5


def cosine(a, b):
    na, nb = flat_norm(a), flat_norm(b)
    return flat_dot(a, b) / (na * nb) if na > 0 and nb > 0 else float("nan")


def build_networks(cfg, meta, device):
    args = cfg["args"]
    n_obs = int(meta["n_obs"])
    n_act = int(meta["n_act"])
    n_critic_obs = int(meta["n_critic_obs"])
    actor = Actor(n_obs=n_obs, n_act=n_act, num_envs=int(args["num_envs"]),
                  device=device, init_scale=float(args["init_scale"]),
                  hidden_dim=int(args["actor_hidden_dim"]),
                  std_min=float(args["std_min"]), std_max=float(args["std_max"]))
    critic = Critic(n_obs=n_critic_obs, n_act=n_act,
                    num_atoms=int(args["num_atoms"]), v_min=float(args["v_min"]),
                    v_max=float(args["v_max"]),
                    hidden_dim=int(args["critic_hidden_dim"]), device=device)
    return actor, critic


@torch.enable_grad()
def actor_ascent_gradient(actor, critic, obs, critic_obs, params, use_cdq):
    """∇_θ E_s[ Q_L(s, π_θ(s)) ]，分块累加后除以样本数。

    Q_L = min(Q1, Q2)（与 update_pol 在 use_cdq=True 下的取法一致）。
    这是 **ascent** 方向：actor 要最大化它。
    """
    total = [torch.zeros_like(p) for p in params]
    n = int(obs.shape[0])
    for i in range(0, n, CHUNK):
        o = obs[i:i + CHUNK]
        co = critic_obs[i:i + CHUNK]
        a = actor(o)
        q1, q2 = critic(co, a)
        v1 = critic.get_value(torch.softmax(q1, dim=1))
        v2 = critic.get_value(torch.softmax(q2, dim=1))
        v = torch.minimum(v1, v2) if use_cdq else (v1 + v2) / 2.0
        g = torch.autograd.grad(v.sum(), params, retain_graph=False)
        for t, gi in zip(total, g):
            t.add_(gi)
    return [t / max(1, n) for t in total]


def probe_seed(task: str, seed: int, device, actor_at: str = "20k") -> dict:
    """``actor_at='20k'``：冻结判据要求的口径（scaffold 结束时刻的 actor/critic）。

    ``actor_at='10k'``：**post-hoc 诊断**，用 scaffold 尚未开始时（A0）的
    actor/critic 评估同一批 scaffold 状态。理由：20k 的 actor 已在混合分布上
    训练了 10k 步，此时两组梯度对齐可能是"适应的结果"而非"从未冲突"。
    该口径**不参与 T1 裁决**。
    """
    adir = ANCHOR_ROOT / f"{task}_s{seed}_scaf_k20000"
    ldir = adir if actor_at == "20k" else ANCHOR_ROOT / f"{task}_s{seed}_k10000"
    learner = torch.load(ldir / "learner.pt", map_location="cpu", weights_only=False)
    cfg = learner["configuration"]
    args = cfg["args"]
    use_cdq = bool(args["use_cdq"])

    # ── 实际剂量（名义 mass 0.5 不等于实际占用时间）──────────────────
    # 剂量恒从 A1 读——A0 是 empty-bank prefix，没有 source 执行记录。
    aux = torch.load(adir / "learner.pt", map_location="cpu",
                     weights_only=False)["auxiliary_state"] if actor_at != "20k" \
        else learner["auxiliary_state"]
    ec = torch.as_tensor(aux["admission_execution_counts"]).double()
    names = list(aux.get("source_names") or [])
    tot_exec = float(ec.sum())
    behavior_share = float(ec[:-1].sum()) / tot_exec
    per_source = {n: round(float(ec[i]) / tot_exec, 4) for i, n in enumerate(names)}

    blob = torch.load(adir / "replay.pt", map_location="cpu", weights_only=False)
    meta = blob["metadata"]
    tensors = blob["tensors"]
    prov = blob["provenance"]

    # ── 按 provenance 切分 scaffold 期状态 ───────────────────────────
    step = torch.as_tensor(prov["learner_step"])
    written = torch.as_tensor(prov["provenance_written"]).bool()
    in_window = (step >= SCAFFOLD_LO) & (step < SCAFFOLD_HI) & written
    is_src = torch.as_tensor(prov["executed_group_mask"]).any(dim=-1) & in_window
    is_stu = (~torch.as_tensor(prov["executed_group_mask"]).any(dim=-1)) & in_window

    obs_all = tensors["observations"]
    asym = bool(meta["asymmetric_obs"])
    cobs_all = tensors["critic_observations"] if asym else obs_all

    def take(mask):
        idx = mask.nonzero(as_tuple=False)
        return obs_all[idx[:, 0], idx[:, 1]], cobs_all[idx[:, 0], idx[:, 1]]

    src_obs, src_cobs = take(is_src)
    stu_obs, stu_cobs = take(is_stu)
    del blob, tensors, prov

    n_src, n_stu = int(src_obs.shape[0]), int(stu_obs.shape[0])
    if n_src == 0 or n_stu == 0:
        return {"seed": seed, "status": "INCOMPLETE",
                "reason": f"scaffold 窗口内 z=1 有 {n_src} 条、z=0 有 {n_stu} 条"}

    # ── 重建 20k 时刻的 actor / critic / normalizer ──────────────────
    actor, critic = build_networks(cfg, meta, device)
    actor.load_state_dict(learner["modules"]["actor"])
    critic.load_state_dict(learner["modules"]["critic"])
    actor.eval()
    critic.eval()
    for p in critic.parameters():
        p.requires_grad_(False)
    params = [p for p in actor.parameters() if p.requires_grad]

    # replay 存 raw obs，采样后才 normalize（train_ptf.py:3235）。
    # 必须用 anchor 里 20k 时刻的 normalizer，不能重新估。
    from fast_td3_utils import EmpiricalNormalization  # type: ignore

    obs_norm = EmpiricalNormalization(shape=int(meta["n_obs"]), device=device)
    obs_norm.load_state_dict(learner["modules"]["obs_normalizer"])
    obs_norm.eval()
    if asym:
        cobs_norm = EmpiricalNormalization(shape=int(meta["n_critic_obs"]), device=device)
        cobs_norm.load_state_dict(learner["modules"]["critic_obs_normalizer"])
        cobs_norm.eval()

    @torch.no_grad()
    def prep(o, co):
        o = obs_norm(o.to(device), update=False)
        co = cobs_norm(co.to(device), update=False) if asym else o
        return o, co

    src_o, src_c = prep(src_obs, src_cobs)
    stu_o, stu_c = prep(stu_obs, stu_cobs)

    def grad_of(o, c):
        return actor_ascent_gradient(actor, critic, o, c, params, use_cdq)

    g_src = grad_of(src_o, src_c)
    g_stu = grad_of(stu_o, stu_c)

    # ── 组内 split-half：把估计噪声与真冲突分开 ─────────────────────
    gen = torch.Generator(device="cpu").manual_seed(20260808 + seed)

    def split_half_cos(o, c):
        n = o.shape[0]
        perm = torch.randperm(n, generator=gen).to(o.device)
        a, b = perm[: n // 2], perm[n // 2:]
        return cosine(grad_of(o[a], c[a]), grad_of(o[b], c[b]))

    return {
        "seed": seed,
        "status": "OK",
        "behavior_source_share": round(behavior_share, 4),
        "per_source_share": per_source,
        "n_source_states": n_src,
        "n_student_states": n_stu,
        "cos_src_stu": round(cosine(g_src, g_stu), 6),
        "dot_src_stu": flat_dot(g_src, g_stu),
        "norm_ratio_src_over_stu": round(flat_norm(g_src) / max(1e-12, flat_norm(g_stu)), 6),
        "split_half_cos_source": round(split_half_cos(src_o, src_c), 6),
        "split_half_cos_student": round(split_half_cos(stu_o, stu_c), 6),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=["truck", "stair"])
    ap.add_argument("--actor-at", choices=["20k", "10k"], default="20k",
                    help="20k=冻结判据口径；10k=post-hoc 诊断，不参与裁决")
    ap.add_argument("--out", default="docs/data/pdau_probe_v1/gradient_probe.json")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    report = {"primary_task": "truck", "actor_at": args.actor_at, "criterion": {
        "conflict": "cos(g_src, g_stu) < 0",
        "stability": f"split-half cosine > {SPLIT_HALF_MIN} 于两组",
        "advance_to_T2": "truck 至少 2/3 seed 同时满足 conflict 与 stability",
        "note": "每 seed 一个 cosine；batch 重复不是 learner replication",
    }, "per_task": {}}

    for task in args.tasks:
        rows = []
        for s in SEEDS:
            print(f"[probe] {task} s{s} ...", flush=True)
            rows.append(probe_seed(task, s, device, actor_at=args.actor_at))
            print(f"    {rows[-1]}", flush=True)
        report["per_task"][task] = rows

    truck = report["per_task"].get("truck", [])
    ok = [r for r in truck if r.get("status") == "OK"]
    if len(ok) < len(SEEDS):
        verdict = "INCOMPLETE"
    else:
        hits = [r for r in ok
                if r["cos_src_stu"] < 0
                and min(r["split_half_cos_source"], r["split_half_cos_student"]) > SPLIT_HALF_MIN]
        verdict = "CONFLICT_SUPPORTED" if len(hits) >= 2 else "NO_CONFLICT_STOP"
        report["n_conflicting_seeds"] = len(hits)
    # 10k 口径是 post-hoc 诊断，不得产生 T1 裁决——冻结判据只认 20k。
    report["verdict"] = verdict if args.actor_at == "20k" else f"DIAGNOSTIC_ONLY({verdict})"

    text = json.dumps(report, indent=2, ensure_ascii=False)
    print(text)
    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text + "\n")
    return 0 if verdict in ("CONFLICT_SUPPORTED", "NO_CONFLICT_STOP") else 1


if __name__ == "__main__":
    sys.exit(main())
