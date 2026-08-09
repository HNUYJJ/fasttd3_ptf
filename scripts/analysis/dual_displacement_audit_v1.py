#!/usr/bin/env python3
"""T3 Dual Displacement Audit（零环境交互）。

判据与放大律冻结于 `docs/experiments/dual_displacement_audit_prereg_20260808.md`，
本脚本在计算任何 displacement 数值**之前**写就。之后只允许改路径参数。

Part A —— replay displacement：
    rho_S = physical source slots / all valid slots
    q_S   = critic source samples / all critic samples
    A     = (q_S/rho_S) / ((1-q_S)/(1-rho_S))
理论（无自由参数，直接来自 replay sampling rule）：
    rho_S(u) = m*u/(H+u),   q_S = m,   A(u) = 1 + H/((1-m)*u)
Gate A / T2 的 H=10k, m=0.5, u=10k  =>  rho_S=0.25, q_S=0.5, A=3.

Part B —— behavior event yield：直接用 truck reward 的真实离散事件
（+100 抬起/放置、-100 掉落、+1000 成功），按 provenance 分组统计。
**只报告，不设通过阈值，不作为新 proxy。**

用法：python scripts/analysis/dual_displacement_audit_v1.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
ANCHORS = REPO / "artifacts/pare_gate_a_v1/anchors"
SEEDS = (1, 2, 3)

H_STEPS, M_MASS, U_STEPS = 10_000, 0.5, 10_000
A_CONFIRM_MIN = 2.5          # 预注册 §1 判据
A_REFUTE_MAX = 1.5
RHO_TOL = 0.02

POS_EVENT, NEG_EVENT, SUCC_EVENT = 50.0, -50.0, 900.0


def theory():
    rho = M_MASS * U_STEPS / (H_STEPS + U_STEPS)
    a = 1.0 + H_STEPS / ((1.0 - M_MASS) * U_STEPS)
    return {"rho_S": rho, "q_S": M_MASS, "q_over_rho": M_MASS / rho, "A": a}


def amplification(q, rho):
    if rho <= 0 or rho >= 1 or q <= 0 or q >= 1:
        return float("nan")
    return (q / rho) / ((1.0 - q) / (1.0 - rho))


def audit_anchor(task: str, seed: int, arm: str) -> dict:
    """arm: 'scaf'(joint) 或 'bonly'。"""
    name = f"{task}_s{seed}_scaf_k20000" if arm == "scaf" else f"{task}_s{seed}_bonly_k20000"
    adir = ANCHORS / name
    if not adir.exists():
        return {"status": "MISSING", "anchor": name}

    blob = torch.load(adir / "replay.pt", map_location="cpu", weights_only=False)
    meta, prov = blob["metadata"], blob["provenance"]
    tensors = blob["tensors"]
    valid = int(meta["valid_size"])
    n_env = int(meta["n_env"])
    buf = int(meta["buffer_size"])

    written = torch.as_tensor(prov["provenance_written"]).bool()[:, :valid]
    is_src = torch.as_tensor(prov["executed_group_mask"]).any(dim=-1)[:, :valid] & written
    n_slots = valid * n_env
    n_src = int(is_src.sum())
    rho_S = n_src / n_slots

    sc = (blob.get("admission_sampling") or {}).get("sample_counts") or {}
    q_S, crit_raw = None, None
    if "critic" in sc:
        c = torch.as_tensor(sc["critic"]).double()
        crit_raw = [int(x) for x in c]
        tot = float(c.sum())
        if tot > 0:
            q_S = float(c[:-1].sum()) / tot

    # ── Part B：真实离散 reward 事件，按 provenance 分组 ──────────────
    events = {}
    if "rewards" in tensors:
        r = torch.as_tensor(tensors["rewards"])[:, :valid]
        stu = written & ~is_src

        def rate(mask, cond):
            n = int(mask.sum())
            return {"n_transitions": n,
                    "count": int((cond & mask).sum()),
                    "rate": round(float((cond & mask).sum()) / n, 8) if n else None}

        for label, cond in (("positive", r > POS_EVENT),
                            ("negative", r < NEG_EVENT),
                            ("success", r > SUCC_EVENT)):
            events[label] = {"source": rate(is_src, cond), "student": rate(stu, cond)}

    # arms 组被 source 接管的比例（provenance groups: [legs_torso, arms]）
    gm = torch.as_tensor(prov["executed_group_mask"])[:, :valid]
    per_group = {}
    if gm.ndim == 3:
        for gi in range(gm.shape[-1]):
            per_group[f"group_{gi}"] = round(float((gm[..., gi] & written).sum()) / n_slots, 6)

    # ρ_S(t)：按 learner_step 每 1k 重建
    step = torch.as_tensor(prov["learner_step"])[:, :valid]
    curve = []
    for t in range(1000, U_STEPS + H_STEPS + 1, 1000):
        upto = written & (step < t) & (step >= 0)
        n_up = int(upto.sum())
        if n_up == 0:
            continue
        s_up = int((is_src & upto).sum())
        u_eff = max(0, t - H_STEPS)
        th_rho = (M_MASS * u_eff / t) if t > 0 else 0.0
        th_A = (1.0 + H_STEPS / ((1 - M_MASS) * u_eff)) if u_eff > 0 else None
        curve.append({"step": t, "rho_S_measured": round(s_up / n_up, 6),
                      "rho_S_theory": round(th_rho, 6),
                      "A_theory": None if th_A is None else round(th_A, 4)})

    del blob, tensors, prov
    out = {
        "status": "OK", "anchor": name,
        "valid_size": valid, "n_env": n_env, "buffer_size_per_env": buf,
        "buffer_not_wrapped": valid < buf,
        "n_slots": n_slots, "n_source_slots": n_src,
        "rho_S": round(rho_S, 6),
        "q_S": None if q_S is None else round(q_S, 6),
        "critic_counts_raw": crit_raw,
        "q_over_rho": None if q_S is None else round(q_S / rho_S, 6),
        "A": None if q_S is None else round(amplification(q_S, rho_S), 6),
        "group_source_share": per_group,
        "rho_curve": curve,
        "reward_events": events,
    }
    return out


def main() -> int:
    th = theory()
    report = {
        "prereg": "docs/experiments/dual_displacement_audit_prereg_20260808.md",
        "theory": {"H": H_STEPS, "m": M_MASS, "u": U_STEPS,
                   "formula_rho": "rho_S(u)=m*u/(H+u)",
                   "formula_A": "A(u)=1+H/((1-m)*u)",
                   **{k: round(v, 6) for k, v in th.items()}},
        "criterion": {"confirm": f"truck 3/3 seed 满足 A>={A_CONFIRM_MIN} 且 |rho_S-0.25|<={RHO_TOL}",
                      "refute": f"truck 出现 A<{A_REFUTE_MAX} 的 seed",
                      "note": "阈值先于任何实测数值冻结；stair 仅对照不参与判定"},
        "per_task": {},
    }

    for task in ("truck", "stair"):
        rows = []
        for s in SEEDS:
            print(f"[audit] {task} s{s} joint ...", flush=True)
            rows.append({"seed": s, "joint": audit_anchor(task, s, "scaf")})
            if task == "truck":
                print(f"[audit] {task} s{s} b-only ...", flush=True)
                rows[-1]["bonly"] = audit_anchor(task, s, "bonly")
        report["per_task"][task] = rows

    truck = [r["joint"] for r in report["per_task"]["truck"] if r["joint"]["status"] == "OK"]
    if len(truck) < len(SEEDS) or any(r["A"] is None for r in truck):
        verdict = "INCOMPLETE"
    else:
        As = [r["A"] for r in truck]
        rhos = [r["rho_S"] for r in truck]
        confirm = all(a >= A_CONFIRM_MIN for a in As) and \
            all(abs(x - th["rho_S"]) <= RHO_TOL for x in rhos)
        refute = any(a < A_REFUTE_MAX for a in As)
        verdict = ("AMPLIFICATION_CONFIRMED" if confirm
                   else "AMPLIFICATION_REFUTED" if refute
                   else "AMPLIFICATION_PARTIAL")
        report["truck_A"] = [round(a, 4) for a in As]
        report["truck_rho"] = [round(x, 4) for x in rhos]
    report["verdict"] = verdict

    text = json.dumps(report, indent=2, ensure_ascii=False)
    out = REPO / "docs/data/dual_displacement_v1/audit.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text + "\n")
    print(f"\nverdict = {verdict}")
    return 0 if verdict != "INCOMPLETE" else 1


if __name__ == "__main__":
    sys.exit(main())
