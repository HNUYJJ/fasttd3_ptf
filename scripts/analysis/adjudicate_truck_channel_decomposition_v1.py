#!/usr/bin/env python3
"""T2 判决：Truck 10k→20k 的 behavior / replay 因果分解。

判据冻结于 `docs/experiments/truck_channel_decomposition_prereg_20260808.md`，
本脚本在 B-only 臂产出评估结果**之前**写就。之后只允许改路径参数。

    U^B        = J_Bonly - J_scratch          behavior 通道
    Δ^{R|B}    = J_joint - J_Bonly            在 behavior 已开的条件下再加 replay
    U^{BR}     = J_joint - J_scratch          Gate A 已测得的联合效应

恒等式 U^{BR} = U^B + Δ^{R|B} 作数值一致性自检。
逐 learner seed 报告，**用 learner 间方差，绝不用 episode 面板 SE**（M16）。

工程 gate（预注册 §2）先于科学判定；任一不过则本轮作废。
任何缺失一律 INCOMPLETE 且非零退出，且独立扫描全部组合不 continue。

用法：python scripts/analysis/adjudicate_truck_channel_decomposition_v1.py
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
SEEDS = (1, 2, 3)
EVAL_ROOT = REPO / "docs/data/pare_gate_a_v1/source_free_eval"
BONLY_EVAL_ROOT = REPO / "docs/data/truck_channel_v1/source_free_eval"
GATE_A_ANCHORS = REPO / "artifacts/pare_gate_a_v1/anchors"
SHARE_TOL = 0.01


def load_return(path: Path):
    if not path.exists():
        return None, f"缺文件 {path.name}"
    blob = json.loads(path.read_text())
    agg = blob.get("aggregate") or {}
    if "return_mean" not in agg:
        return None, f"{path.name} 内无 aggregate.return_mean"
    return float(agg["return_mean"]), None


def mean_sd(xs):
    n = len(xs)
    m = sum(xs) / n
    if n < 2:
        return m, float("nan")
    return m, math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))


def engineering_gate(seed: int) -> dict:
    """E1–E4。B-only 的定义就是 E2 与 E3 **同时**成立。"""
    out = {"seed": seed}
    b_dir = GATE_A_ANCHORS / f"truck_s{seed}_bonly_k20000"
    j_dir = GATE_A_ANCHORS / f"truck_s{seed}_scaf_k20000"
    for d in (b_dir, j_dir):
        if not d.exists():
            out["status"] = "INCOMPLETE"
            out["reason"] = f"缺 anchor {d.name}"
            return out

    bl = torch.load(b_dir / "learner.pt", map_location="cpu", weights_only=False)
    jl = torch.load(j_dir / "learner.pt", map_location="cpu", weights_only=False)

    def beh_share(learner):
        ec = torch.as_tensor(
            learner["auxiliary_state"]["admission_execution_counts"]).double()
        return float(ec[:-1].sum()) / float(ec.sum())

    b_share, j_share = beh_share(bl), beh_share(jl)
    out["E1_behavior_share"] = {
        "bonly": round(b_share, 6), "joint": round(j_share, 6),
        "abs_diff": round(abs(b_share - j_share), 6),
        "pass": abs(b_share - j_share) <= SHARE_TOL,
    }

    blob = torch.load(b_dir / "replay.pt", map_location="cpu", weights_only=False)
    sc = (blob.get("admission_sampling") or {}).get("sample_counts") or {}
    crit = torch.as_tensor(sc.get("critic", [])).double() if "critic" in sc else None
    if crit is None or crit.numel() < 2:
        out["E2_critic_source_samples"] = {"pass": False,
                                           "reason": "无 critic sample_counts"}
    else:
        n_src = int(crit[:-1].sum())
        out["E2_critic_source_samples"] = {
            "source_samples": n_src, "student_samples": int(crit[-1]),
            "pass": n_src == 0,
            "note": "student_only 下 source 配额恒 0，故必须严格为 0",
        }

    prov = blob["provenance"]
    written = torch.as_tensor(prov["provenance_written"]).bool()
    is_src = torch.as_tensor(prov["executed_group_mask"]).any(dim=-1) & written
    n_phys = int(is_src.sum())
    out["E3_source_physical_transitions"] = {
        "count": n_phys, "pass": n_phys > 0,
        "note": "source 仍有 behavior authority 并写入 physical buffer",
    }

    valid = int(blob["metadata"]["valid_size"])
    n_env = int(blob["metadata"]["n_env"])
    n_written = int(written[:, :valid].sum())
    out["E4_provenance_complete"] = {"pass": n_written == valid * n_env,
                                     "written": n_written,
                                     "expected": valid * n_env}

    out["status"] = "OK"
    out["all_pass"] = all(
        out[k].get("pass", False)
        for k in ("E1_behavior_share", "E2_critic_source_samples",
                  "E3_source_physical_transitions", "E4_provenance_complete")
    )
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/data/truck_channel_v1/channel_verdict.json")
    args = ap.parse_args()

    gates = [engineering_gate(s) for s in SEEDS]
    gate_ok = all(g.get("status") == "OK" and g.get("all_pass") for g in gates)

    # 独立扫描全部 (arm, seed)，不因前置缺失而 continue
    vals, missing = {}, []
    for s in SEEDS:
        for arm, path in (
            ("scratch", EVAL_ROOT / f"truck_scratch_s{s}_step20000.json"),
            ("joint", EVAL_ROOT / f"truck_scaf_s{s}_step20000.json"),
            ("bonly", BONLY_EVAL_ROOT / f"truck_bonly_s{s}_step20000.json"),
        ):
            v, why = load_return(path)
            if v is None:
                missing.append(why)
            else:
                vals[(arm, s)] = v

    report = {"prereg": "docs/experiments/truck_channel_decomposition_prereg_20260808.md",
              "engineering_gate": {"all_pass": gate_ok, "per_seed": gates},
              "scope": "根因定位，非论文 confirmatory result"}

    if missing:
        report["verdict"] = "INCOMPLETE"
        report["missing"] = missing
    elif not gate_ok:
        report["verdict"] = "ENGINEERING_GATE_FAILED"
    else:
        U_B, D_RB, U_BR = [], [], []
        for s in SEEDS:
            j0, jb, jbr = vals[("scratch", s)], vals[("bonly", s)], vals[("joint", s)]
            U_B.append(jb - j0)
            D_RB.append(jbr - jb)
            U_BR.append(jbr - j0)
        ident_err = max(abs((a + b) - c) for a, b, c in zip(U_B, D_RB, U_BR))

        def blk(xs):
            m, sd = mean_sd(xs)
            se = sd / math.sqrt(len(xs))
            return {"per_seed": [round(x, 3) for x in xs], "mean": round(m, 3),
                    "sd_learner": round(sd, 3), "se_learner": round(se, 3),
                    "t": round(m / se, 3) if se > 0 else None,
                    "all_negative": all(x < 0 for x in xs),
                    "sign_flip": (max(xs) > 0) and (min(xs) < 0)}

        b_blk, r_blk, br_blk = blk(U_B), blk(D_RB), blk(U_BR)
        report["decomposition"] = {
            "U_B_behavior_channel": b_blk,
            "Delta_R_given_B_replay_channel": r_blk,
            "U_BR_joint_gate_a": br_blk,
            "identity_max_abs_error": round(ident_err, 9),
            "per_seed_returns": {f"s{s}": {a: round(vals[(a, s)], 3)
                                           for a in ("scratch", "bonly", "joint")}
                                 for s in SEEDS},
        }

        # 预注册 §3：跨 seed 翻符号优先——它是实质裁决之一，不是失败
        if b_blk["sign_flip"] or r_blk["sign_flip"]:
            verdict = "CHANNEL_UNRESOLVED"
        elif b_blk["all_negative"] and not r_blk["all_negative"]:
            verdict = "BEHAVIOR_SIDE_CANDIDATE"
        elif r_blk["all_negative"] and not b_blk["all_negative"]:
            verdict = "REPLAY_SIDE_CANDIDATE"
        elif b_blk["all_negative"] and r_blk["all_negative"]:
            verdict = "JOINT_HARM_CANDIDATE"
        else:
            verdict = "CHANNEL_UNRESOLVED"
        report["verdict"] = verdict

    text = json.dumps(report, indent=2, ensure_ascii=False)
    print(text)
    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text + "\n")
    return 0 if report["verdict"] not in ("INCOMPLETE", "ENGINEERING_GATE_FAILED") else 1


if __name__ == "__main__":
    sys.exit(main())
