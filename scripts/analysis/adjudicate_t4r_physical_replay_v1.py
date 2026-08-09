#!/usr/bin/env python3
"""T4-R 判决：physical replay（q_S=rho_S）vs B-only（q_S=0）vs fixed quota（q_S≈0.5）。

判据冻结于 `docs/experiments/t4r_physical_replay_prereg_20260808.md`，
本脚本在 Rphys 臂产出评估结果**之前**写就。之后只允许改路径参数。

三臂的 source behavior 完全相同，唯一差别是 replay 曝光：

    R0    = B-only   q_S = 0
    Rphys = physical q_S = rho_S      <- 本轮新增
    Rfix  = joint    q_S ≈ 0.5        （fixed provenance quota，T3 实测 A≈2.95）

主判据是 3/3 符号（与 T2 同口径）；n=3 下 t 只作报告。
工程 gate G1–G5 先于科学判定；G3/G4 是本臂的定义性检查。
任何缺失 → INCOMPLETE 且非零退出，独立扫描全部组合不 continue。
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
ANCHORS = REPO / "artifacts/pare_gate_a_v1/anchors"
EVAL_GATE_A = REPO / "docs/data/pare_gate_a_v1/source_free_eval"
EVAL_BONLY = REPO / "docs/data/truck_channel_v1/source_free_eval"
EVAL_PHYS = REPO / "docs/data/t4r_phys_v1/source_free_eval"

SHARE_TOL = 0.01
Q_RHO_TOL = 0.03      # G3
A_MAX = 1.3           # G4
UBR_SE = 39.463       # T2 的 U_BR learner 间 SE，用于 "≈" 的操作化


def load_return(path: Path):
    if not path.exists():
        return None, f"缺文件 {path.name}"
    agg = (json.loads(path.read_text()).get("aggregate") or {})
    if "return_mean" not in agg:
        return None, f"{path.name} 内无 aggregate.return_mean"
    return float(agg["return_mean"]), None


def mean_sd(xs):
    n = len(xs)
    m = sum(xs) / n
    return (m, math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))) if n > 1 else (m, float("nan"))


def gate(seed: int) -> dict:
    out = {"seed": seed}
    p_dir = ANCHORS / f"truck_s{seed}_phys_k20000"
    j_dir = ANCHORS / f"truck_s{seed}_scaf_k20000"
    for d in (p_dir, j_dir):
        if not d.exists():
            return {"seed": seed, "status": "INCOMPLETE", "reason": f"缺 anchor {d.name}"}

    def beh(learner):
        ec = torch.as_tensor(learner["auxiliary_state"]["admission_execution_counts"]).double()
        return float(ec[:-1].sum()) / float(ec.sum())

    pb = beh(torch.load(p_dir / "learner.pt", map_location="cpu", weights_only=False))
    jb = beh(torch.load(j_dir / "learner.pt", map_location="cpu", weights_only=False))
    out["G1_behavior_share"] = {"phys": round(pb, 6), "joint": round(jb, 6),
                                "abs_diff": round(abs(pb - jb), 6),
                                "pass": abs(pb - jb) <= SHARE_TOL}

    blob = torch.load(p_dir / "replay.pt", map_location="cpu", weights_only=False)
    meta, prov = blob["metadata"], blob["provenance"]
    v, n_env = int(meta["valid_size"]), int(meta["n_env"])
    written = torch.as_tensor(prov["provenance_written"]).bool()[:, :v]
    is_src = torch.as_tensor(prov["executed_group_mask"]).any(dim=-1)[:, :v] & written
    n_src = int(is_src.sum())
    rho = n_src / (v * n_env)
    out["G2_source_physical"] = {"count": n_src, "pass": n_src > 0}

    sc = (blob.get("admission_sampling") or {}).get("sample_counts") or {}
    q = None
    if "critic" in sc:
        c = torch.as_tensor(sc["critic"]).double()
        tot = float(c.sum())
        if tot > 0:
            q = float(c[:-1].sum()) / tot
    if q is None:
        out["G3_q_tracks_rho"] = {"pass": False, "reason": "无 critic sample_counts"}
        out["G4_amplification"] = {"pass": False}
    else:
        A = (q / rho) / ((1 - q) / (1 - rho)) if 0 < rho < 1 and 0 < q < 1 else float("nan")
        out["G3_q_tracks_rho"] = {"rho_S": round(rho, 6), "q_S": round(q, 6),
                                  "abs_diff": round(abs(q - rho), 6),
                                  "pass": abs(q - rho) <= Q_RHO_TOL}
        out["G4_amplification"] = {"A": round(A, 6), "threshold": A_MAX,
                                   "pass": bool(A <= A_MAX),
                                   "note": "fixed quota 下 T3 实测 A≈2.95"}

    n_written = int(written.sum())
    out["G5_provenance"] = {"pass": n_written == v * n_env,
                            "written": n_written, "expected": v * n_env}
    out["status"] = "OK"
    out["all_pass"] = all(out[k].get("pass", False) for k in
                          ("G1_behavior_share", "G2_source_physical", "G3_q_tracks_rho",
                           "G4_amplification", "G5_provenance"))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/data/t4r_phys_v1/t4r_verdict.json")
    args = ap.parse_args()

    gates = [gate(s) for s in SEEDS]
    gate_ok = all(g.get("status") == "OK" and g.get("all_pass") for g in gates)

    vals, missing = {}, []
    for s in SEEDS:
        for arm, path in (
            ("bonly", EVAL_BONLY / f"truck_bonly_s{s}_step20000.json"),
            ("phys", EVAL_PHYS / f"truck_phys_s{s}_step20000.json"),
            ("fixed", EVAL_GATE_A / f"truck_scaf_s{s}_step20000.json"),
            ("scratch", EVAL_GATE_A / f"truck_scratch_s{s}_step20000.json"),
        ):
            v, why = load_return(path)
            if v is None:
                missing.append(why)
            else:
                vals[(arm, s)] = v

    report = {"prereg": "docs/experiments/t4r_physical_replay_prereg_20260808.md",
              "engineering_gate": {"all_pass": gate_ok, "per_seed": gates},
              "scope": "只补 q_S=rho_S 这一格；不搜索 q_S 最优值"}

    if missing:
        report["verdict"] = "INCOMPLETE"
        report["missing"] = missing
    elif not gate_ok:
        report["verdict"] = "ENGINEERING_GATE_FAILED"
    else:
        d_pf = [vals[("phys", s)] - vals[("fixed", s)] for s in SEEDS]   # P1
        d_pb = [vals[("phys", s)] - vals[("bonly", s)] for s in SEEDS]   # P2/P3
        d_ps = [vals[("phys", s)] - vals[("scratch", s)] for s in SEEDS]

        def blk(xs):
            m, sd = mean_sd(xs)
            se = sd / math.sqrt(len(xs))
            return {"per_seed": [round(x, 3) for x in xs], "mean": round(m, 3),
                    "sd_learner": round(sd, 3), "se_learner": round(se, 3),
                    "t": round(m / se, 3) if se > 0 else None,
                    "all_positive": all(x > 0 for x in xs),
                    "all_negative": all(x < 0 for x in xs),
                    "sign_flip": (max(xs) > 0) and (min(xs) < 0)}

        pf, pb, ps = blk(d_pf), blk(d_pb), blk(d_ps)
        report["comparisons"] = {
            "phys_minus_fixed_P1": pf,
            "phys_minus_bonly": pb,
            "phys_minus_scratch": ps,
            "per_seed_returns": {f"s{s}": {a: round(vals[(a, s)], 3)
                                           for a in ("scratch", "bonly", "phys", "fixed")}
                                 for s in SEEDS},
        }

        # 预注册 §4：跨 seed 翻符号优先
        if pf["sign_flip"]:
            verdict = "REPLAY_MODE_UNRESOLVED"
        elif pf["all_positive"]:
            # P1 成立，再按 phys vs bonly 细分 P2 / P3
            verdict = "SOURCE_DATA_VALUABLE" if pb["all_positive"] \
                else "SOURCE_DATA_NEUTRAL_OR_NEGATIVE"
        elif abs(pf["mean"]) < UBR_SE:
            verdict = "ENTRY_AMPLIFICATION_REFUTED"
        else:
            verdict = "ENTRY_AMPLIFICATION_REFUTED" if pf["all_negative"] \
                else "REPLAY_MODE_UNRESOLVED"
        report["verdict"] = verdict
        report["p1_supported"] = bool(pf["all_positive"])

    text = json.dumps(report, indent=2, ensure_ascii=False)
    print(text)
    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text + "\n")
    return 0 if report["verdict"] not in ("INCOMPLETE", "ENGINEERING_GATE_FAILED") else 1


if __name__ == "__main__":
    sys.exit(main())
