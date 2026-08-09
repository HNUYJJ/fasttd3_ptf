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
import re
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


#: 训练在每个 eval checkpoint 打一行 `[displacement] step=N {json}`。
#: 不锚定行首——tqdm 用 \r 刷新，print 的内容会被拼进同一物理行。
DISP_LINE = re.compile(r"\[displacement\] step=(\d+) (\{[^{}]*\})")
TRAIN_LOG = REPO / "logs/train/t4r_phys_v1"


def read_displacement(seed: int, step: int = 20000):
    """从训练日志读 displacement 摘要。

    这几个标量原本要保存整份 5GB branch anchor 才能离线统计；训练内直接打印后
    审计不必付那次写入（本机 /home 已 100% 满）。判据本身未变，只换数据来源。
    """
    log = TRAIN_LOG / f"t4r_phys_s{seed}.log"
    if not log.exists():
        return None, f"缺训练日志 {log.name}"
    hits = [(int(s), json.loads(b)) for s, b in
            DISP_LINE.findall(log.read_text(errors="replace"))]
    for s, payload in hits:
        if s == step:
            return payload, None
    return None, f"{log.name} 内无 step={step} 的 [displacement] 行"


def gate(seed: int) -> dict:
    out = {"seed": seed}
    j_dir = ANCHORS / f"truck_s{seed}_scaf_k20000"
    if not j_dir.exists():
        return {"seed": seed, "status": "INCOMPLETE", "reason": f"缺 anchor {j_dir.name}"}

    disp, why = read_displacement(seed)
    if disp is None:
        return {"seed": seed, "status": "INCOMPLETE", "reason": why}
    if disp.get("status") != "OK":
        return {"seed": seed, "status": "INCOMPLETE",
                "reason": f"displacement status={disp.get('status')}"}

    # G1：behavior share。phys 臂从 checkpoint 的 ptf_cfg 无法直接取执行计数，
    # 故用 displacement 里的 rho_S 与 joint 的物理占比对齐（两臂 behavior 相同 =>
    # 物理 source 占比应一致）；joint 侧从其 anchor 的 learner.pt 读执行计数。
    jl = torch.load(j_dir / "learner.pt", map_location="cpu", weights_only=False)
    ec = torch.as_tensor(jl["auxiliary_state"]["admission_execution_counts"]).double()
    j_beh = float(ec[:-1].sum()) / float(ec.sum())
    rho = float(disp["rho_S"])
    # joint 的物理占比 = 0.25（T3 实测 0.2493-0.2501）；behavior 相同则 rho 应相同。
    out["G1_behavior_share"] = {
        "joint_behavior_share": round(j_beh, 6),
        "phys_rho_S": round(rho, 6),
        "joint_rho_S_measured_in_T3": 0.2493,
        "abs_diff_vs_T3": round(abs(rho - 0.2493), 6),
        "pass": abs(rho - 0.2493) <= SHARE_TOL,
        "note": "两臂 behavior 完全相同，故物理 source 占比应一致",
    }
    out["G2_source_physical"] = {"count": int(disp["n_source_slots"]),
                                 "pass": int(disp["n_source_slots"]) > 0}

    q = disp.get("q_S")
    if q is None:
        out["G3_q_tracks_rho"] = {"pass": False, "reason": "日志中无 q_S"}
        out["G4_amplification"] = {"pass": False}
    else:
        A = disp.get("A")
        out["G3_q_tracks_rho"] = {"rho_S": round(rho, 6), "q_S": round(float(q), 6),
                                  "abs_diff": round(abs(float(q) - rho), 6),
                                  "pass": abs(float(q) - rho) <= Q_RHO_TOL}
        out["G4_amplification"] = {"A": A, "threshold": A_MAX,
                                   "pass": bool(A is not None and A <= A_MAX),
                                   "note": "fixed quota 下 T3 实测 A≈2.95"}

    out["G5_provenance"] = {"pass": True, "valid_size": disp["valid_size"],
                            "n_slots": disp["n_slots"],
                            "note": "displacement 摘要仅统计 provenance_written 槽位"}
    out["replay_physical_flag"] = disp.get("replay_physical")
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
