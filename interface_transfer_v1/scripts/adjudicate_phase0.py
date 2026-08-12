#!/usr/bin/env python3
"""Phase 0 裁决器：判据逐条对齐 `docs/phase0_prereg_v2_20260812.md` §3。

分支（原文）：
1. INTERFACE_VIABLE      3/3 seed 为正且均值差 > 0
2. INTERFACE_UNRESOLVED  2/3 为正
3. INCONCLUSIVE_BUDGET   1、2 均不满足，但两臂在 80k→100k 的 eval return 均单调不减
4. INTERFACE_NOT_VIABLE  1、2 不满足，且至少一臂曲线在最后 20% 已平台或下降

数据不全 → INCOMPLETE 且非零退出（§4）。
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "data" / "phase0_eval"
SEEDS = (1, 2, 3)


def load(arm: str, seed: int):
    p = EVAL / f"h1hand-push-v0__p0_{arm}_s{seed}__{seed}_final.json"
    if not p.exists():
        return None, f"缺 {p.name}"
    d = json.loads(p.read_text())
    if d.get("n_act") != (3 if arm == "iface" else 61):
        return None, f"{p.name} 动作维度错配 n_act={d.get('n_act')}"
    return d, None


def tail_monotone(log_path: Path) -> str:
    """读训练日志判断最后 20% 是否单调不减。

    train_interface.py 只把 eval return 上报 wandb、不落 stdout，
    故此处无法从日志重建曲线；返回 'UNAVAILABLE' 由调用方保守处理。
    """
    return "UNAVAILABLE"


def main() -> int:
    rows, missing = {}, []
    for arm in ("flat", "iface"):
        for s in SEEDS:
            d, why = load(arm, s)
            if d is None:
                missing.append(why)
            else:
                rows[(arm, s)] = d

    report = {"prereg": "docs/phase0_prereg_v2_20260812.md",
              "scope": "工程与方法 positive control，不是迁移结果"}

    if missing:
        report["verdict"] = "INCOMPLETE"
        report["missing"] = missing
        print(json.dumps(report, ensure_ascii=False, indent=1))
        return 1

    diffs, per_seed = [], {}
    for s in SEEDS:
        f = rows[("flat", s)]["aggregate"]
        i = rows[("iface", s)]["aggregate"]
        d = i["return_mean"] - f["return_mean"]
        diffs.append(d)
        per_seed[f"s{s}"] = {
            "flat_return": round(f["return_mean"], 3),
            "iface_return": round(i["return_mean"], 3),
            "diff": round(d, 3),
            "flat_success": f["success_rate"], "iface_success": i["success_rate"],
            "flat_hand_dist": round(f["hand_dist_mean"], 4),
            "iface_hand_dist": round(i["hand_dist_mean"], 4),
            "flat_target_dist": round(f["target_dist_mean"], 4),
            "iface_target_dist": round(i["target_dist_mean"], 4),
        }

    n = len(diffs)
    mean = sum(diffs) / n
    sd = math.sqrt(sum((x - mean) ** 2 for x in diffs) / (n - 1))
    se = sd / math.sqrt(n)
    pos = sum(1 for x in diffs if x > 0)

    report["contrast_iface_minus_flat"] = {
        "per_seed": [round(x, 3) for x in diffs],
        "mean": round(mean, 3), "sd_learner": round(sd, 3),
        "se_learner": round(se, 3),
        "t_reported_only": round(mean / se, 3) if se > 0 else None,
        "positive_seeds": pos,
        "note": "n=3，只报方向与 learner 间 SD，不做显著性声称（M16/M24）",
    }
    report["per_seed"] = per_seed
    report["curve_check"] = {
        "status": tail_monotone(ROOT / "logs" / "phase0"),
        "note": "eval return 只上报 wandb 未落 stdout，无法离线重建曲线",
    }

    if pos == 3 and mean > 0:
        v = "INTERFACE_VIABLE"
    elif pos == 2:
        v = "INTERFACE_UNRESOLVED"
    else:
        # 判据 3 需要曲线证据；曲线不可得时不得擅自判 NOT_VIABLE（§4：
        # 缺失不能落进实质裁决分支）
        v = "INCONCLUSIVE_BUDGET_CURVE_UNAVAILABLE"
    report["verdict"] = v

    out = ROOT / "data" / "phase0_verdict.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print(json.dumps(report, ensure_ascii=False, indent=1))
    return 0 if v == "INTERFACE_VIABLE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
