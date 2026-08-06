#!/usr/bin/env python3
"""Analyze the frozen three-seed Hurdle RBO confirmation matrix."""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from pathlib import Path


ARMS = ("scratch", "walk", "run")
SEEDS = (1, 2, 3)
EXPECTED_STEPS = (5000, 10000, 15000, 20000, 25000)
EVAL_RE = re.compile(r"\[eval\]\s+step=(\d+)\s+return=([-+0-9.eE]+)")
T_95_DF2 = 2.9199855803537256  # two-sided 90% CI


def _parse_online(path: Path) -> dict[int, float]:
    points: dict[int, float] = {}
    for line in path.read_text(errors="replace").splitlines():
        match = EVAL_RE.search(line)
        if match:
            step = int(match.group(1))
            if step in EXPECTED_STEPS:
                points[step] = float(match.group(2))
    missing = [step for step in EXPECTED_STEPS if step not in points]
    if missing:
        raise ValueError(f"{path}: missing eval steps {missing}")
    return points


def _nauc(points: dict[int, float]) -> float:
    area = sum(
        (right - left) * (points[left] + points[right]) / 2.0
        for left, right in zip(EXPECTED_STEPS[:-1], EXPECTED_STEPS[1:])
    )
    return area / (EXPECTED_STEPS[-1] - EXPECTED_STEPS[0])


def _ci90(values: list[float]) -> list[float]:
    mean = statistics.mean(values)
    half = T_95_DF2 * statistics.stdev(values) / math.sqrt(len(values))
    return [mean - half, mean + half]


def _root_for(seed: int, seed1_root: Path, confirm_root: Path) -> Path:
    return seed1_root if seed == 1 else confirm_root


def _record(arm: str, seed: int, root: Path) -> dict:
    meta_path = root / f"{arm}_s{seed}.meta.json"
    eval_path = root / "source_free_eval" / f"{arm}_s{seed}_step30000.json"
    meta = json.loads(meta_path.read_text())
    if int(meta.get("exit_code", -1)) != 0:
        raise ValueError(f"{arm}/s{seed}: exit_code={meta.get('exit_code')}")
    frozen = json.loads(eval_path.read_text())
    if int(frozen["checkpoint"]["global_step"]) != 30000:
        raise ValueError(f"{arm}/s{seed}: endpoint checkpoint is not step 30000")
    online = _parse_online(root / f"{arm}_s{seed}.log")
    return {
        "online_eval_return": {str(k): online[k] for k in EXPECTED_STEPS},
        "nauc_5k_25k": _nauc(online),
        "frozen_30k_return_mean": float(frozen["aggregate"]["return_mean"]),
        "frozen_30k_progress_max_dx_mean": float(
            frozen["aggregate"]["progress_max_dx_mean"]
        ),
        "checkpoint": meta["completed_step_checkpoint"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed1-root", required=True)
    parser.add_argument("--confirm-root", required=True)
    parser.add_argument("--out-prefix", required=True)
    args = parser.parse_args()
    seed1_root = Path(args.seed1_root)
    confirm_root = Path(args.confirm_root)
    out_prefix = Path(args.out_prefix)

    records = {
        arm: {
            str(seed): _record(
                arm, seed, _root_for(seed, seed1_root, confirm_root)
            )
            for seed in SEEDS
        }
        for arm in ARMS
    }
    metrics = ("nauc_5k_25k", "frozen_30k_return_mean")
    summary: dict[str, dict] = {}
    for arm in ARMS:
        summary[arm] = {}
        for metric in metrics:
            values = [records[arm][str(seed)][metric] for seed in SEEDS]
            summary[arm][metric] = {
                "values": values,
                "mean": statistics.mean(values),
                "sd": statistics.stdev(values),
                "ci90": _ci90(values),
            }

    paired: dict[str, dict] = {}
    for left, right in (("walk", "scratch"), ("run", "scratch"), ("run", "walk")):
        key = f"{left}_minus_{right}"
        paired[key] = {}
        for metric in metrics:
            values = [
                records[left][str(seed)][metric]
                - records[right][str(seed)][metric]
                for seed in SEEDS
            ]
            paired[key][metric] = {
                "values": values,
                "mean": statistics.mean(values),
                "sd": statistics.stdev(values),
                "ci90": _ci90(values),
            }

    per_seed_full_order = {
        str(seed): {
            metric: (
                records["run"][str(seed)][metric]
                > records["walk"][str(seed)][metric]
                > records["scratch"][str(seed)][metric]
            )
            for metric in metrics
        }
        for seed in SEEDS
    }
    all_full = all(
        passed
        for seed_values in per_seed_full_order.values()
        for passed in seed_values.values()
    )
    aggregate_full = all(
        summary["run"][metric]["mean"]
        > summary["walk"][metric]["mean"]
        > summary["scratch"][metric]["mean"]
        for metric in metrics
    )
    if all_full:
        decision = "FULL_ORDER_REPLICATED"
    elif aggregate_full:
        decision = "AGGREGATE_ORDER_ONLY"
    else:
        decision = "ORDER_NOT_REPLICATED"

    payload = {
        "schema_version": 1,
        "experiment": "hurdle_equal_dose_source_calibration_multiseed_v1",
        "claim_scope": "three_seed_rbo_intervention_labels",
        "records": records,
        "summary": summary,
        "paired_differences": paired,
        "per_seed_full_order": per_seed_full_order,
        "decision": decision,
        "classic_ptf_multisource_selector_preference": "walk",
        "comparison_boundary": (
            "The classic PTF observation is a multisource selector preference, "
            "not an independently validated walk-only teacher-value label."
        ),
    }

    json_path = out_prefix.with_suffix(".json")
    md_path = out_prefix.with_suffix(".md")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    if json_path.exists() or md_path.exists():
        raise FileExistsError(f"refusing to overwrite {json_path} or {md_path}")
    json_path.write_text(json.dumps(payload, indent=2) + "\n")

    lines = [
        "# Hurdle等剂量单源RBO：3-seed确认结果",
        "",
        "| arm | nAUC seeds 1/2/3 | nAUC mean | 30k source-free seeds 1/2/3 | endpoint mean |",
        "|---|---|---:|---|---:|",
    ]
    for arm in ARMS:
        a = summary[arm]["nauc_5k_25k"]
        e = summary[arm]["frozen_30k_return_mean"]
        lines.append(
            f"| {arm} | {' / '.join(f'{x:.3f}' for x in a['values'])} | "
            f"{a['mean']:.3f} | "
            f"{' / '.join(f'{x:.3f}' for x in e['values'])} | "
            f"{e['mean']:.3f} |"
        )
    lines.extend(
        [
            "",
            f"- 各seed双视角完整排序：`{per_seed_full_order}`",
            f"- 裁决：`{decision}`",
            "",
            "边界：这里得到的是RBO完整干预包的三seed标签，不是已经部署的迁移性指标。",
            "classic PTF中的walk仅是多教师调度器的观测选择偏好，不是已验证的",
            "walk-only最佳教师标签。",
            "",
        ]
    )
    md_path.write_text("\n".join(lines))
    print(json.dumps(payload, indent=2))
    print(f"[analysis] wrote {json_path} and {md_path}")


if __name__ == "__main__":
    main()
