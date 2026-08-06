#!/usr/bin/env python3
"""Analyze the seed-1 equal-dose source-specific RBO calibration.

The script deliberately reports a feasibility screen rather than a statistical
source-separation claim. It combines:
  * in-training deterministic evaluations at 5k..25k;
  * the frozen 32-episode source-free panel at completed step 30k;
  * admission behavior/replay treatment shares from the 30k checkpoint.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import re
from pathlib import Path

import torch


ARMS = ("scratch", "stand", "walk", "run")
EXPECTED_STEPS = (5000, 10000, 15000, 20000, 25000)
EVAL_RE = re.compile(r"\[eval\]\s+step=(\d+)\s+return=([-+0-9.eE]+)")


def _normalized_auc(points: dict[int, float]) -> float:
    xs = list(EXPECTED_STEPS)
    missing = [step for step in xs if step not in points]
    if missing:
        raise ValueError(f"missing online eval steps: {missing}")
    area = 0.0
    for left, right in zip(xs[:-1], xs[1:]):
        area += (right - left) * (points[left] + points[right]) / 2.0
    return area / float(xs[-1] - xs[0])


def _parse_online(path: Path) -> dict[int, float]:
    points: dict[int, float] = {}
    for line in path.read_text(errors="replace").splitlines():
        # tqdm writes the evaluation marker after a carriage-return progress
        # fragment, so it is not guaranteed to begin the physical log line.
        match = EVAL_RE.search(line)
        if match:
            points[int(match.group(1))] = float(match.group(2))
    extra = sorted(set(points) - set(EXPECTED_STEPS))
    if extra:
        # Extra diagnostics are retained in the log but excluded by protocol.
        points = {step: value for step, value in points.items() if step in EXPECTED_STEPS}
    return points


def _safe_share(values: list[int] | None) -> float | None:
    if not values:
        return None
    total = sum(int(value) for value in values)
    if total <= 0:
        return None
    return float(sum(int(value) for value in values[:-1]) / total)


def _checkpoint_audit(path: Path) -> dict:
    state = torch.load(path, map_location="cpu", weights_only=False)
    try:
        audit = state.get("admission_audit")
        if audit is None:
            return {
                "global_step": int(state["global_step"]),
                "candidate_masses": None,
                "source_behavior_share": None,
                "source_critic_replay_share": None,
            }
        return {
            "global_step": int(state["global_step"]),
            "candidate_masses": audit.get("candidate_masses"),
            "source_behavior_share": _safe_share(audit.get("execution_counts")),
            "source_critic_replay_share": _safe_share(
                audit.get("critic_sample_counts")
            ),
            "execution_counts": audit.get("execution_counts"),
            "critic_sample_counts": audit.get("critic_sample_counts"),
            "main_buffer_counts": audit.get("main_buffer_counts"),
        }
    finally:
        del state
        gc.collect()


def _fmt(value: float | None) -> str:
    return "NA" if value is None or not math.isfinite(value) else f"{value:.3f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--out-prefix", default=None)
    parser.add_argument(
        "--experiment", default="hurdle_equal_dose_source_calibration_v1"
    )
    parser.add_argument(
        "--title", default="Hurdle等剂量单source RBO标定：seed-1结果"
    )
    parser.add_argument("--selector-preference", default=None)
    parser.add_argument(
        "--decision-mode", choices=("positive", "negative"), default="positive"
    )
    args = parser.parse_args()

    root = Path(args.run_root)
    if not root.is_dir():
        raise FileNotFoundError(root)
    out_prefix = (
        Path(args.out_prefix)
        if args.out_prefix
        else root / f"{args.experiment}_results"
    )

    records: dict[str, dict] = {}
    for arm in ARMS:
        meta_path = root / f"{arm}_s1.meta.json"
        eval_path = root / "source_free_eval" / f"{arm}_s1_step30000.json"
        if not meta_path.is_file():
            raise FileNotFoundError(meta_path)
        if not eval_path.is_file():
            raise FileNotFoundError(eval_path)
        meta = json.loads(meta_path.read_text())
        if int(meta.get("exit_code", -1)) != 0:
            raise ValueError(f"{arm}: training exit_code={meta.get('exit_code')}")
        log_path = root / f"{arm}_s1.log"
        checkpoint = Path(meta["completed_step_checkpoint"])
        online = _parse_online(log_path)
        frozen = json.loads(eval_path.read_text())
        if int(frozen["checkpoint"]["global_step"]) != 30000:
            raise ValueError(f"{arm}: frozen evaluator did not use step 30000")
        audit = _checkpoint_audit(checkpoint)
        records[arm] = {
            "online_eval_return": {str(k): online[k] for k in sorted(online)},
            "nauc_5k_25k": _normalized_auc(online),
            "frozen_30k_return_mean": float(
                frozen["aggregate"]["return_mean"]
            ),
            "frozen_30k_return_sd_episode": float(
                frozen["aggregate"]["return_std"]
            ),
            "frozen_30k_progress_max_dx_mean": float(
                frozen["aggregate"]["progress_max_dx_mean"]
            ),
            "checkpoint": str(checkpoint),
            "treatment_audit": audit,
        }

    scratch_auc = records["scratch"]["nauc_5k_25k"]
    scratch_endpoint = records["scratch"]["frozen_30k_return_mean"]
    for arm in ("stand", "walk", "run"):
        records[arm]["delta_vs_scratch"] = {
            "nauc_5k_25k": records[arm]["nauc_5k_25k"] - scratch_auc,
            "frozen_30k_return_mean": (
                records[arm]["frozen_30k_return_mean"] - scratch_endpoint
            ),
        }

    source_arms = ("stand", "walk", "run")
    auc_order = sorted(
        source_arms, key=lambda arm: records[arm]["nauc_5k_25k"], reverse=True
    )
    endpoint_order = sorted(
        source_arms,
        key=lambda arm: records[arm]["frozen_30k_return_mean"],
        reverse=True,
    )
    positive_both = [
        arm
        for arm in source_arms
        if records[arm]["delta_vs_scratch"]["nauc_5k_25k"] > 0
        and records[arm]["delta_vs_scratch"]["frozen_30k_return_mean"] > 0
    ]
    nonpositive = [arm for arm in source_arms if arm not in positive_both]
    if args.decision_mode == "negative":
        decision = (
            "NEGATIVE_SOURCE_FOUND"
            if nonpositive
            else "NO_INDIVIDUAL_NEGATIVE_SOURCE"
        )
    elif not positive_both:
        decision = "STOP_NO_SOURCE_POSITIVE_IN_BOTH_VIEWS"
    elif auc_order[0] == endpoint_order[0]:
        decision = "ADVANCE_PROVISIONAL_TOP_AND_RUNNER_UP"
    else:
        decision = "AMBIGUOUS_ACCELERATION_ENDPOINT_ORDER"

    payload = {
        "schema_version": 1,
        "experiment": args.experiment,
        "claim_scope": "seed1_feasibility_screen_only",
        "records": records,
        "source_auc_order": auc_order,
        "source_endpoint_order": endpoint_order,
        "sources_positive_vs_scratch_in_both_views": positive_both,
        "sources_nonpositive_in_at_least_one_view": nonpositive,
        "decision": decision,
    }
    if args.selector_preference is not None:
        payload.update(
            {
                "classic_ptf_multisource_selector_preference": (
                    args.selector_preference
                ),
                "selector_preference_matches_rbo_auc_top": (
                    auc_order[0] == args.selector_preference
                ),
                "selector_preference_matches_rbo_endpoint_top": (
                    endpoint_order[0] == args.selector_preference
                ),
            }
        )
    json_path = out_prefix.with_suffix(".json")
    md_path = out_prefix.with_suffix(".md")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    if json_path.exists() or md_path.exists():
        raise FileExistsError(f"refusing to overwrite {json_path} or {md_path}")
    json_path.write_text(json.dumps(payload, indent=2) + "\n")

    lines = [
        f"# {args.title}",
        "",
        "> 本结果只作feasibility screen，不构成source可分性的统计证明。",
        "",
        "| arm | 5k–25k nAUC | ΔAUC vs scratch | 30k source-free | Δ30k vs scratch | behavior source share | critic source share |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        record = records[arm]
        delta = record.get("delta_vs_scratch") or {}
        audit = record["treatment_audit"]
        lines.append(
            f"| {arm} | {record['nauc_5k_25k']:.3f} | "
            f"{_fmt(delta.get('nauc_5k_25k'))} | "
            f"{record['frozen_30k_return_mean']:.3f} | "
            f"{_fmt(delta.get('frozen_30k_return_mean'))} | "
            f"{_fmt(audit.get('source_behavior_share'))} | "
            f"{_fmt(audit.get('source_critic_replay_share'))} |"
        )
    lines.extend(
        [
            "",
            f"- AUC排序：`{' > '.join(auc_order)}`",
            f"- 30k终点排序：`{' > '.join(endpoint_order)}`",
            f"- 两个视角均优于scratch：`{positive_both}`",
            f"- 至少一个视角不优于scratch：`{nonpositive}`",
            f"- 自动裁决：`{decision}`",
            "",
            "语义边界：这里测量的是完整RBO干预包效应，包括轨迹内容、实际执行长度、",
            "occupancy变化和replay暴露；它不是纯粹的数据质量分数，也还不是在线迁移性指标。",
            "",
        ]
    )
    if args.selector_preference is not None:
        lines[-1:-1] = [
            f"classic PTF多教师实验中观测到的{args.selector_preference}选择偏好"
            "与RBO AUC/终点top是否一致："
            f"`{payload['selector_preference_matches_rbo_auc_top']}` / "
            f"`{payload['selector_preference_matches_rbo_endpoint_top']}`",
            "该偏好不是独立单source因果标签。",
        ]
    md_path.write_text("\n".join(lines))
    print(json.dumps(payload, indent=2))
    print(f"[analysis] wrote {json_path} and {md_path}")


if __name__ == "__main__":
    main()
