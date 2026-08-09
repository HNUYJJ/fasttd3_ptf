#!/usr/bin/env python3
"""Frozen N1 engineering audit and learner-seed sign adjudication."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import torch


REPO = Path(__file__).resolve().parents[2]
SEEDS = (4, 5, 6, 7, 8)
ARMS = ("s", "ff", "fp", "lp")
EVAL_ROOT = REPO / "docs/data/n1_non_displacing_transfer_v1/source_free_eval"
LOG_ROOT = REPO / "logs/train/n1_non_displacing_transfer_v1"
PREREG = "docs/experiments/n1_non_displacing_transfer_prereg_20260809.md"
T90_DF4 = 2.1318
DISP_LINE = re.compile(r"\[displacement\] step=(\d+) (\{[^{}]*\})")


def _finite(value: object) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _eval_path(arm: str, seed: int) -> Path:
    return EVAL_ROOT / f"truck_{arm}_s{seed}_step20000.json"


def _load_eval(arm: str, seed: int) -> tuple[dict | None, str | None]:
    path = _eval_path(arm, seed)
    if not path.exists():
        return None, f"missing evaluation {path.relative_to(REPO)}"
    try:
        payload = json.loads(path.read_text())
    except Exception as exc:  # malformed evidence is engineering-invalid
        return None, f"cannot parse {path.name}: {exc}"
    protocol = payload.get("protocol") or {}
    checkpoint = payload.get("checkpoint") or {}
    aggregate = payload.get("aggregate") or {}
    errors = []
    if payload.get("env_name") != "h1hand-truck-v0":
        errors.append("wrong env")
    if protocol.get("deterministic") is not True:
        errors.append("not deterministic")
    if "structural" not in str(protocol.get("source_free", "")):
        errors.append("not structurally source-free")
    if len(payload.get("episodes") or []) != 128:
        errors.append("episode count is not 128")
    if aggregate.get("episode_count") != 128:
        errors.append("aggregate episode_count is not 128")
    if checkpoint.get("identity_checked") is not True:
        errors.append("checkpoint identity not checked")
    if checkpoint.get("global_step") != 20000:
        errors.append("wrong checkpoint step")
    if not _finite(aggregate.get("return_mean")):
        errors.append("non-finite return_mean")
    if errors:
        return None, f"{path.name}: " + "; ".join(errors)
    return payload, None


def _load_checkpoint(eval_payload: dict) -> tuple[dict | None, str | None]:
    path = Path(eval_payload["checkpoint"]["path"])
    if not path.exists():
        return None, f"missing checkpoint {path}"
    try:
        return torch.load(path, map_location="cpu", weights_only=False), None
    except Exception as exc:
        return None, f"cannot load checkpoint {path.name}: {exc}"


def _load_displacement(arm: str, seed: int) -> tuple[dict | None, str | None]:
    log = LOG_ROOT / f"n1_{arm}_truck_s{seed}.log"
    if not log.exists():
        return None, f"missing train log {log.relative_to(REPO)}"
    try:
        hits = [(int(step), json.loads(body)) for step, body in
                DISP_LINE.findall(log.read_text(errors="replace"))]
    except Exception as exc:
        return None, f"cannot parse displacement from {log.name}: {exc}"
    rows = [payload for step, payload in hits if step == 20000]
    if len(rows) != 1:
        return None, f"{log.name}: expected one step-20000 displacement, got {len(rows)}"
    return rows[0], None


def _source_share(counts: list[int | float]) -> float:
    total = float(sum(counts))
    return float(sum(counts[:-1])) / total if total else float("nan")


def audit_cell(arm: str, seed: int) -> dict:
    out: dict = {"arm": arm, "seed": seed, "checks": {}}
    evaluation, error = _load_eval(arm, seed)
    if error:
        return {**out, "valid": False, "error": error}
    checkpoint, error = _load_checkpoint(evaluation)
    if error:
        return {**out, "valid": False, "error": error}
    displacement, error = _load_displacement(arm, seed)
    if error:
        return {**out, "valid": False, "error": error}

    args = checkpoint.get("args") or {}
    cfg = checkpoint.get("ptf_cfg") or {}
    audit = checkpoint.get("admission_audit") or {}
    expected_mode = "none" if arm == "s" else "all"
    expected_replay = "physical" if arm in ("fp", "lp") else "shared"
    expected_groups = ["legs_torso"] if arm == "lp" else ["legs_torso", "arms"]
    expected_anchor = (
        f"artifacts/n1_non_displacing_transfer_v1/anchors/truck_s{seed}_k10000"
    )
    frozen = {
        "env_name": args.get("env_name") == "h1hand-truck-v0",
        "seed": args.get("seed") == seed,
        "global_step": checkpoint.get("global_step") == 20000,
        "num_envs": args.get("num_envs") == 128,
        "batch_size": args.get("batch_size") == 32768,
        "buffer_size": args.get("buffer_size") == 51200,
        "num_updates": args.get("num_updates") == 2,
        "total_timesteps": args.get("total_timesteps") == 100000,
        "bank": cfg.get("source_bank")
        == "configs/source_banks/h1hand_hurdle4_wfix_truck.yaml",
        "mcg_groups": cfg.get("mcg_groups") == ["legs_torso", "arms"],
        "behavior_source_groups": cfg.get("mcg_behavior_source_groups")
        == expected_groups,
        "warmup_mode": cfg.get("mcg_warmup_mode") == "admission_bootstrap",
        "ablation": cfg.get("mcg_ablation") == "bootstrap_only",
        "horizon": cfg.get("mcg_warmup_min_steps") == 25,
        "admission_mode": cfg.get("admission_mode") == expected_mode,
        "replay_mode": cfg.get("admission_replay_mode") == expected_replay,
        "anchor": str(cfg.get("anchor_resume")) == expected_anchor,
        "resume_noise_seed": cfg.get("resume_noise_seed") == 95000 + seed,
    }
    out["checks"]["frozen_configuration"] = frozen

    execution = [int(x) for x in audit.get("execution_counts") or []]
    critic = [int(x) for x in audit.get("critic_sample_counts") or []]
    if len(execution) != 5 or len(critic) != 5:
        return {**out, "valid": False, "error": "missing five-stratum audit counts"}
    execution_share = _source_share(execution)
    critic_share = _source_share(critic)
    if arm == "s":
        dose_ok = sum(execution[:-1]) == 0 and sum(critic[:-1]) == 0
    else:
        dose_ok = 0.45 <= execution_share <= 0.55
        if arm == "ff":
            dose_ok = dose_ok and 0.45 <= critic_share <= 0.55
        else:
            dose_ok = dose_ok and 0.13 <= critic_share <= 0.18
    out["checks"]["dose"] = {
        "execution_source_share": execution_share,
        "critic_source_share_cumulative": critic_share,
        "pass": dose_ok,
    }

    expected_physical = arm in ("fp", "lp")
    if arm == "s":
        # Exact abstention releases source authority immediately. Depending on
        # snapshot timing, the empty-source arm may report the quota phase or
        # the post-release physical phase. Phase naming is not S's treatment
        # invariant; strict zero source execution/storage/sampling is.
        replay_ok = (
            bool(audit.get("replay_physical", False)) is False
            and audit.get("sampling_phase")
            in ("authority_quota", "physical_allowed")
        )
    else:
        replay_ok = (
            bool(audit.get("replay_physical", False)) is expected_physical
            and audit.get("sampling_phase")
            == ("physical_allowed" if expected_physical else "authority_quota")
        )
    out["checks"]["replay_semantics"] = {
        "replay_physical": audit.get("replay_physical"),
        "sampling_phase": audit.get("sampling_phase"),
        "pass": replay_ok,
    }

    group_rho = displacement.get("rho_endpoint_by_group")
    if not isinstance(group_rho, list) or len(group_rho) != 2:
        group_ok = False
    elif arm == "s":
        group_ok = group_rho == [0.0, 0.0]
    elif arm == "lp":
        group_ok = float(group_rho[0]) > 0.0 and float(group_rho[1]) == 0.0
    else:
        group_ok = (
            float(group_rho[0]) > 0.0
            and float(group_rho[1]) > 0.0
            and abs(float(group_rho[0]) - float(group_rho[1])) <= 1e-6
        )
    out["checks"]["group_provenance"] = {
        "rho_endpoint_by_group": group_rho,
        "pass": group_ok,
    }
    out["checks"]["evaluation"] = {"pass": True, "episode_count": 128}
    out["return_mean"] = float(evaluation["aggregate"]["return_mean"])
    out["valid"] = (
        all(frozen.values())
        and dose_ok
        and replay_ok
        and group_ok
        and displacement.get("status") == "OK"
        and displacement.get("buffer_not_wrapped") is True
    )
    return out


def summarize(values: list[float]) -> dict:
    n = len(values)
    mean = sum(values) / n
    sd = math.sqrt(sum((x - mean) ** 2 for x in values) / (n - 1))
    se = sd / math.sqrt(n)
    positive = sum(x > 0 for x in values)
    if mean > 0 and positive >= 4:
        classification = "DIRECTIONAL_SUPPORT"
    elif mean <= 0 or positive <= 2:
        classification = "DIRECTIONAL_REFUTATION"
    else:
        classification = "UNRESOLVED"
    return {
        "per_seed": [round(x, 6) for x in values],
        "mean": round(mean, 6),
        "sd_learner": round(sd, 6),
        "se_learner": round(se, 6),
        "ci90": [round(mean - T90_DF4 * se, 6), round(mean + T90_DF4 * se, 6)],
        "positive_seeds": positive,
        "classification": classification,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        default="docs/data/n1_non_displacing_transfer_v1/n1_verdict.json",
    )
    args = parser.parse_args()

    cells = [audit_cell(arm, seed) for seed in SEEDS for arm in ARMS]
    report: dict = {"prereg": PREREG, "seeds": list(SEEDS), "cells": cells}
    if not all(cell.get("valid") is True for cell in cells):
        report["verdict"] = "ENGINEERING_INVALID"
        report["invalid_cells"] = [
            {"arm": cell["arm"], "seed": cell["seed"], "error": cell.get("error")}
            for cell in cells if cell.get("valid") is not True
        ]
        exit_code = 1
    else:
        returns = {(cell["arm"], cell["seed"]): cell["return_mean"] for cell in cells}
        contrasts = {
            "H_R_FP_minus_FF": summarize(
                [returns[("fp", seed)] - returns[("ff", seed)] for seed in SEEDS]
            ),
            "H_A_LP_minus_FP": summarize(
                [returns[("lp", seed)] - returns[("fp", seed)] for seed in SEEDS]
            ),
            "H_REC_LP_minus_S": summarize(
                [returns[("lp", seed)] - returns[("s", seed)] for seed in SEEDS]
            ),
        }
        report["returns"] = {
            f"s{seed}": {arm: round(returns[(arm, seed)], 6) for arm in ARMS}
            for seed in SEEDS
        }
        report["contrasts"] = contrasts
        classes = [item["classification"] for item in contrasts.values()]
        if all(value == "DIRECTIONAL_SUPPORT" for value in classes):
            report["verdict"] = "NDT_DIRECTIONALLY_SUPPORTED"
        elif any(value == "DIRECTIONAL_REFUTATION" for value in classes):
            report["verdict"] = "NDT_NOT_SUPPORTED"
        else:
            report["verdict"] = "NDT_UNRESOLVED"
        exit_code = 0

    text = json.dumps(report, indent=2, ensure_ascii=False)
    print(text)
    output = REPO / args.out
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.write_text(text + "\n")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
