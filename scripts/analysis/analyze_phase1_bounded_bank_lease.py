#!/usr/bin/env python3
"""Frozen Phase-1 analysis; run only after the 9-run matrix completes."""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from pathlib import Path

import torch

EVAL_RE = re.compile(rb"\[eval\] step=(\d+) return=([-+0-9.eE]+)")
SEEDS = (1, 2, 3)
MAIN_STEPS = tuple(range(35_000, 80_001, 5_000))
PERSIST_STEPS = (80_000, 85_000, 90_000, 95_000)
FULL_EVAL_STEPS = tuple(range(5_000, 95_001, 5_000))
CKPT_STEPS = (30_000, 60_000, 80_000, 90_000, 100_000)
T90_DF2 = 1.8856
STAMP = "20260719T130500Z"


def curve(path: Path) -> dict[int, float]:
    pairs = EVAL_RE.findall(path.read_bytes())
    out = {int(step): float(value) for step, value in pairs}
    if len(out) != len(pairs):
        raise ValueError(f"duplicate eval step: {path}")
    return out


def n_auc(values: dict[int, float], steps: tuple[int, ...]) -> float:
    missing = [step for step in steps if step not in values]
    if missing:
        raise ValueError(f"missing eval steps {missing}")
    area = sum(
        (values[a] + values[b]) * 0.5 * (b - a)
        for a, b in zip(steps, steps[1:])
    )
    return area / (steps[-1] - steps[0])


def assert_full_eval_grid(values: dict[int, float], path: Path) -> None:
    if tuple(sorted(values)) != FULL_EVAL_STEPS:
        raise ValueError(f"unexpected eval grid in {path}: {tuple(sorted(values))}")


def paired_stats(deltas: list[float], delta: float) -> dict:
    mean = statistics.mean(deltas)
    sd = statistics.stdev(deltas)
    radius = T90_DF2 * sd / math.sqrt(3)
    lower, upper = mean - radius, mean + radius
    if any(value > delta for value in deltas) and any(value < -delta for value in deltas):
        label = "HETEROGENEOUS"
    elif lower > delta:
        label = "IMPROVEMENT"
    elif upper < -delta:
        label = "HARM"
    elif lower >= -delta and upper <= delta:
        label = "EQUIVALENT"
    else:
        label = "UNCERTAIN"
    return {
        "per_seed": deltas,
        "mean": mean,
        "sample_sd": sd,
        "ci90": [lower, upper],
        "sesoi": delta,
        "classification": label,
    }


def new_log(root: Path, cell: str, seed: int) -> Path:
    return root / f"logs/train/phase1_bounded_bank_lease_{STAMP}/{cell}_s{seed}.log"


def basketball_retention_log(root: Path, seed: int) -> Path:
    return root / f"logs/train/adaptive_admission_v1_20260714T110054Z/basketball_static_s{seed}.log"


def checkpoint(root: Path, cell: str, seed: int, step: int) -> Path:
    task = "basketball" if cell.startswith("basketball") else "truck"
    exp = f"phase1_bounded_lease_formal_{cell}_s{seed}_{STAMP}"
    return root / "models" / f"h1hand-{task}-v0__{exp}__{seed}_{step}.pt"


def counts_share(values: list[int], n: int) -> float:
    total = sum(values)
    if total <= 0:
        raise ValueError("zero count denominator")
    return sum(values[:n]) / total


def audit_new_run(root: Path, cell: str, seed: int, epsilon: float, bands: dict) -> dict:
    payloads = {}
    for step in CKPT_STEPS:
        path = checkpoint(root, cell, seed, step)
        if not path.is_file():
            raise FileNotFoundError(path)
        value = torch.load(path, map_location="cpu", weights_only=False)
        if int(value["global_step"]) != step:
            raise ValueError(f"checkpoint step mismatch: {path}")
        payloads[step] = value
    a30 = payloads[30_000]["admission_audit"]
    n = len(a30["admitted_sources"])
    behavior_share = counts_share(a30["execution_counts"], n)
    critic_share = counts_share(a30["critic_sample_counts"], n)
    behavior_band = bands["behavior"]
    critic_band = bands["critic"]
    checks = {
        "behavior_dose": behavior_band[0] <= behavior_share <= behavior_band[1],
        "critic_dose": critic_band[0] <= critic_share <= critic_band[1],
    }
    rows = []
    if cell.endswith("hard_exit"):
        final_audit = payloads[100_000]["admission_audit"]
        revoke = [
            event for event in final_audit["policy_events"]
            if event.get("event") == "admission_policy"
            and not any(event["admitted_sources"])
        ][-1]
        decision = [
            event for event in final_audit["decision_history"]
            if event.get("step") == 30_000
        ][-1]
        base_critic = revoke["sample_counts_at_apply"]["critic"]
        base_execution = decision["execution_counts_at_apply"]
        for step in (60_000, 80_000, 90_000, 100_000):
            audit = payloads[step]["admission_audit"]
            source_exec = sum(
                audit["execution_counts"][i] - base_execution[i] for i in range(n)
            )
            source_critic = sum(
                audit["critic_sample_counts"][i] - base_critic[i] for i in range(n)
            )
            active = sum(audit["active_buffer_counts"][:n])
            physical = sum(audit["main_buffer_counts"][:n])
            row = {
                "step": step,
                "source_execution_increment": source_exec,
                "source_critic_increment": source_critic,
                "active_source_slots": active,
                "physical_source_slots": physical,
            }
            rows.append(row)
            checks[f"hard_exit_{step}"] = (
                source_exec == 0 and source_critic == 0 and active == 0
            )
    else:
        qs = {}
        for step, payload in payloads.items():
            audit = payload["admission_audit"]
            q = counts_share(audit["main_buffer_counts"], n)
            qs[step] = q
            checks[f"active_equals_main_{step}"] = (
                audit["active_buffer_counts"] == audit["main_buffer_counts"]
            )
        checks["retention_q_monotone"] = all(
            qs[a] + 1e-15 >= qs[b] for a, b in zip(CKPT_STEPS, CKPT_STEPS[1:])
        )
        checks["retention_q_90_100_zero"] = qs[90_000] == 0 and qs[100_000] == 0
        for a, b in zip(CKPT_STEPS, CKPT_STEPS[1:]):
            left = payloads[a]["admission_audit"]["critic_sample_counts"]
            right = payloads[b]["admission_audit"]["critic_sample_counts"]
            delta_counts = [right[i] - left[i] for i in range(n + 1)]
            r = counts_share(delta_counts, n)
            ok = qs[b] - epsilon <= r <= qs[a] + epsilon
            checks[f"retention_envelope_{a}_{b}"] = ok
            rows.append({"interval": [a, b], "q_start": qs[a], "q_end": qs[b], "r": r})
    return {
        "cell": cell,
        "seed": seed,
        "behavior_source_share_30k": behavior_share,
        "critic_source_share_30k": critic_share,
        "checks": checks,
        "rows": rows,
        "pass": all(checks.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    delta_payload = json.loads((root / "docs/data/p1_bounded_bank_lease/delta_frozen.json").read_text())
    gate_payload = json.loads((root / "docs/data/p1_bounded_bank_lease/gate_b_tolerance_frozen.json").read_text())
    deltas = delta_payload["deltas"]
    epsilon = float(gate_payload["epsilon"]["eps_frozen"])
    bands = {
        "basketball": {"behavior": [0.4696, 0.5110], "critic": [0.4773, 0.5174]},
        "truck": {"behavior": [0.4795, 0.5210], "critic": [0.4791, 0.5192]},
    }
    perf = {}
    scratch_descriptive = {}
    for task in ("basketball", "truck"):
        hard_cell = f"{task}_hard_exit"
        hard_paths = [new_log(root, hard_cell, seed) for seed in SEEDS]
        hard = [curve(path) for path in hard_paths]
        if task == "basketball":
            retention_paths = [basketball_retention_log(root, seed) for seed in SEEDS]
        else:
            retention_paths = [new_log(root, "truck_retention", seed) for seed in SEEDS]
        retention = [curve(path) for path in retention_paths]
        for path, values in zip(hard_paths + retention_paths, hard + retention):
            assert_full_eval_grid(values, path)
        j_hard = [n_auc(value, MAIN_STEPS) for value in hard]
        j_ret = [n_auc(value, MAIN_STEPS) for value in retention]
        p_hard = [n_auc(value, PERSIST_STEPS) for value in hard]
        p_ret = [n_auc(value, PERSIST_STEPS) for value in retention]
        delta = float(deltas[task]["delta_sesoi"])
        perf[task] = {
            "main_steps": MAIN_STEPS,
            "hard_exit_nauc": j_hard,
            "retention_nauc": j_ret,
            "delta_exit": paired_stats([a - b for a, b in zip(j_hard, j_ret)], delta),
            "persistence_delta": paired_stats([a - b for a, b in zip(p_hard, p_ret)], delta),
            "endpoint_95k_hard_exit": [value[95_000] for value in hard],
            "endpoint_95k_retention": [value[95_000] for value in retention],
        }
        scratch = [
            float(delta_payload["sources"][f"{task}_s{seed}"]["nauc_35k_80k"])
            for seed in SEEDS
        ]
        scratch_descriptive[task] = {
            "causal_status": "DESCRIPTIVE_ONLY",
            "scratch_nauc": scratch,
            "hard_exit_minus_scratch": paired_stats(
                [a - b for a, b in zip(j_hard, scratch)], delta
            ),
            "retention_minus_scratch": paired_stats(
                [a - b for a, b in zip(j_ret, scratch)], delta
            ),
        }
    audits = []
    for seed in SEEDS:
        audits.append(audit_new_run(root, "basketball_hard_exit", seed, epsilon, bands["basketball"]))
        audits.append(audit_new_run(root, "truck_retention", seed, epsilon, bands["truck"]))
        audits.append(audit_new_run(root, "truck_hard_exit", seed, epsilon, bands["truck"]))
    matrix = json.loads(
        (root / f"logs/phase1_bounded_bank_lease/{STAMP}/result.json").read_text()
    )
    matrix_pass = (
        matrix.get("status") == "PASS"
        and len(matrix.get("completed", [])) == 9
        and all(int(row["exit_code"]) == 0 for row in matrix["completed"])
    )
    result = {
        "schema_version": 1,
        "experiment": "phase1_bounded_bank_lease",
        "matrix_pass": matrix_pass,
        "engineering_valid": matrix_pass and all(value["pass"] for value in audits),
        "new_run_audits": audits,
        "performance": perf,
        "scratch_descriptive": scratch_descriptive,
        "historical_basketball_retention_gate_b": gate_payload["per_task"]["basketball"],
    }
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

