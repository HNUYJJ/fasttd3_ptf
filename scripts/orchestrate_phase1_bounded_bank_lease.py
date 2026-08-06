#!/usr/bin/env python3
"""Run the frozen Phase-1 bounded-bank-lease matrix with at most two GPUs."""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JOBS = [
    *(('basketball_hard_exit', seed) for seed in (1, 2, 3)),
    *(('truck_retention', seed) for seed in (1, 2, 3)),
    *(('truck_hard_exit', seed) for seed in (1, 2, 3)),
]
FROZEN = [
    ROOT / "docs/data/p1_bounded_bank_lease/gate_a_report.json",
    ROOT / "docs/data/p1_bounded_bank_lease/delta_frozen.json",
    ROOT / "docs/data/p1_bounded_bank_lease/gate_b_tolerance_frozen.json",
]


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _preflight(formal: bool) -> str:
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if formal and subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True
    ).strip():
        raise RuntimeError("formal execution requires a clean committed worktree")
    for path in FROZEN:
        payload = json.loads(path.read_text())
        if path.name == "gate_a_report.json" and payload.get("status") != "PASS":
            raise RuntimeError("Gate A has not passed")
        if path.name != "gate_a_report.json" and payload.get("status") != "frozen":
            raise RuntimeError(f"artifact is not frozen: {path}")
    return head


def _terminate(active: dict[int, tuple[subprocess.Popen, object, str]]) -> None:
    for proc, _, _ in active.values():
        if proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    deadline = time.time() + 20
    while time.time() < deadline and any(proc.poll() is None for proc, _, _ in active.values()):
        time.sleep(0.2)
    for proc, _, _ in active.values():
        if proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--stamp")
    args = parser.parse_args()
    gpus = [int(value) for value in args.gpus.split(",") if value.strip()]
    if not gpus or len(gpus) > 2 or len(set(gpus)) != len(gpus):
        raise ValueError("use one or two distinct GPU ids")
    head = _preflight(formal=not args.smoke)
    stamp = args.stamp or _utc_stamp()
    jobs = [("basketball_hard_exit", 1)] if args.smoke else list(JOBS)
    plan = {
        "schema_version": 1,
        "experiment": "phase1_bounded_bank_lease",
        "kind": "smoke" if args.smoke else "formal",
        "stamp": stamp,
        "git_head": head,
        "gpus": gpus,
        "jobs": [{"cell": cell, "seed": seed} for cell, seed in jobs],
    }
    plan_dir = ROOT / "logs/phase1_bounded_bank_lease" / stamp
    plan_dir.mkdir(parents=True, exist_ok=True)
    (plan_dir / "plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    print(json.dumps(plan, indent=2), flush=True)
    if not args.execute:
        return 0

    available = list(gpus)
    pending = list(jobs)
    active: dict[int, tuple[subprocess.Popen, object, str]] = {}
    completed: list[dict] = []
    try:
        while pending or active:
            while pending and available:
                gpu = available.pop(0)
                cell, seed = pending.pop(0)
                label = f"{cell}_s{seed}"
                log_handle = (plan_dir / f"launcher_{label}.log").open("w")
                env = os.environ.copy()
                env.update({
                    "CELL": cell,
                    "SEED": str(seed),
                    "GPU_ID": str(gpu),
                    "STAMP": stamp,
                    "SMOKE": "1" if args.smoke else "0",
                })
                proc = subprocess.Popen(
                    ["bash", "scripts/run_phase1_bounded_bank_lease.sh"],
                    cwd=ROOT,
                    env=env,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                active[gpu] = (proc, log_handle, label)
                print(f"LAUNCHED {label} gpu={gpu} pid={proc.pid}", flush=True)
            time.sleep(2)
            for gpu, (proc, handle, label) in list(active.items()):
                code = proc.poll()
                if code is None:
                    continue
                handle.close()
                completed.append({"label": label, "gpu": gpu, "exit_code": code})
                del active[gpu]
                available.append(gpu)
                available.sort()
                print(f"FINISHED {label} gpu={gpu} exit={code}", flush=True)
                if code != 0:
                    raise RuntimeError(f"job failed: {label} exit={code}")
    except BaseException:
        _terminate(active)
        raise
    finally:
        for _, handle, _ in active.values():
            handle.close()

    result = {**plan, "completed": completed, "status": "PASS"}
    (plan_dir / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"ALL_DONE result={plan_dir / 'result.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
