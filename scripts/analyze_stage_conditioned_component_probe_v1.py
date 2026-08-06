#!/usr/bin/env python3
"""Apply the frozen bidirectional feasibility gate to v1 probe artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _summary(report: dict) -> dict:
    return {
        "task": report["task"],
        "global_step": report["student"]["global_step"],
        "admitted_order": report["admitted_order"],
        "exact_abstention": report["exact_abstention"],
        "sources": {
            name: {
                "admitted": result["admitted"],
                "dR_lcb90": result["intervals"]["return"]["lcb90"],
                "dP_lcb90": result["intervals"]["progress"]["lcb90"],
                "dF_lcb90": min(
                    (
                        item["lcb90"]
                        for item in result["feasibility_intervals"].values()
                    ),
                    default=0.0,
                ),
            }
            for name, result in report["classifications"].items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hurdle", required=True)
    parser.add_argument("--crawl", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md", required=True)
    args = parser.parse_args()

    hurdle_path = Path(args.hurdle).resolve()
    crawl_path = Path(args.crawl).resolve()
    hurdle = json.loads(hurdle_path.read_text())
    crawl = json.loads(crawl_path.read_text())
    if hurdle["task"] != "hurdle" or crawl["task"] != "crawl":
        raise ValueError("task artifacts are swapped or invalid")
    if hurdle["student"]["global_step"] != 10_000 or crawl["student"]["global_step"] != 10_000:
        raise ValueError("primary gate requires 10k student checkpoints")
    if hurdle["protocol"] != crawl["protocol"]:
        raise ValueError("positive and negative task protocols differ")

    hurdle_order = hurdle["admitted_order"]
    hurdle_pass = (
        "run" in hurdle_order
        and "walk" in hurdle_order
        and hurdle_order.index("run") < hurdle_order.index("walk")
    )
    crawl_pass = crawl["exact_abstention"] and not crawl["admitted_order"]
    decision = (
        "BIDIRECTIONAL_FEASIBILITY_PASS"
        if hurdle_pass and crawl_pass
        else "CANDIDATE_REJECTED"
    )
    result = {
        "experiment": "stage_conditioned_component_probe_v1",
        "decision": decision,
        "primary_gate": {
            "hurdle_admit_run_walk_rank_run_over_walk": hurdle_pass,
            "crawl_reject_all": crawl_pass,
        },
        "hurdle": _summary(hurdle),
        "crawl": _summary(crawl),
        "inputs": {"hurdle": str(hurdle_path), "crawl": str(crawl_path)},
        "claim_boundary": (
            "Passing only authorizes an online low-frequency feasibility test."
            if decision == "BIDIRECTIONAL_FEASIBILITY_PASS"
            else "The frozen short matched-component proxy is rejected; do not retune it on these labels."
        ),
    }
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    if out_json.exists():
        raise FileExistsError(out_json)
    out_json.write_text(json.dumps(result, indent=2) + "\n")

    lines = [
        "# Stage-conditioned component probe v1 — result",
        "",
        f"**Decision: `{decision}`**",
        "",
        "| Primary gate | Pass | Observed |",
        "|---|---:|---|",
        f"| Hurdle admits run and walk, ranks run > walk | {hurdle_pass} | "
        f"`{hurdle_order or 'NONE'}` |",
        f"| Crawl rejects all sources | {crawl_pass} | "
        f"`{crawl['admitted_order'] or 'NONE'}` |",
        "",
        "## Conservative lower bounds",
        "",
        "| Task | Source | Admitted | LCB90 ΔR | LCB90 ΔP | LCB90 ΔF |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for task_report in (hurdle, crawl):
        for name in ("stand", "walk", "run"):
            item = task_report["classifications"][name]
            intervals = item["intervals"]
            feasibility_lcb = min(
                (
                    value["lcb90"]
                    for value in item["feasibility_intervals"].values()
                ),
                default=0.0,
            )
            lines.append(
                f"| {task_report['task']} | {name} | {item['admitted']} | "
                f"{intervals['return']['lcb90']:.4f} | "
                f"{intervals['progress']['lcb90']:.4f} | "
                f"{feasibility_lcb:.4f} |"
            )
    lines += [
        "",
        "## Claim boundary",
        "",
        result["claim_boundary"],
        "",
        "The full RBO training outcomes are causal intervention labels; this probe is only a cheap predictor.",
    ]
    out_md = Path(args.out_md)
    if out_md.exists():
        raise FileExistsError(out_md)
    out_md.write_text("\n".join(lines) + "\n")
    print(f"{decision} -> {out_json}, {out_md}")


if __name__ == "__main__":
    main()
