"""Build single-source stand/walk/run banks for the deconfounding audit.

The source entry, observation adapter and action metadata are copied from each
target's existing final-method bank.  Only source identity changes; bootstrap
weight, horizon and all training hyperparameters remain controlled.

By default this script only prints the planned outputs.  Pass ``--write`` to
materialize the YAML banks and ``--emit-commands`` to print the seed-1 pilot
training commands.  It never launches training.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

TASK_BASE_BANK = {
    "cabinet": "configs/source_banks/h1hand_loco_wfix_cabinet.yaml",
    "maze": "configs/source_banks/h1hand_loco_wfix_maze.yaml",
    "powerlift": "configs/source_banks/h1hand_std9_wfix_powerlift.yaml",
    "basketball": "configs/source_banks/h1hand_std9_wfix_basketball.yaml",
}
SOURCES = ("stand", "walk", "run")


def single_source_bank(base_path: str | Path, source_name: str) -> dict:
    base = yaml.safe_load(Path(base_path).read_text())
    matches = [source for source in base.get("sources", []) if source.get("name") == source_name]
    if len(matches) != 1:
        raise ValueError(f"{base_path}: expected one source named {source_name!r}, found {len(matches)}")
    source = matches[0]
    source["bootstrap"] = {"weight": 1.0, "horizon": 25}
    return {"null_option": True, "sources": [source]}


def command(task: str, source: str, seed: int) -> str:
    bank = f"configs/source_banks/audit/h1hand_{task}_sd_{source}.yaml"
    return (
        f"ENV_NAME=h1hand-{task}-v0 "
        f"EXP_NAME=h1hand_{task}_sd_{source}_s{seed} SEED={seed} "
        f"SOURCE_BANK={bank} TOTAL_TIMESTEPS=100000 "
        "PTF_MCG=1 PTF_MCG_GROUPS=legs_torso,arms "
        "PTF_MCG_WARMUP_STEPS=30000 PTF_MCG_WARMUP_EXEC_PROB=0.5 "
        "PTF_MCG_MIN_STEPS=25 PTF_MCG_WARMUP_MODE=safe_bootstrap "
        "PTF_MCG_ABLATION=bootstrap_only "
        "bash scripts/official_fasttd3_train_target_ptf.sh"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", choices=sorted(TASK_BASE_BANK), default=sorted(TASK_BASE_BANK))
    parser.add_argument("--sources", nargs="+", choices=SOURCES, default=list(SOURCES))
    parser.add_argument("--seeds", nargs="+", type=int, default=[1])
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--emit-commands", action="store_true")
    args = parser.parse_args()

    out_dir = Path("configs/source_banks/audit")
    if args.write:
        out_dir.mkdir(parents=True, exist_ok=True)
    for task in args.tasks:
        for source in args.sources:
            bank = single_source_bank(TASK_BASE_BANK[task], source)
            out_path = out_dir / f"h1hand_{task}_sd_{source}.yaml"
            print(f"{task:12s} {source:5s} -> {out_path}")
            if args.write:
                header = (
                    f"# Stability-deconfounded audit: {source}-only RBO for {task}.\n"
                    "# Controlled: weight=1, horizon=25, warmup=30k, exec_prob=0.5.\n"
                )
                out_path.write_text(header + yaml.safe_dump(bank, sort_keys=False, allow_unicode=True))
    if args.emit_commands:
        print("\n# Pilot commands (not executed)")
        for task in args.tasks:
            for source in args.sources:
                for seed in args.seeds:
                    print(command(task, source, seed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
