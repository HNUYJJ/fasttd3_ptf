"""Capture the exact dirty-worktree source identity before the scientific run."""
from __future__ import annotations

import hashlib
import json
import subprocess
import tarfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/data/slide_hard_exit_v1"
SOURCE_MODEL = ROOT / "models/h1hand-walk-v0__h1hand_walk_source_official__1_final.pt"
FILES = (
    "fasttd3_ptf/official_fasttd3_ptf/train_ptf.py",
    "fasttd3_ptf/official_fasttd3_ptf/anchor_io.py",
    "fasttd3_ptf/official_fasttd3_ptf/ptf_replay.py",
    "fasttd3_ptf/official_fasttd3_ptf/admission_control.py",
    "fasttd3_ptf/official_fasttd3_ptf/humanoid_bench_env.py",
    "fasttd3_ptf/official_fasttd3_ptf/rng_isolation.py",
    "fasttd3_ptf/ptf/mcg.py",
    "fasttd3_ptf/ptf/source_bank.py",
    "fasttd3_ptf/ptf/action_schema.py",
    "fasttd3_ptf/official_code/FastTD3/fast_td3/fast_td3.py",
    "fasttd3_ptf/official_code/FastTD3/fast_td3/fast_td3_utils.py",
    "fasttd3_ptf/official_code/FastTD3/fast_td3/hyperparams.py",
    "scripts/official_fasttd3_train_target_ptf.sh",
    "scripts/run_slide_hard_exit_v1.sh",
    "scripts/eval_slide_hard_exit_v1.sh",
    "scripts/analysis/analyze_slide_hard_exit_v1.py",
    "configs/source_banks/calibration/h1hand_slide_rbo_walk.yaml",
    "checkpoints/official_sources/h1hand_walk/manifest.json",
    "docs/run_card_interventional_bootstrap_racing_v1.md",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=ROOT, text=True, stderr=subprocess.STDOUT
    ).strip()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if any(not (ROOT / relative).is_file() for relative in FILES):
        missing = [relative for relative in FILES if not (ROOT / relative).is_file()]
        raise FileNotFoundError(f"runtime source file missing: {missing}")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive = OUT / f"runtime_source_snapshot_{stamp}.tar.gz"
    manifest_path = OUT / f"runtime_manifest_{stamp}.json"
    if archive.exists() or manifest_path.exists():
        raise FileExistsError("runtime capture output already exists")
    with tarfile.open(archive, "w:gz") as tar:
        for relative in FILES:
            tar.add(ROOT / relative, arcname=relative, recursive=False)
    status = git("status", "--porcelain=v1")
    manifest = {
        "schema": "slide_hard_exit_runtime.v1",
        "capture_utc": datetime.now(timezone.utc).isoformat(),
        "git": {
            "head": git("rev-parse", "HEAD"),
            "dirty": bool(status),
            "status_sha256": hashlib.sha256(status.encode()).hexdigest(),
        },
        "source_snapshot": {
            "path": str(archive.relative_to(ROOT)),
            "sha256": sha256(archive),
            "bytes": archive.stat().st_size,
        },
        "file_sha256": {
            relative: sha256(ROOT / relative) for relative in FILES
        },
        "source_model": {
            "path": str(SOURCE_MODEL.relative_to(ROOT)),
            "sha256": sha256(SOURCE_MODEL),
            "bytes": SOURCE_MODEL.stat().st_size,
        },
        "experiment": {
            "target": "h1hand-slide-v0",
            "seeds": [1, 2, 3],
            "prefix_stop": 30000,
            "continuation_stop": 100000,
            "arms": ["matched_continuous", "hard_exit"],
            "source_free_eval_episodes": 128,
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(manifest_path.relative_to(ROOT))
    print(sha256(manifest_path))


if __name__ == "__main__":
    main()
