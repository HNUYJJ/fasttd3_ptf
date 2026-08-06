from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _condition(progress_mean: float, progress_t: float, return_mean: float) -> dict:
    aggregate = {
        "common_prefix_progress_delta": {
            "n_train_seeds": 3,
            "mean": progress_mean,
            "sd": 0.01,
            "t_vs_zero": progress_t,
        },
        "return_delta": {
            "n_train_seeds": 3,
            "mean": return_mean,
            "sd": 1.0,
            "t_vs_zero": return_mean,
        },
    }
    return {
        "by_checkpoint_step": {
            "100000": {"aggregate_over_train_seeds": aggregate}
        }
    }


def test_preregistered_performance_adjudicator(tmp_path: Path) -> None:
    summary = {
        "coverage": {"complete": True},
        "exact_reset_pairing_validated": True,
        "tasks": {
            "basketball": {
                "comparisons": {
                    "admission_none": _condition(-0.01, -0.5, -2.0),
                    "wfix": _condition(-0.20, -4.0, -60.0),
                }
            },
            "powerlift": {
                "comparisons": {
                    "admission_all": _condition(0.0002, 3.2, 10.0),
                    "wfix": _condition(0.00025, 3.5, 14.0),
                }
            },
        },
    }
    summary_path = tmp_path / "summary.json"
    json_path = tmp_path / "verdict.json"
    md_path = tmp_path / "verdict.md"
    summary_path.write_text(json.dumps(summary))
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/adjudicate_admission_core_v1.py"),
            "--summary",
            str(summary_path),
            "--json-out",
            str(json_path),
            "--md-out",
            str(md_path),
        ],
        check=True,
    )
    verdict = json.loads(json_path.read_text())
    assert verdict["basketball_negative_safety"]["pass"] is True
    assert verdict["powerlift_positive_retention"]["pass"] is True
    assert verdict["both_performance_gates_pass"] is True
    assert "Both gates: **PASS**" in md_path.read_text()
