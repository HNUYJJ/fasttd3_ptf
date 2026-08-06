from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.analysis.p1_gate_a_equivalence import run_gate


def test_basketball_all_and_schedule_step0_all_are_equivalent() -> None:
    result = run_gate()
    assert result["status"] == "PASS"
    assert result["dynamic_trace_steps"] == 100
    assert result["segment_boundaries_crossed"] >= 3
