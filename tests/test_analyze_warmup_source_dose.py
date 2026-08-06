import sys
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.analyze_warmup_source_dose import (  # noqa: E402
    aggregate_runs,
    analyze_history_rows,
)


def _row(step, stand, run, teacher=None):
    teacher = stand + run if teacher is None else teacher
    return {
        "_step": step,
        "mcg/exec_share_src0": stand,
        "mcg/exec_share_src1": run,
        "mcg/exec_env_frac": teacher,
        "mcg/exec_part_frac": teacher,
    }


def test_strict_warmup_filter_and_cross_sectional_seed_stats():
    rows_seed1 = [
        _row(-1, 0.9, 0.0),
        _row(0, 0.2, 0.3),
        _row(100, 0.1, 0.4),
        _row(29_999, 0.3, 0.2),
        _row(30_000, 0.0, 0.9),
    ]
    seed1 = analyze_history_rows(rows_seed1, ["stand", "run"])
    assert seed1["n_history_cross_sections"] == 3
    assert seed1["selected_step_min"] == 0
    assert seed1["selected_step_max"] == 29_999
    assert seed1["source_absolute_shares"]["stand"] == pytest.approx(0.2)
    assert seed1["source_absolute_shares"]["run"] == pytest.approx(0.3)
    assert seed1["teacher_share"] == pytest.approx(0.5)
    assert seed1["student_share"] == pytest.approx(0.5)
    assert seed1["validation"]["passed"] is True

    seed2 = analyze_history_rows([_row(100, 0.4, 0.2)], ["stand", "run"])
    aggregate = aggregate_runs([seed1, seed2], ["stand", "run"])
    assert aggregate["teacher_share"]["mean"] == pytest.approx(0.55)
    assert aggregate["teacher_share"]["sd"] == pytest.approx(0.0707106781)
    assert aggregate["source_absolute_shares"]["run"]["mean"] == pytest.approx(0.25)


def test_source_shares_must_sum_to_teacher_at_every_cross_section():
    with pytest.raises(ValueError, match="source shares sum"):
        analyze_history_rows(
            [_row(100, 0.2, 0.3, teacher=0.6)],
            ["stand", "run"],
            share_tolerance=1e-8,
        )
