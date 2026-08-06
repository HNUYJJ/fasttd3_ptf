"""Post-hoc economics audit for the Hurdle fixed-K source portfolio.

This script does not adjudicate a new experiment.  It reconciles the units and
the evidence that already exist for:

* Hurdle source racing at K=10k vector steps;
* the independently trained full-horizon run and scratch arms; and
* the 128-episode source-free panels used to rank arms.

The audit deliberately distinguishes source-identity screening from
winner-state continuation.  The latter has not yet been executed in the
repository and therefore cannot be certified from the existing curves.
"""
from __future__ import annotations

import json
import statistics as st
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SPEEDUP = ROOT / "docs/data/hurdle_speedup_v1/hurdle_speedup_v1_results.json"
RACING = ROOT / "docs/data/racing_min_horizon_v1/correct_lr/results.json"
OUT = ROOT / "docs/data/hurdle_portfolio_economics_v1/results.json"

NUM_ENVS = 128
K_VECTOR_STEPS = 10_000
MAX_VECTOR_STEPS = 100_000
N_SOURCES = 3
EVAL_EPISODES_PER_ARM = 128
EVAL_MAX_EPISODE_STEPS = 1_000
THRESHOLDS = (200.0, 300.0)


def _first_crossing(curve: list[list[float]], threshold: float) -> tuple[float, bool]:
    """Linearly interpolated first crossing, right-censored at the horizon."""
    points = [(float(x), float(y)) for x, y in curve]
    if points[0][1] >= threshold:
        return points[0][0], False
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if y0 < threshold <= y1:
            frac = (threshold - y0) / (y1 - y0)
            return x0 + frac * (x1 - x0), False
    return float(points[-1][0]), True


def _sustained_crossing(curve: list[list[float]], threshold: float) -> tuple[float, bool]:
    """Earliest interpolated crossing after which all observed checkpoints pass.

    A censored value is set to the observed horizon.  Therefore a benefit that
    uses this value is an optimistic upper bound when the source arm is
    censored: its true sustained crossing is later than the recorded horizon.
    """
    points = [(float(x), float(y)) for x, y in curve]
    passing_suffix = [False] * len(points)
    suffix = True
    for idx in range(len(points) - 1, -1, -1):
        suffix = suffix and points[idx][1] >= threshold
        passing_suffix[idx] = suffix
    for idx, passed in enumerate(passing_suffix):
        if not passed:
            continue
        if idx == 0:
            return points[0][0], False
        x0, y0 = points[idx - 1]
        x1, y1 = points[idx]
        if y0 < threshold <= y1:
            frac = (threshold - y0) / (y1 - y0)
            return x0 + frac * (x1 - x0), False
        return x1, False
    return float(points[-1][0]), True


def _median(values: list[float]) -> float:
    return float(st.median(values))


def main() -> None:
    speedup = json.loads(SPEEDUP.read_text())
    racing = json.loads(RACING.read_text())
    curves = speedup["curves"]

    threshold_audit: dict[str, object] = {}
    for threshold in THRESHOLDS:
        rows = []
        for seed in (1, 2, 3):
            scratch_curve = curves["scratch"][str(seed)]
            source_curve = curves["source"][str(seed)]
            scratch_first, scratch_first_c = _first_crossing(scratch_curve, threshold)
            source_first, source_first_c = _first_crossing(source_curve, threshold)
            scratch_stable, scratch_stable_c = _sustained_crossing(scratch_curve, threshold)
            source_stable, source_stable_c = _sustained_crossing(source_curve, threshold)
            rows.append(
                {
                    "seed": seed,
                    "first": {
                        "scratch": scratch_first,
                        "source": source_first,
                        "benefit": scratch_first - source_first,
                        "scratch_censored": scratch_first_c,
                        "source_censored": source_first_c,
                    },
                    "sustained_observed_checkpoints": {
                        "scratch": scratch_stable,
                        "source": source_stable,
                        "benefit_upper_if_censored": scratch_stable - source_stable,
                        "scratch_censored": scratch_stable_c,
                        "source_censored": source_stable_c,
                    },
                }
            )
        threshold_audit[str(int(threshold))] = {
            "per_seed": rows,
            "median_first_benefit_vector_steps": _median(
                [r["first"]["benefit"] for r in rows]
            ),
            "median_sustained_benefit_vector_steps_upper_if_censored": _median(
                [r["sustained_observed_checkpoints"]["benefit_upper_if_censored"] for r in rows]
            ),
        }

    # The evaluator does not persist actual episode lengths.  One 128-episode
    # panel therefore costs between 1 and 1000 vector-step equivalents for a
    # 128-env training job.  The maximum is 10% of K, not 128k vector steps.
    eval_vector_equivalent_per_arm = {
        "lower": EVAL_EPISODES_PER_ARM / NUM_ENVS,
        "upper": EVAL_EPISODES_PER_ARM * EVAL_MAX_EPISODE_STEPS / NUM_ENVS,
        "exact_observable": False,
        "reason": "p0_evaluator records episode_steps cap but not realized episode length",
    }

    source_only_arms = N_SOURCES
    source_plus_student_arms = N_SOURCES + 1
    cost_scenarios = {
        "identity_screen_then_restart_selected_source": {
            "meaning": "all source pilots are discarded; a fresh selected-source learner is trained",
            "race_overhead_vector_steps_without_eval": N_SOURCES * K_VECTOR_STEPS,
            "selection_eval_vector_steps_equivalent": {
                "lower": source_only_arms * eval_vector_equivalent_per_arm["lower"],
                "upper": source_only_arms * eval_vector_equivalent_per_arm["upper"],
            },
        },
        "winner_state_continuation_source_only": {
            "meaning": "three source branches race and the realized winning learner state continues",
            "race_overhead_vector_steps_without_eval": (N_SOURCES - 1) * K_VECTOR_STEPS,
            "selection_eval_vector_steps_equivalent": {
                "lower": source_only_arms * eval_vector_equivalent_per_arm["lower"],
                "upper": source_only_arms * eval_vector_equivalent_per_arm["upper"],
            },
            "evidence_status": "not executed; existing racing branches were not faithfully continued",
        },
        "winner_state_continuation_with_student_abstention_arm": {
            "meaning": "three source branches plus one student branch race; winner continues",
            "race_overhead_vector_steps_without_eval": N_SOURCES * K_VECTOR_STEPS,
            "selection_eval_vector_steps_equivalent": {
                "lower": source_plus_student_arms * eval_vector_equivalent_per_arm["lower"],
                "upper": source_plus_student_arms * eval_vector_equivalent_per_arm["upper"],
            },
            "evidence_status": "not executed; no full-state winner continuation or matched scratch population",
        },
    }

    # A descriptive calculation only: the three full scratch curves can show
    # what a K=10k return selector would have picked, but they are different
    # initial seeds rather than same-anchor paired branches.
    scratch_at_k = {
        int(seed): float(dict(curves["scratch"][seed])[K_VECTOR_STEPS])
        for seed in ("1", "2", "3")
    }
    selected_scratch_seed = max(scratch_at_k, key=scratch_at_k.get)
    scratch_population = {
        "population_size": 3,
        "return_at_K": scratch_at_k,
        "selected_seed_at_K": selected_scratch_seed,
        "selected_seed_thresholds": {},
        "hindsight_fastest_seed_thresholds": {},
        "status": "descriptive_only",
        "limitations": [
            "different learner initializations, not same-anchor paired branches",
            "only three full scratch curves; a 3-source plus student portfolio requires four candidates",
            "source racing winners were not continued, so no matched end-to-end portfolio comparison exists",
        ],
    }
    for threshold in THRESHOLDS:
        selected_curve = curves["scratch"][str(selected_scratch_seed)]
        selected_cross, selected_censored = _sustained_crossing(selected_curve, threshold)
        all_crossings = {
            seed: _sustained_crossing(curves["scratch"][str(seed)], threshold)
            for seed in (1, 2, 3)
        }
        fastest_seed = min(all_crossings, key=lambda s: all_crossings[s][0])
        scratch_population["selected_seed_thresholds"][str(int(threshold))] = {
            "vector_steps": selected_cross,
            "censored": selected_censored,
        }
        scratch_population["hindsight_fastest_seed_thresholds"][str(int(threshold))] = {
            "seed": fastest_seed,
            "vector_steps": all_crossings[fastest_seed][0],
            "censored": all_crossings[fastest_seed][1],
        }

    report = {
        "analysis_type": "posthoc_economics_audit",
        "inputs": {
            "speedup": str(SPEEDUP.relative_to(ROOT)),
            "racing": str(RACING.relative_to(ROOT)),
            "racing_verdict": racing.get("verdict"),
            "racing_K_star": racing.get("K_star"),
        },
        "units": {
            "reported_training_step": "vector step",
            "num_envs": NUM_ENVS,
            "environment_transitions_per_vector_step": NUM_ENVS,
            "K_vector_steps": K_VECTOR_STEPS,
            "K_environment_transitions_per_arm": K_VECTOR_STEPS * NUM_ENVS,
        },
        "threshold_audit": threshold_audit,
        "selection_evaluation_cost": eval_vector_equivalent_per_arm,
        "cost_scenarios": cost_scenarios,
        "scratch_population_descriptive": scratch_population,
        "verdict": "END_TO_END_ECONOMICS_NOT_IDENTIFIED",
        "why_not_identified": [
            "the existing source racing branches were evaluated but not faithfully continued",
            "the available long-horizon run arm is an independent source-identity validation, not the realized racing winner",
            "no equal-candidate same-anchor scratch population with winner continuation exists",
            "realized evaluation episode lengths were not recorded, so selection interaction cost is bounded rather than exact",
        ],
        "bounded_conclusions": [
            "evaluation cost was omitted from prior arithmetic, but its per-arm maximum is 1000 vector-step equivalents, not 128000 vector steps",
            "first-crossing savings overstate deployment-stable savings when the source arm later falls below threshold",
            "with a student abstention arm, the no-evaluation racing overhead is 30000 vector steps for three sources",
            "a new experiment must continue the realized winner and include a same-candidate-count scratch population before claiming net portfolio benefit",
        ],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
