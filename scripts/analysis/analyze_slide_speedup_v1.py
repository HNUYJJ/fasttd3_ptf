"""Slide 样本效率加速倍数冻结裁决。

判据来自 docs/experiments/slide_speedup_v1_prereg_20260731.md：
  θ ∈ {250, 375, 500}
  SPEEDUP_CONFIRMED:
    至少 2/3 阈值的 speedup 中位数 >= 2.0，且相应阈值 3/3 seed >= 1.5
  SPEEDUP_REFUTED:
    全部阈值中位数 < 1.5，或 100k 的 walk 均值被 scratch 均值反超
  其余为 SPEEDUP_PARTIAL。
"""
from __future__ import annotations

import glob
import hashlib
import json
import math
import statistics as st
from pathlib import Path

import torch

ROOT = Path("docs/data/slide_speedup_v1")
EVAL = ROOT / "source_free_eval"
OUT = ROOT / "slide_speedup_v1_results.json"
TRAIN_LOG = Path("logs/train/slide_speedup_v1")
SEEDS = (1, 2, 3)
STEPS = (10000, 20000, 30000, 50000, 75000, 100000)
THRESHOLDS = (250.0, 375.0, 500.0)
CENSOR = float(STEPS[-1])
DOSE_BAND = (0.45, 0.55)
ENV_NAME = "h1hand-slide-v0"
EVAL_SEEDS = (11, 23, 37, 53, 71, 89, 103, 113,
              131, 149, 163, 179, 193, 211, 227, 241)
RANKS = tuple(range(8))
EPISODE_STEPS = 1000
PANEL_SEEDS = [
    seed * 1000 + rank
    for seed in EVAL_SEEDS
    for rank in RANKS
]

COMMON_ARGS = {
    "env_name": ENV_NAME,
    "project": "ptf_fasttd3_slide_speedup",
    "total_timesteps": 100000,
    "num_envs": 128,
    "batch_size": 32768,
    "buffer_size": 51200,
    "learning_starts": 10,
    "num_updates": 2,
    "save_interval": 0,
    "eval_interval": 0,
    "render_interval": 0,
    "compile": False,
    "amp": True,
    "torch_deterministic": True,
}
COMMON_PTF = {
    "eval_checkpoint_steps": list(STEPS),
    "execute_sources": False,
    "admission_replay_mode": "shared",
}
ARM_PTF = {
    "scratch": {
        "source_bank": "configs/source_banks/empty.yaml",
        "admission_mode": "legacy",
        "mcg": False,
    },
    "walk": {
        "source_bank": "configs/source_banks/calibration/h1hand_slide_rbo_walk.yaml",
        "mcg": True,
        "mcg_groups": ["legs_torso", "arms", "hands"],
        "mcg_warmup_steps": 100000,
        "mcg_warmup_min_steps": 25,
        "mcg_warmup_mode": "admission_bootstrap",
        "mcg_warmup_exec_prob": 0.5,
        "mcg_ablation": "bootstrap_only",
        "admission_mode": "all",
        "admission_student_logit": 0.0,
        "admission_expected_source_mass": 0.5,
        "admission_replay_recency_half_life": 0.0,
        "admission_replay_uniform_mix": 1.0,
        "admission_replay_priority_alpha": 0.0,
        "admission_replay_handoff": "physical_after_authority",
    },
}


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check_fields(
    defects: list[str], tag: str, actual: dict, expected: dict
) -> None:
    for field, value in expected.items():
        if actual.get(field) != value:
            defects.append(
                f"{tag}: {field}={actual.get(field)!r}, expected {value!r}"
            )


def _curve(arm: str, seed: int) -> list[tuple[int, float]]:
    return [
        (
            step,
            float(
                json.loads(
                    (EVAL / f"{arm}_s{seed}_step{step}.json").read_text()
                )["aggregate"]["return_mean"]
            ),
        )
        for step in STEPS
    ]


def _steps_to(points: list[tuple[int, float]], theta: float) -> tuple[float, bool]:
    previous = None
    for step, value in points:
        if value >= theta:
            if previous is None:
                return float(step), False
            step0, value0 = previous
            if value == value0:
                return float(step), False
            return (
                step0 + (theta - value0) / (value - value0) * (step - step0),
                False,
            )
        previous = (step, value)
    return CENSOR, True


def _validate() -> tuple[list[str], dict[str, float]]:
    defects: list[str] = []
    dose: dict[str, float] = {}

    for arm in ("scratch", "walk"):
        for seed in SEEDS:
            log = TRAIN_LOG / f"sspd_{arm}_s{seed}.log"
            if not log.exists():
                defects.append(f"{arm} s{seed}: missing training log")
            for step in STEPS:
                matches = glob.glob(f"models/*sspd_{arm}_s{seed}__*_{step}.pt")
                if len(matches) != 1:
                    defects.append(
                        f"{arm} s{seed} step {step}: checkpoint matches={len(matches)}"
                    )
                eval_path = EVAL / f"{arm}_s{seed}_step{step}.json"
                if not eval_path.exists():
                    defects.append(f"{arm} s{seed} step {step}: missing eval")
                    continue
                try:
                    data = json.loads(eval_path.read_text())
                    checkpoint = data["checkpoint"]
                    protocol = data["protocol"]
                    episodes = data["episodes"]
                    aggregate = data["aggregate"]
                except Exception as exc:
                    defects.append(
                        f"{arm} s{seed} step {step}: invalid eval {type(exc).__name__}"
                    )
                    continue
                tag = f"{arm} s{seed} step {step}"
                if data.get("env_name") != ENV_NAME:
                    defects.append(f"{tag}: wrong env")
                if checkpoint.get("global_step") != step:
                    defects.append(f"{tag}: wrong checkpoint step")
                if checkpoint.get("identity_checked") is not True:
                    defects.append(f"{tag}: identity not checked")
                if f"sspd_{arm}_s{seed}__" not in str(checkpoint.get("path", "")):
                    defects.append(f"{tag}: wrong checkpoint identity")
                if protocol.get("eval_seeds") != list(EVAL_SEEDS):
                    defects.append(f"{tag}: wrong eval seed bases")
                if protocol.get("ranks") != list(RANKS):
                    defects.append(f"{tag}: wrong eval ranks")
                if protocol.get("episode_steps") != EPISODE_STEPS:
                    defects.append(f"{tag}: wrong episode_steps")
                if protocol.get("compat_subpanel_episodes") != 32:
                    defects.append(f"{tag}: wrong compatibility subpanel size")
                if protocol.get("deterministic") is not True:
                    defects.append(f"{tag}: nondeterministic eval")
                if protocol.get("source_free") != (
                    "structural (no bank/option/admission components constructed)"
                ):
                    defects.append(f"{tag}: source-free declaration mismatch")
                if len(episodes) != 128 or aggregate.get("episode_count") != 128:
                    defects.append(f"{tag}: wrong episode count")
                if [row.get("seed") for row in episodes] != PANEL_SEEDS:
                    defects.append(f"{tag}: wrong seed panel")
                if not all(
                    isinstance(row.get("return"), (int, float))
                    and math.isfinite(float(row["return"]))
                    for row in episodes
                ):
                    defects.append(f"{tag}: non-finite return")

                if len(matches) == 1:
                    try:
                        current_checkpoint = Path(matches[0]).resolve()
                        if Path(str(checkpoint.get("path", ""))).resolve() != current_checkpoint:
                            defects.append(f"{tag}: eval/checkpoint path mismatch")
                        if checkpoint.get("sha256") != _sha256(current_checkpoint):
                            defects.append(f"{tag}: eval/checkpoint sha256 mismatch")
                        state = torch.load(
                            current_checkpoint, map_location="cpu", weights_only=False
                        )
                        if state.get("global_step") != step:
                            defects.append(f"{tag}: checkpoint global_step mismatch")
                        ckpt_args = state.get("args") or {}
                        _check_fields(defects, f"{tag} args", ckpt_args, COMMON_ARGS)
                        _check_fields(
                            defects,
                            f"{tag} args",
                            ckpt_args,
                            {"seed": seed, "exp_name": f"sspd_{arm}_s{seed}"},
                        )
                        ptf_cfg = state.get("ptf_cfg") or {}
                        _check_fields(defects, f"{tag} ptf_cfg", ptf_cfg, COMMON_PTF)
                        _check_fields(defects, f"{tag} ptf_cfg", ptf_cfg, ARM_PTF[arm])
                        expected_names = ["null"] if arm == "scratch" else ["walk"]
                        if state.get("source_names") != expected_names:
                            defects.append(
                                f"{tag}: source_names={state.get('source_names')!r}, "
                                f"expected {expected_names!r}"
                            )
                        if arm == "walk":
                            audit = state.get("admission_audit") or {}
                            for field in ("execution_counts", "critic_sample_counts"):
                                values = audit.get(field)
                                if not isinstance(values, (list, tuple)) or len(values) != 2:
                                    defects.append(f"{tag}: invalid {field}")
                                    continue
                                counts = [int(value) for value in values]
                                if min(counts) < 0 or sum(counts) <= 0:
                                    defects.append(f"{tag}: invalid {field} counts")
                                    continue
                                share = counts[0] / sum(counts)
                                if not DOSE_BAND[0] <= share <= DOSE_BAND[1]:
                                    defects.append(
                                        f"{tag}: {field} share={share:.4f} "
                                        f"outside {DOSE_BAND}"
                                    )
                                if step == STEPS[-1]:
                                    dose[f"walk_s{seed}_{field}"] = round(share, 4)
                    except Exception as exc:
                        defects.append(
                            f"{tag}: checkpoint load failed {type(exc).__name__}"
                        )
    return defects, dose


def main() -> None:
    defects, dose = _validate()
    report: dict = {
        "prereg": "docs/experiments/slide_speedup_v1_prereg_20260731.md",
        "threshold_source": "preregistered r@end fractions; theta=250,375,500",
        "dose": dose,
    }
    ROOT.mkdir(parents=True, exist_ok=True)
    if defects:
        report["verdict"] = "VOID_ENGINEERING"
        report["defects"] = defects
        OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        print("VERDICT: VOID_ENGINEERING")
        for defect in defects[:30]:
            print(f"  {defect}")
        raise SystemExit(2)

    curves = {
        arm: {seed: _curve(arm, seed) for seed in SEEDS}
        for arm in ("scratch", "walk")
    }
    report["curves"] = {
        arm: {str(seed): points for seed, points in by_seed.items()}
        for arm, by_seed in curves.items()
    }
    report["thresholds"] = {}
    passes: list[bool] = []
    threshold_medians: dict[float, float] = {}
    for theta in THRESHOLDS:
        rows = []
        speedups = []
        for seed in SEEDS:
            scratch_steps, scratch_censored = _steps_to(
                curves["scratch"][seed], theta
            )
            walk_steps, walk_censored = _steps_to(curves["walk"][seed], theta)
            speedup = scratch_steps / walk_steps
            speedups.append(speedup)
            rows.append(
                {
                    "seed": seed,
                    "steps_scratch": round(scratch_steps),
                    "steps_walk": round(walk_steps),
                    "speedup": round(speedup, 3),
                    "censored": {
                        "scratch": scratch_censored,
                        "walk": walk_censored,
                    },
                }
            )
        median = st.median(speedups)
        threshold_medians[theta] = median
        passed = median >= 2.0 and all(value >= 1.5 for value in speedups)
        passes.append(passed)
        report["thresholds"][str(int(theta))] = {
            "per_seed": rows,
            "speedup_median": round(median, 3),
            "speedup_mean": round(st.mean(speedups), 3),
            "all_seeds_ge_1.5": all(value >= 1.5 for value in speedups),
            "pass": passed,
        }

    endpoint_raw = {
        arm: st.mean(
            value
            for seed in SEEDS
            for step, value in curves[arm][seed]
            if step == STEPS[-1]
        )
        for arm in ("scratch", "walk")
    }
    endpoint = {arm: round(value, 3) for arm, value in endpoint_raw.items()}
    overtaken = endpoint_raw["scratch"] > endpoint_raw["walk"]
    report["endpoint_100k"] = endpoint
    report["scratch_overtook_walk"] = overtaken

    all_low = all(threshold_medians[theta] < 1.5 for theta in THRESHOLDS)
    if sum(passes) >= 2 and not overtaken:
        verdict = "SPEEDUP_CONFIRMED"
    elif all_low or overtaken:
        verdict = "SPEEDUP_REFUTED"
    else:
        verdict = "SPEEDUP_PARTIAL"
    report["verdict"] = verdict
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")

    print("=" * 88)
    print(f"VERDICT: {verdict}")
    print("=" * 88)
    for arm in ("scratch", "walk"):
        for seed in SEEDS:
            print(
                f"{arm:7s} s{seed}: "
                + "  ".join(
                    f"{step // 1000}k={value:.1f}"
                    for step, value in curves[arm][seed]
                )
            )
    for theta in THRESHOLDS:
        result = report["thresholds"][str(int(theta))]
        print(
            f"theta={int(theta):3d} median={result['speedup_median']:.3f} "
            f"per_seed={[row['speedup'] for row in result['per_seed']]} "
            f"{'PASS' if result['pass'] else '----'}"
        )
    print(f"100k endpoint={endpoint}, scratch_overtook_walk={overtaken}")
    print(f"written {OUT}")


if __name__ == "__main__":
    main()
