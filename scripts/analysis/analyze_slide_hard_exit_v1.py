"""Frozen adjudicator for docs/run_card_interventional_bootstrap_racing_v1.md."""
from __future__ import annotations

import glob
import hashlib
import json
import math
import statistics as st
from pathlib import Path

import torch

from fasttd3_ptf.official_fasttd3_ptf.anchor_io import verify_anchor_bundle

DATA = Path("docs/data/slide_hard_exit_v1")
EVAL = DATA / "source_free_eval"
ARTIFACT = Path("artifacts/slide_hard_exit_v1")
OUT = DATA / "slide_hard_exit_v1_results.json"
SEEDS = (1, 2, 3)
STEPS = (30000, 50000, 75000, 100000)
EVAL_BASES = (11, 23, 37, 53, 71, 89, 103, 113,
              131, 149, 163, 179, 193, 211, 227, 241)
RANKS = tuple(range(8))
PANEL = [base * 1000 + rank for base in EVAL_BASES for rank in RANKS]
T90_DF2 = 1.8856


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def interval(values: list[float]) -> dict[str, float | list[float]]:
    mean = st.mean(values)
    sd = st.stdev(values)
    half = T90_DF2 * sd / math.sqrt(len(values))
    return {
        "per_seed": [round(value, 6) for value in values],
        "mean": round(mean, 6),
        "sd": round(sd, 6),
        "lcb90": round(mean - half, 6),
        "ucb90": round(mean + half, 6),
    }


def checkpoint(arm: str, seed: int, step: int) -> Path | None:
    matches = sorted(glob.glob(f"models/*shev1_{arm}_s{seed}__*_{step}.pt"))
    return Path(matches[0]).resolve() if len(matches) == 1 else None


def validate_eval(
    defects: list[str], arm: str, seed: int, step: int, mode: str
) -> float | None:
    tag = f"{arm} s{seed} step{step}"
    path = EVAL / f"{arm}_s{seed}_step{step}.json"
    ckpt = checkpoint(arm, seed, step)
    if ckpt is None:
        defects.append(f"{tag}: checkpoint identity is not unique")
        return None
    if not path.is_file():
        defects.append(f"{tag}: missing eval JSON")
        return None
    try:
        data = json.loads(path.read_text())
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        protocol = data["protocol"]
        episodes = data["episodes"]
        identity = data["checkpoint"]
    except Exception as exc:
        defects.append(f"{tag}: unreadable artifact ({type(exc).__name__})")
        return None
    if data.get("env_name") != "h1hand-slide-v0":
        defects.append(f"{tag}: wrong environment")
    if identity.get("global_step") != step or state.get("global_step") != step:
        defects.append(f"{tag}: wrong completed step")
    if identity.get("identity_checked") is not True:
        defects.append(f"{tag}: evaluator did not check identity")
    if Path(str(identity.get("path", ""))).resolve() != ckpt:
        defects.append(f"{tag}: checkpoint path mismatch")
    if identity.get("sha256") != sha256(ckpt):
        defects.append(f"{tag}: checkpoint SHA mismatch")
    if protocol.get("eval_seeds") != list(EVAL_BASES):
        defects.append(f"{tag}: wrong eval bases")
    if protocol.get("ranks") != list(RANKS):
        defects.append(f"{tag}: wrong eval ranks")
    if protocol.get("episode_steps") != 1000:
        defects.append(f"{tag}: wrong episode horizon")
    if protocol.get("deterministic") is not True:
        defects.append(f"{tag}: nondeterministic evaluator")
    if protocol.get("source_free") != (
        "structural (no bank/option/admission components constructed)"
    ):
        defects.append(f"{tag}: evaluator is not structurally source-free")
    if len(episodes) != 128 or data.get("aggregate", {}).get("episode_count") != 128:
        defects.append(f"{tag}: wrong panel size")
    if [row.get("seed") for row in episodes] != PANEL:
        defects.append(f"{tag}: wrong episode seed panel")
    if not all(math.isfinite(float(row.get("return"))) for row in episodes):
        defects.append(f"{tag}: nonfinite return")
    args = state.get("args") or {}
    expected_args = {
        "env_name": "h1hand-slide-v0",
        "seed": seed,
        "total_timesteps": 100000,
        "num_envs": 128,
        "batch_size": 32768,
        "buffer_size": 51200,
        "learning_starts": 10,
        "num_updates": 2,
        "compile": False,
        "amp": True,
        "torch_deterministic": True,
        "reward_normalization": False,
    }
    for key, expected in expected_args.items():
        if args.get(key) != expected:
            defects.append(f"{tag}: args.{key}={args.get(key)!r}, expected {expected!r}")
    ptf = state.get("ptf_cfg") or {}
    expected_ptf = {
        "source_bank": "configs/source_banks/calibration/h1hand_slide_rbo_walk.yaml",
        "mcg": True,
        "mcg_groups": ["legs_torso", "arms", "hands"],
        "mcg_warmup_steps": 100000,
        "mcg_warmup_min_steps": 25,
        "mcg_warmup_mode": "admission_bootstrap",
        "mcg_ablation": "bootstrap_only",
        "admission_mode": mode,
        "admission_replay_handoff": "physical_after_authority",
        "admission_replay_uniform_mix": 1.0,
        "admission_replay_recency_half_life": 0.0,
        "admission_replay_priority_alpha": 0.0,
        "run_stop_step": 30000 if arm == "prefix" else 100000,
    }
    for key, expected in expected_ptf.items():
        if ptf.get(key) != expected:
            defects.append(f"{tag}: ptf.{key}={ptf.get(key)!r}, expected {expected!r}")
    if state.get("source_names") != ["walk"]:
        defects.append(f"{tag}: wrong source identity")
    return float(data["aggregate"]["return_mean"])


def main() -> None:
    defects: list[str] = []
    curves: dict[str, dict[str, list[list[float]]]] = {
        "cont": {}, "exit": {}
    }
    audit: dict[str, dict] = {}

    for seed in SEEDS:
        anchor = ARTIFACT / "anchors" / f"slide_s{seed}_walk_k30000"
        try:
            manifest = verify_anchor_bundle(anchor)
            learner = torch.load(anchor / "learner.pt", map_location="cpu", weights_only=False)
            replay = torch.load(anchor / "replay.pt", map_location="cpu", weights_only=False)
        except Exception as exc:
            defects.append(f"s{seed}: invalid branch anchor ({type(exc).__name__})")
            continue
        aux = learner.get("auxiliary_state") or {}
        adm = replay.get("admission_sampling") or {}
        if manifest.get("completed_vector_steps") != 30000:
            defects.append(f"s{seed}: wrong anchor step")
        if not aux.get("branch_anchor") or aux.get("source_names") != ["walk"]:
            defects.append(f"s{seed}: wrong branch-anchor identity")
        anchor_exec = [int(x) for x in aux.get("admission_execution_counts", [])]
        anchor_critic_raw = (adm.get("sample_counts") or {}).get("critic")
        anchor_critic = (
            [int(x) for x in anchor_critic_raw.tolist()]
            if isinstance(anchor_critic_raw, torch.Tensor) else []
        )
        if len(anchor_exec) != 2 or len(anchor_critic) != 2:
            defects.append(f"s{seed}: missing anchor treatment counts")
            continue
        for label, counts in (("execution", anchor_exec), ("critic", anchor_critic)):
            share = counts[0] / sum(counts)
            if not 0.45 <= share <= 0.55:
                defects.append(f"s{seed}: anchor {label} share={share:.4f}")

        prefix_value = validate_eval(defects, "prefix", seed, 30000, "all")
        if prefix_value is None:
            continue
        for arm, mode in (("cont", "all"), ("exit", "none")):
            points: list[list[float]] = [[30000, prefix_value]]
            for step in STEPS[1:]:
                value = validate_eval(defects, arm, seed, step, mode)
                if value is not None:
                    points.append([step, value])
            curves[arm][str(seed)] = points

        cont_ckpt = checkpoint("cont", seed, 100000)
        exit_ckpt = checkpoint("exit", seed, 100000)
        if cont_ckpt is None or exit_ckpt is None:
            continue
        cont = torch.load(cont_ckpt, map_location="cpu", weights_only=False)
        stopped = torch.load(exit_ckpt, map_location="cpu", weights_only=False)
        for arm_name, state in (("cont", cont), ("exit", stopped)):
            ptf = state.get("ptf_cfg") or {}
            lineage = ptf.get("anchor_resume_manifest") or {}
            if Path(str(lineage.get("bundle", ""))).resolve() != anchor.resolve():
                defects.append(f"s{seed}: {arm_name} has wrong branch-anchor lineage")
            if ptf.get("resume_noise_seed") != 880000 + seed:
                defects.append(f"s{seed}: {arm_name} has wrong paired resume-noise seed")
        cont_audit = cont.get("admission_audit") or {}
        exit_audit = stopped.get("admission_audit") or {}
        cont_exec = [int(x) for x in cont_audit.get("execution_counts", [])]
        cont_critic = [int(x) for x in cont_audit.get("critic_sample_counts", [])]
        exit_exec = [int(x) for x in exit_audit.get("execution_counts", [])]
        exit_critic = [int(x) for x in exit_audit.get("critic_sample_counts", [])]
        if not all(len(x) == 2 for x in (cont_exec, cont_critic, exit_exec, exit_critic)):
            defects.append(f"s{seed}: malformed final treatment audit")
            continue
        if exit_exec[0] != anchor_exec[0]:
            defects.append(f"s{seed}: source behavior continued after hard exit")
        if exit_critic[0] != anchor_critic[0]:
            defects.append(f"s{seed}: source critic sampling continued after hard exit")
        if (exit_audit.get("active_buffer_counts") or [None])[0] != 0:
            defects.append(f"s{seed}: hard-exit source remains active")
        if exit_audit.get("candidate_masses") != [0.0, 1.0]:
            defects.append(f"s{seed}: hard-exit replay masses are not student-only")
        delta_exec = [cont_exec[i] - anchor_exec[i] for i in range(2)]
        delta_critic = [cont_critic[i] - anchor_critic[i] for i in range(2)]
        for label, counts in (("execution", delta_exec), ("critic", delta_critic)):
            if min(counts) <= 0:
                defects.append(f"s{seed}: continuous {label} did not expose both strata")
            else:
                share = counts[0] / sum(counts)
                if not 0.45 <= share <= 0.55:
                    defects.append(f"s{seed}: continuous delta {label} share={share:.4f}")
        audit[str(seed)] = {
            "anchor_execution": anchor_exec,
            "anchor_critic": anchor_critic,
            "continuous_delta_execution": delta_exec,
            "continuous_delta_critic": delta_critic,
            "hard_exit_delta_execution": [exit_exec[i] - anchor_exec[i] for i in range(2)],
            "hard_exit_delta_critic": [exit_critic[i] - anchor_critic[i] for i in range(2)],
            "hard_exit_main_buffer_counts": exit_audit.get("main_buffer_counts"),
            "hard_exit_active_buffer_counts": exit_audit.get("active_buffer_counts"),
        }

    report: dict = {
        "prereg": "docs/run_card_interventional_bootstrap_racing_v1.md",
        "curves": curves,
        "treatment_audit": audit,
    }
    DATA.mkdir(parents=True, exist_ok=True)
    if defects:
        report["verdict"] = "ENGINEERING_INVALID"
        report["defects"] = defects
        OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        print("VERDICT: ENGINEERING_INVALID")
        for defect in defects:
            print(" -", defect)
        raise SystemExit(2)

    d_exit: list[float] = []
    a_exit: list[float] = []
    d_scratch: list[float] = []
    historical = json.loads(
        Path("docs/data/slide_speedup_v1/slide_speedup_v1_results.json").read_text()
    )
    for seed in SEEDS:
        cont = curves["cont"][str(seed)]
        stopped = curves["exit"][str(seed)]
        d_exit.append(stopped[-1][1] - cont[-1][1])
        auc_cont = sum(
            (cont[i + 1][0] - cont[i][0]) * (cont[i + 1][1] + cont[i][1]) / 2
            for i in range(len(cont) - 1)
        ) / 70000.0
        auc_exit = sum(
            (stopped[i + 1][0] - stopped[i][0])
            * (stopped[i + 1][1] + stopped[i][1]) / 2
            for i in range(len(stopped) - 1)
        ) / 70000.0
        a_exit.append(auc_exit - auc_cont)
        scratch100 = dict(historical["curves"]["scratch"][str(seed)])[100000]
        d_scratch.append(stopped[-1][1] - float(scratch100))

    stats = {
        "D_exit_endpoint": interval(d_exit),
        "A_exit_nAUC_30k_100k": interval(a_exit),
        "D_scratch_endpoint": interval(d_scratch),
    }
    report["statistics"] = stats
    d = stats["D_exit_endpoint"]
    a = stats["A_exit_nAUC_30k_100k"]
    s = stats["D_scratch_endpoint"]
    if d["ucb90"] < 0 or a["ucb90"] < 0 or (
        d["lcb90"] <= 0 and a["lcb90"] <= 0
    ):
        verdict = "HARD_EXIT_REFUTED"
    elif d["lcb90"] > 0 and a["lcb90"] > 0 and s["ucb90"] >= 0:
        verdict = "HARD_EXIT_SUPPORTED"
    else:
        verdict = "HARD_EXIT_PARTIAL"
    report["verdict"] = verdict
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print("VERDICT:", verdict)
    for name, values in stats.items():
        print(name, values)


if __name__ == "__main__":
    main()
