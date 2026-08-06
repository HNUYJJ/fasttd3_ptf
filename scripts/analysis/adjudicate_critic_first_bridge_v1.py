#!/usr/bin/env python3
"""Engineering audit and frozen single-seed verdict for critic-first bridge v1."""
from __future__ import annotations

import json
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
MODEL_ROOT = ROOT / "models"
EVAL_ROOT = ROOT / "docs/data/critic_first_bridge_v1/source_free_eval"
OUT = ROOT / "docs/data/critic_first_bridge_v1/feasibility_result.json"
ARMS = ("student_freeze", "interleaved", "critic_first")
TASK_SOURCE = {"slide": "walk", "door": "run"}


def _checkpoint(task: str, arm: str, step: int) -> dict:
    path = MODEL_ROOT / f"h1hand-{task}-v0__cfb_{task}_{arm}_s1__1_{step}.pt"
    if not path.is_file():
        raise FileNotFoundError(path)
    return torch.load(path, map_location="cpu", weights_only=False)


def _anchor_counts(task: str) -> tuple[int, int]:
    path = ROOT / f"artifacts/{task}_bac_gate_v1/anchors/s1/learner.pt"
    if task == "door":
        path = ROOT / "artifacts/door_at10k_gate_v1/anchors/s1/learner.pt"
    state = torch.load(path, map_location="cpu", weights_only=False)
    audit = state["auxiliary_state"]
    return int(audit["critic_update_count"]), int(audit["actor_update_count"])


def _source_counts_at_exit(audit: dict, step: int) -> tuple[list[int], list[int]]:
    decisions = [x for x in audit["decision_history"] if int(x["step"]) == step]
    if len(decisions) != 1 or not decisions[0]["exact_abstain"]:
        raise AssertionError(f"expected one exact-abstain decision at {step}")
    exec_counts = [int(x) for x in decisions[0]["execution_counts_at_apply"]]
    events = [
        x for x in audit["policy_events"]
        if x["event"] == "admission_policy" and int(x["replay_ptr"]) == step
    ]
    if len(events) != 1:
        raise AssertionError(f"expected one replay policy event at {step}")
    critic_counts = [int(x) for x in events[0]["sample_counts_at_apply"]["critic"]]
    return exec_counts, critic_counts


def _share(counts: list[int]) -> float:
    if len(counts) != 2 or sum(counts) <= 0:
        raise AssertionError(f"invalid source/student counts: {counts}")
    return counts[0] / sum(counts)


def _eval(task: str, arm: str) -> tuple[float, dict]:
    path = EVAL_ROOT / f"{task}_{arm}_s1_step20000.json"
    report = json.loads(path.read_text())
    protocol = report["protocol"]
    if not protocol["deterministic"] or "structural" not in protocol["source_free"]:
        raise AssertionError(f"{path}: evaluator is not deterministic/source-free")
    if len(report["episodes"]) != 128 or protocol["episode_steps"] != 1000:
        raise AssertionError(f"{path}: expected frozen 128 x 1000 panel")
    if not report["checkpoint"]["identity_checked"]:
        raise AssertionError(f"{path}: checkpoint identity was not checked")
    return float(report["aggregate"]["return_mean"]), report


def main() -> None:
    engineering: dict[str, dict] = {}
    returns: dict[str, dict[str, float]] = {}
    for task, source in TASK_SOURCE.items():
        anchor_critic, anchor_actor = _anchor_counts(task)
        engineering[task] = {}
        returns[task] = {}
        for arm in ARMS:
            bridge = _checkpoint(task, arm, 12000)
            final = _checkpoint(task, arm, 20000)
            if bridge["global_step"] != 12000 or final["global_step"] != 20000:
                raise AssertionError(f"{task}/{arm}: checkpoint step mismatch")
            btrain = bridge["training_audit"]
            ftrain = final["training_audit"]
            if int(btrain["critic_update_count"]) != anchor_critic + 4000:
                raise AssertionError(f"{task}/{arm}: critic did not receive 4000 bridge updates")
            expected_bridge_actor = anchor_actor if arm != "interleaved" else anchor_actor + 2000
            expected_final_actor = anchor_actor + (8000 if arm != "interleaved" else 10000)
            if int(btrain["actor_update_count"]) != expected_bridge_actor:
                raise AssertionError(f"{task}/{arm}: actor bridge update count mismatch")
            if int(ftrain["actor_update_count"]) != expected_final_actor:
                raise AssertionError(f"{task}/{arm}: actor final update count mismatch")

            arm_audit: dict[str, object] = {
                "anchor_critic_updates": anchor_critic,
                "anchor_actor_updates": anchor_actor,
                "bridge_training_audit": btrain,
                "final_training_audit": ftrain,
            }
            if arm != "student_freeze":
                audit = final["admission_audit"]
                if final["source_names"] != [source]:
                    raise AssertionError(f"{task}/{arm}: source identity mismatch")
                exec_at_exit, critic_at_exit = _source_counts_at_exit(audit, 12000)
                behavior_share = _share(exec_at_exit)
                critic_share = _share(critic_at_exit)
                if not 0.45 <= behavior_share <= 0.55:
                    raise AssertionError(f"{task}/{arm}: behavior share {behavior_share:.4f}")
                if not 0.45 <= critic_share <= 0.55:
                    raise AssertionError(f"{task}/{arm}: critic share {critic_share:.4f}")
                if int(audit["execution_counts"][0]) != exec_at_exit[0]:
                    raise AssertionError(f"{task}/{arm}: source behavior continued after hard exit")
                if int(audit["critic_sample_counts"][0]) != critic_at_exit[0]:
                    raise AssertionError(f"{task}/{arm}: source replay continued after hard exit")
                if audit["source_authority_active"] or int(audit["active_buffer_counts"][0]) != 0:
                    raise AssertionError(f"{task}/{arm}: source remained active after hard exit")
                arm_audit.update({
                    "source_behavior_counts_at_exit": exec_at_exit,
                    "source_critic_counts_at_exit": critic_at_exit,
                    "behavior_source_share": behavior_share,
                    "critic_source_share": critic_share,
                    "post_exit_source_execution_delta": 0,
                    "post_exit_source_critic_delta": 0,
                })
            engineering[task][arm] = arm_audit
            returns[task][arm], _ = _eval(task, arm)

    slide = returns["slide"]
    door = returns["door"]
    slide_pass = (
        slide["critic_first"] > slide["interleaved"]
        and slide["critic_first"] > slide["student_freeze"]
    )
    door_vs_interleaved = door["critic_first"] > door["interleaved"]
    door_safe = door["critic_first"] >= door["student_freeze"]
    if slide_pass and door_vs_interleaved and door_safe:
        verdict = "DUAL_GATE_PASS"
    elif slide_pass and door_vs_interleaved:
        verdict = "POSITIVE_ONLY"
    else:
        verdict = "FAIL"

    result = {
        "protocol": "critic-first bridge v1; single-seed feasibility only",
        "engineering_gate": "PASS",
        "returns": returns,
        "contrasts": {
            task: {
                "critic_first_minus_interleaved": values["critic_first"] - values["interleaved"],
                "critic_first_minus_student_freeze": values["critic_first"] - values["student_freeze"],
            }
            for task, values in returns.items()
        },
        "decision_bits": {
            "slide_positive_preserved": slide_pass,
            "door_better_than_interleaved": door_vs_interleaved,
            "door_not_worse_than_student_freeze": door_safe,
        },
        "verdict": verdict,
        "engineering": engineering,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUT}")
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
