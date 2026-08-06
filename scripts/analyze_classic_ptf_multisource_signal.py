"""Frozen-rollout Q/beta diagnostic for a classic-PTF multi-source checkpoint."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from analyze_classic_ptf_signal_offline import (  # noqa: E402
    _basic_summary,
    _collect_frozen_rollout_states,
    _conditional_beta_summary,
)
from probe_lib import load_student  # noqa: E402
from fasttd3_ptf.ptf.option_module import OptionModule  # noqa: E402
from fasttd3_ptf.ptf.option_update import termination_margin  # noqa: E402


@torch.no_grad()
def analyze(checkpoint: Path, device: torch.device) -> dict:
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    names = list(state["source_names"])
    actor, _critic, obs_norm, _critic_norm, step = load_student(
        str(checkpoint), device
    )
    module = OptionModule(**state["option_kwargs"]).to(device)
    module.load_state_dict(state["option_state_dict"], strict=True)
    module.eval()
    raw_obs, episodes = _collect_frozen_rollout_states(
        str((state.get("args") or {})["env_name"]), actor, obs_norm, device
    )

    q_parts: list[torch.Tensor] = []
    beta_parts: list[torch.Tensor] = []
    logit_parts: list[torch.Tensor] = []
    for begin in range(0, raw_obs.shape[0], 2048):
        obs = obs_norm(raw_obs[begin : begin + 2048])
        hidden = module.trunk(obs)
        logit_parts.append(module.beta_head(hidden).cpu())
        q, beta = module(obs)
        q_parts.append(q.cpu())
        beta_parts.append(beta.cpu())
    q = torch.cat(q_parts)
    beta = torch.cat(beta_parts)
    logits = torch.cat(logit_parts)
    greedy = q.argmax(dim=1)
    margin = termination_margin(q, xi=0.0)
    top2 = torch.topk(q, k=2, dim=1).values
    q_gap = top2[:, 0] - top2[:, 1]
    all_options_saturated = (q.abs() > 0.95).all(dim=1)

    options = {}
    for index, name in enumerate(names):
        conditioned = _conditional_beta_summary(beta, q, index)
        greedy_mean = conditioned["when_greedy"]["mean"]
        non_greedy_mean = conditioned["when_non_greedy"]["mean"]
        raw_advantage = q[:, index] - q.max(dim=1).values + margin
        if module.released_code_fidelity:
            training_advantage = raw_advantage
        else:
            training_advantage = raw_advantage.clamp(
                min=-margin.clamp_min(1e-6),
                max=margin.clamp_min(1e-6),
            )
        options[name] = {
            "q": _basic_summary(q[:, index]),
            "argmax_fraction": float((greedy == index).float().mean()),
            "beta": _basic_summary(beta[:, index]),
            "beta_logit": _basic_summary(logits[:, index]),
            "termination_advantage": _basic_summary(training_advantage),
            "termination_up_fraction": float((training_advantage < 0).float().mean()),
            "termination_down_fraction": float((training_advantage > 0).float().mean()),
            "beta_conditioned": conditioned,
            "non_greedy_minus_greedy": (
                None
                if greedy_mean is None or non_greedy_mean is None
                else float(non_greedy_mean - greedy_mean)
            ),
        }

    return {
        "checkpoint": str(checkpoint.resolve()),
        "global_step": int(step),
        "source_names": names,
        "state_count": int(raw_obs.shape[0]),
        "episodes": episodes,
        "aggregate": {
            "mean_abs_q": float(q.abs().mean()),
            "all_options_abs_gt_0_95_fraction": float(
                all_options_saturated.float().mean()
            ),
            "any_option_abs_gt_0_95_fraction": float(
                (q.abs() > 0.95).any(dim=1).float().mean()
            ),
            "q_gap": _basic_summary(q_gap),
            "gap_lt_0_01_fraction": float((q_gap < 0.01).float().mean()),
            "all_options_saturated_low_gap_fraction": float(
                (all_options_saturated & (q_gap < 0.01)).float().mean()
            ),
        },
        "options": options,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(f"refusing to overwrite {out}")
    result = analyze(Path(args.checkpoint), torch.device(args.device))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote multi-source signal diagnostic -> {out}")


if __name__ == "__main__":
    main()
