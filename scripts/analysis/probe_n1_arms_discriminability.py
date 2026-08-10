#!/usr/bin/env python3
"""N1 辨别力探针：arms 动作对 truck 早期 reward 是否存在因果通路。

判据冻结于 `docs/experiments/n1_discriminability_probe_prereg_20260810.md`，
本脚本在任何探针输出产生**之前**写就。DIAGNOSTIC，不改动 N1 的任何 verdict。

纯前向 rollout：加载已训练好的 actor，在 `env.step()` 之前按动作子空间扰动，
比较 return。`rand_legs_torso` 是方法有效性对照——它若不掉，说明扰动没生效。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from probe_lib import load_student  # noqa: E402
from p0_evaluator import _make_env, EPISODE_STEPS  # noqa: E402

# 与 p0_evaluator 的 panel 前 16 个 episode 逐位相同（循环序 for eval_seed: for rank）
PANEL = [s * 1000 + r for s in (11, 23) for r in range(8)]

# action_schema.py:51-56 —— legs 0-10, torso 10-11, arms 11-21, hands 21-61
SLICES = {"legs_torso": (0, 11), "arms": (11, 21)}

# 预注册 §3 的冻结阈值
VALIDITY_MIN = 0.20      # rand_legs_torso 至少要掉这么多，否则 PROBE_INVALID
NEGLIGIBLE_MAX = 0.05    # arms 两条件都低于此 → ARMS_PATHWAY_NEGLIGIBLE
SUBSTANTIAL_MIN = 0.15   # rand_arms 高于此 → ARMS_PATHWAY_SUBSTANTIAL


@torch.no_grad()
def run_episode(env, actor, obs_norm, device, seed: int, condition: str) -> float:
    """跑一个 episode，按 condition 扰动动作后再 step。返回 episode return。"""
    np.random.seed(seed)
    obs, _ = env.reset(seed=seed)
    # 扰动用独立 RNG，随 seed 与 condition 确定 —— 可复现且不污染环境流
    rng = np.random.RandomState(seed * 31 + hash(condition) % 1000)
    total = 0.0
    for _ in range(EPISODE_STEPS):
        obs_t = torch.as_tensor(obs, device=device, dtype=torch.float32).unsqueeze(0)
        action = actor(obs_norm(obs_t)).squeeze(0).cpu().numpy()
        if condition != "intact":
            kind, _, group = condition.partition("_")
            lo, hi = SLICES[group]
            if kind == "rand":
                action[lo:hi] = rng.uniform(-1.0, 1.0, size=hi - lo)
            elif kind == "zero":
                action[lo:hi] = 0.0
            else:
                raise ValueError(f"未知扰动类型: {condition}")
        obs, reward, terminated, truncated, _ = env.step(action)
        total += float(reward)
        if terminated or truncated:
            break
    return total


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    actor, _, obs_norm, _ = load_student(args.checkpoint, device)[:4]
    env = _make_env("h1hand-truck-v0")

    conditions = ["intact", "rand_arms", "zero_arms", "rand_legs_torso"]
    per_cond: dict[str, list[float]] = {}
    for cond in conditions:
        per_cond[cond] = [run_episode(env, actor, obs_norm, device, s, cond)
                          for s in PANEL]

    base = float(np.mean(per_cond["intact"]))
    out = {
        "prereg": "docs/experiments/n1_discriminability_probe_prereg_20260810.md",
        "status": "DIAGNOSTIC",
        "checkpoint": args.checkpoint,
        "arm": args.arm,
        "learner_seed": args.seed,
        "panel_episodes": len(PANEL),
        "intact_mean": base,
        "conditions": {},
    }
    for cond in conditions:
        m = float(np.mean(per_cond[cond]))
        out["conditions"][cond] = {
            "mean": m,
            "sd": float(np.std(per_cond[cond], ddof=1)),
            "delta": m - base,
            "rel_delta": abs(m - base) / base if base else None,
            "per_episode": [round(x, 3) for x in per_cond[cond]],
        }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"[{args.arm} s{args.seed}] intact={base:.1f} " + " ".join(
        f"{c}={out['conditions'][c]['mean']:.1f}(δ={out['conditions'][c]['rel_delta']:.3f})"
        for c in conditions[1:]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
