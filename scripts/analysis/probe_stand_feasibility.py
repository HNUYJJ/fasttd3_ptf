#!/usr/bin/env python3
"""Stand-only 可行性前置探针：source 在 truck 场景的 zero-shot 姿态维持能力。

判据冻结于 `docs/experiments/stand_feasibility_probe_prereg_20260811.md`，
本脚本写于任何探针输出之前。零训练、纯前向，不改动任何 verdict。

关键设计：源动作经其 bank 内**声明的** obs adapter 得到，与训练时同一条链路
（`SourcePolicy.act`），避免我另写一份 adapter 造成口径分叉。
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

from p0_evaluator import _make_env, EPISODE_STEPS  # noqa: E402
from fasttd3_ptf.ptf.source_bank import SourcePolicyBank  # noqa: E402

PANEL = [s * 1000 + r for s in (11, 23) for r in range(8)]
BANK = "configs/source_banks/h1hand_hurdle4_wfix_truck.yaml"
ACTION_DIM = 61


@torch.no_grad()
def run_episode(env, bank, src_idx: int, device, seed: int) -> dict:
    np.random.seed(seed)
    obs, _ = env.reset(seed=seed)
    x0 = float(env.unwrapped.data.qpos[0])
    upr, pkg, rew, max_dx, prev_x, path = [], [], 0.0, 0.0, x0, 0.0
    act_log = []
    for t in range(EPISODE_STEPS):
        obs_t = torch.as_tensor(obs, device=device, dtype=torch.float32).unsqueeze(0)
        all_actions, _ = bank.act_all(obs_t)          # (1, n_src, 61)
        action = all_actions[0, src_idx].cpu().numpy()
        if t < 100:
            act_log.append(action.copy())
        obs, reward, terminated, truncated, info = env.step(action)
        rew += float(reward)
        if "upright" in info:
            upr.append(float(info["upright"]))
        if "reward_robot_package_truck" in info:
            pkg.append(float(info["reward_robot_package_truck"]))
        x = float(env.unwrapped.data.qpos[0])
        path += abs(x - prev_x); prev_x = x
        max_dx = max(max_dx, x - x0)
        if terminated or truncated:
            break
    A = np.asarray(act_log)
    return {
        "seed": seed, "return": rew,
        "upright_mean": float(np.mean(upr)) if upr else None,
        "package_mean": float(np.mean(pkg)) if pkg else None,
        "progress_max_dx": max_dx, "path_length": path,
        "action_std_over_time": float(A.std(axis=0).mean()) if len(A) > 1 else 0.0,
        "action_absmax": float(np.abs(A).max()) if len(A) else 0.0,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/data/stand_feasibility_v1/probe.json")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bank = SourcePolicyBank.from_config(BANK, device=device, target_action_dim=ACTION_DIM)
    names = [s.name for s in bank.sources]
    print(f"bank 源顺序: {names}")

    env = _make_env("h1hand-truck-v0")
    out = {
        "prereg": "docs/experiments/stand_feasibility_probe_prereg_20260811.md",
        "status": "DIAGNOSTIC_ZERO_TRAINING",
        "bank": BANK, "source_order": names,
        "u_scratch_reference": 0.8776,
        "conditions": {},
    }
    for i, nm in enumerate(names):
        eps = [run_episode(env, bank, i, device, s) for s in PANEL]
        def m(k):
            v = [e[k] for e in eps if e[k] is not None]
            return float(np.mean(v)) if v else None
        rec = {k: m(k) for k in ("return", "upright_mean", "package_mean",
                                 "progress_max_dx", "path_length",
                                 "action_std_over_time", "action_absmax")}
        rec["episodes"] = eps
        out["conditions"][nm] = rec
        print(f"[{nm:<8}] upright={rec['upright_mean']:.4f} return={rec['return']:8.1f} "
              f"dx={rec['progress_max_dx']:6.3f} path={rec['path_length']:7.2f} "
              f"a_std={rec['action_std_over_time']:.5f} a_max={rec['action_absmax']:.3f}")

    # adapter 自检（预注册 §2 前置）
    bad = [n for n, r in out["conditions"].items() if r["action_std_over_time"] < 1e-6]
    out["adapter_check"] = {"constant_action_sources": bad,
                            "pass": len(bad) == 0}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"\nadapter 自检: {'PASS' if not bad else 'FAIL ' + str(bad)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
