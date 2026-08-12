#!/usr/bin/env python3
"""Phase 0 独立评估器：deterministic actor，两臂各用其自身的动作接口。

判据冻结于 `docs/phase0_prereg_v2_20260812.md`。本脚本在看到任何 return 之前写就。

为什么独立评估而不读 wandb：`train_interface.py` 的 eval 结果只进 `logs` dict
上报 wandb、不落 stdout；独立评估让口径完全可控，并能同时取出预注册要求的
诊断指标（`hand_dist` / `target_dist` / `success`）。

臂身份由 checkpoint 的 `args.exp_name` 决定，**不由命令行指定**——
避免用错接口评估（iface 的 actor 输出 3 维，若按 flat 构造 61 维环境会静默错配）。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "vendor" / "fast_td3"))

from hb_interface_env import make_env  # noqa: E402

EPISODE_STEPS = 500          # push 的 max_episode_steps（vendor 短 episode 表）
PANEL = [s * 1000 + r for s in (11, 23, 37, 53) for r in range(8)]   # 32 episodes


class FrozenNorm:
    """复刻 train 时的 obs normalizer（EmpiricalNormalization 的推理路径）。"""

    def __init__(self, state, device):
        self.mean = torch.as_tensor(state["_mean"], device=device, dtype=torch.float32)
        self.std = torch.as_tensor(state["_std"], device=device, dtype=torch.float32)

    def __call__(self, x):
        return (x - self.mean) / (self.std + 1e-8)


def load_actor(ckpt_path: str, device):
    from fast_td3 import Actor
    st = torch.load(ckpt_path, map_location=device, weights_only=False)
    args = st["args"]
    sd = {k: v for k, v in st["actor_state_dict"].items() if k != "noise_scales"}
    n_obs = int(torch.as_tensor(st["obs_normalizer_state"]["_mean"]).shape[-1])
    n_act = int(sd["net.0.weight"].shape[0]) if False else None
    # 从最后一层权重推动作维度，避免依赖外部信息
    last = [k for k in sd if k.endswith(".weight")][-1]
    n_act = int(sd[last].shape[0])
    actor = Actor(
        n_obs=n_obs, n_act=n_act, num_envs=1, device=device,
        init_scale=float(args.get("init_scale", 0.01)),
        hidden_dim=int(args.get("actor_hidden_dim", 512)),
        std_min=float(args.get("std_min", 0.001)),
        std_max=float(args.get("std_max", 0.4)),
    ).to(device)
    actor.load_state_dict(sd, strict=False)
    actor.eval()
    norm = FrozenNorm(st["obs_normalizer_state"], device)
    return actor, norm, args, n_act


@torch.no_grad()
def run_episode(env, actor, norm, device, seed: int) -> dict:
    np.random.seed(seed)
    obs, _ = env.reset(seed=seed)
    total, hand, tgt, succ = 0.0, [], [], False
    for _ in range(EPISODE_STEPS):
        o = torch.as_tensor(obs, device=device, dtype=torch.float32).unsqueeze(0)
        a = actor(norm(o)).squeeze(0).cpu().numpy()
        obs, r, term, trunc, info = env.step(a)
        total += float(r)
        if "hand_dist" in info:
            hand.append(float(info["hand_dist"]))
        if "target_dist" in info:
            tgt.append(float(info["target_dist"]))
        if info.get("success"):
            succ = True
        if term or trunc:
            break
    return {"seed": seed, "return": total, "success": succ,
            "hand_dist_mean": float(np.mean(hand)) if hand else None,
            "target_dist_mean": float(np.mean(tgt)) if tgt else None,
            "target_dist_min": float(np.min(tgt)) if tgt else None}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    actor, norm, args, n_act = load_actor(a.checkpoint, device)

    exp = str(args.get("exp_name", ""))
    if "_iface_" in exp:
        ptype, expect = "reach_single", 3
    elif "_flat_" in exp:
        ptype, expect = None, 61
    else:
        print(f"[FATAL] 无法从 exp_name='{exp}' 判定臂身份"); return 2
    if n_act != expect:
        print(f"[FATAL] 动作维度不符：actor={n_act} 期望={expect}（臂={exp}）"); return 2

    env = make_env("h1hand-push-v0", policy_type=ptype, max_episode_steps=EPISODE_STEPS)
    eps = [run_episode(env, actor, norm, device, s) for s in PANEL]

    def m(k):
        v = [e[k] for e in eps if e[k] is not None]
        return float(np.mean(v)) if v else None

    out = {
        "prereg": "docs/phase0_prereg_v2_20260812.md",
        "checkpoint": a.checkpoint, "exp_name": exp,
        "arm": "iface" if ptype else "flat",
        "policy_type": ptype or "flat", "n_act": n_act,
        "global_step": int(torch.load(a.checkpoint, map_location="cpu",
                                      weights_only=False).get("global_step", -1)),
        "episode_steps": EPISODE_STEPS, "episode_count": len(eps),
        "aggregate": {
            "return_mean": m("return"),
            "return_std": float(np.std([e["return"] for e in eps], ddof=1)),
            "success_rate": float(np.mean([e["success"] for e in eps])),
            "hand_dist_mean": m("hand_dist_mean"),
            "target_dist_mean": m("target_dist_mean"),
            "target_dist_min_mean": m("target_dist_min"),
        },
        "episodes": eps,
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1))
    g = out["aggregate"]
    print(f"[{exp}] n_act={n_act} return={g['return_mean']:.2f} "
          f"succ={g['success_rate']:.3f} hand={g['hand_dist_mean']:.4f} "
          f"tgt={g['target_dist_mean']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
