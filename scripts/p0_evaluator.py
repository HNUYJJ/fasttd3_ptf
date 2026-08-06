"""P0 独立冻结 evaluator(run card §6;五次复核阻塞问题 2)。

结构性 source-free:本脚本只加载 actor + 冻结 obs normalizer,从不构建
bank/option/MCG/admission 组件——eval 路径不可能触碰 source。

协议(冻结):
- 面板 = eval seeds × ranks(默认 4×8,lease/abstain 分支使用完全相同面板);
- 每 episode:独立创建 env,双播种 reset(np.random.seed + reset(seed=...),
  E11:legacy unwrapped.seed 只播 NumPy 全局,basketball 类任务又用全局——
  与 humanoid_bench_env._SeededResetWrapper 同语义);
- deterministic actor(无探索噪声),episode 1000 步(crawl/truck 标准长度);
- 指标(§6 精确定义):
  return         = episode 奖励和(全任务 primary);
  crawl progress = max_t(x_t − x_0),x=root x 坐标(qpos[0]);confirmatory;
  crawl posture  = min(crawling, crawling_head) 的 episode mean(描述性);
  truck progress = info["reward_robot_package_truck"] 的 episode mean(描述性);
  truck success  = get_terminated 语义=任务成功,单独报告(描述性,预期恒 0)。

用法:
  python scripts/p0_evaluator.py --checkpoint models/<branch>_13000.pt \
      --env-name h1hand-crawl-v0 --out <dir>/eval_13000.json
输出 JSON 含 per-episode 明细与面板聚合,附 checkpoint/git 溯源。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# HB env 默认 render_mode="rgb_array",headless 节点须在 mujoco 加载前指定
# EGL(项目全部离线脚本的既有惯例)。
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from probe_lib import load_student  # noqa: E402

EVAL_SEEDS = (11, 23, 37, 53)      # 冻结:4 eval seeds(默认面板 32 episodes)
RANKS = tuple(range(8))            # 冻结:8 ranks
EPISODE_STEPS = 1000               # 冻结:crawl/truck 均为标准 1000 步

# 扩展面板(Door@10k gate 起启用):Cabinet gate 暴露出 32 episodes 的面板 SE 会
# 淹没干预效应,故允许用 --eval-seeds 传入更长的 seed 列表。**前 4 个必须保持
# (11,23,37,53)**——循环顺序是 for eval_seed: for rank,故前 32 个 episode 的
# reset seed 与既有 32-episode 面板逐位相同,构成向后兼容的 secondary 子面板。
EVAL_SEEDS_128 = (11, 23, 37, 53, 71, 89, 103, 113,
                  131, 149, 163, 179, 193, 211, 227, 241)   # 16 × 8 = 128


def _make_env(env_name: str):
    from fasttd3_ptf.official_fasttd3_ptf.paths import ensure_humanoidbench_import_path

    ensure_humanoidbench_import_path()
    import gymnasium as gym
    import humanoid_bench  # noqa: F401
    from gymnasium.wrappers import TimeLimit

    env = gym.make(env_name)
    return TimeLimit(env, max_episode_steps=EPISODE_STEPS)


@torch.no_grad()
def _run_episode(env, actor, obs_norm, device, seed: int) -> dict:
    # 双播种(E11 语义):全局 NumPy + Gymnasium np_random。
    np.random.seed(seed)
    obs, _ = env.reset(seed=seed)
    x0 = float(env.unwrapped.data.qpos[0])
    total_return = 0.0
    max_progress = 0.0
    posture_terms: list[float] = []
    package_terms: list[float] = []
    success = False
    for _ in range(EPISODE_STEPS):
        obs_tensor = torch.as_tensor(obs, device=device, dtype=torch.float32).unsqueeze(0)
        action = actor(obs_norm(obs_tensor)).squeeze(0).cpu().numpy()
        obs, reward, terminated, truncated, info = env.step(action)
        total_return += float(reward)
        max_progress = max(max_progress, float(env.unwrapped.data.qpos[0]) - x0)
        if "crawling" in info and "crawling_head" in info:
            posture_terms.append(min(float(info["crawling"]), float(info["crawling_head"])))
        if "reward_robot_package_truck" in info:
            package_terms.append(float(info["reward_robot_package_truck"]))
        if terminated:
            success = True
            break
        if truncated:
            break
    return {
        "seed": seed,
        "return": total_return,
        "progress_max_dx": max_progress,
        "posture_mean": float(np.mean(posture_terms)) if posture_terms else None,
        "package_reward_mean": float(np.mean(package_terms)) if package_terms else None,
        "terminated_success": success,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--env-name", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    # 身份验证(六次复核高优 2):正式 P0 评估必须显式声明预期身份,
    # checkpoint 与声明不符即拒绝——防止喂错 checkpoint 静默产出合法外观的
    # eval JSON。toy/调试评估可不传(输出中会标记 identity_checked=false)。
    parser.add_argument("--expect-global-step", type=int, default=None)
    parser.add_argument("--expect-seed", type=int, default=None)
    parser.add_argument("--expect-admission-mode", default=None,
                        help="lease 分支=all,abstain 分支=none")
    parser.add_argument("--eval-seeds", default=None,
                        help="逗号分隔的 eval seed 列表;传 'panel128' 用冻结的 16-seed "
                             "扩展面板(128 episodes,前 32 个与默认面板逐位相同)。"
                             "不传则用冻结的默认 4-seed 面板(32 episodes)。")
    args = parser.parse_args()

    if args.eval_seeds is None:
        eval_seeds = EVAL_SEEDS
    elif args.eval_seeds == "panel128":
        eval_seeds = EVAL_SEEDS_128
    else:
        eval_seeds = tuple(int(x) for x in args.eval_seeds.split(","))
    if eval_seeds[:len(EVAL_SEEDS)] != EVAL_SEEDS:
        raise ValueError(
            f"eval seed 列表的前 {len(EVAL_SEEDS)} 个必须是 {EVAL_SEEDS},否则前 32 个 "
            f"episode 无法与既有面板兼容;收到 {eval_seeds[:len(EVAL_SEEDS)]}"
        )

    device = torch.device(args.device)
    checkpoint = Path(args.checkpoint).resolve()
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    ckpt_args = state.get("args") or {}
    identity_errors = []
    if ckpt_args.get("env_name") != args.env_name:
        identity_errors.append(
            f"checkpoint env_name={ckpt_args.get('env_name')} != --env-name {args.env_name}"
        )
    if args.expect_global_step is not None and state.get("global_step") != args.expect_global_step:
        identity_errors.append(
            f"checkpoint global_step={state.get('global_step')} != expected {args.expect_global_step}"
        )
    if args.expect_seed is not None and ckpt_args.get("seed") != args.expect_seed:
        identity_errors.append(
            f"checkpoint seed={ckpt_args.get('seed')} != expected {args.expect_seed}"
        )
    if args.expect_admission_mode is not None:
        actual_mode = (state.get("ptf_cfg") or {}).get("admission_mode")
        if actual_mode != args.expect_admission_mode:
            identity_errors.append(
                f"checkpoint admission_mode={actual_mode} != expected {args.expect_admission_mode}"
            )
    if identity_errors:
        raise ValueError("checkpoint identity mismatch: " + "; ".join(identity_errors))
    del state
    identity_checked = all(
        value is not None
        for value in (args.expect_global_step, args.expect_seed, args.expect_admission_mode)
    )
    actor, _critic, obs_norm, _critic_norm, global_step = load_student(str(checkpoint), device)

    episodes = []
    env = _make_env(args.env_name)
    try:
        for eval_seed in eval_seeds:
            for rank in RANKS:
                # (seed, rank) → 唯一 reset seed;面板冻结,分支间逐位相同。
                episodes.append(_run_episode(env, actor, obs_norm, device, eval_seed * 1000 + rank))
    finally:
        env.close()

    returns = [e["return"] for e in episodes]
    progress = [e["progress_max_dx"] for e in episodes]
    postures = [e["posture_mean"] for e in episodes if e["posture_mean"] is not None]
    packages = [e["package_reward_mean"] for e in episodes if e["package_reward_mean"] is not None]
    git_head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False,
    ).stdout.strip()
    report = {
        "protocol": {
            "eval_seeds": list(eval_seeds),
            "ranks": list(RANKS),
            "compat_subpanel_episodes": len(EVAL_SEEDS) * len(RANKS),
            "episode_steps": EPISODE_STEPS,
            "deterministic": True,
            "source_free": "structural (no bank/option/admission components constructed)",
        },
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": _sha256(checkpoint),
            "global_step": global_step,
            "identity_checked": identity_checked,
        },
        "env_name": args.env_name,
        "git_head": git_head,
        "utc": datetime.now(timezone.utc).isoformat(),
        "episodes": episodes,
        "aggregate": {
            "return_mean": float(np.mean(returns)),
            "return_std": float(np.std(returns, ddof=1)),
            "progress_max_dx_mean": float(np.mean(progress)),
            "posture_mean": float(np.mean(postures)) if postures else None,
            "package_reward_mean": float(np.mean(packages)) if packages else None,
            "success_count": int(sum(e["terminated_success"] for e in episodes)),
            "episode_count": len(episodes),
        },
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        raise FileExistsError(f"refusing to overwrite existing eval artifact: {out_path}")
    out_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"[p0_evaluator] {args.env_name} step={global_step}: "
          f"return_mean={report['aggregate']['return_mean']:.2f} "
          f"({len(episodes)} episodes) -> {out_path}")


if __name__ == "__main__":
    main()
