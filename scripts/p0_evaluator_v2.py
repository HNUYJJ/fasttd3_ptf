"""P0 独立冻结 evaluator —— schema v2。

契约冻结于 docs/experiments/evaluator_schema_v2_prereg_20260806.md。

与 v1（scripts/p0_evaluator.py）的**唯一**差别是任务语义层：

    v1   if terminated: success = True     → terminated_success
    v2   环境事实（terminated/truncated）与任务语义（task_success）分离，
         语义一律由 fasttd3_ptf.evaluation.task_metrics 的 registry 判定。

**数值路径必须与 v1 逐位相同**：相同的面板常量、相同的双播种、相同的步进循环、
相同的 break 条件。T4 集成测试对同一 checkpoint 跑两条路径并逐 episode 比对
return / progress_max_dx / episode_length / reset seed —— 这条测试防的是
"改口径导致数字变化"被误当成"修好了 bug"。

结构性 source-free：只加载 actor + 冻结 obs normalizer，从不构建
bank/option/MCG/admission 组件。
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

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np  # noqa: E402
import torch  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from fasttd3_ptf.evaluation import schema_v2, task_metrics  # noqa: E402
from probe_lib import load_student  # noqa: E402

# ── 面板常量：必须与 v1 逐位相同，禁止在此处"顺手改进" ──────────────
EVAL_SEEDS = (11, 23, 37, 53)
EVAL_SEEDS_128 = (11, 23, 37, 53, 71, 89, 103, 113,
                  131, 149, 163, 179, 193, 211, 227, 241)
RANKS = tuple(range(8))
EPISODE_STEPS = 1000


def _make_env(env_name: str):
    from fasttd3_ptf.official_fasttd3_ptf.paths import ensure_humanoidbench_import_path

    ensure_humanoidbench_import_path()
    import gymnasium as gym
    from gymnasium.wrappers import TimeLimit

    import humanoid_bench  # noqa: F401

    env = gym.make(env_name)
    return TimeLimit(env, max_episode_steps=EPISODE_STEPS)


def _basketball_state(env) -> dict | None:
    """basketball 判定成败需要球心与 hoop 的距离（info 里没有）。"""
    try:
        named = env.unwrapped._env.named.data
        dist = float(np.linalg.norm(named.xpos["basketball"] - named.site_xpos["hoop_center"]))
        return {"ball_to_hoop_dist": dist}
    except (AttributeError, KeyError):
        return None


@torch.no_grad()
def _run_episode_v2(env, actor, obs_norm, device, seed: int, env_name: str) -> dict:
    """单 episode。**数值部分与 v1 的 _run_episode 逐行对齐。**"""
    # 双播种(E11 语义):全局 NumPy + Gymnasium np_random。与 v1 相同。
    np.random.seed(seed)
    obs, _ = env.reset(seed=seed)
    x0 = float(env.unwrapped.data.qpos[0])
    total_return = 0.0
    max_progress = 0.0
    steps = 0
    terminated = truncated = False
    info: dict = {}
    info_history: list[dict] = []

    for _ in range(EPISODE_STEPS):
        obs_tensor = torch.as_tensor(obs, device=device, dtype=torch.float32).unsqueeze(0)
        action = actor(obs_norm(obs_tensor)).squeeze(0).cpu().numpy()
        obs, reward, terminated, truncated, info = env.step(action)
        total_return += float(reward)
        max_progress = max(max_progress, float(env.unwrapped.data.qpos[0]) - x0)
        steps += 1
        info_history.append(dict(info))
        if terminated:
            break
        if truncated:
            break

    spec = task_metrics.TASK_METRIC_REGISTRY.get(env_name)
    mj_state = _basketball_state(env) if (spec and spec.needs_mujoco_state) else None
    required = spec.required_info_keys if spec else ()

    task_success, semantics, status, milestones = task_metrics.resolve_task_outcome(
        env_name=env_name, terminated=terminated, truncated=truncated,
        info=info, mj_state=mj_state,
    )
    diagnostics, unsupported = schema_v2.summarize_info(info_history, required_keys=required)

    return schema_v2.build_episode_record(
        seed=seed,
        total_return=total_return,
        progress_max_dx=max_progress,
        episode_length=steps,
        terminated=terminated,
        truncated=truncated,
        task_success=task_success,
        termination_semantics=semantics,
        metric_status=status,
        milestones=milestones,
        info_diagnostics=diagnostics,
        info_diagnostics_unsupported=unsupported,
    )


def run_panel_v2(checkpoint: str, env_name: str, eval_seeds=EVAL_SEEDS,
                 device: str | None = None, n_episodes: int | None = None) -> list[dict]:
    """跑冻结面板，返回 v2 episode 记录列表。供 T4 与主流程共用。"""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    # load_student → (actor, critic, obs_norm, critic_norm, global_step)；
    # 结构性 source-free：critic 与 critic_norm 在 eval 路径上一律不使用。
    actor, _critic, obs_norm, _critic_norm, _global_step = load_student(str(checkpoint), device)
    actor.eval()
    env = _make_env(env_name)
    episodes: list[dict] = []
    try:
        for eval_seed in eval_seeds:
            for rank in RANKS:
                episodes.append(
                    _run_episode_v2(env, actor, obs_norm, device, eval_seed * 1000 + rank, env_name)
                )
                if n_episodes is not None and len(episodes) >= n_episodes:
                    return episodes
    finally:
        env.close()
    return episodes


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _aggregate(episodes: list[dict]) -> dict:
    """面板聚合。

    **不再输出 success_count**——v1 的该字段读自 terminated，在 Walk 系上是
    摔倒计数（CLAUDE.md §6）。v2 改为分列三态计数，缺 adapter 的任务
    task_success_unknown 会等于 episode 数，一眼可见"没测"而不是"没成功"。
    """
    returns = [e["return"] for e in episodes]
    progress = [e["progress_max_dx"] for e in episodes]
    successes = [e["task_success"] for e in episodes]
    return {
        "return_mean": float(np.mean(returns)),
        "return_std": float(np.std(returns, ddof=1)) if len(returns) > 1 else 0.0,
        "progress_max_dx_mean": float(np.mean(progress)),
        "episode_length_mean": float(np.mean([e["episode_length"] for e in episodes])),
        "terminated_count": int(sum(e["terminated"] for e in episodes)),
        "truncated_count": int(sum(e["truncated"] for e in episodes)),
        "task_success_true": int(sum(1 for s in successes if s is True)),
        "task_success_false": int(sum(1 for s in successes if s is False)),
        "task_success_unknown": int(sum(1 for s in successes if s is None)),
        "metric_status_counts": {
            st: int(sum(1 for e in episodes if e["metric_status"] == st))
            for st in sorted({e["metric_status"] for e in episodes})
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--env-name", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--eval-seeds", default=None,
                        help="逗号分隔，或 'panel128'；前 4 个必须为 (11,23,37,53)")
    args = parser.parse_args()

    if args.eval_seeds is None:
        eval_seeds = EVAL_SEEDS
    elif args.eval_seeds == "panel128":
        eval_seeds = EVAL_SEEDS_128
    else:
        eval_seeds = tuple(int(x) for x in args.eval_seeds.split(","))
    if eval_seeds[:len(EVAL_SEEDS)] != EVAL_SEEDS:
        raise SystemExit(
            f"面板前 4 个 eval seed 必须为 {EVAL_SEEDS} 以保持向后兼容，"
            f"收到 {eval_seeds[:len(EVAL_SEEDS)]}"
        )

    if args.env_name not in task_metrics.TASK_METRIC_REGISTRY:
        print(
            f"[WARN] {args.env_name} 未在 TASK_METRIC_REGISTRY 中注册："
            f"task_success 将全部为 null（UNREGISTERED）。这是 fail-closed 行为，"
            f"不是错误——但该 checkpoint 无法用于任何 milestone 判据。",
            file=sys.stderr,
        )

    episodes = run_panel_v2(args.checkpoint, args.env_name, eval_seeds, args.device)

    ckpt_path = Path(args.checkpoint)
    payload = {
        "schema_version": schema_v2.SCHEMA_VERSION,
        "protocol": {
            "eval_seeds": list(eval_seeds),
            "ranks": list(RANKS),
            "episode_steps": EPISODE_STEPS,
            "deterministic": True,
            "source_free": "structural: actor + frozen obs normalizer only",
        },
        "checkpoint": {
            "path": str(ckpt_path),
            "sha256": _sha256(ckpt_path),
        },
        "env_name": args.env_name,
        "task_metrics_registered": args.env_name in task_metrics.TASK_METRIC_REGISTRY,
        "task_metrics_source": (
            task_metrics.TASK_METRIC_REGISTRY[args.env_name].source
            if args.env_name in task_metrics.TASK_METRIC_REGISTRY else None
        ),
        "git_head": subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
            cwd=str(REPO_ROOT),
        ).stdout.strip(),
        "utc": datetime.now(timezone.utc).isoformat(),
        "episodes": episodes,
        "aggregate": _aggregate(episodes),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out_path}")
    print(json.dumps(payload["aggregate"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
