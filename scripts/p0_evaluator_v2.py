"""P0 独立冻结 evaluator —— schema v2.1。

契约冻结于 docs/experiments/evaluator_schema_v2_prereg_20260806.md
与 docs/experiments/evaluator_v21_hardening_prereg_20260806.md。

与 v1（scripts/p0_evaluator.py）的差别有两层：

**(a) 任务语义层**（v2 引入）：

    v1   if terminated: success = True     → terminated_success
    v2   环境事实（terminated/truncated）与任务语义（task_success）分离，
         语义一律由 fasttd3_ptf.evaluation.task_metrics 的 registry 判定。

**(b) 安全能力层**（v2.1 恢复）：v2 初版丢掉了 v1 的 checkpoint 身份校验
与拒绝覆盖，导致喂错 checkpoint/env 仍会产出外观合法的 JSON。
v2.1 恢复二者并加了 `panel_digest` —— 不同面板产出的数字不可比，
而此前无法从输出中判断两份结果是否用了同一面板。

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


def _digest_obj(obj) -> str:
    """对任意可 JSON 化对象取稳定摘要（键排序）；不可 JSON 化时降级为 repr。"""
    try:
        blob = json.dumps(obj, sort_keys=True, default=str, ensure_ascii=False)
    except (TypeError, ValueError):
        blob = repr(obj)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def panel_digest(eval_seeds, ranks, episode_steps: int, deterministic: bool) -> str:
    """面板指纹。不同面板产出的数字不可比，而此前无法从输出中判断这一点。"""
    return _digest_obj({
        "eval_seeds": list(eval_seeds),
        "ranks": list(ranks),
        "episode_steps": episode_steps,
        "deterministic": deterministic,
    })


def verify_checkpoint_identity(
    checkpoint: str,
    env_name: str,
    expect_global_step=None,
    expect_seed=None,
    expect_admission_mode=None,
) -> dict:
    """核对 checkpoint 内部身份，不符即抛错。恢复自 v1（p0_evaluator.py:152-175）。

    强制项：checkpoint 内 ``args["env_name"]`` 必须等于命令行 ``--env-name``。
    喂错 env 会产出完全合法但语义错误的结果，且无法从输出中看出来。

    返回身份摘要 dict，供写入评估产物。
    """
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    ckpt_args = state.get("args") or {}
    ptf_cfg = state.get("ptf_cfg") or {}

    errors = []
    if ckpt_args.get("env_name") != env_name:
        errors.append(
            f"checkpoint env_name={ckpt_args.get('env_name')!r} != --env-name {env_name!r}")
    if expect_global_step is not None and state.get("global_step") != expect_global_step:
        errors.append(
            f"checkpoint global_step={state.get('global_step')} != expected {expect_global_step}")
    if expect_seed is not None and ckpt_args.get("seed") != expect_seed:
        errors.append(f"checkpoint seed={ckpt_args.get('seed')} != expected {expect_seed}")
    if expect_admission_mode is not None:
        actual = ptf_cfg.get("admission_mode")
        if actual != expect_admission_mode:
            errors.append(
                f"checkpoint admission_mode={actual!r} != expected {expect_admission_mode!r}")
    if errors:
        raise ValueError("checkpoint identity mismatch: " + "; ".join(errors))

    explicit = [
        name for name, val in (
            ("expect_global_step", expect_global_step),
            ("expect_seed", expect_seed),
            ("expect_admission_mode", expect_admission_mode),
        ) if val is not None
    ]
    info = {
        "global_step": state.get("global_step"),
        "learner_seed": ckpt_args.get("seed"),
        "env_name_in_checkpoint": ckpt_args.get("env_name"),
        "training_commit": ckpt_args.get("git_commit") or ckpt_args.get("commit") or "UNKNOWN",
        "args_digest": _digest_obj(ckpt_args),
        "ptf_cfg_digest": _digest_obj(ptf_cfg),
        "ptf_cfg_summary": {
            "admission_mode": ptf_cfg.get("admission_mode", "UNKNOWN"),
            "admission_replay_mode": ptf_cfg.get("admission_replay_mode", "UNKNOWN"),
            "mcg_warmup_steps": ptf_cfg.get("mcg_warmup_steps", "UNKNOWN"),
            "source_names": state.get("source_names", "UNKNOWN"),
        },
        # 只做强制 env 核对而未显式声明任何 expect_* 时为 False：
        # 输出仍产生，但明确标记为未经完整身份声明。
        "identity_checked": len(explicit) >= 1,
        "identity_expectations": explicit,
    }
    del state
    return info


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
    # 身份声明（恢复自 v1）：正式评估必须显式声明预期身份，
    # checkpoint 与声明不符即拒绝——防止喂错 checkpoint 静默产出合法外观的 JSON。
    parser.add_argument("--expect-global-step", type=int, default=None)
    parser.add_argument("--expect-seed", type=int, default=None)
    parser.add_argument("--expect-admission-mode", default=None,
                        help="lease 分支=all，abstain 分支=none")
    parser.add_argument("--allow-overwrite", action="store_true",
                        help="显式允许覆盖已存在的评估产物（默认硬拒绝）")
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

    # ── 防覆盖：在跑任何 episode 之前检查，避免白跑几十分钟 ──────────
    out_path = Path(args.out)
    if out_path.exists() and not args.allow_overwrite:
        raise SystemExit(
            f"拒绝覆盖已存在的评估产物：{out_path}\n"
            f"若确需覆盖请显式传 --allow-overwrite")

    # ── 身份校验：不符即拒绝，不产出任何字节 ─────────────────────────
    identity = verify_checkpoint_identity(
        args.checkpoint, args.env_name,
        args.expect_global_step, args.expect_seed, args.expect_admission_mode)
    if not identity["identity_checked"]:
        print(
            "[WARN] 未显式声明任何 --expect-*：identity_checked=false。"
            "强制 env 核对已通过，但正式评估应至少声明一项。",
            file=sys.stderr)

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
        "schema_version": "2.1",
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
            **{k: v for k, v in identity.items()
               if k not in ("identity_checked", "identity_expectations")},
        },
        "identity_checked": identity["identity_checked"],
        "identity_expectations": identity["identity_expectations"],
        "panel_digest": panel_digest(eval_seeds, RANKS, EPISODE_STEPS, True),
        "overwrote_existing": bool(out_path.exists() and args.allow_overwrite),
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

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out_path}")
    print(json.dumps(payload["aggregate"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
