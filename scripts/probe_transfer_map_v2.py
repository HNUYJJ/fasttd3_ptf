"""Cross-Task Transfer Map v2（RBO-PTF Day2-3：snippet-level + safe-horizon）。

v1 是 episode-level zero-shot(整段 rollout 取 return),在 window 这类"短 prefix
站立有用、长 episode 摔倒有害"的脆弱任务上会误判(v1: window 所有源 19-36 步全摔
→ 判全员 OOD,却 seed1 +76%)。v2 改成 **snippet-level**:对每个 (source o,
target T) 执行教师整动作 prefix,逐步记录累计 reward / progress / 摔倒,从一条
max_h 轨迹同时读出所有 horizon h∈{5,10,15,25,50} 的前缀量,据此算:

  S(o,T,h) = [R_h(o) − R_h(zero)]            # prefix reward gain(携带回报)
           + λ·[Φ_h(o) − Φ_h(zero)]         # progress-event gain
           − α·P_fall(o,T,h)                # safety penalty

  h*(o,T) = argmax_h S(o,T,h)  s.t. P_fall(o,T,h*) < δ   # safe horizon
  safe_horizon(o,T) = max{ h : P_fall(o,T,h) < δ }       # 可安全执行的最长前缀

输出每个 (target, source) 一行 JSON,含 per-horizon 量 + safe_horizon + best
score/h + time-to-fall 分位。直接喂给 Day4 的 safe-horizon weighted bootstrap
(p(o|T)=softmax(maxₕS/τ), 锁存步数=safe_horizon)。也支持 Spearman 预测力验证
(score vs 已测 transfer ROI)。

复用 v1 的源加载(stand/walk/run, proprio_adapter)与 obs 布局摸底。

用法:
  python scripts/probe_transfer_map_v2.py [--targets t1 ...] [--repeats 4]
      [--max-h 50] [--num-envs 16] [--out logs/probe/transfer_map_v2.jsonl]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from fasttd3_ptf.official_fasttd3_ptf.paths import (
    ensure_fasttd3_import_path,
    ensure_humanoidbench_import_path,
)
from fasttd3_ptf.ptf.source_policy import SourcePolicy

ACTION_DIM = 61
HORIZONS = [5, 10, 15, 25, 50]
REWRITTEN_PROPRIO_PREFIX = {"h1hand-push-v0", "h1hand-package-v0", "h1hand-reach-v0"}

DEFAULT_TARGETS = [
    "h1hand-hurdle-v0", "h1hand-cabinet-v0", "h1hand-powerlift-v0", "h1hand-maze-v0",
    "h1hand-window-v0", "h1hand-balance_hard-v0", "h1hand-truck-v0",
    "h1hand-spoon-v0", "h1hand-door-v0",
]
PROPRIO_SOURCES = {
    "stand": "checkpoints/official_sources/h1hand_stand/manifest.json",
    "walk": "checkpoints/official_sources/h1hand_walk/manifest.json",
    "run": "checkpoints/official_sources/h1hand_run/manifest.json",
    # 扩源第一批(2026-07-04): terrain scr s1 final 作策略源——obs 151/nq 76
    # 与 loco 源完全同布局(hb_task_layouts 核实), identity 直连零 adapter 风险
    "crawl": "checkpoints/terrain_sources/h1hand_crawl/manifest.json",
    "pole": "checkpoints/terrain_sources/h1hand_pole/manifest.json",
    "slide": "checkpoints/terrain_sources/h1hand_slide/manifest.json",
    "stair": "checkpoints/terrain_sources/h1hand_stair/manifest.json",
    # 扩源 v2(正交技能, 2026-07-04): hurdle=跨障(151 identity);
    # reach=伸手 manipulation(源 obs 157=151 proprio+6 目标位, 跨任务用
    # identity+allow_pad 把 proprio 视图补 6 个零——目标位=原点, 有偏但可测)
    "hurdle": "checkpoints/terrain_sources/h1hand_hurdle/manifest.json",
    "reach": {
        "manifest": "checkpoints/official_sources/h1hand_reach/manifest.json",
        "obs_adapter": {"type": "identity", "output_dim": 157, "allow_pad": True},
    },
}
# 每任务 task-progress 白名单(Day1 audit 学到的硬完成度字段;balance 无独立进度)
PROGRESS_KEYS = {
    "hurdle": ["move"],
    "cabinet": ["success_subtasks", "door_openness_reward"],
    "powerlift": ["reward_dumbbell_lifted"],
    "maze": ["success_subtasks", "checkpoint_proximity_reward", "move"],
    "window": ["window_contact_total_reward"],
    "balance_hard": [],
    "truck": ["reward_robot_package_truck"],
    "spoon": ["reward_spoon_in_cup", "spoon_spinning_reward"],
    "door": ["passage_reward", "door_openness_reward"],
}

LAMBDA, ALPHA, DELTA = 1.0, 1.0, 0.5  # score 权重 + safe-horizon fall 阈值


def make_env_fn(env_id: str, rank: int, max_steps: int, seed: int):
    def _init():
        ensure_humanoidbench_import_path()
        import gymnasium as gym
        import humanoid_bench  # noqa: F401
        from gymnasium.wrappers import TimeLimit

        env = gym.make(env_id)
        env = TimeLimit(env, max_episode_steps=max_steps)
        env.unwrapped.seed(seed + rank)
        return env

    return _init


def proprio_adapter(env_id: str, nq: int):
    if env_id in REWRITTEN_PROPRIO_PREFIX:
        return lambda o: o[:, :151]
    return lambda o: np.concatenate([o[:, :76], o[:, nq:nq + 75]], axis=1)


def find_scratch_early(task: str, step: int = 10000, seed: int = 1) -> str | None:
    """每任务 scratch@step 的 checkpoint(作 opportunity baseline:教师 prefix 相对
    scratch 早期 policy 的增量,才是 scratch 学不到的真对价 —— ChatGPT Opportunity)。"""
    cands = glob.glob(
        f"models/h1hand-{task}-v0__h1hand_{task}_pilot_scr_s{seed}_*__{seed}_{step}.pt")
    pat = re.compile(rf"_pilot_scr_s{seed}_[0-9TZ]+__{seed}_{step}\.pt$")
    cands = [c for c in cands if pat.search(os.path.basename(c))]
    return cands[0] if cands else None


@torch.no_grad()
def rollout_snippet(envs, act_fn, num_envs, max_h, repeats, progress_keys):
    """从 reset 执行 act_fn 的 prefix,逐步累计 reward/progress,记录摔倒步。

    返回 per-step(1..max_h)的: prefix_reward[h]/prefix_progress[h] 均值、
    fall_prob[h](到第 h 步为止已摔比例)、time-to-fall 分布(未摔记 max_h+1)。
    """
    cum_r_all, cum_p_all, ttf_all = [], [], []
    for _ in range(repeats):
        obs = envs.reset()
        active = np.ones(num_envs, dtype=bool)
        acc_r = np.zeros(num_envs)
        acc_p = np.zeros(num_envs)
        fall_step = np.full(num_envs, max_h + 1, dtype=np.int64)
        cum_r = np.zeros((num_envs, max_h))
        cum_p = np.zeros((num_envs, max_h))
        for h in range(max_h):
            act = act_fn(obs)
            obs, rew, dones, infos = envs.step(act)
            psum = np.array([sum(float(infos[i].get(k, 0.0)) for k in progress_keys)
                             for i in range(num_envs)]) if progress_keys else np.zeros(num_envs)
            for i in range(num_envs):
                if active[i]:
                    acc_r[i] += float(rew[i])
                    acc_p[i] += psum[i]
                    if dones[i]:
                        if not bool(infos[i].get("TimeLimit.truncated", False)):
                            fall_step[i] = h + 1  # 非超时终止 = 摔
                        active[i] = False
                cum_r[i, h] = acc_r[i]
                cum_p[i, h] = acc_p[i]
        cum_r_all.append(cum_r)
        cum_p_all.append(cum_p)
        ttf_all.append(fall_step)
    cum_r = np.concatenate(cum_r_all, axis=0)      # [R*E, max_h]
    cum_p = np.concatenate(cum_p_all, axis=0)
    ttf = np.concatenate(ttf_all, axis=0)          # [R*E]
    per_h = {}
    for h in HORIZONS:
        if h > max_h:
            continue
        per_h[h] = dict(
            prefix_reward=float(cum_r[:, h - 1].mean()),
            prefix_progress=float(cum_p[:, h - 1].mean()),
            fall_prob=float(np.mean(ttf <= h)),
        )
    return per_h, ttf


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", nargs="+", default=DEFAULT_TARGETS)
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--max-h", type=int, default=50)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", default="logs/probe/transfer_map_v2.jsonl")
    args = parser.parse_args()

    ensure_fasttd3_import_path()
    ensure_humanoidbench_import_path()
    from stable_baselines3.common.vec_env import SubprocVecEnv

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    layouts = json.loads(Path("logs/probe/hb_task_layouts.json").read_text())
    recs = {r["task"]: r for r in layouts} if isinstance(layouts, list) else layouts

    def load_sp(name, manifest):
        # manifest 可为 str(默认 identity adapter)或 dict(带 obs_adapter override,
        # 供 reach 这类源 obs 含 task 维的源用 pad 方案)
        if isinstance(manifest, dict):
            spec = {"name": name, "manifest": manifest["manifest"],
                    "obs_adapter": manifest.get("obs_adapter", {"type": "identity"}),
                    "action_adapter": {"type": "passthrough"}, "action_mask": {"type": "full"}}
        else:
            spec = {"name": name, "manifest": manifest, "obs_adapter": {"type": "identity"},
                    "action_adapter": {"type": "passthrough"}, "action_mask": {"type": "full"}}
        return SourcePolicy.from_spec(spec, device=device, target_action_dim=ACTION_DIM)

    proprio_sps = {n: load_sp(n, m) for n, m in PROPRIO_SOURCES.items()}

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fout = out_path.open("a")

    for target in args.targets:
        task = target.replace("h1hand-", "").replace("-v0", "")
        nq = recs[target]["nq"]
        obs_dim = int(recs[target]["obs_dim"])
        adapter = proprio_adapter(target, nq)
        pkeys = PROGRESS_KEYS.get(task, [])
        envs = SubprocVecEnv([make_env_fn(target, i, args.max_h + 2, args.seed)
                              for i in range(args.num_envs)])

        # 先跑 baseline cells: zero(dense-shaping 偏置) + scratch_early(opportunity
        # 基线)。教师 prefix 相对 scratch_early 的增量才是 scratch 早期学不到的真
        # 对价 —— v1 的 Spearman 失败正因漏了这一项(door 高估/cabinet 低估)。
        cells = [("zero", lambda o: np.zeros((o.shape[0], ACTION_DIM), dtype=np.float32))]
        scr_path = find_scratch_early(task)
        if scr_path:
            scr_sp = SourcePolicy(Path(scr_path).stem, scr_path, device=device,
                                  target_action_dim=ACTION_DIM, source_obs_dim=obs_dim,
                                  source_action_dim=ACTION_DIM)
            cells.append(("scratch_early", lambda o, _sp=scr_sp: _sp.act(
                torch.as_tensor(o, device=device, dtype=torch.float32)).cpu().numpy()))
        for n, sp in proprio_sps.items():
            cells.append((n, lambda o, _sp=sp: _sp.act(
                torch.as_tensor(adapter(o), device=device, dtype=torch.float32)).cpu().numpy()))

        zero_per_h, scr_per_h = None, None
        for sname, fn in cells:
            t0 = time.time()
            per_h, ttf = rollout_snippet(envs, fn, args.num_envs, args.max_h, args.repeats, pkeys)
            if sname == "zero":
                zero_per_h = per_h
                continue
            if sname == "scratch_early":
                scr_per_h = per_h
                continue
            base = scr_per_h if scr_per_h is not None else zero_per_h
            base_name = "scratch_early" if scr_per_h is not None else "zero"
            scored = {}
            best_score, best_h = -1e9, None
            safe_h = 0
            for h in HORIZONS:
                if h not in per_h:
                    continue
                rg = per_h[h]["prefix_reward"] - base[h]["prefix_reward"]   # opportunity
                pg = per_h[h]["prefix_progress"] - base[h]["prefix_progress"]
                rg0 = per_h[h]["prefix_reward"] - zero_per_h[h]["prefix_reward"]  # vs zero(参考)
                fp = per_h[h]["fall_prob"]
                S = rg + LAMBDA * pg - ALPHA * fp
                scored[h] = dict(reward_gain=round(rg, 4), reward_gain_vs_zero=round(rg0, 4),
                                 progress_gain=round(pg, 4), fall_prob=round(fp, 4),
                                 score=round(S, 4))
                if fp < DELTA:
                    safe_h = max(safe_h, h)
                if S > best_score:
                    best_score, best_h = S, h
            ttf_clip = ttf[ttf <= args.max_h]
            row = dict(
                target=target, source=sname, baseline=base_name,
                per_horizon=scored, safe_horizon=safe_h,
                best_score=round(best_score, 4), best_h=best_h,
                fall_rate_full=round(float(np.mean(ttf <= args.max_h)), 4),
                ttf_q10=int(np.quantile(ttf_clip, 0.1)) if len(ttf_clip) else args.max_h + 1,
                ttf_q50=int(np.quantile(ttf_clip, 0.5)) if len(ttf_clip) else args.max_h + 1,
                progress_keys=pkeys,
                evaluated_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
            fout.write(json.dumps(row) + "\n")
            fout.flush()
            print(f"{target:24s} {sname:6s} safe_h={safe_h:2d} best_h={str(best_h):>3s} "
                  f"score={best_score:7.3f} (base={base_name}) fall_full={row['fall_rate_full']:4.0%} "
                  f"ttf_q10={row['ttf_q10']:2d} ({time.time()-t0:.0f}s)", flush=True)
        envs.close()
    fout.close()
    print(f"\nsaved -> {out_path}")


if __name__ == "__main__":
    main()
