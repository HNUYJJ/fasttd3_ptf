"""Task-Progress Audit（RBO-PTF Day-1，回答审稿核心质疑）。

ChatGPT 的关键提醒:若 cabinet/powerlift 等任务的增益只是 alive/stand
multiplier 上升,论文声称"提升复杂任务完成度"会被审稿人一句"你只是站得更久"
打掉。本脚本把 total return 拆成 **task-progress** 与 **alive/stand** 两部分,
证明 full(RBO) 相对 scratch 的增益落在任务进度上,而不只是站立时长。

做法(全部基于已落盘的 pilot checkpoint,不重训):
  - 对 (task, method∈{mcg=full, scr=scratch}, seed) 的 checkpoint 序列,用
    SourcePolicy 加载训练出的 actor(自动复现训练时的 obs normalizer)做确定性
    rollout;
  - 采集 HB info 的细粒度 reward 分量 per-step 均值 + total return + fall +
    ep_len(SubprocVecEnv 的 TimeLimit.truncated 区分摔倒 vs 超时);
  - 按 ALIVE_CONTROL 黑名单把数值分量分成 alive/control 与 task-progress 两组,
    另对已知任务给出 PROGRESS_KEYS 白名单(更直观,如 door_openness_reward);
  - 沿 step 对 total / task-progress / alive 各算 AUC,输出 full vs scratch 对比。

用法:
  python scripts/task_progress_audit.py [--tasks hurdle cabinet ...]
      [--methods mcg scr] [--seeds 1] [--steps 10000 30000 50000 70000]
      [--num-envs 16] [--episode-steps 1000]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import time
from collections import defaultdict
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

# alive / 控制平滑 / 每步总量类分量 —— 不计入 task-progress(否则会把"站着"
# 当成"完成任务",正是要避免的混淆)。其余数值分量视为 task-progress。
ALIVE_CONTROL = {
    "stand_reward", "standing", "upright", "small_control", "dont_move",
    "per_timestep_reward", "TimeLimit.truncated",
}

# 各任务最直观的 task-progress 白名单(便于直接读关键字段;sum 仍用黑名单法
# 计算,二者互为佐证)。未列任务自动退回黑名单法。
PROGRESS_KEYS = {
    "hurdle": ["move"],
    "cabinet": ["door_openness_reward", "success", "subtask_complete", "success_subtasks"],
    "powerlift": ["reward_dumbbell_lifted"],
    "maze": ["checkpoint_proximity_reward", "stage_convert_reward", "move", "success_subtasks"],
    "window": ["window_contact_reward", "window_contact_total_reward", "moving_wipe_reward",
               "hand_tool_proximity_reward"],
    "balance_hard": [],  # 任务本身=站立平衡,无独立 manipulation 进度,诚实标注
}


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


@torch.no_grad()
def rollout(envs, act_fn, num_envs: int, episode_steps: int):
    """每个并行 env 恰好 1 个 episode;采集 return / fall / 所有数值 info 分量。"""
    obs = envs.reset()
    ep_ret = np.zeros(num_envs)
    ep_len = np.zeros(num_envs, dtype=np.int64)
    active = np.ones(num_envs, dtype=bool)
    rets, lens, falls = [], [], []
    info_sums: dict[str, float] = defaultdict(float)
    info_steps = 0
    for _ in range(episode_steps + 5):
        act = act_fn(obs)
        obs, rew, dones, infos = envs.step(act)
        ep_ret[active] += rew[active]
        ep_len[active] += 1
        # 仅累加仍 active 的 env 的 info 分量(避免把 done 后 reset 的新 episode 混入)
        for i in np.nonzero(active)[0]:
            for k, v in infos[i].items():
                if isinstance(v, bool) or isinstance(v, np.bool_):
                    info_sums[k] += float(v)
                elif isinstance(v, (int, float, np.floating, np.integer)):
                    info_sums[k] += float(v)
            info_steps += 1
        for i in np.nonzero(dones & active)[0]:
            truncated = bool(infos[i].get("TimeLimit.truncated", False))
            rets.append(float(ep_ret[i]))
            lens.append(int(ep_len[i]))
            falls.append(0.0 if truncated else 1.0)  # 非超时终止 = 摔倒
            active[i] = False
        if not active.any():
            break
    for i in np.nonzero(active)[0]:  # 跑满步数仍未 done(罕见)
        rets.append(float(ep_ret[i]))
        lens.append(int(ep_len[i]))
        falls.append(0.0)
    info_means = {k: v / max(info_steps, 1) for k, v in info_sums.items()
                  if k != "TimeLimit.truncated"}
    return dict(
        return_mean=float(np.mean(rets)), return_std=float(np.std(rets)),
        ep_len_mean=float(np.mean(lens)), fall=float(np.mean(falls)),
        episodes=len(rets), info_means=info_means,
    )


def split_progress(task: str, info_means: dict[str, float]) -> tuple[float, float, dict, dict]:
    """返回 (progress_sum, alive_sum, progress_components, alive_components)。"""
    prog, alive = {}, {}
    for k, v in info_means.items():
        if k in ALIVE_CONTROL:
            alive[k] = v
        else:
            prog[k] = v
    return sum(prog.values()), sum(alive.values()), prog, alive


def find_checkpoints(task: str, method: str, seed: int, steps: list[int]):
    """返回 [(step, path)]，含 final(标为 100000)。选 ckpt 最全的 stamp。"""
    pat = re.compile(
        rf"h1hand_{task}_pilot_{method}_s{seed}_(?P<stamp>[0-9TZ]+)__\d+_(?P<step>\d+|final)\.pt$")
    by_stamp: dict[str, dict[str, str]] = defaultdict(dict)
    for f in glob.glob(f"models/h1hand-{task}-v0__h1hand_{task}_pilot_{method}_s{seed}_*.pt"):
        m = pat.search(os.path.basename(f))
        if m:
            by_stamp[m["stamp"]][m["step"]] = f
    if not by_stamp:
        return []
    stamp = max(by_stamp, key=lambda s: len(by_stamp[s]))  # ckpt 最全
    files = by_stamp[stamp]
    out = []
    for s in steps:
        if str(s) in files:
            out.append((s, files[str(s)]))
    if "final" in files:
        out.append((100000, files["final"]))  # final 当 100k 末点
    return out


def auc(steps, vals):
    if len(steps) < 2:
        return float("nan")
    order = np.argsort(steps)
    s, v = np.asarray(steps)[order], np.asarray(vals)[order]
    return float(np.trapz(v, s) / (s[-1] - s[0]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+",
                        default=["hurdle", "cabinet", "powerlift", "maze", "window", "balance_hard"])
    parser.add_argument("--methods", nargs="+", default=["mcg", "scr"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1])
    parser.add_argument("--steps", nargs="+", type=int, default=[10000, 30000, 50000, 70000])
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--episode-steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=7, help="env seed")
    parser.add_argument("--out", default="logs/probe/task_progress_audit.jsonl")
    args = parser.parse_args()

    ensure_fasttd3_import_path()
    ensure_humanoidbench_import_path()
    from stable_baselines3.common.vec_env import SubprocVecEnv

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    layouts = json.loads(Path("logs/probe/hb_task_layouts.json").read_text())
    recs = {r["task"]: r for r in layouts} if isinstance(layouts, list) else layouts

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fout = out_path.open("a")
    # rows[(task,seed)][method] = [(step, total, prog, alive, fall), ...]
    rows: dict = defaultdict(lambda: defaultdict(list))

    for task in args.tasks:
        env_id = f"h1hand-{task}-v0"
        obs_dim = int(recs[env_id]["obs_dim"])
        envs = SubprocVecEnv([make_env_fn(env_id, i, args.episode_steps, args.seed)
                              for i in range(args.num_envs)])
        for seed in args.seeds:
            for method in args.methods:
                ckpts = find_checkpoints(task, method, seed, args.steps)
                if not ckpts:
                    print(f"[skip] {task} {method} s{seed}: no checkpoints", flush=True)
                    continue
                for step, path in ckpts:
                    sp = SourcePolicy(Path(path).stem, path, device=device,
                                      target_action_dim=ACTION_DIM,
                                      source_obs_dim=obs_dim, source_action_dim=ACTION_DIM)
                    act_fn = lambda o, _sp=sp: _sp.act(
                        torch.as_tensor(o, device=device, dtype=torch.float32)).cpu().numpy()
                    t0 = time.time()
                    res = rollout(envs, act_fn, args.num_envs, args.episode_steps)
                    psum, asum, pc, ac = split_progress(task, res["info_means"])
                    wl = PROGRESS_KEYS.get(task)
                    wl_sum = (sum(res["info_means"].get(k, 0.0) for k in wl)
                              if wl is not None else None)
                    row = dict(task=task, method=method, seed=seed, step=step,
                               total_return=res["return_mean"], return_std=res["return_std"],
                               progress_sum=psum, alive_sum=asum,
                               whitelist_progress=wl_sum, fall=res["fall"],
                               ep_len=res["ep_len_mean"], episodes=res["episodes"],
                               progress_components=pc, alive_components=ac,
                               evaluated_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
                    fout.write(json.dumps(row) + "\n")
                    fout.flush()
                    rows[(task, seed)][method].append((step, res["return_mean"], psum, asum, res["fall"]))
                    wlstr = f" wlprog={wl_sum:7.3f}" if wl_sum is not None else ""
                    print(f"{task:12s} {method} s{seed} {step:6d}  "
                          f"ret={res['return_mean']:8.1f} prog={psum:8.3f} alive={asum:7.3f}"
                          f"{wlstr} fall={res['fall']:4.0%} ({time.time()-t0:.0f}s)", flush=True)
        envs.close()

    # AUC 汇总 full vs scratch
    print("\n=== AUC 汇总(full=mcg vs scratch=scr)===")
    print(f"{'task/seed':16s} {'metric':10s} {'scr':>10s} {'full':>10s} {'ROI':>8s}")
    summary = {}
    for (task, seed), md in sorted(rows.items()):
        if "mcg" not in md or "scr" not in md:
            continue
        summary[f"{task}_s{seed}"] = {}
        for mi, mname in [(1, "total"), (2, "progress"), (3, "alive")]:
            scr = md["scr"]; full = md["mcg"]
            a_scr = auc([r[0] for r in scr], [r[mi] for r in scr])
            a_full = auc([r[0] for r in full], [r[mi] for r in full])
            roi = (a_full - a_scr) / abs(a_scr) if a_scr else float("nan")
            print(f"{task+'/s'+str(seed):16s} {mname:10s} {a_scr:10.2f} {a_full:10.2f} {roi:+8.0%}")
            summary[f"{task}_s{seed}"][mname] = dict(scr=a_scr, full=a_full, roi=roi)
    fout.close()
    Path("logs/probe/task_progress_audit_summary.json").write_text(json.dumps(summary, indent=1))
    print("\nsaved rows -> logs/probe/task_progress_audit.jsonl")
    print("saved summary -> logs/probe/task_progress_audit_summary.json")


if __name__ == "__main__":
    main()
