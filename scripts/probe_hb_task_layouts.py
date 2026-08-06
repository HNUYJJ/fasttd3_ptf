"""HB 全任务 obs 布局摸底(Transfer Map 基础设施第一步)。

对每个 h1hand 任务实例化一次,输出: obs_dim / nq / nv / 是否默认布局
(obs == concat(qpos, qvel)) / object qpos 维数(nq-76) / success_bar /
step info 键。结果写 logs/probe/hb_task_layouts.json + 控制台表格。

默认布局任务的 proprio 适配器是机械的:
  proprio151 = concat(obs[0:76], obs[nq : nq+75])
重写 get_obs 的 7 个任务(cube/spoon/kitchen/bookshelf/reach/package/push)
需要逐个读源码登记字段。
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from fasttd3_ptf.official_fasttd3_ptf.paths import ensure_humanoidbench_import_path

ensure_humanoidbench_import_path()
import gymnasium as gym
import humanoid_bench  # noqa: F401


def main() -> None:
    ids = sorted(k for k in gym.registry if k.startswith("h1hand-"))
    rows = []
    print(f"{'task':28s} {'obs':>5s} {'nq':>4s} {'nv':>4s} {'default':>8s} "
          f"{'obj_nq':>7s} {'bar':>6s} info_keys")
    for env_id in ids:
        try:
            env = gym.make(env_id)
            obs, _ = env.reset(seed=0)
            u = env.unwrapped
            nq, nv = u.model.nq, u.model.nv
            default = (
                len(obs) == nq + nv
                and np.allclose(obs[:nq], u.data.qpos, atol=1e-8)
                and np.allclose(obs[nq:nq + nv], u.data.qvel, atol=1e-8)
            )
            bar = getattr(u.task, "success_bar", None)
            a = env.action_space.sample() * 0
            _, _, _, _, info = env.step(a)
            row = dict(
                task=env_id, obs_dim=int(len(obs)), nq=int(nq), nv=int(nv),
                default_layout=bool(default), object_nq=int(nq - 76),
                success_bar=bar, info_keys=sorted(info.keys()),
            )
            rows.append(row)
            print(f"{env_id:28s} {len(obs):5d} {nq:4d} {nv:4d} {str(default):>8s} "
                  f"{nq-76:7d} {str(bar):>6s} {','.join(row['info_keys'][:6])}", flush=True)
            env.close()
        except Exception as e:  # noqa: BLE001
            print(f"{env_id:28s} FAILED: {type(e).__name__}: {e}", flush=True)
    out = Path("logs/probe/hb_task_layouts.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=1))
    print(f"\nsaved {len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
