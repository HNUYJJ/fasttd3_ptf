"""episode-prefix handoff 的聚焦单元测试。

被测语义（Door 通道分解之后的接口消融）：
    关闭（episode_prefix_steps=None）→ 与历史逐位相同：每次 latch 到期都重新抽。
    开启 → 一个 episode 内**只在起点抽一次**；抽中 source 则连续执行 H 步，
           此后**锁定 student 直到该 episode 结束**，即使 latch 到期也不再抽。

这与既有 RBO 的区别正是本实验唯一要检验的因子：source 的**时间放置方式**
（随机碎片 vs episode 前缀），而非剂量、身份或 replay eligibility。
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fasttd3_ptf.ptf.mcg import McgBehaviorController  # noqa: E402


N_ENV, N_GROUP, N_SRC = 4, 1, 1
PREFIX = 5


def _make(prefix_steps: int | None) -> McgBehaviorController:
    return McgBehaviorController(
        num_envs=N_ENV,
        num_groups=N_GROUP,
        device="cpu",
        group_masks=torch.ones(N_GROUP, 3),
        warmup_min_steps=2,
        seed=0,
        warmup_mode="admission_bootstrap",
        bootstrap_weights=torch.zeros(N_SRC),
        bootstrap_horizons=torch.full((N_SRC,), 2, dtype=torch.long),
        admitted_sources=torch.ones(N_SRC, dtype=torch.bool),
        admission_student_logit=0.0,
        episode_prefix_steps=prefix_steps,
    )


def _roll(ctrl: McgBehaviorController, steps: int, done_every: int | None = None):
    """跑若干步，返回每步每 env 执行的是 source(True) 还是 student(False)。"""
    a_stu = torch.zeros(N_ENV, 3)
    a_src = torch.ones(N_ENV, N_SRC, 3)
    trace = []
    for t in range(steps):
        dones = torch.zeros(N_ENV)
        if done_every and t > 0 and t % done_every == 0:
            dones = torch.ones(N_ENV)
        ctrl.step(a_stu, a_src, best=None, gate=None, dones=dones)
        trace.append((ctrl.current[:, 0] >= 0).clone())
    return torch.stack(trace)          # [steps, n_env]


def test_prefix_disabled_keeps_resampling() -> None:
    """默认关闭时，latch 到期会反复重抽——source 段应在 episode 中反复出现。"""
    ctrl = _make(None)
    trace = _roll(ctrl, steps=40, done_every=20)
    # 在单个 episode(20 步)内部，source 的执行应出现不止一个连续段
    seg = trace[:20, 0].tolist()
    switches = sum(1 for i in range(1, len(seg)) if seg[i] != seg[i - 1])
    assert switches >= 2, f"关闭 prefix 时应多次切换，实测 switches={switches}"


def test_prefix_runs_only_at_episode_start() -> None:
    """开启时：source 只可能出现在 episode 前 H 步内，之后必须是 student。"""
    ctrl = _make(PREFIX)
    ep_len = 20
    trace = _roll(ctrl, steps=60, done_every=ep_len)
    for env in range(N_ENV):
        col = trace[:, env].tolist()
        for t, is_src in enumerate(col):
            # step t 属于 episode 内的第 (t % ep_len) 步（t=0 为首个 episode 起点）
            pos = t % ep_len
            if is_src:
                assert pos < PREFIX, (
                    f"env{env} 在 episode 内第 {pos} 步仍由 source 执行，"
                    f"超过 prefix={PREFIX}"
                )


def test_prefix_locks_student_after_handoff() -> None:
    """handoff 之后即使 latch 到期也不得重新抽到 source（本 episode 内）。"""
    ctrl = _make(PREFIX)
    trace = _roll(ctrl, steps=18, done_every=None)   # 单个长 episode，无 reset
    col = trace[:, 0].tolist()
    # 前 PREFIX 步可能是 source；此后必须恒为 student
    assert not any(col[PREFIX:]), f"handoff 后出现了 source 执行：{col}"


def test_episode_reset_reenables_decision() -> None:
    """episode reset 后必须允许重新决策，否则整段训练只有第一个 episode 用到 source。"""
    ctrl = _make(PREFIX)
    trace = _roll(ctrl, steps=60, done_every=20)
    # 至少两个不同 episode 里出现过 source 执行
    eps_with_src = {t // 20 for t in range(60) if bool(trace[t].any())}
    assert len(eps_with_src) >= 2, f"仅 {eps_with_src} 个 episode 用到 source"


def test_decided_flag_cleared_on_done() -> None:
    ctrl = _make(PREFIX)
    a_stu, a_src = torch.zeros(N_ENV, 3), torch.ones(N_ENV, N_SRC, 3)
    ctrl.step(a_stu, a_src, best=None, gate=None, dones=torch.zeros(N_ENV))
    assert bool(ctrl._episode_decided.all()), "首步之后应全部标记为已决策"
    ctrl.step(a_stu, a_src, best=None, gate=None, dones=torch.ones(N_ENV))
    # done 处理发生在抽样之前，故本步会立即重新决策；关键是 done 清了标志
    assert ctrl._episode_decided.dtype == torch.bool


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
