"""迁移通道解耦的聚焦单元测试（Door@10k 结果驱动）。

被测机制：在 ``warmup_mode=admission_bootstrap`` 下，"谁执行"(behavior authority)
与 "critic 按什么来源配额采样"(replay eligibility) 本来共用同一个
student-inclusive categorical。``admission_replay_mode`` 只覆盖 **replay 侧**，
使 B-only 臂 :math:`(B{=}1, R{=}0)` 成为可能。

关键区别（也是本文件存在的理由）：既有的
``test_exact_replay_revocation_samples_only_student`` 测的是**撤销准入**
(``admitted=False``)，那种情况下 source 连车都开不了。这里测的是 source
**仍被准入**（因而仍有 behavior authority、仍写入 physical buffer），
但 replay 配额为 0。
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from tensordict import TensorDict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fasttd3_ptf.official_fasttd3_ptf.paths import ensure_fasttd3_import_path

ensure_fasttd3_import_path()
from fast_td3_utils import SimpleReplayBuffer  # type: ignore  # noqa: E402

from fasttd3_ptf.official_fasttd3_ptf.ptf_replay import PTFReplayWrapper  # noqa: E402
from fasttd3_ptf.official_fasttd3_ptf.train_ptf import (  # noqa: E402
    actor_updates_enabled,
    replay_candidate_masses,
)


def test_critic_first_actor_update_boundary_and_default() -> None:
    assert actor_updates_enabled(global_step=10_000, start_step=None)
    assert not actor_updates_enabled(global_step=11_999, start_step=12_000)
    assert actor_updates_enabled(global_step=12_000, start_step=12_000)
    assert actor_updates_enabled(global_step=12_001, start_step=12_000)


def _make_replay(capacity: int = 8, n_env: int = 2) -> PTFReplayWrapper:
    return PTFReplayWrapper(
        SimpleReplayBuffer(
            n_env=n_env,
            buffer_size=capacity,
            n_obs=3,
            n_act=2,
            n_critic_obs=3,
            n_steps=1,
            device=torch.device("cpu"),
        )
    )


def _transition(value: int, n_env: int = 2) -> TensorDict:
    env_offset = torch.arange(n_env, dtype=torch.float32).view(n_env, 1)
    return TensorDict(
        {
            "observations": torch.full((n_env, 3), float(value)) + env_offset,
            "actions": torch.full((n_env, 2), float(value) / 10.0) + env_offset,
            "next": TensorDict(
                {
                    "observations": torch.full((n_env, 3), float(value) + 0.5),
                    "rewards": torch.full((n_env,), float(value)),
                    "dones": torch.zeros(n_env),
                    "truncations": torch.zeros(n_env),
                },
                batch_size=(n_env,),
            ),
        },
        batch_size=(n_env,),
    )


def _filled(n_sources: int = 2) -> PTFReplayWrapper:
    """交替写入 student(-1) 与两个 source 的 provenance。"""
    replay = _make_replay()
    rows = [
        torch.tensor([-1, -1]),
        torch.tensor([0, 0]),
        torch.tensor([1, 1]),
        torch.tensor([-1, -1]),
        torch.tensor([0, 0]),
        torch.tensor([1, 1]),
    ]
    for value, options in enumerate(rows):
        replay.extend(_transition(value), options)
    return replay


# ---------------------------------------------------------------- 纯函数


def test_shared_mode_leaves_masses_untouched() -> None:
    """默认模式必须逐位等于输入，否则所有既有实验的数值会改变。"""
    masses = torch.tensor([0.3, 0.2, 0.5])
    out = replay_candidate_masses(masses, "shared")
    assert torch.equal(out, masses)


def test_student_only_zeroes_every_source_stratum() -> None:
    masses = torch.tensor([0.3, 0.2, 0.5])
    out = replay_candidate_masses(masses, "student_only")
    assert torch.equal(out, torch.tensor([0.0, 0.0, 1.0]))
    # student 质量必须是 1，否则 _admission_slot_weights 会因 mass_sum<=0 报错
    assert float(out[-1]) == 1.0
    assert float(out[:-1].sum()) == 0.0


def test_unknown_mode_is_rejected() -> None:
    try:
        replay_candidate_masses(torch.tensor([0.5, 0.5]), "bogus")
    except ValueError as exc:
        assert "bogus" in str(exc)
    else:
        raise AssertionError("unknown replay mode must raise")


# ------------------------------------------------- 与 replay 采样的真实交互


def test_admitted_source_can_still_get_zero_replay_quota() -> None:
    """核心：source 仍被准入（保留 behavior authority），但 critic 一条都不采。

    这与撤销准入不同——撤销后 source 连 behavior 都没有，无法构成 B-only 臂。
    """
    replay = _filled()
    replay.set_admission_policy(
        admitted_sources=torch.tensor([True, True]),          # 仍然准入
        candidate_masses=replay_candidate_masses(
            torch.tensor([0.25, 0.25, 0.5]), "student_only"
        ),
        uniform_mix=1.0,
    )
    batch = replay.sample(1024)
    assert torch.all(batch["options"] == -1), "student_only 下不得采到任何 source slot"

    audit = replay.admission_audit()
    assert audit is not None
    # source stratum 的 critic 采样数必须严格为 0
    assert audit["critic_sample_counts"][0] == 0
    assert audit["critic_sample_counts"][1] == 0
    assert audit["critic_sample_counts"][-1] > 0
    # 但 physical buffer 里 source 数据必须仍然存在（B 通道确实写入了）
    assert audit["active_buffer_counts"][0] > 0


def test_shared_mode_still_samples_sources() -> None:
    """对照：同一份 buffer 在 shared 模式下必须采得到 source，否则上一个测试无意义。"""
    replay = _filled()
    replay.set_admission_policy(
        admitted_sources=torch.tensor([True, True]),
        candidate_masses=replay_candidate_masses(
            torch.tensor([0.25, 0.25, 0.5]), "shared"
        ),
        uniform_mix=1.0,
    )
    replay.sample(1024)
    audit = replay.admission_audit()
    assert audit is not None
    assert sum(audit["critic_sample_counts"][:-1]) > 0


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
