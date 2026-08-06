"""load_anchor_core 白名单加载路径的单元测试(run card v2.1.2 附录 A)。

覆盖等价性测试清单的 toy 级验证:
- 测试 1/2:state digest 与 scheduler(LR) 恢复一致性;
- 测试 7:round-trip(load → re-save 白名单组件逐位一致);
- 白名单纪律:多传/少传/bundle 缺核心组件/带状态 reward_normalizer 均拒绝;
- named generator 不从 anchor 恢复(全局 RNG 恢复、generator 保持调用方状态)。

GPU 上的真实分支等价性(控制臂/duplicate/option 不参与)由
scripts/p0_equivalence_tests.py 在实机执行。
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from tensordict import TensorDict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fasttd3_ptf.official_fasttd3_ptf.anchor_io import (
    ANCHOR_CORE_MODULES,
    ANCHOR_CORE_OPTIMIZERS,
    ANCHOR_CORE_SCHEDULERS,
    load_anchor_core,
    save_anchor_bundle,
)
from fasttd3_ptf.official_fasttd3_ptf.paths import ensure_fasttd3_import_path

ensure_fasttd3_import_path()
from fast_td3_utils import SimpleReplayBuffer  # type: ignore  # noqa: E402

from fasttd3_ptf.official_fasttd3_ptf.ptf_replay import PTFReplayWrapper  # noqa: E402


def _replay() -> PTFReplayWrapper:
    return PTFReplayWrapper(
        SimpleReplayBuffer(
            n_env=2,
            buffer_size=8,
            n_obs=3,
            n_act=2,
            n_critic_obs=3,
            n_steps=1,
            device=torch.device("cpu"),
        )
    )


def _core_objects(seed: int):
    """构建与白名单键名一一对应的 toy learner 组件集。"""
    torch.manual_seed(seed)
    modules = {
        "actor": torch.nn.Linear(3, 2),
        "critic": torch.nn.Linear(3, 1),
        "critic_target": torch.nn.Linear(3, 1),
        "obs_normalizer": torch.nn.BatchNorm1d(3),
        "critic_obs_normalizer": torch.nn.BatchNorm1d(3),
    }
    modules["critic_target"].load_state_dict(modules["critic"].state_dict())
    optimizers = {
        "actor": torch.optim.AdamW(modules["actor"].parameters(), lr=1e-2),
        "critic": torch.optim.AdamW(modules["critic"].parameters(), lr=1e-2),
    }
    schedulers = {
        name: torch.optim.lr_scheduler.CosineAnnealingLR(optimizers[name], T_max=20)
        for name in ("actor", "critic")
    }
    scaler = torch.amp.GradScaler(enabled=False)
    return modules, optimizers, schedulers, scaler, _replay()


def _add_transition(replay: PTFReplayWrapper, value: int) -> None:
    obs = torch.full((2, 3), float(value))
    replay.extend(
        TensorDict(
            {
                "observations": obs,
                "actions": torch.full((2, 2), value / 10.0),
                "next": {
                    "rewards": torch.full((2,), float(value)),
                    "dones": torch.zeros(2, dtype=torch.long),
                    "truncations": torch.zeros(2, dtype=torch.long),
                    "observations": obs + 1,
                },
            },
            batch_size=2,
        ),
        torch.full((2,), -1),
    )


def _step_all(modules, optimizers, schedulers, steps: int) -> None:
    for _ in range(steps):
        for name in ("actor", "critic"):
            optimizers[name].zero_grad(set_to_none=True)
            x = torch.randn(4, 3)
            out = modules[name](modules["obs_normalizer"](x))
            out.pow(2).mean().backward()
            optimizers[name].step()
            schedulers[name].step()


def _save_anchor(tmp_path, modules, optimizers, schedulers, scaler, replay, *, extra_modules=None, generators=None):
    """按 train_ptf anchor 钩子的真实结构保存:含 option 族等白名单以外组件。"""
    full_modules = dict(modules)
    # anchor 真实 bundle 中的额外组件(option 族+无状态 reward_normalizer)。
    full_modules.setdefault("reward_normalizer", torch.nn.Identity())
    full_modules.update(extra_modules or {"option": torch.nn.Linear(3, 4), "option_target": torch.nn.Linear(3, 4)})
    option_optimizer = torch.optim.Adam(full_modules["option"].parameters(), lr=1e-3)
    return save_anchor_bundle(
        tmp_path / "anchor",
        completed_vector_steps=int(replay.ptr),
        num_envs=2,
        modules=full_modules,
        optimizers={**optimizers, "option": option_optimizer, "beta": option_optimizer},
        schedulers=dict(schedulers),
        scaler=scaler,
        replay=replay,
        configuration={"env": "toy"},
        auxiliary_state={"critic_update_count": 3, "actor_update_count": 1},
        generators=generators or {"option_selector": torch.Generator().manual_seed(123)},
        provenance_default={"behavior_source": "student"},
        repo_root=Path(__file__).resolve().parents[1],
        code_paths=[Path(__file__).resolve()],
    )


def test_core_resume_restores_state_lr_and_global_rng(tmp_path):
    random.seed(11)
    np.random.seed(11)
    torch.manual_seed(11)
    modules, optimizers, schedulers, scaler, replay = _core_objects(11)
    for step in range(3):
        _add_transition(replay, step)
    _step_all(modules, optimizers, schedulers, steps=3)
    # scheduler last_epoch 必须等于 anchor 完成步数(load_anchor_core 的断言语义)。
    assert int(schedulers["actor"].state_dict()["last_epoch"]) == int(replay.ptr)
    bundle = _save_anchor(tmp_path, modules, optimizers, schedulers, scaler, replay)
    anchor_lr = float(schedulers["actor"].get_last_lr()[0])

    # 基准:保存后全局 RNG 的下一次消耗与下一次参数更新。
    baseline_random = (random.random(), np.random.rand(), torch.rand(3))
    _step_all(modules, optimizers, schedulers, steps=1)
    baseline_actor = {k: v.detach().clone() for k, v in modules["actor"].state_dict().items()}

    # 分支进程:不同 seed 构建 → core-only 恢复 → 必须逐位复现基准。
    modules_b, optimizers_b, schedulers_b, scaler_b, replay_b = _core_objects(999)
    branch_generator = torch.Generator().manual_seed(777)
    generator_probe_before = torch.rand(3, generator=branch_generator)
    branch_generator.manual_seed(777)
    loaded = load_anchor_core(
        bundle,
        modules=modules_b,
        optimizers=optimizers_b,
        schedulers=schedulers_b,
        scaler=scaler_b,
        replay=replay_b,
    )
    assert loaded["completed_vector_steps"] == 3
    assert loaded["auxiliary_state"]["critic_update_count"] == 3

    # 全局 RNG 已恢复:三类流逐位复现。
    branch_random = (random.random(), np.random.rand(), torch.rand(3))
    assert baseline_random[0] == branch_random[0]
    assert baseline_random[1] == branch_random[1]
    torch.testing.assert_close(baseline_random[2], branch_random[2], rtol=0.0, atol=0.0)
    # named generator 未被 anchor 触碰:仍是调用方自己播种的流。
    torch.testing.assert_close(
        torch.rand(3, generator=branch_generator), generator_probe_before, rtol=0.0, atol=0.0
    )

    # 测试 2:resume 时刻的 LR 逐位等于 anchor 保存时刻的 LR。
    assert float(schedulers_b["actor"].get_last_lr()[0]) == anchor_lr
    # 测试 1 延伸:下一次参数更新逐位复现。
    _step_all(modules_b, optimizers_b, schedulers_b, steps=1)
    for name, value in modules_b["actor"].state_dict().items():
        torch.testing.assert_close(value, baseline_actor[name], rtol=0.0, atol=0.0)

    # replay 逐位一致。
    original = replay.gather(torch.tensor([[0, 2], [1, 0]]))
    restored = replay_b.gather(torch.tensor([[0, 2], [1, 0]]))
    for key in original.keys(True, True):
        assert torch.equal(original[key], restored[key]), key


def test_core_resume_roundtrip_resave_is_bitwise_identical(tmp_path):
    """等价性测试 7:load → 立即 re-save,白名单组件逐位一致。"""
    random.seed(23)
    np.random.seed(23)
    torch.manual_seed(23)
    modules, optimizers, schedulers, scaler, replay = _core_objects(23)
    for step in range(2):
        _add_transition(replay, step)
    _step_all(modules, optimizers, schedulers, steps=2)
    bundle = _save_anchor(tmp_path, modules, optimizers, schedulers, scaler, replay)

    modules_b, optimizers_b, schedulers_b, scaler_b, replay_b = _core_objects(555)
    load_anchor_core(
        bundle,
        modules=modules_b,
        optimizers=optimizers_b,
        schedulers=schedulers_b,
        scaler=scaler_b,
        replay=replay_b,
    )
    resaved = _save_anchor(
        tmp_path / "resave", modules_b, optimizers_b, schedulers_b, scaler_b, replay_b
    )
    original_learner = torch.load(bundle / "learner.pt", weights_only=False)
    resaved_learner = torch.load(resaved / "learner.pt", weights_only=False)
    for name in ANCHOR_CORE_MODULES:
        a, b = original_learner["modules"][name], resaved_learner["modules"][name]
        assert set(a) == set(b), name
        for key in a:
            assert torch.equal(a[key], b[key]), f"{name}.{key}"
    for name in ANCHOR_CORE_OPTIMIZERS:
        a = original_learner["optimizers"][name]["state"]
        b = resaved_learner["optimizers"][name]["state"]
        assert set(a) == set(b), name
        for idx in a:
            for key in a[idx]:
                lhs, rhs = a[idx][key], b[idx][key]
                if isinstance(lhs, torch.Tensor):
                    assert torch.equal(lhs, rhs), f"{name}.{idx}.{key}"
                else:
                    assert lhs == rhs, f"{name}.{idx}.{key}"
    for name in ANCHOR_CORE_SCHEDULERS:
        a = original_learner["schedulers"][name]
        b = resaved_learner["schedulers"][name]
        for key in a:
            lhs, rhs = a[key], b[key]
            if isinstance(lhs, torch.Tensor):
                assert torch.equal(lhs, rhs), f"{name}.{key}"
            else:
                assert lhs == rhs, f"{name}.{key}"
    original_replay = torch.load(bundle / "replay.pt", weights_only=False)
    resaved_replay = torch.load(resaved / "replay.pt", weights_only=False)
    for key, value in original_replay["tensors"].items():
        assert torch.equal(value, resaved_replay["tensors"][key]), key


def test_core_resume_rejects_non_whitelist_supplied(tmp_path):
    modules, optimizers, schedulers, scaler, replay = _core_objects(31)
    _add_transition(replay, 0)
    _step_all(modules, optimizers, schedulers, steps=1)
    bundle = _save_anchor(tmp_path, modules, optimizers, schedulers, scaler, replay)
    modules_b, optimizers_b, schedulers_b, scaler_b, replay_b = _core_objects(32)
    # 多传 option → 拒绝(白名单必须恰好相等,防越权加载)。
    with pytest.raises(ValueError, match="must equal the whitelist"):
        load_anchor_core(
            bundle,
            modules={**modules_b, "option": torch.nn.Linear(3, 4)},
            optimizers=optimizers_b,
            schedulers=schedulers_b,
            scaler=scaler_b,
            replay=replay_b,
        )
    # 少传核心模块 → 拒绝(防漏加载)。
    missing = dict(modules_b)
    missing.pop("critic_target")
    with pytest.raises(ValueError, match="must equal the whitelist"):
        load_anchor_core(
            bundle,
            modules=missing,
            optimizers=optimizers_b,
            schedulers=schedulers_b,
            scaler=scaler_b,
            replay=replay_b,
        )


def test_core_resume_rejects_bundle_missing_core_or_stateful_reward_norm(tmp_path):
    modules, optimizers, schedulers, scaler, replay = _core_objects(41)
    _add_transition(replay, 0)
    _step_all(modules, optimizers, schedulers, steps=1)
    # anchor 缺核心模块(篡改场景):从保存字典中剔除 critic_target。
    bundle = _save_anchor(tmp_path, modules, optimizers, schedulers, scaler, replay)
    learner = torch.load(bundle / "learner.pt", weights_only=False)
    learner["modules"].pop("critic_target")
    torch.save(learner, bundle / "learner.pt")
    # checksums 会先失败——这是预期防线;绕过校验的缺组件场景由白名单自身兜底,
    # 此处验证完整性防线即可。
    modules_b, optimizers_b, schedulers_b, scaler_b, replay_b = _core_objects(42)
    with pytest.raises(IOError, match="integrity"):
        load_anchor_core(
            bundle,
            modules=modules_b,
            optimizers=optimizers_b,
            schedulers=schedulers_b,
            scaler=scaler_b,
            replay=replay_b,
        )

    # 带状态 reward_normalizer → 拒绝(P0 冻结 reward_normalization=False)。
    modules2, optimizers2, schedulers2, scaler2, replay2 = _core_objects(43)
    _add_transition(replay2, 0)
    _step_all(modules2, optimizers2, schedulers2, steps=1)
    bundle2 = _save_anchor(
        tmp_path / "stateful",
        modules2,
        optimizers2,
        schedulers2,
        scaler2,
        replay2,
        extra_modules={
            "option": torch.nn.Linear(3, 4),
            "option_target": torch.nn.Linear(3, 4),
            "reward_normalizer": torch.nn.BatchNorm1d(1),
        },
    )
    modules_c, optimizers_c, schedulers_c, scaler_c, replay_c = _core_objects(44)
    with pytest.raises(ValueError, match="reward_normalizer"):
        load_anchor_core(
            bundle2,
            modules=modules_c,
            optimizers=optimizers_c,
            schedulers=schedulers_c,
            scaler=scaler_c,
            replay=replay_c,
        )


@pytest.mark.parametrize("group_count", [2, 3])
def test_core_resume_provenance_roundtrip_all_fields(tmp_path, group_count):
    """五次复核审计缺口 1:provenance 全字段(含 segment_id/env_rank/learner_step
    标量元数据)在 2-group(truck)与 3-group(crawl)schema 下逐位往返。"""
    modules, optimizers, schedulers, scaler, replay = _core_objects(61 + group_count)
    replay.enable_provenance(group_count)
    for step in range(3):
        obs = torch.full((2, 3), float(step))
        replay.extend(
            TensorDict(
                {
                    "observations": obs,
                    "actions": torch.full((2, 2), step / 10.0),
                    "next": {
                        "rewards": torch.full((2,), float(step)),
                        "dones": torch.zeros(2, dtype=torch.long),
                        "truncations": torch.zeros(2, dtype=torch.long),
                        "observations": obs + 1,
                    },
                },
                batch_size=2,
            ),
            torch.full((2,), -1),
            provenance={
                "behavior_source": torch.full((2,), -1, dtype=torch.int16),
                "source_by_group": torch.full((2, group_count), -1, dtype=torch.int16),
                "executed_group_mask": torch.zeros(2, group_count, dtype=torch.bool),
                "segment_id": torch.tensor([step * 2, step * 2 + 1], dtype=torch.int64),
                "segment_step": torch.full((2,), step, dtype=torch.int16),
                "anchor_id": torch.full((2,), -1, dtype=torch.int32),
                "env_rank": torch.tensor([0, 1], dtype=torch.int16),
                "learner_step": torch.full((2,), step, dtype=torch.int64),
            },
        )
    _step_all(modules, optimizers, schedulers, steps=3)
    assert replay.max_provenance_segment_id() == 5
    bundle = _save_anchor(tmp_path, modules, optimizers, schedulers, scaler, replay)

    modules_b, optimizers_b, schedulers_b, scaler_b, replay_b = _core_objects(70 + group_count)
    # 分支按运行时组数预启用(优先方案:anchor 组数=任务目标组数,精确匹配)。
    replay_b.enable_provenance(group_count)
    load_anchor_core(
        bundle,
        modules=modules_b,
        optimizers=optimizers_b,
        schedulers=schedulers_b,
        scaler=scaler_b,
        replay=replay_b,
    )
    assert replay_b.provenance_enabled
    assert replay_b.max_provenance_segment_id() == 5
    original = replay.export_valid(require_complete_provenance=True)
    restored = replay_b.export_valid(require_complete_provenance=True)
    assert original["metadata"]["provenance_group_count"] == group_count
    assert restored["metadata"]["provenance_group_count"] == group_count
    for name, value in original["provenance"].items():
        assert torch.equal(value, restored["provenance"][name]), f"provenance.{name}"

    # 组数不匹配的 bundle 必须被拒绝(enable_provenance 的组数守卫)。
    modules_c, optimizers_c, schedulers_c, scaler_c, replay_c = _core_objects(80 + group_count)
    replay_c.enable_provenance(group_count + 1)
    with pytest.raises(ValueError, match="provenance already configured"):
        load_anchor_core(
            bundle,
            modules=modules_c,
            optimizers=optimizers_c,
            schedulers=schedulers_c,
            scaler=scaler_c,
            replay=replay_c,
        )


def test_core_resume_rejects_scheduler_step_mismatch(tmp_path):
    """scheduler last_epoch 与 anchor 步数不一致 → 拒绝(防 LR 日程错位)。"""
    modules, optimizers, schedulers, scaler, replay = _core_objects(51)
    for step in range(3):
        _add_transition(replay, step)
    # 只走 2 次 scheduler.step,replay.ptr=3 → 不一致。
    _step_all(modules, optimizers, schedulers, steps=2)
    bundle = _save_anchor(tmp_path, modules, optimizers, schedulers, scaler, replay)
    modules_b, optimizers_b, schedulers_b, scaler_b, replay_b = _core_objects(52)
    with pytest.raises(AssertionError, match="last_epoch"):
        load_anchor_core(
            bundle,
            modules=modules_b,
            optimizers=optimizers_b,
            schedulers=schedulers_b,
            scaler=scaler_b,
            replay=replay_b,
        )
