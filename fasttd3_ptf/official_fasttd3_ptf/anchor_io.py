"""Paper-grade learner anchor bundles for controlled FastTD3 forks."""

from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
import uuid
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from fasttd3_ptf.official_fasttd3_ptf.ptf_replay import PTFReplayWrapper


ANCHOR_SCHEMA_VERSION = 1


def _cpu_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    return value


def _module_state(module: Any) -> dict[str, Any]:
    if hasattr(module, "module"):
        module = module.module
    return _cpu_tree(module.state_dict())


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.device):
        return str(value)
    return repr(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_state(repo_root: Path) -> dict[str, Any]:
    def run(*args: str) -> str | None:
        result = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else None

    head = run("rev-parse", "HEAD")
    status = run("status", "--porcelain=v1", "--untracked-files=normal")
    return {
        "head": head,
        "dirty": bool(status),
        "status_sha256": None
        if status is None
        else hashlib.sha256(status.encode("utf-8")).hexdigest(),
    }


def capture_rng_state(
    generators: Mapping[str, torch.Generator] | None = None,
) -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state().cpu(),
        # Capture only the active training device. get_rng_state_all() creates
        # CUDA contexts on every shared-server GPU and is unnecessary for this
        # single-device experiment.
        "torch_cuda": {
            "device_index": int(torch.cuda.current_device()),
            "state": torch.cuda.get_rng_state().cpu(),
        }
        if torch.cuda.is_available()
        else None,
        "generators": {
            name: {
                "device": str(generator.device),
                "state": generator.get_state().cpu(),
            }
            for name, generator in (generators or {}).items()
        },
    }


def restore_global_rng_state(state: Mapping[str, Any]) -> None:
    """只恢复全局四类 RNG（python/numpy/torch_cpu/torch_cuda），不触碰 named
    generators——core-only resume 的 named generator（option_selector）按分支
    seed 面板重新播种，不从 anchor 继承（run card v2.1.2 附录 A.1）。"""

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(torch.as_tensor(state["torch_cpu"], device="cpu", dtype=torch.uint8))
    cuda_state = state.get("torch_cuda")
    if cuda_state:
        if not torch.cuda.is_available():
            raise RuntimeError("anchor contains a CUDA RNG state but CUDA is unavailable")
        if isinstance(cuda_state, dict):
            torch.cuda.set_rng_state(cuda_state["state"].cpu())
        else:
            # Backward compatibility for schema-v1 bundles written before the
            # active-device-only fix. Restore the current device entry without
            # creating contexts on every GPU.
            current = int(torch.cuda.current_device())
            if current >= len(cuda_state):
                raise RuntimeError(
                    f"legacy CUDA RNG state lacks current device {current}"
                )
            torch.cuda.set_rng_state(cuda_state[current].cpu())


def restore_rng_state(
    state: Mapping[str, Any],
    generators: Mapping[str, torch.Generator] | None = None,
) -> None:
    """Restore RNG last, after model/optimizer construction and state loading."""

    restore_global_rng_state(state)
    supplied = generators or {}
    expected_names = set((state.get("generators") or {}).keys())
    if set(supplied) != expected_names:
        raise ValueError(
            f"named generator mismatch: anchor={sorted(expected_names)}, "
            f"runtime={sorted(supplied)}"
        )
    for name, generator in supplied.items():
        generator.set_state(state["generators"][name]["state"].cpu())


def save_anchor_bundle(
    output_dir: str | Path,
    *,
    completed_vector_steps: int,
    num_envs: int,
    modules: Mapping[str, Any],
    optimizers: Mapping[str, torch.optim.Optimizer],
    schedulers: Mapping[str, Any],
    scaler: Any,
    replay: PTFReplayWrapper,
    configuration: Mapping[str, Any],
    auxiliary_state: Mapping[str, Any] | None = None,
    generators: Mapping[str, torch.Generator] | None = None,
    provenance_default: Mapping[str, Any] | None = None,
    require_complete_replay_provenance: bool = False,
    repo_root: str | Path | None = None,
    code_paths: Sequence[str | Path] = (),
) -> Path:
    """Atomically write ``manifest/learner/replay/rng/checksums`` files."""

    target = Path(output_dir).resolve()
    if target.exists():
        raise FileExistsError(f"anchor bundle already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.parent / f".{target.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
    temporary.mkdir()

    completed_vector_steps = int(completed_vector_steps)
    if int(replay.ptr) != completed_vector_steps:
        raise AssertionError(
            f"anchor boundary mismatch: replay.ptr={replay.ptr}, "
            f"completed_vector_steps={completed_vector_steps}"
        )
    learner = {
        "schema_version": ANCHOR_SCHEMA_VERSION,
        "completed_vector_steps": completed_vector_steps,
        "environment_transitions": completed_vector_steps * int(num_envs),
        "modules": {name: _module_state(module) for name, module in modules.items()},
        "optimizers": {
            name: _cpu_tree(optimizer.state_dict()) for name, optimizer in optimizers.items()
        },
        "schedulers": {
            name: _cpu_tree(scheduler.state_dict()) for name, scheduler in schedulers.items()
        },
        "scaler": _cpu_tree(scaler.state_dict()),
        "configuration": _cpu_tree(dict(configuration)),
        "auxiliary_state": _cpu_tree(dict(auxiliary_state or {})),
    }
    replay_state = replay.export_valid(
        require_complete_provenance=require_complete_replay_provenance
    )
    rng_state = capture_rng_state(generators)

    torch.save(learner, temporary / "learner.pt")
    torch.save(replay_state, temporary / "replay.pt")
    torch.save(rng_state, temporary / "rng.pt")

    root = Path(repo_root).resolve() if repo_root is not None else Path.cwd().resolve()
    code_hashes: dict[str, str] = {}
    for path_like in code_paths:
        path = Path(path_like).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"anchor code dependency does not exist: {path}")
        try:
            label = str(path.relative_to(root))
        except ValueError:
            label = str(path)
        code_hashes[label] = _sha256(path)

    checksums = {
        name: {
            "sha256": _sha256(temporary / name),
            "bytes": int((temporary / name).stat().st_size),
        }
        for name in ("learner.pt", "replay.pt", "rng.pt")
    }
    (temporary / "checksums.json").write_text(
        json.dumps(checksums, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": ANCHOR_SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "completed_vector_steps": completed_vector_steps,
        "environment_transitions": completed_vector_steps * int(num_envs),
        "num_envs": int(num_envs),
        "replay_ptr": int(replay.ptr),
        "replay_valid_size": int(replay.valid_size),
        "modules": sorted(modules),
        "optimizers": sorted(optimizers),
        "schedulers": sorted(schedulers),
        "provenance_default": _jsonable(provenance_default),
        "replay_has_per_transition_provenance": bool(replay._provenance),
        "configuration": _jsonable(configuration),
        "git": _git_state(root),
        "code_sha256": code_hashes,
        "files": checksums,
    }
    (temporary / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, target)
    return target


def verify_anchor_bundle(bundle_dir: str | Path) -> dict[str, Any]:
    bundle = Path(bundle_dir).resolve()
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    checksums = json.loads((bundle / "checksums.json").read_text(encoding="utf-8"))
    for name, expected in checksums.items():
        path = bundle / name
        actual_hash = _sha256(path)
        actual_size = int(path.stat().st_size)
        if actual_hash != expected["sha256"] or actual_size != int(expected["bytes"]):
            raise IOError(
                f"anchor integrity failure for {name}: "
                f"sha256={actual_hash}, bytes={actual_size}"
            )
    if manifest.get("files") != checksums:
        raise IOError("manifest/checksums disagreement")
    return manifest


def load_anchor_bundle(
    bundle_dir: str | Path,
    *,
    modules: Mapping[str, Any],
    optimizers: Mapping[str, torch.optim.Optimizer],
    schedulers: Mapping[str, Any],
    scaler: Any,
    replay: PTFReplayWrapper,
    generators: Mapping[str, torch.Generator] | None = None,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> dict[str, Any]:
    """Verify and restore a complete anchor, with RNG restoration last."""

    bundle = Path(bundle_dir).resolve()
    manifest = verify_anchor_bundle(bundle)
    learner = torch.load(bundle / "learner.pt", map_location=map_location, weights_only=False)
    replay_state = torch.load(bundle / "replay.pt", map_location="cpu", weights_only=False)
    rng_state = torch.load(bundle / "rng.pt", map_location="cpu", weights_only=False)
    if int(learner.get("schema_version", -1)) != ANCHOR_SCHEMA_VERSION:
        raise ValueError("unsupported learner anchor schema")

    expected_components = {
        "modules": set(modules),
        "optimizers": set(optimizers),
        "schedulers": set(schedulers),
    }
    for category, expected in expected_components.items():
        present = set(learner[category])
        if strict and present != expected:
            raise ValueError(
                f"anchor {category} mismatch: anchor={sorted(present)}, "
                f"runtime={sorted(expected)}"
            )
    for name, module in modules.items():
        target = module.module if hasattr(module, "module") else module
        target.load_state_dict(learner["modules"][name], strict=strict)
    for name, optimizer in optimizers.items():
        optimizer.load_state_dict(learner["optimizers"][name])
    for name, scheduler in schedulers.items():
        scheduler.load_state_dict(learner["schedulers"][name])
    scaler.load_state_dict(learner["scaler"])
    replay.import_valid(replay_state, strict=strict)

    completed = int(learner["completed_vector_steps"])
    if replay.ptr != completed or completed != int(manifest["completed_vector_steps"]):
        raise AssertionError("loaded anchor step/replay pointer mismatch")
    restore_rng_state(rng_state, generators)
    return {
        "manifest": manifest,
        "completed_vector_steps": completed,
        "environment_transitions": int(learner["environment_transitions"]),
        "configuration": learner["configuration"],
        "auxiliary_state": learner["auxiliary_state"],
    }


# core-only resume 的严格白名单（run card v2.1.2 附录 A.1）：只恢复核心
# learner；option 族（module/target/optimizer）与 admission/MCG 状态由分支
# 配置重新构建——bootstrap_only+gate 关下它们不参与任何行为与 loss。
# 键名与 save_anchor_bundle 保存时的命名一致（train_ptf.py anchor 钩子）。
ANCHOR_CORE_MODULES = (
    "actor",
    "critic",
    "critic_target",
    "obs_normalizer",
    "critic_obs_normalizer",
)
ANCHOR_CORE_OPTIMIZERS = ("actor", "critic")
ANCHOR_CORE_SCHEDULERS = ("actor", "critic")


def load_anchor_core(
    bundle_dir: str | Path,
    *,
    modules: Mapping[str, Any],
    optimizers: Mapping[str, torch.optim.Optimizer],
    schedulers: Mapping[str, Any],
    scaler: Any,
    replay: PTFReplayWrapper,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """白名单加载核心 learner；bundle 中白名单以外的组件（option 族等）被忽略。

    与 :func:`load_anchor_bundle` 的区别：
    - 调用方必须恰好传入白名单集合（防漏加载/防越权加载）；
    - bundle 可以包含额外组件（anchor 保存了 option 族），只加载白名单；
    - named generators 一律不恢复（由调用方按分支 seed 面板重新播种）；
    - 强制校验 reward_normalizer 为无状态（冻结 reward_normalization=False）。
    """

    bundle = Path(bundle_dir).resolve()
    manifest = verify_anchor_bundle(bundle)
    learner = torch.load(bundle / "learner.pt", map_location=map_location, weights_only=False)
    replay_state = torch.load(bundle / "replay.pt", map_location="cpu", weights_only=False)
    rng_state = torch.load(bundle / "rng.pt", map_location="cpu", weights_only=False)
    if int(learner.get("schema_version", -1)) != ANCHOR_SCHEMA_VERSION:
        raise ValueError("unsupported learner anchor schema")

    whitelist = {
        "modules": set(ANCHOR_CORE_MODULES),
        "optimizers": set(ANCHOR_CORE_OPTIMIZERS),
        "schedulers": set(ANCHOR_CORE_SCHEDULERS),
    }
    supplied = {
        "modules": set(modules),
        "optimizers": set(optimizers),
        "schedulers": set(schedulers),
    }
    for category, expected in whitelist.items():
        if supplied[category] != expected:
            raise ValueError(
                f"core resume {category} must equal the whitelist: "
                f"expected={sorted(expected)}, supplied={sorted(supplied[category])}"
            )
        missing = expected - set(learner[category])
        if missing:
            raise ValueError(
                f"anchor bundle lacks core {category}: {sorted(missing)}"
            )

    # 冻结校验：P0 全分支 reward_normalization=False（h1hand 历史默认），
    # 对应 nn.Identity 的空 state_dict；非空即 anchor 与协议不符，直接拒绝。
    reward_state = learner["modules"].get("reward_normalizer")
    if reward_state:
        raise ValueError(
            "anchor reward_normalizer has state; P0 requires reward_normalization=False"
        )

    for name in ANCHOR_CORE_MODULES:
        module = modules[name]
        target = module.module if hasattr(module, "module") else module
        target.load_state_dict(learner["modules"][name], strict=True)
    for name in ANCHOR_CORE_OPTIMIZERS:
        optimizers[name].load_state_dict(learner["optimizers"][name])
    for name in ANCHOR_CORE_SCHEDULERS:
        schedulers[name].load_state_dict(learner["schedulers"][name])
    scaler.load_state_dict(learner["scaler"])
    replay.import_valid(replay_state, strict=True)

    completed = int(learner["completed_vector_steps"])
    if replay.ptr != completed or completed != int(manifest["completed_vector_steps"]):
        raise AssertionError("loaded anchor step/replay pointer mismatch")
    # scheduler 恢复正确性断言：last_epoch 必须等于 anchor 的完成步数
    # （每 vector step 调一次 scheduler.step 的训练循环语义）。
    for name in ANCHOR_CORE_SCHEDULERS:
        last_epoch = int(schedulers[name].state_dict()["last_epoch"])
        if last_epoch != completed:
            raise AssertionError(
                f"scheduler '{name}' last_epoch={last_epoch} != anchor step {completed}"
            )
    restore_global_rng_state(rng_state)
    return {
        "manifest": manifest,
        "completed_vector_steps": completed,
        "environment_transitions": int(learner["environment_transitions"]),
        "configuration": learner["configuration"],
        "auxiliary_state": learner["auxiliary_state"],
    }
