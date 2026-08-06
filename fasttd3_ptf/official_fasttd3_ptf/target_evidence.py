"""Task-agnostic target evidence contract for transferability probes.

The admission algorithm consumes a fixed tuple:

    target return, target progress, target feasibility components

Task semantics live in a declarative YAML adapter.  This is analogous to an
observation/action adapter: Crawl may declare posture/tunnel constraints while
a manipulation task may declare object progress/contact constraints, without
adding task-name branches to the admission mechanism.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml


def _reduce(values: list[float], reducer: str) -> float:
    if not values:
        raise ValueError("cannot reduce an empty evidence trace")
    array = np.asarray(values, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError("target evidence contains a non-finite value")
    if reducer == "sum":
        return float(array.sum())
    if reducer == "mean":
        return float(array.mean())
    if reducer == "min":
        return float(array.min())
    if reducer == "max":
        return float(array.max())
    if reducer == "final":
        return float(array[-1])
    raise ValueError(f"unsupported reducer={reducer!r}")


@dataclass(frozen=True)
class ProgressSpec:
    kind: str
    direction: str
    array: str | None = None
    index: int | None = None
    info_key: str | None = None
    temporal_reducer: str = "sum"
    gate_components: tuple[str, ...] = ()
    gate_reducer: str = "product"

    @property
    def sign(self) -> float:
        if self.direction == "maximize":
            return 1.0
        if self.direction == "minimize":
            return -1.0
        raise ValueError(f"unsupported progress direction={self.direction!r}")


@dataclass(frozen=True)
class FeasibilitySpec:
    name: str
    info_keys: tuple[str, ...]
    step_reducer: str
    temporal_reducer: str
    direction: str

    @property
    def sign(self) -> float:
        if self.direction == "maximize":
            return 1.0
        if self.direction == "minimize":
            return -1.0
        raise ValueError(f"unsupported feasibility direction={self.direction!r}")


@dataclass(frozen=True)
class TargetEvidenceContract:
    name: str
    env_name: str
    progress: ProgressSpec
    feasibility: tuple[FeasibilitySpec, ...]
    hard_constraints: tuple[str, ...]

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "TargetEvidenceContract":
        if int(raw.get("schema_version", -1)) != 1:
            raise ValueError("target evidence schema_version must equal 1")
        progress_raw = raw.get("progress")
        if not isinstance(progress_raw, Mapping):
            raise ValueError("target evidence progress must be a mapping")
        kind = str(progress_raw.get("kind"))
        progress = ProgressSpec(
            kind=kind,
            direction=str(progress_raw.get("direction", "maximize")),
            array=progress_raw.get("array"),
            index=(
                int(progress_raw["index"])
                if progress_raw.get("index") is not None
                else None
            ),
            info_key=progress_raw.get("info_key"),
            temporal_reducer=str(progress_raw.get("temporal_reducer", "sum")),
            gate_components=tuple(
                str(name) for name in (progress_raw.get("gate_components") or ())
            ),
            gate_reducer=str(progress_raw.get("gate_reducer", "product")),
        )
        if kind == "sim_state_delta":
            if progress.array not in {"qpos", "qvel"} or progress.index is None:
                raise ValueError("sim_state_delta progress requires array=qpos|qvel and index")
        elif kind == "info":
            if not progress.info_key:
                raise ValueError("info progress requires info_key")
        else:
            raise ValueError(f"unsupported progress kind={kind!r}")
        _ = progress.sign

        feasibility = []
        for item in raw.get("feasibility") or []:
            if not isinstance(item, Mapping):
                raise ValueError("each feasibility item must be a mapping")
            keys = tuple(str(key) for key in (item.get("info_keys") or ()))
            if not keys:
                raise ValueError("feasibility item requires at least one info_key")
            spec = FeasibilitySpec(
                name=str(item.get("name")),
                info_keys=keys,
                step_reducer=str(item.get("step_reducer", "min")),
                temporal_reducer=str(item.get("temporal_reducer", "mean")),
                direction=str(item.get("direction", "maximize")),
            )
            if not spec.name:
                raise ValueError("feasibility item requires a name")
            _ = spec.sign
            feasibility.append(spec)
        names = [spec.name for spec in feasibility]
        if len(names) != len(set(names)):
            raise ValueError("feasibility names must be unique")
        unknown_gates = sorted(set(progress.gate_components).difference(names))
        if unknown_gates:
            raise ValueError(f"unknown progress gate components: {unknown_gates}")
        if progress.gate_reducer not in {"product", "min", "mean"}:
            raise ValueError(
                f"unsupported progress gate_reducer={progress.gate_reducer!r}"
            )
        hard_constraints = tuple(str(name) for name in (raw.get("hard_constraints") or ()))
        unknown_hard = sorted(set(hard_constraints).difference(names))
        if unknown_hard:
            raise ValueError(f"unknown hard constraints: {unknown_hard}")
        if len(hard_constraints) != len(set(hard_constraints)):
            raise ValueError("hard constraint names must be unique")
        return cls(
            name=str(raw.get("name")),
            env_name=str(raw.get("env_name")),
            progress=progress,
            feasibility=tuple(feasibility),
            hard_constraints=hard_constraints,
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TargetEvidenceContract":
        path = Path(path)
        raw = yaml.safe_load(path.read_text())
        if not isinstance(raw, Mapping):
            raise ValueError(f"{path} must contain a mapping")
        return cls.from_mapping(raw)

    def new_accumulator(self, env) -> "TargetEvidenceAccumulator":
        return TargetEvidenceAccumulator(self, env)


class TargetEvidenceAccumulator:
    def __init__(self, contract: TargetEvidenceContract, env) -> None:
        self.contract = contract
        self.env = env
        self._progress_start = self._read_sim_progress() if contract.progress.kind == "sim_state_delta" else None
        self._progress_trace: list[float] = []
        self._feasibility_traces = {spec.name: [] for spec in contract.feasibility}

    def _read_sim_progress(self) -> float:
        spec = self.contract.progress
        array = getattr(self.env.unwrapped.data, str(spec.array))
        return float(array[int(spec.index)])

    def observe(self, info: Mapping[str, Any]) -> None:
        progress = self.contract.progress
        if progress.kind == "info":
            if progress.info_key not in info:
                raise KeyError(f"missing progress info key {progress.info_key!r}")
            self._progress_trace.append(float(info[progress.info_key]))
        for spec in self.contract.feasibility:
            missing = [key for key in spec.info_keys if key not in info]
            if missing:
                raise KeyError(f"missing feasibility info keys for {spec.name}: {missing}")
            step_values = [float(info[key]) for key in spec.info_keys]
            self._feasibility_traces[spec.name].append(
                _reduce(step_values, spec.step_reducer)
            )

    def finish(self) -> dict[str, Any]:
        progress = self.contract.progress
        if progress.kind == "sim_state_delta":
            value = self._read_sim_progress() - float(self._progress_start)
        else:
            value = _reduce(self._progress_trace, progress.temporal_reducer)
        feasibility = {
            spec.name: spec.sign
            * _reduce(self._feasibility_traces[spec.name], spec.temporal_reducer)
            for spec in self.contract.feasibility
        }
        if progress.gate_components:
            gate_values = [feasibility[name] for name in progress.gate_components]
            if progress.gate_reducer == "product":
                gate = float(np.prod(gate_values))
            else:
                gate = _reduce(gate_values, progress.gate_reducer)
            value *= gate
        return {
            "progress": progress.sign * value,
            "feasibility": feasibility,
        }
