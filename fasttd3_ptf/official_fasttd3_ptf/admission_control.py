"""Source-admission snapshots and the optional adaptive revocation state machine.

The training loop consumes an immutable admission decision; it does not infer
source utility.  Explicit snapshots keep exact abstention, replay lifecycle,
and optional MCG authority independent from the estimator.  The adaptive
controller below is a pure-CPU, segment-window heuristic that only emits new
immutable snapshots; it never mutates behavior or replay itself.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch

from fasttd3_ptf.config import load_yaml


SCHEMA_VERSION = 1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class AdmissionSnapshot:
    """不可变的准入决策快照:训练循环只消费它,不做任何 utility 推断。

    admission_bootstrap 的核心数据结构——admitted 源 + student 组成单一
    categorical(见 candidate_probabilities),rejected 源概率严格为 0;
    全拒绝(exact_abstain)时行为确定性回退纯 student,不消耗抽样 RNG。
    """

    source_names: tuple[str, ...]
    admitted: tuple[bool, ...]
    source_logits: tuple[float, ...]
    student_logit: float
    decision_id: str
    mode: str
    quarantine_artifact: str | None = None
    quarantine_digest: str | None = None

    def __post_init__(self) -> None:
        size = len(self.source_names)
        if len(self.admitted) != size or len(self.source_logits) != size:
            raise ValueError("source_names, admitted, and source_logits must have equal length")
        if len(set(self.source_names)) != size:
            raise ValueError("source_names must be unique")

    @property
    def exact_abstain(self) -> bool:
        return not any(self.admitted)

    @property
    def admitted_names(self) -> tuple[str, ...]:
        return tuple(name for name, keep in zip(self.source_names, self.admitted) if keep)

    def admitted_tensor(self, device: torch.device | str) -> torch.Tensor:
        return torch.tensor(self.admitted, dtype=torch.bool, device=device)

    def candidate_probabilities(
        self,
        *,
        tau: float,
        device: torch.device | str = "cpu",
    ) -> torch.Tensor:
        """Return probabilities for ``sources + student``.

        Rejected sources have exactly zero probability.  If every source is
        rejected, the result is exactly one-hot on the student and no random
        draw is required by the behavior controller.
        """

        if tau <= 0:
            raise ValueError("admission tau must be positive")
        size = len(self.source_names)
        if self.exact_abstain:
            out = torch.zeros(size + 1, dtype=torch.float32, device=device)
            out[-1] = 1.0
            return out
        logits = torch.tensor(
            [*self.source_logits, float(self.student_logit)],
            dtype=torch.float32,
            device=device,
        ) / float(tau)
        admitted = self.admitted_tensor(device)
        logits[:-1] = torch.where(
            admitted,
            logits[:-1],
            torch.full_like(logits[:-1], float("-inf")),
        )
        return torch.softmax(logits, dim=0)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "decision_id": self.decision_id,
            "mode": self.mode,
            "source_names": list(self.source_names),
            "admitted_sources": list(self.admitted_names),
            "source_logits": list(self.source_logits),
            "student_logit": float(self.student_logit),
            "exact_abstain": self.exact_abstain,
            "quarantine_artifact": self.quarantine_artifact,
            "quarantine_digest": self.quarantine_digest,
        }


@dataclass(frozen=True)
class AdmissionSchedule:
    """Explicit step-indexed admission decisions with no embedded estimator."""

    decisions: tuple[tuple[int, AdmissionSnapshot], ...]

    def __post_init__(self) -> None:
        steps = [step for step, _ in self.decisions]
        if not steps or steps[0] != 0:
            raise ValueError("admission schedule must start at step 0")
        if steps != sorted(set(steps)):
            raise ValueError("admission schedule steps must be unique and increasing")

    def snapshot_at(self, step: int) -> AdmissionSnapshot:
        current = self.decisions[0][1]
        for change_step, snapshot in self.decisions:
            if change_step > int(step):
                break
            current = snapshot
        return current


@dataclass(frozen=True)
class CandidateWindowStatistics:
    """Immutable normal-approximate statistics for one completed stage window."""

    candidate: str
    count: int
    mean: float | None
    variance: float | None
    standard_error: float | None
    lower_bound: float | None
    upper_bound: float | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidate": self.candidate,
            "count": int(self.count),
            "mean": self.mean,
            "variance": self.variance,
            "standard_error": self.standard_error,
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
        }


@dataclass(frozen=True)
class AdaptiveAdmissionWindowResult:
    """One immutable window audit, optionally carrying a new admission decision."""

    completed_step: int
    window_index: int
    statistics: tuple[CandidateWindowStatistics, ...]
    positive_votes: tuple[bool, ...]
    persistence_counts: tuple[int, ...]
    revoked_sources: tuple[str, ...]
    snapshot: AdmissionSnapshot | None

    @property
    def decision_applied(self) -> bool:
        return self.snapshot is not None

    def as_dict(self) -> dict[str, Any]:
        return {
            "event": "adaptive_admission_window",
            "completed_step": int(self.completed_step),
            "window_index": int(self.window_index),
            "statistics": [value.as_dict() for value in self.statistics],
            "positive_votes": list(self.positive_votes),
            "persistence_counts": list(self.persistence_counts),
            "revoked_sources": list(self.revoked_sources),
            "decision_applied": self.decision_applied,
            "decision": self.snapshot.as_dict() if self.snapshot is not None else None,
        }


@dataclass
class _RunningMoments:
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def update(self, value: float) -> None:
        value = float(value)
        if not math.isfinite(value):
            raise ValueError("segment mean reward must be finite")
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (value - self.mean)

    def snapshot(self, *, candidate: str, z: float) -> CandidateWindowStatistics:
        if self.count == 0:
            return CandidateWindowStatistics(candidate, 0, None, None, None, None, None)
        if self.count < 2:
            return CandidateWindowStatistics(
                candidate, self.count, float(self.mean), None, None, None, None
            )
        variance = max(0.0, self.m2 / (self.count - 1))
        standard_error = math.sqrt(variance / self.count)
        radius = float(z) * standard_error
        return CandidateWindowStatistics(
            candidate=candidate,
            count=self.count,
            mean=float(self.mean),
            variance=float(variance),
            standard_error=float(standard_error),
            lower_bound=float(self.mean - radius),
            upper_bound=float(self.mean + radius),
        )


class AdaptiveAdmissionController:
    """Stage-local conservative behavioral-source revocation.

    Completed segment mean rewards are accumulated without RNG use.  At each
    fixed, non-overlapping window boundary, every currently admitted source is
    compared with the student at most once.  Revocations are irreversible and
    batched into one immutable :class:`AdmissionSnapshot`.

    The normal bounds are a controller heuristic, not a finite-sample or
    sequential-testing confidence guarantee.

    机制概要(adaptive_admission_v1 实验的撤销状态机,2026-07-15 预注册裁决
    FAIL——行为 reward 信号第三次独立否定,保留实现供审计复现与后续换信号族
    时参考):
    - 固定不重叠 stage 窗口(默认 3000 步),窗口末 Welford 统计清零;
    - 每源每窗至多一票:UCB(source) < LCB(student) 记 positive(双方本窗
      segment 数须 >= min_segments,证据不足则 persistence 清零);
    - 连续 persistence(默认 3)窗 positive → 不可逆撤销,同窗多源合并成
      单一 immutable snapshot 原子应用;
    - 纯 CPU、零 RNG:不触碰行为/replay 本身,只发布新 snapshot。
    """

    def __init__(
        self,
        *,
        initial_snapshot: AdmissionSnapshot,
        stage_window_steps: int = 3000,
        confidence_z: float = 1.645,
        min_segments: int = 20,
        persistence: int = 3,
    ) -> None:
        if stage_window_steps <= 0:
            raise ValueError("stage_window_steps must be positive")
        if not math.isfinite(confidence_z) or confidence_z <= 0:
            raise ValueError("confidence_z must be finite and positive")
        if min_segments < 2:
            raise ValueError("min_segments must be at least 2")
        if persistence <= 0:
            raise ValueError("persistence must be positive")
        self.stage_window_steps = int(stage_window_steps)
        self.confidence_z = float(confidence_z)
        self.min_segments = int(min_segments)
        self.persistence = int(persistence)
        self._current_snapshot = initial_snapshot
        self._candidate_names = (*initial_snapshot.source_names, "student")
        self._moments = [_RunningMoments() for _ in self._candidate_names]
        self._persistence_counts = [0 for _ in initial_snapshot.source_names]
        self._last_closed_step = 0

    @property
    def current_snapshot(self) -> AdmissionSnapshot:
        return self._current_snapshot

    @property
    def student_candidate(self) -> int:
        return len(self._current_snapshot.source_names)

    @property
    def persistence_counts(self) -> tuple[int, ...]:
        return tuple(self._persistence_counts)

    def record_segments(
        self,
        candidate_ids: Iterable[int] | torch.Tensor,
        segment_mean_rewards: Iterable[float] | torch.Tensor,
    ) -> None:
        """Record a batch of naturally completed segments for the active window."""

        if isinstance(candidate_ids, torch.Tensor):
            ids = candidate_ids.detach().cpu().view(-1).tolist()
        else:
            ids = list(candidate_ids)
        if isinstance(segment_mean_rewards, torch.Tensor):
            values = segment_mean_rewards.detach().cpu().view(-1).tolist()
        else:
            values = list(segment_mean_rewards)
        if len(ids) != len(values):
            raise ValueError("candidate ids and segment rewards must have equal length")
        for candidate, value in zip(ids, values):
            candidate = int(candidate)
            if not 0 <= candidate < len(self._moments):
                raise ValueError(f"unknown adaptive admission candidate id: {candidate}")
            self._moments[candidate].update(float(value))

    def maybe_close_window(
        self, completed_step: int
    ) -> AdaptiveAdmissionWindowResult | None:
        """Close exactly one due stage window and return its immutable audit."""

        completed_step = int(completed_step)
        expected = self._last_closed_step + self.stage_window_steps
        if completed_step < expected:
            return None
        if completed_step != expected:
            raise ValueError(
                f"adaptive window boundary skipped or repeated: expected {expected}, "
                f"got {completed_step}"
            )

        statistics = tuple(
            moments.snapshot(candidate=name, z=self.confidence_z)
            for name, moments in zip(self._candidate_names, self._moments)
        )
        student = statistics[-1]
        admitted_after = list(self._current_snapshot.admitted)
        positive_votes = [False for _ in admitted_after]
        revoked_indices: list[int] = []
        for index, admitted in enumerate(self._current_snapshot.admitted):
            if not admitted:
                self._persistence_counts[index] = 0
                continue
            source = statistics[index]
            enough = (
                source.count >= self.min_segments
                and student.count >= self.min_segments
                and source.upper_bound is not None
                and student.lower_bound is not None
            )
            positive = bool(
                enough and float(source.upper_bound) < float(student.lower_bound)
            )
            positive_votes[index] = positive
            self._persistence_counts[index] = (
                self._persistence_counts[index] + 1 if positive else 0
            )
            if self._persistence_counts[index] >= self.persistence:
                admitted_after[index] = False
                revoked_indices.append(index)

        window_index = completed_step // self.stage_window_steps
        decision = None
        if revoked_indices:
            decision = AdmissionSnapshot(
                source_names=self._current_snapshot.source_names,
                admitted=tuple(admitted_after),
                source_logits=self._current_snapshot.source_logits,
                student_logit=self._current_snapshot.student_logit,
                decision_id=f"adaptive-window-{window_index}-step-{completed_step}",
                mode="adaptive",
                quarantine_artifact=self._current_snapshot.quarantine_artifact,
                quarantine_digest=self._current_snapshot.quarantine_digest,
            )
            self._current_snapshot = decision

        result = AdaptiveAdmissionWindowResult(
            completed_step=completed_step,
            window_index=window_index,
            statistics=statistics,
            positive_votes=tuple(positive_votes),
            persistence_counts=tuple(self._persistence_counts),
            revoked_sources=tuple(
                self._current_snapshot.source_names[index] for index in revoked_indices
            ),
            snapshot=decision,
        )
        self._moments = [_RunningMoments() for _ in self._candidate_names]
        self._last_closed_step = completed_step
        return result


def desired_admission_source_authority(
    snapshot: AdmissionSnapshot,
    *,
    global_step: int,
    warmup_steps: int,
    warmup_authority: bool,
    post_warmup_authority: bool,
) -> bool:
    """Return phase authority while making exact abstention non-revivable."""

    phase_authority = (
        bool(warmup_authority)
        if int(global_step) < int(warmup_steps)
        else bool(post_warmup_authority)
    )
    return phase_authority and not snapshot.exact_abstain


def _parse_names(value: str | Iterable[str] | None) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return tuple(item.strip() for item in value.split(",") if item.strip())
    return tuple(str(item).strip() for item in value if str(item).strip())


def _load_manifest(path: Path, source_names: tuple[str, ...]) -> dict[str, Any]:
    raw = load_yaml(path)
    if int(raw.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ValueError("unsupported admission manifest schema")
    manifest_sources = tuple(str(name) for name in raw.get("source_names", []))
    if manifest_sources != source_names:
        raise ValueError(
            f"admission manifest source order mismatch: {manifest_sources} != {source_names}"
        )
    quarantine = raw.get("quarantine") or {}
    artifact_value = quarantine.get("artifact")
    digest_value = quarantine.get("sha256")
    if artifact_value is not None:
        artifact = Path(str(artifact_value))
        if not artifact.is_absolute():
            artifact = (path.parent / artifact).resolve()
        if not artifact.is_file():
            raise FileNotFoundError(artifact)
        actual = _sha256(artifact)
        if digest_value is not None and str(digest_value) != actual:
            raise ValueError("quarantine artifact digest mismatch")
        quarantine = {"artifact": str(artifact), "sha256": actual}
    elif digest_value is not None:
        raise ValueError("quarantine sha256 requires an artifact path")
    raw["quarantine"] = quarantine
    return raw


def build_admission_snapshot(
    *,
    mode: str,
    source_names: Iterable[str],
    source_logits: Iterable[float],
    student_logit: float = 0.0,
    admitted_sources: str | Iterable[str] | None = None,
    manifest_path: str | Path | None = None,
) -> AdmissionSnapshot:
    names = tuple(str(name) for name in source_names)
    logits = tuple(float(value) for value in source_logits)
    if len(logits) != len(names):
        raise ValueError("source logits do not match source bank size")
    if mode not in {"all", "none", "static", "manifest", "target_evidence"}:
        raise ValueError(f"unknown admission mode: {mode}")

    decision_id = f"explicit-{mode}"
    quarantine_artifact = None
    quarantine_digest = None
    if mode == "all":
        selected = names
    elif mode in {"none", "target_evidence"}:
        selected = ()
    elif mode == "static":
        selected = _parse_names(admitted_sources)
    else:
        if manifest_path is None:
            raise ValueError("manifest admission mode requires manifest_path")
        manifest = _load_manifest(Path(manifest_path), names)
        selected = _parse_names(manifest.get("admitted_sources"))
        decision_id = str(manifest.get("decision_id") or "manifest")
        student_logit = float(manifest.get("student_logit", student_logit))
        if "source_logits" in manifest:
            logits = tuple(float(value) for value in manifest["source_logits"])
            if len(logits) != len(names):
                raise ValueError("manifest source_logits do not match source bank size")
        quarantine = manifest.get("quarantine") or {}
        quarantine_artifact = quarantine.get("artifact")
        quarantine_digest = quarantine.get("sha256")

    unknown = sorted(set(selected) - set(names))
    if unknown:
        raise ValueError(f"admission decision references unknown sources: {unknown}")
    selected_set = set(selected)
    admitted = tuple(name in selected_set for name in names)
    return AdmissionSnapshot(
        source_names=names,
        admitted=admitted,
        source_logits=logits,
        student_logit=float(student_logit),
        decision_id=decision_id,
        mode=mode,
        quarantine_artifact=quarantine_artifact,
        quarantine_digest=quarantine_digest,
    )


def build_admission_schedule(
    *,
    schedule_path: str | Path,
    source_names: Iterable[str],
    source_logits: Iterable[float],
    student_logit: float = 0.0,
) -> AdmissionSchedule:
    """Load an explicit admission schedule.

    The schedule contains decisions, not scores or a transferability estimator.
    Every optional quarantine artifact is only hashed/bound to the decision; its
    transitions are never loaded into the learner replay by this module.
    """

    path = Path(schedule_path)
    raw = load_yaml(path)
    if int(raw.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ValueError("unsupported admission schedule schema")
    names = tuple(str(name) for name in source_names)
    declared = tuple(str(name) for name in raw.get("source_names", []))
    if declared != names:
        raise ValueError(f"admission schedule source order mismatch: {declared} != {names}")
    defaults = tuple(float(value) for value in source_logits)
    if len(defaults) != len(names):
        raise ValueError("source logits do not match source bank size")
    entries = raw.get("decisions")
    if not isinstance(entries, list) or not entries:
        raise ValueError("admission schedule requires non-empty decisions")

    decisions: list[tuple[int, AdmissionSnapshot]] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError("admission schedule decisions must be mappings")
        step = int(entry.get("step", -1))
        selected = _parse_names(entry.get("admitted_sources"))
        unknown = sorted(set(selected) - set(names))
        if unknown:
            raise ValueError(f"admission decision references unknown sources: {unknown}")
        logits = tuple(float(value) for value in entry.get("source_logits", defaults))
        if len(logits) != len(names):
            raise ValueError("scheduled source_logits do not match source bank size")
        quarantine = entry.get("quarantine") or {}
        artifact_value = quarantine.get("artifact")
        digest_value = quarantine.get("sha256")
        artifact_path = None
        actual_digest = None
        if artifact_value is not None:
            artifact = Path(str(artifact_value))
            if not artifact.is_absolute():
                artifact = (path.parent / artifact).resolve()
            if not artifact.is_file():
                raise FileNotFoundError(artifact)
            actual_digest = _sha256(artifact)
            if digest_value is not None and str(digest_value) != actual_digest:
                raise ValueError("quarantine artifact digest mismatch")
            artifact_path = str(artifact)
        elif digest_value is not None:
            raise ValueError("quarantine sha256 requires an artifact path")
        selected_set = set(selected)
        snapshot = AdmissionSnapshot(
            source_names=names,
            admitted=tuple(name in selected_set for name in names),
            source_logits=logits,
            student_logit=float(entry.get("student_logit", student_logit)),
            decision_id=str(entry.get("decision_id") or f"schedule-{index}-step-{step}"),
            mode="schedule",
            quarantine_artifact=artifact_path,
            quarantine_digest=actual_digest,
        )
        decisions.append((step, snapshot))
    return AdmissionSchedule(tuple(decisions))
