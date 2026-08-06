"""Quarantine bank 结构与完整性校验(admission lifecycle 审计链使用)。

历史说明:本模块原为 SHU(stage-conditioned handoff utility)gate 的完整分析
工具箱;SHU 路线 2026-07-12 预注册失败后,其 gate 配置校验与 paired-effect
分析函数已随该路线移除(原件见 git 快照 a5cec9d)。保留的
validate_quarantine_bank 服务于 admission 线的 quarantine probe 数据审计
(analyze_admission_core_v1.py 消费)。
"""

from __future__ import annotations

from typing import Any

import torch


QUARANTINE_SCHEMA_VERSION = 1


def validate_quarantine_bank(bank: dict[str, Any]) -> None:
    if int(bank.get("schema_version", -1)) != QUARANTINE_SCHEMA_VERSION:
        raise ValueError("unsupported SHU quarantine schema")
    metadata = bank.get("metadata") or {}
    if not bool(metadata.get("quarantine_only", False)):
        raise ValueError("artifact is not declared quarantine-only")
    if int(metadata.get("learner_updates", -1)) != 0:
        raise ValueError("quarantine artifact reports learner updates")
    if int(metadata.get("main_replay_writes", -1)) != 0:
        raise ValueError("quarantine artifact reports main replay writes")
    anchors = int(metadata["valid_anchors"])
    total = int(metadata["source_horizon"]) + int(metadata["followup_horizon"])
    if anchors <= 0 or total <= 1:
        raise ValueError("invalid quarantine dimensions")
    provenance_groups = metadata.get("provenance_groups")
    if not isinstance(provenance_groups, list) or not provenance_groups:
        raise ValueError("metadata.provenance_groups must be a non-empty list")
    group_count = len(provenance_groups)
    paths = bank.get("paths") or {}
    required_paths = {"student", "student_duplicate", "student_null"}
    if not required_paths.issubset(paths):
        raise ValueError(f"missing baseline paths: {sorted(required_paths - set(paths))}")
    sources = bank.get("sources") or {}
    if not sources:
        raise ValueError("quarantine bank has no source paths")
    tensor_fields = (
        "observations",
        "actions",
        "rewards",
        "dones",
        "truncations",
        "active",
        "next_observations",
    )
    provenance_shapes = {
        "behavior_source": (anchors, total),
        "source_by_group": (anchors, total, group_count),
        "executed_group_mask": (anchors, total, group_count),
        "segment_id": (anchors, total),
        "segment_step": (anchors, total),
        "anchor_id": (anchors, total),
        "learner_step": (anchors, total),
    }
    for name, path in {**paths, **sources}.items():
        for field in tensor_fields:
            if field not in path:
                raise ValueError(f"{name}: missing {field}")
            if tuple(path[field].shape[:2]) != (anchors, total):
                raise ValueError(f"{name}/{field}: invalid shape {tuple(path[field].shape)}")
        provenance = path.get("provenance")
        if not isinstance(provenance, dict):
            raise ValueError(f"{name}: missing provenance")
        for field, expected in provenance_shapes.items():
            if field not in provenance or tuple(provenance[field].shape) != expected:
                shape = None if field not in provenance else tuple(provenance[field].shape)
                raise ValueError(f"{name}/provenance/{field}: invalid shape {shape}")
    left = paths["student"]
    duplicate = paths["student_duplicate"]
    for field in tensor_fields:
        if not torch.equal(left[field], duplicate[field]):
            raise ValueError(f"student duplicate mismatch: {field}")
    for field in provenance_shapes:
        if not torch.equal(left["provenance"][field], duplicate["provenance"][field]):
            raise ValueError(f"student duplicate provenance mismatch: {field}")
    left_diagnostics = left.get("diagnostics") or {}
    duplicate_diagnostics = duplicate.get("diagnostics") or {}
    if set(left_diagnostics) != set(duplicate_diagnostics):
        raise ValueError("student duplicate diagnostic schema mismatch")
    for field in left_diagnostics:
        if not torch.equal(left_diagnostics[field], duplicate_diagnostics[field]):
            raise ValueError(f"student duplicate diagnostic mismatch: {field}")
    left_task = left.get("task_diagnostics") or {}
    duplicate_task = duplicate.get("task_diagnostics") or {}
    if set(left_task) != set(duplicate_task):
        raise ValueError("student duplicate task-diagnostic schema mismatch")
    for field in left_task:
        if not torch.allclose(
            left_task[field], duplicate_task[field], rtol=0.0, atol=0.0, equal_nan=True
        ):
            raise ValueError(f"student duplicate task-diagnostic mismatch: {field}")

