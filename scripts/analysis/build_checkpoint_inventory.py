#!/usr/bin/env python3
"""Checkpoint inventory 第一遍：只读文件名与文件系统元数据。

判据冻结于 docs/experiments/checkpoint_inventory_v1_prereg_20260806.md（提交 76fa2e2）。

**本脚本产出的 manifest 不得用于任何统计或场地判断**（预注册 §5）——
它只回答"有哪些文件、哪些值得深扫"。所有需要 method_family 的判断必须等第二遍。

防 winner's curse 的关键：canonical 选取只依赖 global_step 这一个与性能无关的量，
本脚本**不 import torch，也不读取任何 return / eval 数据**——这一点可由代码审查
直接验证（预注册 §6 的 8.1）。
"""

from __future__ import annotations

import json
import os
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_DIR = os.path.join(REPO, "docs/data/checkpoint_inventory_v1")

# 预注册 §1：扫描范围
SCAN_ROOTS = ("models", "artifacts")
EXCLUDE_PATH_PARTS = ("p0_anchors",)
ANCHOR_BUNDLE_NAMES = {"learner.pt", "replay.pt", "rng.pt"}

# 预注册 §3：canonical 只依赖固定步数点，与性能无关
CANONICAL_STEPS = {20000, 50000, 100000}
CANONICAL_FINAL = "FINAL"

# 预注册 §4：smoke 关键词（大小写不敏感，只匹配 run_name 段）
SMOKE_KEYWORDS = ("smoke", "toy", "debug", "dbg", "test", "trial")

# models/{env}__{run_name}__{seed}_{step}.pt
NAME_RE = re.compile(r"^(?P<env>[^_]+(?:_[^_]+)*?)__(?P<run>.+)__(?P<seed>\d+)_(?P<step>\w+)\.pt$")

DEEP_SCAN_FIELDS = (
    "method_family", "training_commit", "source_bank_digest",
    "bootstrap_budget", "exit_policy", "completion_status",
)
PENDING = "UNKNOWN_NEEDS_DEEP_SCAN"


def _iter_checkpoints():
    for root in SCAN_ROOTS:
        base = os.path.join(REPO, root)
        if not os.path.isdir(base):
            continue
        for dirpath, _dirnames, filenames in os.walk(base):
            rel_dir = os.path.relpath(dirpath, REPO)
            for name in filenames:
                if not name.endswith(".pt"):
                    continue
                yield os.path.join(rel_dir, name), name, dirpath


def _parse_step(raw: str):
    """'20000' → 20000；'final' → 'FINAL'；其余 → None。"""
    if raw.isdigit():
        return int(raw)
    if raw.lower() == "final":
        return CANONICAL_FINAL
    return None


def _classify(rel_path: str, name: str) -> dict:
    """只用文件名与路径做判定，不读文件内容。"""
    row = {
        "path": rel_path,
        "checkpoint_sha256": PENDING,
        "env_name": None,
        "run_name": None,
        "run_group_id": None,
        "learner_seed": None,
        "global_step": None,
        "eligibility": None,
        "exclusion_reason": None,
        "is_canonical": False,
    }
    for f in DEEP_SCAN_FIELDS:
        row[f] = PENDING

    # ── anchor bundle：非评估对象 ──────────────────────────────────
    if name in ANCHOR_BUNDLE_NAMES or any(p in rel_path for p in EXCLUDE_PATH_PARTS):
        row["eligibility"] = "EXCLUDED"
        row["exclusion_reason"] = "ANCHOR_BUNDLE"
        return row

    m = NAME_RE.match(name)
    if not m:
        row["eligibility"] = "EXCLUDED"
        row["exclusion_reason"] = "UNPARSEABLE_NAME"
        return row

    env, run, seed, step_raw = m.group("env"), m.group("run"), m.group("seed"), m.group("step")
    step = _parse_step(step_raw)
    row.update({
        "env_name": env,
        "run_name": run,
        "run_group_id": run,          # 预注册 §2：同一 run 的所有 step 共享
        "learner_seed": int(seed),
        "global_step": step,
    })

    if step is None:
        row["eligibility"] = "EXCLUDED"
        row["exclusion_reason"] = "UNPARSEABLE_NAME"
        return row

    # ── smoke / debug：只匹配 run_name 段，避免误伤 env 名或路径 ──
    low = run.lower()
    if any(k in low for k in SMOKE_KEYWORDS):
        row["eligibility"] = "EXCLUDED"
        row["exclusion_reason"] = "SMOKE_OR_DEBUG"
        return row

    # ── canonical：**只看 global_step**，不接触任何性能数据 ────────
    row["is_canonical"] = (step == CANONICAL_FINAL) or (step in CANONICAL_STEPS)
    row["eligibility"] = "PENDING_DEEP_SCAN"
    return row


def build() -> dict:
    rows = []
    for rel_path, name, dirpath in _iter_checkpoints():
        row = _classify(rel_path, name)
        try:
            st = os.stat(os.path.join(dirpath, name))
            row["file_size_bytes"] = st.st_size
            row["file_mtime_utc"] = datetime.fromtimestamp(
                st.st_mtime, tz=timezone.utc).isoformat()
        except OSError:
            row["file_size_bytes"] = None
            row["file_mtime_utc"] = None
        rows.append(row)

    # ── 按 run_group_id 去重的统计（预注册 §7 要求可直接核对）────────
    # 同一 run 的多个 step **不是**独立样本，故 n_seeds 只数不同 learner_seed。
    by_env: dict[str, dict] = defaultdict(
        lambda: {"run_groups": set(), "seeds": set(), "canonical_files": 0, "excluded": 0}
    )
    for r in rows:
        env = r["env_name"]
        if env is None:
            continue
        entry = by_env[env]
        if r["eligibility"] == "EXCLUDED":
            entry["excluded"] += 1
            continue
        entry["run_groups"].add(r["run_group_id"])
        entry["seeds"].add(r["learner_seed"])
        if r["is_canonical"]:
            entry["canonical_files"] += 1

    per_env = {
        env: {
            "n_run_groups": len(v["run_groups"]),
            "n_distinct_seeds": len(v["seeds"]),
            "n_canonical_files": v["canonical_files"],
            "n_excluded": v["excluded"],
        }
        for env, v in sorted(by_env.items())
    }

    excl = defaultdict(int)
    for r in rows:
        if r["eligibility"] == "EXCLUDED":
            excl[r["exclusion_reason"]] += 1

    n_pending = sum(1 for r in rows if r["eligibility"] == "PENDING_DEEP_SCAN")
    n_canonical = sum(1 for r in rows if r["is_canonical"])

    return {
        "prereg": "docs/experiments/checkpoint_inventory_v1_prereg_20260806.md",
        "prereg_commit": "76fa2e2",
        "pass": 1,
        "warning": (
            "第一遍 manifest：深度字段均为 UNKNOWN_NEEDS_DEEP_SCAN。"
            "按预注册 §5，本产物**不得用于任何统计或场地判断**。"
        ),
        "canonical_steps": sorted(CANONICAL_STEPS) + [CANONICAL_FINAL],
        "totals": {
            "n_files": len(rows),
            "n_excluded": sum(excl.values()),
            "n_pending_deep_scan": n_pending,
            "n_canonical": n_canonical,
        },
        "exclusion_breakdown": dict(sorted(excl.items())),
        "per_env": per_env,
        "rows": rows,
    }


def main() -> int:
    res = build()
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, "manifest.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)

    t = res["totals"]
    print(f"扫描 {t['n_files']} 个 .pt")
    print(f"  排除          {t['n_excluded']}")
    print(f"  待深扫        {t['n_pending_deep_scan']}")
    print(f"  canonical     {t['n_canonical']}")
    print("\n排除原因：")
    for k, v in res["exclusion_breakdown"].items():
        print(f"  {k:24s} {v}")
    print(f"\n{'env':32s} {'runs':>5s} {'seeds':>6s} {'canon':>6s} {'excl':>5s}")
    print("-" * 60)
    for env, v in res["per_env"].items():
        print(f"{env:32s} {v['n_run_groups']:5d} {v['n_distinct_seeds']:6d} "
              f"{v['n_canonical_files']:6d} {v['n_excluded']:5d}")
    print(f"\nwrote {os.path.relpath(out, REPO)}")

    if t["n_pending_deep_scan"] > 0:
        print(
            f"\nINCOMPLETE: {t['n_pending_deep_scan']} 条待第二遍 deep scan；"
            f"manifest 尚不完整，不得用于统计（预注册 §5）",
            file=sys.stderr,
        )
        return 2
    if t["n_files"] - t["n_excluded"] == 0:
        print("\nNO_ELIGIBLE_CHECKPOINTS", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
