#!/usr/bin/env python3
"""校验 run_card_registry_v1.json 的每条证据（P2.1 预注册 §6 / §10.2）。

**为什么需要它**：registry 由我编写，存在"为了让 sentinel 通过而写 registry"
的风险。约束是每条 entry 必须指向具体的文件 + 行号，且：

  1. `evidence_excerpt` 必须**真的出现在**该文件的该行——行号写错会被抓出；
  2. `source_commit` 必须**早于**预注册 commit——事后补造的证据不算；
  3. 必需字段齐全，`execution_role` 只能取冻结的三个值之一。

任一不满足即非零退出。这样 registry 就不是"我说了算"，而是可机械复核的。

用法::

    python scripts/analysis/validate_run_card_registry.py
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
REGISTRY = REPO / "docs/data/run_cards/run_card_registry_v1.json"

#: 预注册 commit。registry 引用的证据必须早于它（§10.2 的防混淆约束）。
PREREG_COMMIT = "af3cacb"

REQUIRED_FIELDS = (
    "id", "match", "experiment_role", "execution_role", "match_group",
    "evidence_path", "evidence_line", "evidence_excerpt",
    "source_commit", "mapping_rule",
)

VALID_EXECUTION_ROLES = frozenset({
    "FORMAL", "REPEATABILITY_DUPLICATE", "UNKNOWN_EXECUTION_ROLE"})


def commit_time(rev: str) -> int | None:
    r = subprocess.run(["git", "-C", str(REPO), "show", "-s", "--format=%ct", rev],
                       capture_output=True, text=True)
    if r.returncode != 0:
        return None
    try:
        return int(r.stdout.strip().splitlines()[0])
    except (ValueError, IndexError):
        return None


def main() -> int:
    if not REGISTRY.exists():
        print(f"INCOMPLETE: registry 不存在：{REGISTRY}")
        return 1
    data = json.loads(REGISTRY.read_text(encoding="utf-8"))
    entries = data.get("entries") or []
    if not entries:
        print("INCOMPLETE: registry 为空")
        return 1

    prereg_ts = commit_time(PREREG_COMMIT)
    if prereg_ts is None:
        print(f"INCOMPLETE: 无法解析预注册 commit {PREREG_COMMIT}")
        return 1

    failures, ids = [], set()
    for e in entries:
        eid = e.get("id", "<无 id>")

        missing = [f for f in REQUIRED_FIELDS if e.get(f) in (None, "")]
        if missing:
            failures.append(f"[{eid}] 缺必需字段 {missing}")
            continue

        if eid in ids:
            failures.append(f"[{eid}] id 重复")
        ids.add(eid)

        if e["execution_role"] not in VALID_EXECUTION_ROLES:
            failures.append(
                f"[{eid}] execution_role={e['execution_role']!r} 不在冻结取值内 "
                f"{sorted(VALID_EXECUTION_ROLES)}")

        # ① 证据行必须真的包含声明的片段
        path = REPO / e["evidence_path"]
        if not path.exists():
            failures.append(f"[{eid}] evidence_path 不存在：{e['evidence_path']}")
            continue
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        ln = int(e["evidence_line"])
        if not (1 <= ln <= len(lines)):
            failures.append(f"[{eid}] evidence_line {ln} 超出 {e['evidence_path']} "
                            f"的行数 {len(lines)}")
            continue
        actual = lines[ln - 1]
        if e["evidence_excerpt"] not in actual:
            failures.append(
                f"[{eid}] {e['evidence_path']}:{ln} 的实际内容不含声明的片段\n"
                f"        声明: {e['evidence_excerpt']!r}\n"
                f"        实际: {actual.strip()!r}")

        # ② 证据 commit 必须早于预注册
        ts = commit_time(e["source_commit"])
        if ts is None:
            failures.append(f"[{eid}] source_commit {e['source_commit']} 无法解析")
        elif ts >= prereg_ts:
            failures.append(
                f"[{eid}] source_commit {e['source_commit']} 不早于预注册 "
                f"{PREREG_COMMIT} —— 事后补造的证据不算")

        # ③ 正则必须可编译
        rx = (e.get("match") or {}).get("exp_name_regex")
        if rx:
            try:
                re.compile(rx)
            except re.error as exc:
                failures.append(f"[{eid}] exp_name_regex 无法编译：{exc}")

    print(f"run card registry 校验：{len(entries)} 条 entry")
    if failures:
        print(f"\n{len(failures)} 处失败：\n")
        for f in failures:
            print(f"  ! {f}")
        return 1
    print("  全部通过：证据行内容逐条比对一致，source_commit 均早于预注册")
    for e in entries:
        print(f"    {e['id']:26s} {e['experiment_role']:20s} "
              f"{e['execution_role']:24s} {e['evidence_path']}:{e['evidence_line']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
