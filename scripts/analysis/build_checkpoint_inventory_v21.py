#!/usr/bin/env python3
"""Checkpoint inventory v2.1 —— provenance / identity 修正（P2.1）。

判据冻结于 docs/experiments/checkpoint_inventory_v21_prereg_20260807.md。

与 v2 的差别（每条都对应 v2 的一个实测缺陷）：

  文件名解析   支持含 `_` 的 env；解析失败 → EXCLUDED_UNPARSEABLE_NAME（v2 是 fail-open）
  endpoint     run_stop_step if 显式设置 else total_timesteps（禁 min()）
  canonical    只保留 <= effective_endpoint，超出记 out_of_scope
  身份         三层：run_family / execution_instance / learner_replication
  alias        正式路径与 archive A 是同一 execution；SHA 不符 → FORMAL_ALIAS_INTEGRITY_FAILURE
  digest       ptf_cfg / treatment / pairing_invariant 三分
  full scan    独立扫 filesystem roots，v1 manifest 只作差异对照

用法::

    python scripts/analysis/build_checkpoint_inventory_v21.py                 # sentinel
    python scripts/analysis/build_checkpoint_inventory_v21.py --full --approved-by "<who>"
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from fasttd3_ptf.evaluation import inventory_identity as ident  # noqa: E402

REGISTRY_PATH = REPO / "docs/data/run_cards/run_card_registry_v1.json"
V1_MANIFEST = REPO / "docs/data/checkpoint_inventory_v1/manifest.json"
OUT_DIR = REPO / "docs/data/checkpoint_inventory_v21"

#: full scan 的**冻结** filesystem roots（预注册 §7）。
#: v1 manifest 只作差异对照，不作数据来源——否则 v1 漏掉的 v2 永远发现不了。
SCAN_ROOTS = ("models", "checkpoints")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_registry() -> list:
    if not REGISTRY_PATH.exists():
        raise SystemExit(f"INCOMPLETE: run card registry 不存在：{REGISTRY_PATH}")
    return json.loads(REGISTRY_PATH.read_text(encoding="utf-8")).get("entries") or []


def discover_filesystem() -> list:
    """独立发现数据宇宙（预注册 §7）。不读 v1 manifest。"""
    found = []
    for root in SCAN_ROOTS:
        base = REPO / root
        if not base.exists():
            continue
        found.extend(str(p.relative_to(REPO)) for p in base.rglob("*.pt"))
    return sorted(found)


def read_checkpoint(path: Path) -> dict:
    """只读元数据，不构建模型。"""
    import torch

    try:
        state = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:  # noqa: BLE001
        return {"readable": False, "error": f"{type(exc).__name__}: {exc}"}
    args = state.get("args") or {}
    ptf_cfg = state.get("ptf_cfg") or {}
    out = {
        "readable": True,
        "args": args,
        "ptf_cfg": ptf_cfg,
        "source_names": state.get("source_names"),
        "agent_type": state.get("agent_type"),
        "inner_env_name": args.get("env_name"),
        "inner_seed": args.get("seed"),
        "inner_global_step": state.get("global_step"),
        "exp_name": args.get("exp_name"),
    }
    del state
    return out


def classify_mechanism(ptf_cfg, source_names) -> str:
    if not ptf_cfg:
        return "NO_PTF"
    names = list(source_names) if source_names else []
    return "PTF_NULL_BANK" if (not names or names == ["null"]) else "PTF_WITH_SOURCES"


def check_identity_conflict(fname: dict, inner: dict) -> list:
    """文件名解析值与 checkpoint 内部值逐项比对。

    调用方保证只在 ``fname_parsed=True`` 时调用——
    v2 在解析失败时返回空冲突列表，等于放行，那是 fail-open。
    """
    conflicts = []
    pairs = [("env_name", fname["fname_env"], inner.get("inner_env_name")),
             ("learner_seed", fname["fname_seed"], inner.get("inner_seed"))]
    if not fname["fname_is_final"]:
        pairs.append(("global_step", fname["fname_step"], inner.get("inner_global_step")))
    for field, from_name, from_ckpt in pairs:
        if from_name is None or from_ckpt is None:
            continue
        if str(from_name) != str(from_ckpt):
            conflicts.append({"field": field, "from_filename": from_name,
                              "from_checkpoint": from_ckpt})
    return conflicts


def process(rel_path: str, registry: list, reason: str | None = None) -> dict:
    p = rel_path if Path(rel_path).is_absolute() else str(REPO / rel_path)
    path = Path(p)
    row = {"path": rel_path, "sentinel_reason": reason}

    if not path.exists():
        row.update({"eligibility": ident.EXCLUDED_UNREADABLE, "error": "文件不存在"})
        return row

    fname = ident.parse_filename(path.name)
    row.update(fname)

    # 解析失败即排除。**无法核对 != 核对通过**（v2 在这里 fail-open）。
    if not fname["fname_parsed"]:
        row["eligibility"] = ident.EXCLUDED_UNPARSEABLE_NAME
        row["checkpoint_id"] = sha256_file(path)
        return row

    row["checkpoint_id"] = sha256_file(path)
    inner = read_checkpoint(path)
    if not inner["readable"]:
        row.update({"eligibility": ident.EXCLUDED_UNREADABLE, "error": inner["error"]})
        return row

    args, ptf_cfg = inner["args"], inner["ptf_cfg"]
    row.update({k: inner[k] for k in
                ("inner_env_name", "inner_seed", "inner_global_step", "exp_name",
                 "source_names", "agent_type")})

    conflicts = check_identity_conflict(fname, inner)
    row["identity_conflicts"] = conflicts
    if conflicts:
        row["eligibility"] = ident.EXCLUDED_IDENTITY_MISMATCH
        return row

    card = ident.match_run_card(registry, inner["exp_name"], rel_path)
    row["run_card"] = card
    row["experiment_role"] = card["experiment_role"] if card else ident.UNKNOWN_ROLE
    row["execution_role"] = card["execution_role"] if card else ident.UNKNOWN_EXECUTION_ROLE
    row["match_group"] = card["match_group"] if card else None
    row["alias_of_formal_path"] = bool(card and card["alias_of_formal_path"])
    row["counts_as_new_learner_replication"] = (
        card["counts_as_new_learner_replication"] if card else True)

    row["mechanism_family"] = classify_mechanism(ptf_cfg, inner["source_names"])
    row.update(ident.compute_digests(args, ptf_cfg, inner["source_names"]))

    ep = ident.effective_endpoint(args, ptf_cfg)
    row["endpoint"] = ep
    row["canonical"] = ident.canonical_steps(
        args, ptf_cfg, (card or {}).get("registry_canonical_points") or ())

    row.update(ident.build_identity(
        inner["inner_env_name"], inner["exp_name"], inner["inner_seed"],
        row["execution_role"]))

    row["eligibility"] = (ident.INVALID_ENDPOINT_CONFIG
                          if ep["source"] == ident.INVALID_ENDPOINT_CONFIG
                          else ident.ELIGIBLE)
    return row


def resolve_aliases(rows: list) -> tuple[list, list, dict]:
    """alias 去重与 execution 冲突分流（预注册 §2.2）。

    返回 ``(ambiguous, integrity_failures, alias_notes)``。
    """
    ambiguous, integrity, notes = [], [], {}

    # 同一 execution_instance + 同 step 的多份文件
    by_exec = defaultdict(list)
    for r in rows:
        if r.get("eligibility") != ident.ELIGIBLE:
            continue
        key = (r.get("execution_instance_id"), r.get("inner_global_step"))
        if key[0] is not None and key[1] is not None:
            by_exec[key].append(r)

    for (exec_id, step), items in by_exec.items():
        shas = {i["checkpoint_id"] for i in items}
        if len(items) == 1:
            continue
        if len(shas) == 1:
            keep = min(items, key=lambda r: r["path"])
            for r in items:
                if r is not keep:
                    r["alias_status"] = ident.EXACT_ALIAS
                    r["alias_of"] = keep["path"]
                    r["is_canonical_file"] = False
            notes[f"{exec_id}@{step}"] = f"{len(items)} 个文件 SHA 相同，已 alias 去重"
        else:
            # 同一 execution 内部不该出现不同 SHA —— 若其中含 formal alias 对，
            # 那是协议被破坏；否则是无法区分的执行歧义。
            has_alias_pair = any(r.get("alias_of_formal_path") for r in items)
            if has_alias_pair:
                integrity.append({
                    "execution_instance_id": exec_id, "global_step": step,
                    "n_distinct_sha": len(shas),
                    "paths": sorted(r["path"] for r in items),
                    "reason": "正式路径与 archive A 应逐字节相同（协议要求正式路径恢复 A）"})
                for r in items:
                    r["eligibility"] = ident.FORMAL_ALIAS_INTEGRITY_FAILURE
            else:
                ambiguous.append({
                    "execution_instance_id": exec_id, "global_step": step,
                    "n_distinct_sha": len(shas),
                    "paths": sorted(r["path"] for r in items),
                    "reason": "同一 execution 内出现不同 SHA，且无冻结证据可区分"})
                for r in items:
                    r["eligibility"] = ident.AMBIGUOUS_EXECUTION
    return ambiguous, integrity, notes


def compute_run_completion(rows: list) -> dict:
    """completion 在 run（execution）层判断。"""
    by_exec = defaultdict(list)
    for r in rows:
        if r.get("eligibility") == ident.ELIGIBLE and r.get("execution_instance_id"):
            by_exec[r["execution_instance_id"]].append(r)

    out = {}
    for exec_id, group in by_exec.items():
        steps = [r["inner_global_step"] for r in group
                 if r.get("inner_global_step") is not None]
        endpoints = {(r.get("endpoint") or {}).get("endpoint") for r in group}
        endpoints.discard(None)
        observed_end = max(steps) if steps else None
        for r in group:
            r["is_run_endpoint"] = (r.get("inner_global_step") == observed_end)
        if len(endpoints) != 1:
            out[exec_id] = {"completion_status": ident.UNKNOWN_COMPLETION,
                            "observed_end": observed_end,
                            "reason": f"endpoint 不唯一或缺失：{sorted(endpoints)}"}
            continue
        endpoint = endpoints.pop()
        out[exec_id] = {
            "completion_status": ident.completion_status(observed_end, endpoint),
            "observed_end": observed_end,
            "effective_endpoint": endpoint,
            "endpoint_source": (group[0].get("endpoint") or {}).get("source"),
            "counts_as_new_learner_replication":
                group[0].get("counts_as_new_learner_replication", True),
        }
    return out


# ══════════════════════════════════════════════════════════════════════
# Sentinel 选取（预注册 §9，规则先于运行冻结）
# ══════════════════════════════════════════════════════════════════════

def select_sentinels(universe: list) -> tuple[list, list]:
    picked, seen, unavailable = [], set(), []

    def take(pred, n, tag):
        got = 0
        for rel in universe:
            if got >= n:
                break
            if rel in seen or not pred(rel):
                continue
            seen.add(rel)
            picked.append({"path": rel, "reason": tag})
            got += 1
        if got < n:
            unavailable.append({"rule": tag, "wanted": n, "got": got,
                                "status": "SENTINEL_UNAVAILABLE"})

    name = lambda r: Path(r).name                                     # noqa: E731
    # #1/#2 P0 正式路径 + archive A + archive B（同 step）
    take(lambda r: "p0_dup_archive" not in r and "p0_crawl_abstain" in r
         and "_13000.pt" in r, 1, "p0_formal_path")
    take(lambda r: "p0_dup_archive/crawl_A" in r and "_13000.pt" in r, 1, "p0_archive_A")
    take(lambda r: "p0_dup_archive/crawl_B" in r and "_13000.pt" in r, 1, "p0_archive_B")
    # #4 underscore env
    take(lambda r: "balance_hard" in name(r) or "bookshelf_simple" in name(r),
         2, "underscore_env")
    # #9 Racing 角色（实际前缀是 rck / rad，不是 "racing"——v2 用错关键词导致 0 命中）
    take(lambda r: "__rck_" in name(r), 2, "racing_min_horizon")
    take(lambda r: "__rad_" in name(r), 2, "racing_admission")
    # Slide 三臂
    for arm in ("prefix", "cont", "exit"):
        take(lambda r, a=arm: f"__shev1_{a}_s" in name(r), 1, f"slide_{arm}")
    # 覆盖 scratch / null bank / 各 target
    take(lambda r: "scr" in name(r), 2, "scr_named")
    for env in ("h1hand-crawl-v0", "h1hand-slide-v0", "h1hand-truck-v0"):
        take(lambda r, e=env: name(r).startswith(e + "__"), 1, f"target:{env}")
    take(lambda r: r.endswith("_final.pt"), 2, "final_file")
    return picked, unavailable


def make_injected(tmpdir: Path, donor: Path) -> list:
    """构造注入件（预注册 §9 的 #3 / #5 / #10）。不污染 models/。"""
    out = []
    m = ident.FNAME_RE.match(donor.name)
    if not m:
        return out
    env, run, seed, step = (m.group("env"), m.group("run"),
                            m.group("seed"), m.group("step"))
    # #10 身份冲突：改错 seed / 改错 step
    bad_seed = tmpdir / f"{env}__{run}__{int(seed) + 77}_{step}.pt"
    shutil.copy2(donor, bad_seed)
    out.append({"path": str(bad_seed), "expect": ident.EXCLUDED_IDENTITY_MISMATCH,
                "reason": "injected_seed_conflict"})
    if step != "final":
        bad_step = tmpdir / f"{env}__{run}__{seed}_{int(step) + 12345}.pt"
        shutil.copy2(donor, bad_step)
        out.append({"path": str(bad_step), "expect": ident.EXCLUDED_IDENTITY_MISMATCH,
                    "reason": "injected_step_conflict"})
    # #5 不可解析文件名
    bad_name = tmpdir / "not_a_valid_checkpoint_name.pt"
    shutil.copy2(donor, bad_name)
    out.append({"path": str(bad_name), "expect": ident.EXCLUDED_UNPARSEABLE_NAME,
                "reason": "injected_unparseable_name"})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--approved-by", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.full and not args.approved_by:
        raise SystemExit("全量 deep scan 需要 --approved-by（预注册 §8）")

    registry = load_registry()
    universe = discover_filesystem()
    print(f"独立发现 filesystem universe：{len(universe)} 个 .pt（roots={SCAN_ROOTS}）",
          flush=True)

    if args.full:
        targets = [{"path": r, "reason": "FULL_SCAN"} for r in universe]
        injected, tmpdir = [], None
        unavailable = []
    else:
        targets, unavailable = select_sentinels(universe)
        tmpdir = Path(tempfile.mkdtemp(prefix="inv21_injected_"))
        donor = next((REPO / t["path"] for t in targets
                      if (REPO / t["path"]).exists()
                      and not t["path"].endswith("_final.pt")), None)
        injected = make_injected(tmpdir, donor) if donor else []

    print(f"处理 {len(targets)} 个目标 + {len(injected)} 个注入件", flush=True)
    rows = [process(t["path"], registry, t.get("reason")) for t in targets]
    injected_rows = [process(t["path"], registry, t["reason"]) for t in injected]

    injection_failures = [
        {"path": r["path"], "expected": t["expect"], "actual": r.get("eligibility")}
        for t, r in zip(injected, injected_rows) if r.get("eligibility") != t["expect"]]

    ambiguous, integrity, alias_notes = resolve_aliases(rows)
    run_completion = compute_run_completion(rows)

    n_unparseable = sum(1 for r in rows if r.get("eligibility") == ident.EXCLUDED_UNPARSEABLE_NAME)
    n_invalid_ep = sum(1 for r in rows if r.get("eligibility") == ident.INVALID_ENDPOINT_CONFIG)
    n_role = sum(1 for r in rows if r.get("experiment_role") not in (None, ident.UNKNOWN_ROLE))

    failures = []
    if injection_failures:
        failures.append(f"{len(injection_failures)} 个注入件未按预期分类")
    if integrity:
        failures.append(f"{len(integrity)} 组 FORMAL_ALIAS_INTEGRITY_FAILURE")
    if ambiguous:
        failures.append(f"{len(ambiguous)} 组 AMBIGUOUS_EXECUTION")
    if n_unparseable:
        failures.append(f"{n_unparseable} 个 EXCLUDED_UNPARSEABLE_NAME（真实数据中）")
    if n_invalid_ep:
        failures.append(f"{n_invalid_ep} 个 INVALID_ENDPOINT_CONFIG")
    if not rows:
        failures.append("INCOMPLETE：无任何文件被处理")

    # v1 ↔ v2 universe diff（预注册 §7）
    universe_diff = {}
    if V1_MANIFEST.exists():
        v1_paths = {r["path"] for r in
                    json.loads(V1_MANIFEST.read_text(encoding="utf-8"))["rows"]}
        v2_paths = set(universe)
        universe_diff = {
            "v1_total": len(v1_paths), "v2_total": len(v2_paths),
            "common": len(v1_paths & v2_paths),
            "v1_only_count": len(v1_paths - v2_paths),
            "v2_only_count": len(v2_paths - v1_paths),
            "v1_only_sample": sorted(v1_paths - v2_paths)[:20],
            "v2_only_sample": sorted(v2_paths - v1_paths)[:20],
        }

    payload = {
        "prereg": "docs/experiments/checkpoint_inventory_v21_prereg_20260807.md",
        "registry": "docs/data/run_cards/run_card_registry_v1.json",
        "mode": "full" if args.full else "sentinel",
        "verdict": ("SENTINEL_PASS" if not failures else "SENTINEL_FAILED")
                   if not args.full else
                   ("FULL_SCAN_COMPLETE" if not failures else "FULL_SCAN_FAILED"),
        "failures": failures,
        "scan_roots": list(SCAN_ROOTS),
        "universe_size": len(universe),
        "universe_diff_vs_v1": universe_diff,
        "totals": {
            "n_processed": len(rows),
            "n_eligible": sum(1 for r in rows if r.get("eligibility") == ident.ELIGIBLE),
            "n_unparseable": n_unparseable,
            "n_identity_mismatch": sum(
                1 for r in rows if r.get("eligibility") == ident.EXCLUDED_IDENTITY_MISMATCH),
            "n_role_resolved": n_role,
            "n_injected": len(injected_rows),
        },
        "sentinel_unavailable": unavailable,
        "injection_checks": injection_failures or "全部注入件均按预期分类",
        "ambiguous_executions": ambiguous,
        "formal_alias_integrity_failures": integrity,
        "alias_notes": alias_notes,
        "run_completion": run_completion,
        "rows": rows,
        "injected_rows": injected_rows,
    }

    out_path = Path(args.out) if args.out else (
        OUT_DIR / ("full.json" if args.full else "sentinel.json"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, out_path)
    if tmpdir and tmpdir.exists():
        shutil.rmtree(tmpdir, ignore_errors=True)

    t = payload["totals"]
    print(f"\nVERDICT: {payload['verdict']}")
    print(f"  处理 {t['n_processed']} | ELIGIBLE {t['n_eligible']} | "
          f"角色已解析 {t['n_role_resolved']} | 不可解析 {t['n_unparseable']}")
    print(f"  注入件 {t['n_injected']} 个：{payload['injection_checks']}")
    if universe_diff:
        print(f"  universe diff vs v1：common={universe_diff['common']} "
              f"v1_only={universe_diff['v1_only_count']} "
              f"v2_only={universe_diff['v2_only_count']}")
    if unavailable:
        print(f"  SENTINEL_UNAVAILABLE: {unavailable}")
    for f in failures:
        print(f"  ! {f}")
    print(f"wrote {out_path.relative_to(REPO)}")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
