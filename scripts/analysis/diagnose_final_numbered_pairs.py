#!/usr/bin/env python3
"""诊断 final-vs-numbered pair 的语义关系（P2.2 protocol §2.1，**先于全量执行**）。

protocol 要求：全量执行前，先对 ≥20 对跨 slide / racing / stair / truck /
cabinet / P0 的真实 pair 做 field 级与 digest 级诊断，产出三态的实际分布。

**选取规则先于运行冻结**（见 `FAMILY_PATTERNS` 与 `PER_FAMILY`）：
按实验族分层抽样，每族取前 N 对（按路径字典序，不挑数据）。
族命中不足时如实记 `FAMILY_UNAVAILABLE`，不从别处补足。

只读元数据与状态，**不跑任何 episode**。
"""

from __future__ import annotations

import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from fasttd3_ptf.evaluation import state_digest as sd  # noqa: E402

OUT = REPO / "docs/data/inventory_v22_diagnosis"

#: 分层抽样的族定义（冻结）。key 是族名，value 是匹配 exp_name 的正则。
FAMILY_PATTERNS = {
    "slide": re.compile(r"slide|shev1"),
    "racing": re.compile(r"^(rck|rad)_"),
    "stair": re.compile(r"stair"),
    "truck": re.compile(r"truck"),
    "cabinet": re.compile(r"cabinet"),
    "p0": re.compile(r"^p0_"),
}
PER_FAMILY = 4          # 每族取 4 对 → 6 族 24 对 ≥ 20


def load_pairs_from_full_scan() -> list:
    """从 P2.1 full scan 的 ambiguous 列表取 pair（那里已列出全部 263 组）。"""
    p = REPO / "docs/data/checkpoint_inventory_v21/full.json"
    if not p.exists():
        raise SystemExit(f"INCOMPLETE: 需要 P2.1 full scan 输出：{p}")
    data = json.loads(p.read_text(encoding="utf-8"))
    pairs = []
    for a in data.get("ambiguous_executions") or []:
        paths = a["paths"]
        num = [x for x in paths if not x.endswith("_final.pt")]
        fin = [x for x in paths if x.endswith("_final.pt")]
        if len(num) == 1 and len(fin) == 1:
            exec_id = a["execution_instance_id"]
            exp = exec_id.split("|")[1] if "|" in exec_id else exec_id
            pairs.append({"exp_name": exp, "global_step": a["global_step"],
                          "numbered": num[0], "final": fin[0]})
    return pairs


def select_stratified(pairs: list) -> tuple[list, list]:
    by_family = defaultdict(list)
    for p in sorted(pairs, key=lambda x: x["numbered"]):
        for fam, rx in FAMILY_PATTERNS.items():
            if rx.search(p["exp_name"]):
                by_family[fam].append(p)
                break
    picked, unavailable = [], []
    for fam in FAMILY_PATTERNS:
        got = by_family.get(fam, [])[:PER_FAMILY]
        for g in got:
            picked.append({**g, "family": fam})
        if len(got) < PER_FAMILY:
            unavailable.append({"family": fam, "wanted": PER_FAMILY, "got": len(got),
                                "status": "FAMILY_UNAVAILABLE"})
    return picked, unavailable


def diagnose(pair: dict) -> dict:
    import torch

    row = dict(pair)
    pa, pb = REPO / pair["numbered"], REPO / pair["final"]
    if not pa.exists() or not pb.exists():
        row["error"] = "文件缺失"
        return row
    try:
        a = torch.load(pa, map_location="cpu", weights_only=False)
        b = torch.load(pb, map_location="cpu", weights_only=False)
    except Exception as exc:  # noqa: BLE001
        row["error"] = f"{type(exc).__name__}: {exc}"
        return row

    row["inner_global_step_numbered"] = a.get("global_step")
    row["inner_global_step_final"] = b.get("global_step")
    row["top_level_keys_numbered"] = sorted(a.keys())
    row["top_level_keys_final"] = sorted(b.keys())
    row.update(sd.compare_states(a, b))
    del a, b
    return row


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    pairs = load_pairs_from_full_scan()
    picked, unavailable = select_stratified(pairs)
    print(f"候选 pair {len(pairs)} 对；分层抽样选中 {len(picked)} 对"
          f"（{len(FAMILY_PATTERNS)} 族 × {PER_FAMILY}）", flush=True)
    if unavailable:
        print(f"FAMILY_UNAVAILABLE: {unavailable}", flush=True)

    rows = []
    for i, p in enumerate(picked, 1):
        r = diagnose(p)
        rows.append(r)
        v = r.get("verdict") or r.get("error")
        print(f"  [{i:2d}/{len(picked)}] {p['family']:8s} {v}", flush=True)

    counts = defaultdict(int)
    for r in rows:
        counts[r.get("verdict") or "ERROR"] += 1

    # 差异键的频次：回答"final 与 numbered 到底差在哪"
    keyfreq = defaultdict(int)
    for r in rows:
        for k in r.get("differing_top_level_keys") or []:
            keyfreq[k] += 1

    payload = {
        "protocol": "docs/experiments/checkpoint_inventory_v22_protocol_20260807.md",
        "note": "诊断先于全量执行（protocol §2.1）。只读状态，不跑 episode。",
        "n_candidate_pairs": len(pairs),
        "n_diagnosed": len(rows),
        "family_unavailable": unavailable,
        "verdict_counts": dict(counts),
        "differing_key_frequency": dict(sorted(keyfreq.items(), key=lambda x: -x[1])),
        "rows": rows,
    }
    out = OUT / "final_numbered_diagnosis.json"
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, out)

    print(f"\n三态分布：{dict(counts)}")
    if keyfreq:
        print("差异 top-level key 频次：")
        for k, v in sorted(keyfreq.items(), key=lambda x: -x[1]):
            print(f"  {k:34s} {v}/{len(rows)}")
    print(f"wrote {out.relative_to(REPO)}")

    # 诊断本身不设通过/失败——它的目的是**测量**三态分布。
    # 但出现 FINAL_POLICY_DIVERGENCE 必须显式提示（那是真问题）。
    if counts.get("FINAL_POLICY_DIVERGENCE"):
        print(f"\n!! {counts['FINAL_POLICY_DIVERGENCE']} 对 FINAL_POLICY_DIVERGENCE"
              f" —— actor/obs norm 不同，需单独调查")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
