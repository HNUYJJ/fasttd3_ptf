"""hurdle 加速实验的剂量验收(预注册 docs/experiments/hurdle_speedup_v1_prereg_20260730.md §3)。

预注册要求:source 臂的 behavior source share 必须落在 [0.48, 0.52],
否则该 seed 作废重跑(EQD30K 实测为 0.500–0.502)。

share 直接从 checkpoint 的 admission_audit 累计计数读出:
    behavior share = execution_counts[0] / sum(execution_counts)
    critic   share = critic_sample_counts[0] / sum(critic_sample_counts)
索引 0 = source,索引 1 = student。

用法: python scripts/analysis/audit_hurdle_speedup_dose_v1.py [ckpt ...]
     无参数时审计 hspd_source_s{1,2,3} 的 100k checkpoint。
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import torch

BAND = (0.48, 0.52)


def share(counts) -> float:
    c = [int(x) for x in counts]
    if len(c) != 2 or sum(c) <= 0:
        raise AssertionError(f"invalid source/student counts: {c}")
    return c[0] / sum(c)


def audit(path: str) -> dict:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    a = ck.get("admission_audit")
    if a is None:
        raise AssertionError(f"{path}: no admission_audit (是 scratch 臂?)")
    out = {
        "checkpoint": path,
        "global_step": int(ck.get("global_step", -1)),
        "source_names": ck.get("source_names"),
        "execution_counts": [int(x) for x in a["execution_counts"]],
        "critic_sample_counts": [int(x) for x in a["critic_sample_counts"]],
    }
    out["behavior_share"] = share(a["execution_counts"])
    out["critic_share"] = share(a["critic_sample_counts"])
    out["behavior_in_band"] = BAND[0] <= out["behavior_share"] <= BAND[1]
    out["critic_in_band"] = BAND[0] <= out["critic_share"] <= BAND[1]
    return out


def main() -> None:
    args = sys.argv[1:]
    if args:
        paths = args
    else:
        paths = sorted(
            p for s in (1, 2, 3)
            for p in glob.glob(f"models/*hspd_source_s{s}__*_100000.pt")
        )
    if not paths:
        print("no checkpoints found"); return

    rows, ok = [], True
    print(f"剂量验收带 = [{BAND[0]}, {BAND[1]}]  (EQD30K 实测 0.500–0.502)")
    print("=" * 96)
    for p in paths:
        try:
            r = audit(p)
        except AssertionError as e:
            print(f"  SKIP {e}"); continue
        rows.append(r)
        flag_b = "OK " if r["behavior_in_band"] else "FAIL"
        flag_c = "OK " if r["critic_in_band"] else "FAIL"
        ok = ok and r["behavior_in_band"]
        print(f"  {Path(p).name}")
        print(f"     step={r['global_step']}  sources={r['source_names']}")
        print(f"     behavior share = {r['behavior_share']:.4f}  [{flag_b}]   "
              f"counts={r['execution_counts']}")
        print(f"     critic   share = {r['critic_share']:.4f}  [{flag_c}]   "
              f"counts={r['critic_sample_counts']}")
    print("=" * 96)
    print(f"剂量验收: {'PASS' if ok else 'FAIL —— 预注册要求作废重跑'}")

    out = Path("docs/data/hurdle_speedup_v1/dose_audit.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"band": list(BAND), "rows": rows, "pass": ok},
                              indent=2, ensure_ascii=False))
    print(f"written {out}")


if __name__ == "__main__":
    main()
