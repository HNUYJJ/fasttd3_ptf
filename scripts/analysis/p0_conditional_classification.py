"""P0 条件分类生成脚本（posthoc；十二次复核修复 5——可复现入口）。

在"critic 累计占比下偏已由冷启动机制解释"的**条件**下，调用冻结的
`p0_adjudicate._classify_task/_joint`（判序未修改）对正式 adjudication
manifest 计算条件分类。跳过的唯一环节是 treatment 审计（即条件本身）。

输出记录全部输入文件 SHA256、裁决器脚本 SHA256 与 git HEAD，保证可复现。
本脚本不修改任何正式产物；预注册正式结论 ENGINEERING_INVALID 不受影响。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
from p0_adjudicate import _classify_task, _joint, _load  # noqa: E402

INPUTS = {
    "crawl_manifest": "logs/p0_lease_oracle/adjudication_manifest_crawl.json",
    "truck_manifest": "logs/p0_lease_oracle/adjudication_manifest_truck.json",
    "crawl_delta": "configs/experiments/p0_frozen_delta_crawl.json",
    "truck_delta": "configs/experiments/p0_frozen_delta_truck.json",
}
ADJUDICATOR = "scripts/p0_adjudicate.py"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default=str(REPO / "docs" / "data" / "p0_posthoc" / "conditional_classification.json"),
    )
    args = parser.parse_args()
    results = {}
    for task in ("crawl", "truck"):
        manifest = _load(REPO / INPUTS[f"{task}_manifest"])
        delta = _load(REPO / INPUTS[f"{task}_delta"])
        dup_a = _load(manifest["duplicate"]["eval_a"])["aggregate"]["return_mean"]
        dup_b = _load(manifest["duplicate"]["eval_b"])["aggregate"]["return_mean"]
        results[task] = _classify_task(task, manifest, delta, abs(dup_a - dup_b))
    joint = _joint(results["crawl"]["classification"], results["truck"]["classification"])
    git_head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False,
    ).stdout.strip()
    out = {
        "analysis_type": "posthoc_engineering_sensitivity",
        "preregistered_verdict_unchanged": "ENGINEERING_INVALID",
        "condition": (
            "critic cumulative fraction shortfall accepted as explained "
            "cold-start mechanism effect (treatment audit skipped by construction)"
        ),
        "conditional_tasks": results,
        "conditional_joint": joint,
        "provenance": {
            "git_head": git_head,
            "adjudicator_sha256": _sha256(REPO / ADJUDICATOR),
            "input_sha256": {k: _sha256(REPO / v) for k, v in INPUTS.items()},
        },
        "utc": datetime.now(timezone.utc).isoformat(),
        "document": "docs/p0_posthoc_engineering_sensitivity_20260718.md",
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    print(f"conditional joint = {joint['verdict']} -> {out_path}")


if __name__ == "__main__":
    main()
