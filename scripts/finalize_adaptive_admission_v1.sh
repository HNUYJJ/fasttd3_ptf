#!/usr/bin/env bash
set -euo pipefail

STAMP="${STAMP:?set the completed adaptive_admission_v1 stamp}"
PYTHON_BIN="${PYTHON_BIN:-/home/yjj/miniconda3/envs/FastTD3/bin/python}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${ROOT}/logs/train/adaptive_admission_v1_${STAMP}"
OUT_DIR="${ROOT}/artifacts/adaptive_admission_v1/${STAMP}"
VERIFY_DIR="${OUT_DIR}/training_verification"

[[ -f "${LOG_DIR}/COMPLETE" ]] || {
  echo "missing orchestrator COMPLETE marker: ${LOG_DIR}/COMPLETE" >&2
  exit 2
}
[[ ! -e "${LOG_DIR}/FAILED" ]] || {
  echo "orchestrator FAILED marker exists: ${LOG_DIR}/FAILED" >&2
  exit 2
}
mkdir -p "${VERIFY_DIR}"

audit_one() {
  local task="$1" mode="$2" seed="$3"
  local checkpoint="${ROOT}/models/h1hand-${task}-v0__h1hand_${task}_adaptive_admission_v1_${mode}_s${seed}_${STAMP}__${seed}_final.pt"
  local out="${VERIFY_DIR}/${task}_${mode}_s${seed}.json"
  [[ -f "${checkpoint}" ]] || {
    echo "missing final checkpoint: ${checkpoint}" >&2
    return 2
  }
  "${PYTHON_BIN}" "${ROOT}/scripts/audit_adaptive_admission_checkpoint.py" \
    --checkpoint "${checkpoint}" \
    --expected-step 100000 \
    --expected-adaptive "$([[ "${mode}" == "adaptive" ]] && echo true || echo false)" \
    --out "${out}"
}

for task in crawl truck powerlift basketball; do
  for seed in 1 2 3; do
    audit_one "${task}" adaptive "${seed}"
  done
done
for task in crawl basketball; do
  for seed in 1 2 3; do
    audit_one "${task}" static "${seed}"
  done
done

# --allow-incomplete suppresses a non-zero exit for a legitimate scientific
# FAIL.  The validation block below still requires all runs and audits to be
# complete; PASS versus FAIL remains the report's result, not a shell failure.
"${PYTHON_BIN}" "${ROOT}/scripts/analyze_adaptive_admission_v1.py" \
  --root "${ROOT}" --stamp "${STAMP}" --allow-incomplete --out-dir "${OUT_DIR}"

"${PYTHON_BIN}" - "${OUT_DIR}" "${STAMP}" <<'PY'
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

out_dir = Path(sys.argv[1])
stamp = sys.argv[2]
report_path = out_dir / f"analysis_{stamp}.json"
report = json.loads(report_path.read_text())
audit_paths = sorted((out_dir / "training_verification").glob("*.json"))
audits = [json.loads(path.read_text()) for path in audit_paths]
checks = {
    "orchestrator_complete": True,
    "analysis_complete": report.get("complete") is True,
    "evaluation_grid_complete": report.get("evaluation_grid_complete") is True,
    "eighteen_checkpoint_audits_present": len(audits) == 18,
    "eighteen_checkpoint_audits_pass": len(audits) == 18
    and all(row.get("pass") is True for row in audits),
    "embedded_lifecycle_audits_pass": report["mechanism_audit"][
        "all_available_pass"
    ]
    is True,
    "no_trigger_regression_pass": report["no_trigger_regression"]["pass"]
    is True,
    "historical_comparators_verified": report[
        "historical_comparator_verification"
    ]["pass"]
    is True,
}
summary = {
    "schema_version": 1,
    "experiment": "adaptive_admission_v1",
    "stamp": stamp,
    "finalized_at": datetime.now(timezone.utc).isoformat(),
    "scientific_status": report["overall_status"],
    "checks": checks,
    "pass": all(checks.values()),
    "analysis": str(report_path),
    "checkpoint_audits": [str(path) for path in audit_paths],
}
path = out_dir / "finalization_summary.json"
path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps(summary, indent=2, sort_keys=True))
if not summary["pass"]:
    raise SystemExit(4)
PY
