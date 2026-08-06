#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

PYTHON_BIN="${PYTHON_BIN:-/home/yjj/miniconda3/envs/FastTD3/bin/python}"
LOG_DIR="logs/train/admission_core_v1_20260712TFINALV2Z"
RNG_LOG_DIR="${LOG_DIR}"
ARTIFACT_DIR="artifacts/admission_core_v1"
PROBE_DIR="logs/probe/admission_core_v1"
mkdir -p "${ARTIFACT_DIR}" "${PROBE_DIR}"

expected_powerlift=(powerlift_retain_all_s1 powerlift_retain_all_s2 powerlift_retain_all_s3)
expected_basketball=(basketball_exact_none_s1 basketball_exact_none_s2 basketball_exact_none_s3)

cell_complete() {
  local meta="$1"
  [[ -f "${meta}" ]] && grep -q '^exit_code=0$' "${meta}"
}

while true; do
  complete=0
  for cell in "${expected_powerlift[@]}"; do
    meta="${LOG_DIR}/${cell}.meta.txt"
    if cell_complete "${meta}"; then
      complete=$((complete + 1))
    elif [[ -f "${meta}" ]] && grep -q '^exit_code=' "${meta}"; then
      echo "failed training cell: ${cell}" >&2
      grep '^exit_code=' "${meta}" >&2
      exit 3
    fi
  done
  for cell in "${expected_basketball[@]}"; do
    meta="${RNG_LOG_DIR}/${cell}.meta.txt"
    if cell_complete "${meta}"; then
      complete=$((complete + 1))
    elif [[ -f "${meta}" ]] && grep -q '^exit_code=' "${meta}"; then
      echo "failed RNG-isolated training cell: ${cell}" >&2
      grep '^exit_code=' "${meta}" >&2
      exit 3
    fi
  done
  printf '[finalizer] %s complete=%d/6\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${complete}"
  [[ "${complete}" -eq 6 ]] && break
  sleep 30
done

"${PYTHON_BIN}" scripts/analyze_admission_core_v1.py

CUDA_VISIBLE_DEVICES="${EVAL_GPU:-4}" "${PYTHON_BIN}" scripts/stability_deconfounded_audit.py collect \
  --spec configs/experiments/admission_core_v1_eval.json \
  --out "${PROBE_DIR}/powerlift_all_conditions.jsonl" \
  --tasks powerlift --overwrite

CUDA_VISIBLE_DEVICES="${EVAL_GPU:-4}" "${PYTHON_BIN}" scripts/stability_deconfounded_audit.py collect \
  --spec configs/experiments/admission_core_v1_eval.json \
  --out "${PROBE_DIR}/basketball_all_conditions.jsonl" \
  --tasks basketball --overwrite

"${PYTHON_BIN}" scripts/stability_deconfounded_audit.py summarize \
  --spec configs/experiments/admission_core_v1_eval.json \
  --input \
    "${PROBE_DIR}/powerlift_all_conditions.jsonl" \
    "${PROBE_DIR}/basketball_all_conditions.jsonl" \
  --json-out "${ARTIFACT_DIR}/paired_evaluation.json" \
  --md-out "${ARTIFACT_DIR}/paired_evaluation.md" \
  --require-complete

"${PYTHON_BIN}" scripts/adjudicate_admission_core_v1.py \
  --summary "${ARTIFACT_DIR}/paired_evaluation.json" \
  --json-out "${ARTIFACT_DIR}/performance_verdict.json" \
  --md-out "${ARTIFACT_DIR}/performance_verdict.md"

echo "[finalizer] complete"
