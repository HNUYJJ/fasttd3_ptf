#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

STAMP="${STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
GPU_A="${GPU_A:-2}"
GPU_B="${GPU_B:-3}"
PYTHON_BIN="${PYTHON_BIN:-/home/yjj/miniconda3/envs/FastTD3/bin/python}"
LOG_DIR="logs/train/admission_handoff_v1_${STAMP}"
ARTIFACT_DIR="artifacts/admission_handoff_v1/${STAMP}"
HASH_FILE="${ARTIFACT_DIR}/frozen_implementation.sha256"
mkdir -p "${LOG_DIR}" "${ARTIFACT_DIR}/training_verification"

FILES=(
  configs/experiments/admission_handoff_v1.yaml
  fasttd3_ptf/official_fasttd3_ptf/admission_control.py
  fasttd3_ptf/official_fasttd3_ptf/ptf_replay.py
  fasttd3_ptf/official_fasttd3_ptf/train_ptf.py
  fasttd3_ptf/ptf/mcg.py
  scripts/official_fasttd3_train_target_ptf.sh
  scripts/run_admission_handoff_v1.sh
  scripts/verify_admission_handoff.py
)
sha256sum "${FILES[@]}" > "${HASH_FILE}"

verify_hashes() {
  sha256sum -c "${HASH_FILE}"
}

run_cell() {
  local cell="$1" seed="$2" gpu="$3"
  verify_hashes
  CELL="${cell}" SEED="${seed}" GPU_ID="${gpu}" STAMP="${STAMP}" \
    bash scripts/run_admission_handoff_v1.sh
  verify_hashes
  "${PYTHON_BIN}" scripts/verify_admission_handoff.py \
    --cell "${cell}" --seed "${seed}" --stamp "${STAMP}" \
    --log "${LOG_DIR}/${cell}_s${seed}.log" \
    --out "${ARTIFACT_DIR}/training_verification/${cell}_s${seed}.json"
}

# Two queues are the hard concurrency ceiling. The ordering completes all
# powerlift seeds after the second wave while already using the otherwise idle
# slot for truck; the last wave finishes the two remaining truck seeds.
queue_a() {
  run_cell powerlift_admission_all_fix 1 "${GPU_A}"
  run_cell powerlift_admission_all_fix 3 "${GPU_A}"
  run_cell truck_admission_h4_fix 2 "${GPU_A}"
}

queue_b() {
  run_cell powerlift_admission_all_fix 2 "${GPU_B}"
  run_cell truck_admission_h4_fix 1 "${GPU_B}"
  run_cell truck_admission_h4_fix 3 "${GPU_B}"
}

echo "[orchestrator] stamp=${STAMP} GPUs=${GPU_A},${GPU_B}; max_concurrency=2"
set +e
queue_a &
pid_a=$!
queue_b &
pid_b=$!
wait "${pid_a}"
status_a=$?
wait "${pid_b}"
status_b=$?
set -e
if [[ "${status_a}" -ne 0 || "${status_b}" -ne 0 ]]; then
  echo "formal queue failed: a=${status_a} b=${status_b}" >&2
  exit 6
fi
verify_hashes
"${PYTHON_BIN}" scripts/analyze_admission_handoff.py \
  --stamp "${STAMP}" --out-dir "${ARTIFACT_DIR}"
echo "[orchestrator] completed and adjudicated stamp=${STAMP}"
