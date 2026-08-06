#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

STAMP="20260712TFINALV2Z"
LOG_DIR="logs/train/admission_core_v1_${STAMP}"
ARTIFACT_DIR="artifacts/admission_core_v1"
HASH_FILE="${ARTIFACT_DIR}/final_v2_implementation_sha256.txt"
PYTHON_BIN="${PYTHON_BIN:-/home/yjj/miniconda3/envs/FastTD3/bin/python}"
mkdir -p "${LOG_DIR}" "${ARTIFACT_DIR}/training_verification"

verify_hashes() {
  sha256sum -c "${HASH_FILE}"
}

run_queue_a() {
  CELL=powerlift_retain_all SEED=1 GPU_ID=3 STAMP="${STAMP}" bash scripts/run_admission_core_v1.sh
  CELL=basketball_exact_none SEED=1 GPU_ID=3 STAMP="${STAMP}" bash scripts/run_admission_core_v1.sh
  CELL=powerlift_retain_all SEED=3 GPU_ID=3 STAMP="${STAMP}" bash scripts/run_admission_core_v1.sh
}

run_queue_b() {
  CELL=powerlift_retain_all SEED=2 GPU_ID=4 STAMP="${STAMP}" bash scripts/run_admission_core_v1.sh
  CELL=basketball_exact_none SEED=2 GPU_ID=4 STAMP="${STAMP}" bash scripts/run_admission_core_v1.sh
  CELL=basketball_exact_none SEED=3 GPU_ID=4 STAMP="${STAMP}" bash scripts/run_admission_core_v1.sh
}

verify_hashes
echo "[orchestrator] starting two frozen-code queues"
set +e
run_queue_a &
pid_a=$!
run_queue_b &
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

for seed in 1 2 3; do
  "${PYTHON_BIN}" scripts/verify_admission_training.py \
    --cell powerlift_retain_all --seed "${seed}" --stamp "${STAMP}" \
    --log "${LOG_DIR}/powerlift_retain_all_s${seed}.log" \
    --out "${ARTIFACT_DIR}/training_verification/powerlift_retain_all_s${seed}.json"
  "${PYTHON_BIN}" scripts/verify_admission_training.py \
    --cell basketball_exact_none --seed "${seed}" --stamp "${STAMP}" \
    --log "${LOG_DIR}/basketball_exact_none_s${seed}.log" \
    --out "${ARTIFACT_DIR}/training_verification/basketball_exact_none_s${seed}.json"
done

echo "[orchestrator] all formal training verified; starting finalizer"
bash scripts/finalize_admission_core_v1.sh
