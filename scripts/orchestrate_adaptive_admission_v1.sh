#!/usr/bin/env bash
set -euo pipefail

STAMP="${STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
GPU_A="${GPU_A:-0}"
GPU_B="${GPU_B:-2}"
PYTHON_BIN="${PYTHON_BIN:-/home/yjj/miniconda3/envs/FastTD3/bin/python}"
LOG_DIR="logs/train/adaptive_admission_v1_${STAMP}"
STATUS_FILE="${LOG_DIR}/orchestrator_status.tsv"
ORCH_LOG="${LOG_DIR}/orchestrator.log"
CURRENT_FILE="${LOG_DIR}/current_pair.tsv"

mkdir -p "${LOG_DIR}"
printf 'cell\tseed\tgpu\tstate\tutc\texit_code\n' > "${STATUS_FILE}"
{
  echo "stamp=${STAMP}"
  echo "started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "gpu_a=${GPU_A}"
  echo "gpu_b=${GPU_B}"
  echo "python=${PYTHON_BIN}"
  echo "git_head=$(git rev-parse HEAD)"
  echo "protocol_sha256=$(sha256sum configs/experiments/adaptive_admission_v1.yaml | awk '{print $1}')"
  echo "launcher_sha256=$(sha256sum scripts/run_adaptive_admission_v1.sh | awk '{print $1}')"
  echo "implementation_sha256=$(sha256sum \
    fasttd3_ptf/official_fasttd3_ptf/admission_control.py \
    fasttd3_ptf/official_fasttd3_ptf/ptf_replay.py \
    fasttd3_ptf/official_fasttd3_ptf/train_ptf.py \
    fasttd3_ptf/ptf/mcg.py \
    scripts/official_fasttd3_train_target_ptf.sh \
    scripts/run_adaptive_admission_v1.sh | sha256sum | awk '{print $1}')"
} > "${LOG_DIR}/orchestrator.meta.txt"

run_one() {
  local cell="$1" seed="$2" gpu="$3"
  printf '%s\t%s\t%s\trunning\t%s\t\n' \
    "${cell}" "${seed}" "${gpu}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${STATUS_FILE}"
  echo "[orchestrator] start cell=${cell} seed=${seed} gpu=${gpu}" >> "${ORCH_LOG}"
  set +e
  CELL="${cell}" SEED="${seed}" GPU_ID="${gpu}" STAMP="${STAMP}" \
    PYTHON_BIN="${PYTHON_BIN}" scripts/run_adaptive_admission_v1.sh >> "${ORCH_LOG}" 2>&1
  local status=$?
  set -e
  local state="completed"
  [[ "${status}" == "0" ]] || state="failed"
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${cell}" "${seed}" "${gpu}" "${state}" \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${status}" >> "${STATUS_FILE}"
  echo "[orchestrator] finish cell=${cell} seed=${seed} gpu=${gpu} exit=${status}" >> "${ORCH_LOG}"
  return "${status}"
}

cleanup() {
  local children
  children="$(jobs -pr || true)"
  if [[ -n "${children}" ]]; then
    kill ${children} 2>/dev/null || true
  fi
}
trap cleanup INT TERM

# Matched cells share a pair whenever possible so host contention is balanced.
PAIRS=(
  "crawl_adaptive:1 crawl_static:1"
  "crawl_adaptive:2 crawl_static:2"
  "crawl_adaptive:3 crawl_static:3"
  "truck_adaptive:1 powerlift_adaptive:1"
  "truck_adaptive:2 powerlift_adaptive:2"
  "truck_adaptive:3 powerlift_adaptive:3"
  "basketball_adaptive:1 basketball_static:1"
  "basketball_adaptive:2 basketball_static:2"
  "basketball_adaptive:3 basketball_static:3"
)

for pair in "${PAIRS[@]}"; do
  read -r left right <<< "${pair}"
  left_cell="${left%%:*}"; left_seed="${left##*:}"
  right_cell="${right%%:*}"; right_seed="${right##*:}"
  run_one "${left_cell}" "${left_seed}" "${GPU_A}" &
  left_pid=$!
  run_one "${right_cell}" "${right_seed}" "${GPU_B}" &
  right_pid=$!
  printf 'cell\tseed\tgpu\tpid\n%s\t%s\t%s\t%s\n%s\t%s\t%s\t%s\n' \
    "${left_cell}" "${left_seed}" "${GPU_A}" "${left_pid}" \
    "${right_cell}" "${right_seed}" "${GPU_B}" "${right_pid}" > "${CURRENT_FILE}"

  set +e
  wait "${left_pid}"; left_status=$?
  wait "${right_pid}"; right_status=$?
  set -e
  if [[ "${left_status}" != "0" || "${right_status}" != "0" ]]; then
    echo "failed_pair=${pair}" > "${LOG_DIR}/FAILED"
    echo "left_exit=${left_status}" >> "${LOG_DIR}/FAILED"
    echo "right_exit=${right_status}" >> "${LOG_DIR}/FAILED"
    exit 1
  fi
done

rm -f "${CURRENT_FILE}"
{
  echo "stamp=${STAMP}"
  echo "completed_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "runs=18"
} > "${LOG_DIR}/COMPLETE"
echo "[orchestrator] all 18 runs completed" >> "${ORCH_LOG}"
