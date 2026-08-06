#!/usr/bin/env bash
set -euo pipefail

RUN_TAG="${RUN_TAG:-formal_$(date -u +%Y%m%dT%H%M%SZ)}"
PTF_GPU="${PTF_GPU:-2}"
SCRATCH_GPU="${SCRATCH_GPU:-3}"
PYTHON_BIN="${PYTHON_BIN:-/home/yjj/miniconda3/envs/FastTD3/bin/python}"
MATRIX_LOG_DIR="${MATRIX_LOG_DIR:-logs/train/classic_ptf_hurdle_single_source_v1/matrix_${RUN_TAG}}"
mkdir -p "${MATRIX_LOG_DIR}"

run_queue() {
  local gpu="$1"
  local arm="$2"
  local seed
  for seed in 1 2 3; do
    echo "[$(date -u +%FT%TZ)] START arm=${arm} seed=${seed} gpu=${gpu}"
    env \
      PYTHON_BIN="${PYTHON_BIN}" \
      CUDA_VISIBLE_DEVICES="${gpu}" \
      DEVICE_RANK=0 \
      ARM="${arm}" \
      SEED="${seed}" \
      RUN_TAG="${RUN_TAG}" \
      TOTAL_TIMESTEPS=100000 \
      EVAL_INTERVAL=5000 \
      SAVE_INTERVAL=25000 \
      WANDB=1 \
      WANDB_INIT_TIMEOUT=300 \
      bash scripts/run_classic_ptf_hurdle_single_source_v1.sh
    echo "[$(date -u +%FT%TZ)] DONE arm=${arm} seed=${seed} gpu=${gpu}"
  done
}

echo "Classic PTF Hurdle matrix: tag=${RUN_TAG}, ptf_gpu=${PTF_GPU}, scratch_gpu=${SCRATCH_GPU}"
run_queue "${PTF_GPU}" ptf >"${MATRIX_LOG_DIR}/ptf_queue.log" 2>&1 &
ptf_pid=$!
run_queue "${SCRATCH_GPU}" scratch >"${MATRIX_LOG_DIR}/scratch_queue.log" 2>&1 &
scratch_pid=$!

status=0
wait "${ptf_pid}" || status=$?
wait "${scratch_pid}" || status=$?
if [[ "${status}" -ne 0 ]]; then
  echo "Classic PTF Hurdle matrix FAILED status=${status}; inspect ${MATRIX_LOG_DIR}" >&2
  exit "${status}"
fi
echo "Classic PTF Hurdle matrix COMPLETE: ${MATRIX_LOG_DIR}"
