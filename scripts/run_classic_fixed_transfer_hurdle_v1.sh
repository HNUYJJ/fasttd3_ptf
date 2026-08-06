#!/usr/bin/env bash
set -euo pipefail

RUN_TAG="${RUN_TAG:-fixed_formal_$(date -u +%Y%m%dT%H%M%SZ)}"
GPU="${GPU:-2}"
PYTHON_BIN="${PYTHON_BIN:-/home/yjj/miniconda3/envs/FastTD3/bin/python}"
QUEUE_DIR="${QUEUE_DIR:-logs/train/classic_ptf_hurdle_single_source_v1/matrix_${RUN_TAG}}"
mkdir -p "${QUEUE_DIR}"

for seed in 1 2 3; do
  echo "[$(date -u +%FT%TZ)] START arm=fixed seed=${seed} gpu=${GPU}"
  env \
    PYTHON_BIN="${PYTHON_BIN}" \
    CUDA_VISIBLE_DEVICES="${GPU}" \
    DEVICE_RANK=0 \
    ARM=fixed \
    SEED="${seed}" \
    RUN_TAG="${RUN_TAG}" \
    TOTAL_TIMESTEPS=100000 \
    EVAL_INTERVAL=5000 \
    SAVE_INTERVAL=25000 \
    WANDB=1 \
    WANDB_INIT_TIMEOUT=300 \
    bash scripts/run_classic_ptf_hurdle_single_source_v1.sh
  echo "[$(date -u +%FT%TZ)] DONE arm=fixed seed=${seed} gpu=${GPU}"
done

echo "Classic fixed-transfer Hurdle COMPLETE: ${QUEUE_DIR}"
