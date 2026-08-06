#!/usr/bin/env bash
set -euo pipefail

ARM="${ARM:-ptf}"
SEED="${SEED:-1}"
RUN_TAG="${RUN_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}"

case "${ARM}" in
  ptf)
    SOURCE_BANK="configs/source_banks/pure_ptf/h1hand_hurdle_walk.yaml"
    PTF_CONFIG="configs/experiments/classic_ptf_hurdle_single_source_v1.yaml"
    ;;
  fixed)
    SOURCE_BANK="configs/source_banks/pure_ptf/h1hand_hurdle_walk_fixed.yaml"
    PTF_CONFIG="configs/experiments/classic_fixed_transfer_hurdle_walk_v1.yaml"
    ;;
  scratch)
    SOURCE_BANK="configs/source_banks/empty.yaml"
    PTF_CONFIG="configs/experiments/classic_ptf_hurdle_single_source_v1.yaml"
    ;;
  *)
    echo "ARM must be ptf, fixed, or scratch, got: ${ARM}" >&2
    exit 2
    ;;
esac

export ENV_NAME="h1hand-hurdle-v0"
export EXP_NAME="classic_ptf_hurdle_walk_v1_${ARM}_s${SEED}_${RUN_TAG}"
export PROJECT="${PROJECT:-ptf_fasttd3_classic_revisit}"
export SEED
export SOURCE_BANK
export PTF_CONFIG
export TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-100000}"
export EVAL_INTERVAL="${EVAL_INTERVAL:-5000}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-25000}"
export NUM_ENVS="${NUM_ENVS:-128}"
export COMPILE="${COMPILE:-0}"
export AMP="${AMP:-1}"
export WANDB="${WANDB:-1}"
export WANDB_INIT_TIMEOUT="${WANDB_INIT_TIMEOUT:-300}"

LOG_DIR="${LOG_DIR:-logs/train/classic_ptf_hurdle_single_source_v1}"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/${EXP_NAME}.log"

echo "Classic PTF Hurdle v1: arm=${ARM} seed=${SEED} bank=${SOURCE_BANK} log=${LOG_FILE}"
bash scripts/official_fasttd3_train_target_ptf.sh 2>&1 | tee "${LOG_FILE}"
