#!/usr/bin/env bash
set -euo pipefail

TARGET="${TARGET:?set TARGET to hurdle or crawl}"
ARM="${ARM:-online}"
SEED="${SEED:-1}"
GPU_ID="${GPU_ID:-0}"
STAMP="${STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
SMOKE="${SMOKE:-0}"
RUN_ROOT="${RUN_ROOT:-logs/train/stage_target_evidence_online_v1_${STAMP}}"
PYTHON_BIN="${PYTHON_BIN:-/home/yjj/miniconda3/envs/FastTD3/bin/python}"

case "${TARGET}" in hurdle|crawl) ;; *) echo "unsupported TARGET=${TARGET}" >&2; exit 2 ;; esac
case "${ARM}" in online|scratch) ;; *) echo "unsupported ARM=${ARM}" >&2; exit 2 ;; esac
case "${SEED}" in 1|2|3) ;; *) echo "SEED must be 1, 2, or 3" >&2; exit 2 ;; esac

ENV_NAME="h1hand-${TARGET}-v0"
EVIDENCE="configs/target_evidence/humanoidbench_${TARGET}_v1.yaml"
if [[ "${SMOKE}" == "1" ]]; then
  RUN_STOP_STEP=200
  WARMUP_STEPS=200
  PROBE_STEPS=100
  EVAL_INTERVAL=0
  CHECKPOINT_STEPS=200
  WANDB=0
  RUN_KIND=smoke
else
  RUN_STOP_STEP=30000
  WARMUP_STEPS=30000
  PROBE_STEPS=10000,20000
  EVAL_INTERVAL=5000
  CHECKPOINT_STEPS=30000
  WANDB=1
  RUN_KIND=formal_gate
fi

if [[ "${ARM}" == "online" ]]; then
  BANK="configs/source_banks/calibration/h1hand_loco3_rbo_equal_h25.yaml"
  MCG=1
  ADMISSION_MODE=target_evidence
else
  BANK="configs/source_banks/empty.yaml"
  MCG=0
  ADMISSION_MODE=legacy
fi

EXP_NAME="stage_target_evidence_${TARGET}_${ARM}_${RUN_KIND}_s${SEED}_${STAMP}"
RUN_NAME="${ENV_NAME}__${EXP_NAME}__${SEED}"
LOG_FILE="${RUN_ROOT}/${TARGET}_${ARM}_s${SEED}.log"
PROBE_DIR="${RUN_ROOT}/quarantine/${TARGET}_${ARM}_s${SEED}"

RUN_ENV=(
  env
  "CUDA_VISIBLE_DEVICES=${GPU_ID}"
  "OMP_NUM_THREADS=1"
  "MKL_NUM_THREADS=1"
  "OPENBLAS_NUM_THREADS=1"
  "NUMEXPR_NUM_THREADS=1"
  "PYTHONUNBUFFERED=1"
  "WANDB_INIT_TIMEOUT=300"
  "WANDB_SILENT=true"
  "MUJOCO_GL=egl"
  "PYTHON_BIN=${PYTHON_BIN}"
  "ENV_NAME=${ENV_NAME}"
  "EXP_NAME=${EXP_NAME}"
  "PROJECT=ptf_fasttd3_target_evidence"
  "SEED=${SEED}"
  "DEVICE_RANK=0"
  "SOURCE_BANK=${BANK}"
  "TOTAL_TIMESTEPS=100000"
  "NUM_ENVS=128"
  "BATCH_SIZE=32768"
  "BUFFER_SIZE=51200"
  "LEARNING_STARTS=10"
  "NUM_UPDATES=2"
  "SAVE_INTERVAL=0"
  "EVAL_INTERVAL=${EVAL_INTERVAL}"
  "RENDER_INTERVAL=0"
  "COMPILE=0"
  "AMP=1"
  "WANDB=${WANDB}"
  "PTF_MCG=${MCG}"
  "PTF_MCG_GROUPS=legs_torso,arms,hands"
  "PTF_MCG_WARMUP_STEPS=${WARMUP_STEPS}"
  "PTF_MCG_WARMUP_MIN_STEPS=25"
  "PTF_MCG_WARMUP_MODE=admission_bootstrap"
  "PTF_MCG_ABLATION=bootstrap_only"
  "PTF_ADMISSION_MODE=${ADMISSION_MODE}"
  "PTF_ADMISSION_STUDENT_LOGIT=0"
  "PTF_ADMISSION_REPLAY_RECENCY_HALF_LIFE=0"
  "PTF_ADMISSION_REPLAY_UNIFORM_MIX=1"
  "PTF_ADMISSION_REPLAY_PRIORITY_ALPHA=0"
  "PTF_ADMISSION_REPLAY_HANDOFF=physical_after_authority"
  "PTF_RUN_STOP_STEP=${RUN_STOP_STEP}"
  "PTF_EVAL_CHECKPOINT_STEPS=${CHECKPOINT_STEPS}"
)
if [[ "${ARM}" == "online" ]]; then
  RUN_ENV+=(
    "PTF_ADMISSION_TARGET_EVIDENCE=${EVIDENCE}"
    "PTF_ADMISSION_PROBE_STEPS=${PROBE_STEPS}"
    "PTF_ADMISSION_PROBE_OUTPUT_DIR=${PROBE_DIR}"
  )
fi

CMD=(bash scripts/official_fasttd3_train_target_ptf.sh)
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf 'target=%s arm=%s seed=%s gpu=%s run=%s\n' \
    "${TARGET}" "${ARM}" "${SEED}" "${GPU_ID}" "${RUN_NAME}"
  printf 'command:'; printf ' %q' "${RUN_ENV[@]}" "${CMD[@]}"; printf '\n'
  exit 0
fi

mkdir -p "${RUN_ROOT}"
nice -n 5 "${RUN_ENV[@]}" "${CMD[@]}" >"${LOG_FILE}" 2>&1
MODEL_FILE="models/${RUN_NAME}_${RUN_STOP_STEP}.pt"
if [[ ! -f "${MODEL_FILE}" ]]; then
  echo "missing completed-step checkpoint: ${MODEL_FILE}" >&2
  exit 3
fi
echo "${TARGET}/${ARM}/s${SEED} complete log=${LOG_FILE} model=${MODEL_FILE}"
