#!/usr/bin/env bash
set -euo pipefail

CELL="${CELL:?set CELL to basketball_exact_none or powerlift_retain_all}"
SEED="${SEED:-1}"
GPU_ID="${GPU_ID:-2}"
STAMP="${STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
DRY_RUN="${DRY_RUN:-0}"
PYTHON_BIN="${PYTHON_BIN:-/home/yjj/miniconda3/envs/FastTD3/bin/python}"

case "${SEED}" in 1|2|3) ;; *) echo "SEED must be 1, 2, or 3" >&2; exit 2 ;; esac
case "${CELL}" in
  basketball_exact_none)
    TASK="basketball"
    BANK="configs/source_banks/h1hand_std9_wfix_basketball.yaml"
    ADMISSION_MODE="none"
    STUDENT_LOGIT="0.0"
    ;;
  powerlift_retain_all)
    TASK="powerlift"
    BANK="configs/source_banks/h1hand_std9_wfix_powerlift.yaml"
    ADMISSION_MODE="all"
    STUDENT_LOGIT="2.6708791088087076"
    ;;
  *) echo "unsupported CELL=${CELL}" >&2; exit 2 ;;
esac

ENV_NAME="h1hand-${TASK}-v0"
EXP_NAME="h1hand_${TASK}_admission_v1_${ADMISSION_MODE}_s${SEED}_${STAMP}"
LOG_DIR="logs/train/admission_core_v1_${STAMP}"
LOG_FILE="${LOG_DIR}/${CELL}_s${SEED}.log"
META_FILE="${LOG_DIR}/${CELL}_s${SEED}.meta.txt"

RUN_ENV=(
  env
  "CUDA_VISIBLE_DEVICES=${GPU_ID}"
  "OMP_NUM_THREADS=1"
  "MKL_NUM_THREADS=1"
  "OPENBLAS_NUM_THREADS=1"
  "NUMEXPR_NUM_THREADS=1"
  "PYTHONUNBUFFERED=1"
  "PYTHON_BIN=${PYTHON_BIN}"
  "ENV_NAME=${ENV_NAME}"
  "EXP_NAME=${EXP_NAME}"
  "PROJECT=fasttd3_ptf"
  "SEED=${SEED}"
  "DEVICE_RANK=0"
  "SOURCE_BANK=${BANK}"
  "TOTAL_TIMESTEPS=100000"
  "NUM_ENVS=128"
  "BATCH_SIZE=32768"
  "BUFFER_SIZE=51200"
  "LEARNING_STARTS=10"
  "NUM_UPDATES=2"
  "SAVE_INTERVAL=30000"
  "EVAL_INTERVAL=5000"
  "RENDER_INTERVAL=0"
  "COMPILE=0"
  "AMP=1"
  "WANDB=0"
  "PTF_MCG=1"
  "PTF_MCG_GROUPS=legs_torso,arms"
  "PTF_MCG_WARMUP_STEPS=30000"
  "PTF_MCG_WARMUP_MODE=admission_bootstrap"
  "PTF_MCG_ABLATION=bootstrap_only"
  "PTF_ADMISSION_MODE=${ADMISSION_MODE}"
  "PTF_ADMISSION_STUDENT_LOGIT=${STUDENT_LOGIT}"
  "PTF_ADMISSION_REPLAY_RECENCY_HALF_LIFE=0"
  "PTF_ADMISSION_REPLAY_UNIFORM_MIX=1"
  "PTF_ADMISSION_REPLAY_PRIORITY_ALPHA=0"
)
CMD=(bash scripts/official_fasttd3_train_target_ptf.sh)

if [[ "${DRY_RUN}" == "1" ]]; then
  printf 'cell=%s seed=%s gpu=%s exp=%s\n' "${CELL}" "${SEED}" "${GPU_ID}" "${EXP_NAME}"
  printf 'command:'; printf ' %q' "${RUN_ENV[@]}" "${CMD[@]}"; printf '\n'
  exit 0
fi

mkdir -p "${LOG_DIR}"
{
  echo "started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "cell=${CELL}"
  echo "seed=${SEED}"
  echo "gpu_id=${GPU_ID}"
  echo "exp_name=${EXP_NAME}"
  echo "bank=${BANK}"
  echo "bank_sha256=$(sha256sum "${BANK}" | awk '{print $1}')"
  echo "protocol_sha256=$(sha256sum configs/experiments/admission_core_v1.yaml | awk '{print $1}')"
  echo "implementation_sha256=$(sha256sum \
    fasttd3_ptf/official_fasttd3_ptf/admission_control.py \
    fasttd3_ptf/official_fasttd3_ptf/humanoid_bench_env.py \
    fasttd3_ptf/official_fasttd3_ptf/ptf_replay.py \
    fasttd3_ptf/official_fasttd3_ptf/rng_isolation.py \
    fasttd3_ptf/official_fasttd3_ptf/train_ptf.py \
    fasttd3_ptf/ptf/mcg.py \
    scripts/official_fasttd3_train_target_ptf.sh \
    scripts/run_admission_core_v1.sh | sha256sum | awk '{print $1}')"
  echo "git_head=$(git rev-parse HEAD)"
} > "${META_FILE}"

set +e
nice -n 5 "${RUN_ENV[@]}" "${CMD[@]}" > "${LOG_FILE}" 2>&1
status=$?
set -e
{
  echo "finished_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "exit_code=${status}"
} >> "${META_FILE}"
echo "${CELL}/s${SEED} exit=${status} log=${LOG_FILE}"
exit "${status}"
