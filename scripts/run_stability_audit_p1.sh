#!/usr/bin/env bash
set -euo pipefail

# Run one preregistered P1 single-source cell. This launcher deliberately runs a
# single cell so GPU/CPU scheduling remains explicit on the shared server.

TASK="${TASK:?set TASK to cabinet, maze, powerlift, or basketball}"
SOURCE="${SOURCE:?set SOURCE to stand, walk, or run}"
SEED="${SEED:-1}"
GPU_ID="${GPU_ID:-0}"
STAMP="${STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
DRY_RUN="${DRY_RUN:-0}"
CONDITION="${CONDITION:-sd_${SOURCE}}"
AUDIT_TAG="${AUDIT_TAG:-p1}"
WARMUP_EXEC_PROB="${WARMUP_EXEC_PROB:-0.5}"

case "${TASK}" in
  cabinet|maze|powerlift|basketball) ;;
  *) echo "unsupported TASK=${TASK}" >&2; exit 2 ;;
esac
case "${SOURCE}" in
  stand|walk|run) ;;
  *) echo "unsupported SOURCE=${SOURCE}" >&2; exit 2 ;;
esac
[[ "${CONDITION}" =~ ^[A-Za-z0-9_]+$ ]] || {
  echo "CONDITION must contain only letters, digits, or underscores: ${CONDITION}" >&2
  exit 2
}
[[ "${AUDIT_TAG}" =~ ^[A-Za-z0-9_]+$ ]] || {
  echo "AUDIT_TAG must contain only letters, digits, or underscores: ${AUDIT_TAG}" >&2
  exit 2
}
awk -v p="${WARMUP_EXEC_PROB}" 'BEGIN { exit !(p >= 0 && p <= 1) }' || {
  echo "WARMUP_EXEC_PROB must be in [0, 1]: ${WARMUP_EXEC_PROB}" >&2
  exit 2
}

BANK="configs/source_banks/audit/h1hand_${TASK}_sd_${SOURCE}.yaml"
[[ -f "${BANK}" ]] || { echo "missing bank: ${BANK}" >&2; exit 2; }

ENV_NAME="h1hand-${TASK}-v0"
EXP_NAME="h1hand_${TASK}_${CONDITION}_s${SEED}_${STAMP}"
LOG_DIR="logs/train/stability_audit_${AUDIT_TAG}_${STAMP}"
LOG_FILE="${LOG_DIR}/${TASK}_${CONDITION}_s${SEED}.log"
META_FILE="${LOG_DIR}/${TASK}_${CONDITION}_s${SEED}.meta.txt"
PYTHON_BIN="${PYTHON_BIN:-/home/yjj/miniconda3/envs/FastTD3/bin/python}"

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
  "COMPILE=0"
  "AMP=1"
  "WANDB=1"
  "PTF_MCG=1"
  "PTF_MCG_GROUPS=legs_torso,arms"
  "PTF_MCG_WARMUP_STEPS=30000"
  "PTF_MCG_WARMUP_EXEC_PROB=${WARMUP_EXEC_PROB}"
  "PTF_MCG_WARMUP_MODE=safe_bootstrap"
  "PTF_MCG_ABLATION=bootstrap_only"
)
CMD=(bash scripts/official_fasttd3_train_target_ptf.sh)

if [[ "${DRY_RUN}" == "1" ]]; then
  printf 'cell=%s/%s/s%s gpu=%s\n' "${TASK}" "${CONDITION}" "${SEED}" "${GPU_ID}"
  printf 'bank=%s\nexp_name=%s\n' "${BANK}" "${EXP_NAME}"
  printf 'source=%s\nwarmup_exec_prob=%s\naudit_tag=%s\n' \
    "${SOURCE}" "${WARMUP_EXEC_PROB}" "${AUDIT_TAG}"
  printf 'command:'
  printf ' %q' "${RUN_ENV[@]}" "${CMD[@]}"
  printf '\n'
  exit 0
fi

mkdir -p "${LOG_DIR}"
cp "${BANK}" "${LOG_DIR}/$(basename "${BANK}")"
{
  echo "started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "task=${TASK}"
  echo "source=${SOURCE}"
  echo "condition=${CONDITION}"
  echo "audit_tag=${AUDIT_TAG}"
  echo "warmup_exec_prob=${WARMUP_EXEC_PROB}"
  echo "seed=${SEED}"
  echo "gpu_id=${GPU_ID}"
  echo "exp_name=${EXP_NAME}"
  echo "bank=${BANK}"
  echo "git_head=$(git rev-parse HEAD)"
  echo "bank_sha256=$(sha256sum "${BANK}" | awk '{print $1}')"
  echo "launcher_sha256=$(sha256sum "$0" | awk '{print $1}')"
  echo "working_tree_begin"
  git status --short
  echo "working_tree_end"
} > "${META_FILE}"

set +e
nice -n 5 "${RUN_ENV[@]}" "${CMD[@]}" 2>&1 | tee "${LOG_FILE}"
status=${PIPESTATUS[0]}
set -e

{
  echo "finished_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "exit_code=${status}"
} >> "${META_FILE}"
echo "Audit cell ${TASK}/${CONDITION}/s${SEED} exit=${status} log=${LOG_FILE}"
exit "${status}"
