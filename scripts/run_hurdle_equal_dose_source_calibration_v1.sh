#!/usr/bin/env bash
set -euo pipefail

ARM="${ARM:?set ARM to scratch, stand, walk, or run}"
SEED="${SEED:-1}"
GPU_ID="${GPU_ID:-0}"
TARGET="${TARGET:-hurdle}"
STAMP="${STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
SMOKE="${SMOKE:-0}"
RUN_ROOT="${RUN_ROOT:-logs/train/${TARGET}_equal_dose_source_calibration_v1_${STAMP}}"
PYTHON_BIN="${PYTHON_BIN:-/home/yjj/miniconda3/envs/FastTD3/bin/python}"
EXPERIMENT_CONFIG="configs/experiments/${TARGET}_equal_dose_source_calibration_v1.yaml"

case "${SEED}" in 1|2|3) ;; *) echo "SEED must be 1, 2, or 3" >&2; exit 2 ;; esac
case "${ARM}" in
  scratch)
    BANK="configs/source_banks/empty.yaml"
    MCG=0
    ADMISSION_MODE="legacy"
    ;;
  stand|walk|run)
    BANK="configs/source_banks/calibration/h1hand_${TARGET}_rbo_${ARM}.yaml"
    MCG=1
    ADMISSION_MODE="all"
    ;;
  *) echo "unsupported ARM=${ARM}" >&2; exit 2 ;;
esac

if [[ "${SMOKE}" == "1" ]]; then
  RUN_STOP_STEP=200
  CHECKPOINT_STEPS=200
  EVAL_INTERVAL=0
  WANDB=0
  RUN_KIND=smoke
else
  RUN_STOP_STEP=30000
  CHECKPOINT_STEPS=30000
  EVAL_INTERVAL=5000
  WANDB=1
  if [[ "${SEED}" == "1" ]]; then
    RUN_KIND=seed1_gate
  else
    RUN_KIND=multiseed_confirmation
  fi
fi

ENV_NAME="h1hand-${TARGET}-v0"
EXP_NAME="${TARGET}_equal_dose_rbo_${ARM}_${RUN_KIND}_s${SEED}_${STAMP}"
RUN_NAME="${ENV_NAME}__${EXP_NAME}__${SEED}"
LOG_FILE="${RUN_ROOT}/${ARM}_s${SEED}.log"
META_FILE="${RUN_ROOT}/${ARM}_s${SEED}.meta.json"

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
  "PROJECT=ptf_fasttd3_source_calibration"
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
  "PTF_MCG_WARMUP_STEPS=30000"
  "PTF_MCG_WARMUP_MIN_STEPS=25"
  "PTF_MCG_WARMUP_MODE=admission_bootstrap"
  "PTF_MCG_ABLATION=bootstrap_only"
  "PTF_ADMISSION_MODE=${ADMISSION_MODE}"
  "PTF_ADMISSION_STUDENT_LOGIT=0.0"
  "PTF_ADMISSION_REPLAY_RECENCY_HALF_LIFE=0"
  "PTF_ADMISSION_REPLAY_UNIFORM_MIX=1"
  "PTF_ADMISSION_REPLAY_PRIORITY_ALPHA=0"
  "PTF_ADMISSION_REPLAY_HANDOFF=physical_after_authority"
  "PTF_RUN_STOP_STEP=${RUN_STOP_STEP}"
  "PTF_EVAL_CHECKPOINT_STEPS=${CHECKPOINT_STEPS}"
)
if [[ "${ARM}" != "scratch" ]]; then
  RUN_ENV+=("PTF_ADMISSION_EXPECTED_SOURCE_MASS=0.5")
fi

CMD=(bash scripts/official_fasttd3_train_target_ptf.sh)

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf 'arm=%s seed=%s gpu=%s run=%s\n' "${ARM}" "${SEED}" "${GPU_ID}" "${RUN_NAME}"
  printf 'command:'; printf ' %q' "${RUN_ENV[@]}" "${CMD[@]}"; printf '\n'
  exit 0
fi

mkdir -p "${RUN_ROOT}"
"${PYTHON_BIN}" - "${META_FILE}" <<PY
import hashlib, json, pathlib, subprocess, sys
paths = [
    pathlib.Path("${BANK}"),
    pathlib.Path("${EXPERIMENT_CONFIG}"),
]
payload = {
    "started_at": subprocess.check_output(
        ["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"], text=True
    ).strip(),
    "arm": "${ARM}",
    "seed": ${SEED},
    "gpu_id": "${GPU_ID}",
    "exp_name": "${EXP_NAME}",
    "run_name": "${RUN_NAME}",
    "run_kind": "${RUN_KIND}",
    "run_stop_step": ${RUN_STOP_STEP},
    "source_bank": "${BANK}",
    "admission_mode": "${ADMISSION_MODE}",
    "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "git_dirty": bool(
        subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
    ),
    "inputs": {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths
    },
}
pathlib.Path(sys.argv[1]).write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n"
)
PY

set +e
nice -n 5 "${RUN_ENV[@]}" "${CMD[@]}" > "${LOG_FILE}" 2>&1
status=$?
set -e

MODEL_FILE="models/${RUN_NAME}_${RUN_STOP_STEP}.pt"
FINAL_FILE="models/${RUN_NAME}_final.pt"
"${PYTHON_BIN}" - "${META_FILE}" "${status}" "${MODEL_FILE}" "${FINAL_FILE}" <<'PY'
import json, pathlib, subprocess, sys
path = pathlib.Path(sys.argv[1])
payload = json.loads(path.read_text())
payload.update({
    "finished_at": subprocess.check_output(
        ["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"], text=True
    ).strip(),
    "exit_code": int(sys.argv[2]),
    "completed_step_checkpoint": sys.argv[3],
    "completed_step_checkpoint_exists": pathlib.Path(sys.argv[3]).is_file(),
    "final_checkpoint": sys.argv[4],
    "final_checkpoint_exists": pathlib.Path(sys.argv[4]).is_file(),
})
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY

if [[ "${status}" -eq 0 && ! -f "${MODEL_FILE}" ]]; then
  echo "missing completed-step checkpoint: ${MODEL_FILE}" >&2
  exit 3
fi
echo "${ARM}/s${SEED} exit=${status} log=${LOG_FILE} model=${MODEL_FILE}"
exit "${status}"
