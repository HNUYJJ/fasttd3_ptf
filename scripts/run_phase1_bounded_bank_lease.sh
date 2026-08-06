#!/usr/bin/env bash
set -euo pipefail

CELL="${CELL:?set CELL to basketball_hard_exit, truck_retention, or truck_hard_exit}"
SEED="${SEED:-1}"
GPU_ID="${GPU_ID:-0}"
STAMP="${STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
SMOKE="${SMOKE:-0}"
PYTHON_BIN="${PYTHON_BIN:-/home/yjj/miniconda3/envs/FastTD3/bin/python}"

case "${SEED}" in 1|2|3) ;; *) echo "SEED must be 1, 2, or 3" >&2; exit 2 ;; esac
case "${CELL}" in
  basketball_hard_exit)
    TASK="basketball"
    BANK="configs/source_banks/h1hand_std9_wfix_basketball.yaml"
    SCHEDULE="configs/admission_schedules/phase1_basketball_hard_exit.yaml"
    STUDENT_LOGIT="3.5892126423877646"
    ;;
  truck_retention)
    TASK="truck"
    BANK="configs/source_banks/h1hand_hurdle4_wfix_truck.yaml"
    SCHEDULE="configs/admission_schedules/phase1_truck_retention.yaml"
    STUDENT_LOGIT="14.216676716804526"
    ;;
  truck_hard_exit)
    TASK="truck"
    BANK="configs/source_banks/h1hand_hurdle4_wfix_truck.yaml"
    SCHEDULE="configs/admission_schedules/phase1_truck_hard_exit.yaml"
    STUDENT_LOGIT="14.216676716804526"
    ;;
  *) echo "unsupported CELL=${CELL}" >&2; exit 2 ;;
esac

if [[ "${SMOKE}" == "1" ]]; then
  RUN_STOP_STEP=200
  CHECKPOINT_STEPS=200
  RUN_KIND="smoke"
  LOG_ROOT="logs/train/phase1_bounded_bank_lease_smoke_${STAMP}"
else
  RUN_STOP_STEP=100000
  CHECKPOINT_STEPS="30000,60000,80000,90000,100000"
  RUN_KIND="formal"
  LOG_ROOT="logs/train/phase1_bounded_bank_lease_${STAMP}"
fi

ENV_NAME="h1hand-${TASK}-v0"
EXP_NAME="phase1_bounded_lease_${RUN_KIND}_${CELL}_s${SEED}_${STAMP}"
RUN_NAME="${ENV_NAME}__${EXP_NAME}__${SEED}"
LOG_FILE="${LOG_ROOT}/${CELL}_s${SEED}.log"
META_FILE="${LOG_ROOT}/${CELL}_s${SEED}.meta.json"

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
  "SAVE_INTERVAL=0"
  "EVAL_INTERVAL=5000"
  "RENDER_INTERVAL=0"
  "COMPILE=0"
  "AMP=1"
  "WANDB=1"
  "PTF_MCG=1"
  "PTF_MCG_GROUPS=legs_torso,arms"
  "PTF_MCG_WARMUP_STEPS=30000"
  "PTF_MCG_WARMUP_MIN_STEPS=25"
  "PTF_MCG_WARMUP_MODE=admission_bootstrap"
  "PTF_MCG_ABLATION=bootstrap_only"
  "PTF_ADMISSION_MODE=schedule"
  "PTF_ADMISSION_SCHEDULE=${SCHEDULE}"
  "PTF_ADMISSION_STUDENT_LOGIT=${STUDENT_LOGIT}"
  "PTF_ADMISSION_EXPECTED_SOURCE_MASS=0.5"
  "PTF_ADMISSION_REPLAY_RECENCY_HALF_LIFE=0"
  "PTF_ADMISSION_REPLAY_UNIFORM_MIX=1"
  "PTF_ADMISSION_REPLAY_PRIORITY_ALPHA=0"
  "PTF_ADMISSION_REPLAY_HANDOFF=physical_after_authority"
  "PTF_RUN_STOP_STEP=${RUN_STOP_STEP}"
  "PTF_EVAL_CHECKPOINT_STEPS=${CHECKPOINT_STEPS}"
)
CMD=(bash scripts/official_fasttd3_train_target_ptf.sh)

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf 'cell=%s seed=%s gpu=%s run=%s\n' "${CELL}" "${SEED}" "${GPU_ID}" "${RUN_NAME}"
  printf 'command:'; printf ' %q' "${RUN_ENV[@]}" "${CMD[@]}"; printf '\n'
  exit 0
fi

mkdir -p "${LOG_ROOT}"
python - "${META_FILE}" <<PY
import hashlib, json, pathlib, subprocess, sys
paths = [pathlib.Path(p) for p in ["${BANK}", "${SCHEDULE}", "configs/experiments/phase1_bounded_bank_lease.yaml"]]
payload = {
    "started_at": subprocess.check_output(["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"], text=True).strip(),
    "cell": "${CELL}", "task": "${TASK}", "arm": "${CELL#*_}", "seed": ${SEED},
    "gpu_id": "${GPU_ID}", "exp_name": "${EXP_NAME}", "run_name": "${RUN_NAME}",
    "run_kind": "${RUN_KIND}", "run_stop_step": ${RUN_STOP_STEP},
    "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()),
    "inputs": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
}
pathlib.Path(sys.argv[1]).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY

set +e
nice -n 5 "${RUN_ENV[@]}" "${CMD[@]}" > "${LOG_FILE}" 2>&1
status=$?
set -e

MODEL_FILE="models/${RUN_NAME}_${RUN_STOP_STEP}.pt"
FINAL_FILE="models/${RUN_NAME}_final.pt"
python - "${META_FILE}" "${status}" "${MODEL_FILE}" "${FINAL_FILE}" <<'PY'
import json, pathlib, subprocess, sys
path = pathlib.Path(sys.argv[1]); payload = json.loads(path.read_text())
payload.update({
    "finished_at": subprocess.check_output(["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"], text=True).strip(),
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
echo "${CELL}/s${SEED} exit=${status} log=${LOG_FILE} model=${MODEL_FILE}"
exit "${status}"
