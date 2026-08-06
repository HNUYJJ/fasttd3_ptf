#!/usr/bin/env bash
# Slide hard-exit v1: one 0->30k walk prefix, then two matched continuations.
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/home/yjj/miniconda3/envs/FastTD3/bin/python}"
GPU="${GPU:?set GPU}"
SEED="${SEED:?set SEED}"
PROJECT="${PROJECT:-ptf_fasttd3_slide_hard_exit_v1}"
ROOT=artifacts/slide_hard_exit_v1
LOGDIR=logs/train/slide_hard_exit_v1
ANCHOR="${ROOT}/anchors/slide_s${SEED}_walk_k30000"
mkdir -p "${ROOT}/anchors" "${LOGDIR}"

if [[ -e "${ANCHOR}" ]]; then
  echo "[REFUSE] branch anchor already exists: ${ANCHOR}" >&2
  exit 2
fi
for arm in prefix cont exit; do
  name="shev1_${arm}_s${SEED}"
  if compgen -G "models/*${name}__*.pt" >/dev/null || [[ -e "${LOGDIR}/${name}.log" ]]; then
    echo "[REFUSE] existing output for ${name}" >&2
    exit 2
  fi
done

common=(
  PYTHONUNBUFFERED=1 MUJOCO_GL=egl WANDB_INIT_TIMEOUT=300 WANDB_SILENT=true
  PYTHON_BIN="${PYTHON_BIN}" CUDA_VISIBLE_DEVICES="${GPU}" DEVICE_RANK=0
  ENV_NAME=h1hand-slide-v0 PROJECT="${PROJECT}" SEED="${SEED}"
  TOTAL_TIMESTEPS=100000 NUM_ENVS=128 BATCH_SIZE=32768 BUFFER_SIZE=51200
  LEARNING_STARTS=10 NUM_UPDATES=2 SAVE_INTERVAL=0 EVAL_INTERVAL=0
  RENDER_INTERVAL=0 COMPILE=0 AMP=1 WANDB=0
  SOURCE_BANK=configs/source_banks/calibration/h1hand_slide_rbo_walk.yaml
  PTF_MCG=1 PTF_MCG_GROUPS=legs_torso,arms,hands
  PTF_MCG_WARMUP_STEPS=100000 PTF_MCG_WARMUP_MIN_STEPS=25
  PTF_MCG_WARMUP_MODE=admission_bootstrap PTF_MCG_ABLATION=bootstrap_only
  PTF_ADMISSION_STUDENT_LOGIT=0.0
  PTF_ADMISSION_REPLAY_RECENCY_HALF_LIFE=0
  PTF_ADMISSION_REPLAY_UNIFORM_MIX=1
  PTF_ADMISSION_REPLAY_PRIORITY_ALPHA=0
  PTF_ADMISSION_REPLAY_HANDOFF=physical_after_authority
)

run_arm() {
  local name="$1"; shift
  env "${common[@]}" EXP_NAME="${name}" "$@" \
    bash scripts/official_fasttd3_train_target_ptf.sh \
    > "${LOGDIR}/${name}.log" 2>&1
  echo "[$(date -u +%FT%TZ)] ${name} DONE"
}

# The prefix is run once. Both continuations consume this immutable bundle.
run_arm "shev1_prefix_s${SEED}" \
  PTF_ADMISSION_MODE=all PTF_ADMISSION_EXPECTED_SOURCE_MASS=0.5 \
  PTF_RUN_STOP_STEP=30000 PTF_EVAL_CHECKPOINT_STEPS=30000 \
  PTF_BRANCH_ANCHOR_STEP=30000 PTF_BRANCH_ANCHOR_DIR="${ANCHOR}"

RESUME_NOISE_SEED=$((880000 + SEED))
run_arm "shev1_cont_s${SEED}" \
  PTF_ADMISSION_MODE=all PTF_ADMISSION_EXPECTED_SOURCE_MASS=0.5 \
  PTF_ANCHOR_RESUME="${ANCHOR}" PTF_RESUME_NOISE_SEED="${RESUME_NOISE_SEED}" \
  PTF_RUN_STOP_STEP=100000 PTF_EVAL_CHECKPOINT_STEPS=50000,75000,100000

run_arm "shev1_exit_s${SEED}" \
  PTF_ADMISSION_MODE=none PTF_ADMISSION_EXPECTED_SOURCE_MASS=0.0 \
  PTF_ANCHOR_RESUME="${ANCHOR}" PTF_RESUME_NOISE_SEED="${RESUME_NOISE_SEED}" \
  PTF_RUN_STOP_STEP=100000 PTF_EVAL_CHECKPOINT_STEPS=50000,75000,100000

echo "SLIDE HARD-EXIT SEED ${SEED} COMPLETE"
