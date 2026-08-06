#!/usr/bin/env bash
# Critic-first bridge v1.  Pre-registered design:
# docs/run_card_critic_first_bridge_v1.md
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/home/yjj/miniconda3/envs/FastTD3/bin/python}"
GPU="${GPU:?set GPU}"
TASK="${TASK:?set TASK: slide|door}"
ARM="${ARM:?set ARM: student_freeze|interleaved|critic_first}"
SEED="${SEED:-1}"
SMOKE="${SMOKE:-0}"

case "${TASK}" in
  slide)
    ENV_NAME=h1hand-slide-v0
    ANCHOR_ROOT=artifacts/slide_bac_gate_v1/anchors
    SOURCE=walk
    BANK=configs/source_banks/calibration/h1hand_slide_rbo_walk.yaml
    SCHEDULE=configs/experiments/critic_first_bridge_slide_walk_schedule_v1.yaml
    ;;
  door)
    ENV_NAME=h1hand-door-v0
    ANCHOR_ROOT=artifacts/door_at10k_gate_v1/anchors
    SOURCE=run
    BANK=configs/source_banks/calibration/h1hand_door_rbo_run.yaml
    SCHEDULE=configs/experiments/critic_first_bridge_door_run_schedule_v1.yaml
    ;;
  *) echo "unsupported TASK=${TASK}" >&2; exit 2 ;;
esac

BRIDGE_END=12000
RUN_STOP=20000
CHECKPOINTS=12000,20000
if [[ "${SMOKE}" == "1" ]]; then
  [[ "${TASK}" == "slide" && "${ARM}" == "critic_first" ]] || {
    echo "SMOKE=1 is restricted to slide critic_first" >&2; exit 2;
  }
  BRIDGE_END=10100
  RUN_STOP=10200
  CHECKPOINTS=10100,10200
  SCHEDULE=configs/experiments/critic_first_bridge_slide_walk_smoke_schedule_v1.yaml
fi

ROOT=artifacts/critic_first_bridge_v1
LOGDIR=logs/train/critic_first_bridge_v1
mkdir -p "${ROOT}" "${LOGDIR}"
NAME="cfb_${TASK}_${ARM}_s${SEED}"
[[ "${SMOKE}" == "1" ]] && NAME="${NAME}_smoke"

COMMON=(
  PYTHONUNBUFFERED=1 MUJOCO_GL=egl WANDB_INIT_TIMEOUT=300 WANDB_SILENT=true
  PYTHON_BIN="${PYTHON_BIN}" CUDA_VISIBLE_DEVICES="${GPU}" DEVICE_RANK=0
  ENV_NAME="${ENV_NAME}" PROJECT=ptf_fasttd3_critic_first_bridge
  TOTAL_TIMESTEPS=100000 NUM_ENVS=128 BATCH_SIZE=32768 BUFFER_SIZE=51200
  LEARNING_STARTS=10 NUM_UPDATES=2 SAVE_INTERVAL=0 EVAL_INTERVAL=0 RENDER_INTERVAL=0
  COMPILE=0 AMP=1 WANDB=0 SEED="${SEED}" EXP_NAME="${NAME}"
  PTF_ANCHOR_RESUME="${ANCHOR_ROOT}/s${SEED}"
  PTF_RESUME_NOISE_SEED="$((91000 + SEED))"
  PTF_RUN_STOP_STEP="${RUN_STOP}" PTF_EVAL_CHECKPOINT_STEPS="${CHECKPOINTS}"
)

ARM_ENV=()
case "${ARM}" in
  student_freeze)
    ARM_ENV+=(SOURCE_BANK=configs/source_banks/empty.yaml PTF_ADMISSION_MODE=legacy)
    ARM_ENV+=(PTF_ACTOR_UPDATE_START_STEP="${BRIDGE_END}")
    ;;
  interleaved|critic_first)
    ARM_ENV+=(
      SOURCE_BANK="${BANK}"
      PTF_MCG=1 PTF_MCG_GROUPS=legs_torso,arms,hands
      PTF_MCG_WARMUP_STEPS="${BRIDGE_END}" PTF_MCG_WARMUP_MIN_STEPS=25
      PTF_MCG_WARMUP_MODE=admission_bootstrap PTF_MCG_ABLATION=bootstrap_only
      PTF_ADMISSION_MODE=schedule PTF_ADMISSION_SCHEDULE="${SCHEDULE}"
      PTF_ADMISSION_STUDENT_LOGIT=0.0 PTF_ADMISSION_EXPECTED_SOURCE_MASS=0.5
      PTF_ADMISSION_REPLAY_RECENCY_HALF_LIFE=0
      PTF_ADMISSION_REPLAY_UNIFORM_MIX=1
      PTF_ADMISSION_REPLAY_PRIORITY_ALPHA=0
      PTF_ADMISSION_REPLAY_HANDOFF=physical_after_authority
    )
    [[ "${ARM}" == "critic_first" ]] && \
      ARM_ENV+=(PTF_ACTOR_UPDATE_START_STEP="${BRIDGE_END}")
    ;;
  *) echo "unsupported ARM=${ARM}" >&2; exit 2 ;;
esac

env "${COMMON[@]}" "${ARM_ENV[@]}" \
  bash scripts/official_fasttd3_train_target_ptf.sh \
  > "${LOGDIR}/${NAME}.log" 2>&1

echo "[$(date -u +%FT%TZ)] task=${TASK} arm=${ARM} seed=${SEED} smoke=${SMOKE} DONE"
