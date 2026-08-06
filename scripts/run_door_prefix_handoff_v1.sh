#!/usr/bin/env bash
# Door fixed-horizon option handoff feasibility ablation — Prefix B-only 臂。
# 与 door_channel_decomposition_v1 的 Segment B-only 臂**逐项相同**，
# 唯一差异是 PTF_MCG_EPISODE_PREFIX_STEPS（source 改为 episode 前缀连续执行）。
set -euo pipefail
PYTHON_BIN="${PYTHON_BIN:-/home/yjj/miniconda3/envs/FastTD3/bin/python}"
GPU="${GPU:?set GPU}"; SEEDS="${SEEDS:?set SEEDS}"; SOURCE="${SOURCE:-run}"
PREFIX="${PREFIX:?set PREFIX (episode prefix steps)}"
STOP="${STOP:-20000}"; NAME_SUFFIX="${NAME_SUFFIX:-}"
ANCHOR_ROOT=artifacts/door_at10k_gate_v1/anchors
LOGDIR=logs/train/door_prefix_handoff_v1
PROJECT="${PROJECT:-ptf_fasttd3_source_calibration}"
mkdir -p "${LOGDIR}"
for SEED in ${SEEDS}; do
  NAME="door_prefix_${SOURCE}_s${SEED}${NAME_SUFFIX}"
  env PYTHONUNBUFFERED=1 MUJOCO_GL=egl WANDB_INIT_TIMEOUT=300 WANDB_SILENT=true \
    PYTHON_BIN="${PYTHON_BIN}" CUDA_VISIBLE_DEVICES="${GPU}" DEVICE_RANK=0 \
    ENV_NAME=h1hand-door-v0 PROJECT="${PROJECT}" \
    TOTAL_TIMESTEPS=100000 NUM_ENVS=128 BATCH_SIZE=32768 BUFFER_SIZE=51200 \
    LEARNING_STARTS=10 NUM_UPDATES=2 SAVE_INTERVAL=0 EVAL_INTERVAL=0 RENDER_INTERVAL=0 \
    COMPILE=0 AMP=1 \
    SEED="${SEED}" WANDB="${WANDB:-1}" EXP_NAME="${NAME}" \
    PTF_ANCHOR_RESUME="${ANCHOR_ROOT}/s${SEED}" \
    PTF_RUN_STOP_STEP="${STOP}" PTF_EVAL_CHECKPOINT_STEPS="${STOP}" \
    PTF_RESUME_NOISE_SEED=$((91000 + SEED)) \
    SOURCE_BANK="configs/source_banks/calibration/h1hand_door_rbo_${SOURCE}.yaml" \
    PTF_MCG=1 PTF_MCG_GROUPS=legs_torso,arms,hands \
    PTF_MCG_WARMUP_STEPS=30000 PTF_MCG_WARMUP_MIN_STEPS=25 \
    PTF_MCG_WARMUP_MODE=admission_bootstrap PTF_MCG_ABLATION=bootstrap_only \
    PTF_MCG_EPISODE_PREFIX_STEPS="${PREFIX}" \
    PTF_ADMISSION_MODE=all PTF_ADMISSION_STUDENT_LOGIT=0.0 \
    PTF_ADMISSION_EXPECTED_SOURCE_MASS=0.5 \
    PTF_ADMISSION_REPLAY_MODE=student_only \
    PTF_ADMISSION_REPLAY_RECENCY_HALF_LIFE=0 PTF_ADMISSION_REPLAY_UNIFORM_MIX=1 \
    PTF_ADMISSION_REPLAY_PRIORITY_ALPHA=0 \
    PTF_ADMISSION_REPLAY_HANDOFF=physical_after_authority \
    bash scripts/official_fasttd3_train_target_ptf.sh > "${LOGDIR}/${NAME}.log" 2>&1
  echo "[$(date -u +%FT%TZ)] prefix ${SOURCE} seed=${SEED} H=${PREFIX} DONE"
done
echo "PREFIX COMPLETE seeds='${SEEDS}' H=${PREFIX}"
