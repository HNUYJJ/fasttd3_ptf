#!/usr/bin/env bash
# T2：Truck 10k→20k 的 B-only 臂（source behavior 在，source replay 不参与学习）。
# 预注册：docs/experiments/truck_channel_decomposition_prereg_20260808.md
#
# scratch 臂与 joint 臂复用 Gate A 已有产物，本脚本**只跑缺的第三臂**。
# 与 Gate A 的 joint 臂逐项相同，唯一差别是 PTF_ADMISSION_REPLAY_MODE=student_only：
#   critic 只采 student provenance 的 slot（source 配额恒 0），
#   而 source 仍照常获得 behavior authority、照常写入 physical buffer
#   （train_ptf.py:653-677 的 replay_candidate_masses）。
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/home/yjj/miniconda3/envs/FastTD3/bin/python}"
GPU="${GPU:?set GPU}"
SEED="${SEED:?set SEED}"
PROJECT="${PROJECT:-ptf_fasttd3_truck_channel_v1}"
GATE_A_ROOT=artifacts/pare_gate_a_v1
LOGDIR=logs/train/truck_channel_v1
mkdir -p "${LOGDIR}"

A0="${GATE_A_ROOT}/anchors/truck_s${SEED}_k10000"
[[ -d "${A0}" ]] || { echo "[FATAL] 缺 A0 anchor: ${A0}" >&2; exit 2; }

NAME="tchan_bonly_s${SEED}"
if compgen -G "models/*${NAME}__*.pt" >/dev/null || [[ -e "${LOGDIR}/${NAME}.log" ]]; then
  echo "[REFUSE] existing output for ${NAME}" >&2; exit 2
fi

# 与 run_pare_gate_a_v1.sh 的 truck scaffold 臂逐项一致（同 bank、同 student_logit
# 给出 mass 0.5、同 2 组、同 horizon、同 noise seed），只加 replay mode。
env \
  PYTHONUNBUFFERED=1 MUJOCO_GL=egl WANDB_INIT_TIMEOUT=300 WANDB_SILENT=true \
  PYTHON_BIN="${PYTHON_BIN}" CUDA_VISIBLE_DEVICES="${GPU}" DEVICE_RANK=0 \
  ENV_NAME=h1hand-truck-v0 PROJECT="${PROJECT}" SEED="${SEED}" \
  EXP_NAME="${NAME}" \
  TOTAL_TIMESTEPS=100000 NUM_ENVS=128 BATCH_SIZE=32768 BUFFER_SIZE=51200 \
  LEARNING_STARTS=10 NUM_UPDATES=2 SAVE_INTERVAL=0 EVAL_INTERVAL=0 \
  RENDER_INTERVAL=0 COMPILE=0 AMP=1 WANDB=1 \
  SOURCE_BANK=configs/source_banks/h1hand_hurdle4_wfix_truck.yaml \
  PTF_MCG=1 PTF_MCG_GROUPS=legs_torso,arms \
  PTF_MCG_WARMUP_STEPS=100000 PTF_MCG_WARMUP_MIN_STEPS=25 \
  PTF_MCG_WARMUP_MODE=admission_bootstrap PTF_MCG_ABLATION=bootstrap_only \
  PTF_ADMISSION_STUDENT_LOGIT=14.216676716804526 \
  PTF_ADMISSION_EXPECTED_SOURCE_MASS=0.5 \
  PTF_ADMISSION_REPLAY_RECENCY_HALF_LIFE=0 \
  PTF_ADMISSION_REPLAY_UNIFORM_MIX=1 \
  PTF_ADMISSION_REPLAY_PRIORITY_ALPHA=0 \
  PTF_ADMISSION_REPLAY_HANDOFF=physical_after_authority \
  PTF_ANCHOR_PROVENANCE_GROUPS=2 \
  PTF_ADMISSION_MODE=all \
  PTF_ADMISSION_REPLAY_MODE=student_only \
  PTF_ANCHOR_RESUME="${A0}" PTF_RESUME_NOISE_SEED=$((92000 + SEED)) \
  PTF_RUN_STOP_STEP=20000 PTF_EVAL_CHECKPOINT_STEPS=20000 \
  PTF_BRANCH_ANCHOR_STEP=20000 \
  PTF_BRANCH_ANCHOR_DIR="${GATE_A_ROOT}/anchors/truck_s${SEED}_bonly_k20000" \
  bash scripts/official_fasttd3_train_target_ptf.sh \
  > "${LOGDIR}/${NAME}.log" 2>&1

echo "[$(date -u +%FT%TZ)] TRUCK CHANNEL B-ONLY SEED ${SEED} COMPLETE"
