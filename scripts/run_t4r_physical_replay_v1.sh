#!/usr/bin/env bash
# T4-R：Truck 10k→20k 的 physical-replay 臂（q_S = rho_S）。
# 预注册：docs/experiments/t4r_physical_replay_prereg_20260808.md
#
# B-only 臂（T2）与 joint 臂（Gate A）复用已有产物，本脚本只跑缺的中间格。
# 与 Gate A 的 joint 臂逐项相同，唯一差别是 PTF_ADMISSION_REPLAY_MODE=physical：
#   replay 走 physical-uniform over allowed slots，使 q_S 跟随物理占比 rho_S，
#   而不是 fixed provenance quota 的 q_S = m；
#   source 的 behavior authority 完全不变，照常执行动作、照常写 physical buffer。
# 目的：T2 比较的是 q_S=0 与 q_S≈0.5，缺的正是 q_S=rho_S 这一格。
# T3 实测 fixed quota 下 A≈2.95（理论 3.0），本臂应使 A≈1。
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/home/yjj/miniconda3/envs/FastTD3/bin/python}"
GPU="${GPU:?set GPU}"
SEED="${SEED:?set SEED}"
PROJECT="${PROJECT:-ptf_fasttd3_t4r_phys_v1}"
GATE_A_ROOT=artifacts/pare_gate_a_v1
LOGDIR=logs/train/t4r_phys_v1
mkdir -p "${LOGDIR}"

A0="${GATE_A_ROOT}/anchors/truck_s${SEED}_k10000"
[[ -d "${A0}" ]] || { echo "[FATAL] 缺 A0 anchor: ${A0}" >&2; exit 2; }

NAME="t4r_phys_s${SEED}"
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
  PTF_ADMISSION_REPLAY_MODE=physical \
  PTF_ANCHOR_RESUME="${A0}" PTF_RESUME_NOISE_SEED=$((92000 + SEED)) \
  PTF_RUN_STOP_STEP=20000 PTF_EVAL_CHECKPOINT_STEPS=20000 \
  bash scripts/official_fasttd3_train_target_ptf.sh \
  > "${LOGDIR}/${NAME}.log" 2>&1

echo "[$(date -u +%FT%TZ)] TRUCK T4R PHYSICAL-REPLAY SEED ${SEED} COMPLETE"
