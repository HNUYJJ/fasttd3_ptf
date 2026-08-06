#!/usr/bin/env bash
# RACING_REJECT v1（预注册 docs/experiments/racing_reject_door_v1_prereg_20260730.md）。
#
# 逐项复用 scripts/run_door_at10k_gate_v1.sh 的 arms 阶段：同 anchor、
# 同 PTF_RESUME_NOISE_SEED(91000+seed)、同 dose/h/bank/warmup。
# **唯一改动**：PTF_EVAL_CHECKPOINT_STEPS 由 20000 改为 12000,15000,20000，
# 即在 K=2000/5000/10000 处各存一个 checkpoint。
# 因此 K=10000 必须复现 door_at10k_gate_v1 的已发表 ground truth（内建 sanity check）。
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/home/yjj/miniconda3/envs/FastTD3/bin/python}"
GPU="${GPU:?set GPU}"
SEEDS="${SEEDS:?set SEEDS}"
ARMS="${ARMS:-student stand walk run}"
EXP_PREFIX="${EXP_PREFIX:-rjd}"   # 批1=rjd(seeds 1-3), 批2=rjd2(seeds 4-6)
ROOT=artifacts/door_at10k_gate_v1          # 复用现成 anchor，不重建
LOGDIR=logs/train/racing_reject_door_v1
PROJECT="${PROJECT:-ptf_fasttd3_racing_reject}"
mkdir -p "${LOGDIR}"

common_env() {
  echo PYTHONUNBUFFERED=1 MUJOCO_GL=egl WANDB_INIT_TIMEOUT=300 WANDB_SILENT=true \
    PYTHON_BIN="${PYTHON_BIN}" CUDA_VISIBLE_DEVICES="${GPU}" DEVICE_RANK=0 \
    ENV_NAME=h1hand-door-v0 PROJECT="${PROJECT}" \
    TOTAL_TIMESTEPS=100000 NUM_ENVS=128 BATCH_SIZE=32768 BUFFER_SIZE=51200 \
    LEARNING_STARTS=10 NUM_UPDATES=2 SAVE_INTERVAL=0 EVAL_INTERVAL=0 RENDER_INTERVAL=0 \
    COMPILE=0 AMP=1
}

for SEED in ${SEEDS}; do
  if [[ ! -d "${ROOT}/anchors/s${SEED}" ]]; then
    echo "[FATAL] 缺 anchor: ${ROOT}/anchors/s${SEED}" >&2; exit 1
  fi
  for ARM in ${ARMS}; do
    NAME="${EXP_PREFIX}_${ARM}_s${SEED}"
    ARM_ENV=(SEED="${SEED}" WANDB=1 EXP_NAME="${NAME}"
             PTF_ANCHOR_RESUME="${ROOT}/anchors/s${SEED}"
             PTF_RUN_STOP_STEP=20000
             PTF_EVAL_CHECKPOINT_STEPS=12000,15000,20000
             PTF_RESUME_NOISE_SEED=$((91000 + SEED)))
    if [[ "${ARM}" == "student" ]]; then
      ARM_ENV+=(SOURCE_BANK=configs/source_banks/empty.yaml PTF_ADMISSION_MODE=legacy)
    else
      ARM_ENV+=(SOURCE_BANK="configs/source_banks/calibration/h1hand_door_rbo_${ARM}.yaml"
                PTF_MCG=1 PTF_MCG_GROUPS=legs_torso,arms,hands
                PTF_MCG_WARMUP_STEPS=30000 PTF_MCG_WARMUP_MIN_STEPS=25
                PTF_MCG_WARMUP_MODE=admission_bootstrap PTF_MCG_ABLATION=bootstrap_only
                PTF_ADMISSION_MODE=all PTF_ADMISSION_STUDENT_LOGIT=0.0
                PTF_ADMISSION_EXPECTED_SOURCE_MASS=0.5
                PTF_ADMISSION_REPLAY_RECENCY_HALF_LIFE=0
                PTF_ADMISSION_REPLAY_UNIFORM_MIX=1
                PTF_ADMISSION_REPLAY_PRIORITY_ALPHA=0
                PTF_ADMISSION_REPLAY_HANDOFF=physical_after_authority)
    fi
    env $(common_env) "${ARM_ENV[@]}" \
      bash scripts/official_fasttd3_train_target_ptf.sh \
      > "${LOGDIR}/${NAME}.log" 2>&1
    echo "[$(date -u +%FT%TZ)] arm=${ARM} seed=${SEED} DONE"
  done
done
echo "SEEDS='${SEEDS}' ARMS='${ARMS}' COMPLETE"
