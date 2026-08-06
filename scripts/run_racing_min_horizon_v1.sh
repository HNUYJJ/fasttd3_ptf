#!/usr/bin/env bash
# RACING_K v1：自动选源的最小测量代价（预注册 docs/experiments/racing_min_horizon_v1_prereg_20260730.md）。
#
# 四臂 × 3 seeds × 10k，单条训练内在 2k/5k/10k 存 checkpoint。
# 源臂的配置与 EQD30K 逐项相同（剂量 0.5/0.5，horizon 25），只把 K 缩短到 10k。
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/home/yjj/miniconda3/envs/FastTD3/bin/python}"
GPU="${GPU:?set GPU}"
SEEDS="${SEEDS:?set SEEDS, e.g. '1 2 3'}"
ARM="${ARM:?set ARM: student|run|walk|stand}"
STEPS="${STEPS:-10000}"            # 实际训练步数(通过 run_stop_step 控制)
LR_HORIZON="${LR_HORIZON:-100000}" # CosineAnnealingLR 的 T_max,必须等于部署时的长训练长度
EXP_PREFIX="${EXP_PREFIX:-rck}"
PROJECT="${PROJECT:-ptf_fasttd3_racing_k}"
LOGDIR=logs/train/racing_min_horizon_v1
mkdir -p "${LOGDIR}"

common_env() {
  echo PYTHONUNBUFFERED=1 MUJOCO_GL=egl WANDB_INIT_TIMEOUT=300 WANDB_SILENT=true \
    PYTHON_BIN="${PYTHON_BIN}" CUDA_VISIBLE_DEVICES="${GPU}" DEVICE_RANK=0 \
    ENV_NAME=h1hand-hurdle-v0 PROJECT="${PROJECT}" \
    TOTAL_TIMESTEPS="${LR_HORIZON}" PTF_RUN_STOP_STEP="${STEPS}" NUM_ENVS=128 BATCH_SIZE=32768 BUFFER_SIZE=51200 \
    LEARNING_STARTS=10 NUM_UPDATES=2 SAVE_INTERVAL=0 EVAL_INTERVAL=0 RENDER_INTERVAL=0 \
    COMPILE=0 AMP=1 \
    PTF_EVAL_CHECKPOINT_STEPS=2000,5000,10000
}

for SEED in ${SEEDS}; do
  NAME="${EXP_PREFIX}_${ARM}_s${SEED}"
  if [[ "${ARM}" == "student" ]]; then
    ARM_ENV=(SOURCE_BANK=configs/source_banks/empty.yaml PTF_ADMISSION_MODE=legacy)
  else
    ARM_ENV=(SOURCE_BANK="configs/source_banks/calibration/h1hand_hurdle_rbo_${ARM}.yaml"
             PTF_MCG=1 PTF_MCG_GROUPS=legs_torso,arms,hands
             PTF_MCG_WARMUP_STEPS="${LR_HORIZON}" PTF_MCG_WARMUP_MIN_STEPS=25
             PTF_MCG_WARMUP_MODE=admission_bootstrap PTF_MCG_ABLATION=bootstrap_only
             PTF_ADMISSION_MODE=all PTF_ADMISSION_STUDENT_LOGIT=0.0
             PTF_ADMISSION_EXPECTED_SOURCE_MASS=0.5
             PTF_ADMISSION_REPLAY_RECENCY_HALF_LIFE=0
             PTF_ADMISSION_REPLAY_UNIFORM_MIX=1
             PTF_ADMISSION_REPLAY_PRIORITY_ALPHA=0
             PTF_ADMISSION_REPLAY_HANDOFF=physical_after_authority)
  fi
  env $(common_env) SEED="${SEED}" WANDB=1 EXP_NAME="${NAME}" "${ARM_ENV[@]}" \
    bash scripts/official_fasttd3_train_target_ptf.sh \
    > "${LOGDIR}/${NAME}.log" 2>&1
  echo "[$(date -u +%FT%TZ)] ARM=${ARM} seed=${SEED} DONE -> ${LOGDIR}/${NAME}.log"
done
echo "ARM=${ARM} SEEDS='${SEEDS}' COMPLETE"
