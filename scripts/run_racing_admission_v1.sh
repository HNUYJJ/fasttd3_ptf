#!/usr/bin/env bash
# racing 准入能力 v1（预注册 docs/experiments/racing_admission_v1_prereg_20260804.md §5）。
#
# 四臂 × 3 seeds × K=10000，与 run_racing_min_horizon_v1.sh 逐项同构，
# 只把 target 参数化（hurdle 的同协议数据直接复用，不在本脚本内重跑）。
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/home/yjj/miniconda3/envs/FastTD3/bin/python}"
GPU="${GPU:?set GPU}"
TARGET="${TARGET:?set TARGET: crawl|slide}"
ARM="${ARM:?set ARM: student|stand|walk|run}"
SEEDS="${SEEDS:?set SEEDS, e.g. '1 2 3'}"
STEPS="${STEPS:-10000}"             # 实际训练步数（PTF_RUN_STOP_STEP）
LR_HORIZON="${LR_HORIZON:-100000}"  # CosineAnnealingLR 的 T_max = 部署时长训练长度
PROJECT="${PROJECT:-ptf_fasttd3_racing_admission}"
LOGDIR=logs/train/racing_admission_v1
mkdir -p "${LOGDIR}"

case "${TARGET}" in
  crawl|slide) ;;
  *) echo "[REFUSE] TARGET must be crawl|slide (hurdle 复用既有数据)" >&2; exit 2 ;;
esac

common_env() {
  echo PYTHONUNBUFFERED=1 MUJOCO_GL=egl WANDB_INIT_TIMEOUT=300 WANDB_SILENT=true \
    PYTHON_BIN="${PYTHON_BIN}" CUDA_VISIBLE_DEVICES="${GPU}" DEVICE_RANK=0 \
    ENV_NAME="h1hand-${TARGET}-v0" PROJECT="${PROJECT}" \
    TOTAL_TIMESTEPS="${LR_HORIZON}" PTF_RUN_STOP_STEP="${STEPS}" \
    NUM_ENVS=128 BATCH_SIZE=32768 BUFFER_SIZE=51200 \
    LEARNING_STARTS=10 NUM_UPDATES=2 SAVE_INTERVAL=0 EVAL_INTERVAL=0 RENDER_INTERVAL=0 \
    COMPILE=0 AMP=1 WANDB=0 \
    PTF_EVAL_CHECKPOINT_STEPS=2000,5000,10000
}

for SEED in ${SEEDS}; do
  NAME="rad_${TARGET}_${ARM}_s${SEED}"
  if compgen -G "models/*${NAME}__*.pt" >/dev/null || [[ -e "${LOGDIR}/${NAME}.log" ]]; then
    echo "[REFUSE] existing output for ${NAME}" >&2
    exit 2
  fi
  if [[ "${ARM}" == "student" ]]; then
    ARM_ENV=(SOURCE_BANK=configs/source_banks/empty.yaml PTF_ADMISSION_MODE=legacy)
  else
    BANK="configs/source_banks/calibration/h1hand_${TARGET}_rbo_${ARM}.yaml"
    [[ -f "${BANK}" ]] || { echo "[REFUSE] missing bank ${BANK}" >&2; exit 2; }
    ARM_ENV=(SOURCE_BANK="${BANK}"
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
  env $(common_env) SEED="${SEED}" EXP_NAME="${NAME}" "${ARM_ENV[@]}" \
    bash scripts/official_fasttd3_train_target_ptf.sh \
    > "${LOGDIR}/${NAME}.log" 2>&1
  echo "[$(date -u +%FT%TZ)] ${TARGET} ${ARM} seed=${SEED} DONE"
done
echo "TARGET=${TARGET} ARM=${ARM} SEEDS='${SEEDS}' COMPLETE"
