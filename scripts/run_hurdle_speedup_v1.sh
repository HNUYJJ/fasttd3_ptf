#!/usr/bin/env bash
# hurdle 样本效率加速倍数实验(预注册 docs/experiments/hurdle_speedup_v1_prereg_20260730.md)。
#
# 单因素:只改"是否有源"。处理臂与已验证的 EQD30K.hurdle.run(U=+379.66,3 seeds)
# 逐项相同,唯一差别是把干预窗口从 30k 延长到全程 100k。
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/home/yjj/miniconda3/envs/FastTD3/bin/python}"
GPU="${GPU:?set GPU}"
SEEDS="${SEEDS:?set SEEDS, e.g. '1 2 3'}"
ARM="${ARM:?set ARM: scratch|source}"
STEPS="${STEPS:-100000}"
PROJECT="${PROJECT:-ptf_fasttd3_hurdle_speedup}"
LOGDIR=logs/train/hurdle_speedup_v1
mkdir -p "${LOGDIR}"

common_env() {
  echo PYTHONUNBUFFERED=1 MUJOCO_GL=egl WANDB_INIT_TIMEOUT=300 WANDB_SILENT=true \
    PYTHON_BIN="${PYTHON_BIN}" CUDA_VISIBLE_DEVICES="${GPU}" DEVICE_RANK=0 \
    ENV_NAME=h1hand-hurdle-v0 PROJECT="${PROJECT}" \
    TOTAL_TIMESTEPS="${STEPS}" NUM_ENVS=128 BATCH_SIZE=32768 BUFFER_SIZE=51200 \
    LEARNING_STARTS=10 NUM_UPDATES=2 SAVE_INTERVAL=0 EVAL_INTERVAL=0 RENDER_INTERVAL=0 \
    COMPILE=0 AMP=1 \
    PTF_EVAL_CHECKPOINT_STEPS=10000,20000,30000,50000,75000,100000
}

for SEED in ${SEEDS}; do
  NAME="hspd_${ARM}_s${SEED}"
  if [[ "${ARM}" == "scratch" ]]; then
    ARM_ENV=(SOURCE_BANK=configs/source_banks/empty.yaml PTF_ADMISSION_MODE=legacy)
  else
    # ARM=source 沿用 run（向后兼容 hurdle_speedup_v1 的既有数据）；
    # ARM=stand/walk/run 时用同名 bank（hurdle_selection_value_v1 的 argmin 对照臂）。
    SRC="${ARM}"; [[ "${ARM}" == "source" ]] && SRC=run
    ARM_ENV=(SOURCE_BANK="configs/source_banks/calibration/h1hand_hurdle_rbo_${SRC}.yaml"
             PTF_MCG=1 PTF_MCG_GROUPS=legs_torso,arms,hands
             PTF_MCG_WARMUP_STEPS="${STEPS}" PTF_MCG_WARMUP_MIN_STEPS=25
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
