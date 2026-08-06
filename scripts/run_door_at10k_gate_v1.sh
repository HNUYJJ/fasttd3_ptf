#!/usr/bin/env bash
# Door@10k standardized equal-dose calibration gate (PI approved 2026-07-27).
#
# 定位(PI 冻结):这是**有历史行为先验的定向 RBO 学习效用标定**,不是盲测也不是
# 外部验证。Transfer Map v1 已记录 door 上的行为层异质性(run 101 vs zero 64,
# +56%;walk 25 且 62% 摔,负迁移;stand 59)。本实验回答的是一个不同的问题:
# 该行为层异质性是否转化为 reward-bearing replay 的**学习效用**异质性,即
#     U_i(10k,10k) = J_sf@20k(source_i) - J_sf@20k(student)
#
# 冻结协议(与 Cabinet gate 完全一致,只换 target 与 bank):每 seed 一个 10k
# exact-abstention 纯 student anchor,分叉 stand/walk/run/student 四臂,
# teacher/student=0.5/0.5、h=25、bootstrap_only,训到 20k。
# 唯一的协议升级:评估面板从 32 → 128 deterministic episodes(见 eval 脚本)。
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/home/yjj/miniconda3/envs/FastTD3/bin/python}"
GPU="${GPU:?set GPU}"
SEEDS="${SEEDS:?set SEEDS, e.g. '1 2'}"
STAGE="${STAGE:?set STAGE: anchor|arms}"
ROOT=artifacts/door_at10k_gate_v1
LOGDIR=logs/train/door_at10k_gate_v1
PROJECT="${PROJECT:-ptf_fasttd3_source_calibration}"
mkdir -p "${ROOT}/anchors" "${LOGDIR}"

common_env() {
  echo PYTHONUNBUFFERED=1 MUJOCO_GL=egl WANDB_INIT_TIMEOUT=300 WANDB_SILENT=true \
    PYTHON_BIN="${PYTHON_BIN}" CUDA_VISIBLE_DEVICES="${GPU}" DEVICE_RANK=0 \
    ENV_NAME=h1hand-door-v0 PROJECT="${PROJECT}" \
    TOTAL_TIMESTEPS=100000 NUM_ENVS=128 BATCH_SIZE=32768 BUFFER_SIZE=51200 \
    LEARNING_STARTS=10 NUM_UPDATES=2 SAVE_INTERVAL=0 EVAL_INTERVAL=0 RENDER_INTERVAL=0 \
    COMPILE=0 AMP=1
}

for SEED in ${SEEDS}; do
  if [[ "${STAGE}" == "anchor" ]]; then
    # 10k pure-student exact-abstention anchor (empty bank => target-only path).
    env $(common_env) SEED="${SEED}" WANDB=0 \
      EXP_NAME="door_at10k_anchor_s${SEED}" \
      SOURCE_BANK=configs/source_banks/empty.yaml \
      PTF_ANCHOR_STEP=10000 PTF_ANCHOR_DIR="${ROOT}/anchors/s${SEED}" \
      PTF_RUN_STOP_STEP=10000 \
      bash scripts/official_fasttd3_train_target_ptf.sh \
      > "${LOGDIR}/anchor_s${SEED}.log" 2>&1
    echo "[$(date -u +%FT%TZ)] anchor seed=${SEED} DONE"
    continue
  fi

  for ARM in ${ARMS:-student stand walk run}; do
    NAME="door_at10k_${ARM}_s${SEED}"
    # Paired noise reseed: identical across the four arms of one seed, so the
    # arms differ only by source identity.
    ARM_ENV=(SEED="${SEED}" WANDB=1 EXP_NAME="${NAME}"
             PTF_ANCHOR_RESUME="${ROOT}/anchors/s${SEED}"
             PTF_RUN_STOP_STEP=20000 PTF_EVAL_CHECKPOINT_STEPS=20000
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
echo "STAGE=${STAGE} SEEDS='${SEEDS}' COMPLETE"
