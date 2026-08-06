#!/usr/bin/env bash
# Slide 等剂量 BAC 判决场（预注册 2026-07-28，见
# docs/experiments/bottleneck_aligned_coverage_v1_prereg_20260728.md）。
#
# 定位：这是 Bottleneck-Aligned Coverage 指标的**前瞻判决实验**。
# 之所以选 slide，是因为两个指标在此做出极端相反的预测：
#     zero-shot return   stand 88.5(最高) > walk 45.7 > run 27.8
#     NET(BAC)           stand 0.0129    ≪ walk 0.5153 ≈ run 0.5086   （差 40 倍）
# 机制透明：slide 是纯乘性 stand_reward × small_control × move；
# stand 的 move 仅 0.188（几乎不动），靠不摔活满 episode 刷出高 return，
# 而 walk/run 的 move 达 0.82/0.84，每步 reward 是 stand 的 3 倍。
#
# 被估量与 door/cabinet 系列一致：
#     U_i(10k,10k) = J_sf@20k(source_i) - J_sf@20k(student)
#
# 冻结协议（与 Door gate 逐项一致，只换 target 与 bank）：每 seed 一个 10k
# exact-abstention 纯 student anchor，分叉 stand/walk/run/student 四臂，
# teacher/student=0.5/0.5、h=25、bootstrap_only，训到 20k，
# 128 deterministic episodes 冻结 source-free 面板。
#
# 三级裁决已在预注册文档冻结，失败后不得调 BOTTLENECK_MASS / SIGN_EPS /
# SEPARATION_MIN 或更换瓶颈定义来抢救。
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/home/yjj/miniconda3/envs/FastTD3/bin/python}"
GPU="${GPU:?set GPU}"
SEEDS="${SEEDS:?set SEEDS, e.g. '1 2'}"
STAGE="${STAGE:?set STAGE: anchor|arms}"
ROOT=artifacts/stair_bac_gate_v1
LOGDIR=logs/train/stair_bac_gate_v1
PROJECT="${PROJECT:-ptf_fasttd3_source_calibration}"
mkdir -p "${ROOT}/anchors" "${LOGDIR}"

common_env() {
  echo PYTHONUNBUFFERED=1 MUJOCO_GL=egl WANDB_INIT_TIMEOUT=300 WANDB_SILENT=true \
    PYTHON_BIN="${PYTHON_BIN}" CUDA_VISIBLE_DEVICES="${GPU}" DEVICE_RANK=0 \
    ENV_NAME=h1hand-stair-v0 PROJECT="${PROJECT}" \
    TOTAL_TIMESTEPS=100000 NUM_ENVS=128 BATCH_SIZE=32768 BUFFER_SIZE=51200 \
    LEARNING_STARTS=10 NUM_UPDATES=2 SAVE_INTERVAL=0 EVAL_INTERVAL=0 RENDER_INTERVAL=0 \
    COMPILE=0 AMP=1
}

for SEED in ${SEEDS}; do
  if [[ "${STAGE}" == "anchor" ]]; then
    # 10k pure-student exact-abstention anchor (empty bank => target-only path).
    env $(common_env) SEED="${SEED}" WANDB=0 \
      EXP_NAME="stair_bac_anchor_s${SEED}" \
      SOURCE_BANK=configs/source_banks/empty.yaml \
      PTF_ANCHOR_STEP=10000 PTF_ANCHOR_DIR="${ROOT}/anchors/s${SEED}" \
      PTF_RUN_STOP_STEP=10000 \
      bash scripts/official_fasttd3_train_target_ptf.sh \
      > "${LOGDIR}/anchor_s${SEED}.log" 2>&1
    echo "[$(date -u +%FT%TZ)] anchor seed=${SEED} DONE"
    continue
  fi

  for ARM in ${ARMS:-student stand walk run}; do
    NAME="stair_bac_${ARM}_s${SEED}"
    # Paired noise reseed: identical across the four arms of one seed, so the
    # arms differ only by source identity.
    ARM_ENV=(SEED="${SEED}" WANDB=1 EXP_NAME="${NAME}"
             PTF_ANCHOR_RESUME="${ROOT}/anchors/s${SEED}"
             PTF_RUN_STOP_STEP=20000 PTF_EVAL_CHECKPOINT_STEPS=20000
             PTF_RESUME_NOISE_SEED=$((91000 + SEED)))
    if [[ "${ARM}" == "student" ]]; then
      ARM_ENV+=(SOURCE_BANK=configs/source_banks/empty.yaml PTF_ADMISSION_MODE=legacy)
    else
      ARM_ENV+=(SOURCE_BANK="configs/source_banks/calibration/h1hand_stair_rbo_${ARM}.yaml"
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
