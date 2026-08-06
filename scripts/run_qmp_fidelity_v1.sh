#!/usr/bin/env bash
# QMP-fidelity behavior-only 验证(run card docs/run_card_qmp_fidelity_v1.md)。
#
# 机制:per-state 完整策略 argmax min_h Q_h(s, π_i(s)),候选 {student,stand,walk,run},
# ties→student,只改行为采集——无蒸馏、无 null margin、无身体组、无锁存,
# replay 保持 uniform,transfer_loss ≡ 0,不更新 option/termination。
#
# 协议与 door@10k / slide BAC gate 的固定源臂**逐位对齐**:同一 10k anchor resume、
# 同一 PTF_RESUME_NOISE_SEED=91000+SEED、同一 stop/eval step。唯一差异是行为模式。
# 因此历史 student/stand/walk/run 臂可直接作为对照,无需重跑。
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/home/yjj/miniconda3/envs/FastTD3/bin/python}"
GPU="${GPU:?set GPU}"
SEEDS="${SEEDS:?set SEEDS, e.g. '1 2 3'}"
TASK="${TASK:?set TASK: door|slide}"
SMOKE="${SMOKE:-0}"          # 1 = 200-step forced-student 隔离等价性检查
PROJECT="${PROJECT:-ptf_fasttd3_source_calibration}"

case "${TASK}" in
  door)  ANCHOR_ROOT=artifacts/door_at10k_gate_v1/anchors ;;
  slide) ANCHOR_ROOT=artifacts/slide_bac_gate_v1/anchors ;;
  *) echo "unsupported TASK=${TASK}" >&2; exit 1 ;;
esac
BANK="configs/source_banks/calibration/h1hand_${TASK}_qmp_loco3.yaml"
LOGDIR="logs/train/qmp_fidelity_v1"
mkdir -p "${LOGDIR}"

common_env() {
  echo PYTHONUNBUFFERED=1 MUJOCO_GL=egl WANDB_INIT_TIMEOUT=300 WANDB_SILENT=true \
    PYTHON_BIN="${PYTHON_BIN}" CUDA_VISIBLE_DEVICES="${GPU}" DEVICE_RANK=0 \
    ENV_NAME="h1hand-${TASK}-v0" PROJECT="${PROJECT}" \
    TOTAL_TIMESTEPS=100000 NUM_ENVS=128 BATCH_SIZE=32768 BUFFER_SIZE=51200 \
    LEARNING_STARTS=10 NUM_UPDATES=2 SAVE_INTERVAL=0 EVAL_INTERVAL=0 RENDER_INTERVAL=0 \
    COMPILE=0 AMP=1
}

for SEED in ${SEEDS}; do
  ANCHOR="${ANCHOR_ROOT}/s${SEED}"
  [[ -f "${ANCHOR}/manifest.json" ]] || { echo "missing anchor ${ANCHOR}" >&2; exit 1; }

  if [[ "${SMOKE}" == "1" ]]; then
    NAME="qmp_smoke_${TASK}_s${SEED}"
    # 200 步 forced-student:验证 QMP 分支除动作选择外无任何副作用
    env $(common_env) SEED="${SEED}" WANDB=0 EXP_NAME="${NAME}" \
      SOURCE_BANK="${BANK}" PTF_QMP=1 PTF_QMP_FORCE_STUDENT=1 \
      PTF_ADMISSION_MODE=legacy \
      PTF_ANCHOR_RESUME="${ANCHOR}" \
      PTF_RUN_STOP_STEP=10200 \
      PTF_RESUME_NOISE_SEED=$((91000 + SEED)) \
      bash scripts/official_fasttd3_train_target_ptf.sh \
      > "${LOGDIR}/${NAME}.log" 2>&1
    echo "[$(date -u +%FT%TZ)] SMOKE ${TASK} seed=${SEED} DONE -> ${LOGDIR}/${NAME}.log"
    continue
  fi

  NAME="qmp_${TASK}_s${SEED}"
  env $(common_env) SEED="${SEED}" WANDB=1 EXP_NAME="${NAME}" \
    SOURCE_BANK="${BANK}" PTF_QMP=1 \
    PTF_ADMISSION_MODE=legacy \
    PTF_ANCHOR_RESUME="${ANCHOR}" \
    PTF_RUN_STOP_STEP=20000 PTF_EVAL_CHECKPOINT_STEPS=20000 \
    PTF_RESUME_NOISE_SEED=$((91000 + SEED)) \
    bash scripts/official_fasttd3_train_target_ptf.sh \
    > "${LOGDIR}/${NAME}.log" 2>&1
  echo "[$(date -u +%FT%TZ)] ARM ${TASK} seed=${SEED} DONE -> ${LOGDIR}/${NAME}.log"
done
echo "TASK=${TASK} SEEDS='${SEEDS}' SMOKE=${SMOKE} COMPLETE"
