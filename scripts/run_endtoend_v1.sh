#!/usr/bin/env bash
# 端到端系统 v1 的增量训练（预注册 docs/experiments/endtoend_v1_prereg_20260804.md §5）。
#
# 只跑 9 条缺失的臂；slide 三臂与 hurdle 的 A/B 臂全部复用既有数据。
#   JOB=hurdle_C   run + 30k 硬退出（两阶段：prefix 存 anchor → 从 anchor 以 none 续训）
#   JOB=crawl_A    scratch（决策为 REJECT，故同时充当端到端臂）
#   JOB=crawl_B    run 全程注入（盲目基线，源由 zero-shot 位移 argmax 选出）
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/home/yjj/miniconda3/envs/FastTD3/bin/python}"
GPU="${GPU:?set GPU}"
JOB="${JOB:?set JOB: hurdle_C|crawl_A|crawl_B}"
SEEDS="${SEEDS:?set SEEDS, e.g. '1 2 3'}"
PROJECT="${PROJECT:-ptf_fasttd3_endtoend_v1}"
ROOT=artifacts/endtoend_v1
LOGDIR=logs/train/endtoend_v1
mkdir -p "${ROOT}/anchors" "${LOGDIR}"

common() {
  local env_name="$1"
  echo PYTHONUNBUFFERED=1 MUJOCO_GL=egl WANDB_INIT_TIMEOUT=300 WANDB_SILENT=true \
    PYTHON_BIN="${PYTHON_BIN}" CUDA_VISIBLE_DEVICES="${GPU}" DEVICE_RANK=0 \
    ENV_NAME="${env_name}" PROJECT="${PROJECT}" \
    TOTAL_TIMESTEPS=100000 NUM_ENVS=128 BATCH_SIZE=32768 BUFFER_SIZE=51200 \
    LEARNING_STARTS=10 NUM_UPDATES=2 SAVE_INTERVAL=0 EVAL_INTERVAL=0 \
    RENDER_INTERVAL=0 COMPILE=0 AMP=1 WANDB=0
}

SRC_ARGS=(PTF_MCG=1 PTF_MCG_GROUPS=legs_torso,arms,hands
          PTF_MCG_WARMUP_STEPS=100000 PTF_MCG_WARMUP_MIN_STEPS=25
          PTF_MCG_WARMUP_MODE=admission_bootstrap PTF_MCG_ABLATION=bootstrap_only
          PTF_ADMISSION_STUDENT_LOGIT=0.0
          PTF_ADMISSION_REPLAY_RECENCY_HALF_LIFE=0
          PTF_ADMISSION_REPLAY_UNIFORM_MIX=1
          PTF_ADMISSION_REPLAY_PRIORITY_ALPHA=0
          PTF_ADMISSION_REPLAY_HANDOFF=physical_after_authority)

run_arm() {
  local name="$1"; shift
  if compgen -G "models/*${name}__*.pt" >/dev/null || [[ -e "${LOGDIR}/${name}.log" ]]; then
    echo "[REFUSE] existing output for ${name}" >&2; exit 2
  fi
  env "$@" EXP_NAME="${name}" bash scripts/official_fasttd3_train_target_ptf.sh \
    > "${LOGDIR}/${name}.log" 2>&1
  echo "[$(date -u +%FT%TZ)] ${name} DONE"
}

for SEED in ${SEEDS}; do
case "${JOB}" in
  hurdle_C)
    ANCHOR="${ROOT}/anchors/hurdle_s${SEED}_run_k30000"
    [[ -e "${ANCHOR}" ]] && { echo "[REFUSE] anchor exists: ${ANCHOR}" >&2; exit 2; }
    # 阶段 1：0→30k 带 run 源，存 branch anchor
    run_arm "e2e_hurdle_prefix_s${SEED}" $(common h1hand-hurdle-v0) SEED="${SEED}" \
      SOURCE_BANK=configs/source_banks/calibration/h1hand_hurdle_rbo_run.yaml \
      "${SRC_ARGS[@]}" PTF_ADMISSION_MODE=all PTF_ADMISSION_EXPECTED_SOURCE_MASS=0.5 \
      PTF_RUN_STOP_STEP=30000 PTF_EVAL_CHECKPOINT_STEPS=30000 \
      PTF_BRANCH_ANCHOR_STEP=30000 PTF_BRANCH_ANCHOR_DIR="${ANCHOR}"
    # 阶段 2：从同一 anchor 硬退出续训至 100k（与 slide_hard_exit_v1 的 exit 臂同构）
    run_arm "e2e_hurdle_exit_s${SEED}" $(common h1hand-hurdle-v0) SEED="${SEED}" \
      SOURCE_BANK=configs/source_banks/calibration/h1hand_hurdle_rbo_run.yaml \
      "${SRC_ARGS[@]}" PTF_ADMISSION_MODE=none PTF_ADMISSION_EXPECTED_SOURCE_MASS=0.0 \
      PTF_ANCHOR_RESUME="${ANCHOR}" PTF_RESUME_NOISE_SEED=$((770000 + SEED)) \
      PTF_RUN_STOP_STEP=100000 PTF_EVAL_CHECKPOINT_STEPS=50000,75000,100000
    ;;
  crawl_A)
    run_arm "e2e_crawl_scratch_s${SEED}" $(common h1hand-crawl-v0) SEED="${SEED}" \
      SOURCE_BANK=configs/source_banks/empty.yaml PTF_ADMISSION_MODE=legacy \
      PTF_EVAL_CHECKPOINT_STEPS=30000,50000,75000,100000
    ;;
  crawl_B)
    run_arm "e2e_crawl_blind_s${SEED}" $(common h1hand-crawl-v0) SEED="${SEED}" \
      SOURCE_BANK=configs/source_banks/calibration/h1hand_crawl_rbo_run.yaml \
      "${SRC_ARGS[@]}" PTF_ADMISSION_MODE=all PTF_ADMISSION_EXPECTED_SOURCE_MASS=0.5 \
      PTF_EVAL_CHECKPOINT_STEPS=30000,50000,75000,100000
    ;;
  *) echo "[REFUSE] unknown JOB=${JOB}" >&2; exit 2 ;;
esac
done
echo "JOB=${JOB} SEEDS='${SEEDS}' COMPLETE"
