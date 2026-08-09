#!/usr/bin/env bash
# N1 fresh-seed four-arm gate.  See preregistration:
# docs/experiments/n1_non_displacing_transfer_prereg_20260809.md
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/home/yjj/miniconda3/envs/FastTD3/bin/python}"
GPU="${GPU:?set GPU}"
SEED="${SEED:?set SEED (4..8)}"
[[ "${SEED}" =~ ^[4-8]$ ]] || { echo "[FATAL] SEED must be one of 4..8" >&2; exit 2; }

PROJECT="${PROJECT:-ptf_fasttd3_n1_ndt_v1}"
ROOT=artifacts/n1_non_displacing_transfer_v1
LOGDIR=logs/train/n1_non_displacing_transfer_v1
A0="${ROOT}/anchors/truck_s${SEED}_k10000"
mkdir -p "${ROOT}/anchors" "${LOGDIR}"

for arm in prefix s ff fp lp; do
  name="n1_${arm}_truck_s${SEED}"
  if compgen -G "models/*${name}__*.pt" >/dev/null || [[ -e "${LOGDIR}/${name}.log" ]]; then
    echo "[REFUSE] existing output for ${name}" >&2; exit 2
  fi
done
[[ ! -e "${A0}" ]] || { echo "[REFUSE] anchor exists: ${A0}" >&2; exit 2; }

common=(
  PYTHONUNBUFFERED=1 MUJOCO_GL=egl WANDB_INIT_TIMEOUT=300 WANDB_SILENT=true
  PYTHON_BIN="${PYTHON_BIN}" CUDA_VISIBLE_DEVICES="${GPU}" DEVICE_RANK=0
  ENV_NAME=h1hand-truck-v0 PROJECT="${PROJECT}" SEED="${SEED}"
  TOTAL_TIMESTEPS=100000 NUM_ENVS=128 BATCH_SIZE=32768 BUFFER_SIZE=51200
  LEARNING_STARTS=10 NUM_UPDATES=2 SAVE_INTERVAL=0 EVAL_INTERVAL=0
  RENDER_INTERVAL=0 COMPILE=0 AMP=1 WANDB=1
)

scaffold=(
  SOURCE_BANK=configs/source_banks/h1hand_hurdle4_wfix_truck.yaml
  PTF_MCG=1 PTF_MCG_GROUPS=legs_torso,arms
  PTF_MCG_WARMUP_STEPS=100000 PTF_MCG_WARMUP_MIN_STEPS=25
  PTF_MCG_WARMUP_MODE=admission_bootstrap PTF_MCG_ABLATION=bootstrap_only
  PTF_ADMISSION_STUDENT_LOGIT=14.216676716804526
  PTF_ADMISSION_REPLAY_RECENCY_HALF_LIFE=0
  PTF_ADMISSION_REPLAY_UNIFORM_MIX=1
  PTF_ADMISSION_REPLAY_PRIORITY_ALPHA=0
  PTF_ADMISSION_REPLAY_HANDOFF=physical_after_authority
  PTF_ANCHOR_PROVENANCE_GROUPS=2
)

run_arm() {
  local name="$1"; shift
  echo "[$(date -u +%FT%TZ)] START ${name}"
  env "${common[@]}" EXP_NAME="${name}" "$@" \
    bash scripts/official_fasttd3_train_target_ptf.sh \
    > "${LOGDIR}/${name}.log" 2>&1
  echo "[$(date -u +%FT%TZ)] DONE ${name}"
}

run_arm "n1_prefix_truck_s${SEED}" \
  SOURCE_BANK=configs/source_banks/empty.yaml PTF_ADMISSION_MODE=legacy \
  PTF_ANCHOR_PROVENANCE_GROUPS=2 \
  PTF_ANCHOR_STEP=10000 PTF_ANCHOR_DIR="${A0}" PTF_RUN_STOP_STEP=10000

NOISE=$((95000 + SEED))
branch_common=(
  "${scaffold[@]}"
  PTF_ANCHOR_RESUME="${A0}" PTF_RESUME_NOISE_SEED="${NOISE}"
  PTF_RUN_STOP_STEP=20000 PTF_EVAL_CHECKPOINT_STEPS=20000
)

run_arm "n1_s_truck_s${SEED}" \
  "${branch_common[@]}" PTF_ADMISSION_MODE=none \
  PTF_ADMISSION_EXPECTED_SOURCE_MASS=0.0 \
  PTF_MCG_BEHAVIOR_SOURCE_GROUPS=legs_torso,arms \
  PTF_ADMISSION_REPLAY_MODE=shared

run_arm "n1_ff_truck_s${SEED}" \
  "${branch_common[@]}" PTF_ADMISSION_MODE=all \
  PTF_ADMISSION_EXPECTED_SOURCE_MASS=0.5 \
  PTF_MCG_BEHAVIOR_SOURCE_GROUPS=legs_torso,arms \
  PTF_ADMISSION_REPLAY_MODE=shared

run_arm "n1_fp_truck_s${SEED}" \
  "${branch_common[@]}" PTF_ADMISSION_MODE=all \
  PTF_ADMISSION_EXPECTED_SOURCE_MASS=0.5 \
  PTF_MCG_BEHAVIOR_SOURCE_GROUPS=legs_torso,arms \
  PTF_ADMISSION_REPLAY_MODE=physical

run_arm "n1_lp_truck_s${SEED}" \
  "${branch_common[@]}" PTF_ADMISSION_MODE=all \
  PTF_ADMISSION_EXPECTED_SOURCE_MASS=0.5 \
  PTF_MCG_BEHAVIOR_SOURCE_GROUPS=legs_torso \
  PTF_ADMISSION_REPLAY_MODE=physical

echo "N1 TRUCK SEED ${SEED} COMPLETE"
