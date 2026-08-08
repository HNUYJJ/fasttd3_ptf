#!/usr/bin/env bash
# PARE 2k smoke（spec §12 第 3 项）。**只验工程正确性，不产出任何科学结论。**
#
# 四项判据，逐条对齐 spec 原文，不换更容易过的代理：
#   S1  provenance 两类都存在   scaffold 段 z=1 与 z=0 都非空
#   S2  D 输出有限              pare/d_loss 与 pare/source_affinity 全程有限
#   S3  actor 梯度有限          actor_grad_norm 有限，且 PARE 分支未抛异常
#   S4  PARE 生效且可关          PARE-on 与 PARE-off 从同一 anchor 出发结果不同
#
# S4 的方向：这里要证的是"PARE 不是空操作"。`pare_runtime is None` 时
# update_pol 逐行走原路径，该等价性由 357 项回归测试覆盖，不在此重复。
#
# 规模刻意压小（32 env / 2k 步），因为它检的是代码路径而非学习效果。
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/home/yjj/miniconda3/envs/FastTD3/bin/python}"
GPU="${GPU:?set GPU}"
ROOT=artifacts/pare_smoke_v1
LOGDIR=logs/train/pare_smoke_v1
ANCHOR="${ROOT}/anchor_k1000"
rm -rf "${ROOT}" "${LOGDIR}"
mkdir -p "${ROOT}" "${LOGDIR}"

common=(
  PYTHONUNBUFFERED=1 MUJOCO_GL=egl WANDB_SILENT=true WANDB=0
  PYTHON_BIN="${PYTHON_BIN}" CUDA_VISIBLE_DEVICES="${GPU}" DEVICE_RANK=0
  ENV_NAME=h1hand-stair-v0 PROJECT=pare_smoke SEED=1
  TOTAL_TIMESTEPS=2000 NUM_ENVS=32 BATCH_SIZE=4096 BUFFER_SIZE=5000
  LEARNING_STARTS=10 NUM_UPDATES=2 SAVE_INTERVAL=0 EVAL_INTERVAL=0
  RENDER_INTERVAL=0 COMPILE=0 AMP=1
)

src=(
  SOURCE_BANK=configs/source_banks/calibration/h1hand_stair_rbo_slidesrc.yaml
  PTF_MCG=1 PTF_MCG_GROUPS=legs_torso,arms,hands
  PTF_MCG_WARMUP_STEPS=2000 PTF_MCG_WARMUP_MIN_STEPS=25
  PTF_MCG_WARMUP_MODE=admission_bootstrap PTF_MCG_ABLATION=bootstrap_only
  PTF_ADMISSION_STUDENT_LOGIT=0.0
  PTF_ADMISSION_REPLAY_RECENCY_HALF_LIFE=0
  PTF_ADMISSION_REPLAY_UNIFORM_MIX=1
  PTF_ADMISSION_REPLAY_PRIORITY_ALPHA=0
  PTF_ADMISSION_REPLAY_HANDOFF=physical_after_authority
  PTF_ANCHOR_PROVENANCE_GROUPS=3
)

run() {
  local name="$1"; shift
  echo "[$(date -u +%FT%TZ)] START ${name}"
  env "${common[@]}" EXP_NAME="${name}" "$@" \
    bash scripts/official_fasttd3_train_target_ptf.sh \
    > "${LOGDIR}/${name}.log" 2>&1
  echo "[$(date -u +%FT%TZ)] DONE  ${name}"
}

# ── scaffold：0→1k，source 有行为权，产出含 z=1 的 release anchor ──────
run "psmoke_scaf" "${src[@]}" \
  PTF_ADMISSION_MODE=all PTF_ADMISSION_EXPECTED_SOURCE_MASS=0.5 \
  PTF_RUN_STOP_STEP=1000 \
  PTF_BRANCH_ANCHOR_STEP=1000 PTF_BRANCH_ANCHOR_DIR="${ANCHOR}"

# ── 两臂：同一 anchor、同一 resume noise，唯一差别是 PTF_PARE ─────────
for arm in off on; do
  extra=()
  [[ "${arm}" == "on" ]] && extra=(PTF_PARE=1)
  run "psmoke_${arm}" "${src[@]}" \
    PTF_ADMISSION_MODE=none PTF_ADMISSION_EXPECTED_SOURCE_MASS=0.0 \
    PTF_ANCHOR_RESUME="${ANCHOR}" PTF_RESUME_NOISE_SEED=777 \
    PTF_RUN_STOP_STEP=2000 PTF_EVAL_CHECKPOINT_STEPS=2000 \
    "${extra[@]}"
done

echo "PARE SMOKE RUNS COMPLETE — 判据由 scripts/analysis/adjudicate_pare_smoke_v1.py 裁定"
