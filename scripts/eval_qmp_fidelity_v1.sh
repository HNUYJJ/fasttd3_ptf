#!/usr/bin/env bash
# QMP-fidelity v1 的冻结 source-free 评估(run card docs/run_card_qmp_fidelity_v1.md §5)。
#
# 面板与 door@10k / slide BAC gate 逐项一致:128 deterministic episodes
# (16 eval seeds × 8 ranks)。评估的是**纯 student**(源在评估时不在场)
# ——这是全项目不变的终评口径,QMP 臂也不例外。
#
# --expect-* 是防串台断言。QMP 臂的 admission_mode 恒为 legacy(启动即断言)。
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/home/yjj/miniconda3/envs/FastTD3/bin/python}"
GPU="${GPU:?set GPU}"
SEEDS="${SEEDS:?set SEEDS, e.g. '1 2 3'}"
TASK="${TASK:?set TASK: door|slide}"
EVAL_ROOT=docs/data/qmp_fidelity_v1/source_free_eval
mkdir -p "${EVAL_ROOT}"

for SEED in ${SEEDS}; do
  CKPT=$(ls -1 models/*qmp_${TASK}_s${SEED}__*_20000.pt 2>/dev/null | head -1)
  if [[ -z "${CKPT}" ]]; then
    echo "[MISSING] task=${TASK} seed=${SEED} 无 20000 步 checkpoint" >&2
    exit 1
  fi
  CUDA_VISIBLE_DEVICES="${GPU}" MUJOCO_GL=egl "${PYTHON_BIN}" scripts/p0_evaluator.py \
    --checkpoint "${CKPT}" \
    --env-name "h1hand-${TASK}-v0" \
    --out "${EVAL_ROOT}/qmp_${TASK}_s${SEED}_step20000.json" \
    --expect-global-step 20000 \
    --expect-seed "${SEED}" \
    --expect-admission-mode legacy \
    --eval-seeds panel128 \
    > "${EVAL_ROOT}/qmp_${TASK}_s${SEED}.log" 2>&1
  echo "[$(date -u +%FT%TZ)] eval task=${TASK} seed=${SEED} DONE"
done
echo "EVAL TASK=${TASK} SEEDS='${SEEDS}' COMPLETE"
