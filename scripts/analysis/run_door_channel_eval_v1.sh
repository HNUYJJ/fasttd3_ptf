#!/usr/bin/env bash
# B-only 臂的冻结 source-free 评估（与 joint/student 完全同一 128-episode 面板）。
set -euo pipefail
PYTHON_BIN="${PYTHON_BIN:-/home/yjj/miniconda3/envs/FastTD3/bin/python}"
GPU="${GPU:?set GPU}"; SEEDS="${SEEDS:?set SEEDS}"; SOURCE="${SOURCE:-run}"
OUT=docs/data/door_channel_decomposition_v1/source_free_eval
mkdir -p "${OUT}"
for SEED in ${SEEDS}; do
  CKPT=$(ls models/*door_Bonly_${SOURCE}_s${SEED}*_20000.pt 2>/dev/null | head -1)
  [ -z "${CKPT}" ] && { echo "MISSING checkpoint seed=${SEED}"; exit 1; }
  MUJOCO_GL=egl CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" scripts/p0_evaluator.py \
    --checkpoint "${CKPT}" --env-name h1hand-door-v0 --eval-seeds panel128 \
    --out "${OUT}/${SOURCE}_s${SEED}_step20000.json" 2>&1 | grep -E "p0_evaluator" || true
done
echo "EVAL_DONE seeds='${SEEDS}' panel=128"
