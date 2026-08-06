#!/usr/bin/env bash
set -euo pipefail
PYTHON_BIN="${PYTHON_BIN:-/home/yjj/miniconda3/envs/FastTD3/bin/python}"
GPU="${GPU:?set GPU}"; SEEDS="${SEEDS:?set SEEDS}"; SOURCE="${SOURCE:-run}"
OUT=docs/data/door_prefix_handoff_v1/source_free_eval
mkdir -p "${OUT}"
for SEED in ${SEEDS}; do
  CKPT=$(ls models/*door_prefix_${SOURCE}_s${SEED}__*_20000.pt 2>/dev/null | head -1)
  [ -z "${CKPT}" ] && { echo "MISSING checkpoint seed=${SEED}"; exit 1; }
  MUJOCO_GL=egl CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" scripts/p0_evaluator.py \
    --checkpoint "${CKPT}" --env-name h1hand-door-v0 --eval-seeds panel128 \
    --out "${OUT}/${SOURCE}_s${SEED}_step20000.json" 2>&1 | grep -E "p0_evaluator" || true
done
echo "EVAL_DONE seeds='${SEEDS}' panel=128"
