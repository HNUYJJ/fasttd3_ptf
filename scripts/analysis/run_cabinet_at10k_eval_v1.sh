#!/usr/bin/env bash
# Frozen source-free evaluation of every Cabinet@10k gate arm at step 20000.
set -euo pipefail
PYTHON_BIN="${PYTHON_BIN:-/home/yjj/miniconda3/envs/FastTD3/bin/python}"
GPU="${GPU:?set GPU}"; ARMS="${ARMS:-student stand walk run}"; SEEDS="${SEEDS:?set SEEDS}"
OUT=docs/data/cabinet_at10k_gate_v1/source_free_eval
mkdir -p "${OUT}"
for SEED in ${SEEDS}; do for ARM in ${ARMS}; do
  CKPT=$(ls models/*cabinet_at10k_${ARM}_s${SEED}*_20000.pt 2>/dev/null | head -1)
  [ -z "${CKPT}" ] && { echo "MISSING checkpoint arm=${ARM} seed=${SEED}"; exit 1; }
  MUJOCO_GL=egl CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" scripts/p0_evaluator.py \
    --checkpoint "${CKPT}" --env-name h1hand-cabinet-v0 \
    --out "${OUT}/${ARM}_s${SEED}_step20000.json" 2>&1 | grep -E "p0_evaluator" || true
done; done
echo "EVAL_DONE arms='${ARMS}' seeds='${SEEDS}'"
