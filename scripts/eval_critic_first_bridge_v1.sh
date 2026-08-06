#!/usr/bin/env bash
# Frozen source-free evaluation for docs/run_card_critic_first_bridge_v1.md.
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/home/yjj/miniconda3/envs/FastTD3/bin/python}"
GPU="${GPU:?set GPU}"
TASK="${TASK:?set TASK: slide|door}"
SEED="${SEED:-1}"
EVAL_ROOT=docs/data/critic_first_bridge_v1/source_free_eval
mkdir -p "${EVAL_ROOT}"

for ARM in student_freeze interleaved critic_first; do
  MODE=schedule
  [[ "${ARM}" == "student_freeze" ]] && MODE=legacy
  CKPT=$(find models -maxdepth 1 -type f \
    -name "h1hand-${TASK}-v0__cfb_${TASK}_${ARM}_s${SEED}__${SEED}_20000.pt" \
    -print -quit)
  if [[ -z "${CKPT}" ]]; then
    echo "missing 20k checkpoint: task=${TASK} arm=${ARM} seed=${SEED}" >&2
    exit 1
  fi
  OUT="${EVAL_ROOT}/${TASK}_${ARM}_s${SEED}_step20000.json"
  CUDA_VISIBLE_DEVICES="${GPU}" MUJOCO_GL=egl "${PYTHON_BIN}" scripts/p0_evaluator.py \
    --checkpoint "${CKPT}" \
    --env-name "h1hand-${TASK}-v0" \
    --out "${OUT}" \
    --expect-global-step 20000 \
    --expect-seed "${SEED}" \
    --expect-admission-mode "${MODE}" \
    --eval-seeds panel128 \
    > "${EVAL_ROOT}/${TASK}_${ARM}_s${SEED}.log" 2>&1
  echo "[$(date -u +%FT%TZ)] eval task=${TASK} arm=${ARM} seed=${SEED} DONE"
done
