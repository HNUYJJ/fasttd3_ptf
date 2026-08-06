#!/usr/bin/env bash
# Slide speedup v1 的结构性 source-free 128-episode 冻结面板。
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/home/yjj/miniconda3/envs/FastTD3/bin/python}"
GPU="${GPU:?set GPU}"
SEEDS="${SEEDS:?set SEEDS}"
ARM="${ARM:?set ARM: scratch|walk}"
STEPS_LIST="${STEPS_LIST:-10000 20000 30000 50000 75000 100000}"
EVAL_ROOT=docs/data/slide_speedup_v1/source_free_eval
mkdir -p "${EVAL_ROOT}"

if [[ "${ARM}" != "scratch" && "${ARM}" != "walk" ]]; then
  echo "ARM must be scratch or walk; got ${ARM}" >&2
  exit 2
fi

MODE=legacy
[[ "${ARM}" == "walk" ]] && MODE=all

for SEED in ${SEEDS}; do
  for ST in ${STEPS_LIST}; do
    mapfile -t CKPTS < <(compgen -G "models/*sspd_${ARM}_s${SEED}__*_${ST}.pt" | sort || true)
    OUT="${EVAL_ROOT}/${ARM}_s${SEED}_step${ST}.json"
    if [[ -e "${OUT}" ]]; then
      echo "[REFUSE] existing eval artifact must be archived before rerun: ${OUT}" >&2
      exit 2
    fi
    if [[ ${#CKPTS[@]} -ne 1 ]]; then
      echo "[INVALID] arm=${ARM} seed=${SEED} step=${ST} checkpoint matches=${#CKPTS[@]}" >&2
      exit 2
    fi
    CKPT="${CKPTS[0]}"
    CUDA_VISIBLE_DEVICES="${GPU}" MUJOCO_GL=egl "${PYTHON_BIN}" scripts/p0_evaluator.py \
      --checkpoint "${CKPT}" \
      --env-name h1hand-slide-v0 \
      --out "${OUT}" \
      --expect-global-step "${ST}" \
      --expect-seed "${SEED}" \
      --expect-admission-mode "${MODE}" \
      --eval-seeds panel128 \
      > "${EVAL_ROOT}/${ARM}_s${SEED}_step${ST}.log" 2>&1
    echo "[$(date -u +%FT%TZ)] eval ${ARM} s${SEED} step=${ST} DONE"
  done
done
echo "EVAL ARM=${ARM} SEEDS='${SEEDS}' COMPLETE"
