#!/usr/bin/env bash
# hurdle 加速实验的 source-free 评估(预注册 §3)。
# 面板与项目全部历史一致:128 deterministic episodes,纯 student(源不在场)。
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/home/yjj/miniconda3/envs/FastTD3/bin/python}"
GPU="${GPU:?set GPU}"
SEEDS="${SEEDS:?set SEEDS}"
ARM="${ARM:?set ARM: scratch|source|stand}"
STEPS_LIST="${STEPS_LIST:-10000 20000 30000 50000 75000 100000}"
if [[ "${ARM}" == "stand" ]]; then
  # hurdle_selection_value_v1 的新 argmin 对照臂；run/scratch 继续复用
  # hurdle_speedup_v1 的已冻结面板。
  EVAL_ROOT=docs/data/hurdle_selection_value_v1/source_free_eval
else
  EVAL_ROOT=docs/data/hurdle_speedup_v1/source_free_eval
fi
mkdir -p "${EVAL_ROOT}"

MODE=legacy
[[ "${ARM}" != "scratch" ]] && MODE=all

for SEED in ${SEEDS}; do
  for ST in ${STEPS_LIST}; do
    CKPT=$(compgen -G "models/*hspd_${ARM}_s${SEED}__*_${ST}.pt" | sort | head -1 || true)
    OUT="${EVAL_ROOT}/${ARM}_s${SEED}_step${ST}.json"
    [[ -f "${OUT}" ]] && { echo "[skip] ${OUT}"; continue; }
    if [[ -z "${CKPT}" ]]; then
      echo "[MISSING] arm=${ARM} seed=${SEED} step=${ST}" >&2
      continue
    fi
    CUDA_VISIBLE_DEVICES="${GPU}" MUJOCO_GL=egl "${PYTHON_BIN}" scripts/p0_evaluator.py \
      --checkpoint "${CKPT}" \
      --env-name h1hand-hurdle-v0 \
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
