#!/usr/bin/env bash
# RACING_K v1 的 source-free 评估（预注册 §4）。128 deterministic episodes，纯 student。
#
# 注意：并行调用时各条链的 (ARM, SEEDS, STEPS_LIST) 必须互斥——
# 本脚本的 `-f && skip` 只在启动瞬间检查，不是原子锁
# （hurdle_speedup_v1 曾因步数集合重叠导致同一文件被两进程并发写）。
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/home/yjj/miniconda3/envs/FastTD3/bin/python}"
GPU="${GPU:?set GPU}"
SEEDS="${SEEDS:?set SEEDS}"
ARM="${ARM:?set ARM: student|run|walk|stand}"
STEPS_LIST="${STEPS_LIST:-2000 5000 10000}"
EXP_PREFIX="${EXP_PREFIX:-rck}"
EVAL_ROOT="${EVAL_ROOT:-docs/data/racing_min_horizon_v1/source_free_eval}"
mkdir -p "${EVAL_ROOT}"

MODE=all
[[ "${ARM}" == "student" ]] && MODE=legacy

for SEED in ${SEEDS}; do
  for ST in ${STEPS_LIST}; do
    CKPT=$(ls -1 models/*${EXP_PREFIX}_${ARM}_s${SEED}__*_${ST}.pt 2>/dev/null | head -1)
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
