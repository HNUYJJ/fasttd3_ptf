#!/usr/bin/env bash
# RACING_REJECT v1 的 source-free 评估（预注册 §4）。128 deterministic episodes。
# 并行时各链的 (ARM, SEEDS) 必须互斥：`-f && skip` 不是原子锁。
set -euo pipefail
PYTHON_BIN="${PYTHON_BIN:-/home/yjj/miniconda3/envs/FastTD3/bin/python}"
GPU="${GPU:?set GPU}"
SEEDS="${SEEDS:?set SEEDS}"
ARM="${ARM:?set ARM: student|stand|walk|run}"
STEPS_LIST="${STEPS_LIST:-12000 15000 20000}"
EXP_PREFIX="${EXP_PREFIX:-rjd}"
EVAL_ROOT="${EVAL_ROOT:-docs/data/racing_reject_door_v1/source_free_eval}"
mkdir -p "${EVAL_ROOT}"
MODE=all
[[ "${ARM}" == "student" ]] && MODE=legacy
for SEED in ${SEEDS}; do
  for ST in ${STEPS_LIST}; do
    # `|| true` 防止 set -e + pipefail 在无匹配时直接终止（否则下面的 MISSING 分支不可达）
    CKPT=$(ls -1 models/*${EXP_PREFIX}_${ARM}_s${SEED}__*_${ST}.pt 2>/dev/null | head -1 || true)
    OUT="${EVAL_ROOT}/${ARM}_s${SEED}_step${ST}.json"
    [[ -f "${OUT}" ]] && { echo "[skip] ${OUT}"; continue; }
    [[ -z "${CKPT}" ]] && { echo "[MISSING] ${ARM} s${SEED} ${ST}" >&2; continue; }
    CUDA_VISIBLE_DEVICES="${GPU}" MUJOCO_GL=egl "${PYTHON_BIN}" scripts/p0_evaluator.py \
      --checkpoint "${CKPT}" --env-name h1hand-door-v0 --out "${OUT}" \
      --expect-global-step "${ST}" --expect-seed "${SEED}" \
      --expect-admission-mode "${MODE}" --eval-seeds panel128 \
      > "${EVAL_ROOT}/${ARM}_s${SEED}_step${ST}.log" 2>&1
    echo "[$(date -u +%FT%TZ)] eval ${ARM} s${SEED} step=${ST} DONE"
  done
done
echo "EVAL ARM=${ARM} SEEDS='${SEEDS}' COMPLETE"
