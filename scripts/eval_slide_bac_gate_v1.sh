#!/usr/bin/env bash
# Slide BAC 判决场的冻结 source-free 评估（预注册协议）。
#
# 面板与 door/cabinet 系列逐项一致：128 deterministic episodes
# （16 eval seeds × 8 ranks），前 32 与既往所有 32-episode 面板逐位兼容。
# 评估的是**纯 student**（源在评估时不在场）——这是全项目不变的终评口径。
#
# --expect-* 三项是防串台断言：checkpoint 的 global_step / seed / admission_mode
# 必须与臂的身份一致，不一致立即失败而不是静默产出一份错面板。
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/home/yjj/miniconda3/envs/FastTD3/bin/python}"
GPU="${GPU:?set GPU}"
SEEDS="${SEEDS:?set SEEDS, e.g. '1 2 3'}"
EVAL_ROOT="${EVAL_ROOT:-docs/data/slide_bac_gate_v1/source_free_eval}"
mkdir -p "${EVAL_ROOT}"

for SEED in ${SEEDS}; do
  for ARM in ${ARMS:-student stand walk run}; do
    MODE=all
    [[ "${ARM}" == "student" ]] && MODE=legacy
    CKPT=$(ls -1 models/*slide_bac_${ARM}_s${SEED}__*_20000.pt 2>/dev/null | head -1)
    if [[ -z "${CKPT}" ]]; then
      echo "[MISSING] arm=${ARM} seed=${SEED} 无 20000 步 checkpoint" >&2
      exit 1
    fi
    CUDA_VISIBLE_DEVICES="${GPU}" MUJOCO_GL=egl "${PYTHON_BIN}" scripts/p0_evaluator.py \
      --checkpoint "${CKPT}" \
      --env-name h1hand-slide-v0 \
      --out "${EVAL_ROOT}/${ARM}_s${SEED}_step20000.json" \
      --expect-global-step 20000 \
      --expect-seed "${SEED}" \
      --expect-admission-mode "${MODE}" \
      --eval-seeds panel128 \
      > "${EVAL_ROOT}/${ARM}_s${SEED}.log" 2>&1
    echo "[$(date -u +%FT%TZ)] eval arm=${ARM} seed=${SEED} DONE"
  done
done
echo "EVAL SEEDS='${SEEDS}' COMPLETE"
