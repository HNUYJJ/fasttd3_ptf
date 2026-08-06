#!/usr/bin/env bash
# racing 准入 v1 的冻结 128-episode source-free 面板（预注册 §5）。
# 只评 K=10000（主判据）；2000/5000 的 checkpoint 保留但不进判据。
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/home/yjj/miniconda3/envs/FastTD3/bin/python}"
GPU="${GPU:?set GPU}"
TARGET="${TARGET:?set TARGET: crawl|slide}"
SEEDS="${SEEDS:-1 2 3}"
ARMS="${ARMS:-student stand walk run}"
STEP="${STEP:-10000}"
ROOT="docs/data/racing_admission_v1/${TARGET}/source_free_eval"
mkdir -p "${ROOT}"

for seed in ${SEEDS}; do
  for arm in ${ARMS}; do
    name="rad_${TARGET}_${arm}_s${seed}"
    out="${ROOT}/${arm}_s${seed}_step${STEP}.json"
    mapfile -t ckpts < <(compgen -G "models/*${name}__*_${STEP}.pt" | sort || true)
    [[ ${#ckpts[@]} -eq 1 ]] || {
      echo "[INVALID] ${TARGET} ${arm} s${seed} step${STEP}: matches=${#ckpts[@]}" >&2; exit 2;
    }
    [[ ! -e "${out}" ]] || { echo "[REFUSE] existing evaluation: ${out}" >&2; exit 2; }
    # student 臂 admission_mode=legacy，源臂=all（预注册 §5）
    mode=all; [[ "${arm}" == "student" ]] && mode=legacy
    CUDA_VISIBLE_DEVICES="${GPU}" MUJOCO_GL=egl "${PYTHON_BIN}" scripts/p0_evaluator.py \
      --checkpoint "${ckpts[0]}" --env-name "h1hand-${TARGET}-v0" --out "${out}" \
      --expect-global-step "${STEP}" --expect-seed "${seed}" \
      --expect-admission-mode "${mode}" --eval-seeds panel128 \
      > "${ROOT}/${arm}_s${seed}_step${STEP}.log" 2>&1
    echo "[$(date -u +%FT%TZ)] eval ${TARGET} ${arm} s${seed} step${STEP} DONE"
  done
done
echo "TARGET=${TARGET} SEEDS='${SEEDS}' EVAL COMPLETE"
