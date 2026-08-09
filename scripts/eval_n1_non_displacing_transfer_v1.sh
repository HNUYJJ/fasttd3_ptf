#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/home/yjj/miniconda3/envs/FastTD3/bin/python}"
GPU="${GPU:?set GPU}"
SEEDS="${SEEDS:-4 5 6 7 8}"
ROOT=docs/data/n1_non_displacing_transfer_v1/source_free_eval
mkdir -p "${ROOT}"

for seed in ${SEEDS}; do
  for arm in s ff fp lp; do
    name="n1_${arm}_truck_s${seed}"
    out="${ROOT}/truck_${arm}_s${seed}_step20000.json"
    [[ ! -e "${out}" ]] || { echo "[REFUSE] existing: ${out}" >&2; exit 2; }
    mapfile -t ckpts < <(compgen -G "models/*${name}__*_20000.pt" | sort || true)
    [[ ${#ckpts[@]} -eq 1 ]] || {
      echo "[INVALID] ${name}: checkpoint matches=${#ckpts[@]}" >&2; exit 2; }
    mode=all
    [[ "${arm}" == s ]] && mode=none
    CUDA_VISIBLE_DEVICES="${GPU}" MUJOCO_GL=egl "${PYTHON_BIN}" scripts/p0_evaluator.py \
      --checkpoint "${ckpts[0]}" --env-name h1hand-truck-v0 --out "${out}" \
      --expect-global-step 20000 --expect-seed "${seed}" \
      --expect-admission-mode "${mode}" --eval-seeds panel128 \
      > "${ROOT}/truck_${arm}_s${seed}_step20000.log" 2>&1
    echo "[$(date -u +%FT%TZ)] EVAL ${arm} s${seed} DONE"
  done
done
