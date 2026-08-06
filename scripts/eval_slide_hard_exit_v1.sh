#!/usr/bin/env bash
# Frozen 128-episode source-free panel for Slide hard-exit v1.
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/home/yjj/miniconda3/envs/FastTD3/bin/python}"
GPU="${GPU:?set GPU}"
SEEDS="${SEEDS:-1 2 3}"
ROOT=docs/data/slide_hard_exit_v1/source_free_eval
mkdir -p "${ROOT}"

eval_one() {
  local arm="$1" seed="$2" step="$3" mode="$4"
  local name="shev1_${arm}_s${seed}"
  local out="${ROOT}/${arm}_s${seed}_step${step}.json"
  mapfile -t ckpts < <(compgen -G "models/*${name}__*_${step}.pt" | sort || true)
  [[ ${#ckpts[@]} -eq 1 ]] || {
    echo "[INVALID] ${arm} s${seed} step${step}: matches=${#ckpts[@]}" >&2; exit 2;
  }
  [[ ! -e "${out}" ]] || {
    echo "[REFUSE] existing evaluation: ${out}" >&2; exit 2;
  }
  CUDA_VISIBLE_DEVICES="${GPU}" MUJOCO_GL=egl "${PYTHON_BIN}" scripts/p0_evaluator.py \
    --checkpoint "${ckpts[0]}" --env-name h1hand-slide-v0 --out "${out}" \
    --expect-global-step "${step}" --expect-seed "${seed}" \
    --expect-admission-mode "${mode}" --eval-seeds panel128 \
    > "${ROOT}/${arm}_s${seed}_step${step}.log" 2>&1
  echo "[$(date -u +%FT%TZ)] eval ${arm} s${seed} step${step} DONE"
}

for seed in ${SEEDS}; do
  eval_one prefix "${seed}" 30000 all
  for step in 50000 75000 100000; do
    eval_one cont "${seed}" "${step}" all
    eval_one exit "${seed}" "${step}" none
  done
done
