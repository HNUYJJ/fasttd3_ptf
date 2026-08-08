#!/usr/bin/env bash
# T2 的 B-only 臂 20k source-free 面板（scratch/joint 复用 Gate A 已有评估）。
# 128 deterministic episodes，只取 return。
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/home/yjj/miniconda3/envs/FastTD3/bin/python}"
GPU="${GPU:?set GPU}"
SEEDS="${SEEDS:-1 2 3}"
ROOT=docs/data/truck_channel_v1/source_free_eval
mkdir -p "${ROOT}"

for seed in ${SEEDS}; do
  name="tchan_bonly_s${seed}"
  out="${ROOT}/truck_bonly_s${seed}_step20000.json"
  [[ ! -e "${out}" ]] || { echo "[REFUSE] existing: ${out}" >&2; exit 2; }
  mapfile -t ckpts < <(compgen -G "models/*${name}__*_20000.pt" | sort || true)
  [[ ${#ckpts[@]} -eq 1 ]] || {
    echo "[INVALID] ${name}: matches=${#ckpts[@]}" >&2; exit 2; }

  CUDA_VISIBLE_DEVICES="${GPU}" MUJOCO_GL=egl "${PYTHON_BIN}" scripts/p0_evaluator.py \
    --checkpoint "${ckpts[0]}" --env-name h1hand-truck-v0 --out "${out}" \
    --expect-global-step 20000 --expect-seed "${seed}" \
    --expect-admission-mode all --eval-seeds panel128 \
    > "${ROOT}/truck_bonly_s${seed}_step20000.log" 2>&1
  echo "[$(date -u +%FT%TZ)] eval bonly s${seed} DONE"
done
echo "TRUCK CHANNEL B-ONLY EVAL COMPLETE"
