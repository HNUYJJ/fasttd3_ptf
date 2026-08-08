#!/usr/bin/env bash
# PARE Gate A 的 source-free 评估面板（判据见
# docs/experiments/pare_gate_a_prereg_20260808.md）。
#
# 128 个 deterministic episode，source-free。只取 return——success_count 在
# locomotion 上读的是 terminated（摔倒早停），与 return 强反向（CLAUDE.md §6）。
#
# 每个 (arm, step) 的 checkpoint 必须**恰好匹配一个**文件，否则 INVALID 退出；
# 已存在的评估结果一律 REFUSE，不覆盖。
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/home/yjj/miniconda3/envs/FastTD3/bin/python}"
GPU="${GPU:?set GPU}"
TASK="${TASK:?set TASK: stair|truck}"
SEEDS="${SEEDS:-1 2 3}"
ROOT=docs/data/pare_gate_a_v1/source_free_eval
mkdir -p "${ROOT}"

case "${TASK}" in
  stair) ENV_NAME=h1hand-stair-v0 ;;
  truck) ENV_NAME=h1hand-truck-v0 ;;
  *) echo "[FATAL] unknown TASK=${TASK}" >&2; exit 2 ;;
esac

eval_one() {
  local arm="$1" seed="$2" step="$3" mode="$4"
  local name="pgav1_${arm}_${TASK}_s${seed}"
  local out="${ROOT}/${TASK}_${arm}_s${seed}_step${step}.json"
  [[ ! -e "${out}" ]] || { echo "[REFUSE] existing: ${out}" >&2; exit 2; }
  mapfile -t ckpts < <(compgen -G "models/*${name}__*_${step}.pt" | sort || true)
  [[ ${#ckpts[@]} -eq 1 ]] || {
    echo "[INVALID] ${name} step${step}: matches=${#ckpts[@]}" >&2; exit 2; }

  CUDA_VISIBLE_DEVICES="${GPU}" MUJOCO_GL=egl "${PYTHON_BIN}" scripts/p0_evaluator.py \
    --checkpoint "${ckpts[0]}" --env-name "${ENV_NAME}" --out "${out}" \
    --expect-global-step "${step}" --expect-seed "${seed}" \
    --expect-admission-mode "${mode}" --eval-seeds panel128 \
    > "${ROOT}/${TASK}_${arm}_s${seed}_step${step}.log" 2>&1
  echo "[$(date -u +%FT%TZ)] eval ${TASK} ${arm} s${seed} step${step} DONE"
}

for seed in ${SEEDS}; do
  # release 点：exit 臂的 20k 状态就是 scaffold run 的 20k checkpoint
  eval_one scaf "${seed}" 20000 all
  for step in 50000 100000; do
    eval_one exit "${seed}" "${step}" none
  done
  for step in 20000 50000 100000; do
    eval_one scratch "${seed}" "${step}" legacy
  done
done
echo "PARE GATE-A EVAL ${TASK} SEEDS='${SEEDS}' COMPLETE"
