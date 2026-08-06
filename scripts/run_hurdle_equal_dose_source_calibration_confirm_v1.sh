#!/usr/bin/env bash
set -euo pipefail

GPU_A="${GPU_A:-0}"
GPU_B="${GPU_B:-1}"
STAMP="${STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_ROOT="${RUN_ROOT:-logs/train/hurdle_equal_dose_source_calibration_confirm_v1_${STAMP}}"
SEED1_ROOT="${SEED1_ROOT:-logs/train/hurdle_equal_dose_source_calibration_v1_20260723T132917Z}"
PYTHON_BIN="${PYTHON_BIN:-/home/yjj/miniconda3/envs/FastTD3/bin/python}"

if [[ "${GPU_A}" == "${GPU_B}" ]]; then
  echo "GPU_A and GPU_B must be distinct" >&2
  exit 2
fi
mkdir -p "${RUN_ROOT}"

active_pids=()
cleanup() {
  local pid
  for pid in "${active_pids[@]:-}"; do
    kill "${pid}" 2>/dev/null || true
  done
}
trap cleanup INT TERM

run_pair() {
  local arm_a="$1" seed_a="$2" arm_b="$3" seed_b="$4"
  set +e
  ARM="${arm_a}" SEED="${seed_a}" GPU_ID="${GPU_A}" STAMP="${STAMP}" \
    RUN_ROOT="${RUN_ROOT}" PYTHON_BIN="${PYTHON_BIN}" \
    bash scripts/run_hurdle_equal_dose_source_calibration_v1.sh &
  local pid_a=$!
  ARM="${arm_b}" SEED="${seed_b}" GPU_ID="${GPU_B}" STAMP="${STAMP}" \
    RUN_ROOT="${RUN_ROOT}" PYTHON_BIN="${PYTHON_BIN}" \
    bash scripts/run_hurdle_equal_dose_source_calibration_v1.sh &
  local pid_b=$!
  active_pids=("${pid_a}" "${pid_b}")
  wait "${pid_a}"; local status_a=$?
  wait "${pid_b}"; local status_b=$?
  active_pids=()
  set -e
  if [[ "${status_a}" -ne 0 || "${status_b}" -ne 0 ]]; then
    echo "training pair failed: ${arm_a}/s${seed_a}=${status_a}, ${arm_b}/s${seed_b}=${status_b}" >&2
    exit 3
  fi
}

run_pair scratch 2 run 2
run_pair walk 2 scratch 3
run_pair run 3 walk 3

EVAL_ROOT="${RUN_ROOT}/source_free_eval"
mkdir -p "${EVAL_ROOT}"

evaluate_one() {
  local arm="$1" seed="$2" gpu="$3"
  local mode="all"
  [[ "${arm}" == "scratch" ]] && mode="legacy"
  local checkpoint
  checkpoint="$("${PYTHON_BIN}" - "${RUN_ROOT}/${arm}_s${seed}.meta.json" <<'PY'
import json, pathlib, sys
print(json.loads(pathlib.Path(sys.argv[1]).read_text())["completed_step_checkpoint"])
PY
)"
  CUDA_VISIBLE_DEVICES="${gpu}" MUJOCO_GL=egl "${PYTHON_BIN}" scripts/p0_evaluator.py \
    --checkpoint "${checkpoint}" \
    --env-name h1hand-hurdle-v0 \
    --out "${EVAL_ROOT}/${arm}_s${seed}_step30000.json" \
    --expect-global-step 30000 \
    --expect-seed "${seed}" \
    --expect-admission-mode "${mode}" \
    > "${EVAL_ROOT}/${arm}_s${seed}.log" 2>&1
}

eval_pair() {
  local arm_a="$1" seed_a="$2" arm_b="$3" seed_b="$4"
  set +e
  evaluate_one "${arm_a}" "${seed_a}" "${GPU_A}" & local pid_a=$!
  evaluate_one "${arm_b}" "${seed_b}" "${GPU_B}" & local pid_b=$!
  active_pids=("${pid_a}" "${pid_b}")
  wait "${pid_a}"; local status_a=$?
  wait "${pid_b}"; local status_b=$?
  active_pids=()
  set -e
  if [[ "${status_a}" -ne 0 || "${status_b}" -ne 0 ]]; then
    echo "evaluation pair failed: ${arm_a}/s${seed_a}=${status_a}, ${arm_b}/s${seed_b}=${status_b}" >&2
    exit 4
  fi
}

eval_pair scratch 2 run 2
eval_pair walk 2 scratch 3
eval_pair run 3 walk 3

"${PYTHON_BIN}" scripts/analyze_hurdle_equal_dose_source_calibration_multiseed_v1.py \
  --seed1-root "${SEED1_ROOT}" \
  --confirm-root "${RUN_ROOT}" \
  --out-prefix "${RUN_ROOT}/hurdle_equal_dose_source_calibration_multiseed_v1_results" \
  > "${RUN_ROOT}/analysis_stdout.log" 2>&1

echo "confirmation matrix complete: ${RUN_ROOT}"
