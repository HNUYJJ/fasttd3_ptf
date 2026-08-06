#!/usr/bin/env bash
set -euo pipefail

GPU_A="${GPU_A:-0}"
GPU_B="${GPU_B:-1}"
SEED="${SEED:-1}"
TARGET="${TARGET:-hurdle}"
STAMP="${STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_ROOT="${RUN_ROOT:-logs/train/${TARGET}_equal_dose_source_calibration_v1_${STAMP}}"
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
  local arm_a="$1"
  local arm_b="$2"
  set +e
  ARM="${arm_a}" SEED="${SEED}" GPU_ID="${GPU_A}" TARGET="${TARGET}" STAMP="${STAMP}" \
    RUN_ROOT="${RUN_ROOT}" PYTHON_BIN="${PYTHON_BIN}" \
    bash scripts/run_hurdle_equal_dose_source_calibration_v1.sh &
  local pid_a=$!
  ARM="${arm_b}" SEED="${SEED}" GPU_ID="${GPU_B}" TARGET="${TARGET}" STAMP="${STAMP}" \
    RUN_ROOT="${RUN_ROOT}" PYTHON_BIN="${PYTHON_BIN}" \
    bash scripts/run_hurdle_equal_dose_source_calibration_v1.sh &
  local pid_b=$!
  active_pids=("${pid_a}" "${pid_b}")
  wait "${pid_a}"; local status_a=$?
  wait "${pid_b}"; local status_b=$?
  active_pids=()
  set -e
  if [[ "${status_a}" -ne 0 || "${status_b}" -ne 0 ]]; then
    echo "training pair failed: ${arm_a}=${status_a}, ${arm_b}=${status_b}" >&2
    exit 3
  fi
}

run_pair scratch stand
run_pair walk run

EVAL_ROOT="${RUN_ROOT}/source_free_eval"
mkdir -p "${EVAL_ROOT}"

evaluate_arm() {
  local arm="$1"
  local gpu="$2"
  local mode="all"
  if [[ "${arm}" == "scratch" ]]; then
    mode="legacy"
  fi
  local checkpoint
  checkpoint="$("${PYTHON_BIN}" - "${RUN_ROOT}/${arm}_s${SEED}.meta.json" <<'PY'
import json, pathlib, sys
print(json.loads(pathlib.Path(sys.argv[1]).read_text())["completed_step_checkpoint"])
PY
)"
  CUDA_VISIBLE_DEVICES="${gpu}" MUJOCO_GL=egl "${PYTHON_BIN}" scripts/p0_evaluator.py \
    --checkpoint "${checkpoint}" \
    --env-name "h1hand-${TARGET}-v0" \
    --out "${EVAL_ROOT}/${arm}_s${SEED}_step30000.json" \
    --expect-global-step 30000 \
    --expect-seed "${SEED}" \
    --expect-admission-mode "${mode}" \
    > "${EVAL_ROOT}/${arm}_s${SEED}.log" 2>&1
}

set +e
evaluate_arm scratch "${GPU_A}" & eval_a=$!
evaluate_arm stand "${GPU_B}" & eval_b=$!
active_pids=("${eval_a}" "${eval_b}")
wait "${eval_a}"; eval_a_status=$?
wait "${eval_b}"; eval_b_status=$?
active_pids=()
set -e
if [[ "${eval_a_status}" -ne 0 || "${eval_b_status}" -ne 0 ]]; then
  echo "first evaluation pair failed" >&2
  exit 4
fi

set +e
evaluate_arm walk "${GPU_A}" & eval_a=$!
evaluate_arm run "${GPU_B}" & eval_b=$!
active_pids=("${eval_a}" "${eval_b}")
wait "${eval_a}"; eval_a_status=$?
wait "${eval_b}"; eval_b_status=$?
active_pids=()
set -e
if [[ "${eval_a_status}" -ne 0 || "${eval_b_status}" -ne 0 ]]; then
  echo "second evaluation pair failed" >&2
  exit 5
fi

analysis_extra=()
if [[ "${TARGET}" == "hurdle" ]]; then
  analysis_extra+=(--selector-preference walk --decision-mode positive)
else
  analysis_extra+=(--decision-mode negative)
fi

"${PYTHON_BIN}" scripts/analyze_hurdle_equal_dose_source_calibration_v1.py \
  --run-root "${RUN_ROOT}" \
  --experiment "${TARGET}_equal_dose_source_calibration_v1" \
  --title "${TARGET^}等剂量单source RBO标定：seed-1结果" \
  "${analysis_extra[@]}" \
  > "${RUN_ROOT}/analysis_stdout.log" 2>&1

echo "matrix complete: ${RUN_ROOT}"
