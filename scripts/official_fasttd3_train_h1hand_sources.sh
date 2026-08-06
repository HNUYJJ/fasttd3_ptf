#!/usr/bin/env bash
set -euo pipefail

export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

PYTHON_BIN="${PYTHON_BIN:-python}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Keep the default training path self-contained inside this project.
FASTTD3_ROOT="${ROOT_DIR}/fasttd3_ptf/official_code/FastTD3"
FASTTD3_TRAIN="${FASTTD3_ROOT}/fast_td3/train.py"
HUMANOID_BENCH_ROOT="${ROOT_DIR}/fasttd3_ptf/official_code/humanoid-bench"
PROJECT="${PROJECT:-fasttd3_ptf}"
SEED="${SEED:-1}"
COMPILE="${COMPILE:-0}"

if [[ ! -f "${FASTTD3_TRAIN}" ]]; then
  echo "Could not find FastTD3 train.py at ${FASTTD3_TRAIN}" >&2
  echo "Expected the in-project official code under fasttd3_ptf/official_code/FastTD3." >&2
  exit 1
fi

export PYTHONPATH="${FASTTD3_ROOT}/fast_td3:${HUMANOID_BENCH_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
COMMON_ARGS=(
  --project "${PROJECT}"
  --seed "${SEED}"
)

if [[ -n "${TOTAL_TIMESTEPS:-}" ]]; then
  COMMON_ARGS+=(--total-timesteps "${TOTAL_TIMESTEPS}")
fi
if [[ -n "${NUM_ENVS:-}" ]]; then
  COMMON_ARGS+=(--num-envs "${NUM_ENVS}")
fi
if [[ -n "${BATCH_SIZE:-}" ]]; then
  COMMON_ARGS+=(--batch-size "${BATCH_SIZE}")
fi
if [[ -n "${BUFFER_SIZE:-}" ]]; then
  COMMON_ARGS+=(--buffer-size "${BUFFER_SIZE}")
fi
if [[ -n "${LEARNING_STARTS:-}" ]]; then
  COMMON_ARGS+=(--learning-starts "${LEARNING_STARTS}")
fi
if [[ -n "${NUM_UPDATES:-}" ]]; then
  COMMON_ARGS+=(--num-updates "${NUM_UPDATES}")
fi
if [[ -n "${POLICY_FREQUENCY:-}" ]]; then
  COMMON_ARGS+=(--policy-frequency "${POLICY_FREQUENCY}")
fi
if [[ -n "${DEVICE_RANK:-}" ]]; then
  COMMON_ARGS+=(--device-rank "${DEVICE_RANK}")
fi
if [[ "${CUDA:-1}" == "0" ]]; then
  COMMON_ARGS+=(--no-cuda)
fi
if [[ "${WANDB:-1}" == "0" ]]; then
  COMMON_ARGS+=(--no-use-wandb)
fi
if [[ "${AMP:-1}" == "0" ]]; then
  COMMON_ARGS+=(--no-amp)
fi
if [[ "${COMPILE}" == "0" ]]; then
  COMMON_ARGS+=(--no-compile)
fi
if [[ "${RENDER_INTERVAL:-}" != "" ]]; then
  COMMON_ARGS+=(--render-interval "${RENDER_INTERVAL}")
fi
if [[ "${EVAL_INTERVAL:-}" != "" ]]; then
  COMMON_ARGS+=(--eval-interval "${EVAL_INTERVAL}")
fi

run_source() {
  local env_name="$1"
  local exp_name="$2"
  "${PYTHON_BIN}" "${FASTTD3_TRAIN}" \
    --env-name "${env_name}" \
    --exp-name "${exp_name}" \
    "${COMMON_ARGS[@]}"
}

run_source h1hand-stand-v0 h1hand_stand_source_official
run_source h1hand-walk-v0 h1hand_walk_source_official
run_source h1hand-run-v0 h1hand_run_source_official
run_source h1hand-reach-v0 h1hand_reach_source_official
