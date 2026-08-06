#!/usr/bin/env bash
set -euo pipefail

export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

PYTHON_BIN="${PYTHON_BIN:-python}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Keep the default training path self-contained inside this project.
FASTTD3_ROOT="${ROOT_DIR}/fasttd3_ptf/official_code/FastTD3"
FASTTD3_TRAIN="${FASTTD3_ROOT}/fast_td3/train.py"
HUMANOID_BENCH_ROOT="${ROOT_DIR}/fasttd3_ptf/official_code/humanoid-bench"
ENV_NAME="${ENV_NAME:-h1hand-push-v0}"
EXP_NAME="${EXP_NAME:-${ENV_NAME//-/_}_scratch_official}"
PROJECT="${PROJECT:-fasttd3_ptf}"
SEED="${SEED:-1}"
COMPILE="${COMPILE:-0}"

if [[ ! -f "${FASTTD3_TRAIN}" ]]; then
  echo "Could not find FastTD3 train.py at ${FASTTD3_TRAIN}" >&2
  echo "Expected the in-project official code under fasttd3_ptf/official_code/FastTD3." >&2
  exit 1
fi

export PYTHONPATH="${FASTTD3_ROOT}/fast_td3:${HUMANOID_BENCH_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

ARGS=(
  --env-name "${ENV_NAME}"
  --exp-name "${EXP_NAME}"
  --project "${PROJECT}"
  --seed "${SEED}"
)

if [[ -n "${TOTAL_TIMESTEPS:-}" ]]; then
  ARGS+=(--total-timesteps "${TOTAL_TIMESTEPS}")
fi
if [[ -n "${NUM_ENVS:-}" ]]; then
  ARGS+=(--num-envs "${NUM_ENVS}")
fi
if [[ -n "${BATCH_SIZE:-}" ]]; then
  ARGS+=(--batch-size "${BATCH_SIZE}")
fi
if [[ -n "${BUFFER_SIZE:-}" ]]; then
  ARGS+=(--buffer-size "${BUFFER_SIZE}")
fi
if [[ -n "${LEARNING_STARTS:-}" ]]; then
  ARGS+=(--learning-starts "${LEARNING_STARTS}")
fi
if [[ -n "${NUM_UPDATES:-}" ]]; then
  ARGS+=(--num-updates "${NUM_UPDATES}")
fi
if [[ -n "${POLICY_FREQUENCY:-}" ]]; then
  ARGS+=(--policy-frequency "${POLICY_FREQUENCY}")
fi
if [[ -n "${DEVICE_RANK:-}" ]]; then
  ARGS+=(--device-rank "${DEVICE_RANK}")
fi
if [[ "${CUDA:-1}" == "0" ]]; then
  ARGS+=(--no-cuda)
fi
if [[ "${WANDB:-1}" == "0" ]]; then
  ARGS+=(--no-use-wandb)
fi
if [[ "${AMP:-1}" == "0" ]]; then
  ARGS+=(--no-amp)
fi
if [[ "${COMPILE}" == "0" ]]; then
  ARGS+=(--no-compile)
fi
if [[ "${RENDER_INTERVAL:-}" != "" ]]; then
  ARGS+=(--render-interval "${RENDER_INTERVAL}")
fi
if [[ "${EVAL_INTERVAL:-}" != "" ]]; then
  ARGS+=(--eval-interval "${EVAL_INTERVAL}")
fi

"${PYTHON_BIN}" "${FASTTD3_TRAIN}" "${ARGS[@]}"
