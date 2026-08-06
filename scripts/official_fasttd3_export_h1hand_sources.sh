#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
SEED="${SEED:-1}"
MODEL_DIR="${MODEL_DIR:-models}"
OUT_ROOT="${OUT_ROOT:-checkpoints/official_sources}"

export_one() {
  local key="$1"
  local env_name="$2"
  local exp_name="$3"
  local checkpoint="${MODEL_DIR}/${env_name}__${exp_name}__${SEED}_final.pt"
  local output="${OUT_ROOT}/${key}/manifest.json"
  "${PYTHON_BIN}" -m fasttd3_ptf.source_bank.exporter \
    --checkpoint "${checkpoint}" \
    --env-id "${env_name}" \
    --name "${key#h1hand_}" \
    --output "${output}"
}

export_one h1hand_stand h1hand-stand-v0 h1hand_stand_source_official
export_one h1hand_walk h1hand-walk-v0 h1hand_walk_source_official
export_one h1hand_run h1hand-run-v0 h1hand_run_source_official
export_one h1hand_reach h1hand-reach-v0 h1hand_reach_source_official
