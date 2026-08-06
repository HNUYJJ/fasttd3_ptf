#!/usr/bin/env bash
set -euo pipefail

STAMP="${STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_ROOT="${RUN_ROOT:-logs/train/stage_target_evidence_online_v1_confirm_${STAMP}}"
GPU_A="${GPU_A:-2}"
GPU_B="${GPU_B:-3}"

mkdir -p "${RUN_ROOT}"
trap 'rc=$?; printf "%s\n" "${rc}" >"${RUN_ROOT}/matrix.exit"' EXIT

run_pair() {
  local target_a="$1"
  local arm_a="$2"
  local seed_a="$3"
  local target_b="$4"
  local arm_b="$5"
  local seed_b="$6"

  STAMP="${STAMP}" RUN_ROOT="${RUN_ROOT}" TARGET="${target_a}" ARM="${arm_a}" \
    SEED="${seed_a}" GPU_ID="${GPU_A}" \
    bash scripts/run_stage_conditioned_target_evidence_online_v1.sh &
  local pid_a=$!
  STAMP="${STAMP}" RUN_ROOT="${RUN_ROOT}" TARGET="${target_b}" ARM="${arm_b}" \
    SEED="${seed_b}" GPU_ID="${GPU_B}" \
    bash scripts/run_stage_conditioned_target_evidence_online_v1.sh &
  local pid_b=$!

  local status=0
  wait "${pid_a}" || status=$?
  wait "${pid_b}" || status=$?
  return "${status}"
}

# Minimum confirmation matrix:
# - hurdle: online + current-checkout scratch for seeds 2 and 3;
# - crawl: online only for seeds 2 and 3, because exact abstention itself
#   guarantees zero source behavior and zero source replay exposure.
run_pair hurdle online 2 hurdle scratch 2
run_pair hurdle online 3 crawl online 2
run_pair hurdle scratch 3 crawl online 3

echo "stage-conditioned target-evidence confirmation complete: ${RUN_ROOT}"
