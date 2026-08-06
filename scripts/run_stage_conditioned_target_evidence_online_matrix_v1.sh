#!/usr/bin/env bash
set -euo pipefail

STAMP="${STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_ROOT="${RUN_ROOT:-logs/train/stage_target_evidence_online_v1_${STAMP}}"
GPU_A="${GPU_A:-2}"
GPU_B="${GPU_B:-3}"
SEED="${SEED:-1}"

mkdir -p "${RUN_ROOT}"

run_pair() {
  local target_a="$1"
  local arm_a="$2"
  local target_b="$3"
  local arm_b="$4"

  STAMP="${STAMP}" RUN_ROOT="${RUN_ROOT}" TARGET="${target_a}" ARM="${arm_a}" \
    SEED="${SEED}" GPU_ID="${GPU_A}" \
    bash scripts/run_stage_conditioned_target_evidence_online_v1.sh &
  local pid_a=$!
  STAMP="${STAMP}" RUN_ROOT="${RUN_ROOT}" TARGET="${target_b}" ARM="${arm_b}" \
    SEED="${SEED}" GPU_ID="${GPU_B}" \
    bash scripts/run_stage_conditioned_target_evidence_online_v1.sh &
  local pid_b=$!

  local status=0
  wait "${pid_a}" || status=$?
  wait "${pid_b}" || status=$?
  return "${status}"
}

# Two concurrent 128-env jobs fit the node memory budget. Run online arms first
# so their 10k/20k admission decisions become available as early as possible,
# then run matching current-checkout scratch controls.
run_pair hurdle online crawl online
run_pair hurdle scratch crawl scratch

echo "stage-conditioned target-evidence seed-${SEED} matrix complete: ${RUN_ROOT}"
