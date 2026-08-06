#!/usr/bin/env bash
set -euo pipefail
CHECKPOINT=${1:?usage: scripts/export_source_policy.sh CHECKPOINT ENV_ID OUTPUT_JSON}
ENV_ID=${2:?usage: scripts/export_source_policy.sh CHECKPOINT ENV_ID OUTPUT_JSON}
OUTPUT=${3:?usage: scripts/export_source_policy.sh CHECKPOINT ENV_ID OUTPUT_JSON}
python -m fasttd3_ptf.source_bank.exporter --checkpoint "$CHECKPOINT" --env-id "$ENV_ID" --output "$OUTPUT"
