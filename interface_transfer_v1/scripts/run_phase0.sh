#!/usr/bin/env bash
# Phase 0：flat vs goal-conditioned interface，h1hand-push-v0。
# 判据冻结于 docs/phase0_prereg_v2_20260812.md（先于本次运行）。
#
#   ARM=flat|iface SEED=1..3 GPU=<id> bash scripts/run_phase0.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-/home/yjj/miniconda3/envs/FastTD3/bin/python}"
ARM="${ARM:?set ARM=flat|iface}"
SEED="${SEED:?set SEED}"
GPU="${GPU:?set GPU}"
STEPS="${STEPS:-100000}"

case "$ARM" in
  flat)  PTYPE="" ;;
  iface) PTYPE="reach_single" ;;
  *) echo "[FATAL] ARM 必须是 flat 或 iface" >&2; exit 2 ;;
esac

NAME="p0_${ARM}_s${SEED}"
LOGDIR=logs/phase0
mkdir -p "$LOGDIR" models

# §3：输出集合两两不相交 + 幂等拒绝
if compgen -G "models/*${NAME}__*.pt" >/dev/null || [[ -e "${LOGDIR}/${NAME}.log" ]]; then
  echo "[REFUSE] 已存在 ${NAME} 的输出" >&2; exit 2
fi

echo "[$(date -u +%FT%TZ)] START ${NAME} (policy_type=${PTYPE:-flat}, steps=${STEPS}, gpu=${GPU})"

INTERFACE_POLICY_TYPE="$PTYPE" \
MUJOCO_GL=egl PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="$GPU" \
WANDB_INIT_TIMEOUT=300 WANDB_SILENT=true \
"$PYTHON_BIN" src/train_interface.py \
  --env_name h1hand-push-v0 \
  --seed "$SEED" \
  --total_timesteps "$STEPS" \
  --eval_interval 5000 \
  --render_interval 0 \
  --save_interval 0 \
  --no-compile \
  --project ptf_interface_transfer_v1 \
  --exp_name "$NAME" \
  > "${LOGDIR}/${NAME}.log" 2>&1

echo "[$(date -u +%FT%TZ)] DONE ${NAME}"
