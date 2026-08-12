#!/usr/bin/env bash
# 等第一批 3 条完成后自动启动第二批 3 条（并行度硬上限 3，RAM 瓶颈）。
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
while true; do
  n=$(ls models/*p0_*_final.pt 2>/dev/null | wc -l)
  alive=$(ps -eo args --no-headers | grep -cE "^/home/yjj/miniconda3/envs/FastTD3/bin/python src/train_interface")
  [[ "$n" -ge 3 || "$alive" -eq 0 ]] && break
  sleep 60
done
echo "[$(date -u +%FT%TZ)] 第一批结束（final=$n），启动第二批"
for spec in "iface 2 0" "flat 3 6" "iface 3 7"; do
  set -- $spec
  setsid tmux new-session -d -s "p0_$1$2" "ARM=$1 SEED=$2 GPU=$3 bash scripts/run_phase0.sh"
  sleep 5
done
echo "[$(date -u +%FT%TZ)] 第二批已启动"
