#!/usr/bin/env bash
# RACING_K v1 训练队列：12 条 (臂, seed) 组合，每批 3 条并行，共 4 批串行。
#
# 并行度硬上限 3：本机瓶颈是 RAM 不是显存，超过 3 个训练进程会被静默杀掉
# （无 Traceback、无退出码，日志停在半途）——2026-06-15 与 2026-07-29 各犯过一次。
set -uo pipefail

cd /home/yjj/fasttd3_ptf

BATCHES=(
  "student:1 student:2 student:3"
  "run:1 run:2 run:3"
  "walk:1 walk:2 walk:3"
  "stand:1 stand:2 stand:3"
)

for bi in "${!BATCHES[@]}"; do
  echo "=== 批次 $((bi+1))/${#BATCHES[@]}: ${BATCHES[$bi]}  $(date -u +%FT%TZ) ==="
  free -g | sed -n 2p
  gpu=0
  pids=()
  for job in ${BATCHES[$bi]}; do
    arm="${job%%:*}"; seed="${job##*:}"
    # 幂等：已有 10k checkpoint 则跳过，使本脚本可安全重跑
    if ls -1 models/*${EXP_PREFIX:-rck}_${arm}_s${seed}__*_10000.pt >/dev/null 2>&1; then
      echo "[skip] ${arm} s${seed} 已完成"
      continue
    fi
    GPU="${gpu}" SEEDS="${seed}" ARM="${arm}" EXP_PREFIX="${EXP_PREFIX:-rck}" \
      LR_HORIZON="${LR_HORIZON:-100000}" bash scripts/run_racing_min_horizon_v1.sh &
    pids+=($!)
    gpu=$((gpu+1))
    sleep 5
  done
  [[ ${#pids[@]} -eq 0 ]] && { echo "=== 批次 $((bi+1)) 全部已完成，跳过 ==="; continue; }
  fail=0
  for p in "${pids[@]}"; do wait "$p" || fail=1; done
  echo "=== 批次 $((bi+1)) 结束 (fail=${fail})  $(date -u +%FT%TZ) ==="
done

echo "ALL TRAINING COMPLETE $(date -u +%FT%TZ)"
ls -1 models/*rck_*_{2000,5000,10000}.pt 2>/dev/null | wc -l
