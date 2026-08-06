#!/usr/bin/env bash
# Frozen source-free evaluation of every Door@10k gate arm at step 20000.
#
# 面板升级(PI 冻结,训练前决定,非看结果后改标准):Cabinet gate 证明 32 episodes
# 的面板 SE 会淹没干预效应,故 Door 的 primary label 用 **128 deterministic
# episodes**(16 eval seeds × 8 ranks)。循环顺序是 for eval_seed: for rank,
# 且前 4 个 eval seed 保持 (11,23,37,53),因此 **前 32 个 episode 的 reset seed
# 与既有 32-episode 面板逐位相同**,构成向后兼容的 secondary 子面板。
set -euo pipefail
PYTHON_BIN="${PYTHON_BIN:-/home/yjj/miniconda3/envs/FastTD3/bin/python}"
GPU="${GPU:?set GPU}"; ARMS="${ARMS:-student stand walk run}"; SEEDS="${SEEDS:?set SEEDS}"
OUT=docs/data/door_at10k_gate_v1/source_free_eval
mkdir -p "${OUT}"
for SEED in ${SEEDS}; do for ARM in ${ARMS}; do
  CKPT=$(ls models/*door_at10k_${ARM}_s${SEED}*_20000.pt 2>/dev/null | head -1)
  [ -z "${CKPT}" ] && { echo "MISSING checkpoint arm=${ARM} seed=${SEED}"; exit 1; }
  MUJOCO_GL=egl CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" scripts/p0_evaluator.py \
    --checkpoint "${CKPT}" --env-name h1hand-door-v0 --eval-seeds panel128 \
    --out "${OUT}/${ARM}_s${SEED}_step20000.json" 2>&1 | grep -E "p0_evaluator" || true
done; done
echo "EVAL_DONE arms='${ARMS}' seeds='${SEEDS}' panel=128"
