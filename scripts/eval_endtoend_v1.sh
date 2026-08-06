#!/usr/bin/env bash
# 端到端 v1 增量臂的冻结 128-episode source-free 面板（预注册 §5）。
# 只评新跑的 9 条；slide 三臂与 hurdle 的 A/B 臂复用既有评估，不重评。
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/home/yjj/miniconda3/envs/FastTD3/bin/python}"
GPU="${GPU:?set GPU}"
JOB="${JOB:?set JOB: hurdle_C|crawl_A|crawl_B}"
SEEDS="${SEEDS:-1 2 3}"
STEPS="${STEPS:-30000 50000 75000 100000}"

case "${JOB}" in
  hurdle_C) ENV=h1hand-hurdle-v0; SUB=hurdle; ARM=exit;    MODE=none;   PREFIX=e2e_hurdle_exit ;;
  crawl_A)  ENV=h1hand-crawl-v0;  SUB=crawl;  ARM=scratch; MODE=legacy; PREFIX=e2e_crawl_scratch ;;
  crawl_B)  ENV=h1hand-crawl-v0;  SUB=crawl;  ARM=blind;   MODE=all;    PREFIX=e2e_crawl_blind ;;
  *) echo "[REFUSE] unknown JOB=${JOB}" >&2; exit 2 ;;
esac
ROOT="docs/data/endtoend_v1/${SUB}/source_free_eval"
mkdir -p "${ROOT}"

for seed in ${SEEDS}; do
  for step in ${STEPS}; do
    # hurdle_C 的 30k 点在 prefix 臂上（分叉前），其余步在 exit 臂上
    name="${PREFIX}_s${seed}"
    if [[ "${JOB}" == "hurdle_C" && "${step}" == "30000" ]]; then
      name="e2e_hurdle_prefix_s${seed}"; emode=all
    else
      emode="${MODE}"
    fi
    out="${ROOT}/${ARM}_s${seed}_step${step}.json"
    mapfile -t ckpts < <(compgen -G "models/*${name}__*_${step}.pt" | sort || true)
    [[ ${#ckpts[@]} -eq 1 ]] || { echo "[INVALID] ${name} step${step}: matches=${#ckpts[@]}" >&2; exit 2; }
    [[ ! -e "${out}" ]] || { echo "[REFUSE] existing: ${out}" >&2; exit 2; }
    CUDA_VISIBLE_DEVICES="${GPU}" MUJOCO_GL=egl "${PYTHON_BIN}" scripts/p0_evaluator.py \
      --checkpoint "${ckpts[0]}" --env-name "${ENV}" --out "${out}" \
      --expect-global-step "${step}" --expect-seed "${seed}" \
      --expect-admission-mode "${emode}" --eval-seeds panel128 \
      > "${ROOT}/${ARM}_s${seed}_step${step}.log" 2>&1
    echo "[$(date -u +%FT%TZ)] eval ${JOB} s${seed} step${step} DONE"
  done
done
echo "JOB=${JOB} EVAL COMPLETE"
