#!/usr/bin/env bash
set -euo pipefail

export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

PYTHON_BIN="${PYTHON_BIN:-python}"
ENV_NAME="${ENV_NAME:-h1hand-push-v0}"
EXP_NAME="${EXP_NAME:-${ENV_NAME//-/_}_ptf_official}"
PROJECT="${PROJECT:-fasttd3_ptf}"
SEED="${SEED:-1}"
SOURCE_BANK="${SOURCE_BANK:-configs/source_banks/official/h1hand_basic_sources.yaml}"
COMPILE="${COMPILE:-0}"

ARGS=(
  --env-name "${ENV_NAME}"
  --exp-name "${EXP_NAME}"
  --project "${PROJECT}"
  --seed "${SEED}"
  --ptf-source-bank "${SOURCE_BANK}"
)

if [[ -n "${PTF_CONFIG:-}" ]]; then
  ARGS+=(--ptf-config "${PTF_CONFIG}")
fi

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
if [[ -n "${SAVE_INTERVAL:-}" ]]; then
  ARGS+=(--save-interval "${SAVE_INTERVAL}")
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
if [[ "${COMPILE}" == "0" ]]; then
  ARGS+=(--no-compile)
fi
if [[ "${AMP:-1}" == "0" ]]; then
  ARGS+=(--no-amp)
fi
if [[ "${RENDER_INTERVAL:-}" != "" ]]; then
  ARGS+=(--render-interval "${RENDER_INTERVAL}")
fi
if [[ "${EVAL_INTERVAL:-}" != "" ]]; then
  ARGS+=(--eval-interval "${EVAL_INTERVAL}")
fi
if [[ "${PTF_ANCHOR_STEP:-}" != "" ]]; then
  ARGS+=(--ptf-anchor-step "${PTF_ANCHOR_STEP}")
fi
if [[ "${PTF_ANCHOR_DIR:-}" != "" ]]; then
  ARGS+=(--ptf-anchor-dir "${PTF_ANCHOR_DIR}")
fi
if [[ "${PTF_BRANCH_ANCHOR_STEP:-}" != "" ]]; then
  ARGS+=(--ptf-branch-anchor-step "${PTF_BRANCH_ANCHOR_STEP}")
fi
if [[ "${PTF_BRANCH_ANCHOR_DIR:-}" != "" ]]; then
  ARGS+=(--ptf-branch-anchor-dir "${PTF_BRANCH_ANCHOR_DIR}")
fi
if [[ "${PTF_ANCHOR_RESUME:-}" != "" ]]; then
  ARGS+=(--ptf-anchor-resume "${PTF_ANCHOR_RESUME}")
fi
# anchor 的 provenance 组数必须与运行时 MCG 组数一致（train_ptf.py:1121 分配、
# :1390 的守卫复核）。truck 等 manipulation target 用 2 组（legs_torso,arms），
# 与 DEFAULT_GROUPS=3 不同，不透传会在 enable_provenance 处直接 ValueError。
if [[ "${PTF_ANCHOR_PROVENANCE_GROUPS:-}" != "" ]]; then
  ARGS+=(--ptf-anchor-provenance-groups "${PTF_ANCHOR_PROVENANCE_GROUPS}")
fi
# PARE（docs/PARE_ALGORITHM_SPEC_v1.md）。PTF_PARE=1 开启，必须与
# PTF_ADMISSION_MODE=none + PTF_ANCHOR_RESUME=<release anchor> 同用。
if [[ "${PTF_PARE:-}" == "1" ]]; then
  ARGS+=(--ptf-pare)
fi
if [[ "${PTF_PARE_RESERVOIR_CAPACITY:-}" != "" ]]; then
  ARGS+=(--ptf-pare-reservoir-capacity "${PTF_PARE_RESERVOIR_CAPACITY}")
fi
if [[ "${PTF_PARE_D_LR:-}" != "" ]]; then
  ARGS+=(--ptf-pare-d-lr "${PTF_PARE_D_LR}")
fi
if [[ "${PTF_RESUME_NOISE_SEED:-}" != "" ]]; then
  ARGS+=(--ptf-resume-noise-seed "${PTF_RESUME_NOISE_SEED}")
fi
if [[ "${PTF_ACTOR_UPDATE_START_STEP:-}" != "" ]]; then
  ARGS+=(--ptf-actor-update-start-step "${PTF_ACTOR_UPDATE_START_STEP}")
fi
if [[ "${PTF_RUN_STOP_STEP:-}" != "" ]]; then
  ARGS+=(--ptf-run-stop-step "${PTF_RUN_STOP_STEP}")
fi
if [[ "${PTF_EVAL_CHECKPOINT_STEPS:-}" != "" ]]; then
  ARGS+=(--ptf-eval-checkpoint-steps "${PTF_EVAL_CHECKPOINT_STEPS}")
fi
if [[ "${PTF_ADMISSION_EXPECTED_SOURCE_MASS:-}" != "" ]]; then
  ARGS+=(--ptf-admission-expected-source-mass "${PTF_ADMISSION_EXPECTED_SOURCE_MASS}")
fi
if [[ "${TRANSFER_LAMBDA_START:-}" != "" ]]; then
  ARGS+=(--ptf-transfer-lambda-start "${TRANSFER_LAMBDA_START}")
fi
if [[ "${TRANSFER_LAMBDA_END:-}" != "" ]]; then
  ARGS+=(--ptf-transfer-lambda-end "${TRANSFER_LAMBDA_END}")
fi
if [[ "${TRANSFER_DECAY_STEPS:-}" != "" ]]; then
  ARGS+=(--ptf-transfer-decay-steps "${TRANSFER_DECAY_STEPS}")
fi
if [[ "${PTF_XI:-}" != "" ]]; then
  ARGS+=(--ptf-xi "${PTF_XI}")
fi
if [[ "${BETA_WEIGHTED:-1}" == "0" ]]; then
  ARGS+=(--ptf-no-beta-weighted-transfer)
fi
if [[ "${UPDATE_ALL_COMPATIBLE:-1}" == "0" ]]; then
  ARGS+=(--ptf-no-update-all-compatible-options)
fi
if [[ "${PTF_BETA_LR:-}" != "" ]]; then
  ARGS+=(--ptf-beta-lr "${PTF_BETA_LR}")
fi
if [[ "${PTF_OPTION_REWARD_SCALE:-}" != "" ]]; then
  ARGS+=(--ptf-option-reward-scale "${PTF_OPTION_REWARD_SCALE}")
fi
if [[ "${PTF_BETA_WARMUP_STEPS:-}" != "" ]]; then
  ARGS+=(--ptf-beta-warmup-steps "${PTF_BETA_WARMUP_STEPS}")
fi
if [[ "${PTF_BETA_UPDATE_MODE:-}" != "" ]]; then
  ARGS+=(--ptf-beta-update-mode "${PTF_BETA_UPDATE_MODE}")
fi
if [[ "${PTF_BETA_LOGIT_CLIP:-}" != "" ]]; then
  ARGS+=(--ptf-beta-logit-clip "${PTF_BETA_LOGIT_CLIP}")
fi
if [[ "${PTF_OPTION_SEED:-}" != "" ]]; then
  ARGS+=(--ptf-option-seed "${PTF_OPTION_SEED}")
fi
if [[ "${PTF_EXECUTE_SOURCES:-0}" == "1" ]]; then
  ARGS+=(--ptf-execute-sources)
fi
if [[ "${PTF_OPTION_MIN_STEPS:-}" != "" ]]; then
  ARGS+=(--ptf-option-min-steps "${PTF_OPTION_MIN_STEPS}")
fi
if [[ "${PTF_OPTION_EPSILON_END:-}" != "" ]]; then
  ARGS+=(--ptf-option-epsilon-end "${PTF_OPTION_EPSILON_END}")
fi
if [[ "${PTF_QMP:-0}" == "1" ]]; then
  ARGS+=(--ptf-qmp)
fi
if [[ "${PTF_QMP_FORCE_STUDENT:-0}" == "1" ]]; then
  ARGS+=(--ptf-qmp-force-student)
fi
if [[ "${PTF_MCG:-0}" == "1" ]]; then
  ARGS+=(--ptf-mcg)
fi
if [[ "${PTF_MCG_GROUPS:-}" != "" ]]; then
  ARGS+=(--ptf-mcg-groups "${PTF_MCG_GROUPS}")
fi
if [[ "${PTF_MCG_BEHAVIOR_SOURCE_GROUPS:-}" != "" ]]; then
  ARGS+=(--ptf-mcg-behavior-source-groups "${PTF_MCG_BEHAVIOR_SOURCE_GROUPS}")
fi
if [[ "${PTF_MCG_MARGIN:-}" != "" ]]; then
  ARGS+=(--ptf-mcg-margin "${PTF_MCG_MARGIN}")
fi
if [[ "${PTF_MCG_WARMUP_STEPS:-}" != "" ]]; then
  ARGS+=(--ptf-mcg-warmup-steps "${PTF_MCG_WARMUP_STEPS}")
fi
if [[ "${PTF_MCG_EXEC_PROB:-}" != "" ]]; then
  ARGS+=(--ptf-mcg-exec-prob "${PTF_MCG_EXEC_PROB}")
fi
if [[ "${PTF_MCG_WARMUP_EXEC_PROB:-}" != "" ]]; then
  ARGS+=(--ptf-mcg-warmup-exec-prob "${PTF_MCG_WARMUP_EXEC_PROB}")
fi
if [[ "${PTF_MCG_MIN_STEPS:-}" != "" ]]; then
  ARGS+=(--ptf-mcg-min-steps "${PTF_MCG_MIN_STEPS}")
fi
if [[ "${PTF_MCG_WARMUP_MIN_STEPS:-}" != "" ]]; then
  ARGS+=(--ptf-mcg-warmup-min-steps "${PTF_MCG_WARMUP_MIN_STEPS}")
fi
if [[ "${PTF_MCG_DISTILL_SUBSAMPLE:-}" != "" ]]; then
  ARGS+=(--ptf-mcg-distill-subsample "${PTF_MCG_DISTILL_SUBSAMPLE}")
fi
if [[ "${PTF_MCG_GATE_MODE:-}" != "" ]]; then
  ARGS+=(--ptf-mcg-gate-mode "${PTF_MCG_GATE_MODE}")
fi
if [[ "${PTF_MCG_NULL_QUANTILE:-}" != "" ]]; then
  ARGS+=(--ptf-mcg-null-quantile "${PTF_MCG_NULL_QUANTILE}")
fi
if [[ "${PTF_MCG_CONF_TAU:-}" != "" ]]; then
  ARGS+=(--ptf-mcg-conf-tau "${PTF_MCG_CONF_TAU}")
fi
if [[ "${PTF_MCG_WARMUP_MODE:-}" != "" ]]; then
  ARGS+=(--ptf-mcg-warmup-mode "${PTF_MCG_WARMUP_MODE}")
fi
if [[ "${PTF_MCG_ABLATION:-}" != "" ]]; then
  ARGS+=(--ptf-mcg-ablation "${PTF_MCG_ABLATION}")
fi
if [[ "${PTF_ADMISSION_MODE:-}" != "" ]]; then
  ARGS+=(--ptf-admission-mode "${PTF_ADMISSION_MODE}")
fi
if [[ "${PTF_ADMITTED_SOURCES:-}" != "" ]]; then
  ARGS+=(--ptf-admitted-sources "${PTF_ADMITTED_SOURCES}")
fi
if [[ "${PTF_ADMISSION_MANIFEST:-}" != "" ]]; then
  ARGS+=(--ptf-admission-manifest "${PTF_ADMISSION_MANIFEST}")
fi
if [[ "${PTF_ADMISSION_SCHEDULE:-}" != "" ]]; then
  ARGS+=(--ptf-admission-schedule "${PTF_ADMISSION_SCHEDULE}")
fi
if [[ "${PTF_ADMISSION_TARGET_EVIDENCE:-}" != "" ]]; then
  ARGS+=(--ptf-admission-target-evidence "${PTF_ADMISSION_TARGET_EVIDENCE}")
fi
if [[ "${PTF_ADMISSION_PROBE_STEPS:-}" != "" ]]; then
  ARGS+=(--ptf-admission-probe-steps "${PTF_ADMISSION_PROBE_STEPS}")
fi
if [[ "${PTF_ADMISSION_PROBE_OUTPUT_DIR:-}" != "" ]]; then
  ARGS+=(--ptf-admission-probe-output-dir "${PTF_ADMISSION_PROBE_OUTPUT_DIR}")
fi
if [[ "${PTF_ADMISSION_STUDENT_LOGIT:-}" != "" ]]; then
  ARGS+=(--ptf-admission-student-logit "${PTF_ADMISSION_STUDENT_LOGIT}")
fi
if [[ "${PTF_ADMISSION_REPLAY_RECENCY_HALF_LIFE:-}" != "" ]]; then
  ARGS+=(--ptf-admission-replay-recency-half-life "${PTF_ADMISSION_REPLAY_RECENCY_HALF_LIFE}")
fi
if [[ "${PTF_ADMISSION_REPLAY_UNIFORM_MIX:-}" != "" ]]; then
  ARGS+=(--ptf-admission-replay-uniform-mix "${PTF_ADMISSION_REPLAY_UNIFORM_MIX}")
fi
if [[ "${PTF_ADMISSION_REPLAY_PRIORITY_ALPHA:-}" != "" ]]; then
  ARGS+=(--ptf-admission-replay-priority-alpha "${PTF_ADMISSION_REPLAY_PRIORITY_ALPHA}")
fi
if [[ "${PTF_MCG_EPISODE_PREFIX_STEPS:-}" != "" ]]; then
  ARGS+=(--ptf-mcg-episode-prefix-steps "${PTF_MCG_EPISODE_PREFIX_STEPS}")
fi
if [[ "${PTF_ADMISSION_REPLAY_MODE:-}" != "" ]]; then
  ARGS+=(--ptf-admission-replay-mode "${PTF_ADMISSION_REPLAY_MODE}")
fi
if [[ "${PTF_ADMISSION_REPLAY_HANDOFF:-}" != "" ]]; then
  ARGS+=(--ptf-admission-replay-handoff "${PTF_ADMISSION_REPLAY_HANDOFF}")
fi
if [[ "${PTF_ADMISSION_ADAPTIVE:-0}" == "1" ]]; then
  ARGS+=(--ptf-admission-adaptive)
fi
if [[ "${PTF_ADMISSION_STAGE_WINDOW_STEPS:-}" != "" ]]; then
  ARGS+=(--ptf-admission-stage-window-steps "${PTF_ADMISSION_STAGE_WINDOW_STEPS}")
fi
if [[ "${PTF_ADMISSION_CONFIDENCE_Z:-}" != "" ]]; then
  ARGS+=(--ptf-admission-confidence-z "${PTF_ADMISSION_CONFIDENCE_Z}")
fi
if [[ "${PTF_ADMISSION_MIN_SEGMENTS:-}" != "" ]]; then
  ARGS+=(--ptf-admission-min-segments "${PTF_ADMISSION_MIN_SEGMENTS}")
fi
if [[ "${PTF_ADMISSION_PERSISTENCE:-}" != "" ]]; then
  ARGS+=(--ptf-admission-persistence "${PTF_ADMISSION_PERSISTENCE}")
fi

"${PYTHON_BIN}" -m fasttd3_ptf.official_fasttd3_ptf.train_ptf "${ARGS[@]}"
