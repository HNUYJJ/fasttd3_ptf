#!/usr/bin/env bash
# PARE Experiment A —— fresh phenomenon gate（见 docs/PARE_ALGORITHM_SPEC_v1.md §8 F1）。
#
# 目的**不是**发论文，只回答一个问题：
#     有没有值得 PARE 去解决的 residual problem？
# 即某任务是否同时表现出 (a) early scaffold gain 与 (b) post-exit residual headroom。
# 两个任务都没有 → 不存在 post-release expansion 现象 → 诚实关闭 PARE（F1）。
#
# 四段结构（每 seed 一条串行链，source 曝光固定 10k，**不调 dose**）：
#     A0  prefix   0    → 10k   empty bank 纯 student            → anchor A0
#     A1  scaffold 10k  → 20k   真 bank, ADMISSION_MODE=all      → anchor A1
#         exit     20k  → 100k  从 A1, ADMISSION_MODE=none       （hard exit 臂）
#         scratch  10k  → 100k  从 A0, empty bank                （对照臂）
#
# exit 与 scratch 的 target interactions 完全相同（都到 100k），并共享同一段
# 0-10k prefix；唯一差别是 10k-20k 那段是否有 source scaffold。
#
# 剂量口径：两任务的 admission source mass 都是 **0.500000**。
#   stair  slidesrc bank（1 源 weight=0.0, null_option:false）+ student_logit=0.0
#   truck  hurdle4 bank（4 源）+ student_logit=logsumexp(weights)=14.216676716804526
# 统一剂量是为了排除混淆：否则"stair 有 gain 而 truck 没有"无法区分是任务差异
# 还是剂量差异（CLAUDE.md §8.2）。注意 p0_orchestrator 的 truck student_logit
# =16.4139012941 对应 mass=0.1，是 P0 专用的低剂量口径，**此处不沿用**。
#
# 评估点固定 20k / 50k / 100k，不做全库扫描、不批量重评旧 checkpoint。
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/home/yjj/miniconda3/envs/FastTD3/bin/python}"
GPU="${GPU:?set GPU}"
SEED="${SEED:?set SEED}"
TASK="${TASK:?set TASK: stair|truck}"
PROJECT="${PROJECT:-ptf_fasttd3_pare_gate_a_v1}"
ROOT=artifacts/pare_gate_a_v1
LOGDIR=logs/train/pare_gate_a_v1
mkdir -p "${ROOT}/anchors" "${LOGDIR}"

case "${TASK}" in
  stair)
    ENV_NAME=h1hand-stair-v0
    BANK=configs/source_banks/calibration/h1hand_stair_rbo_slidesrc.yaml
    MCG_GROUPS=legs_torso,arms,hands
    PROV_GROUPS=3
    STUDENT_LOGIT=0.0
    ;;
  truck)
    ENV_NAME=h1hand-truck-v0
    BANK=configs/source_banks/h1hand_hurdle4_wfix_truck.yaml
    # manipulation target：与 p0_orchestrator / run_admission_handoff_v1 /
    # run_phase1_bounded_bank_lease 三处历史配置一致，loco 源不接管 hands。
    MCG_GROUPS=legs_torso,arms
    PROV_GROUPS=2
    STUDENT_LOGIT=14.216676716804526
    ;;
  *) echo "[FATAL] unknown TASK=${TASK}" >&2; exit 2 ;;
esac

A0="${ROOT}/anchors/${TASK}_s${SEED}_k10000"
A1="${ROOT}/anchors/${TASK}_s${SEED}_scaf_k20000"

# ── 前置拒绝：绝不覆盖已有产物（CLAUDE.md §3）────────────────────────
for arm in prefix scaf exit scratch; do
  name="pgav1_${arm}_${TASK}_s${SEED}"
  if compgen -G "models/*${name}__*.pt" >/dev/null || [[ -e "${LOGDIR}/${name}.log" ]]; then
    echo "[REFUSE] existing output for ${name}" >&2; exit 2
  fi
done
for a in "${A0}" "${A1}"; do
  [[ -e "${a}" ]] && { echo "[REFUSE] anchor already exists: ${a}" >&2; exit 2; }
done

common=(
  PYTHONUNBUFFERED=1 MUJOCO_GL=egl WANDB_INIT_TIMEOUT=300 WANDB_SILENT=true
  PYTHON_BIN="${PYTHON_BIN}" CUDA_VISIBLE_DEVICES="${GPU}" DEVICE_RANK=0
  ENV_NAME="${ENV_NAME}" PROJECT="${PROJECT}" SEED="${SEED}"
  TOTAL_TIMESTEPS=100000 NUM_ENVS=128 BATCH_SIZE=32768 BUFFER_SIZE=51200
  LEARNING_STARTS=10 NUM_UPDATES=2 SAVE_INTERVAL=0 EVAL_INTERVAL=0
  RENDER_INTERVAL=0 COMPILE=0 AMP=1 WANDB=1
)

# scaffold 期的 source 配置。bootstrap_only：source 只执行动作并把 target
# reward transition 写进 replay，**无 distillation**——这正是本项目实际成功的
# 路径（见 spec §0），不是原始 PTF 的 option/termination 那套。
scaffold_cfg=(
  SOURCE_BANK="${BANK}"
  PTF_MCG=1 PTF_MCG_GROUPS="${MCG_GROUPS}"
  PTF_MCG_WARMUP_STEPS=100000 PTF_MCG_WARMUP_MIN_STEPS=25
  PTF_MCG_WARMUP_MODE=admission_bootstrap PTF_MCG_ABLATION=bootstrap_only
  PTF_ADMISSION_STUDENT_LOGIT="${STUDENT_LOGIT}"
  PTF_ADMISSION_EXPECTED_SOURCE_MASS=0.5
  PTF_ADMISSION_REPLAY_RECENCY_HALF_LIFE=0
  PTF_ADMISSION_REPLAY_UNIFORM_MIX=1
  PTF_ADMISSION_REPLAY_PRIORITY_ALPHA=0
  PTF_ADMISSION_REPLAY_HANDOFF=physical_after_authority
  PTF_ANCHOR_PROVENANCE_GROUPS="${PROV_GROUPS}"
)

run_arm() {
  local name="$1"; shift
  echo "[$(date -u +%FT%TZ)] START ${name}"
  # 直接重定向而非 tee：tee 会把自己的退出码盖住真实退出码（E7）。
  env "${common[@]}" EXP_NAME="${name}" "$@" \
    bash scripts/official_fasttd3_train_target_ptf.sh \
    > "${LOGDIR}/${name}.log" 2>&1
  echo "[$(date -u +%FT%TZ)] DONE  ${name}"
}

# ── A0：0→10k 纯 student prefix（两臂共享）────────────────────────────
run_arm "pgav1_prefix_${TASK}_s${SEED}" \
  SOURCE_BANK=configs/source_banks/empty.yaml PTF_ADMISSION_MODE=legacy \
  PTF_ANCHOR_PROVENANCE_GROUPS="${PROV_GROUPS}" \
  PTF_ANCHOR_STEP=10000 PTF_ANCHOR_DIR="${A0}" PTF_RUN_STOP_STEP=10000

# scaffold 与 scratch 同从 A0 分叉，用同一 resume noise seed 配对，
# 使二者在 10k→20k 段除 source 有无之外 RNG 起点一致。
NOISE_A0=$((92000 + SEED))
NOISE_A1=$((93000 + SEED))

# ── A1：10k→20k scaffold（source mass 0.5，固定 10k 曝光）─────────────
run_arm "pgav1_scaf_${TASK}_s${SEED}" \
  "${scaffold_cfg[@]}" PTF_ADMISSION_MODE=all \
  PTF_ANCHOR_RESUME="${A0}" PTF_RESUME_NOISE_SEED="${NOISE_A0}" \
  PTF_RUN_STOP_STEP=20000 PTF_EVAL_CHECKPOINT_STEPS=20000 \
  PTF_BRANCH_ANCHOR_STEP=20000 PTF_BRANCH_ANCHOR_DIR="${A1}"

# ── hard-exit 臂：20k→100k，source 永久退出 ───────────────────────────
run_arm "pgav1_exit_${TASK}_s${SEED}" \
  "${scaffold_cfg[@]}" PTF_ADMISSION_MODE=none \
  PTF_ADMISSION_EXPECTED_SOURCE_MASS=0.0 \
  PTF_ANCHOR_RESUME="${A1}" PTF_RESUME_NOISE_SEED="${NOISE_A1}" \
  PTF_RUN_STOP_STEP=100000 PTF_EVAL_CHECKPOINT_STEPS=50000,100000

# ── scratch 对照臂：10k→100k，全程纯 student ──────────────────────────
run_arm "pgav1_scratch_${TASK}_s${SEED}" \
  SOURCE_BANK=configs/source_banks/empty.yaml PTF_ADMISSION_MODE=legacy \
  PTF_ANCHOR_PROVENANCE_GROUPS="${PROV_GROUPS}" \
  PTF_ANCHOR_RESUME="${A0}" PTF_RESUME_NOISE_SEED="${NOISE_A0}" \
  PTF_RUN_STOP_STEP=100000 PTF_EVAL_CHECKPOINT_STEPS=20000,50000,100000

echo "PARE GATE-A ${TASK} SEED ${SEED} COMPLETE"
