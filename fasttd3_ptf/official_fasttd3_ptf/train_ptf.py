from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ["TORCHDYNAMO_INLINE_INBUILT_NN_MODULES"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
if sys.platform != "darwin":
    os.environ["MUJOCO_GL"] = "egl"
else:
    os.environ["MUJOCO_GL"] = "glfw"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["JAX_DEFAULT_MATMUL_PRECISION"] = "highest"

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm
import wandb
from tensordict import TensorDict
from torch.amp import GradScaler, autocast

try:
    import isaacgym  # noqa: F401
except ImportError:
    pass

try:
    import jax.numpy as jnp
except ImportError:
    jnp = None

from fasttd3_ptf.config import load_config
from fasttd3_ptf.ptf.option_module import OptionModule
from fasttd3_ptf.official_fasttd3_ptf import ensure_fasttd3_import_path, ensure_humanoidbench_import_path
from fasttd3_ptf.official_fasttd3_ptf.anchor_io import load_anchor_core, save_anchor_bundle
from fasttd3_ptf.official_fasttd3_ptf.admission_control import (
    AdaptiveAdmissionController,
    build_admission_schedule,
    build_admission_snapshot,
    desired_admission_source_authority,
)
from fasttd3_ptf.official_fasttd3_ptf.target_evidence import TargetEvidenceContract
from fasttd3_ptf.official_fasttd3_ptf.target_evidence_probe import (
    TargetEvidenceProbeProtocol,
    build_top1_admission_snapshot,
    run_target_evidence_probe,
)
from fasttd3_ptf.official_fasttd3_ptf.pare import (
    PARERuntime,
    SourceTransitionReservoir,
    apply_pare_actor_gradient as pare_apply_actor_gradient,
)
from fasttd3_ptf.official_fasttd3_ptf.ptf_replay import PTFReplayWrapper
from fasttd3_ptf.official_fasttd3_ptf.rng_isolation import (
    GlobalRngState,
)
from fasttd3_ptf.ptf.compatibility import gaussian_action_compatibility_all
from fasttd3_ptf.ptf.distillation import masked_action_distillation_loss
from fasttd3_ptf.ptf.mcg import (
    DEFAULT_GROUPS,
    AdmissionSegmentTracker,
    McgBehaviorController,
    ModularGating,
    mcg_distillation_loss,
)
from fasttd3_ptf.ptf.qmp import QmpSelector
from fasttd3_ptf.ptf.option_update import (
    compatible_option_q_loss,
    option_td_target,
    option_u_value,
    released_code_option_u_value,
    select_termination_batch,
    termination_loss_at_next_state,
)
from fasttd3_ptf.ptf.option_selector import OptionSelector
from fasttd3_ptf.ptf.source_bank import SourcePolicyBank
from fasttd3_ptf.utils.schedules import LinearScheduler, ReleasedPTFTanhScheduler

ensure_fasttd3_import_path()
ensure_humanoidbench_import_path()

from fast_td3_utils import (  # type: ignore  # noqa: E402
    EmpiricalNormalization,
    PerTaskRewardNormalizer,
    RewardNormalizer,
    SimpleReplayBuffer,
    cpu_state,
    mark_step,
)
from hyperparams import get_args  # type: ignore  # noqa: E402

torch.set_float32_matmul_precision("high")


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_ptf_cli() -> dict[str, Any]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--ptf_config", "--ptf-config", default=None)
    parser.add_argument(
        "--ptf_released_code_fidelity",
        "--ptf-released-code-fidelity",
        action="store_true",
        help=(
            "Isolated PTF author-released-code fidelity mode. This restores "
            "the released option/termination dynamics while retaining only "
            "the necessary FastTD3/HumanoidBench adapters."
        ),
    )
    parser.add_argument("--ptf_source_bank", "--ptf-source-bank", default=None)
    parser.add_argument("--ptf_option_hidden_dims", "--ptf-option-hidden-dims", default=None)
    parser.add_argument("--ptf_option_lr", "--ptf-option-lr", type=float, default=None)
    parser.add_argument("--ptf_beta_lr", "--ptf-beta-lr", type=float, default=None)
    parser.add_argument(
        "--ptf_option_reward_scale",
        "--ptf-option-reward-scale",
        type=float,
        default=None,
        help=(
            "Fixed reward scale used only by Q_omega TD targets; the FastTD3 "
            "critic reward path is unchanged."
        ),
    )
    parser.add_argument("--ptf_beta_warmup_steps", "--ptf-beta-warmup-steps", type=int, default=None)
    parser.add_argument(
        "--ptf_beta_update_mode",
        "--ptf-beta-update-mode",
        choices=["replay", "current_transition"],
        default=None,
    )
    parser.add_argument(
        "--ptf_beta_logit_clip",
        "--ptf-beta-logit-clip",
        type=float,
        default=None,
        help=(
            "Optional straight-through forward clamp for beta-head logits; "
            "ordinary hard clamp is intentionally not used because it has zero outside gradient."
        ),
    )
    parser.add_argument("--ptf_option_tau", "--ptf-option-tau", type=float, default=None)
    parser.add_argument("--ptf_option_seed", "--ptf-option-seed", type=int, default=None)
    parser.add_argument("--ptf_transfer_lambda_start", "--ptf-transfer-lambda-start", type=float, default=None)
    parser.add_argument("--ptf_transfer_lambda_end", "--ptf-transfer-lambda-end", type=float, default=None)
    parser.add_argument("--ptf_transfer_decay_steps", "--ptf-transfer-decay-steps", type=int, default=None)
    parser.add_argument("--ptf_option_epsilon_start", "--ptf-option-epsilon-start", type=float, default=None)
    parser.add_argument("--ptf_option_epsilon_end", "--ptf-option-epsilon-end", type=float, default=None)
    parser.add_argument("--ptf_option_epsilon_decay_steps", "--ptf-option-epsilon-decay-steps", type=int, default=None)
    parser.add_argument("--ptf_transfer_loss", "--ptf-transfer-loss", default=None)
    parser.add_argument("--ptf_transfer_huber_delta", "--ptf-transfer-huber-delta", type=float, default=None)
    parser.add_argument("--ptf_transfer_loss_clip", "--ptf-transfer-loss-clip", type=float, default=None)
    parser.add_argument("--ptf_xi", "--ptf-xi", type=float, default=None)
    parser.add_argument("--ptf_no_beta_weighted_transfer", "--ptf-no-beta-weighted-transfer", action="store_true")
    parser.add_argument("--ptf_no_update_all_compatible_options", "--ptf-no-update-all-compatible-options", action="store_true")
    # 行为层 call-and-return(PTF 原文语义):rollout 时被选中的非 null 源 option
    # 在其 action mask 内的维度真正执行到环境(mask 外维度仍由 actor 控制),
    # buffer 记录实际执行的动作。distill-only 模式下 buffer 缺少"源带到的状态",
    # 蒸馏在分布外失效(pilot v2 的根因,见 docs/experiments)。
    parser.add_argument("--ptf_execute_sources", "--ptf-execute-sources", action="store_true")
    parser.add_argument("--ptf_option_min_steps", "--ptf-option-min-steps", type=int, default=None)
    # MCG(Modular Critic-Guided transfer):option 推广为(教师,身体组),调度/蒸馏
    # gating 由 target critic 的 masked-candidate Δ 直接给出(不学 Q_o)。
    # QMP-fidelity behavior-only 模式(run card docs/run_card_qmp_fidelity_v1.md):
    # per-state 完整策略 argmax min_h Q_h;复用 target-only 的 classic-PTF 隔离,
    # 只替换 rollout 的动作选择。与 MCG 互斥。
    parser.add_argument("--ptf_qmp", "--ptf-qmp", action="store_true")
    # smoke-only:强制 QMP 退化为纯 student 选择,用于 run card §7 的隔离等价性
    # 检查(验证 QMP 分支除动作选择外没有引入任何副作用)。**不用于正式实验。**
    parser.add_argument("--ptf_qmp_force_student", "--ptf-qmp-force-student",
                        action="store_true")
    parser.add_argument("--ptf_mcg", "--ptf-mcg", action="store_true")
    parser.add_argument("--ptf_mcg_groups", "--ptf-mcg-groups", default=None)
    parser.add_argument("--ptf_mcg_margin", "--ptf-mcg-margin", type=float, default=None)
    parser.add_argument("--ptf_mcg_warmup_steps", "--ptf-mcg-warmup-steps", type=int, default=None)
    parser.add_argument("--ptf_mcg_exec_prob", "--ptf-mcg-exec-prob", type=float, default=None)
    parser.add_argument("--ptf_mcg_warmup_exec_prob", "--ptf-mcg-warmup-exec-prob", type=float, default=None)
    parser.add_argument("--ptf_mcg_min_steps", "--ptf-mcg-min-steps", type=int, default=None)
    parser.add_argument("--ptf_mcg_warmup_min_steps", "--ptf-mcg-warmup-min-steps", type=int, default=None)
    # episode-prefix handoff：source 只在 episode 前缀连续执行,之后锁定 student。
    # 不设则保持历史的随机碎片式 latch 行为。
    parser.add_argument("--ptf_mcg_episode_prefix_steps", "--ptf-mcg-episode-prefix-steps", type=int, default=None)
    parser.add_argument("--ptf_mcg_distill_subsample", "--ptf-mcg-distill-subsample", type=int, default=None)
    # v1.1: gate 模式(null=显著性校准, sign=v1 的 Δ>margin,留作 ablation)
    parser.add_argument("--ptf_mcg_gate_mode", "--ptf-mcg-gate-mode", choices=["null", "sign"], default=None)
    # warmup 调度模式(活跃四种;历史的 chain 模式随 package 专项一并移除):
    # random=均匀抽源(rand 对照组);
    # safe_bootstrap=静态 RBO 主方法,按 T⁰ probe 的 reward-bearing weight 抽源
    #   +bank horizon 锁存(weight 是相对 allocation prior 而非 ROI);
    # online_bootstrap=student-as-arm(T^critic 线 Step A):学生与教师作(S+1)个
    #   平等 arm,权重为执行期 reward 在线 EMA;
    # admission_bootstrap=admission lifecycle 主路径:admitted 源+student 进单一
    #   categorical(softmax([source_logits, student_logit]/τ)),rejected 源=-inf。
    parser.add_argument("--ptf_mcg_warmup_mode", "--ptf-mcg-warmup-mode",
                        choices=["random", "safe_bootstrap", "online_bootstrap", "admission_bootstrap"], default=None)
    # Metric-agnostic admission snapshot.  The decision source is explicit;
    # automatic utility estimation is deliberately outside this implementation.
    parser.add_argument("--ptf_admission_mode", "--ptf-admission-mode",
                        choices=["legacy", "all", "none", "static", "manifest", "schedule", "target_evidence"], default=None)
    parser.add_argument("--ptf_admitted_sources", "--ptf-admitted-sources", default=None)
    parser.add_argument("--ptf_admission_manifest", "--ptf-admission-manifest", default=None)
    parser.add_argument("--ptf_admission_schedule", "--ptf-admission-schedule", default=None)
    parser.add_argument(
        "--ptf_admission_target_evidence",
        "--ptf-admission-target-evidence",
        default=None,
    )
    parser.add_argument(
        "--ptf_admission_probe_steps",
        "--ptf-admission-probe-steps",
        default=None,
        help="Comma-separated completed learner steps for low-frequency target-evidence probes.",
    )
    parser.add_argument(
        "--ptf_admission_probe_output_dir",
        "--ptf-admission-probe-output-dir",
        default=None,
    )
    parser.add_argument("--ptf_admission_student_logit", "--ptf-admission-student-logit", type=float, default=None)
    parser.add_argument("--ptf_admission_replay_recency_half_life", "--ptf-admission-replay-recency-half-life", type=float, default=None)
    parser.add_argument("--ptf_admission_replay_uniform_mix", "--ptf-admission-replay-uniform-mix", type=float, default=None)
    parser.add_argument("--ptf_admission_replay_priority_alpha", "--ptf-admission-replay-priority-alpha", type=float, default=None)
    parser.add_argument(
        "--ptf_admission_replay_handoff",
        "--ptf-admission-replay-handoff",
        choices=["fixed_quota", "physical_after_authority"],
        default=None,
    )
    # 迁移通道解耦(Door@10k 结果驱动):behavior authority 与 replay eligibility
    # 在 admission_bootstrap 下本来共用同一个 student-inclusive categorical。
    # 该开关只覆盖 **replay 侧** 配额,behavior 侧一律保持原配置。
    parser.add_argument(
        "--ptf_admission_replay_mode",
        "--ptf-admission-replay-mode",
        # physical(T4-R)：source 保持 behavior authority，但 replay 始终
        # physical-uniform over allowed slots，使 q_S 跟随 rho_S 而非固定 quota。
        choices=["shared", "student_only", "physical"],
        default=None,
    )
    parser.add_argument(
        "--ptf_admission_adaptive",
        "--ptf-admission-adaptive",
        action="store_true",
    )
    parser.add_argument(
        "--ptf_admission_stage_window_steps",
        "--ptf-admission-stage-window-steps",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--ptf_admission_confidence_z",
        "--ptf-admission-confidence-z",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--ptf_admission_min_segments",
        "--ptf-admission-min-segments",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--ptf_admission_persistence",
        "--ptf-admission-persistence",
        type=int,
        default=None,
    )
    # online_bootstrap 超参: tau=在线 softmax 温度(作用在跨 arm zscore 上,尺度
    # 不变); eps=uniform 探索下限(防 EMA 死锁); prior_frac=先验分支线性衰减窗口
    # 占 warmup 的比例; ema_n=EMA 等效窗口(样本数)
    parser.add_argument("--ptf_mcg_online_tau", "--ptf-mcg-online-tau", type=float, default=None)
    parser.add_argument("--ptf_mcg_online_eps", "--ptf-mcg-online-eps", type=float, default=None)
    parser.add_argument("--ptf_mcg_online_prior_frac", "--ptf-mcg-online-prior-frac", type=float, default=None)
    parser.add_argument("--ptf_mcg_online_ema_n", "--ptf-mcg-online-ema-n", type=float, default=None)
    # horizon-arm(ChatGPT 裁定 2026-07-03): 逗号分隔档位如 "25,50" → 每源
    # len(H) 个 (source,horizon) arm, arm 空间 S×H+1。stair 的 horizon 敏感
    # 边界(h25 全负而 safe h50 超 scr)由 arm 竞争自动解决, 禁任务名分支。
    parser.add_argument("--ptf_mcg_online_horizons", "--ptf-mcg-online-horizons", type=str, default=None)
    # Step B(导师意见1): replay 按 arm value 对源驻留轨迹降权采样(只降不升)。
    # 动机: Step A 证明减少坏教师执行不足以修 crawl——buffer 残留毒害是主因。
    parser.add_argument("--ptf_mcg_replay_reweight", "--ptf-mcg-replay-reweight", action="store_true")
    parser.add_argument("--ptf_mcg_replay_tau", "--ptf-mcg-replay-tau", type=float, default=None)
    parser.add_argument("--ptf_mcg_replay_floor", "--ptf-mcg-replay-floor", type=float, default=None)
    # actor/critic split replay(ChatGPT 第二轮裁定 2026-07-02): 坏源数据对
    # critic 是有用负样本(保守降权),对 actor 状态分布是污染(强降权)。
    # both=obrw(双路径同权重 floor=replay_floor); actor_only/critic_only=归因
    # 消融(同 floor 只差路径,变量控制); split=设计版(actor_floor/critic_floor)。
    parser.add_argument("--ptf_mcg_replay_mode", "--ptf-mcg-replay-mode",
                        choices=["off", "both", "actor_only", "critic_only", "split"], default=None)
    parser.add_argument("--ptf_mcg_replay_actor_floor", "--ptf-mcg-replay-actor-floor", type=float, default=None)
    parser.add_argument("--ptf_mcg_replay_critic_floor", "--ptf-mcg-replay-critic-floor", type=float, default=None)
    # T-gated transfer/abstain(ChatGPT 裁定 2026-07-02, 合并硬 abstain 与降权自适应
    # 激活): 全源劣于 student−δ 持续 K 步 → abstain(exec→probe floor + replay 源
    # 权重→floor); 否则 transfer mode(replay 保持 uniform, 修 pole 误伤)。
    parser.add_argument("--ptf_mcg_abstain_gate", "--ptf-mcg-abstain-gate", action="store_true")
    parser.add_argument("--ptf_mcg_abstain_delta_frac", "--ptf-mcg-abstain-delta-frac", type=float, default=None)
    parser.add_argument("--ptf_mcg_abstain_k_steps", "--ptf-mcg-abstain-k-steps", type=int, default=None)
    parser.add_argument("--ptf_mcg_abstain_eps", "--ptf-mcg-abstain-eps", type=float, default=None)
    parser.add_argument("--ptf_mcg_abstain_replay_floor", "--ptf-mcg-abstain-replay-floor", type=float, default=None)
    # ablation(分离三个贡献来源,ChatGPT 2026-06-13): full=warmup bootstrap+gate
    # 蒸馏; bootstrap_only=只 warmup 教师执行,gate 期纯 student、无蒸馏(回答
    # "提升是否全靠 warmup 灌 replay"); no_bootstrap=warmup 期纯 student,只 gate
    # 蒸馏(隔离 gate/modular 的独立贡献)。
    parser.add_argument("--ptf_mcg_ablation", "--ptf-mcg-ablation",
                        choices=["full", "bootstrap_only", "no_bootstrap"], default=None)
    parser.add_argument("--ptf_mcg_null_quantile", "--ptf-mcg-null-quantile", type=float, default=None)
    parser.add_argument("--ptf_mcg_conf_tau", "--ptf-mcg-conf-tau", type=float, default=None)
    # paper-anchor 快照钩子:在指定步保存 learner+replay+rng 的完整快照(anchor
    # bundle),供 paired probe/组件①迁移性度量实验离线复用;与训练主链无耦合。
    parser.add_argument("--ptf_anchor_step", "--ptf-anchor-step", type=int, default=None)
    parser.add_argument("--ptf_anchor_dir", "--ptf-anchor-dir", default=None)
    # IBR branch anchor: unlike paper-anchor, this snapshot is allowed to come
    # from a source-bearing/resumed branch.  It captures the complete learner,
    # replay and RNG state at the intervention boundary so the selected branch
    # can continue under exact abstention without rebuilding its learning state.
    parser.add_argument(
        "--ptf_branch_anchor_step", "--ptf-branch-anchor-step", type=int, default=None
    )
    parser.add_argument(
        "--ptf_branch_anchor_dir", "--ptf-branch-anchor-dir", default=None
    )
    # anchor 保存侧 provenance 组数(P0 优先方案):按任务目标 authority 组数
    # 保存(crawl=3/truck=2),使分支 resume 时 schema 精确匹配、零重建——
    # segment_id/env_rank/learner_step 等标量元数据原样保留。
    parser.add_argument("--ptf_anchor_provenance_groups", "--ptf-anchor-provenance-groups", type=int, default=None)
    # P0 core-only anchor-resume(run card v2.1.2 附录 A):白名单恢复核心
    # learner 后继续真实在线训练。run_stop_step 独立控制训练退出——
    # total_timesteps 保持不变以维持 LR 余弦日程;eval_checkpoint_steps 为
    # 显式保存步列表(固定 save_interval 的整除机制覆盖不了任意步);
    # resume_noise_seed 用于 fresh reset 后 noise_scales 的配对重采样。
    parser.add_argument("--ptf_anchor_resume", "--ptf-anchor-resume", default=None)
    # PARE(docs/PARE_ALGORITHM_SPEC_v1.md):post-release 的 provenance-aware
    # occupancy repulsion。release 点就是本 run 的起点——实验设计是
    # branch-at-release(从同一 A1 anchor 分叉 hard-exit 与 PARE 两臂),
    # 故 PARE 臂必须与 admission_mode=none 同用,且 replay 里已有 z=1 历史。
    # 关闭时训练循环不构造任何 PARE 对象,代码路径与既有 hard-exit 逐位一致。
    parser.add_argument("--ptf_pare", "--ptf-pare", action="store_true")
    parser.add_argument("--ptf_pare_reservoir_capacity",
                        "--ptf-pare-reservoir-capacity", type=int, default=262144)
    parser.add_argument("--ptf_pare_d_lr", "--ptf-pare-d-lr", type=float, default=3e-4)
    parser.add_argument("--ptf_run_stop_step", "--ptf-run-stop-step", type=int, default=None)
    parser.add_argument("--ptf_eval_checkpoint_steps", "--ptf-eval-checkpoint-steps", default=None)
    parser.add_argument("--ptf_resume_noise_seed", "--ptf-resume-noise-seed", type=int, default=None)
    # Critic-first bridge bootstrap: keep collecting target-grounded source
    # experience and updating the target critic, but delay target-actor updates
    # until this absolute completed learner step.  None preserves the historical
    # FastTD3 update schedule exactly.
    parser.add_argument(
        "--ptf_actor_update_start_step",
        "--ptf-actor-update-start-step",
        type=int,
        default=None,
    )
    # 预注册的源总 mass 运行时断言(run card §5):student_logit 的 float64 解析
    # 值只是配置输入,实际 softmax 走 float32(admission_control.py)——启动时以
    # 真实路径核对 |源总 mass − 预期| ≤ 1e-6,不声称运行时精确相等。
    parser.add_argument("--ptf_admission_expected_source_mass", "--ptf-admission-expected-source-mass", type=float, default=None)
    ptf_ns, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]

    cfg: dict[str, Any] = {}
    if ptf_ns.ptf_config:
        loaded = load_config(ptf_ns.ptf_config)
        cfg.update(loaded.get("ptf", loaded))
    if ptf_ns.ptf_source_bank is not None:
        cfg["source_bank"] = ptf_ns.ptf_source_bank
    if ptf_ns.ptf_released_code_fidelity:
        cfg["released_code_fidelity"] = True
    if ptf_ns.ptf_option_hidden_dims is not None:
        cfg["option_hidden_dims"] = [
            int(x.strip()) for x in ptf_ns.ptf_option_hidden_dims.split(",") if x.strip()
        ]
    for key in [
        "option_lr",
        "beta_lr",
        "option_reward_scale",
        "beta_warmup_steps",
        "beta_update_mode",
        "beta_logit_clip",
        "option_tau",
        "option_seed",
        "transfer_lambda_start",
        "transfer_lambda_end",
        "transfer_decay_steps",
        "option_epsilon_start",
        "option_epsilon_end",
        "option_epsilon_decay_steps",
        "transfer_loss",
        "transfer_huber_delta",
        "transfer_loss_clip",
        "xi",
        "option_min_steps",
    ]:
        value = getattr(ptf_ns, f"ptf_{key}")
        if value is not None:
            cfg[key] = value
    if ptf_ns.ptf_execute_sources:
        cfg["execute_sources"] = True
    if ptf_ns.ptf_qmp:
        cfg["qmp"] = True
    if ptf_ns.ptf_qmp_force_student:
        cfg["qmp_force_student"] = True
    if ptf_ns.ptf_mcg:
        cfg["mcg"] = True
    if ptf_ns.ptf_mcg_groups is not None:
        cfg["mcg_groups"] = [x.strip() for x in ptf_ns.ptf_mcg_groups.split(",") if x.strip()]
    for key in [
        "mcg_margin",
        "mcg_warmup_steps",
        "mcg_exec_prob",
        "mcg_warmup_exec_prob",
        "mcg_min_steps",
        "mcg_warmup_min_steps",
        "mcg_episode_prefix_steps",
        "mcg_distill_subsample",
        "mcg_gate_mode",
        "mcg_null_quantile",
        "mcg_conf_tau",
        "mcg_warmup_mode",
        "mcg_ablation",
        "mcg_online_tau",
        "mcg_online_eps",
        "mcg_online_prior_frac",
        "mcg_online_ema_n",
        "mcg_online_horizons",
        "mcg_replay_tau",
        "mcg_replay_floor",
        "mcg_replay_mode",
        "mcg_replay_actor_floor",
        "mcg_replay_critic_floor",
        "mcg_abstain_delta_frac",
        "mcg_abstain_k_steps",
        "mcg_abstain_eps",
        "mcg_abstain_replay_floor",
        "admission_mode",
        "admitted_sources",
        "admission_manifest",
        "admission_schedule",
        "admission_target_evidence",
        "admission_probe_output_dir",
        "admission_student_logit",
        "admission_replay_recency_half_life",
        "admission_replay_uniform_mix",
        "admission_replay_priority_alpha",
        "admission_replay_handoff",
        "admission_replay_mode",
        "admission_stage_window_steps",
        "admission_confidence_z",
        "admission_min_segments",
        "admission_persistence",
    ]:
        value = getattr(ptf_ns, f"ptf_{key}")
        if value is not None:
            cfg[key] = value
    if ptf_ns.ptf_admission_probe_steps is not None:
        cfg["admission_probe_steps"] = [
            int(value.strip())
            for value in ptf_ns.ptf_admission_probe_steps.split(",")
            if value.strip()
        ]
    if ptf_ns.ptf_mcg_replay_reweight:
        cfg["mcg_replay_reweight"] = True
    if ptf_ns.ptf_mcg_abstain_gate:
        cfg["mcg_abstain_gate"] = True
    if ptf_ns.ptf_admission_adaptive:
        cfg["admission_adaptive"] = True
    if ptf_ns.ptf_no_beta_weighted_transfer:
        cfg["beta_weighted_transfer"] = False
    if ptf_ns.ptf_no_update_all_compatible_options:
        cfg["update_all_compatible_options"] = False
    if ptf_ns.ptf_anchor_step is not None:
        cfg["anchor_step"] = int(ptf_ns.ptf_anchor_step)
    if ptf_ns.ptf_anchor_dir is not None:
        cfg["anchor_dir"] = str(ptf_ns.ptf_anchor_dir)
    if ptf_ns.ptf_branch_anchor_step is not None:
        cfg["branch_anchor_step"] = int(ptf_ns.ptf_branch_anchor_step)
    if ptf_ns.ptf_branch_anchor_dir is not None:
        cfg["branch_anchor_dir"] = str(ptf_ns.ptf_branch_anchor_dir)
    if ptf_ns.ptf_anchor_provenance_groups is not None:
        cfg["anchor_provenance_groups"] = int(ptf_ns.ptf_anchor_provenance_groups)
    if ptf_ns.ptf_anchor_resume is not None:
        cfg["anchor_resume"] = str(ptf_ns.ptf_anchor_resume)
    if ptf_ns.ptf_pare:
        cfg["pare"] = True
    cfg["pare_reservoir_capacity"] = int(ptf_ns.ptf_pare_reservoir_capacity)
    cfg["pare_d_lr"] = float(ptf_ns.ptf_pare_d_lr)
    if ptf_ns.ptf_run_stop_step is not None:
        cfg["run_stop_step"] = int(ptf_ns.ptf_run_stop_step)
    if ptf_ns.ptf_eval_checkpoint_steps is not None:
        cfg["eval_checkpoint_steps"] = [
            int(x.strip()) for x in ptf_ns.ptf_eval_checkpoint_steps.split(",") if x.strip()
        ]
    if ptf_ns.ptf_resume_noise_seed is not None:
        cfg["resume_noise_seed"] = int(ptf_ns.ptf_resume_noise_seed)
    if ptf_ns.ptf_actor_update_start_step is not None:
        cfg["actor_update_start_step"] = int(ptf_ns.ptf_actor_update_start_step)
    if ptf_ns.ptf_admission_expected_source_mass is not None:
        cfg["admission_expected_source_mass"] = float(ptf_ns.ptf_admission_expected_source_mass)
    cfg.setdefault("released_code_fidelity", False)
    cfg.setdefault("option_hidden_dims", [256, 256])
    cfg.setdefault("option_lr", 3e-4)
    cfg.setdefault("beta_lr", 1e-4)
    cfg.setdefault("option_reward_scale", 1.0)
    if float(cfg["option_reward_scale"]) <= 0.0:
        raise ValueError("option_reward_scale must be positive")
    cfg.setdefault("option_batch_size", 32)
    cfg.setdefault("option_grad_clip", None)
    # β warmup: PTF learns β in lockstep with Q_o; we delay β's gradient updates
    # until Q_o has had enough steps to become at least locally meaningful.
    # Early in training Q_o is dominated by noise (and the on-/off-policy gap in
    # the docstring above), so learning β from it would push termination toward
    # arbitrary rails. During warmup β is only evaluated (no-grad) so it stays
    # usable as a forward-only distillation weight in compute_transfer_loss.
    cfg.setdefault("beta_warmup_steps", 20000)
    # The historical FastTD3 adaptation sampled both Q_omega and beta from
    # replay. The released PTF code instead uses replay only for Q_omega and
    # trains beta from the current transition/current option.
    cfg.setdefault("beta_update_mode", "replay")
    if str(cfg["beta_update_mode"]) not in {"replay", "current_transition"}:
        raise ValueError(
            "beta_update_mode must be 'replay' or 'current_transition', got "
            f"{cfg['beta_update_mode']!r}"
        )
    cfg.setdefault("beta_logit_clip", None)
    if cfg["beta_logit_clip"] is not None and float(cfg["beta_logit_clip"]) <= 0.0:
        raise ValueError("beta_logit_clip must be positive when enabled")
    cfg.setdefault("option_tau", 0.05)
    cfg.setdefault("option_target_update_interval", 1000)
    cfg.setdefault("transfer_schedule", "linear")
    cfg.setdefault("transfer_lambda_start", 0.2)
    cfg.setdefault("transfer_lambda_end", 0.0)
    cfg.setdefault("transfer_decay_steps", 300000)
    cfg.setdefault("option_epsilon_start", 0.3)
    cfg.setdefault("option_epsilon_end", 0.05)
    cfg.setdefault("option_epsilon_decay_steps", 50000)
    cfg.setdefault("transfer_loss", "huber")
    cfg.setdefault("transfer_huber_delta", 1.0)
    cfg.setdefault("transfer_loss_clip", None)
    # xi=0 follows the original PTF code path: adaptive 0.8 * (top1Q - top2Q).
    cfg.setdefault("xi", 0.0)
    cfg.setdefault("beta_weighted_transfer", True)
    cfg.setdefault("execute_sources", False)
    cfg.setdefault("option_min_steps", 1)
    cfg.setdefault("update_all_compatible_options", True)
    cfg.setdefault("anchor_step", None)
    cfg.setdefault("anchor_dir", None)
    cfg.setdefault("branch_anchor_step", None)
    cfg.setdefault("branch_anchor_dir", None)
    cfg.setdefault("anchor_provenance_groups", None)
    cfg.setdefault("anchor_resume", None)
    cfg.setdefault("pare", False)
    cfg.setdefault("pare_reservoir_capacity", 262144)
    cfg.setdefault("pare_d_lr", 3e-4)
    cfg.setdefault("run_stop_step", None)
    cfg.setdefault("eval_checkpoint_steps", None)
    cfg.setdefault("resume_noise_seed", None)
    cfg.setdefault("actor_update_start_step", None)
    cfg.setdefault("admission_expected_source_mass", None)
    # MCG 默认值。warmup=15000 依据离线探针:5k 的 critic 高估学生全拒教师,
    # 25k 已有 part 级结构;margin=0 配 mean-Q 比较(q10 实测过保守)。
    cfg.setdefault("mcg", False)
    cfg.setdefault("mcg_groups", list(DEFAULT_GROUPS))
    cfg.setdefault("mcg_margin", 0.0)
    cfg.setdefault("mcg_warmup_steps", 15000)
    cfg.setdefault("mcg_exec_prob", 0.3)
    cfg.setdefault("mcg_warmup_exec_prob", 0.5)
    cfg.setdefault("mcg_min_steps", 10)
    cfg.setdefault("mcg_warmup_min_steps", 25)
    cfg.setdefault("mcg_episode_prefix_steps", None)
    cfg.setdefault("mcg_distill_subsample", 8192)
    # v1.1 默认走 null 校准 gate(sign 模式在 window 上产生过 −153 净负迁移)。
    cfg.setdefault("mcg_gate_mode", "null")
    cfg.setdefault("mcg_null_quantile", 0.95)
    cfg.setdefault("qmp", False)
    cfg.setdefault("qmp_force_student", False)
    cfg.setdefault("mcg_conf_tau", 0.1)
    cfg.setdefault("mcg_warmup_mode", "random")
    cfg.setdefault("mcg_bootstrap_tau", 1.0)  # safe_bootstrap 抽源 softmax 温度
    cfg.setdefault("mcg_ablation", "full")
    # online_bootstrap(student-as-arm)默认: eps=0.1 保证每 arm 持续被探索;
    # prior_frac=0.3 → warmup 前 30% 先验分支线性衰减(冷启动数据积累);
    # ema_n=2000 ≈ 每 arm 几十个 horizon 段的等效窗口。
    # tau=0.5: 4-5 个 arm 的 zscore 动态范围仅 ±1.5σ, tau=1.0 时"一好三坏"场景
    # 好 arm 只拿 ~65%(单元自测), 0.5 才达 ~87%(负迁移有效关闭且留 eps 探测)
    cfg.setdefault("mcg_online_tau", 0.5)
    cfg.setdefault("mcg_online_eps", 0.1)
    cfg.setdefault("mcg_online_prior_frac", 0.3)
    cfg.setdefault("mcg_online_ema_n", 2000.0)
    # horizon-arm 默认关(None=单 horizon, 走 bank 的 per-source horizon)
    cfg.setdefault("mcg_online_horizons", None)
    # Step B replay 重加权默认关(向后兼容); tau=1.0 配 zscore 尺度,
    # floor=0.1 保留坏源轨迹的负样本价值(off-policy 下坏轨迹≠无用)
    cfg.setdefault("mcg_replay_reweight", False)
    cfg.setdefault("mcg_replay_tau", 1.0)
    cfg.setdefault("mcg_replay_floor", 0.1)
    # split replay 默认: off; actor_floor=0.05(强,防状态分布污染)、
    # critic_floor=0.4(保守,保留负样本)仅 split 模式用;
    # actor_only/critic_only 用统一 mcg_replay_floor(变量控制,只差路径)
    cfg.setdefault("mcg_replay_mode", "off")
    cfg.setdefault("mcg_replay_actor_floor", 0.05)
    cfg.setdefault("mcg_replay_critic_floor", 0.4)
    # T-gated abstain 默认: delta_frac=0.5(crawl/pole 已有数据预演正确分类);
    # k_steps=2000(~80 horizon 段防抖); abstain_eps=0.02(probe floor, ChatGPT
    # 建议 0.01-0.02); abstain 时 replay 源权重 floor=0.05(比 obrw 常开 0.1 更强,
    # 因为只在确认全源劣势后激活)
    cfg.setdefault("mcg_abstain_gate", False)
    cfg.setdefault("mcg_abstain_delta_frac", 0.5)
    cfg.setdefault("mcg_abstain_k_steps", 2000)
    cfg.setdefault("mcg_abstain_eps", 0.02)
    cfg.setdefault("mcg_abstain_replay_floor", 0.05)
    cfg.setdefault("admission_mode", "legacy")
    cfg.setdefault("admitted_sources", None)
    cfg.setdefault("admission_manifest", None)
    cfg.setdefault("admission_schedule", None)
    cfg.setdefault("admission_target_evidence", None)
    cfg.setdefault("admission_probe_steps", None)
    cfg.setdefault("admission_probe_output_dir", None)
    cfg.setdefault("admission_student_logit", 0.0)
    cfg.setdefault("admission_replay_recency_half_life", 5000.0)
    cfg.setdefault("admission_replay_uniform_mix", 0.05)
    cfg.setdefault("admission_replay_priority_alpha", 0.0)
    # Backward compatible by default. Formal lifecycle-fix runs opt into the
    # authority-coupled physical handoff explicitly.
    cfg.setdefault("admission_replay_handoff", "fixed_quota")
    cfg.setdefault("admission_replay_mode", "shared")
    cfg.setdefault("admission_adaptive", False)
    cfg.setdefault("admission_stage_window_steps", 3000)
    cfg.setdefault("admission_confidence_z", 1.645)
    cfg.setdefault("admission_min_segments", 20)
    cfg.setdefault("admission_persistence", 3)

    _validate_released_code_fidelity_config(cfg)
    return cfg


def replay_candidate_masses(
    candidate_masses: "torch.Tensor", replay_mode: str
) -> "torch.Tensor":
    """决定 **replay 侧** 的来源配额,把两条迁移通道解耦。

    在 ``warmup_mode=admission_bootstrap`` 下,谁执行(behavior authority)与
    critic 按什么来源配额采样(replay eligibility)本来共用同一个
    student-inclusive categorical(见 ``McgBehaviorController.step`` 的
    admission_bootstrap 分支:直接对该分布 multinomial,没有外层 teacher
    Bernoulli)。Door@10k 的结果表明这两条通道可能方向相反,故需要分别控制。

    本函数**只**作用于 replay 侧;behavior 侧永远使用未经改动的
    ``candidate_masses``,因此:

    - ``shared``(默认):完全等价于历史行为,任何既有实验的数值不变;
    - ``student_only``:critic 只采 student provenance 的 slot(source 配额恒 0),
      而 source 仍照常获得 behavior authority、照常写入 physical buffer。
      这实现 B-only 臂 :math:`(B{=}1, R{=}0)`。
    """
    if replay_mode == "shared":
        return candidate_masses
    if replay_mode == "student_only":
        masses = torch.zeros_like(candidate_masses)
        masses[-1] = 1.0
        return masses
    if replay_mode == "physical":
        # physical 模式不经 stratum quota：采样由
        # ``PTFReplayWrapper.set_admission_replay_physical`` 切到 physical-uniform，
        # 这里返回原 masses 只为让 admitted/rejected 掩码与审计记录保持一致。
        return candidate_masses
    raise ValueError(f"unknown admission_replay_mode: {replay_mode}")


@torch.no_grad()
def apply_runtime_admission_policy_after_resume(
    *,
    replay: PTFReplayWrapper,
    behavior: McgBehaviorController,
    snapshot: Any,
    device: torch.device | str,
    bootstrap_tau: float,
    replay_mode: str,
    recency_half_life: float,
    uniform_mix: float,
    priority_alpha: float,
) -> torch.Tensor:
    """Make runtime admission authoritative after importing an anchor replay.

    ``load_anchor_core`` deliberately imports the complete historical replay
    state, including the branch's old admission mask and masses.  A continuation
    treatment (notably exact abstention) must override those historical controls
    immediately; otherwise a selected source branch silently keeps sampling its
    source data.  Counts and immutable source bytes remain intact for auditing.
    """

    admitted = snapshot.admitted_tensor(device)
    masses = snapshot.candidate_probabilities(
        tau=float(bootstrap_tau), device=device
    )
    behavior.set_admission_policy(
        admitted_sources=admitted,
        source_logits=torch.tensor(
            snapshot.source_logits, device=device, dtype=torch.float32
        ),
        student_logit=snapshot.student_logit,
    )
    replay.set_admission_policy(
        admitted_sources=admitted,
        candidate_masses=replay_candidate_masses(masses, replay_mode),
        recency_half_life=float(recency_half_life),
        uniform_mix=float(uniform_mix),
        priority_alpha=float(priority_alpha),
    )
    # set_admission_policy 不重置 replay-physical 开关，但 anchor resume 会导入
    # 历史 admission 状态；这里显式重申，避免 resume 后静默退回 fixed quota。
    replay.set_admission_replay_physical(replay_mode == "physical")
    return masses


def actor_updates_enabled(global_step: int, start_step: int | None) -> bool:
    """Return whether target-actor optimization is enabled at ``global_step``.

    The helper deliberately controls only ``update_pol``.  Environment
    interaction, replay insertion, critic/target-critic updates, observation
    normalization, exploration RNG, and learning-rate scheduler time all keep
    their historical semantics.  This makes the critic-first arm differ from
    its interleaved control by one intervention: whether actor gradients are
    applied during the bridge interval.
    """

    return start_step is None or int(global_step) >= int(start_step)


def _validate_released_code_fidelity_config(cfg: dict[str, Any]) -> None:
    """Reject silent hybridization of the released-code fidelity mode."""
    if not bool(cfg.get("released_code_fidelity", False)):
        return
    expected = {
        "option_hidden_dims": [20],
        "option_lr": 5e-4,
        "beta_lr": 5e-4,
        "option_batch_size": 32,
        "option_grad_clip": 10.0,
        "beta_warmup_steps": 0,
        "beta_update_mode": "current_transition",
        "beta_logit_clip": None,
        "option_target_update_interval": 1000,
        "transfer_schedule": "released_tanh",
        "option_epsilon_start": 1.0,
        "option_epsilon_end": 0.1,
        "xi": 0.0,
        "beta_weighted_transfer": False,
        "execute_sources": False,
        "option_min_steps": 1,
        "update_all_compatible_options": True,
        "mcg": False,
        "admission_mode": "legacy",
        "admission_adaptive": False,
    }
    mismatches: dict[str, tuple[Any, Any]] = {}
    for key, wanted in expected.items():
        actual = cfg.get(key)
        if key in {
            "option_lr",
            "beta_lr",
            "option_grad_clip",
            "option_epsilon_start",
            "option_epsilon_end",
            "xi",
        }:
            equal = actual is not None and math.isclose(
                float(actual),
                float(wanted),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        else:
            equal = actual == wanted
        if not equal:
            mismatches[key] = (actual, wanted)
    if mismatches:
        raise ValueError(
            "released-code fidelity configuration contains non-author "
            f"semantics: {mismatches}"
        )


def _make_envs(args, device):
    if args.env_name.startswith("h1hand-") or args.env_name.startswith("h1-"):
        from fasttd3_ptf.official_fasttd3_ptf.humanoid_bench_env import HumanoidBenchEnv

        env_type = "humanoid_bench"
        envs = HumanoidBenchEnv(
            args.env_name,
            args.num_envs,
            device=device,
            seed=args.seed,
        )
        eval_envs = envs
        render_env = HumanoidBenchEnv(
            args.env_name,
            1,
            render_mode="rgb_array",
            device=device,
            seed=args.seed + 2_000_000,
        )
    elif args.env_name.startswith("Isaac-"):
        from environments.isaaclab_env import IsaacLabEnv  # type: ignore

        env_type = "isaaclab"
        envs = IsaacLabEnv(
            args.env_name,
            device.type,
            args.num_envs,
            args.seed,
            action_bounds=args.action_bounds,
        )
        eval_envs = envs
        render_env = envs
    elif args.env_name.startswith("MTBench-"):
        from environments.mtbench_env import MTBenchEnv  # type: ignore

        env_name = "-".join(args.env_name.split("-")[1:])
        env_type = "mtbench"
        envs = MTBenchEnv(env_name, args.device_rank, args.num_envs, args.seed)
        eval_envs = envs
        render_env = envs
    else:
        from environments.mujoco_playground_env import make_env  # type: ignore

        env_type = "mujoco_playground"
        envs, eval_envs, render_env = make_env(
            args.env_name,
            args.seed,
            args.num_envs,
            args.num_eval_envs,
            args.device_rank,
            use_tuned_reward=args.use_tuned_reward,
            use_domain_randomization=args.use_domain_randomization,
            use_push_randomization=args.use_push_randomization,
        )
    return env_type, envs, eval_envs, render_env


def _get_ddp_state_dict(model):
    if hasattr(model, "module"):
        return model.module.state_dict()
    return model.state_dict()


def save_ptf_params(
    global_step,
    actor,
    qnet,
    qnet_target,
    obs_normalizer,
    critic_obs_normalizer,
    args,
    ptf_cfg,
    option_module,
    option_target,
    option_optimizer,
    beta_optimizer,
    source_bank,
    save_path,
    admission_audit=None,
    training_audit=None,
) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    save_dict = {
        "actor_state_dict": cpu_state(_get_ddp_state_dict(actor)),
        "qnet_state_dict": cpu_state(_get_ddp_state_dict(qnet)),
        "qnet_target_state_dict": cpu_state(_get_ddp_state_dict(qnet_target)),
        "obs_normalizer_state": (
            cpu_state(obs_normalizer.state_dict())
            if hasattr(obs_normalizer, "state_dict")
            else None
        ),
        "critic_obs_normalizer_state": (
            cpu_state(critic_obs_normalizer.state_dict())
            if hasattr(critic_obs_normalizer, "state_dict")
            else None
        ),
        "args": vars(args),
        "global_step": global_step,
        "agent_type": "OfficialPTFFastTD3",
        "ptf_cfg": ptf_cfg,
        "source_names": source_bank.names(),
        "option_kwargs": option_module.export_kwargs(),
        "option_state_dict": cpu_state(option_module.state_dict()),
        "option_target_state_dict": cpu_state(option_target.state_dict()),
        "option_optimizer_state_dict": option_optimizer.state_dict(),
        "beta_optimizer_state_dict": beta_optimizer.state_dict(),
        "admission_audit": admission_audit,
        "training_audit": training_audit,
    }
    torch.save(save_dict, save_path, _use_new_zipfile_serialization=True)
    print(f"Saved official PTF parameters and configuration to {save_path}")


def main():
    ptf_cfg = _parse_ptf_cli()
    args = get_args()
    if ptf_cfg.get("option_seed") is None:
        ptf_cfg["option_seed"] = int(args.seed) + 1_000_003
    if args.compile and os.environ.get("FASTTD3_PTF_ALLOW_COMPILE", "0") != "1":
        print(
            "[PTF] Disabling torch.compile for train_ptf.py. "
            "This path adds dynamic option/source-bank logic on top of FastTD3; "
            "set FASTTD3_PTF_ALLOW_COMPILE=1 to override for diagnostics only."
        )
        args.compile = False
    if args.num_steps != 1:
        raise NotImplementedError(
            "official train_ptf.py currently supports num_steps=1. "
            "HumanoidBench FastTD3 defaults to num_steps=1."
        )
    print(args)
    print({"ptf": ptf_cfg})
    run_name = f"{args.env_name}__{args.exp_name}__{args.seed}"

    amp_enabled = args.amp and args.cuda and torch.cuda.is_available()
    amp_device_type = (
        "cuda"
        if args.cuda and torch.cuda.is_available()
        else "mps" if args.cuda and torch.backends.mps.is_available() else "cpu"
    )
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    scaler = GradScaler(enabled=amp_enabled and amp_dtype == torch.float16)

    if args.use_wandb:
        wandb.init(
            project=args.project,
            name=run_name,
            config={**vars(args), "ptf": ptf_cfg},
            save_code=True,
        )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    if not args.cuda:
        device = torch.device("cpu")
    else:
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{args.device_rank}")
        elif torch.backends.mps.is_available():
            device = torch.device(f"mps:{args.device_rank}")
        else:
            raise ValueError("No GPU available")
    print(f"Using device: {device}")

    env_type, envs, eval_envs, render_env = _make_envs(args, device)

    n_act = envs.num_actions
    n_obs = envs.num_obs if type(envs.num_obs) == int else envs.num_obs[0]
    if envs.asymmetric_obs:
        n_critic_obs = (
            envs.num_privileged_obs
            if type(envs.num_privileged_obs) == int
            else envs.num_privileged_obs[0]
        )
    else:
        n_critic_obs = n_obs
    action_low, action_high = -1.0, 1.0

    if args.obs_normalization:
        obs_normalizer = EmpiricalNormalization(shape=n_obs, device=device)
        critic_obs_normalizer = EmpiricalNormalization(shape=n_critic_obs, device=device)
    else:
        obs_normalizer = nn.Identity()
        critic_obs_normalizer = nn.Identity()

    if args.reward_normalization:
        if env_type in ["mtbench"]:
            reward_normalizer = PerTaskRewardNormalizer(
                num_tasks=envs.num_tasks,
                gamma=args.gamma,
                device=device,
                g_max=min(abs(args.v_min), abs(args.v_max)),
            )
        else:
            reward_normalizer = RewardNormalizer(
                gamma=args.gamma,
                device=device,
                g_max=min(abs(args.v_min), abs(args.v_max)),
            )
    else:
        reward_normalizer = nn.Identity()

    actor_kwargs = {
        "n_obs": n_obs,
        "n_act": n_act,
        "num_envs": args.num_envs,
        "device": device,
        "init_scale": args.init_scale,
        "hidden_dim": args.actor_hidden_dim,
        "std_min": args.std_min,
        "std_max": args.std_max,
    }
    critic_kwargs = {
        "n_obs": n_critic_obs,
        "n_act": n_act,
        "num_atoms": args.num_atoms,
        "v_min": args.v_min,
        "v_max": args.v_max,
        "hidden_dim": args.critic_hidden_dim,
        "device": device,
    }
    if env_type == "mtbench":
        actor_kwargs["n_obs"] = n_obs - envs.num_tasks + args.task_embedding_dim
        critic_kwargs["n_obs"] = n_critic_obs - envs.num_tasks + args.task_embedding_dim
        actor_kwargs["num_tasks"] = envs.num_tasks
        actor_kwargs["task_embedding_dim"] = args.task_embedding_dim
        critic_kwargs["num_tasks"] = envs.num_tasks
        critic_kwargs["task_embedding_dim"] = args.task_embedding_dim

    if args.agent == "fasttd3":
        if env_type in ["mtbench"]:
            from fast_td3 import MultiTaskActor, MultiTaskCritic  # type: ignore

            actor_cls = MultiTaskActor
            critic_cls = MultiTaskCritic
        else:
            from fast_td3 import Actor, Critic  # type: ignore

            actor_cls = Actor
            critic_cls = Critic
        print("Using official FastTD3 + PTF")
    elif args.agent == "fasttd3_simbav2":
        if env_type in ["mtbench"]:
            from fast_td3_simbav2 import MultiTaskActor, MultiTaskCritic  # type: ignore

            actor_cls = MultiTaskActor
            critic_cls = MultiTaskCritic
        else:
            from fast_td3_simbav2 import Actor, Critic  # type: ignore

            actor_cls = Actor
            critic_cls = Critic
        print("Using official FastTD3 + SimbaV2 + PTF")
        actor_kwargs.pop("init_scale")
        actor_kwargs.update(
            {
                "scaler_init": math.sqrt(2.0 / args.actor_hidden_dim),
                "scaler_scale": math.sqrt(2.0 / args.actor_hidden_dim),
                "alpha_init": 1.0 / (args.actor_num_blocks + 1),
                "alpha_scale": 1.0 / math.sqrt(args.actor_hidden_dim),
                "expansion": 4,
                "c_shift": 3.0,
                "num_blocks": args.actor_num_blocks,
            }
        )
        critic_kwargs.update(
            {
                "scaler_init": math.sqrt(2.0 / args.critic_hidden_dim),
                "scaler_scale": math.sqrt(2.0 / args.critic_hidden_dim),
                "alpha_init": 1.0 / (args.critic_num_blocks + 1),
                "alpha_scale": 1.0 / math.sqrt(args.critic_hidden_dim),
                "num_blocks": args.critic_num_blocks,
                "expansion": 4,
                "c_shift": 3.0,
            }
        )
    else:
        raise ValueError(f"Agent {args.agent} not supported")

    def make_actor():
        return actor_cls(**actor_kwargs)

    def make_critic():
        return critic_cls(**critic_kwargs)

    actor = make_actor()
    if env_type in ["mtbench"]:
        policy = actor.explore
    else:
        from tensordict import from_module

        actor_detach = make_actor()
        from_module(actor).data.to_module(actor_detach)
        policy = actor_detach.explore

    qnet = make_critic()
    qnet_target = make_critic()
    qnet_target.load_state_dict(qnet.state_dict())

    q_optimizer = optim.AdamW(
        list(qnet.parameters()),
        lr=torch.tensor(args.critic_learning_rate, device=device),
        weight_decay=args.weight_decay,
    )
    actor_optimizer = optim.AdamW(
        list(actor.parameters()),
        lr=torch.tensor(args.actor_learning_rate, device=device),
        weight_decay=args.weight_decay,
    )
    q_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        q_optimizer,
        T_max=args.total_timesteps,
        eta_min=torch.tensor(args.critic_learning_rate_end, device=device),
    )
    actor_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        actor_optimizer,
        T_max=args.total_timesteps,
        eta_min=torch.tensor(args.actor_learning_rate_end, device=device),
    )

    official_rb = SimpleReplayBuffer(
        n_env=args.num_envs,
        buffer_size=args.buffer_size,
        n_obs=n_obs,
        n_act=n_act,
        n_critic_obs=n_critic_obs,
        asymmetric_obs=envs.asymmetric_obs,
        playground_mode=env_type == "mujoco_playground",
        n_steps=args.num_steps,
        gamma=args.gamma,
        device=device,
    )
    rb = PTFReplayWrapper(official_rb)
    # anchor provenance 组数:默认 DEFAULT_GROUPS;P0 anchor 按任务目标
    # authority 组数保存(crawl=3/truck=2),使分支 resume 的 import_valid
    # schema 精确匹配(enable_provenance 的组数守卫要求一致)。
    anchor_provenance_group_count = int(
        ptf_cfg.get("anchor_provenance_groups") or len(DEFAULT_GROUPS)
    )
    if (
        ptf_cfg.get("anchor_step") is not None
        or ptf_cfg.get("branch_anchor_step") is not None
    ):
        # Every complete anchor must have transition-level provenance.  Paper
        # anchors are empty-bank scratch; branch anchors may contain source
        # behavior and therefore rely on the runtime MCG path to overwrite this
        # initial schema with the same group count before any transition write.
        rb.enable_provenance(group_count=anchor_provenance_group_count)

    # Empty-bank scratch and exact admission abstention share one target-only
    # fast path. Neither may inherit RNG consumption from unused transfer-only
    # source/option/MCG construction.
    target_only_rng = GlobalRngState.capture(device)
    _option_hidden = tuple(int(x) for x in ptf_cfg["option_hidden_dims"])

    source_bank_cfg = ptf_cfg.get("source_bank")
    if source_bank_cfg:
        source_bank = SourcePolicyBank.from_config(
            source_bank_cfg, device=device, target_action_dim=n_act
        )
    else:
        source_bank = SourcePolicyBank.from_config(
            {"sources": [], "null_option": {"enabled": True, "name": "no_transfer"}},
            device=device,
            target_action_dim=n_act,
        )
    print(f"Loaded source bank options: {source_bank.names()}")
    released_code_fidelity = bool(ptf_cfg["released_code_fidelity"])
    if released_code_fidelity:
        fidelity_errors = []
        if source_bank.null_option:
            fidelity_errors.append(
                "author-released PTF uses source policies as the option set; "
                "the project null/self option must be disabled"
            )
        if source_bank.num_sources < 2:
            fidelity_errors.append(
                "at least two source options are required to test option "
                "selection/termination rather than a degenerate one-option case"
            )
        if fidelity_errors:
            raise ValueError(
                "invalid released-code fidelity source bank: "
                + "; ".join(fidelity_errors)
            )

    option_module = OptionModule(
        n_obs,
        source_bank.num_options,
        _option_hidden,
        beta_logit_clip=ptf_cfg["beta_logit_clip"],
        released_code_fidelity=released_code_fidelity,
    ).to(device)
    option_target = copy.deepcopy(option_module).to(device)
    option_target.eval()
    option_optimizer = optim.Adam(option_module.parameters(), lr=float(ptf_cfg["option_lr"]))
    beta_optimizer = optim.Adam(option_module.parameters(), lr=float(ptf_cfg["beta_lr"]))
    option_update_count = 0
    if released_code_fidelity:
        transfer_scheduler = ReleasedPTFTanhScheduler(
            scale=float(ptf_cfg["transfer_lambda_start"]),
            duration=int(ptf_cfg["transfer_decay_steps"]),
        )
    else:
        transfer_scheduler = LinearScheduler(
            start=float(ptf_cfg["transfer_lambda_start"]),
            end=float(ptf_cfg["transfer_lambda_end"]),
            duration=int(ptf_cfg["transfer_decay_steps"]),
        )
    epsilon_scheduler = LinearScheduler(
        start=float(ptf_cfg["option_epsilon_start"]),
        end=float(ptf_cfg["option_epsilon_end"]),
        duration=int(ptf_cfg["option_epsilon_decay_steps"]),
    )
    option_selector = OptionSelector(
        num_envs=args.num_envs,
        num_options=source_bank.num_options,
        device=device,
        epsilon=float(ptf_cfg["option_epsilon_start"]),
        initial_option=source_bank.null_option_idx if source_bank.null_option else 0,
        seed=int(ptf_cfg["option_seed"]),
        min_duration=int(ptf_cfg["option_min_steps"]),
        select_on_reset=released_code_fidelity,
        sample_choices_only_when_needed=released_code_fidelity,
    )
    ptf_execute_sources = bool(ptf_cfg["execute_sources"])

    # ---- MCG: critic-guided 的模块化迁移(调度/蒸馏均由 target critic gating) ----
    admission_enabled = False
    admission_snapshot = None
    admission_schedule = None
    admission_schedule_cursor = 0
    admission_history: list[dict[str, Any]] = []
    admission_execution_counts = None
    target_evidence_contract = None
    target_evidence_contract_path = None
    target_evidence_probe_steps: tuple[int, ...] = ()
    target_evidence_probe_cursor = 0
    target_evidence_probe_output_dir = None
    adaptive_admission_enabled = bool(ptf_cfg["admission_adaptive"])
    # replay 侧配额模式;behavior 侧不受其影响(见 replay_candidate_masses)。
    admission_replay_mode = str(ptf_cfg["admission_replay_mode"])
    adaptive_admission_controller = None
    admission_segment_tracker = None
    adaptive_admission_logs: dict[str, float] = {}
    mcg_enabled = bool(ptf_cfg["mcg"]) and source_bank.num_sources > 0
    if mcg_enabled:
        mcg_gating = ModularGating(
            action_dim=n_act,
            groups=tuple(ptf_cfg["mcg_groups"]),
            device=device,
            margin=float(ptf_cfg["mcg_margin"]),
        )
        # safe_bootstrap: 从 bank yaml 读 per-source reward-bearing allocation
        # weight + horizon。默认WFix bank使用固定h25；历史safe bank可使用离线
        # horizon作消融。二者都不解释为自动安全/data-value估计。无
        # bootstrap 字段时 _has_boot=False → controller 退化(weights/horizons=None)。
        from fasttd3_ptf.config import load_yaml as _load_yaml_boot
        _braw = _load_yaml_boot(source_bank_cfg) if isinstance(source_bank_cfg, str) else (source_bank_cfg or {})
        _bsrc = _braw.get("sources", []) if isinstance(_braw, dict) else []
        _has_boot = any(isinstance(s, dict) and "bootstrap" in s for s in _bsrc)
        _boot_w = [float((s.get("bootstrap") or {}).get("weight", 1.0)) for s in _bsrc]
        _boot_h = [int((s.get("bootstrap") or {}).get("horizon", int(ptf_cfg["mcg_warmup_min_steps"]))) for s in _bsrc]
        admission_mode = str(ptf_cfg["admission_mode"])
        admission_enabled = admission_mode != "legacy"
        if admission_enabled:
            if str(ptf_cfg["mcg_warmup_mode"]) != "admission_bootstrap":
                raise ValueError("non-legacy admission requires mcg_warmup_mode=admission_bootstrap")
            if not _has_boot:
                raise ValueError("admission_bootstrap requires per-source bootstrap logits/horizons")
            source_names_only = tuple(source_bank.names()[: source_bank.num_sources])
            if admission_mode == "schedule":
                if not ptf_cfg["admission_schedule"]:
                    raise ValueError("schedule admission mode requires admission_schedule")
                admission_schedule = build_admission_schedule(
                    schedule_path=ptf_cfg["admission_schedule"],
                    source_names=source_names_only,
                    source_logits=_boot_w,
                    student_logit=float(ptf_cfg["admission_student_logit"]),
                )
                admission_snapshot = admission_schedule.snapshot_at(0)
            else:
                admission_snapshot = build_admission_snapshot(
                    mode=admission_mode,
                    source_names=source_names_only,
                    source_logits=_boot_w,
                    student_logit=float(ptf_cfg["admission_student_logit"]),
                    admitted_sources=ptf_cfg["admitted_sources"],
                    manifest_path=ptf_cfg["admission_manifest"],
                )
            if admission_mode == "target_evidence":
                target_evidence_errors = []
                target_evidence_value = ptf_cfg["admission_target_evidence"]
                if not target_evidence_value:
                    target_evidence_errors.append(
                        "admission_target_evidence YAML is required"
                    )
                else:
                    target_evidence_contract_path = Path(
                        str(target_evidence_value)
                    ).resolve()
                    if not target_evidence_contract_path.is_file():
                        target_evidence_errors.append(
                            f"target evidence file does not exist: "
                            f"{target_evidence_contract_path}"
                        )
                    else:
                        target_evidence_contract = TargetEvidenceContract.from_yaml(
                            target_evidence_contract_path
                        )
                        if target_evidence_contract.env_name != args.env_name:
                            target_evidence_errors.append(
                                "target evidence env_name does not match training env"
                            )
                raw_probe_steps = ptf_cfg["admission_probe_steps"]
                if not isinstance(raw_probe_steps, list) or not raw_probe_steps:
                    target_evidence_errors.append(
                        "admission_probe_steps must be a non-empty list"
                    )
                else:
                    target_evidence_probe_steps = tuple(
                        int(step) for step in raw_probe_steps
                    )
                    if (
                        target_evidence_probe_steps
                        != tuple(sorted(set(target_evidence_probe_steps)))
                        or target_evidence_probe_steps[0] <= 0
                    ):
                        target_evidence_errors.append(
                            "admission_probe_steps must be unique, increasing, and positive"
                        )
                    if target_evidence_probe_steps[-1] >= int(
                        ptf_cfg["mcg_warmup_steps"]
                    ):
                        target_evidence_errors.append(
                            "every admission probe step must precede mcg_warmup_steps"
                        )
                if str(ptf_cfg["mcg_ablation"]) != "bootstrap_only":
                    target_evidence_errors.append(
                        "target_evidence admission requires mcg_ablation=bootstrap_only"
                    )
                if str(ptf_cfg["admission_replay_handoff"]) != "physical_after_authority":
                    target_evidence_errors.append(
                        "target_evidence admission requires "
                        "admission_replay_handoff=physical_after_authority"
                    )
                if float(ptf_cfg["admission_student_logit"]) != 0.0:
                    target_evidence_errors.append(
                        "target_evidence admission freezes student_logit=0"
                    )
                if float(ptf_cfg["admission_replay_recency_half_life"]) != 0.0:
                    target_evidence_errors.append(
                        "target_evidence feasibility run freezes replay recency half-life=0"
                    )
                if float(ptf_cfg["admission_replay_uniform_mix"]) != 1.0:
                    target_evidence_errors.append(
                        "target_evidence feasibility run freezes replay uniform_mix=1"
                    )
                if float(ptf_cfg["admission_replay_priority_alpha"]) != 0.0:
                    target_evidence_errors.append(
                        "target_evidence feasibility run freezes replay priority_alpha=0"
                    )
                if any(
                    int(horizon) != 25 for horizon in _boot_h
                ):
                    target_evidence_errors.append(
                        "target_evidence feasibility run freezes every source horizon=25"
                    )
                if args.checkpoint_path or ptf_cfg.get("anchor_resume"):
                    target_evidence_errors.append(
                        "target_evidence feasibility run starts from scratch; resume is unsupported"
                    )
                if adaptive_admission_enabled:
                    target_evidence_errors.append(
                        "target_evidence admission is incompatible with the retired "
                        "reward-window adaptive controller"
                    )
                if target_evidence_errors:
                    raise ValueError(
                        "invalid target_evidence admission configuration: "
                        + "; ".join(target_evidence_errors)
                    )
                target_evidence_probe_output_dir = Path(
                    ptf_cfg["admission_probe_output_dir"]
                    or f"logs/probe/online_target_evidence/{run_name}"
                )
                print(
                    "Target-evidence admission enabled: "
                    f"steps={target_evidence_probe_steps} "
                    f"contract={target_evidence_contract_path}"
                )
            candidate_masses = admission_snapshot.candidate_probabilities(
                tau=float(ptf_cfg["mcg_bootstrap_tau"]), device=device
            )
            # P0 预注册断言:以实际 float32 softmax 路径核对源总 mass(student
            # logit 的 float64 解析值只是配置输入,不声称运行时精确相等)。
            expected_source_mass = ptf_cfg.get("admission_expected_source_mass")
            if expected_source_mass is not None:
                actual_source_mass = float(candidate_masses[:-1].sum())
                if abs(actual_source_mass - float(expected_source_mass)) > 1e-6:
                    raise AssertionError(
                        f"admission source mass {actual_source_mass:.9f} deviates from "
                        f"expected {float(expected_source_mass):.9f} by more than 1e-6"
                    )
            rb.set_admission_policy(
                admitted_sources=admission_snapshot.admitted_tensor(device),
                candidate_masses=replay_candidate_masses(
                        candidate_masses, admission_replay_mode
                    ),
                recency_half_life=float(ptf_cfg["admission_replay_recency_half_life"]),
                uniform_mix=float(ptf_cfg["admission_replay_uniform_mix"]),
                priority_alpha=float(ptf_cfg["admission_replay_priority_alpha"]),
            )
            # T4-R：source 保持 behavior authority，replay 走 physical-uniform。
            rb.set_admission_replay_physical(admission_replay_mode == "physical")
            admission_execution_counts = torch.zeros(
                source_bank.num_sources + 1, device=device, dtype=torch.int64
            )
            rb.enable_provenance(mcg_gating.num_groups)
            print(f"Admission snapshot: {admission_snapshot.as_dict()}")
            admission_history.append(
                {
                    "step": 0,
                    **admission_snapshot.as_dict(),
                    "execution_counts_at_apply": admission_execution_counts.detach().cpu().tolist(),
                }
            )
            if adaptive_admission_enabled:
                adaptive_errors = []
                if admission_mode != "all":
                    adaptive_errors.append("admission_mode=all required")
                if admission_schedule is not None:
                    adaptive_errors.append("explicit admission_schedule is incompatible")
                if str(ptf_cfg["mcg_ablation"]) != "bootstrap_only":
                    adaptive_errors.append("mcg_ablation=bootstrap_only required")
                if str(ptf_cfg["admission_replay_handoff"]) != "physical_after_authority":
                    adaptive_errors.append(
                        "admission_replay_handoff=physical_after_authority required"
                    )
                if any(
                    int(horizon) != int(ptf_cfg["mcg_warmup_min_steps"])
                    for horizon in _boot_h
                ):
                    adaptive_errors.append(
                        "all source horizons must equal mcg_warmup_min_steps"
                    )
                stage_window_steps = int(ptf_cfg["admission_stage_window_steps"])
                if stage_window_steps <= 0:
                    adaptive_errors.append(
                        "admission_stage_window_steps must be positive"
                    )
                elif int(ptf_cfg["mcg_warmup_steps"]) % stage_window_steps != 0:
                    adaptive_errors.append(
                        "mcg_warmup_steps must be divisible by admission_stage_window_steps"
                    )
                if args.checkpoint_path:
                    adaptive_errors.append(
                        "checkpoint resume is not supported for adaptive admission v1"
                    )
                if adaptive_errors:
                    raise ValueError(
                        "invalid adaptive admission configuration: "
                        + "; ".join(adaptive_errors)
                    )
                adaptive_admission_controller = AdaptiveAdmissionController(
                    initial_snapshot=admission_snapshot,
                    stage_window_steps=stage_window_steps,
                    confidence_z=float(ptf_cfg["admission_confidence_z"]),
                    min_segments=int(ptf_cfg["admission_min_segments"]),
                    persistence=int(ptf_cfg["admission_persistence"]),
                )
                admission_segment_tracker = AdmissionSegmentTracker(
                    num_envs=args.num_envs,
                    num_sources=source_bank.num_sources,
                    device=device,
                )
                print(
                    "Adaptive admission enabled: "
                    f"window={stage_window_steps} z={ptf_cfg['admission_confidence_z']} "
                    f"min_segments={ptf_cfg['admission_min_segments']} "
                    f"persistence={ptf_cfg['admission_persistence']}"
                )
        elif adaptive_admission_enabled:
            raise ValueError("adaptive admission requires non-legacy admission")
        mcg_behavior = McgBehaviorController(
            num_envs=args.num_envs,
            num_groups=mcg_gating.num_groups,
            device=device,
            group_masks=mcg_gating.group_masks,
            min_steps=int(ptf_cfg["mcg_min_steps"]),
            warmup_min_steps=int(ptf_cfg["mcg_warmup_min_steps"]),
            episode_prefix_steps=(
                None if ptf_cfg["mcg_episode_prefix_steps"] is None
                else int(ptf_cfg["mcg_episode_prefix_steps"])
            ),
            exec_prob=float(ptf_cfg["mcg_exec_prob"]),
            warmup_exec_prob=float(ptf_cfg["mcg_warmup_exec_prob"]),
            seed=int(ptf_cfg["option_seed"]),
            warmup_mode=str(ptf_cfg["mcg_warmup_mode"]),
            bootstrap_weights=(
                torch.tensor(admission_snapshot.source_logits, device=device)
                if admission_snapshot is not None
                else torch.tensor(_boot_w, device=device) if _has_boot else None
            ),
            bootstrap_horizons=torch.tensor(_boot_h, device=device) if _has_boot else None,
            bootstrap_tau=float(ptf_cfg["mcg_bootstrap_tau"]),
            online_tau=float(ptf_cfg["mcg_online_tau"]),
            online_eps=float(ptf_cfg["mcg_online_eps"]),
            online_prior_steps=int(
                float(ptf_cfg["mcg_online_prior_frac"]) * int(ptf_cfg["mcg_warmup_steps"])
            ),
            online_ema_n=float(ptf_cfg["mcg_online_ema_n"]),
            abstain_gate=bool(ptf_cfg["mcg_abstain_gate"]),
            abstain_delta_frac=float(ptf_cfg["mcg_abstain_delta_frac"]),
            abstain_k_steps=int(ptf_cfg["mcg_abstain_k_steps"]),
            abstain_eps=float(ptf_cfg["mcg_abstain_eps"]),
            online_horizons=(
                tuple(int(x) for x in str(ptf_cfg["mcg_online_horizons"]).split(","))
                if ptf_cfg["mcg_online_horizons"] else None
            ),
            admitted_sources=(
                admission_snapshot.admitted_tensor(device)
                if admission_snapshot is not None
                else None
            ),
            admission_student_logit=float(ptf_cfg["admission_student_logit"]),
        )
        mcg_warmup_steps = int(ptf_cfg["mcg_warmup_steps"])
        # Step B: replay 按源降权采样(online_bootstrap 的 arm_value 驱动)。
        # mcg_replay_mode 是主开关; 旧 --ptf_mcg_replay_reweight 映射为 "both"。
        mcg_replay_mode = str(ptf_cfg["mcg_replay_mode"])
        if bool(ptf_cfg["mcg_replay_reweight"]) and mcg_replay_mode == "off":
            mcg_replay_mode = "both"
        mcg_abstain_gate = bool(ptf_cfg["mcg_abstain_gate"])
        if admission_enabled and (mcg_replay_mode != "off" or mcg_abstain_gate):
            raise ValueError("admission-consistent replay is incompatible with legacy online replay/abstain modes")
        if (mcg_replay_mode != "off" or mcg_abstain_gate) and str(ptf_cfg["mcg_warmup_mode"]) != "online_bootstrap":
            raise ValueError("mcg_replay_mode/mcg_abstain_gate require mcg_warmup_mode=online_bootstrap (arm_value source)")
        if mcg_replay_mode != "off" and mcg_abstain_gate:
            raise ValueError("mcg_replay_mode(连续降权) 与 mcg_abstain_gate(T-gated,已停用路线) 互斥")
        # 需要 buffer 的 options 记录真实执行 arm
        mcg_track_options = admission_enabled or mcg_replay_mode != "off" or mcg_abstain_gate
        # horizon-arm 仅适配 bootstrap_only: gated 分支(full/no_bootstrap 的
        # warmup 后执行)写入 current 的是 source id, current_arm 的 arm 语义
        # 未定义——诚实拒绝而非静默错配
        if ptf_cfg["mcg_online_horizons"] and str(ptf_cfg["mcg_ablation"]) != "bootstrap_only":
            raise NotImplementedError("mcg_online_horizons(horizon-arm) 仅支持 mcg_ablation=bootstrap_only")
        # ablation 两布尔: warmup_bootstrap=warmup 期是否教师执行(注入 replay);
        # gate_active=gate 期是否执行+蒸馏。full=(T,T) bootstrap_only=(T,F) no_bootstrap=(F,T)
        mcg_ablation = str(ptf_cfg["mcg_ablation"])
        mcg_warmup_bootstrap = mcg_ablation in ("full", "bootstrap_only")
        mcg_gate_active = mcg_ablation in ("full", "no_bootstrap")
        admission_physical_handoff = (
            admission_enabled
            and str(ptf_cfg["admission_replay_handoff"])
            == "physical_after_authority"
        )
        mcg_noop_option_ids = torch.full((args.num_envs,), -1, device=device, dtype=torch.long)
        mcg_rollout_info: dict[str, float] = {}
        mcg_gate_mode = str(ptf_cfg["mcg_gate_mode"])
        mcg_null_q = float(ptf_cfg["mcg_null_quantile"])
        mcg_conf_tau = float(ptf_cfg["mcg_conf_tau"])
        # null margin 的 EMA(由蒸馏端校准,行为端复用,避免 rollout 每步翻倍的
        # critic 前向)。None=未校准:行为端保守不放行。
        mcg_margin_state: dict[str, torch.Tensor | None] = {"ema": None}
        mcg_null_gen = torch.Generator(device="cpu")
        mcg_null_gen.manual_seed(int(ptf_cfg["option_seed"]) + 7)

        def qheads_value(critic_obs_batch, action_batch):
            qf1, qf2 = qnet(critic_obs_batch, action_batch)
            v1 = qnet.get_value(F.softmax(qf1, dim=1))
            v2 = qnet.get_value(F.softmax(qf2, dim=1))
            return v1, v2

        print(
            f"MCG enabled: groups={mcg_gating.groups} gate_mode={mcg_gate_mode} "
            f"null_q={mcg_null_q} conf_tau={mcg_conf_tau} margin={mcg_gating.margin} "
            f"warmup={mcg_warmup_steps} exec_prob={ptf_cfg['mcg_exec_prob']} "
            f"warmup_mode={mcg_behavior.warmup_mode} ablation={mcg_ablation} "
            f"(warmup_bootstrap={mcg_warmup_bootstrap} gate_active={mcg_gate_active})"
        )

    if adaptive_admission_enabled and adaptive_admission_controller is None:
        raise ValueError("adaptive admission requires MCG with a non-empty source bank")

    # A schedule may start empty and admit a source later, so only an immutable
    # exact-empty decision may take the static fast path. Runtime revocation in
    # schedule mode continues through the dynamic MCG/replay lifecycle path.
    static_exact_abstention = bool(
        admission_schedule is None
        and admission_snapshot is not None
        and admission_snapshot.exact_abstain
        and admission_snapshot.mode != "target_evidence"
    )
    target_only_behavior = source_bank.num_sources == 0 or static_exact_abstention
    # QMP-fidelity(run card docs/run_card_qmp_fidelity_v1.md):per-state 完整策略
    # argmax。它复用 target-only 已验证的 classic-PTF 隔离(transfer_loss≡0、
    # 不更新 Q_ω/β、不走 OptionSelector、不走 ptf_execute_sources),只替换 rollout
    # 的动作选择;因此 isolate_classic_ptf 而非 target_only_behavior 才是隔离开关。
    # 注意 QMP **不**参与 restore_target_only_rng:target_only_rng 捕获于 source
    # bank 构建之前,其语义是"不继承未使用的 transfer-only 构建消耗";QMP 真实使用
    # source bank,与 fixed-source 臂同语义,必须保留该消耗。
    qmp_enabled = bool(ptf_cfg["qmp"])
    qmp_force_student = bool(ptf_cfg["qmp_force_student"])
    if qmp_force_student and not qmp_enabled:
        raise ValueError("--ptf_qmp_force_student requires --ptf_qmp")
    if qmp_enabled:
        if target_only_behavior:
            raise ValueError("QMP requires a non-empty, non-abstaining source bank")
        if mcg_enabled:
            raise ValueError("QMP and MCG are mutually exclusive behavior modes")
        if admission_enabled:
            raise ValueError("QMP does not support admission control (behavior-only by design)")
        if str(ptf_cfg["mcg_replay_mode"]) != "off" or bool(ptf_cfg["mcg_replay_reweight"]):
            raise ValueError("QMP requires uniform replay sampling (mcg_replay_mode=off)")
        if bool(ptf_cfg["execute_sources"]):
            raise ValueError("QMP must not use the classic ptf_execute_sources path")
    isolate_classic_ptf = bool(target_only_behavior or qmp_enabled)
    qmp_selector = None
    if qmp_enabled:
        qmp_selector = QmpSelector(
            num_sources=source_bank.num_sources, num_envs=args.num_envs, device=device
        )
        # 与 MCG 同语义:buffer 的 options 字段恒 -1(QMP 下 option 模块不更新,
        # 该字段不参与任何学习);实际选择另行记录为 qmp/* 诊断。
        qmp_option_ids = torch.full((args.num_envs,), -1, device=device, dtype=torch.long)
        qmp_diag: dict[str, torch.Tensor] | None = None

        def qmp_qheads(critic_obs_batch, action_batch):
            qf1, qf2 = qnet(critic_obs_batch, action_batch)
            v1 = qnet.get_value(F.softmax(qf1, dim=1))
            v2 = qnet.get_value(F.softmax(qf2, dim=1))
            return v1, v2

        print(
            f"QMP-fidelity enabled: candidates=1+{source_bank.num_sources} "
            f"({['student'] + list(source_bank.names())}) "
            f"score=min_h Q_h, ties->student, behavior-only "
            f"(transfer_loss=0, no option/beta update, uniform replay)"
        )
    restore_target_only_rng = bool(
        target_only_behavior
        or (
            admission_snapshot is not None
            and admission_snapshot.mode == "target_evidence"
            and admission_snapshot.exact_abstain
        )
    )
    if restore_target_only_rng:
        target_only_rng.restore()
        if target_only_behavior:
            print(
                "Target-only fast path: restored learner RNG and disabled "
                "option/source/MCG execution updates"
            )
        else:
            print(
                "Dynamic exact-abstention start: restored learner RNG while "
                "retaining the later target-evidence admission path"
            )

    policy_noise = args.policy_noise
    noise_clip = args.noise_clip

    def evaluate():
        num_eval_envs = eval_envs.num_envs
        episode_returns = torch.zeros(num_eval_envs, device=device)
        episode_lengths = torch.zeros(num_eval_envs, device=device)
        done_masks = torch.zeros(num_eval_envs, dtype=torch.bool, device=device)
        if env_type == "isaaclab":
            obs_eval = eval_envs.reset(random_start_init=False)
        else:
            obs_eval = eval_envs.reset()
        actor.eval()  # deterministic inference
        for _ in range(eval_envs.max_episode_steps):
            with torch.no_grad(), autocast(device_type=amp_device_type, dtype=amp_dtype, enabled=amp_enabled):
                obs_norm_eval = normalize_obs(obs_eval, update=False)
                actions_eval = actor(obs_norm_eval)
            next_obs_eval, rewards_eval, dones_eval, infos_eval = eval_envs.step(actions_eval.float())
            if env_type == "mtbench":
                rewards_eval = infos_eval["episode"]["success"].float() if "episode" in infos_eval else 0.0
            episode_returns = torch.where(~done_masks, episode_returns + rewards_eval, episode_returns)
            episode_lengths = torch.where(~done_masks, episode_lengths + 1, episode_lengths)
            if env_type == "mtbench" and "episode" in infos_eval:
                dones_eval = dones_eval | infos_eval["episode"]["success"]
            done_masks = torch.logical_or(done_masks, dones_eval)
            if done_masks.all():
                break
            obs_eval = next_obs_eval
        actor.train()
        return episode_returns.mean().item(), episode_lengths.mean().item()

    def render_with_rollout():
        if env_type == "humanoid_bench":
            obs_render = render_env.reset()
            renders = [render_env.render()]
        elif env_type in ["isaaclab", "mtbench"]:
            raise NotImplementedError("Rendering is not supported for IsaacLab and MTBench")
        else:
            if jnp is None:
                raise ImportError("jax is required for MuJoCo Playground rendering")
            obs_render = render_env.reset()
            render_env.state.info["command"] = jnp.array([[1.0, 0.0, 0.0]])
            renders = [render_env.state]
        actor.eval()  # deterministic inference
        for i in range(render_env.max_episode_steps):
            with torch.no_grad(), autocast(device_type=amp_device_type, dtype=amp_dtype, enabled=amp_enabled):
                obs_norm_render = normalize_obs(obs_render, update=False)
                actions_render = actor(obs_norm_render)
            next_obs_render, _, done_render, _ = render_env.step(actions_render.float())
            if env_type == "mujoco_playground":
                render_env.state.info["command"] = jnp.array([[1.0, 0.0, 0.0]])
            if i % 2 == 0:
                renders.append(render_env.render() if env_type == "humanoid_bench" else render_env.state)
            if done_render.any():
                break
            obs_render = next_obs_render
        actor.train()
        if env_type == "mujoco_playground":
            renders = render_env.render_trajectory(renders)
        return renders

    # PARE runtime。None = 未启用，update_pol 走原路径（spec §12 smoke 第 4 项）。
    # 在 anchor resume 完成后、训练循环开始前赋值。
    pare_runtime = None

    def pare_q_low(critic_obs, act):
        """conservative value ``Q_L = min(Q1, Q2)``（spec §4）。

        与 ``args.use_cdq`` 无关——``Q_L`` 是 PARE 自己的定义，
        anchor advantage 的保守性不应随 critic 配置摇摆。
        """
        q1, q2 = qnet(critic_obs, act)
        v1 = qnet.get_value(F.softmax(q1, dim=1))
        v2 = qnet.get_value(F.softmax(q2, dim=1))
        return torch.minimum(v1, v2)

    def pare_student_negatives(data):
        """batch 中**确定属于 student** 的 ``(raw_obs, action)``（spec §3）。

        必须滤掉残留的 z=1：否则同一批数据既作正例又作负例，D 学不出东西。
        ``provenance_written`` 为假的槽位来源不明，一并排除——不假定它是 student。
        """
        raw = data["raw_observations"]
        act = data["actions"]
        if "executed_group_mask" not in data.keys():
            raise AssertionError(
                "PARE 需要 replay provenance，但本 batch 没有 executed_group_mask"
            )
        is_source = data["executed_group_mask"].any(dim=-1)
        keep = data["provenance_written"].bool() & ~is_source
        return raw[keep], act[keep]

    def update_main(data, logs_dict):
        replay_priority = None
        with autocast(device_type=amp_device_type, dtype=amp_dtype, enabled=amp_enabled):
            observations = data["observations"]
            next_observations = data["next"]["observations"]
            if envs.asymmetric_obs:
                critic_observations = data["critic_observations"]
                next_critic_observations = data["next"]["critic_observations"]
            else:
                critic_observations = observations
                next_critic_observations = next_observations
            actions = data["actions"]
            rewards = data["next"]["rewards"]
            dones_batch = data["next"]["dones"].bool()
            truncations_batch = data["next"]["truncations"].bool()
            bootstrap = (~dones_batch).float() if args.disable_bootstrap else (truncations_batch | ~dones_batch).float()
            clipped_noise = torch.randn_like(actions).mul(policy_noise).clamp(-noise_clip, noise_clip)
            next_state_actions = (actor(next_observations) + clipped_noise).clamp(action_low, action_high)
            discount = args.gamma ** data["next"]["effective_n_steps"]
            with torch.no_grad():
                qf1_next_target_projected, qf2_next_target_projected = qnet_target.projection(
                    next_critic_observations,
                    next_state_actions,
                    rewards,
                    bootstrap,
                    discount,
                )
                qf1_next_target_value = qnet_target.get_value(qf1_next_target_projected)
                qf2_next_target_value = qnet_target.get_value(qf2_next_target_projected)
                if args.use_cdq:
                    qf_next_target_dist = torch.where(
                        qf1_next_target_value.unsqueeze(1) < qf2_next_target_value.unsqueeze(1),
                        qf1_next_target_projected,
                        qf2_next_target_projected,
                    )
                    qf1_next_target_dist = qf2_next_target_dist = qf_next_target_dist
                else:
                    qf1_next_target_dist, qf2_next_target_dist = (
                        qf1_next_target_projected,
                        qf2_next_target_projected,
                    )
            qf1, qf2 = qnet(critic_observations, actions)
            qf1_loss = -torch.sum(qf1_next_target_dist * F.log_softmax(qf1, dim=1), dim=1).mean()
            qf2_loss = -torch.sum(qf2_next_target_dist * F.log_softmax(qf2, dim=1), dim=1).mean()
            qf_loss = qf1_loss + qf2_loss
            if admission_enabled and float(ptf_cfg["admission_replay_priority_alpha"]) > 0:
                with torch.no_grad():
                    target_value = qnet_target.get_value(qf1_next_target_dist)
                    pred1 = qnet.get_value(F.softmax(qf1, dim=1))
                    pred2 = qnet.get_value(F.softmax(qf2, dim=1))
                    replay_priority = torch.maximum(
                        (target_value - pred1).abs(),
                        (target_value - pred2).abs(),
                    )
        q_optimizer.zero_grad(set_to_none=True)
        scaler.scale(qf_loss).backward()
        scaler.unscale_(q_optimizer)
        if args.use_grad_norm_clipping:
            critic_grad_norm = torch.nn.utils.clip_grad_norm_(
                list(qnet.parameters()),
                max_norm=args.max_grad_norm if args.max_grad_norm > 0 else float("inf"),
            )
        else:
            critic_grad_norm = torch.tensor(0.0, device=device)
        scaler.step(q_optimizer)
        scaler.update()
        logs_dict["critic_grad_norm"] = critic_grad_norm.detach()
        logs_dict["qf_loss"] = qf_loss.detach()
        logs_dict["qf_max"] = qf1_next_target_value.max().detach()
        logs_dict["qf_min"] = qf1_next_target_value.min().detach()
        if replay_priority is not None:
            rb.update_priorities(data["replay_indices"], replay_priority)
            logs_dict["admission/replay_priority_mean"] = replay_priority.mean().detach()
        return logs_dict

    def compute_transfer_loss(data, pi_action, step):
        def empty_metrics(lam: float = 0.0):
            zero = torch.zeros((), device=device)
            return {
                "transfer_loss": zero,
                "transfer_lambda": torch.tensor(float(lam), device=device),
                "ptf_beta_selected_mean": zero,
                "ptf_active_frac": zero,
            }

        option_ids = data["options"].long()
        valid = option_ids >= 0
        if valid.sum() == 0 or source_bank.num_sources == 0:
            return torch.zeros((), device=device), empty_metrics()
        source_action, action_mask, active = source_bank.act_selected(data["raw_observations"], option_ids)
        active = active & valid
        lam = float(transfer_scheduler(step))
        if active.sum() == 0:
            return torch.zeros((), device=device), empty_metrics(lam)
        _, beta = option_module(data["observations"].detach())  # stop-grad on shared encoder
        safe_ids = option_ids.clamp(0, source_bank.num_options - 1)
        beta_o = beta.gather(1, safe_ids.view(-1, 1)).squeeze(1).detach()
        per_sample = masked_action_distillation_loss(
            pi_action,
            source_action,
            action_mask,
            loss_type=str(ptf_cfg["transfer_loss"]),
            delta=float(ptf_cfg["transfer_huber_delta"]),
        )
        transfer_gate = 1.0 - beta_o if bool(ptf_cfg["beta_weighted_transfer"]) else torch.ones_like(beta_o)
        weights = lam * transfer_gate * active.float()
        loss = (weights * per_sample).sum() / active.float().sum().clamp_min(1.0)
        if ptf_cfg.get("transfer_loss_clip") is not None:
            loss = loss.clamp(max=float(ptf_cfg["transfer_loss_clip"]))
        metrics = {
            "transfer_loss": loss.detach(),
            "transfer_lambda": torch.tensor(lam, device=device),
            "ptf_beta_selected_mean": beta_o[active].mean().detach(),
            "ptf_active_frac": active.float().mean().detach(),
            "ptf_transfer_gate_mean": transfer_gate[active].mean().detach(),
            "ptf_transfer_weight_mean": weights[active].mean().detach(),
            "ptf_distill_loss_raw_mean": per_sample[active].mean().detach(),
        }
        return loss, metrics

    def compute_mcg_transfer_loss(data, pi_action, step):
        """MCG 蒸馏:per-(状态,身体组) 由 critic gating 的模块化模仿。

        与经典 PTF 蒸馏的差异:权重不是全局 λ(t)·(1−β),而是 λ(t)·1[Δ_{i*,g}(s)>margin]
        ——"向谁学、学哪个部位、在哪些状态学"全部由 target critic 决定。
        warmup 期 critic 不可信,蒸馏关闭(行为层 bootstrap 已在注入教师数据)。
        """
        zero = torch.zeros((), device=device)
        lam = float(transfer_scheduler(step))
        metrics = {
            "transfer_loss": zero,
            "transfer_lambda": torch.tensor(lam, device=device),
            "mcg/distill_gate_rate": zero,
            "mcg/distill_active_frac": zero,
            "mcg/delta_best_mean": zero,
            "mcg/conf_mean": zero,
        }
        if not mcg_gate_active:
            return zero, metrics  # bootstrap_only ablation: 无 gate 蒸馏与 margin 校准
        if admission_enabled and admission_snapshot.exact_abstain:
            return zero, metrics
        batch = pi_action.shape[0]
        sub = int(ptf_cfg["mcg_distill_subsample"])
        if sub > 0 and batch > sub:
            idx = torch.randperm(batch, device=device)[:sub]
        else:
            idx = torch.arange(batch, device=device)
        raw_obs = data["raw_observations"][idx]
        critic_obs_b = (data["critic_observations"] if envs.asymmetric_obs else data["observations"])[idx].detach()
        pi_sub = pi_action[idx]
        with torch.no_grad():
            src_actions, _ = source_bank.act_all(raw_obs)
            # null margin 校准在 warmup 期也做:行为端的 EMA 必须在 15k 切换
            # 到 gated 模式前就绪,否则第一窗口要么裸奔要么全关。
            if mcg_gate_mode == "null":
                margins_now = mcg_gating.null_margins(
                    qheads_value, critic_obs_b, pi_sub.detach(), src_actions,
                    quantile=mcg_null_q, generator=mcg_null_gen,
                )
                ema = mcg_margin_state["ema"]
                mcg_margin_state["ema"] = (
                    margins_now if ema is None else 0.99 * ema + 0.01 * margins_now
                )
                for g_i, g_name in enumerate(mcg_gating.groups):
                    metrics[f"mcg/null_margin_{g_name}"] = mcg_margin_state["ema"][g_i].detach()
        if step < mcg_warmup_steps or lam <= 0.0:
            return zero, metrics
        with torch.no_grad():
            deltas = mcg_gating.deltas(qheads_value, critic_obs_b, pi_sub.detach(), src_actions)
            margins = mcg_margin_state["ema"] if mcg_gate_mode == "null" else None
            best, sig, gate, conf = mcg_gating.select(
                deltas,
                margins=margins,
                conf_tau=mcg_conf_tau,
                source_mask=(mcg_behavior.admitted_sources if admission_enabled else None),
            )
        # 硬门控×软置信度:gate 不过的 (样本,组) 蒸馏权重严格为 0。纯 conf
        # 软权重在 sig≈0 处仍给 ~0.5 权重,window v1.1 实测平均 16-31% 的
        # 权重漏给无关教师(中后期落后 scratch 的残余伤害源)。
        per_sample, active = mcg_distillation_loss(
            pi_sub,
            src_actions,
            best,
            gate.float() * conf,
            mcg_gating.group_masks,
            loss_type=str(ptf_cfg["transfer_loss"]),
            delta=float(ptf_cfg["transfer_huber_delta"]),
        )
        loss = lam * per_sample.sum() / active.float().sum().clamp_min(1.0)
        if ptf_cfg.get("transfer_loss_clip") is not None:
            loss = loss.clamp(max=float(ptf_cfg["transfer_loss_clip"]))
        metrics.update(
            {
                "transfer_loss": loss.detach(),
                "mcg/distill_gate_rate": gate.float().mean().detach(),
                "mcg/distill_active_frac": active.float().mean().detach(),
                "mcg/delta_best_mean": deltas.max(dim=1).values.mean().detach(),
                "mcg/conf_mean": conf.mean().detach(),
                "mcg/sig_mean": sig.mean().detach(),
            }
        )
        for g_i, g_name in enumerate(mcg_gating.groups):
            metrics[f"mcg/gate_rate_{g_name}"] = gate[:, g_i].float().mean().detach()
        return loss, metrics

    def update_pol(data, logs_dict, step):
        if pare_runtime is not None:
            neg_raw, neg_act = pare_student_negatives(data)
            for k, v in pare_runtime.update_discriminator(
                neg_raw, neg_act, normalize_obs
            ).items():
                logs_dict[k] = v.detach() if torch.is_tensor(v) else torch.tensor(
                    float(v), device=device
                )
        with autocast(device_type=amp_device_type, dtype=amp_dtype, enabled=amp_enabled):
            pol_obs = data["observations"].detach()
            critic_observations = data["critic_observations"].detach() if envs.asymmetric_obs else pol_obs
            pi_action = actor(pol_obs)
            qf1, qf2 = qnet(critic_observations, pi_action)
            qf1_value = qnet.get_value(F.softmax(qf1, dim=1))
            qf2_value = qnet.get_value(F.softmax(qf2, dim=1))
            qf_value = torch.minimum(qf1_value, qf2_value) if args.use_cdq else (qf1_value + qf2_value) / 2.0
            rl_actor_loss = -qf_value.mean()
            if isolate_classic_ptf:
                transfer_loss = torch.zeros((), device=device)
                transfer_metrics = {
                    "transfer_loss": transfer_loss,
                    "transfer_lambda": torch.tensor(
                        float(transfer_scheduler(step)), device=device
                    ),
                }
            elif mcg_enabled:
                transfer_loss, transfer_metrics = compute_mcg_transfer_loss(data, pi_action, step)
            else:
                transfer_loss, transfer_metrics = compute_transfer_loss(data, pi_action, step)
            actor_loss = rl_actor_loss + transfer_loss
        actor_optimizer.zero_grad(set_to_none=True)
        if pare_runtime is None:
            scaler.scale(actor_loss).backward()
        else:
            # PARE(spec §5-§6):第二目标 J_E 与 base 目标 J_Q 分别求梯度,
            # 冲突时投影、范数截到 ‖g_Q‖ 后相加。两路共用同一 scale 因子,
            # 合成对该因子齐次,故下方 unscale_ 一次即可。
            with autocast(device_type=amp_device_type, dtype=amp_dtype,
                          enabled=amp_enabled):
                j_e, pare_metrics = pare_runtime.expansion_objective(
                    pol_obs, critic_observations, pi_action, pare_q_low
                )
            pare_metrics.update(
                pare_apply_actor_gradient(actor, -actor_loss, j_e, scaler)
            )
            for k, v in pare_metrics.items():
                logs_dict[k] = v.detach() if torch.is_tensor(v) else torch.tensor(
                    float(v), device=device
                )
        scaler.unscale_(actor_optimizer)
        if args.use_grad_norm_clipping:
            actor_grad_norm = torch.nn.utils.clip_grad_norm_(
                actor.parameters(),
                max_norm=args.max_grad_norm if args.max_grad_norm > 0 else float("inf"),
            )
        else:
            actor_grad_norm = torch.tensor(0.0, device=device)
        scaler.step(actor_optimizer)
        scaler.update()
        logs_dict["actor_grad_norm"] = actor_grad_norm.detach()
        logs_dict["actor_loss"] = actor_loss.detach()
        logs_dict["rl_actor_loss"] = rl_actor_loss.detach()
        logs_dict["transfer_loss"] = transfer_metrics.get("transfer_loss", torch.zeros((), device=device))
        logs_dict["transfer_lambda"] = transfer_metrics.get("transfer_lambda", torch.zeros((), device=device))
        logs_dict["ptf_beta_selected_mean"] = transfer_metrics.get(
            "ptf_beta_selected_mean",
            torch.zeros((), device=device),
        )
        logs_dict["ptf_active_frac"] = transfer_metrics.get("ptf_active_frac", torch.zeros((), device=device))
        logs_dict["ptf_transfer_gate_mean"] = transfer_metrics.get(
            "ptf_transfer_gate_mean",
            torch.zeros((), device=device),
        )
        logs_dict["ptf_transfer_weight_mean"] = transfer_metrics.get(
            "ptf_transfer_weight_mean",
            torch.zeros((), device=device),
        )
        logs_dict["ptf_distill_loss_raw_mean"] = transfer_metrics.get(
            "ptf_distill_loss_raw_mean",
            torch.zeros((), device=device),
        )
        for _mk, _mv in transfer_metrics.items():
            if _mk.startswith("mcg/"):
                logs_dict[_mk] = _mv
        return logs_dict

    def update_option(data, logs_dict, step, current_beta_transition=None):
        """Update replay-based ``Q_omega`` and the configured termination batch.

        Released PTF already learns ``Q_omega`` off-policy from replay and lets
        every action-compatible option share a transition.  The FastTD3-specific
        change is replacing the released binary ``mu +/- 1 sigma`` support mask
        with a recomputed soft compatibility weight for deterministic teachers.
        For termination, ``replay`` retains the historical project behavior,
        while ``current_transition`` restores the released PTF choice of the
        just-executed option and reached state.
        """
        nonlocal option_update_count
        obs_norm = data["observations"].detach()  # stop-grad: option must not train the shared encoder
        next_obs_norm = data["next"]["observations"]  # already no-grad (target encode / buffer)
        rewards_batch = data["next"]["rewards"]
        dones_batch = data["next"]["dones"].bool()
        truncations_batch = data["next"]["truncations"].bool()
        actions_batch = data["actions"]
        raw_obs_batch = data["raw_observations"]
        option_ids = data["options"].long().clamp(0, source_bank.num_options - 1)
        valid = data["options"].long() >= 0

        selected_oh = F.one_hot(option_ids, num_classes=source_bank.num_options).float()
        selected_oh = selected_oh * valid.float().view(-1, 1)
        # Released PTF also learns Q_o off-policy from replay and stores a binary
        # action-compatibility vector with each transition. Our FastTD3 adapter
        # recomputes a continuous Gaussian compatibility for every source against
        # the stored action. selected_oh is force-maxed below so the option label
        # active during collection always receives an update.
        if bool(ptf_cfg["update_all_compatible_options"]) and source_bank.num_sources > 0:
            with torch.no_grad():
                source_actions, masks = source_bank.act_all(raw_obs_batch)
                compat_src = gaussian_action_compatibility_all(
                    actions_batch,
                    source_actions,
                    masks,
                    sigma=source_bank.source_sigmas,
                )
                if source_bank.null_option:
                    # The null/no-transfer option is compatible with every target-policy
                    # transition because it applies no teacher regularization.
                    null_col = torch.ones_like(selected_oh[:, source_bank.null_option_idx : source_bank.null_option_idx + 1])
                    compat = torch.cat([compat_src, null_col], dim=1)
                else:
                    compat = compat_src
                compat = torch.maximum(compat, selected_oh)
                compat = compat * valid.float().view(-1, 1)
        else:
            compat_src = None
            compat = selected_oh

        beta_next_obs, beta_option_ids, beta_valid = select_termination_batch(
            str(ptf_cfg["beta_update_mode"]),
            replay_next_obs=next_obs_norm,
            replay_option_ids=option_ids,
            replay_valid=valid,
            current_transition=current_beta_transition,
        )

        # The author-released code applies the current-transition termination
        # update before its replay Q_omega update. Legacy mode retains this
        # project's historical Q-first order.
        beta_warmup_steps = int(ptf_cfg["beta_warmup_steps"])
        beta_update_active = step >= beta_warmup_steps

        def update_beta() -> torch.Tensor:
            beta_loss = termination_loss_at_next_state(
                option_module,
                beta_next_obs,
                beta_option_ids,
                valid=beta_valid,
                xi=float(ptf_cfg["xi"]),
                clamp_advantage=not released_code_fidelity,
            )
            beta_optimizer.zero_grad(set_to_none=True)
            beta_loss.backward()
            if ptf_cfg.get("option_grad_clip") is not None:
                torch.nn.utils.clip_grad_norm_(
                    option_module.parameters(),
                    max_norm=float(ptf_cfg["option_grad_clip"]),
                )
            beta_optimizer.step()
            return beta_loss

        def inspect_beta() -> torch.Tensor:
            with torch.no_grad():
                return termination_loss_at_next_state(
                    option_module,
                    beta_next_obs,
                    beta_option_ids,
                    valid=beta_valid,
                    xi=float(ptf_cfg["xi"]),
                    clamp_advantage=not released_code_fidelity,
                )

        if released_code_fidelity:
            beta_loss = update_beta() if beta_update_active else inspect_beta()

        q, beta = option_module(obs_norm)
        with torch.no_grad():
            if released_code_fidelity:
                q_next_online, beta_next_online = option_module(next_obs_norm)
                q_next_target, _ = option_target(next_obs_norm)
                u_next = released_code_option_u_value(
                    q_next_online,
                    beta_next_online,
                    q_next_target,
                )
            else:
                q_next, beta_next = option_target(next_obs_norm)
                u_next = option_u_value(q_next, beta_next)
            bootstrap = (truncations_batch | ~dones_batch).float().view(-1, 1)
            y_all = option_td_target(
                rewards_batch,
                gamma=args.gamma,
                bootstrap=bootstrap,
                u_next=u_next,
                reward_scale=float(ptf_cfg["option_reward_scale"]),
            )
        q_loss = compatible_option_q_loss(
            q,
            y_all,
            compat,
            released_code_reduction=released_code_fidelity,
        )
        option_optimizer.zero_grad(set_to_none=True)
        q_loss.backward()
        if ptf_cfg.get("option_grad_clip") is not None:
            torch.nn.utils.clip_grad_norm_(
                option_module.parameters(),
                max_norm=float(ptf_cfg["option_grad_clip"]),
            )
        option_optimizer.step()

        if not released_code_fidelity:
            beta_loss = update_beta() if beta_update_active else inspect_beta()

        with torch.no_grad():
            opt_hist = torch.bincount(option_ids[valid], minlength=source_bank.num_options).float()
            if opt_hist.sum() > 0:
                opt_hist = opt_hist / opt_hist.sum()
            null_ratio = (
                (option_ids[valid] == source_bank.null_option_idx).float().mean()
                if source_bank.null_option and valid.any()
                else torch.zeros((), device=device)
            )
            if released_code_fidelity:
                option_update_count += 1
                if (
                    option_update_count
                    % int(ptf_cfg["option_target_update_interval"])
                    == 0
                ):
                    option_target.load_state_dict(option_module.state_dict())
            else:
                for p, tp in zip(option_module.parameters(), option_target.parameters()):
                    tp.data.mul_(1.0 - float(ptf_cfg["option_tau"])).add_(
                        p.data,
                        alpha=float(ptf_cfg["option_tau"]),
                    )
        logs_dict["ptf_option_q_loss"] = q_loss.detach()
        logs_dict["ptf_beta_loss"] = beta_loss.detach()
        logs_dict["ptf_beta_update_active"] = torch.tensor(float(beta_update_active), device=device)
        logs_dict["ptf_beta_update_current_transition"] = torch.tensor(
            float(str(ptf_cfg["beta_update_mode"]) == "current_transition"),
            device=device,
        )
        logs_dict["ptf_beta_training_valid_fraction"] = beta_valid.float().mean()
        logs_dict["ptf_beta_mean"] = beta.detach().mean()
        logs_dict["ptf_beta_std"] = beta.detach().std(unbiased=False)
        logs_dict["ptf_q_mean"] = q.detach().mean()
        logs_dict["ptf_option_reward_scale"] = torch.tensor(
            float(ptf_cfg["option_reward_scale"]),
            device=device,
        )
        logs_dict["ptf_option_reward_input_mean"] = rewards_batch.detach().mean()
        logs_dict["ptf_option_reward_scaled_mean"] = (
            rewards_batch.detach().mean() * float(ptf_cfg["option_reward_scale"])
        )
        logs_dict["ptf_option_td_target_min"] = y_all.detach().min()
        logs_dict["ptf_option_td_target_max"] = y_all.detach().max()
        logs_dict["ptf_option_td_target_oob_fraction"] = (
            ((y_all.detach() < -1.0) | (y_all.detach() > 1.0))
            .float()
            .mean()
            if released_code_fidelity
            else torch.zeros((), device=device)
        )
        if q.shape[1] >= 2:
            option_top2 = torch.topk(q.detach(), k=2, dim=1).values
            option_gap = option_top2[:, 0] - option_top2[:, 1]
            option_all_saturated = (q.detach().abs() > 0.95).all(dim=1)
            logs_dict["ptf_option_gap_mean"] = option_gap.mean()
            logs_dict["ptf_option_gap_median"] = option_gap.median()
            logs_dict["ptf_option_all_saturated_fraction"] = (
                option_all_saturated.float().mean()
            )
            logs_dict["ptf_option_all_saturated_low_gap_fraction"] = (
                (option_all_saturated & (option_gap < 0.01)).float().mean()
            )
        logs_dict["ptf_compat_mean"] = compat.detach().mean()
        logs_dict["ptf_null_option_ratio"] = null_ratio.detach()
        option_names = source_bank.names()
        if compat_src is not None:
            logs_dict["ptf_source_compat_mean"] = compat_src.detach().mean()
            for i, name in enumerate(option_names[: source_bank.num_sources]):
                logs_dict[f"ptf_source_compat/{name}"] = compat_src[:, i].detach().mean()
        for i, frac in enumerate(opt_hist.detach().tolist() if valid.any() else []):
            logs_dict[f"ptf_option_frac/{i}"] = torch.tensor(float(frac), device=device)
            if i < len(option_names):
                logs_dict[f"ptf_option_frac/{option_names[i]}"] = torch.tensor(float(frac), device=device)
        for i, name in enumerate(option_names):
            logs_dict[f"ptf_beta_mean/{name}"] = beta[:, i].detach().mean()
            logs_dict[f"ptf_beta_std/{name}"] = beta[:, i].detach().std(unbiased=False)
        return logs_dict

    @torch.no_grad()
    def soft_update(src, tgt, tau: float):
        src_ps = [p.data for p in src.parameters()]
        tgt_ps = [p.data for p in tgt.parameters()]
        torch._foreach_mul_(tgt_ps, 1.0 - tau)
        torch._foreach_add_(tgt_ps, src_ps, alpha=tau)

    if args.compile:
        compile_mode = args.compile_mode
        update_main = torch.compile(update_main, mode=compile_mode)
        policy = torch.compile(policy, mode=None)
        normalize_obs = torch.compile(obs_normalizer.forward, mode=None)
        normalize_critic_obs = torch.compile(critic_obs_normalizer.forward, mode=None)
        if args.reward_normalization:
            update_stats = torch.compile(reward_normalizer.update_stats, mode=None)
        normalize_reward = torch.compile(reward_normalizer.forward, mode=None)
    else:
        normalize_obs = obs_normalizer.forward
        normalize_critic_obs = critic_obs_normalizer.forward
        if args.reward_normalization:
            update_stats = reward_normalizer.update_stats
        normalize_reward = reward_normalizer.forward

    if envs.asymmetric_obs:
        obs, critic_obs = envs.reset_with_critic_obs()
        critic_obs = torch.as_tensor(critic_obs, device=device, dtype=torch.float)
    else:
        obs = envs.reset()
    if args.checkpoint_path:
        torch_checkpoint = torch.load(f"{args.checkpoint_path}", map_location=device, weights_only=False)
        # 拒绝加载 entity-encoder 时代(已移除的死线)的 checkpoint:其网络
        # state_dict 键集与现行结构不相容,直接给出可操作的错误信息。
        if bool((torch_checkpoint.get("ptf_cfg") or {}).get("entity_encoder", {}).get("enabled", False)):
            raise ValueError(
                "checkpoint was trained with the removed entity-encoder path; "
                "it cannot be resumed by the current code (restore from git snapshot a5cec9d if needed)."
            )
        actor.load_state_dict(torch_checkpoint["actor_state_dict"])
        obs_normalizer.load_state_dict(torch_checkpoint["obs_normalizer_state"])
        critic_obs_normalizer.load_state_dict(torch_checkpoint["critic_obs_normalizer_state"])
        qnet.load_state_dict(torch_checkpoint["qnet_state_dict"])
        qnet_target.load_state_dict(torch_checkpoint["qnet_target_state_dict"])
        if "option_state_dict" in torch_checkpoint:
            option_module.load_state_dict(torch_checkpoint["option_state_dict"])
            option_target.load_state_dict(torch_checkpoint.get("option_target_state_dict", torch_checkpoint["option_state_dict"]))
        global_step = torch_checkpoint["global_step"]
    else:
        global_step = 0

    anchor_step = ptf_cfg.get("anchor_step")
    anchor_dir = ptf_cfg.get("anchor_dir")
    if (anchor_step is None) != (anchor_dir is None):
        raise ValueError("paper anchor requires both --ptf-anchor-step and --ptf-anchor-dir")
    branch_anchor_step = ptf_cfg.get("branch_anchor_step")
    branch_anchor_dir = ptf_cfg.get("branch_anchor_dir")
    if (branch_anchor_step is None) != (branch_anchor_dir is None):
        raise ValueError(
            "branch anchor requires both --ptf-branch-anchor-step and "
            "--ptf-branch-anchor-dir"
        )
    if anchor_step is not None and branch_anchor_step is not None:
        raise ValueError("paper anchor and branch anchor are mutually exclusive")
    if anchor_step is not None:
        anchor_step = int(anchor_step)
        anchor_errors = []
        if env_type != "humanoid_bench":
            anchor_errors.append("HumanoidBench environment required")
        if source_bank.num_sources != 0:
            anchor_errors.append("empty source bank required")
        if mcg_enabled or bool(ptf_cfg["mcg"]):
            anchor_errors.append("MCG must be disabled")
        if ptf_execute_sources:
            anchor_errors.append("source execution must be disabled")
        if args.checkpoint_path:
            anchor_errors.append("fresh scratch initialization required")
        if args.eval_interval != 0:
            anchor_errors.append("eval_interval=0 required (training env must not be reset by eval)")
        if args.render_interval != 0:
            anchor_errors.append("render_interval=0 required")
        if args.compile:
            anchor_errors.append("torch.compile must be disabled for the audit anchor")
        if not args.torch_deterministic:
            anchor_errors.append("torch_deterministic must be enabled")
        if anchor_step <= 0 or anchor_step > args.total_timesteps:
            anchor_errors.append("anchor_step must be in (0, total_timesteps]")
        if anchor_errors:
            raise ValueError("invalid paper-anchor run: " + "; ".join(anchor_errors))
    anchor_saved = False
    branch_anchor_saved = False
    critic_update_count = 0
    actor_update_count = 0
    if anchor_step is not None:
        anchor_episode_counts = torch.zeros(args.num_envs, device=device, dtype=torch.int64)
        anchor_episode_steps = torch.zeros(args.num_envs, device=device, dtype=torch.int16)
        anchor_env_ranks = torch.arange(args.num_envs, device=device, dtype=torch.int16)

    # ---- P0 core-only anchor-resume(run card v2.1.2 附录 A) ----
    # 顺序:模块构建(已完成)→ 白名单加载+全局 RNG 恢复(load_anchor_core 内部,
    # RNG 最后恢复)→ actor_detach 重绑定 → noise_scales 配对重采样(独立
    # generator,不触碰已恢复的全局 RNG;同一份采样同时写 actor 与 actor_detach)。
    # env reset 发生在更早的代码自然位置:其随机性由 env seed 面板控制,与
    # 全局 RNG 隔离(E11 wrapper 双播种),不影响分支配对。
    anchor_resume = ptf_cfg.get("anchor_resume")
    run_stop_step = int(ptf_cfg.get("run_stop_step") or args.total_timesteps)
    actor_update_start_step = ptf_cfg.get("actor_update_start_step")
    if actor_update_start_step is not None:
        actor_update_start_step = int(actor_update_start_step)
    eval_checkpoint_set = set(int(x) for x in (ptf_cfg.get("eval_checkpoint_steps") or []))
    resume_errors = []
    if run_stop_step <= 0 or run_stop_step > args.total_timesteps:
        resume_errors.append("run_stop_step must be in (0, total_timesteps]")
    if anchor_step is not None and anchor_step > run_stop_step:
        resume_errors.append("anchor_step must not exceed run_stop_step")
    if branch_anchor_step is not None:
        branch_anchor_step = int(branch_anchor_step)
        if env_type != "humanoid_bench":
            resume_errors.append("branch anchor requires a HumanoidBench environment")
        if not admission_enabled:
            resume_errors.append(
                "branch anchor v1 requires non-legacy admission provenance"
            )
        if branch_anchor_step <= args.learning_starts or branch_anchor_step > run_stop_step:
            resume_errors.append(
                "branch_anchor_step must lie in (learning_starts, run_stop_step]"
            )
        if args.eval_interval != 0:
            resume_errors.append("branch anchor requires eval_interval=0")
        if args.render_interval != 0:
            resume_errors.append("branch anchor requires render_interval=0")
        if args.compile:
            resume_errors.append("branch anchor requires torch.compile disabled")
        if not args.torch_deterministic:
            resume_errors.append("branch anchor requires torch_deterministic enabled")
        if args.reward_normalization:
            resume_errors.append("branch anchor requires reward_normalization=False")
    if actor_update_start_step is not None and not (
        args.learning_starts < actor_update_start_step <= run_stop_step
    ):
        resume_errors.append(
            "actor_update_start_step must lie in (learning_starts, run_stop_step]"
        )
    if eval_checkpoint_set and (
        min(eval_checkpoint_set) <= args.learning_starts
        or max(eval_checkpoint_set) > run_stop_step
    ):
        resume_errors.append("eval_checkpoint_steps must be in (learning_starts, run_stop_step]")
    if anchor_resume is not None:
        if anchor_step is not None:
            resume_errors.append("anchor_resume and anchor_step are mutually exclusive")
        if args.checkpoint_path:
            resume_errors.append("anchor_resume and checkpoint_path are mutually exclusive")
        if env_type != "humanoid_bench":
            resume_errors.append("HumanoidBench environment required")
        if args.eval_interval != 0:
            resume_errors.append("eval_interval=0 required (training env must not be reset by eval)")
        if args.render_interval != 0:
            resume_errors.append("render_interval=0 required")
        if args.compile:
            resume_errors.append("torch.compile must be disabled for anchor-resume branches")
        if not args.torch_deterministic:
            resume_errors.append("torch_deterministic must be enabled")
        if args.reward_normalization:
            resume_errors.append("reward_normalization must be False (P0 frozen protocol)")
        if ptf_cfg.get("run_stop_step") is None:
            resume_errors.append("run_stop_step is required for anchor-resume branches")
        if ptf_cfg.get("resume_noise_seed") is None:
            resume_errors.append("resume_noise_seed is required for anchor-resume branches")
    if resume_errors:
        raise ValueError("invalid anchor-resume run: " + "; ".join(resume_errors))
    if anchor_resume is not None:
        from tensordict import from_module

        core = load_anchor_core(
            anchor_resume,
            modules={
                "actor": actor,
                "critic": qnet,
                "critic_target": qnet_target,
                "obs_normalizer": obs_normalizer,
                "critic_obs_normalizer": critic_obs_normalizer,
            },
            optimizers={"actor": actor_optimizer, "critic": q_optimizer},
            schedulers={"actor": actor_scheduler, "critic": q_scheduler},
            scaler=scaler,
            replay=rb,
            map_location=device,
        )
        global_step = int(core["completed_vector_steps"])
        if global_step >= run_stop_step:
            raise ValueError(
                f"run_stop_step={run_stop_step} must exceed anchor step {global_step}"
            )
        # 关键训练参数与 anchor 逐项一致断言(五次复核阻塞问题 3):否则分支
        # 不再是同一 learner 的配对分支。允许差异仅限预注册项(bank/admission
        # treatment/stop/checkpoint/命名),不在下表内。
        _RESUME_MATCH_KEYS = (
            "env_name", "agent", "seed", "num_envs", "buffer_size", "num_steps",
            "gamma", "tau", "batch_size", "num_updates", "policy_frequency",
            "critic_learning_rate", "actor_learning_rate",
            "critic_learning_rate_end", "actor_learning_rate_end",
            "total_timesteps", "learning_starts", "policy_noise", "noise_clip",
            "std_min", "std_max", "init_scale", "num_atoms", "v_min", "v_max",
            "critic_hidden_dim", "actor_hidden_dim", "critic_num_blocks",
            "actor_num_blocks", "use_cdq", "weight_decay", "amp", "amp_dtype",
            "obs_normalization", "reward_normalization", "compile",
            "torch_deterministic", "disable_bootstrap", "use_grad_norm_clipping",
            "max_grad_norm", "action_bounds",
        )
        anchor_args = (core.get("configuration") or {}).get("args") or {}
        config_mismatches = {
            key: (anchor_args.get(key), getattr(args, key))
            for key in _RESUME_MATCH_KEYS
            if key in anchor_args and anchor_args.get(key) != getattr(args, key)
        }
        if config_mismatches:
            raise ValueError(
                f"anchor/branch configuration mismatch (frozen protocol): {config_mismatches}"
            )
        missing_keys = [key for key in _RESUME_MATCH_KEYS if key not in anchor_args]
        if missing_keys:
            raise ValueError(f"anchor configuration lacks frozen keys: {missing_keys}")
        auxiliary = core["auxiliary_state"]
        critic_update_count = int(auxiliary["critic_update_count"])
        actor_update_count = int(auxiliary["actor_update_count"])
        is_branch_anchor = bool(auxiliary.get("branch_anchor", False))
        if is_branch_anchor:
            if not admission_enabled or admission_execution_counts is None:
                raise ValueError(
                    "a branch anchor must resume through non-legacy admission"
                )
            saved_source_names = list(auxiliary.get("source_names") or [])
            runtime_source_names = list(source_bank.names())
            if saved_source_names != runtime_source_names:
                raise ValueError(
                    "branch anchor source-bank identity mismatch: "
                    f"anchor={saved_source_names}, runtime={runtime_source_names}"
                )
            saved_execution_counts = torch.as_tensor(
                auxiliary.get("admission_execution_counts"),
                device=device,
                dtype=torch.int64,
            ).view(-1)
            if saved_execution_counts.shape != admission_execution_counts.shape:
                raise ValueError(
                    "branch anchor admission execution-count shape mismatch: "
                    f"anchor={tuple(saved_execution_counts.shape)}, "
                    f"runtime={tuple(admission_execution_counts.shape)}"
                )
            admission_execution_counts.copy_(saved_execution_counts)
            saved_history = auxiliary.get("admission_history")
            if not isinstance(saved_history, list):
                raise ValueError("branch anchor lacks admission_history")
            admission_history = copy.deepcopy(saved_history)

        # replay.import_valid() intentionally restores the anchor's historical
        # admission state.  The continuation treatment must nevertheless be
        # controlled by *runtime* configuration (not by the selected branch's
        # old source policy), so reapply it immediately after import.  This is
        # also what atomically releases every latch when runtime mode is exact
        # abstention.
        if admission_enabled:
            assert admission_snapshot is not None
            apply_runtime_admission_policy_after_resume(
                replay=rb,
                behavior=mcg_behavior,
                snapshot=admission_snapshot,
                device=device,
                bootstrap_tau=float(ptf_cfg["mcg_bootstrap_tau"]),
                replay_mode=admission_replay_mode,
                recency_half_life=float(
                    ptf_cfg["admission_replay_recency_half_life"]
                ),
                uniform_mix=float(ptf_cfg["admission_replay_uniform_mix"]),
                priority_alpha=float(ptf_cfg["admission_replay_priority_alpha"]),
            )
            if is_branch_anchor:
                admission_history.append(
                    {
                        "step": global_step,
                        **admission_snapshot.as_dict(),
                        "execution_counts_at_apply": (
                            admission_execution_counts.detach().cpu().tolist()
                        ),
                        "reason": "runtime_policy_after_branch_anchor_resume",
                    }
                )
        # actor_detach 重绑定(无 RNG 消耗,故置于全局 RNG 恢复之后等价)。
        from_module(actor).data.to_module(actor_detach)
        # noise_scales 是 episode 级状态(explore 只在 done 时重采样):fresh
        # reset 后不得沿用 anchor 中上一 episode 的值,按配对 seed 重采样。
        noise_generator = torch.Generator(device=device)
        noise_generator.manual_seed(int(ptf_cfg["resume_noise_seed"]))
        resampled_scales = (
            torch.rand(args.num_envs, 1, generator=noise_generator, device=device)
            * (actor.std_max - actor.std_min)
            + actor.std_min
        )
        actor.noise_scales.copy_(resampled_scales)
        actor_detach.noise_scales.copy_(resampled_scales)
        actor_state = actor.state_dict()
        for buffer_name, buffer_value in actor_detach.state_dict().items():
            if not torch.equal(buffer_value, actor_state[buffer_name]):
                raise AssertionError(
                    f"actor_detach buffer '{buffer_name}' diverged after resume rebind"
                )
        # 审计信息随 ptf_cfg 持久化进后续 checkpoint(anchor 溯源)。
        ptf_cfg["anchor_resume_manifest"] = {
            "bundle": str(anchor_resume),
            "completed_vector_steps": global_step,
            "git_head": (core["manifest"].get("git") or {}).get("head"),
            "files": core["manifest"].get("files"),
        }
        print(
            f"Resumed core learner from anchor {anchor_resume} at step {global_step}; "
            f"training continues to run_stop_step={run_stop_step}"
        )
    if ptf_cfg.get("pare"):
        # PARE 是 post-release 机制：release 点就是本 run 的起点。实验设计是
        # branch-at-release——从同一 release anchor 分叉 hard-exit 与 PARE 两臂，
        # 二者共享 actor / critic / replay / source history / release state，
        # 唯一差别就是这里的 expansion 更新（spec §11 8.2）。
        if anchor_resume is None:
            raise ValueError("PARE 需要 --ptf-anchor-resume 指向 release anchor")
        if str(ptf_cfg["admission_mode"]) != "none":
            raise ValueError(
                "PARE 必须与 admission_mode=none 同用：source 在 release 时已永久退出"
            )
        # fail-closed：provenance 不完整就没有可信的 z，宁可不跑。
        rb.assert_complete_provenance()
        pare_reservoir = SourceTransitionReservoir.from_replay(
            rb, capacity=int(ptf_cfg["pare_reservoir_capacity"])
        )
        pare_runtime = PARERuntime(
            actor=actor,
            obs_dim=n_obs,
            act_dim=n_act,
            reservoir=pare_reservoir,
            device=device,
            d_lr=float(ptf_cfg["pare_d_lr"]),
        )
        # release 时刻的 actor 与 obs normalizer 已完整存在于 release anchor 里，
        # 无需另存一份快照；此处只记指针与 reservoir 的实际截断量（spec §10 D2，
        # 截断不得静默）。
        ptf_cfg["pare_release_manifest"] = {
            "release_anchor": str(anchor_resume),
            "release_step": global_step,
            "reservoir_candidates": pare_reservoir.n_candidates,
            "reservoir_kept": len(pare_reservoir),
            "reservoir_truncated": bool(pare_reservoir.truncated),
            "reservoir_capacity": int(ptf_cfg["pare_reservoir_capacity"]),
        }
        print(
            f"PARE enabled at release step {global_step}: source reservoir kept "
            f"{len(pare_reservoir)}/{pare_reservoir.n_candidates} z=1 transitions "
            f"(capacity {ptf_cfg['pare_reservoir_capacity']})"
        )
    if actor_update_start_step is not None:
        print(
            "Critic-first actor hold enabled: "
            f"actor updates start at completed step {actor_update_start_step}"
        )

    dones = None
    start_time = None
    measure_burnin = 0
    pbar = tqdm.tqdm(total=run_stop_step, initial=global_step)
    desc = ""
    raw_rewards = torch.zeros(args.batch_size, device=device)
    replay_env_ranks = torch.arange(args.num_envs, device=device, dtype=torch.int16)
    admission_segment_counter = torch.zeros(
        args.num_envs, device=device, dtype=torch.int64
    )
    if anchor_resume is not None and rb.provenance_enabled:
        # segment 命名空间续接(五次复核审计缺口 1):anchor 数据的 segment_id
        # 已占用 [0, max];分支从 0 重新编号会碰撞,revocation/审计按 segment
        # 过滤时会误伤。segment_id = counter*num_envs + rank,故 counter 从
        # floor(max_id/num_envs)+1 续接即可保证全局唯一。
        max_segment_id = rb.max_provenance_segment_id()
        if max_segment_id >= 0:
            admission_segment_counter += max_segment_id // args.num_envs + 1
    admission_segment_step = torch.zeros(
        args.num_envs, device=device, dtype=torch.int16
    )
    admission_previous_groups = torch.full(
        (args.num_envs, mcg_gating.num_groups if mcg_enabled else 1),
        -2,
        device=device,
        dtype=torch.long,
    )
    target_only_option_ids = torch.full(
        (args.num_envs,), -1, device=device, dtype=torch.long
    )

    # run_stop_step 独立于 total_timesteps 控制退出:LR 余弦日程的 T_max 仍由
    # total_timesteps 决定,分支训练提前停止不压缩日程(run card v2.1.2 A.2.1)。
    while global_step < run_stop_step:
        pending_adaptive_window = None
        admission_executed_candidate = None
        if admission_schedule is not None:
            decisions = admission_schedule.decisions
            while (
                admission_schedule_cursor + 1 < len(decisions)
                and global_step >= decisions[admission_schedule_cursor + 1][0]
            ):
                admission_schedule_cursor += 1
                decision_step, admission_snapshot = decisions[admission_schedule_cursor]
                admitted_tensor = admission_snapshot.admitted_tensor(device)
                candidate_masses = admission_snapshot.candidate_probabilities(
                    tau=float(ptf_cfg["mcg_bootstrap_tau"]), device=device
                )
                mcg_behavior.set_admission_policy(
                    admitted_sources=admitted_tensor,
                    source_logits=torch.tensor(
                        admission_snapshot.source_logits,
                        device=device,
                        dtype=torch.float32,
                    ),
                    student_logit=admission_snapshot.student_logit,
                )
                rb.set_admission_policy(
                    admitted_sources=admitted_tensor,
                    candidate_masses=replay_candidate_masses(
                        candidate_masses, admission_replay_mode
                    ),
                    recency_half_life=float(ptf_cfg["admission_replay_recency_half_life"]),
                    uniform_mix=float(ptf_cfg["admission_replay_uniform_mix"]),
                    priority_alpha=float(ptf_cfg["admission_replay_priority_alpha"]),
                )
                admission_history.append(
                    {
                        "step": int(decision_step),
                        **admission_snapshot.as_dict(),
                        "execution_counts_at_apply": admission_execution_counts.detach().cpu().tolist(),
                    }
                )
                print(
                    f"Applied admission decision at step {decision_step}: "
                    f"{admission_snapshot.as_dict()}"
                )
        if target_evidence_contract is not None:
            while (
                target_evidence_probe_cursor < len(target_evidence_probe_steps)
                and global_step
                >= target_evidence_probe_steps[target_evidence_probe_cursor]
            ):
                decision_step = target_evidence_probe_steps[
                    target_evidence_probe_cursor
                ]
                target_evidence_probe_cursor += 1
                probe_rng = GlobalRngState.capture(device)

                @torch.no_grad()
                def target_evidence_student_act(
                    raw_obs: np.ndarray,
                ) -> np.ndarray:
                    obs_tensor = torch.as_tensor(
                        raw_obs,
                        device=device,
                        dtype=torch.float32,
                    ).unsqueeze(0)
                    normalized = normalize_obs(obs_tensor, update=False)
                    return actor(normalized).squeeze(0).float().cpu().numpy()

                source_actions = {
                    source.name: (
                        lambda raw_obs, policy=source: policy.act(
                            torch.as_tensor(
                                raw_obs,
                                device=device,
                                dtype=torch.float32,
                            ).unsqueeze(0)
                        )
                        .squeeze(0)
                        .float()
                        .cpu()
                        .numpy()
                    )
                    for source in source_bank.sources
                }
                try:
                    probe_result = run_target_evidence_probe(
                        contract=target_evidence_contract,
                        student_act=target_evidence_student_act,
                        source_actions=source_actions,
                        protocol=TargetEvidenceProbeProtocol(horizon=25),
                    )
                finally:
                    # The sidecar probe is quarantine-only and must not advance
                    # learner exploration, replay, or global CUDA RNG streams.
                    probe_rng.restore()

                admitted_order = list(probe_result["admitted_order"])
                selected = admitted_order[:1]
                source_names = tuple(
                    source_bank.names()[: source_bank.num_sources]
                )
                report = {
                    "experiment": "online_target_evidence_admission_v1",
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "run_name": run_name,
                    "training_step": int(decision_step),
                    "env_name": args.env_name,
                    "target_evidence": {
                        "path": str(target_evidence_contract_path),
                        "sha256": _sha256_file(
                            target_evidence_contract_path
                        ),
                        "name": target_evidence_contract.name,
                    },
                    "source_names": list(source_names),
                    "selection_rule": (
                        "top-1 progress-LCB source among candidates passing "
                        "positive return/progress LCBs; otherwise exact abstention"
                    ),
                    "admitted_order": admitted_order,
                    "selected_source": selected[0] if selected else None,
                    "probe": probe_result,
                }
                assert target_evidence_probe_output_dir is not None
                target_evidence_probe_output_dir.mkdir(
                    parents=True, exist_ok=True
                )
                report_path = (
                    target_evidence_probe_output_dir
                    / f"step_{int(decision_step):06d}.json"
                )
                if report_path.exists():
                    raise FileExistsError(
                        f"refusing to overwrite target-evidence probe {report_path}"
                    )
                report_path.write_text(json.dumps(report, indent=2) + "\n")
                report_digest = _sha256_file(report_path)

                admission_snapshot = build_top1_admission_snapshot(
                    source_names=source_names,
                    probe_result=probe_result,
                    decision_step=decision_step,
                    quarantine_artifact=str(report_path.resolve()),
                    quarantine_digest=report_digest,
                )
                admitted_tensor = admission_snapshot.admitted_tensor(device)
                candidate_masses = admission_snapshot.candidate_probabilities(
                    tau=float(ptf_cfg["mcg_bootstrap_tau"]),
                    device=device,
                )
                if selected:
                    if abs(float(candidate_masses[:-1].sum()) - 0.5) > 1e-6:
                        raise AssertionError(
                            "top-1 target-evidence source mass must equal 0.5"
                        )
                elif float(candidate_masses[-1]) != 1.0:
                    raise AssertionError(
                        "target-evidence exact abstention must be 100% student"
                    )
                mcg_behavior.set_admission_policy(
                    admitted_sources=admitted_tensor,
                    source_logits=torch.zeros(
                        source_bank.num_sources,
                        device=device,
                        dtype=torch.float32,
                    ),
                    student_logit=0.0,
                )
                rb.set_admission_policy(
                    admitted_sources=admitted_tensor,
                    candidate_masses=replay_candidate_masses(
                        candidate_masses, admission_replay_mode
                    ),
                    recency_half_life=float(
                        ptf_cfg["admission_replay_recency_half_life"]
                    ),
                    uniform_mix=float(
                        ptf_cfg["admission_replay_uniform_mix"]
                    ),
                    priority_alpha=float(
                        ptf_cfg["admission_replay_priority_alpha"]
                    ),
                )
                admission_history.append(
                    {
                        "step": int(decision_step),
                        **admission_snapshot.as_dict(),
                        "execution_counts_at_apply": (
                            admission_execution_counts.detach().cpu().tolist()
                        ),
                    }
                )
                print(
                    "Applied target-evidence admission at "
                    f"step {decision_step}: selected="
                    f"{selected[0] if selected else 'NONE'} "
                    f"artifact={report_path}"
                )
        if admission_enabled and admission_physical_handoff:
            desired_source_authority = desired_admission_source_authority(
                admission_snapshot,
                global_step=global_step,
                warmup_steps=mcg_warmup_steps,
                warmup_authority=mcg_warmup_bootstrap,
                post_warmup_authority=mcg_gate_active,
            )
            if (
                rb.admission_source_authority_active
                != desired_source_authority
            ):
                rb.set_admission_source_authority(
                    desired_source_authority,
                    reason=(
                        "warmup_behavior_phase"
                        if global_step < mcg_warmup_steps
                        else "post_warmup_behavior_phase"
                    ),
                )
                print(
                    "Admission replay source authority "
                    f"{'enabled' if desired_source_authority else 'released'} "
                    f"at step {global_step}"
                )
        mark_step()
        logs_dict = TensorDict()
        if admission_enabled:
            logs_dict["ptf_admission/source_authority_active"] = torch.tensor(
                float(rb.admission_source_authority_active), device=device
            )
        if start_time is None and global_step >= args.measure_burnin + args.learning_starts:
            start_time = time.time()
            measure_burnin = global_step

        with torch.no_grad(), autocast(device_type=amp_device_type, dtype=amp_dtype, enabled=amp_enabled):
            norm_obs = normalize_obs(obs)
            executed_source_by_group = None
            eps = float(epsilon_scheduler(global_step))
            if target_only_behavior:
                option_ids = target_only_option_ids
                actions = policy(obs=norm_obs, dones=dones)
                if admission_enabled:
                    executed_source_by_group = torch.full(
                        (args.num_envs, mcg_gating.num_groups),
                        -1,
                        device=device,
                        dtype=torch.long,
                    )
            elif qmp_enabled:
                # QMP-fidelity: per-state 完整策略 argmax min_h Q_h。
                # 噪声时序(run card §3.1,v1 伪码在此处有 bug):policy 默认返回
                # **带噪声**动作,拿它与无噪声 source 比 Q 会系统性偏向 source。
                # 因此先取无噪声 student(deterministic=True 仍会执行 episode 级
                # noise_scales 重采样,RNG 消耗不变),全部候选无噪声打分选择后,
                # 只采样一次噪声加到选中的动作上——与 explore() 逐位等价。
                option_ids = qmp_option_ids
                student_action = policy(obs=norm_obs, dones=dones, deterministic=True)
                src_actions_all, _ = source_bank.act_all(obs)
                a_sel, qmp_choice, qmp_diag = qmp_selector.select(
                    qmp_qheads, norm_obs, student_action, src_actions_all, dones=dones
                )
                if qmp_force_student:
                    # smoke-only(run card §7):退化为纯 student。打分仍照常执行,
                    # 因此这条路径验证的是"除动作选择外 QMP 分支无副作用"。
                    a_sel = student_action
                actions = a_sel + torch.randn_like(a_sel) * actor_detach.noise_scales
            elif mcg_enabled:
                # MCG 行为层:调度不走 OptionSelector/Q_o(option-value 直接从
                # target critic 的 masked-candidate Δ 读出,v3.1 鸡生蛋的解)。
                # buffer 的 options 字段恒 -1(MCG 蒸馏在 update 时重算 gating,
                # 不依赖采样期的 option 标注)。
                option_ids = mcg_noop_option_ids
                executed_source_by_group = torch.full(
                    (args.num_envs, mcg_gating.num_groups),
                    -1,
                    device=device,
                    dtype=torch.long,
                )
                actions = policy(obs=norm_obs, dones=dones)
                in_warmup = global_step < mcg_warmup_steps
                # ablation 门控: bootstrap_only 关 gate 期执行; no_bootstrap 关 warmup 执行
                do_exec = (in_warmup and mcg_warmup_bootstrap) or (
                    not in_warmup and mcg_gate_active)
                if do_exec:
                    exact_abstain = admission_enabled and admission_snapshot.exact_abstain
                    if exact_abstain:
                        src_actions_all = actions.new_empty(
                            args.num_envs, source_bank.num_sources, actions.shape[-1]
                        )
                    else:
                        src_actions_all, _ = source_bank.act_all(obs)
                    if in_warmup:
                        mcg_best = mcg_gate = None
                    else:
                        if exact_abstain:
                            mcg_best = torch.zeros(
                                args.num_envs,
                                mcg_gating.num_groups,
                                device=device,
                                dtype=torch.long,
                            )
                            mcg_gate = torch.zeros_like(mcg_best, dtype=torch.bool)
                        else:
                            mcg_deltas = mcg_gating.deltas(qheads_value, norm_obs, actions, src_actions_all)
                            _m_ema = mcg_margin_state["ema"]
                            if mcg_gate_mode == "null" and _m_ema is None:
                                # margin 未校准(防御分支):保守不放行
                                _m_ema = torch.full(
                                    (mcg_gating.num_groups,), float("inf"), device=device
                                )
                            mcg_best, _mcg_sig, mcg_gate, _mcg_conf = mcg_gating.select(
                                mcg_deltas,
                                margins=_m_ema if mcg_gate_mode == "null" else None,
                                conf_tau=mcg_conf_tau,
                                source_mask=(mcg_behavior.admitted_sources if admission_enabled else None),
                            )
                    actions, mcg_rollout_info = mcg_behavior.step(
                        actions, src_actions_all, best=mcg_best, gate=mcg_gate, dones=dones,
                    )
                    executed_source_by_group = mcg_behavior.current.clone()
                    if adaptive_admission_enabled and in_warmup:
                        admission_executed_candidate = torch.where(
                            mcg_behavior.current_arm[:, 0] < 0,
                            torch.full_like(
                                mcg_behavior.current_arm[:, 0],
                                source_bank.num_sources,
                            ),
                            mcg_behavior.current_arm[:, 0],
                        ).clone()
                    if mcg_track_options:
                        # Step B(导师意见1): buffer 的 options 记录本步真实执行 arm
                        # (仅 do_exec 分支内取——do_exec=False 时 current 是残留值,
                        # 不可用)。MCG 模式下 options 无其他消费方(transfer/option
                        # 模块均走 not mcg_enabled 分支),仅 replay 加权读取。
                        # current_arm=arm 级 id: multi-horizon 时区分同源不同档
                        # (h25/h50 各自按 arm value 加权), 单 horizon 时=current。
                        if admission_enabled:
                            # A modular action may use a source only in arms/hands,
                            # so group-0 is not a valid transition provenance proxy.
                            # Store a canonical admitted source in options for quota
                            # allocation and preserve every contributing source in
                            # source_by_group below for exact revocation filtering.
                            group_sources = mcg_behavior.current
                            sentinel = torch.full_like(
                                group_sources, source_bank.num_sources
                            )
                            canonical = torch.where(
                                group_sources >= 0, group_sources, sentinel
                            ).min(dim=1).values
                            option_ids = torch.where(
                                canonical < source_bank.num_sources,
                                canonical,
                                torch.full_like(canonical, -1),
                            )
                        else:
                            option_ids = mcg_behavior.current_arm[:, 0].clone()
            else:
                option_ids = option_selector.step(norm_obs, option_module, dones=dones, epsilon=eps)
                actions = policy(obs=norm_obs, dones=dones)
            if not mcg_enabled and not qmp_enabled and ptf_execute_sources:
                # 行为层 call-and-return:选中的非 null 源 option 输出完整动作直接
                # 控制环境(教师演示)。必须整动作执行——教师的闭环行为依赖它自己
                # 的全身协调(reach 的"伸手"以它自己的站立腿为前提,若 mask 外维度
                # 交给早期 actor 会摔毁演示;这是 pilot v3 的实现教训)。action mask
                # 只用于蒸馏损失。off-policy 更新对行为策略无要求,buffer 记录实际
                # 执行的动作。
                src_actions, _src_masks, src_active = source_bank.act_selected(obs, option_ids)
                actions = torch.where(src_active.view(-1, 1), src_actions, actions)

        next_obs, rewards, dones, infos = envs.step(actions.float())
        truncations = infos["time_outs"]
        if adaptive_admission_enabled and global_step < mcg_warmup_steps:
            assert adaptive_admission_controller is not None
            assert admission_segment_tracker is not None
            if admission_executed_candidate is None:
                raise AssertionError(
                    "adaptive admission warmup did not expose an executed candidate"
                )
            natural_segment_ends = (
                (mcg_behavior.steps_left[:, 0] <= 0) | dones.view(-1).bool()
            )
            completed_candidates, completed_segment_means = (
                admission_segment_tracker.observe(
                    executed_candidates=admission_executed_candidate,
                    rewards=rewards,
                    natural_ends=natural_segment_ends,
                )
            )
            adaptive_admission_controller.record_segments(
                completed_candidates, completed_segment_means
            )
            pending_adaptive_window = (
                adaptive_admission_controller.maybe_close_window(global_step + 1)
            )
        if (
            mcg_enabled
            and mcg_behavior.warmup_mode == "online_bootstrap"
            and global_step < mcg_warmup_steps
        ):
            # student-as-arm 在线结算: 本步 reward 归属产生它的 arm(EMA)。
            # 必须在 envs.step 后、下一次 mcg_behavior.step 前(current 时序对齐)。
            mcg_behavior.update_arm_reward(rewards)
            if mcg_replay_mode != "off" and global_step % 100 == 0:
                # Step B/split: arm_value → per-source 采样权重。只降不升:
                # 比学生好的源恒 1(不过采样教师), 比学生差的按 exp((T_i−T_stu)/
                # (std·tau)) 降权, floor 保留负样本价值。warmup 结束后权重冻结,
                # 持续控制残留source轨迹的replay exposure。最终归因结果支持
                # actor/critic使用相同权重(role="both")以维持更新分布一致；
                # actor_only/critic_only/split只保留为机制消融，不作主方法。
                _av = mcg_behavior.arm_value
                _scale = _av.std().clamp_min(1e-6) * float(ptf_cfg["mcg_replay_tau"])
                _raw = torch.exp((_av[:-1] - _av[-1]) / _scale)

                def _w(floor):
                    return _raw.clamp(float(floor), 1.0)

                _uni_floor = ptf_cfg["mcg_replay_floor"]
                if mcg_replay_mode == "both":
                    rb.set_source_weights(_w(_uni_floor), role="both")
                elif mcg_replay_mode == "actor_only":
                    rb.set_source_weights(_w(_uni_floor), role="actor")
                    rb.set_source_weights(None, role="critic")
                elif mcg_replay_mode == "critic_only":
                    rb.set_source_weights(_w(_uni_floor), role="critic")
                    rb.set_source_weights(None, role="actor")
                elif mcg_replay_mode == "split":
                    rb.set_source_weights(_w(ptf_cfg["mcg_replay_actor_floor"]), role="actor")
                    rb.set_source_weights(_w(ptf_cfg["mcg_replay_critic_floor"]), role="critic")
                _wlog = _w(_uni_floor)
                for _i in range(int(_raw.shape[0])):
                    # multi-horizon 时 _i 是 arm 索引(source-major), 命名避免误读
                    _k = f"mcg/replay_w_arm{_i}" if mcg_behavior.online_horizons is not None else f"mcg/replay_w_src{_i}"
                    mcg_rollout_info[_k] = float(_wlog[_i])
            elif mcg_abstain_gate and global_step % 100 == 0:
                # T-gated(ChatGPT 裁定): transfer mode 保持 uniform replay(不误伤
                # pole 型正迁移); abstain mode 才把源轨迹降到 floor(修 crawl 型
                # 残留毒害)。warmup 结束后权重维持最后状态。
                if mcg_behavior.abstain_mode:
                    _n_src = int(mcg_behavior.arm_value.shape[0]) - 1
                    rb.set_source_weights(torch.full(
                        (_n_src,), float(ptf_cfg["mcg_abstain_replay_floor"]), device=device
                    ))
                else:
                    rb.set_source_weights(None)

        if args.reward_normalization:
            if env_type == "mtbench":
                task_ids_one_hot = obs[..., -envs.num_tasks :]
                task_indices = torch.argmax(task_ids_one_hot, dim=1)
                update_stats(rewards, dones.float(), task_ids=task_indices)
            else:
                update_stats(rewards, dones.float())

        if envs.asymmetric_obs:
            next_critic_obs = infos["observations"]["critic"]
        true_next_obs = torch.where(dones[:, None] > 0, infos["observations"]["raw"]["obs"], next_obs)
        current_beta_transition = None
        if (
            not isolate_classic_ptf
            and not mcg_enabled
            and str(ptf_cfg["beta_update_mode"]) == "current_transition"
        ):
            # Do not update the shared observation-normalizer statistics a
            # second time merely to construct beta's current-transition batch.
            current_beta_transition = (
                normalize_obs(true_next_obs, update=False).detach(),
                option_ids.detach().clone(),
                (~dones.bool()).detach(),
            )
        if envs.asymmetric_obs:
            true_next_critic_obs = torch.where(
                dones[:, None] > 0,
                infos["observations"]["raw"]["critic_obs"],
                next_critic_obs,
            )

        transition = TensorDict(
            {
                "observations": obs,
                "actions": torch.as_tensor(actions, device=device, dtype=torch.float),
                "next": {
                    "observations": true_next_obs,
                    "rewards": torch.as_tensor(rewards, device=device, dtype=torch.float),
                    "truncations": truncations.long(),
                    "dones": dones.long(),
                },
            },
            batch_size=(envs.num_envs,),
            device=device,
        )
        if envs.asymmetric_obs:
            transition["critic_observations"] = critic_obs
            transition["next"]["critic_observations"] = true_next_critic_obs
        replay_provenance = None
        if admission_enabled:
            assert executed_source_by_group is not None
            source_by_group = executed_source_by_group
            changed = (source_by_group != admission_previous_groups).any(dim=1)
            admission_segment_counter += changed.to(torch.int64)
            admission_segment_step = torch.where(
                changed,
                torch.zeros_like(admission_segment_step),
                admission_segment_step + 1,
            )
            admission_previous_groups.copy_(source_by_group)
            replay_provenance = {
                "behavior_source": option_ids.to(torch.int16),
                "source_by_group": source_by_group.to(torch.int16),
                "executed_group_mask": source_by_group >= 0,
                "segment_id": admission_segment_counter * args.num_envs
                + replay_env_ranks.to(torch.int64),
                "segment_step": admission_segment_step,
                "anchor_id": torch.full(
                    (args.num_envs,), -1, device=device, dtype=torch.int32
                ),
                "env_rank": replay_env_ranks,
                "learner_step": torch.full(
                    (args.num_envs,), global_step, device=device, dtype=torch.int64
                ),
            }
        elif anchor_step is not None:
            replay_provenance = {
                "behavior_source": torch.full(
                    (args.num_envs,), -1, device=device, dtype=torch.int16
                ),
                "source_by_group": torch.full(
                    (args.num_envs, anchor_provenance_group_count),
                    -1,
                    device=device,
                    dtype=torch.int16,
                ),
                "executed_group_mask": torch.zeros(
                    args.num_envs,
                    anchor_provenance_group_count,
                    device=device,
                    dtype=torch.bool,
                ),
                "segment_id": anchor_episode_counts * args.num_envs
                + anchor_env_ranks.to(torch.int64),
                "segment_step": anchor_episode_steps,
                "anchor_id": torch.full(
                    (args.num_envs,), -1, device=device, dtype=torch.int32
                ),
                "env_rank": anchor_env_ranks,
                "learner_step": torch.full(
                    (args.num_envs,), global_step, device=device, dtype=torch.int64
                ),
            }
        if admission_enabled:
            source_slots = option_ids >= 0
            if source_slots.any() and not bool(
                mcg_behavior.admitted_sources[option_ids[source_slots]].all()
            ):
                raise AssertionError("rejected source transition attempted to enter main replay")
            assert admission_execution_counts is not None
            execution_strata = torch.where(
                source_slots,
                option_ids,
                torch.full_like(option_ids, source_bank.num_sources),
            )
            admission_execution_counts += torch.bincount(
                execution_strata, minlength=source_bank.num_sources + 1
            )
        rb.extend(transition, option_ids, provenance=replay_provenance)
        if pending_adaptive_window is not None:
            assert adaptive_admission_controller is not None
            assert admission_segment_tracker is not None
            discarded_partial_segments = 0
            if pending_adaptive_window.snapshot is not None:
                revoked_indices = [
                    admission_snapshot.source_names.index(name)
                    for name in pending_adaptive_window.revoked_sources
                ]
                discarded_partial_segments = admission_segment_tracker.discard_sources(
                    revoked_indices
                )
                admission_snapshot = pending_adaptive_window.snapshot
                if admission_snapshot is not adaptive_admission_controller.current_snapshot:
                    raise AssertionError("adaptive controller and train snapshot diverged")
                admitted_tensor = admission_snapshot.admitted_tensor(device)
                candidate_masses = admission_snapshot.candidate_probabilities(
                    tau=float(ptf_cfg["mcg_bootstrap_tau"]), device=device
                )
                mcg_behavior.set_admission_policy(
                    admitted_sources=admitted_tensor,
                    source_logits=torch.tensor(
                        admission_snapshot.source_logits,
                        device=device,
                        dtype=torch.float32,
                    ),
                    student_logit=admission_snapshot.student_logit,
                )
                rb.set_admission_policy(
                    admitted_sources=admitted_tensor,
                    candidate_masses=replay_candidate_masses(
                        candidate_masses, admission_replay_mode
                    ),
                    recency_half_life=float(
                        ptf_cfg["admission_replay_recency_half_life"]
                    ),
                    uniform_mix=float(ptf_cfg["admission_replay_uniform_mix"]),
                    priority_alpha=float(ptf_cfg["admission_replay_priority_alpha"]),
                )
                if (
                    admission_snapshot.exact_abstain
                    and rb.admission_source_authority_active
                ):
                    rb.set_admission_source_authority(
                        False, reason="adaptive_exact_abstention"
                    )

            replay_audit = rb.admission_audit() or {}
            window_event = {
                **pending_adaptive_window.as_dict(),
                "step": int(pending_adaptive_window.completed_step),
                "execution_counts_at_apply": (
                    admission_execution_counts.detach().cpu().tolist()
                ),
                "discarded_partial_segments": int(discarded_partial_segments),
                "replay_at_apply": {
                    key: replay_audit.get(key)
                    for key in (
                        "admitted_sources",
                        "candidate_masses",
                        "source_authority_active",
                        "sampling_phase",
                        "main_buffer_counts",
                        "active_buffer_counts",
                        "effective_replay_masses",
                        "critic_sample_counts",
                        "actor_independent_sample_counts",
                    )
                },
            }
            admission_history.append(window_event)
            adaptive_admission_logs.update(
                {
                    "ptf_admission/adaptive_window_index": float(
                        pending_adaptive_window.window_index
                    ),
                    "ptf_admission/adaptive_revoked_this_window": float(
                        len(pending_adaptive_window.revoked_sources)
                    ),
                    "ptf_admission/adaptive_admitted_count": float(
                        sum(admission_snapshot.admitted)
                    ),
                    "ptf_admission/adaptive_exact_abstain": float(
                        admission_snapshot.exact_abstain
                    ),
                    "ptf_admission/adaptive_discarded_partial": float(
                        discarded_partial_segments
                    ),
                }
            )
            for index, statistics in enumerate(
                pending_adaptive_window.statistics
            ):
                name = statistics.candidate
                adaptive_admission_logs[
                    f"ptf_admission/window_count/{name}"
                ] = float(statistics.count)
                if statistics.mean is not None:
                    adaptive_admission_logs[
                        f"ptf_admission/window_mean/{name}"
                    ] = float(statistics.mean)
                if statistics.lower_bound is not None:
                    adaptive_admission_logs[
                        f"ptf_admission/window_lcb/{name}"
                    ] = float(statistics.lower_bound)
                    adaptive_admission_logs[
                        f"ptf_admission/window_ucb/{name}"
                    ] = float(statistics.upper_bound)
                if index < source_bank.num_sources:
                    adaptive_admission_logs[
                        f"ptf_admission/persistence/{name}"
                    ] = float(pending_adaptive_window.persistence_counts[index])
                    adaptive_admission_logs[
                        f"ptf_admission/admitted/{name}"
                    ] = float(admission_snapshot.admitted[index])
            print(
                "[adaptive-admission] "
                f"step={pending_adaptive_window.completed_step} "
                f"window={pending_adaptive_window.window_index} "
                f"revoked={list(pending_adaptive_window.revoked_sources)} "
                f"admitted={list(admission_snapshot.admitted_names)} "
                f"discarded_partial={discarded_partial_segments}"
            )
        if anchor_step is not None:
            anchor_episode_steps += 1
            reset_mask = dones.bool()
            anchor_episode_counts += reset_mask.to(torch.int64)
            anchor_episode_steps = torch.where(
                reset_mask, torch.zeros_like(anchor_episode_steps), anchor_episode_steps
            )

        obs = next_obs
        if envs.asymmetric_obs:
            critic_obs = next_critic_obs

        if global_step > args.learning_starts:
            for i in range(args.num_updates):
                data = rb.sample(max(1, args.batch_size // args.num_envs))
                data["raw_observations"] = data["observations"].clone()
                data["next"]["raw_observations"] = data["next"]["observations"].clone()
                data["observations"] = normalize_obs(data["observations"])
                data["next"]["observations"] = normalize_obs(data["next"]["observations"])
                if envs.asymmetric_obs:
                    data["critic_observations"] = normalize_critic_obs(data["critic_observations"])
                    data["next"]["critic_observations"] = normalize_critic_obs(data["next"]["critic_observations"])
                raw_rewards = data["next"]["rewards"]
                if env_type in ["mtbench"] and args.reward_normalization:
                    task_ids_one_hot = data["observations"][..., -envs.num_tasks :]
                    task_indices = torch.argmax(task_ids_one_hot, dim=1)
                    data["next"]["rewards"] = normalize_reward(raw_rewards, task_ids=task_indices)
                else:
                    data["next"]["rewards"] = normalize_reward(raw_rewards)

                logs_dict = update_main(data, logs_dict)
                critic_update_count += 1
                actor_should_update = (
                    (i % args.policy_frequency == 1)
                    if args.num_updates > 1
                    else (global_step % args.policy_frequency == 0)
                ) and actor_updates_enabled(global_step, actor_update_start_step)
                if actor_should_update:
                    if mcg_enabled and mcg_replay_mode in ("actor_only", "critic_only", "split"):
                        # split replay(裁定 2026-07-02): actor 用独立按 actor 权重
                        # 采样的 batch(critic 的主 batch 用 critic 权重)——坏源状态
                        # 不再大量进入 actor 的 Q(s,π(s)) 最大化。
                        # 后处理必须与主 data 完全对齐(首版漏 normalize_obs 导致
                        # actor 在 raw obs 上更新,毁掉第一轮归因矩阵——血的教训)。
                        data_pol = rb.sample(max(1, args.batch_size // args.num_envs), role="actor")
                        data_pol["raw_observations"] = data_pol["observations"].clone()
                        data_pol["next"]["raw_observations"] = data_pol["next"]["observations"].clone()
                        data_pol["observations"] = normalize_obs(data_pol["observations"])
                        data_pol["next"]["observations"] = normalize_obs(data_pol["next"]["observations"])
                        if envs.asymmetric_obs:
                            data_pol["critic_observations"] = normalize_critic_obs(data_pol["critic_observations"])
                            data_pol["next"]["critic_observations"] = normalize_critic_obs(
                                data_pol["next"]["critic_observations"])
                    else:
                        data_pol = data
                    logs_dict = update_pol(data_pol, logs_dict, global_step)
                    actor_update_count += 1
                if (
                    not mcg_enabled
                    and not isolate_classic_ptf
                    and (not released_code_fidelity or i == 0)
                ):
                    # MCG 模式下 Q_o/β 不参与调度与蒸馏,跳过 option 模块训练。
                    # Released PTF performs one option/termination update per
                    # outer environment step; FastTD3 may perform multiple
                    # critic updates, so fidelity mode binds this call to i=0.
                    option_data = data
                    if released_code_fidelity:
                        # Released PTF samples a separate option replay batch
                        # instead of reusing the much larger FastTD3 critic
                        # batch. Vectorized storage requires at least one
                        # transition per environment.
                        option_batch_per_env = max(
                            1,
                            math.ceil(
                                int(ptf_cfg["option_batch_size"])
                                / args.num_envs
                            ),
                        )
                        option_data = rb.sample(option_batch_per_env)
                        option_data["raw_observations"] = option_data[
                            "observations"
                        ].clone()
                        option_data["next"]["raw_observations"] = option_data[
                            "next"
                        ]["observations"].clone()
                        option_data["observations"] = normalize_obs(
                            option_data["observations"],
                            update=False,
                        )
                        option_data["next"]["observations"] = normalize_obs(
                            option_data["next"]["observations"],
                            update=False,
                        )
                        option_raw_rewards = option_data["next"]["rewards"]
                        if env_type in ["mtbench"] and args.reward_normalization:
                            option_task_ids_one_hot = option_data["observations"][
                                ..., -envs.num_tasks :
                            ]
                            option_task_indices = torch.argmax(
                                option_task_ids_one_hot,
                                dim=1,
                            )
                            option_data["next"]["rewards"] = normalize_reward(
                                option_raw_rewards,
                                task_ids=option_task_indices,
                            )
                        else:
                            option_data["next"]["rewards"] = normalize_reward(
                                option_raw_rewards
                            )
                    logs_dict = update_option(
                        option_data,
                        logs_dict,
                        global_step,
                        current_beta_transition=current_beta_transition,
                    )
                soft_update(qnet, qnet_target, args.tau)

            if global_step % 100 == 0 and start_time is not None:
                speed = (global_step - measure_burnin) / (time.time() - start_time)
                pbar.set_description(f"{speed: 4.4f} sps, " + desc)
                with torch.no_grad():
                    logs = {
                        key: value.mean() if isinstance(value, torch.Tensor) else value
                        for key, value in logs_dict.items()
                    }
                    logs.update(
                        {
                            "actor_loss": logs_dict.get("actor_loss", torch.zeros((), device=device)).mean(),
                            "qf_loss": logs_dict.get("qf_loss", torch.zeros((), device=device)).mean(),
                            "qf_max": logs_dict.get("qf_max", torch.zeros((), device=device)).mean(),
                            "qf_min": logs_dict.get("qf_min", torch.zeros((), device=device)).mean(),
                            "actor_grad_norm": logs_dict.get("actor_grad_norm", torch.zeros((), device=device)).mean(),
                            "critic_grad_norm": logs_dict.get("critic_grad_norm", torch.zeros((), device=device)).mean(),
                            "ptf_bootstrap/actor_updates_enabled": torch.tensor(
                                float(
                                    actor_updates_enabled(
                                        global_step, actor_update_start_step
                                    )
                                ),
                                device=device,
                            ),
                            "env_rewards": rewards.mean(),
                            "buffer_rewards": raw_rewards.mean(),
                        }
                    )
                    if pare_runtime is not None:
                        # PARE 的机制量必须在**不依赖 wandb** 的情况下可恢复：
                        # spec §8 的 F3/F7 直接以它们为判据，判据依赖的量不可见
                        # 就等于判据无法实现（已犯过：判据被迫换成更弱的代理）。
                        pare_logs = {
                            k: round(float(v), 6)
                            for k, v in sorted(logs.items())
                            if k.startswith("pare/")
                        }
                        pare_logs["d_skip_count"] = pare_runtime.d_skip_count
                        print(f"[pare] step={global_step} " + json.dumps(pare_logs))
                    if mcg_enabled and mcg_rollout_info:
                        logs.update(mcg_rollout_info)
                    if qmp_enabled and qmp_diag:
                        # 机制诊断,不参与任何 gate(run card §4.2)
                        logs.update(qmp_diag)
                        _rl = qmp_selector.mean_run_lengths()
                        logs["qmp/run_len_student"] = _rl[0]
                        for _i, _n in enumerate(source_bank.names()):
                            logs[f"qmp/run_len_{_n}"] = _rl[_i + 1]
                    if adaptive_admission_enabled and adaptive_admission_logs:
                        logs.update(adaptive_admission_logs)
                    # Classic-PTF observability: these are rollout-state
                    # diagnostics only and do not call either network or consume
                    # RNG. Replay-batch option fractions are already logged by
                    # update_option(); this complementary view tells us which
                    # option is controlling the current call-and-return label.
                    if not isolate_classic_ptf and not mcg_enabled:
                        rollout_hist = torch.bincount(
                            option_selector.current_options,
                            minlength=source_bank.num_options,
                        ).float()
                        rollout_hist = rollout_hist / rollout_hist.sum().clamp_min(1.0)
                        for option_i, option_name in enumerate(source_bank.names()):
                            logs[f"ptf_rollout_option_frac/{option_name}"] = rollout_hist[option_i]
                        logs["ptf_rollout_option_age_mean"] = (
                            option_selector.steps_in_option.float().mean()
                        )
                        with torch.no_grad():
                            rollout_q, rollout_beta = option_module(norm_obs)
                            rollout_current = option_selector.current_options
                            rollout_greedy = rollout_q.argmax(dim=1)
                            current_is_greedy = rollout_current == rollout_greedy
                            current_beta = rollout_beta.gather(
                                1, rollout_current.view(-1, 1)
                            ).squeeze(1)
                            logs["ptf_rollout_current_argmax_rate"] = (
                                current_is_greedy.float().mean()
                            )
                            logs["ptf_rollout_beta_when_argmax"] = (
                                current_beta[current_is_greedy].mean()
                                if current_is_greedy.any()
                                else torch.zeros((), device=device)
                            )
                            logs["ptf_rollout_beta_when_non_argmax"] = (
                                current_beta[~current_is_greedy].mean()
                                if (~current_is_greedy).any()
                                else torch.zeros((), device=device)
                            )
                            if rollout_q.shape[1] >= 2:
                                rollout_top2 = torch.topk(
                                    rollout_q,
                                    k=2,
                                    dim=1,
                                ).values
                                rollout_gap = rollout_top2[:, 0] - rollout_top2[:, 1]
                                rollout_all_saturated = (
                                    rollout_q.abs() > 0.95
                                ).all(dim=1)
                                logs["ptf_rollout_option_gap_mean"] = (
                                    rollout_gap.mean()
                                )
                                logs["ptf_rollout_option_gap_median"] = (
                                    rollout_gap.median()
                                )
                                logs[
                                    "ptf_rollout_all_options_saturated_fraction"
                                ] = rollout_all_saturated.float().mean()
                                logs[
                                    "ptf_rollout_all_options_saturated_low_gap_fraction"
                                ] = (
                                    rollout_all_saturated
                                    & (rollout_gap < 0.01)
                                ).float().mean()
                        for key, value in option_selector.cumulative_diagnostics().items():
                            logs[f"ptf_rollout_{key}"] = value
                    if args.eval_interval > 0 and global_step % args.eval_interval == 0:
                        print(f"Evaluating at global step {global_step}")
                        eval_avg_return, eval_avg_length = evaluate()
                        # Also surface to stdout so eval curves are recoverable without wandb.
                        print(f"[eval] step={global_step} return={eval_avg_return:.3f} length={eval_avg_length:.1f}")
                        if not isolate_classic_ptf and not mcg_enabled:
                            option_diag = " ".join(
                                f"{name}={float(logs[f'ptf_rollout_option_frac/{name}']):.3f}"
                                for name in source_bank.names()
                            )
                            print(
                                f"[ptf] step={global_step} rollout_options({option_diag}) "
                                f"age={float(logs['ptf_rollout_option_age_mean']):.2f} "
                                f"argmax={float(logs['ptf_rollout_current_argmax_rate']):.3f} "
                                f"beta_term_rate={float(logs['ptf_rollout_beta_termination_rate']):.3f} "
                                f"beta_selected={float(logs.get('ptf_beta_selected_mean', 0.0)):.3f} "
                                f"transfer_weight={float(logs.get('ptf_transfer_weight_mean', 0.0)):.4f} "
                                f"compat={float(logs.get('ptf_source_compat_mean', 0.0)):.3f} "
                                f"q_gap={float(logs.get('ptf_rollout_option_gap_median', 0.0)):.4f} "
                                f"q_sat_lowgap={float(logs.get('ptf_rollout_all_options_saturated_low_gap_fraction', 0.0)):.3f} "
                                f"q_target_oob={float(logs.get('ptf_option_td_target_oob_fraction', 0.0)):.3f}"
                            )
                        if env_type in ["humanoid_bench", "isaaclab", "mtbench"]:
                            obs = envs.reset()
                            if released_code_fidelity:
                                option_selector.reset()
                        logs["eval_avg_return"] = eval_avg_return
                        logs["eval_avg_length"] = eval_avg_length
                    if args.render_interval > 0 and global_step % args.render_interval == 0:
                        renders = render_with_rollout()
                        render_video = wandb.Video(
                            np.array(renders).transpose(0, 3, 1, 2),
                            fps=30,
                            format="gif",
                        )
                        logs["render_video"] = render_video
                if args.use_wandb:
                    wandb.log(
                        {
                            "speed": speed,
                            "frame": global_step * args.num_envs,
                            "critic_lr": q_scheduler.get_last_lr()[0],
                            "actor_lr": actor_scheduler.get_last_lr()[0],
                            **logs,
                        },
                        step=global_step,
                    )

            if args.save_interval > 0 and global_step > 0 and global_step % args.save_interval == 0:
                print(f"Saving official PTF model at global step {global_step}")
                save_ptf_params(
                    global_step,
                    actor,
                    qnet,
                    qnet_target,
                    obs_normalizer,
                    critic_obs_normalizer,
                    args,
                    ptf_cfg,
                    option_module,
                    option_target,
                    option_optimizer,
                    beta_optimizer,
                    source_bank,
                    f"models/{run_name}_{global_step}.pt",
                    admission_audit=(
                        {
                            **(rb.admission_audit() or {}),
                            "execution_counts": admission_execution_counts.detach().cpu().tolist(),
                            "actor_sampling": "shared_critic_batch",
                            "decision_history": list(admission_history),
                        }
                        if admission_enabled
                        else None
                    ),
                    training_audit={
                        "critic_update_count": int(critic_update_count),
                        "actor_update_count": int(actor_update_count),
                        "actor_update_start_step": actor_update_start_step,
                    },
                )

        global_step += 1
        actor_scheduler.step()
        q_scheduler.step()
        pbar.update(1)
        # P0 显式 checkpoint:在递增与 scheduler step 之后按"已完成在线步数"
        # 判断——若放在递增前,标号为 N 的文件实际含第 N+1 步数据,且循环退出
        # 前 run_stop_step 本身永远不会被判断到(五次复核阻塞问题 1)。
        if global_step in eval_checkpoint_set:
            print(f"Saving P0 eval checkpoint at completed step {global_step}")
            if mcg_enabled and getattr(mcg_behavior, "episode_prefix_steps", None) is not None:
                _hf = int(getattr(mcg_behavior, "prefix_handoff_count", 0))
                _tr = int(getattr(mcg_behavior, "prefix_truncated_count", 0))
                _tot = _hf + _tr
                print(f"[prefix-audit] handoff={_hf} truncated={_tr} "
                      f"handoff_frac={(_hf/_tot if _tot else float('nan')):.4f}")
            save_ptf_params(
                global_step,
                actor,
                qnet,
                qnet_target,
                obs_normalizer,
                critic_obs_normalizer,
                args,
                ptf_cfg,
                option_module,
                option_target,
                option_optimizer,
                beta_optimizer,
                source_bank,
                f"models/{run_name}_{global_step}.pt",
                admission_audit=(
                    {
                        **(rb.admission_audit() or {}),
                        "execution_counts": admission_execution_counts.detach().cpu().tolist(),
                        "actor_sampling": "shared_critic_batch",
                        "decision_history": list(admission_history),
                    }
                    if admission_enabled
                    else None
                ),
                training_audit={
                    "critic_update_count": int(critic_update_count),
                    "actor_update_count": int(actor_update_count),
                    "actor_update_start_step": actor_update_start_step,
                },
            )
        if anchor_step is not None and global_step == anchor_step:
            if anchor_saved:
                raise AssertionError("paper anchor was already saved")
            module_state = {
                "actor": actor,
                "critic": qnet,
                "critic_target": qnet_target,
                "obs_normalizer": obs_normalizer,
                "critic_obs_normalizer": critic_obs_normalizer,
                "reward_normalizer": reward_normalizer,
                "option": option_module,
                "option_target": option_target,
            }
            option_selector_state = {
                "current_options": option_selector.current_options,
                "steps_in_option": option_selector.steps_in_option,
                "needs_reselection": option_selector.needs_reselection,
                "epsilon": option_selector.epsilon,
                "min_duration": option_selector.min_duration,
                "select_on_reset": option_selector.select_on_reset,
                "sample_choices_only_when_needed": (
                    option_selector.sample_choices_only_when_needed
                ),
                "num_options": option_selector.num_options,
                "num_envs": option_selector.num_envs,
            }
            repo_root = Path(__file__).resolve().parents[2]
            official_fasttd3 = repo_root / "fasttd3_ptf/official_code/FastTD3/fast_td3"
            saved_path = save_anchor_bundle(
                anchor_dir,
                completed_vector_steps=global_step,
                num_envs=args.num_envs,
                modules=module_state,
                optimizers={
                    "actor": actor_optimizer,
                    "critic": q_optimizer,
                    "option": option_optimizer,
                    "beta": beta_optimizer,
                },
                schedulers={"actor": actor_scheduler, "critic": q_scheduler},
                scaler=scaler,
                replay=rb,
                configuration={"args": vars(args), "ptf": ptf_cfg},
                auxiliary_state={
                    "critic_update_count": critic_update_count,
                    "actor_update_count": actor_update_count,
                    "option_selector": option_selector_state,
                    "source_names": source_bank.names(),
                    "scope": "learner/replay anchor; simulator anchors are collected separately",
                },
                generators={"option_selector": option_selector.generator},
                provenance_default={
                    "behavior_source": "student",
                    "reason": "scratch anchor with empty source bank and source execution disabled",
                },
                require_complete_replay_provenance=True,
                repo_root=repo_root,
                code_paths=[
                    Path(__file__),
                    Path(__file__).with_name("anchor_io.py"),
                    Path(__file__).with_name("ptf_replay.py"),
                    Path(__file__).with_name("humanoid_bench_env.py"),
                    official_fasttd3 / "fast_td3.py",
                    official_fasttd3 / "fast_td3_utils.py",
                    official_fasttd3 / "hyperparams.py",
                ],
            )
            anchor_saved = True
            print(f"Saved paper-grade learner anchor to {saved_path}")
        if branch_anchor_step is not None and global_step == branch_anchor_step:
            if branch_anchor_saved:
                raise AssertionError("branch anchor was already saved")
            if not admission_enabled or admission_execution_counts is None:
                raise AssertionError(
                    "branch anchor requires admission execution provenance"
                )
            module_state = {
                "actor": actor,
                "critic": qnet,
                "critic_target": qnet_target,
                "obs_normalizer": obs_normalizer,
                "critic_obs_normalizer": critic_obs_normalizer,
                "reward_normalizer": reward_normalizer,
                "option": option_module,
                "option_target": option_target,
            }
            option_selector_state = {
                "current_options": option_selector.current_options,
                "steps_in_option": option_selector.steps_in_option,
                "needs_reselection": option_selector.needs_reselection,
                "epsilon": option_selector.epsilon,
                "min_duration": option_selector.min_duration,
                "select_on_reset": option_selector.select_on_reset,
                "sample_choices_only_when_needed": (
                    option_selector.sample_choices_only_when_needed
                ),
                "num_options": option_selector.num_options,
                "num_envs": option_selector.num_envs,
            }
            repo_root = Path(__file__).resolve().parents[2]
            official_fasttd3 = repo_root / "fasttd3_ptf/official_code/FastTD3/fast_td3"
            saved_path = save_anchor_bundle(
                branch_anchor_dir,
                completed_vector_steps=global_step,
                num_envs=args.num_envs,
                modules=module_state,
                optimizers={
                    "actor": actor_optimizer,
                    "critic": q_optimizer,
                    "option": option_optimizer,
                    "beta": beta_optimizer,
                },
                schedulers={"actor": actor_scheduler, "critic": q_scheduler},
                scaler=scaler,
                replay=rb,
                configuration={"args": vars(args), "ptf": ptf_cfg},
                auxiliary_state={
                    "critic_update_count": critic_update_count,
                    "actor_update_count": actor_update_count,
                    "option_selector": option_selector_state,
                    "source_names": source_bank.names(),
                    "branch_anchor": True,
                    "admission_execution_counts": (
                        admission_execution_counts.detach().cpu().tolist()
                    ),
                    "admission_history": copy.deepcopy(admission_history),
                    "scope": (
                        "complete source-bearing learner/replay branch anchor; "
                        "simulator state resets on continuation"
                    ),
                },
                generators={"option_selector": option_selector.generator},
                provenance_default={
                    "behavior_source": "runtime admission provenance",
                    "reason": "IBR branch boundary",
                },
                require_complete_replay_provenance=True,
                repo_root=repo_root,
                code_paths=[
                    Path(__file__),
                    Path(__file__).with_name("anchor_io.py"),
                    Path(__file__).with_name("ptf_replay.py"),
                    Path(__file__).with_name("humanoid_bench_env.py"),
                    official_fasttd3 / "fast_td3.py",
                    official_fasttd3 / "fast_td3_utils.py",
                    official_fasttd3 / "hyperparams.py",
                ],
            )
            branch_anchor_saved = True
            print(f"Saved complete branch anchor to {saved_path}")

    save_ptf_params(
        global_step,
        actor,
        qnet,
        qnet_target,
        obs_normalizer,
        critic_obs_normalizer,
        args,
        ptf_cfg,
        option_module,
        option_target,
        option_optimizer,
        beta_optimizer,
        source_bank,
        f"models/{run_name}_final.pt",
        admission_audit=(
            {
                **(rb.admission_audit() or {}),
                "execution_counts": admission_execution_counts.detach().cpu().tolist(),
                "actor_sampling": "shared_critic_batch",
                "decision_history": list(admission_history),
            }
            if admission_enabled
            else None
        ),
        training_audit={
            "critic_update_count": int(critic_update_count),
            "actor_update_count": int(actor_update_count),
            "actor_update_start_step": actor_update_start_step,
        },
    )
    if anchor_step is not None and not anchor_saved:
        raise AssertionError(f"requested paper anchor at step {anchor_step} was not saved")
    if branch_anchor_step is not None and not branch_anchor_saved:
        raise AssertionError(
            f"requested branch anchor at step {branch_anchor_step} was not saved"
        )


if __name__ == "__main__":
    main()
