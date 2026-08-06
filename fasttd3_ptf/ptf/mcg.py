"""MCG(Modular Critic-Guided policy transfer)核心逻辑。

把 PTF 的 option 从"整条教师策略"推广为"(教师 i, 身体组 g)":

- masked candidate: ã_{i,g} = 学生动作中第 g 组替换为教师 i 的该组动作;
- gating: Δ_{i,g}(s) = Qmin(s, ã_{i,g}) − Qmin(s, a_student) > margin 才迁移
  (option-value 不再单独学习,直接从 target critic 读出——这同时解决了
  v3.1 的调度鸡生蛋问题:U(s,o) ≈ Q(s, ã_o(s)) 不需要成功样本);
- 行为层: per-env per-group 锁存执行(temporal smoothing,继承 PTF
  β/termination 的时间一致性语义);warmup 期 critic 不可信,退化为
  无条件 call-and-return bootstrap(随机教师整动作执行)。

设计依据(2026-06-11 离线探针, logs/probe/modular_gating_push.json):
- part≫full 分化稳定(25k: reach-arms Δ≈0/frac+=0.49 vs reach-full Δ=−6.9/0.05);
- 5k 的 critic 高估学生、全拒教师 ⇒ warmup 必须无条件 bootstrap,gating 中期接管;
- 0.1 分位 gating 几乎永不放行(frac+q10≤0.19) ⇒ v1 用 mean+margin。

v1.1(2026-06-11, window 负迁移 −153 后的 safety patch):
- sign-only gate(Δ>0)在 Δ 噪声与真实 advantage 同量级时不具备负迁移免疫力
  (window 实证: Δ_best 全程为负但 gate_rate 仍 0.2-0.3=噪声右尾假阳性);
- **paired head delta**: 每个 critic head 内部做 paired difference 再对 head 取
  min(消除"min head 切换"的额外噪声),替代先各取 min 再相减;
- **null 校准 margin**: 用"教师动作×打乱状态"构造 null delta 分布,取其高分位
  (默认 q95)作为组级 margin m_g——gate 从"Δ>0"变为"Δ 显著高于噪声右尾",
  false-positive rate 可控可报告;
- **confidence 蒸馏**: 0/1 gate → c=σ((Δ−m_g)/τ) 连续置信度加权。
"""
from __future__ import annotations

from typing import Callable

import torch

from fasttd3_ptf.ptf.action_schema import h1hand_default_action_schema
from fasttd3_ptf.ptf.distillation import masked_action_distillation_loss

QMinFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
# 返回两个 head 的标量 Q(各自 [B]),供 paired delta 用
QHeadsFn = Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]

DEFAULT_GROUPS = ("legs_torso", "arms", "hands")


class ModularGating:
    """masked candidate 构造 + Δ_{i,g} 计算 + per-group 教师选择。"""

    def __init__(
        self,
        action_dim: int,
        groups: tuple[str, ...] | list[str] = DEFAULT_GROUPS,
        device: torch.device | str = "cpu",
        margin: float = 0.0,
    ):
        schema = h1hand_default_action_schema()
        if int(action_dim) != schema.dim:
            raise ValueError(f"MCG group schema expects action_dim={schema.dim}, got {action_dim}")
        self.device = torch.device(device)
        self.groups = tuple(groups)
        self.margin = float(margin)
        # [G, A] 0/1 掩码;各组应互不重叠(重叠会让 candidate 语义含糊)
        self.group_masks = torch.stack(
            [schema.get(g).mask(schema.dim, device=self.device) for g in self.groups], dim=0
        )
        overlap = self.group_masks.sum(dim=0).max().item()
        if overlap > 1:
            raise ValueError(f"MCG groups {self.groups} overlap (a dim appears in {int(overlap)} groups)")

    @property
    def num_groups(self) -> int:
        return len(self.groups)

    @torch.no_grad()
    def deltas(
        self,
        qheads_fn: QHeadsFn,
        critic_obs: torch.Tensor,
        a_student: torch.Tensor,
        src_actions: torch.Tensor,
    ) -> torch.Tensor:
        """paired head delta: Δ[B,S,G] = min_h [Q_h(s,ã_{i,g}) − Q_h(s,a_student)]。

        每个 head 内部先做 paired difference 再对 head 取 min——比"两边各取
        min 再相减"少一层 min-head 切换噪声(v1.1 修正)。
        qheads_fn(critic_obs, actions) -> (q1[B], q2[B])。
        """
        batch, num_src = src_actions.shape[0], src_actions.shape[1]
        q1_ref, q2_ref = qheads_fn(critic_obs, a_student)
        out = torch.empty(batch, num_src, self.num_groups, device=a_student.device, dtype=torch.float32)
        for i in range(num_src):
            for g in range(self.num_groups):
                gm = self.group_masks[g].to(a_student.device).bool()
                cand = torch.where(gm, src_actions[:, i, :], a_student)
                q1_c, q2_c = qheads_fn(critic_obs, cand)
                out[:, i, g] = torch.minimum(q1_c - q1_ref, q2_c - q2_ref).float()
        return out

    @torch.no_grad()
    def null_margins(
        self,
        qheads_fn: QHeadsFn,
        critic_obs: torch.Tensor,
        a_student: torch.Tensor,
        src_actions: torch.Tensor,
        quantile: float = 0.95,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """null 校准 margin m[G]:教师动作×打乱状态的 paired delta 高分位。

        null 候选 = 把 batch 内其他状态下的教师动作拼给当前状态——语义上是
        "与当前状态无关的教师建议"。它的 Δ 分布刻画了 critic 对该组动作替换
        的纯噪声响应;真实教师的 Δ 必须显著高于这个右尾才放行。
        每组聚合所有教师的 null 样本(B×S 个)取 quantile。
        """
        batch, num_src = src_actions.shape[0], src_actions.shape[1]
        perm = torch.randperm(batch, generator=generator).to(a_student.device)
        q1_ref, q2_ref = qheads_fn(critic_obs, a_student)
        margins = torch.empty(self.num_groups, device=a_student.device, dtype=torch.float32)
        for g in range(self.num_groups):
            gm = self.group_masks[g].to(a_student.device).bool()
            null_d = []
            for i in range(num_src):
                cand = torch.where(gm, src_actions[perm, i, :], a_student)
                q1_c, q2_c = qheads_fn(critic_obs, cand)
                null_d.append(torch.minimum(q1_c - q1_ref, q2_c - q2_ref))
            margins[g] = torch.quantile(torch.cat(null_d).float(), quantile)
        return margins

    def select(
        self,
        deltas: torch.Tensor,
        margins: torch.Tensor | None = None,
        conf_tau: float = 0.1,
        source_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """每组取 Δ 最大的教师;significance = Δ_best − m_g;gate/confidence。

        margins=None 时退化为 v1 的 sign 模式(m_g=self.margin,供 ablation)。
        返回 (best[B,G] long, sig[B,G], gate[B,G] bool, conf[B,G] float)。
        """
        if source_mask is not None:
            source_mask = source_mask.to(device=deltas.device, dtype=torch.bool).view(-1)
            if source_mask.numel() != deltas.shape[1]:
                raise ValueError("source_mask does not match delta source dimension")
            deltas = torch.where(
                source_mask.view(1, -1, 1),
                deltas,
                torch.full_like(deltas, float("-inf")),
            )
        best_delta, best = deltas.max(dim=1)
        if margins is None:
            m = torch.full_like(best_delta[:1, :], self.margin)
        else:
            m = margins.view(1, -1).clamp_min(self.margin)
        sig = best_delta - m
        gate = sig > 0
        conf = torch.sigmoid(sig / max(conf_tau, 1e-6))
        return best, sig, gate, conf


def mcg_distillation_loss(
    pi_action: torch.Tensor,
    src_actions: torch.Tensor,
    best: torch.Tensor,
    gate: torch.Tensor,
    group_masks: torch.Tensor,
    loss_type: str = "huber",
    delta: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """per-(样本,组) 加权的模块化蒸馏。

    对每组 g 只向该组当前最优教师蒸馏;gate 可以是 bool(v1 硬门控)或 float
    置信度(v1.1: c=σ((Δ−m_g)/τ),显著性越高蒸得越重)。
    返回 (per_sample[B], active[B]);per_sample = Σ_g w_g·l_g(组间独立加权,
    不按组数归一——权重本身携带显著性信息)。
    """
    batch = pi_action.shape[0]
    num_groups = group_masks.shape[0]
    total = torch.zeros(batch, device=pi_action.device, dtype=pi_action.dtype)
    w_sum = torch.zeros(batch, device=pi_action.device, dtype=pi_action.dtype)
    idx = torch.arange(batch, device=pi_action.device)
    for g in range(num_groups):
        target = src_actions[idx, best[:, g], :]
        l_g = masked_action_distillation_loss(
            pi_action, target, group_masks[g].to(pi_action.device), loss_type=loss_type, delta=delta
        )
        w = gate[:, g].to(pi_action.dtype)
        total = total + w * l_g
        w_sum = w_sum + w
    return total, w_sum > 1e-3


class AdmissionSegmentTracker:
    """Reward/length bookkeeping for admission-bootstrap behavior segments.

    Candidate ids are ``0..num_sources-1`` for sources and ``num_sources`` for
    the student.  A segment is naturally closed after its final environment
    step (horizon expiry or done).  Adaptive revocation may discard a partial
    source segment; already written replay transitions remain governed by
    provenance masks, but the partial return is not reused in a later window.
    """

    def __init__(
        self,
        *,
        num_envs: int,
        num_sources: int,
        device: torch.device | str,
    ) -> None:
        if num_envs <= 0:
            raise ValueError("num_envs must be positive")
        if num_sources <= 0:
            raise ValueError("num_sources must be positive")
        self.num_envs = int(num_envs)
        self.num_sources = int(num_sources)
        self.student_candidate = self.num_sources
        self.device = torch.device(device)
        self.active_candidate = torch.full(
            (self.num_envs,), -1, device=self.device, dtype=torch.long
        )
        self.reward_sum = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float32
        )
        self.length = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long
        )

    @torch.no_grad()
    def observe(
        self,
        *,
        executed_candidates: torch.Tensor,
        rewards: torch.Tensor,
        natural_ends: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Accumulate one environment step and return naturally completed segments."""

        candidates = executed_candidates.to(
            self.device, dtype=torch.long
        ).view(self.num_envs)
        if bool(((candidates < 0) | (candidates > self.student_candidate)).any()):
            raise ValueError("executed candidate is outside sources + student")
        rewards = rewards.to(self.device, dtype=torch.float32).view(self.num_envs)
        ends = natural_ends.to(self.device, dtype=torch.bool).view(self.num_envs)
        starting = self.active_candidate < 0
        self.active_candidate[starting] = candidates[starting]
        mismatch = self.active_candidate != candidates
        if bool(mismatch.any()):
            raise AssertionError(
                "admission candidate changed before its segment was closed or discarded"
            )
        self.reward_sum += rewards
        self.length += 1

        completed_ids = self.active_candidate[ends].clone()
        completed_means = (
            self.reward_sum[ends]
            / self.length[ends].clamp_min(1).to(self.reward_sum.dtype)
        ).clone()
        self._reset(ends)
        return completed_ids, completed_means

    @torch.no_grad()
    def discard_sources(self, source_indices: torch.Tensor | list[int] | tuple[int, ...]) -> int:
        """Discard partial segments whose currently executing source was revoked."""

        indices = torch.as_tensor(source_indices, device=self.device, dtype=torch.long).view(-1)
        if indices.numel() == 0:
            return 0
        if bool(((indices < 0) | (indices >= self.num_sources)).any()):
            raise ValueError("discard_sources accepts source ids only")
        discard = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        for index in indices.tolist():
            discard |= self.active_candidate == int(index)
        count = int(discard.sum())
        self._reset(discard)
        return count

    @torch.no_grad()
    def _reset(self, mask: torch.Tensor) -> None:
        if bool(mask.any()):
            self.active_candidate[mask] = -1
            self.reward_sum[mask] = 0.0
            self.length[mask] = 0


class McgBehaviorController:
    """rollout 行为层:per-env per-group 锁存的教师执行(call-and-return 的模块化推广)。

    - warmup 模式(best=None):per-env 以 warmup_exec_prob 抽一个教师整动作执行
      (整动作!教师闭环依赖全身协调,这是 pilot v3 的实现教训),锁 warmup_min_steps;
      具体抽源方式由 warmup_mode 决定(random/safe_bootstrap/online_bootstrap/
      admission_bootstrap,见 __init__ 各参数注释);
    - gated 模式:per-(env,group) 锁存到期时,若 gate 放行且 bernoulli(exec_prob)
      命中则切到该组最优教师,否则回学生;锁存期内保持当前选择(temporal
      consistency——PTF termination 语义的模块化对应物);
    - done 的 env 全组回学生并清零锁存。
    """

    def __init__(
        self,
        num_envs: int,
        num_groups: int,
        device: torch.device | str,
        group_masks: torch.Tensor,
        min_steps: int = 10,
        warmup_min_steps: int = 25,
        exec_prob: float = 0.3,
        warmup_exec_prob: float = 0.5,
        seed: int = 0,
        warmup_mode: str = "random",
        bootstrap_weights: torch.Tensor | None = None,
        bootstrap_horizons: torch.Tensor | None = None,
        bootstrap_tau: float = 1.0,
        online_tau: float = 0.5,
        online_eps: float = 0.1,
        online_prior_steps: int = 0,
        online_ema_n: float = 2000.0,
        abstain_gate: bool = False,
        abstain_delta_frac: float = 0.5,
        abstain_k_steps: int = 2000,
        abstain_eps: float = 0.02,
        online_horizons: tuple[int, ...] | list[int] | None = None,
        admitted_sources: torch.Tensor | None = None,
        admission_student_logit: float = 0.0,
        episode_prefix_steps: int | None = None,
    ):
        self.num_envs = int(num_envs)
        self.num_groups = int(num_groups)
        self.device = torch.device(device)
        self.group_masks = group_masks.to(self.device)
        self.min_steps = int(min_steps)
        self.warmup_min_steps = int(warmup_min_steps)
        self.exec_prob = float(exec_prob)
        self.warmup_exec_prob = float(warmup_exec_prob)
        if warmup_mode not in (
            "random", "safe_bootstrap", "online_bootstrap", "admission_bootstrap"
        ):
            raise ValueError(f"unknown warmup_mode: {warmup_mode}")
        self.warmup_mode = warmup_mode
        # safe_bootstrap(RBO-PTF 主方法): 按 target-environment probe 的
        # per-source reward-bearing weight 抽源(替代 random)，并消费 bank 中的
        # per-source horizon。当前论文默认 WFix bank 统一使用 h25；历史 safe
        # bank 可保留不同 horizon 作消融。probe weight 是相对 allocation prior，
        # 不是校准的 transfer ROI，horizon 也未实现自动安全保证。
        self.bootstrap_weights = None if bootstrap_weights is None else bootstrap_weights.to(self.device).float()
        self.bootstrap_horizons = None if bootstrap_horizons is None else bootstrap_horizons.to(self.device).long()
        self.bootstrap_tau = float(bootstrap_tau)
        if warmup_mode in ("safe_bootstrap", "online_bootstrap", "admission_bootstrap") and (
            self.bootstrap_weights is None or self.bootstrap_horizons is None
        ):
            raise ValueError(f"warmup_mode={warmup_mode} requires bootstrap_weights and bootstrap_horizons")
        n_bootstrap_sources = 0 if self.bootstrap_weights is None else int(self.bootstrap_weights.numel())
        if admitted_sources is None:
            admitted_sources = torch.ones(n_bootstrap_sources, dtype=torch.bool, device=self.device)
        self.admitted_sources = admitted_sources.to(self.device, dtype=torch.bool).view(-1)
        if self.admitted_sources.numel() != n_bootstrap_sources:
            raise ValueError("admitted_sources does not match bootstrap source count")
        self.admission_student_logit = float(admission_student_logit)
        # online_bootstrap(student-as-arm, 导师意见 2026-07-02): 学生与 S 个教师作
        # (S+1) 个平等 arm(idx S=学生)。arm_value=各 arm 执行期 per-step reward 的
        # count-based EMA(update_arm_reward 结算); 选择 = w_p(t) 先验分支(=
        # safe_bootstrap 行为,暖启动+冷启动数据积累,w_p 线性衰减到 0) + (1-w_p)
        # 在线分支 softmax(zscore(value)/tau)+eps·uniform。坏源(如 crawl 的站立
        # 教师)在线 return 低时student share可上升。该信号只解释为execution
        # operational feedback：door上可近零代价soft abstain，但crawl/stair已证明
        # 它不等于replay data utility，也不能保证所有负迁移自动关闭。
        # zscore 跨 arm 归一使tau对reward尺度较不敏感。
        self.online_tau = float(online_tau)
        self.online_eps = float(online_eps)
        self.online_prior_steps = int(online_prior_steps)
        self.online_ema_n = float(online_ema_n)
        self._online_step = 0
        # T-gated transfer/abstain 状态机(ChatGPT 裁定 2026-07-02, 合并"硬 abstain"
        # 与"降权自适应激活"): 先验窗口后每步判定"全源 arm value < student − δ"
        # (δ=delta_frac·std(values), 尺度不变), 连续 k_steps 成立 → abstain mode
        # (teacher 抽样→abstain_eps 的 probe floor, replay 源权重由 train 侧降到
        # floor); 条件连续 k_steps 不成立 → 退回 transfer mode(可逆防误判)。
        # 预演: crawl(0.412/0.431/0.435 vs stu 0.455, δ≈0.009)正确判 abstain;
        # pole(walk 0.748 ≥ stu 0.742−0.041)正确留 transfer。
        self.abstain_gate = bool(abstain_gate)
        self.abstain_delta_frac = float(abstain_delta_frac)
        self.abstain_k_steps = int(abstain_k_steps)
        self.abstain_eps = float(abstain_eps)
        self.abstain_mode = False
        self._abstain_ctr = 0
        self._recover_ctr = 0
        # horizon-arm 扩展(ChatGPT 裁定 2026-07-03): stair 证明 transferability
        # 依赖 (source, horizon) 而非 source 单独——同一源库 h=25 的
        # wfix/onlineb/obrw 全部负迁移, 而 safe(h=50) 超 scratch。arm 空间从
        # S+1 扩为 S×H+1: arm a<S×H 映射 (source=a//H, horizon=H[a%H]),
        # source-major 排列, 学生仍是最后一个 arm。先验分支同源各档同先验
        # (bank 权重抽源 + horizon 档均匀), 靠在线 per-step reward EMA 区分——
        # h50 arm 的 EMA 天然含第 26-50 步的 reward(如登阶 burst), h25 arm
        # 不含, 归属即分辨(无需 snippet-level return, 也避免其 γ^k 折扣
        # 系统性压低长 horizon 后段价值的内在矛盾)。
        self.online_horizons = None
        if online_horizons is not None:
            if warmup_mode != "online_bootstrap":
                raise ValueError("online_horizons requires warmup_mode=online_bootstrap")
            self.online_horizons = torch.tensor(
                [int(h) for h in online_horizons], device=self.device, dtype=torch.long
            )
            if self.online_horizons.numel() == 0:
                raise ValueError("online_horizons must be non-empty")
        if warmup_mode == "online_bootstrap":
            n_src = int(self.bootstrap_weights.shape[0])
            if self.online_horizons is not None:
                n_h = int(self.online_horizons.shape[0])
                # per-arm horizon 表(source-major): arm=src*n_h+h_idx
                self.arm_horizons = self.online_horizons.repeat(n_src)
                n_arm = n_src * n_h + 1
            else:
                self.arm_horizons = None
                n_arm = n_src + 1
            self.arm_value = torch.zeros(n_arm, device=self.device)
            self.arm_count = torch.zeros(n_arm, device=self.device)
        else:
            self.arm_horizons = None
            self.arm_value = None
            self.arm_count = None
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(int(seed))
        # -1 = 学生;>=0 = 教师索引
        self.current = torch.full((self.num_envs, self.num_groups), -1, device=self.device, dtype=torch.long)
        # current 恒存 source id(动作组装/exec_share 消费); current_arm 存 arm id
        # (online_bootstrap 的 EMA 结算与 replay options 消费)。单 horizon 时两者
        # 同值; multi-horizon 时 current = current_arm // n_h。
        self.current_arm = torch.full((self.num_envs, self.num_groups), -1, device=self.device, dtype=torch.long)
        self.steps_left = torch.zeros(self.num_envs, self.num_groups, device=self.device, dtype=torch.long)
        # episode-prefix handoff（Door 通道分解之后的接口消融）：
        # 关闭（None）时行为与历史逐位相同——每次 latch 到期都重新抽。
        # 开启时一个 episode 内**只在起点抽一次**；若抽中 source 则连续执行
        # episode_prefix_steps 步，此后锁定 student 直到该 episode 结束。
        # 这样把"随机碎片式介入"换成"完整 episode 前缀 handoff"，而不引入
        # learned termination。
        self.episode_prefix_steps = (
            None if episode_prefix_steps is None else int(episode_prefix_steps)
        )
        self._episode_decided = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        # 锁定 student 直到 episode 结束所用的哨兵步数（远大于任一 episode 长度）
        self._prefix_lock_steps = 1 << 30
        # prefix 审计（只读计数，不影响行为）：
        #   handoff  = prefix 跑满后成功交给 student 的次数
        #   truncated= episode 在 prefix 未跑满时就结束的次数（source 占满该 episode）
        self.prefix_handoff_count = 0
        self.prefix_truncated_count = 0

    def _rand(self, *shape) -> torch.Tensor:
        return torch.rand(*shape, generator=self.generator).to(self.device)

    @torch.no_grad()
    def set_admitted_sources(self, admitted_sources: torch.Tensor) -> None:
        """Atomically replace the active source set and revoke latched sources."""

        admitted = admitted_sources.to(self.device, dtype=torch.bool).view(-1)
        if admitted.shape != self.admitted_sources.shape:
            raise ValueError("admitted source mask shape changed")
        self.admitted_sources.copy_(admitted)
        active = self.current >= 0
        revoked = active & ~self.admitted_sources[self.current.clamp_min(0)]
        if revoked.any():
            self.current[revoked] = -1
            self.current_arm[revoked] = -1
            self.steps_left[revoked] = 0

    @torch.no_grad()
    def set_admission_policy(
        self,
        *,
        admitted_sources: torch.Tensor,
        source_logits: torch.Tensor | None = None,
        student_logit: float | None = None,
    ) -> None:
        """Apply one explicit decision and immediately release revoked latches."""

        self.set_admitted_sources(admitted_sources)
        if source_logits is not None:
            logits = source_logits.to(self.device, dtype=torch.float32).view(-1)
            if logits.shape != self.bootstrap_weights.shape:
                raise ValueError("admission source logits shape changed")
            self.bootstrap_weights.copy_(logits)
        if student_logit is not None:
            self.admission_student_logit = float(student_logit)

    def admission_probabilities(self) -> torch.Tensor:
        """Probability over ``sources + student`` for admission_bootstrap."""

        num_src = int(self.admitted_sources.numel())
        if not bool(self.admitted_sources.any()):
            out = torch.zeros(num_src + 1, dtype=torch.float32, device=self.device)
            out[-1] = 1.0
            return out
        logits = torch.cat(
            [
                self.bootstrap_weights[:num_src].float(),
                torch.tensor([self.admission_student_logit], device=self.device),
            ]
        ) / self.bootstrap_tau
        logits[:num_src] = torch.where(
            self.admitted_sources,
            logits[:num_src],
            torch.full_like(logits[:num_src], float("-inf")),
        )
        return torch.softmax(logits, dim=0)

    @torch.no_grad()
    def update_arm_reward(self, rewards: torch.Tensor) -> None:
        """online_bootstrap: 本步 reward 归属到各 env 正在执行的 arm(count-based EMA)。

        必须在 envs.step 之后调用——self.current 此刻正是产生该 reward 的 arm
        (done 重置发生在下一次 step() 开头,时序天然对齐)。EMA 步长 alpha =
        n_a / min(count, N_ema):count<N_ema 时是精确 running mean(冷启动快收敛),
        之后退化为窗口≈N_ema 的 EMA(跟踪学生的非平稳进步)。
        """
        if self.warmup_mode != "online_bootstrap":
            return
        s_arm = int(self.arm_value.shape[0]) - 1
        arm = self.current_arm[:, 0]
        arm = torch.where(arm < 0, torch.full_like(arm, s_arm), arm)
        r = rewards.view(-1).to(self.arm_value.dtype)
        for a in arm.unique().tolist():  # arm 数少(S+1),python 循环开销可忽略
            m = arm == a
            n_a = float(m.sum())
            new_cnt = self.arm_count[a] + n_a
            alpha = n_a / float(new_cnt.clamp(max=self.online_ema_n).clamp_min(n_a))
            self.arm_value[a] += alpha * (r[m].mean() - self.arm_value[a])
            self.arm_count[a] = new_cnt

    @torch.no_grad()
    def step(
        self,
        a_student: torch.Tensor,
        src_actions: torch.Tensor,
        best: torch.Tensor | None = None,
        gate: torch.Tensor | None = None,
        dones: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        num_src = src_actions.shape[1]
        if dones is not None:
            d = dones.view(-1).bool()
            if d.any():
                if self.episode_prefix_steps is not None:
                    # episode 结束时仍由 source 执行 => prefix 未跑完，handoff 未发生
                    still_prefix = d & (self.current[:, 0] >= 0)
                    self.prefix_truncated_count += int(still_prefix.sum())
                self.current[d] = -1
                self.current_arm[d] = -1
                self.steps_left[d] = 0
                # 新 episode 起点：允许重新抽一次 prefix 决策
                self._episode_decided[d] = False

        expired = self.steps_left <= 0
        if best is None and self.warmup_mode == "safe_bootstrap":
            # RBO-PTF: 按静态 target-environment reward-bearing weight 抽源，
            # 并按 bank horizon 锁存。最终默认 WFix bank 使用 h25，以便把源选择
            # 与 horizon 解耦；历史 per-source horizon 只作消融。第二批 window /
            # basketball 已表明该静态机制不提供普遍安全或 exact fallback。
            env_exp = expired[:, 0]
            n = int(env_exp.sum())
            if n > 0:
                use_teacher = self._rand(n) < self.warmup_exec_prob
                probs = torch.softmax(self.bootstrap_weights[:num_src] / self.bootstrap_tau, dim=0).cpu()
                tid = torch.multinomial(probs, n, replacement=True, generator=self.generator).to(self.device)
                new = torch.where(use_teacher, tid, torch.full_like(tid, -1))
                self.current[env_exp] = new.view(-1, 1).expand(n, self.num_groups)
                self.current_arm[env_exp] = self.current[env_exp]
                # 锁存步数: 教师 env 用该源 safe horizon; 学生 env 用默认 warmup_min_steps
                horiz = torch.where(new >= 0, self.bootstrap_horizons[new.clamp_min(0)],
                                    torch.full_like(new, self.warmup_min_steps))
                self.steps_left[env_exp] = horiz.view(-1, 1).expand(n, self.num_groups)
        elif best is None and self.warmup_mode == "admission_bootstrap":
            # Student-inclusive execution: sample once from admitted sources +
            # student, with no outer teacher Bernoulli.  Empty admission is an
            # exact deterministic fallback and consumes no source-selection RNG.
            if self.episode_prefix_steps is None:
                env_exp = expired[:, 0]
            else:
                # prefix 模式：只有本 episode 尚未做过决策的 env 才参与抽样；
                # 已决策而 latch 到期的 env 一律锁定为 student 直到 episode 结束。
                env_exp = expired[:, 0] & (~self._episode_decided)
                forced = expired[:, 0] & self._episode_decided
                if bool(forced.any()):
                    # 只有本来在跑 source 的 env 才算一次真实 handoff
                    self.prefix_handoff_count += int((forced & (self.current[:, 0] >= 0)).sum())
                    self.current[forced] = -1
                    self.current_arm[forced] = -1
                    self.steps_left[forced] = self._prefix_lock_steps
            n = int(env_exp.sum())
            if n > 0:
                probs = self.admission_probabilities()
                if not bool(self.admitted_sources.any()):
                    arm = torch.full((n,), num_src, device=self.device, dtype=torch.long)
                else:
                    arm = torch.multinomial(
                        probs.cpu(), n, replacement=True, generator=self.generator
                    ).to(self.device)
                is_student = arm == num_src
                new = torch.where(is_student, torch.full_like(arm, -1), arm)
                self.current[env_exp] = new.view(-1, 1).expand(n, self.num_groups)
                self.current_arm[env_exp] = self.current[env_exp]
                if self.episode_prefix_steps is None:
                    horiz = torch.where(
                        new >= 0,
                        self.bootstrap_horizons[new.clamp_min(0)],
                        torch.full_like(new, self.warmup_min_steps),
                    )
                else:
                    # 抽中 source → 连续执行 prefix 步；抽中 student → 直接锁到 episode 结束
                    horiz = torch.where(
                        new >= 0,
                        torch.full_like(new, self.episode_prefix_steps),
                        torch.full_like(new, self._prefix_lock_steps),
                    )
                    self._episode_decided[env_exp] = True
                self.steps_left[env_exp] = horiz.view(-1, 1).expand(n, self.num_groups)
        elif best is None and self.warmup_mode == "online_bootstrap":
            # student-as-arm: 学生=第 num_src 号 arm 与教师平等竞争,权重在线 EMA。
            # w_p 线性衰减的先验分支保证冷启动数据积累(先验期选的 arm 也在
            # update_arm_reward 里更新 value,在线分支接管时 value 已有数据)。
            self._online_step += 1
            # n_arm_src = 源侧 arm 总数(单 horizon=S, multi-horizon=S×H)
            n_arm_src = int(self.arm_value.shape[0]) - 1
            n_h = 0 if self.online_horizons is None else int(self.online_horizons.shape[0])
            if self.abstain_gate and self._online_step > self.online_prior_steps:
                # T-gated 状态机: 全源劣于 student−δ 持续 k_steps → abstain(可逆)
                v_all = self.arm_value
                delta = self.abstain_delta_frac * float(v_all.std())
                all_worse = bool((v_all[:n_arm_src] < v_all[n_arm_src] - delta).all())
                if not self.abstain_mode:
                    self._abstain_ctr = self._abstain_ctr + 1 if all_worse else 0
                    if self._abstain_ctr >= self.abstain_k_steps:
                        self.abstain_mode = True
                        self._recover_ctr = 0
                else:
                    self._recover_ctr = 0 if all_worse else self._recover_ctr + 1
                    if self._recover_ctr >= self.abstain_k_steps:
                        self.abstain_mode = False
                        self._abstain_ctr = 0
            env_exp = expired[:, 0]
            n = int(env_exp.sum())
            if n > 0:
                if self.abstain_mode:
                    # abstain: 几乎全学生,仅留 abstain_eps 的均匀 probe(维持
                    # arm value 监测,支撑可逆退出)
                    probe = self._rand(n) < self.abstain_eps
                    tid = torch.randint(0, max(n_arm_src, 1), (n,), generator=self.generator).to(self.device)
                    arm = torch.where(probe, tid, torch.full_like(tid, n_arm_src))
                else:
                    w_p = max(0.0, 1.0 - self._online_step / max(1, self.online_prior_steps))
                    use_prior = self._rand(n) < w_p
                    # 先验分支 = safe_bootstrap 行为(bernoulli 学生 + softmax(bank 权重))
                    use_teacher = self._rand(n) < self.warmup_exec_prob
                    p_prior = torch.softmax(self.bootstrap_weights[:num_src] / self.bootstrap_tau, dim=0).cpu()
                    tid = torch.multinomial(p_prior, n, replacement=True, generator=self.generator).to(self.device)
                    if n_h:
                        # 同源各 horizon 档同先验: bank 权重只到源级, 档间均匀,
                        # 由在线 EMA 竞争区分(禁任务名/档位人工先验)
                        h_idx = torch.randint(0, n_h, (n,), generator=self.generator)
                        tid = tid * n_h + h_idx.to(self.device)
                    arm_p = torch.where(use_teacher, tid, torch.full_like(tid, n_arm_src))
                    # 在线分支: zscore over arms(尺度不变) + eps-floor(防 EMA 死锁在先验;
                    # 初期 value 全 0 时 zscore 全 0 → softmax 均匀,自然退化为均匀探索)
                    v = self.arm_value
                    z = (v - v.mean()) / v.std().clamp_min(1e-6)
                    p_on = torch.softmax(z / self.online_tau, dim=0)
                    p_on = (1.0 - self.online_eps) * p_on + self.online_eps / (n_arm_src + 1)
                    arm_o = torch.multinomial(p_on.cpu(), n, replacement=True, generator=self.generator).to(self.device)
                    arm = torch.where(use_prior, arm_p, arm_o)
                is_stu = arm >= n_arm_src
                arm_id = torch.where(is_stu, torch.full_like(arm, -1), arm)
                new = torch.where(is_stu, torch.full_like(arm, -1), arm // n_h if n_h else arm)
                self.current[env_exp] = new.view(-1, 1).expand(n, self.num_groups)
                self.current_arm[env_exp] = arm_id.view(-1, 1).expand(n, self.num_groups)
                if n_h:
                    horiz = torch.where(arm_id >= 0, self.arm_horizons[arm_id.clamp_min(0)],
                                        torch.full_like(arm_id, self.warmup_min_steps))
                else:
                    horiz = torch.where(new >= 0, self.bootstrap_horizons[new.clamp_min(0)],
                                        torch.full_like(new, self.warmup_min_steps))
                self.steps_left[env_exp] = horiz.view(-1, 1).expand(n, self.num_groups)
        elif best is None:
            # random warmup(rand 对照组):组同步,整动作,均匀抽源。
            env_exp = expired[:, 0]
            n = int(env_exp.sum())
            if n > 0:
                use_teacher = self._rand(n) < self.warmup_exec_prob
                tid = torch.randint(0, max(num_src, 1), (n,), generator=self.generator).to(self.device)
                new = torch.where(use_teacher, tid, torch.full_like(tid, -1))
                self.current[env_exp] = new.view(-1, 1).expand(n, self.num_groups)
                self.current_arm[env_exp] = self.current[env_exp]
                self.steps_left[env_exp] = self.warmup_min_steps
        else:
            if self.admitted_sources.numel():
                admitted_best = self.admitted_sources[best.clamp_min(0)]
                gate = gate & admitted_best
            choose = gate & (self._rand(self.num_envs, self.num_groups) < self.exec_prob)
            new = torch.where(choose, best, torch.full_like(best, -1))
            self.current = torch.where(expired, new, self.current)
            # gated 分支 best 是 source id; multi-horizon 下该语义未定义(train
            # 侧已限 bootstrap_only), 单 horizon 下与 current 同值保持同步
            self.current_arm = torch.where(expired, new, self.current_arm)
            self.steps_left = torch.where(
                expired, torch.full_like(self.steps_left, self.min_steps), self.steps_left
            )
        self.steps_left -= 1

        actions = a_student.clone()
        idx = torch.arange(self.num_envs, device=self.device)
        for g in range(self.num_groups):
            sel = self.current[:, g]
            m = sel >= 0
            if m.any():
                picked = src_actions[idx[m], sel[m], :]
                gm = self.group_masks[g].bool()
                actions[m] = torch.where(gm, picked, actions[m])

        any_teacher = (self.current >= 0).any(dim=1)
        info = {
            "mcg/exec_env_frac": float(any_teacher.float().mean()),
            "mcg/exec_part_frac": float((self.current >= 0).float().mean()),
        }
        for i in range(num_src):
            info[f"mcg/exec_share_src{i}"] = float((self.current == i).float().mean())
        if self.warmup_mode == "online_bootstrap" and self.arm_value is not None:
            info["mcg/online_wp"] = max(
                0.0, 1.0 - self._online_step / max(1, self.online_prior_steps)
            )
            if self.online_horizons is not None:
                n_h = int(self.online_horizons.shape[0])
                for a in range(int(self.arm_value.shape[0]) - 1):
                    j, hz = a // n_h, int(self.online_horizons[a % n_h])
                    info[f"mcg/arm_value_src{j}_h{hz}"] = float(self.arm_value[a])
                    info[f"mcg/exec_share_src{j}_h{hz}"] = float(
                        (self.current_arm == a).float().mean()
                    )
            else:
                for i in range(min(num_src, int(self.arm_value.shape[0]) - 1)):
                    info[f"mcg/arm_value_src{i}"] = float(self.arm_value[i])
            info["mcg/arm_value_student"] = float(self.arm_value[-1])
            if self.abstain_gate:
                info["mcg/abstain_mode"] = float(self.abstain_mode)
        if self.warmup_mode == "admission_bootstrap":
            probs = self.admission_probabilities()
            info["mcg/admitted_source_count"] = float(self.admitted_sources.sum())
            info["mcg/admission_exact_abstain"] = float(not bool(self.admitted_sources.any()))
            info["mcg/admission_student_prob"] = float(probs[-1])
            for i in range(num_src):
                info[f"mcg/admission_prob_src{i}"] = float(probs[i])
        return actions, info
