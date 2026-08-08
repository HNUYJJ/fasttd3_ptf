"""PARE —— Provenance-Aware Release and Expansion。

规格见 `docs/PARE_ALGORITHM_SPEC_v1.md`（实现前冻结）。本文件只实现规格里的
四个模块，**不含** selector、threshold scheduler、exit 搜索或任何新 proxy。

核心（spec §6）：post-release 的 actor 更新有两个目标——base RL 的
`g_Q` 与 source-occupancy repulsion 的 `g_E`。二者的合成**没有超参数**：
冲突时把 `g_E` 投影到 `g_Q` 的正交补，再把范数截到 `‖g_Q‖`。由此

    ⟨g_Q, g_PARE⟩ ≥ ‖g_Q‖²                                   (Lemma 1)

即 expansion 永远不会一阶地抵消 base critic 要求的改进方向。
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

#: 仅用于诊断"D 的判别有多饱和"，**不做任何截断**（见 expansion_objective）。
LOGIT_SATURATION_DIAG = 10.0


# ══════════════════════════════════════════════════════════════════════
# 1. 固定 source reservoir
# ══════════════════════════════════════════════════════════════════════
class SourceTransitionReservoir:
    """release 时刻冻结的 source occupancy 样本池 `D_S`。

    存 **raw obs**：replay 本身存 raw、采样后才 normalize
    （`train_ptf.py:3235`），所以 reservoir 也存 raw，用**当前** normalizer
    转换后再喂 discriminator——正负样本必须在同一坐标系里，否则 `D` 学到的
    是 normalizer 漂移而不是 occupancy 差异。

    release 之后 replay 会被新的 student 数据逐步覆盖，source 历史会消失，
    因此必须在 release 当刻复制出来固定住。
    """

    def __init__(self, raw_obs: torch.Tensor, actions: torch.Tensor,
                 *, n_candidates: int) -> None:
        if raw_obs.shape[0] != actions.shape[0]:
            raise ValueError("raw_obs 与 actions 的样本数不一致")
        if raw_obs.shape[0] == 0:
            raise ValueError("source reservoir 为空——release 前没有任何 z=1 transition")
        self.raw_obs = raw_obs
        self.actions = actions
        self.n_candidates = int(n_candidates)

    def __len__(self) -> int:
        return int(self.raw_obs.shape[0])

    @property
    def truncated(self) -> bool:
        return self.n_candidates > len(self)

    @classmethod
    def from_replay(cls, rb, *, capacity: int,
                    generator: torch.Generator | None = None) -> "SourceTransitionReservoir":
        """扫描 replay 中全部 `z=1` 的 transition，超容量时均匀无放回子采样。"""
        raw_obs, actions = rb.source_provenance_samples()
        n_candidates = int(raw_obs.shape[0])
        if n_candidates > capacity:
            # 均匀无放回。截断量必须可见（见 spec §10 D2），调用方负责落盘。
            perm = torch.randperm(n_candidates, device=raw_obs.device,
                                  generator=generator)[:capacity]
            raw_obs = raw_obs[perm]
            actions = actions[perm]
        return cls(raw_obs.contiguous(), actions.contiguous(),
                   n_candidates=n_candidates)

    def sample(self, n: int, *, generator: torch.Generator | None = None):
        idx = torch.randint(0, len(self), (int(n),), device=self.raw_obs.device,
                            generator=generator)
        return self.raw_obs[idx], self.actions[idx]


# ══════════════════════════════════════════════════════════════════════
# 2. provenance discriminator
# ══════════════════════════════════════════════════════════════════════
class SourceOccupancyDiscriminator(nn.Module):
    """`D_φ(s,a)`：判别 `(s,a)` 来自冻结的 source occupancy 还是当前 student。

    先验平衡（正负 batch 等大）时最优解满足

        logit D*(s,a) = log( d_S(s,a) / d_π(s,a) )

    这是标准 density-ratio estimation 的精确结果，不是启发式代理。
    刻意保持小：两层 MLP，**不用** Transformer / VAE / flow / ensemble。
    """

    def __init__(self, obs_dim: int, act_dim: int,
                 hidden: tuple[int, ...] = (256, 256)) -> None:
        super().__init__()
        dims = (obs_dim + act_dim, *hidden)
        layers: list[nn.Module] = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(a, b), nn.ReLU()]
        layers.append(nn.Linear(dims[-1], 1))
        self.net = nn.Sequential(*layers)

    def logit(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([obs, act], dim=-1)).squeeze(-1)

    forward = logit


# ══════════════════════════════════════════════════════════════════════
# 3. 冻结的 competence anchor
# ══════════════════════════════════════════════════════════════════════
class FrozenReleaseActor:
    """`π_B`：release 时刻 actor 的冻结副本，只作 competence reference。

    **共用当前 obs normalizer**（spec §2 决定 D1）。给 π_B 配一个冻结的
    normalizer 会让 π_B 与 π_θ 的输入坐标系不同，使 `Â_B` 的符号混入
    normalizer 漂移而非策略差异。
    """

    def __init__(self, actor: nn.Module) -> None:
        self.module = copy.deepcopy(actor)
        self.module.eval()
        for p in self.module.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def __call__(self, obs: torch.Tensor) -> torch.Tensor:
        return self.module(obs)

    def state_dict(self):
        return self.module.state_dict()


# ══════════════════════════════════════════════════════════════════════
# 4. 梯度合成（spec §6）
# ══════════════════════════════════════════════════════════════════════
def _flat_dot(a: list[torch.Tensor], b: list[torch.Tensor]) -> torch.Tensor:
    return sum((x * y).sum() for x, y in zip(a, b))


def compose_pare_gradient(g_Q: list[torch.Tensor], g_E: list[torch.Tensor],
                          *, eps: float = 1e-12):
    """`g_PARE = g_Q + ḡ_E`，无超参数。

    步骤 1（冲突投影）：若 ``⟨g_Q, g_E⟩ < 0``，把 `g_E` 投影到 `g_Q` 的正交补。
    步骤 2（norm cap）：把结果的范数截到 ``‖g_Q‖``，使 expansion 不会盖过 base RL。

    对 GradScaler 的公共正缩放 `c` **一次齐次**：点积符号不变、投影式齐次、
    norm cap 的比值与 `c` 无关。故传入 scaled 梯度得到的就是 scaled 的正确
    结果，**unscale 一次即可**（spec §6 AMP 齐次性）。
    """
    dot = _flat_dot(g_Q, g_E)
    sq_q = _flat_dot(g_Q, g_Q)
    conflict = bool(dot < 0)
    if conflict:
        coef = dot / sq_q.clamp_min(eps)
        g_E = [e - coef * q for e, q in zip(g_E, g_Q)]

    norm_q = sq_q.clamp_min(0).sqrt()
    norm_e = _flat_dot(g_E, g_E).clamp_min(0).sqrt()
    scale = torch.clamp(norm_q / (norm_e + eps), max=1.0)
    g_E = [e * scale for e in g_E]

    g = [q + e for q, e in zip(g_Q, g_E)]
    ratio = (norm_e * scale / (norm_q + eps)).detach()
    return g, conflict, ratio


class PARERuntime:
    """把 release 后的 discriminator 更新与 actor 梯度合成串起来。

    `enabled=False` 时训练循环根本不构造本对象，代码路径与既有 FastTD3
    hard-exit 逐位一致（spec §12 smoke 第 4 项验此）。
    """

    def __init__(self, *, actor: nn.Module, obs_dim: int, act_dim: int,
                 reservoir: SourceTransitionReservoir, device, d_lr: float = 3e-4,
                 d_hidden: tuple[int, ...] = (256, 256)) -> None:
        self.pi_B = FrozenReleaseActor(actor)
        self.reservoir = reservoir
        self.D = SourceOccupancyDiscriminator(obs_dim, act_dim, d_hidden).to(device)
        self.opt_D = torch.optim.Adam(self.D.parameters(), lr=d_lr)
        self.d_skip_count = 0

    # ── discriminator ────────────────────────────────────────────────
    def update_discriminator(self, student_raw_obs: torch.Tensor,
                             student_actions: torch.Tensor,
                             normalize_obs) -> dict:
        """balanced BCE。student negatives **必须**已过滤掉残留的 z=1。

        负样本为空时跳过本步并计数——不静默把 source 数据当 student 用，
        那会让 `D` 学不出任何东西（同一批数据既作正例又作负例）。
        """
        n = int(student_raw_obs.shape[0])
        if n == 0:
            self.d_skip_count += 1
            return {"pare/d_skipped": 1.0}

        pos_raw, pos_act = self.reservoir.sample(n)
        with torch.no_grad():
            pos_obs = normalize_obs(pos_raw)
            neg_obs = normalize_obs(student_raw_obs)

        pos_logit = self.D.logit(pos_obs, pos_act)
        neg_logit = self.D.logit(neg_obs, student_actions)
        loss = (F.binary_cross_entropy_with_logits(pos_logit, torch.ones_like(pos_logit))
                + F.binary_cross_entropy_with_logits(neg_logit, torch.zeros_like(neg_logit)))

        self.opt_D.zero_grad(set_to_none=True)
        loss.backward()
        self.opt_D.step()
        with torch.no_grad():
            acc = ((pos_logit > 0).float().mean() + (neg_logit <= 0).float().mean()) / 2
        return {"pare/d_loss": loss.detach(), "pare/d_acc": acc,
                "pare/d_skipped": 0.0}

    # ── expansion objective ──────────────────────────────────────────
    def expansion_objective(self, obs: torch.Tensor, critic_obs: torch.Tensor,
                            pi_action: torch.Tensor,
                            q_low_fn) -> tuple[torch.Tensor, dict]:
        """`J_E = E[ m(s) · log(1 - D(s, π_θ(s))) ]`，返回 **ascent** 目标。

        `m(s) = 1[Q_L(s,π_θ) ≥ Q_L(s,π_B)]` 是 competence gate 且 stopgrad：
        只有当前 target critic 认为 π_θ 不比 release policy 差时，
        expansion 才被允许影响该状态上的 actor。

        `obs` 是 actor 观测（π_B 与 D 都吃它），`critic_obs` 是 critic 观测——
        asymmetric_obs 下两者不同，混用会让 Q 读到错的输入。
        """
        with torch.no_grad():
            q_theta = q_low_fn(critic_obs, pi_action.detach())
            q_anchor = q_low_fn(critic_obs, self.pi_B(obs))
            m = (q_theta >= q_anchor).float()

        # D 在 actor 更新中不接收梯度。
        for p in self.D.parameters():
            p.requires_grad_(False)
        logit = self.D.logit(obs, pi_action)
        for p in self.D.parameters():
            p.requires_grad_(True)

        # log(1 - sigmoid(z)) = -softplus(z)。**不得对 z 做 clamp**：
        # d/dz[-softplus(z)] = -sigmoid(z)，在 z→+∞（样本最像 source，
        # 正是最该推离的地方）时趋于 -1，在 z→-∞（已经不像 source）时趋于 0。
        # 梯度本就有界且方向正确，softplus 自身数值稳定。
        # 首版加了 clamp(±10)，smoke 实测 72% 的样本落在截断区——梯度被置零，
        # 恰好把最需要 expansion 的样本全部静音（与 [[project_beta_clamp_fix]]
        # 的 logit 死区同类：护住值域、杀死梯度）。
        log_one_minus_d = -F.softplus(logit)
        j_e = (m * log_one_minus_d).mean()

        with torch.no_grad():
            metrics = {
                "pare/source_affinity": torch.sigmoid(logit).mean(),
                "pare/anchor_adv_positive_frac": m.mean(),
                "pare/anchor_adv_mean": (q_theta - q_anchor).mean(),
                # 纯诊断：D 的判别有多饱和。不再参与任何截断。
                "pare/d_logit_saturated_rate": (logit.abs() > LOGIT_SATURATION_DIAG)
                .float().mean(),
            }
        return j_e, metrics


def apply_pare_actor_gradient(actor: nn.Module, j_q: torch.Tensor,
                              j_e: torch.Tensor, scaler) -> dict:
    """求两路梯度、合成、写回 ``p.grad``。

    `j_q` / `j_e` 都是 **ascent** 目标；optimizer 做 descent，故写回 ``-g``。
    两路用同一个 GradScaler 因子，合成对该因子齐次（见 `compose_pare_gradient`）。
    """
    params = [p for p in actor.parameters() if p.requires_grad]
    g_Q = list(torch.autograd.grad(scaler.scale(j_q), params, retain_graph=True))
    g_E = list(torch.autograd.grad(scaler.scale(j_e), params))

    # fail-closed：非有限梯度一旦写进 p.grad 就会静默污染 actor，
    # 而后续的 GradScaler 只会跳过该步、不会报告原因。
    base_norm = _flat_dot(g_Q, g_Q).clamp_min(0).sqrt()
    finite_or_raise("g_Q", base_norm)
    finite_or_raise("g_E", _flat_dot(g_E, g_E))

    g, conflict, ratio = compose_pare_gradient(g_Q, g_E)
    for p, gi in zip(params, g):
        p.grad = -gi

    return {
        "pare/grad_conflict": torch.tensor(float(conflict), device=ratio.device),
        "pare/expansion_norm_ratio": ratio,
        # scaled 范数——只用于有限性判据与相对比较，不作绝对量解读。
        "pare/base_grad_norm_scaled": base_norm.detach(),
    }


def finite_or_raise(name: str, value: torch.Tensor) -> None:
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError(f"PARE: {name} 出现非有限值")


__all__ = [
    "SourceTransitionReservoir",
    "SourceOccupancyDiscriminator",
    "FrozenReleaseActor",
    "PARERuntime",
    "compose_pare_gradient",
    "apply_pare_actor_gradient",
    "finite_or_raise",
    "LOGIT_SATURATION_DIAG",
]
