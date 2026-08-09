from __future__ import annotations

import torch
from tensordict import TensorDict


class PTFReplayWrapper:
    """Attach option ids to the official FastTD3 replay buffer.

    The wrapped replay buffer is left untouched. This class mirrors the upstream
    n=1 sampling branch so the option id is gathered from the same sampled index.

    num_steps is locked to 1 for a semantic reason, not just an implementation
    one: PTF's option-value Q_o and termination β are defined per single
    transition (the option selected at s, the termination decision at s'). An
    n-step return would span several transitions, each potentially collected
    under a different option, so the bootstrap target would mix option credits
    and the per-step termination semantics would break. HumanoidBench FastTD3
    uses num_steps=1 by default, which is consistent with this; lifting the lock
    would require redesigning option target semantics explicitly.
    """

    def __init__(self, replay_buffer):
        if int(getattr(replay_buffer, "n_steps", 1)) != 1:
            raise NotImplementedError(
                "Official PTF replay supports num_steps=1 only: PTF's Q_o/β are "
                "per-transition quantities, and an n-step return would span "
                "multiple options and break per-step termination semantics. "
                "HumanoidBench FastTD3 defaults to num_steps=1."
            )
        self.base = replay_buffer
        self.options = torch.full(
            (replay_buffer.n_env, replay_buffer.buffer_size),
            -1,
            device=replay_buffer.device,
            dtype=torch.long,
        )
        # replay 重加权(Step B, 导师意见1 2026-07-02): per-source 采样权重。
        # None=uniform(原行为)。非 None 时 sample() 按 transition 的 option id
        # 加权抽样: 学生轨迹(option=-1)恒权重 1; 源 i 的驻留轨迹按权重降权
        # (只降不升,下限>0 保留其负样本价值)。
        # actor/critic 路径消融接口。2026-07-02 的最终归因结果否决了“actor强降、
        # critic保守”的split假设：crawl中actor-only/critic-only/split均劣于both，
        # 表明两条更新路径看到不同source-state distribution时会产生AC mismatch。
        # 主方法使用role="both"；独立role只保留用于因果消融。
        self._src_w_critic: torch.Tensor | None = None
        self._src_w_actor: torch.Tensor | None = None
        # Admission-consistent replay policy. Candidate order is sources then
        # student. Rejected sources receive exact zero mass; student mass must
        # remain positive.  Slot quality is normalized *within* provenance
        # strata, so source/student quotas do not depend on historical counts.
        self._admission_source_mask: torch.Tensor | None = None
        self._admission_candidate_masses: torch.Tensor | None = None
        self._admission_recency_half_life = 0.0
        self._admission_uniform_mix = 0.05
        self._admission_priority_alpha = 0.0
        self._replay_priorities: torch.Tensor | None = None
        self._slot_write_step: torch.Tensor | None = None
        self._admission_sample_counts: dict[str, torch.Tensor] = {}
        self._admission_policy_events: list[dict] = []
        # A fixed provenance quota is meaningful while a source still has
        # behavior authority.  Once that authority ends, retaining the quota
        # would repeatedly oversample the shrinking tail of old source data.
        # The training loop may therefore hand replay back to physical-uniform
        # sampling without deleting source history or weakening exact revoke.
        self._admission_source_authority_active = True
        # T4-R：把 replay authority 从 behavior authority 里解耦出来。
        # `_admission_source_authority_active` 原本一个 flag 同时决定
        # "source 是否还在执行动作"与"replay 用不用 provenance quota"。
        # 置本 flag 后，source 照常拥有 behavior authority，但 replay 始终
        # physical-uniform（q_S = rho_S），用来检验入口侧 replay amplification。
        self._admission_replay_physical = False
        self._provenance: dict[str, torch.Tensor] = {}
        self._provenance_written: torch.Tensor | None = None
        self._provenance_group_count: int | None = None

    _PROVENANCE_DTYPES = {
        "behavior_source": torch.int16,
        "source_by_group": torch.int16,
        "executed_group_mask": torch.bool,
        "segment_id": torch.int64,
        "segment_step": torch.int16,
        "anchor_id": torch.int32,
        "env_rank": torch.int16,
        "learner_step": torch.int64,
    }

    def enable_provenance(self, group_count: int) -> None:
        """Allocate exact behavior provenance for future replay writes.

        This is opt-in so ordinary training does not pay the memory cost. A
        source-intervention collector must supply all fields on every write and
        call :meth:`assert_complete_provenance` before exporting paper data.
        ``behavior_source=-1`` and ``source_by_group=-1`` denote student action;
        missingness is represented separately by ``provenance_written``.
        """

        group_count = int(group_count)
        if group_count <= 0:
            raise ValueError("group_count must be positive")
        if self._provenance_group_count not in (None, group_count):
            raise ValueError(
                f"provenance already configured for {self._provenance_group_count} groups"
            )
        if self._provenance:
            return
        rb = self.base
        scalar_shape = (rb.n_env, rb.buffer_size)
        group_shape = (*scalar_shape, group_count)
        self._provenance = {
            "behavior_source": torch.full(
                scalar_shape, -1, device=rb.device, dtype=torch.int16
            ),
            "source_by_group": torch.full(
                group_shape, -1, device=rb.device, dtype=torch.int16
            ),
            "executed_group_mask": torch.zeros(
                group_shape, device=rb.device, dtype=torch.bool
            ),
            "segment_id": torch.full(
                scalar_shape, -1, device=rb.device, dtype=torch.int64
            ),
            "segment_step": torch.full(
                scalar_shape, -1, device=rb.device, dtype=torch.int16
            ),
            "anchor_id": torch.full(
                scalar_shape, -1, device=rb.device, dtype=torch.int32
            ),
            "env_rank": torch.full(
                scalar_shape, -1, device=rb.device, dtype=torch.int16
            ),
            "learner_step": torch.full(
                scalar_shape, -1, device=rb.device, dtype=torch.int64
            ),
        }
        self._provenance_written = torch.zeros(
            scalar_shape, device=rb.device, dtype=torch.bool
        )
        self._provenance_group_count = group_count

    def set_source_weights(
        self,
        weights: torch.Tensor | None,
        role: str = "both",
    ) -> None:
        """设置 per-source 采样权重([S], (0,1]); None 恢复该路径 uniform。

        role: "both"(critic+actor 同权重, 兼容 obrw) / "critic" / "actor"。
        """
        w = None if weights is None else weights.detach().to(
            device=self.base.device, dtype=torch.float32
        ).clamp(1e-3, 1.0)
        if role in ("both", "critic"):
            self._src_w_critic = w
        if role in ("both", "actor"):
            self._src_w_actor = w
        if role not in ("both", "critic", "actor"):
            raise ValueError(f"unknown role: {role}")

    @torch.no_grad()
    def set_admission_policy(
        self,
        *,
        admitted_sources: torch.Tensor,
        candidate_masses: torch.Tensor,
        recency_half_life: float = 0.0,
        uniform_mix: float = 0.05,
        priority_alpha: float = 0.0,
    ) -> None:
        """Install one shared actor/critic provenance-stratified policy.

        ``candidate_masses`` has shape ``[num_sources + 1]`` with student in
        the last position.  Source revocation is exact: rejected source mass is
        forced to zero, so its historical transitions immediately leave active
        replay without deleting the audit record.
        """

        admitted = admitted_sources.to(self.base.device, dtype=torch.bool).view(-1)
        masses = candidate_masses.to(self.base.device, dtype=torch.float32).view(-1)
        if masses.numel() != admitted.numel() + 1:
            raise ValueError("candidate_masses must contain sources plus student")
        if bool((masses < 0).any()) or not bool(torch.isfinite(masses).all()):
            raise ValueError("candidate masses must be finite and non-negative")
        if float(masses[-1]) <= 0:
            raise ValueError("student replay mass must be positive")
        if not 0.0 <= float(uniform_mix) <= 1.0:
            raise ValueError("uniform_mix must lie in [0, 1]")
        if float(recency_half_life) < 0 or float(priority_alpha) < 0:
            raise ValueError("recency_half_life and priority_alpha must be non-negative")
        masses = masses.clone()
        masses[:-1] = torch.where(admitted, masses[:-1], torch.zeros_like(masses[:-1]))
        masses /= masses.sum()
        self._admission_source_mask = admitted
        self._admission_candidate_masses = masses
        self._admission_recency_half_life = float(recency_half_life)
        self._admission_uniform_mix = float(uniform_mix)
        self._admission_priority_alpha = float(priority_alpha)
        count_shape = admitted.numel() + 1
        if any(value.numel() != count_shape for value in self._admission_sample_counts.values()):
            self._admission_sample_counts = {}
        if not self._admission_sample_counts:
            self._admission_sample_counts = {
                "critic": torch.zeros(count_shape, device=self.base.device, dtype=torch.int64),
                "actor": torch.zeros(count_shape, device=self.base.device, dtype=torch.int64),
            }
        self._admission_policy_events.append(
            {
                "event": "admission_policy",
                "replay_ptr": int(self.base.ptr),
                "source_authority_active": self._admission_source_authority_active,
                "admitted_sources": admitted.detach().cpu().tolist(),
                "candidate_masses": masses.detach().cpu().tolist(),
                "sample_counts_at_apply": {
                    role: values.detach().cpu().tolist()
                    for role, values in self._admission_sample_counts.items()
                },
            }
        )
        if self._replay_priorities is None:
            shape = (self.base.n_env, self.base.buffer_size)
            self._replay_priorities = torch.ones(
                shape, device=self.base.device, dtype=torch.float32
            )
            self._slot_write_step = torch.full(
                shape, -1, device=self.base.device, dtype=torch.int64
            )

    @property
    def admission_source_authority_active(self) -> bool:
        return bool(self._admission_source_authority_active)

    @torch.no_grad()
    def set_admission_source_authority(
        self,
        active: bool,
        *,
        reason: str | None = None,
    ) -> None:
        """Switch admission replay between authority quota and physical handoff.

        ``active=True`` keeps the installed source/student provenance quotas.
        ``active=False`` samples uniformly over physically retained *allowed*
        slots.  Rejected-source slots remain exactly excluded in both phases.
        """

        if self._admission_candidate_masses is None:
            raise RuntimeError("cannot change source authority without admission policy")
        active = bool(active)
        if active == self._admission_source_authority_active:
            return
        self._admission_source_authority_active = active
        assert self._admission_source_mask is not None
        self._admission_policy_events.append(
            {
                "event": "source_authority",
                "replay_ptr": int(self.base.ptr),
                "source_authority_active": active,
                "reason": reason,
                "admitted_sources": self._admission_source_mask.detach().cpu().tolist(),
                "candidate_masses": self._admission_candidate_masses.detach().cpu().tolist(),
                "sample_counts_at_apply": {
                    role: values.detach().cpu().tolist()
                    for role, values in self._admission_sample_counts.items()
                },
            }
        )

    @torch.no_grad()
    def set_admission_replay_physical(self, physical: bool) -> None:
        """Sample replay physically-uniformly while source keeps behavior authority.

        This decouples *replay* authority from *behavior* authority.  With a fixed
        provenance quota the source draws ``q_S = m`` regardless of how much of the
        buffer it actually produced, so a late-entering source is over-sampled by
        ``A = 1 + H/((1-m)u)``.  Setting this flag makes ``q_S`` track ``rho_S``.
        Rejected-source slots stay exactly excluded, as in the retirement path.
        """

        self._admission_replay_physical = bool(physical)

    @property
    def admission_replay_physical(self) -> bool:
        return bool(self._admission_replay_physical)

    @torch.no_grad()
    def clear_admission_policy(self) -> None:
        self._admission_source_mask = None
        self._admission_candidate_masses = None
        self._admission_source_authority_active = True
        self._admission_replay_physical = False

    @torch.no_grad()
    def update_priorities(self, indices: torch.Tensor, values: torch.Tensor) -> None:
        if self._replay_priorities is None:
            return
        indices = torch.as_tensor(indices, device=self.base.device, dtype=torch.long)
        if indices.ndim == 1:
            if indices.numel() % self.base.n_env:
                raise ValueError("flat priority indices do not divide by n_env")
            indices = indices.view(self.base.n_env, -1)
        values = torch.as_tensor(values, device=self.base.device, dtype=torch.float32)
        if values.ndim == 1:
            values = values.view_as(indices)
        if values.shape != indices.shape:
            raise ValueError("priority values must match replay index shape")
        values = values.abs().clamp_min(1e-6)
        self._replay_priorities.scatter_(1, indices, values)

    @property
    def ptr(self) -> int:
        return self.base.ptr

    @property
    def valid_size(self) -> int:
        return min(int(self.base.buffer_size), int(self.base.ptr))

    def chronological_slot_indices(
        self,
        *,
        ptr: int | None = None,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        """Return physical replay slots ordered from oldest to newest."""

        ptr = int(self.base.ptr if ptr is None else ptr)
        capacity = int(self.base.buffer_size)
        valid = min(capacity, ptr)
        out_device = self.base.device if device is None else device
        if valid == 0:
            return torch.empty(0, device=out_device, dtype=torch.long)
        if ptr < capacity:
            return torch.arange(valid, device=out_device, dtype=torch.long)
        start = ptr % capacity
        return torch.cat(
            (
                torch.arange(start, capacity, device=out_device, dtype=torch.long),
                torch.arange(0, start, device=out_device, dtype=torch.long),
            )
        )

    @torch.no_grad()
    def extend(
        self,
        tensor_dict: TensorDict,
        option_ids: torch.Tensor,
        provenance: dict[str, torch.Tensor | int] | None = None,
    ) -> None:
        ptr = self.base.ptr % self.base.buffer_size
        self.options[:, ptr] = option_ids.to(
            device=self.base.device,
            dtype=torch.long,
        ).view(self.base.n_env)
        if provenance is not None and not self._provenance:
            source_by_group = torch.as_tensor(provenance.get("source_by_group"))
            if source_by_group.ndim < 2:
                raise ValueError("source_by_group must have shape [n_env, n_groups]")
            self.enable_provenance(int(source_by_group.shape[-1]))
        if self._provenance:
            assert self._provenance_written is not None
            self._provenance_written[:, ptr] = False
            if provenance is not None:
                missing = set(self._PROVENANCE_DTYPES) - set(provenance)
                if missing:
                    raise ValueError(f"missing provenance fields: {sorted(missing)}")
                for name, storage in self._provenance.items():
                    value = torch.as_tensor(
                        provenance[name],
                        device=self.base.device,
                        dtype=self._PROVENANCE_DTYPES[name],
                    )
                    expected = storage[:, ptr].shape
                    if value.ndim == 0:
                        value = value.expand(expected)
                    if tuple(value.shape) != tuple(expected):
                        raise ValueError(
                            f"{name} has shape {tuple(value.shape)}, expected {tuple(expected)}"
                        )
                    storage[:, ptr].copy_(value)
                self._provenance_written[:, ptr] = True
        if self._replay_priorities is not None:
            assert self._slot_write_step is not None
            self._replay_priorities[:, ptr] = 1.0
            self._slot_write_step[:, ptr] = int(self.base.ptr)
        self.base.extend(tensor_dict)

    def _admission_allowed_slots(
        self, options: torch.Tensor, valid_n: int
    ) -> torch.Tensor:
        assert self._admission_source_mask is not None
        num_sources = int(self._admission_source_mask.numel())
        allowed = torch.ones_like(options, dtype=torch.bool)
        if self._provenance and self._provenance_written is not None:
            source_by_group = self._provenance["source_by_group"][:, :valid_n]
            present = source_by_group >= 0
            source_index = source_by_group.clamp_min(0).long()
            if bool((source_index[present] >= num_sources).any()):
                raise ValueError("replay group provenance contains an unknown source")
            group_allowed = torch.ones_like(present)
            group_allowed[present] = self._admission_source_mask[source_index[present]]
            allowed &= group_allowed.all(dim=-1)
            allowed &= self._provenance_written[:, :valid_n]
        else:
            source_slot = options >= 0
            allowed[source_slot] = self._admission_source_mask[options[source_slot]]
        return allowed

    def _admission_slot_weights(self, valid_n: int) -> torch.Tensor:
        # admission-consistent replay 的配额分配核心(authority 活跃期使用):
        # 1) 按 stratum(每个 admitted 源 + student)分配 candidate mass,
        #    只在"当前 buffer 里确实还有该源数据"的 stratum 间归一
        #    —— `masses * available` 只查非空曾是 repetition divergence
        #    (80k 崩点,oversample 43x)的根源:源物理残留 1.2% 仍拿 50% 配额;
        #    修复 = authority 结束后走 physical_after_authority 分支(见
        #    draw_indices),不再进入本函数;
        # 2) stratum 内按 recency 半衰 + 可选 priority 加权,再混 uniform floor。
        assert self._admission_candidate_masses is not None
        assert self._admission_source_mask is not None
        assert self._replay_priorities is not None
        assert self._slot_write_step is not None
        num_sources = int(self._admission_source_mask.numel())
        student_stratum = num_sources
        options = self.options[:, :valid_n]
        if bool(((options >= num_sources) | (options < -1)).any()):
            raise ValueError("replay option id is incompatible with admission source strata")
        strata = torch.where(options >= 0, options, torch.full_like(options, student_stratum))
        allowed = self._admission_allowed_slots(options, valid_n)
        one = allowed.float()
        counts = torch.zeros(
            self.base.n_env,
            num_sources + 1,
            device=self.base.device,
            dtype=torch.float32,
        ).scatter_add_(1, strata, one)
        available = counts > 0
        masses = self._admission_candidate_masses.view(1, -1) * available
        mass_sum = masses.sum(dim=1, keepdim=True)
        if bool((mass_sum <= 0).any()):
            raise RuntimeError("no active replay stratum is available for at least one environment")
        masses = masses / mass_sum

        quality = one.clone()
        if self._admission_priority_alpha > 0:
            quality *= self._replay_priorities[:, :valid_n].clamp_min(1e-6).pow(
                self._admission_priority_alpha
            )
        if self._admission_recency_half_life > 0:
            ages = (
                int(self.base.ptr) - 1 - self._slot_write_step[:, :valid_n]
            ).clamp_min(0).float()
            quality *= torch.exp2(-ages / self._admission_recency_half_life)
        quality_sums = torch.zeros_like(counts).scatter_add_(1, strata, quality)
        stratum_quality = torch.gather(quality_sums, 1, strata).clamp_min(1e-12)
        stratum_count = torch.gather(counts, 1, strata).clamp_min(1.0)
        within = (
            (1.0 - self._admission_uniform_mix) * quality / stratum_quality
            + self._admission_uniform_mix * one / stratum_count
        )
        slot_masses = torch.gather(masses, 1, strata)
        return slot_masses * within

    def _record_admission_samples(
        self,
        role: str,
        indices: torch.Tensor,
    ) -> None:
        assert self._admission_source_mask is not None
        student = int(self._admission_source_mask.numel())
        sampled_options = torch.gather(self.options, 1, indices)
        sampled_strata = torch.where(
            sampled_options >= 0,
            sampled_options,
            torch.full_like(sampled_options, student),
        )
        self._admission_sample_counts[role] += torch.bincount(
            sampled_strata.reshape(-1), minlength=student + 1
        )

    @torch.no_grad()
    def draw_indices(self, batch_size: int, role: str = "critic") -> torch.Tensor:
        rb = self.base
        valid_n = self.valid_size
        if valid_n <= 0:
            raise RuntimeError("cannot sample an empty replay buffer")
        if role not in ("critic", "actor"):
            raise ValueError(f"unknown replay sampling role: {role}")
        if self._admission_candidate_masses is not None:
            # Startup exact abstention with an all-student history is not merely
            # distribution-equivalent to uniform scratch: use the same randint
            # primitive so the learner's global RNG stream also stays identical.
            if (
                self._admission_source_mask is not None
                and not bool(self._admission_source_mask.any())
                and not bool((self.options[:, :valid_n] >= 0).any())
            ):
                indices = torch.randint(
                    0,
                    valid_n,
                    (rb.n_env, batch_size),
                    device=rb.device,
                )
                student = int(self._admission_source_mask.numel())
                self._admission_sample_counts[role][student] += indices.numel()
                return indices
            # authority-coupled physical handoff(80k repetition divergence 修复):
            # source behavior authority 结束后,配额从 admission mass 切回
            # "allowed 槽位的物理占比"——无 rejected 数据时与 legacy randint
            # 逐位一致(含 RNG 消耗),有 rejected 时用 masked multinomial 排除。
            if not self._admission_source_authority_active or self._admission_replay_physical:
                assert self._admission_source_mask is not None
                options = self.options[:, :valid_n]
                allowed = self._admission_allowed_slots(options, valid_n)
                if bool((allowed.sum(dim=1) <= 0).any()):
                    raise RuntimeError(
                        "no active physical replay slot is available for at least one environment"
                    )
                if bool(allowed.all()):
                    # Exact legacy handoff: same primitive and RNG consumption
                    # as ordinary FastTD3 physical-uniform replay.
                    indices = torch.randint(
                        0,
                        valid_n,
                        (rb.n_env, batch_size),
                        device=rb.device,
                    )
                else:
                    indices = torch.multinomial(
                        allowed.float(), batch_size, replacement=True
                    )
                self._record_admission_samples(role, indices)
                return indices
            indices = torch.multinomial(
                self._admission_slot_weights(valid_n), batch_size, replacement=True
            )
            self._record_admission_samples(role, indices)
            return indices
        src_w = self._src_w_actor if role == "actor" else self._src_w_critic
        if src_w is None:
            return torch.randint(
                0,
                valid_n,
                (rb.n_env, batch_size),
                device=rb.device,
            )
        # 按源加权采样: w[slot] = 1(学生) 或 src_w[option](教师驻留轨迹)。
        # multinomial 每行(env)独立归一,语义 = 原 uniform 的重要性倾斜版。
        opt = self.options[:, :valid_n]
        w = torch.ones_like(opt, dtype=torch.float32)
        m = opt >= 0
        if m.any():
            if src_w.numel() == 0:
                raise ValueError("source weights are empty but source options are present")
            w[m] = src_w[opt[m].clamp_max(src_w.shape[0] - 1)]
        return torch.multinomial(w, batch_size, replacement=True)

    @torch.no_grad()
    def admission_audit(self) -> dict | None:
        """Return compact source-lifecycle evidence for checkpoints.

        Counts are ordered as ``sources + student``.  Buffer counts describe
        the currently retained main replay, while sample counts are cumulative
        since the admission policy was installed.
        """

        if self._admission_candidate_masses is None:
            return None
        assert self._admission_source_mask is not None
        num_sources = int(self._admission_source_mask.numel())
        valid_n = self.valid_size
        if valid_n:
            options = self.options[:, :valid_n]
            strata = torch.where(
                options >= 0,
                options,
                torch.full_like(options, num_sources),
            )
            buffer_counts = torch.bincount(
                strata.reshape(-1), minlength=num_sources + 1
            )
        else:
            buffer_counts = torch.zeros(
                num_sources + 1, device=self.base.device, dtype=torch.int64
            )
        if valid_n:
            allowed = self._admission_allowed_slots(options, valid_n)
            active_buffer_counts = torch.bincount(
                strata[allowed].reshape(-1), minlength=num_sources + 1
            )
        else:
            active_buffer_counts = buffer_counts.clone()
        available = active_buffer_counts > 0
        if self._admission_source_authority_active:
            effective_masses = self._admission_candidate_masses * available
        else:
            effective_masses = active_buffer_counts.float()
        if float(effective_masses.sum()) > 0:
            effective_masses = effective_masses / effective_masses.sum()
        return {
            "candidate_order": [
                *[f"source_{idx}" for idx in range(num_sources)],
                "student",
            ],
            "admitted_sources": self._admission_source_mask.detach().cpu().tolist(),
            "candidate_masses": self._admission_candidate_masses.detach().cpu().tolist(),
            "source_authority_active": self._admission_source_authority_active,
            "sampling_phase": (
                "authority_quota"
                if self._admission_source_authority_active
                else "physical_allowed"
            ),
            "main_buffer_counts": buffer_counts.detach().cpu().tolist(),
            "active_buffer_counts": active_buffer_counts.detach().cpu().tolist(),
            "effective_replay_masses": effective_masses.detach().cpu().tolist(),
            "critic_sample_counts": self._admission_sample_counts["critic"]
            .detach()
            .cpu()
            .tolist(),
            "actor_independent_sample_counts": self._admission_sample_counts["actor"]
            .detach()
            .cpu()
            .tolist(),
            "policy_events": list(self._admission_policy_events),
        }

    @torch.no_grad()
    def gather(self, indices: torch.Tensor) -> TensorDict:
        """Gather an explicit per-env index matrix for deterministic branches."""

        rb = self.base
        indices = torch.as_tensor(indices, device=rb.device, dtype=torch.long)
        if indices.ndim != 2 or indices.shape[0] != rb.n_env:
            raise ValueError(
                f"indices must have shape [n_env, batch], got {tuple(indices.shape)}"
            )
        upper = rb.buffer_size if rb.ptr >= rb.buffer_size else rb.ptr
        if upper <= 0 or bool(((indices < 0) | (indices >= upper)).any()):
            raise IndexError(f"indices must address valid physical slots [0, {upper})")
        batch_size = int(indices.shape[1])
        obs_indices = indices.unsqueeze(-1).expand(-1, -1, rb.n_obs)
        act_indices = indices.unsqueeze(-1).expand(-1, -1, rb.n_act)
        observations = torch.gather(rb.observations, 1, obs_indices).reshape(
            rb.n_env * batch_size,
            rb.n_obs,
        )
        next_observations = torch.gather(
            rb.next_observations,
            1,
            obs_indices,
        ).reshape(rb.n_env * batch_size, rb.n_obs)
        actions = torch.gather(rb.actions, 1, act_indices).reshape(
            rb.n_env * batch_size,
            rb.n_act,
        )
        rewards = torch.gather(rb.rewards, 1, indices).reshape(rb.n_env * batch_size)
        dones = torch.gather(rb.dones, 1, indices).reshape(rb.n_env * batch_size)
        truncations = torch.gather(rb.truncations, 1, indices).reshape(
            rb.n_env * batch_size
        )
        options = torch.gather(self.options, 1, indices).reshape(
            rb.n_env * batch_size
        )
        effective_n_steps = torch.ones_like(dones)

        out = TensorDict(
            {
                "observations": observations,
                "raw_observations": observations.clone(),
                "actions": actions,
                "options": options,
                "replay_indices": indices.reshape(rb.n_env * batch_size),
                "next": {
                    "rewards": rewards,
                    "dones": dones,
                    "truncations": truncations,
                    "observations": next_observations,
                    "raw_observations": next_observations.clone(),
                    "effective_n_steps": effective_n_steps,
                },
            },
            batch_size=rb.n_env * batch_size,
        )
        if rb.asymmetric_obs:
            if rb.playground_mode:
                priv_obs_indices = indices.unsqueeze(-1).expand(
                    -1,
                    -1,
                    rb.privileged_obs_size,
                )
                privileged_observations = torch.gather(
                    rb.privileged_observations,
                    1,
                    priv_obs_indices,
                ).reshape(rb.n_env * batch_size, rb.privileged_obs_size)
                next_privileged_observations = torch.gather(
                    rb.next_privileged_observations,
                    1,
                    priv_obs_indices,
                ).reshape(rb.n_env * batch_size, rb.privileged_obs_size)
                critic_observations = torch.cat(
                    [observations, privileged_observations],
                    dim=1,
                )
                next_critic_observations = torch.cat(
                    [next_observations, next_privileged_observations],
                    dim=1,
                )
            else:
                critic_obs_indices = indices.unsqueeze(-1).expand(
                    -1,
                    -1,
                    rb.n_critic_obs,
                )
                critic_observations = torch.gather(
                    rb.critic_observations,
                    1,
                    critic_obs_indices,
                ).reshape(rb.n_env * batch_size, rb.n_critic_obs)
                next_critic_observations = torch.gather(
                    rb.next_critic_observations,
                    1,
                    critic_obs_indices,
                ).reshape(rb.n_env * batch_size, rb.n_critic_obs)
            out["critic_observations"] = critic_observations
            out["next"]["critic_observations"] = next_critic_observations
        if self._provenance:
            assert self._provenance_written is not None
            written = torch.gather(self._provenance_written, 1, indices).reshape(
                rb.n_env * batch_size
            )
            out["provenance_written"] = written
            for name, storage in self._provenance.items():
                if storage.ndim == 2:
                    value = torch.gather(storage, 1, indices).reshape(
                        rb.n_env * batch_size
                    )
                else:
                    trailing = storage.shape[2:]
                    gather_indices = indices.view(rb.n_env, batch_size, *([1] * len(trailing)))
                    gather_indices = gather_indices.expand(rb.n_env, batch_size, *trailing)
                    value = torch.gather(storage, 1, gather_indices).reshape(
                        rb.n_env * batch_size, *trailing
                    )
                out[name] = value
        return out

    @torch.no_grad()
    def sample(self, batch_size: int, role: str = "critic") -> TensorDict:
        return self.gather(self.draw_indices(batch_size, role=role))

    def _base_tensor_names(self) -> list[str]:
        names = [
            "observations",
            "actions",
            "rewards",
            "dones",
            "truncations",
            "next_observations",
        ]
        if self.base.asymmetric_obs:
            if self.base.playground_mode:
                names.extend(["privileged_observations", "next_privileged_observations"])
            else:
                names.extend(["critic_observations", "next_critic_observations"])
        return names

    @torch.no_grad()
    def export_valid(self, *, require_complete_provenance: bool = False) -> dict:
        """Export the compact valid replay slice in chronological order on CPU."""

        if require_complete_provenance:
            self.assert_complete_provenance()
        order = self.chronological_slot_indices()
        tensors = {
            name: getattr(self.base, name).index_select(1, order).detach().cpu().clone()
            for name in self._base_tensor_names()
        }
        tensors["options"] = self.options.index_select(1, order).detach().cpu().clone()
        provenance = None
        if self._provenance:
            assert self._provenance_written is not None
            provenance = {
                name: value.index_select(1, order).detach().cpu().clone()
                for name, value in self._provenance.items()
            }
            provenance["provenance_written"] = (
                self._provenance_written.index_select(1, order).detach().cpu().clone()
            )
        return {
            "schema_version": 1,
            "storage_order": "chronological_oldest_to_newest",
            "metadata": {
                "n_env": int(self.base.n_env),
                "buffer_size": int(self.base.buffer_size),
                "n_obs": int(self.base.n_obs),
                "n_act": int(self.base.n_act),
                "n_critic_obs": int(self.base.n_critic_obs),
                "asymmetric_obs": bool(self.base.asymmetric_obs),
                "playground_mode": bool(self.base.playground_mode),
                "n_steps": int(self.base.n_steps),
                "gamma": float(self.base.gamma),
                "ptr": int(self.base.ptr),
                "valid_size": self.valid_size,
                "provenance_group_count": self._provenance_group_count,
            },
            "tensors": tensors,
            "provenance": provenance,
            "source_weights": {
                "critic": None
                if self._src_w_critic is None
                else self._src_w_critic.detach().cpu().clone(),
                "actor": None
                if self._src_w_actor is None
                else self._src_w_actor.detach().cpu().clone(),
            },
            "admission_sampling": None
            if self._admission_candidate_masses is None
            else {
                "source_mask": self._admission_source_mask.detach().cpu().clone(),
                "candidate_masses": self._admission_candidate_masses.detach().cpu().clone(),
                "source_authority_active": self._admission_source_authority_active,
                "recency_half_life": self._admission_recency_half_life,
                "uniform_mix": self._admission_uniform_mix,
                "priority_alpha": self._admission_priority_alpha,
                "priorities": self._replay_priorities.index_select(1, order).detach().cpu().clone(),
                "slot_write_step": self._slot_write_step.index_select(1, order).detach().cpu().clone(),
                "sample_counts": {
                    role: values.detach().cpu().clone()
                    for role, values in self._admission_sample_counts.items()
                },
                "policy_events": list(self._admission_policy_events),
            },
        }

    @torch.no_grad()
    def import_valid(self, snapshot: dict, *, strict: bool = True) -> None:
        """Restore an :meth:`export_valid` snapshot into the wrapped buffer."""

        if int(snapshot.get("schema_version", -1)) != 1:
            raise ValueError("unsupported replay snapshot schema")
        if snapshot.get("storage_order") != "chronological_oldest_to_newest":
            raise ValueError("unsupported replay snapshot storage order")
        meta = snapshot["metadata"]
        expected = {
            "n_env": int(self.base.n_env),
            "buffer_size": int(self.base.buffer_size),
            "n_obs": int(self.base.n_obs),
            "n_act": int(self.base.n_act),
            "n_critic_obs": int(self.base.n_critic_obs),
            "asymmetric_obs": bool(self.base.asymmetric_obs),
            "playground_mode": bool(self.base.playground_mode),
            "n_steps": int(self.base.n_steps),
        }
        mismatches = {
            name: (expected[name], meta.get(name))
            for name in expected
            if expected[name] != meta.get(name)
        }
        if mismatches and strict:
            raise ValueError(f"replay metadata mismatch: {mismatches}")
        ptr = int(meta["ptr"])
        valid = min(int(self.base.buffer_size), ptr)
        if valid != int(meta["valid_size"]):
            raise ValueError("invalid replay ptr/valid_size relationship")
        order = self.chronological_slot_indices(ptr=ptr)
        tensors = snapshot["tensors"]
        for name in self._base_tensor_names():
            destination = getattr(self.base, name)
            source = torch.as_tensor(tensors[name], device=destination.device, dtype=destination.dtype)
            if source.shape[0] != self.base.n_env or source.shape[1] != valid:
                raise ValueError(f"invalid {name} snapshot shape: {tuple(source.shape)}")
            destination.zero_()
            destination.index_copy_(1, order, source)
        self.options.fill_(-1)
        source_options = torch.as_tensor(
            tensors["options"], device=self.options.device, dtype=self.options.dtype
        )
        self.options.index_copy_(1, order, source_options)

        provenance = snapshot.get("provenance")
        group_count = meta.get("provenance_group_count")
        if provenance is not None:
            if group_count is None:
                raise ValueError("provenance snapshot is missing group_count")
            self.enable_provenance(int(group_count))
            assert self._provenance_written is not None
            for name, destination in self._provenance.items():
                destination.fill_(False if destination.dtype == torch.bool else -1)
                source = torch.as_tensor(
                    provenance[name], device=destination.device, dtype=destination.dtype
                )
                destination.index_copy_(1, order, source)
            self._provenance_written.zero_()
            written = torch.as_tensor(
                provenance["provenance_written"],
                device=self._provenance_written.device,
                dtype=torch.bool,
            )
            self._provenance_written.index_copy_(1, order, written)
        elif self._provenance:
            for destination in self._provenance.values():
                destination.fill_(False if destination.dtype == torch.bool else -1)
            assert self._provenance_written is not None
            self._provenance_written.zero_()

        weights = snapshot.get("source_weights") or {}
        self._src_w_critic = None
        self._src_w_actor = None
        if weights.get("critic") is not None:
            self.set_source_weights(weights["critic"], role="critic")
        if weights.get("actor") is not None:
            self.set_source_weights(weights["actor"], role="actor")
        admission = snapshot.get("admission_sampling")
        if admission is not None:
            self.set_admission_policy(
                admitted_sources=admission["source_mask"],
                candidate_masses=admission["candidate_masses"],
                recency_half_life=float(admission["recency_half_life"]),
                uniform_mix=float(admission["uniform_mix"]),
                priority_alpha=float(admission["priority_alpha"]),
            )
            self._admission_source_authority_active = bool(
                admission.get("source_authority_active", True)
            )
            assert self._replay_priorities is not None and self._slot_write_step is not None
            priority_values = torch.as_tensor(
                admission["priorities"],
                device=self.base.device,
                dtype=torch.float32,
            )
            write_values = torch.as_tensor(
                admission["slot_write_step"],
                device=self.base.device,
                dtype=torch.int64,
            )
            self._replay_priorities.index_copy_(1, order, priority_values)
            self._slot_write_step.index_copy_(1, order, write_values)
            for role, values in (admission.get("sample_counts") or {}).items():
                if role in self._admission_sample_counts:
                    self._admission_sample_counts[role].copy_(
                        torch.as_tensor(
                            values,
                            device=self.base.device,
                            dtype=torch.int64,
                        )
                    )
            self._admission_policy_events = list(admission.get("policy_events") or [])
        self.base.ptr = ptr

    @property
    def provenance_enabled(self) -> bool:
        return bool(self._provenance)

    @torch.no_grad()
    def max_provenance_segment_id(self) -> int:
        """有效槽位中最大的 segment_id;无 provenance 或空 buffer 时返回 -1。
        anchor-resume 分支用它续接 segment 命名空间,避免与 anchor 数据碰撞。"""

        if not self._provenance or self.valid_size <= 0:
            return -1
        segment_ids = self._provenance["segment_id"][:, : self.valid_size]
        written = self._provenance_written[:, : self.valid_size]
        if not bool(written.any()):
            return -1
        return int(segment_ids[written].max())

    def assert_complete_provenance(self) -> None:
        if not self._provenance or self._provenance_written is None:
            raise AssertionError("behavior provenance was not enabled")
        order = self.chronological_slot_indices()
        if not bool(self._provenance_written.index_select(1, order).all()):
            missing = int((~self._provenance_written.index_select(1, order)).sum().item())
            raise AssertionError(f"{missing} valid transitions lack behavior provenance")

    @torch.no_grad()
    def source_provenance_samples(self) -> tuple[torch.Tensor, torch.Tensor]:
        """当前有效槽位中全部 source-provenance 的 ``(raw_obs, actions)``。

        `z = 1` 定义为**任一身体组实际由 source 执行**，即
        ``executed_group_mask.any(-1)``；用它而不是 ``behavior_source``，
        因为后者是 option id，null option 的取值与"student 执行"不可靠地重合。
        仅统计 ``provenance_written`` 为真的槽位，未写过的不算 student。

        供 PARE 在 release 时刻构建固定 source reservoir（`pare.py`）。
        """
        if not self._provenance or self._provenance_written is None:
            raise AssertionError("behavior provenance was not enabled")
        valid_n = self.valid_size
        if valid_n <= 0:
            raise RuntimeError("cannot scan an empty replay buffer")
        written = self._provenance_written[:, :valid_n]
        is_source = self._provenance["executed_group_mask"][:, :valid_n].any(dim=-1)
        mask = is_source & written
        env_idx, slot_idx = mask.nonzero(as_tuple=True)
        rb = self.base
        return rb.observations[env_idx, slot_idx], rb.actions[env_idx, slot_idx]
