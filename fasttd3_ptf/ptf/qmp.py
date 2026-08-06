"""QMP-fidelity per-state 完整策略选择器。

Run card: docs/run_card_qmp_fidelity_v1.md

对应 QMP(ICLR 2025) 的 per-state behavior-policy selection:

    i*(s) = argmax_i  score_i(s),   score_i(s) = min_h Q_h(s, π_i(s))

与本项目 MCG 的**本质区别**(外部审查 2026-07-29 指出):

- 候选是**完整策略**的实际输出,不是身体组拼接的混合动作
  (拼接动作可能不在任何策略的动作流形上,critic 在该点可能无数据支持);
- 打分是 Q 值本身,不是 MCG 的 paired delta `min_h[Q_h(cand) − Q_h(stu)]`
  ——后者既不等于 QMP 的 soft policy value,也不等于 `min_h Q_h(c) − min_h Q_h(s)`。

因此本模块**不引用** QMP Theorem 5.1 作为安全保证:该定理建立在 tabular SAC、
有限动作空间上,而这里是连续 61 维动作 + 分布式双 critic + 冻结跨任务源。

打分口径 `min_h` 与 FastTD3 actor 的优化目标一致(use_cdq=True 时
actor_loss = −min(qf1,qf2).mean(),见 fast_td3/train.py 与 train_ptf.py:1753)。

**候选索引 0 恒为 student**,`argmax` 并列时返回最小索引,因此自动满足
run card 冻结的 ties→student 规则。
"""
from __future__ import annotations

from typing import Callable

import torch

# qheads_fn(critic_obs[B,O], actions[B,A]) -> (q1[B], q2[B])
QHeadsFn = Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]


class QmpSelector:
    """per-state 完整策略 argmax + 机制诊断累计。

    诊断量只用于观察,**不参与任何 gate**(run card §4.2)。
    """

    def __init__(self, num_sources: int, num_envs: int, device: torch.device | str = "cpu"):
        if int(num_sources) < 1:
            raise ValueError(f"QMP requires a non-empty source bank, got {num_sources}")
        self.num_sources = int(num_sources)
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        # 候选数 = 1(student) + 源数
        self.num_candidates = self.num_sources + 1
        # 上一步的选择(用于 switch_rate 与连续段统计);初始化为 student
        self.prev_choice = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        # 每个候选的累计执行步数与连续段数 → 平均连续执行长度 = sum / count
        self.run_sum = torch.zeros(self.num_candidates, dtype=torch.float64, device=self.device)
        self.run_count = torch.zeros(self.num_candidates, dtype=torch.float64, device=self.device)
        self._started = False

    @torch.no_grad()
    def scores(
        self,
        qheads_fn: QHeadsFn,
        critic_obs: torch.Tensor,
        student_action: torch.Tensor,
        src_actions: torch.Tensor,
    ) -> torch.Tensor:
        """[B, 1+S] 的 min_h Q_h;列 0 = student。

        student_action 与 src_actions 都必须是**无噪声**动作
        ——run card §3.1:拿带噪声的 student 与无噪声的 source 比 Q 会系统性偏向 source。
        """
        if src_actions.shape[1] != self.num_sources:
            raise ValueError(
                f"src_actions has {src_actions.shape[1]} sources, expected {self.num_sources}"
            )
        cand = torch.cat([student_action.unsqueeze(1), src_actions], dim=1)  # [B, 1+S, A]
        out = torch.empty(
            cand.shape[0], self.num_candidates, device=cand.device, dtype=torch.float32
        )
        for k in range(self.num_candidates):
            q1, q2 = qheads_fn(critic_obs, cand[:, k, :])
            out[:, k] = torch.minimum(q1, q2).float()
        if not torch.isfinite(out).all():
            # run card §4.2: 非有限 Q 拒绝启动,不静默跳过
            raise FloatingPointError(
                "QMP candidate scores contain NaN/Inf; refusing to select "
                f"(finite fraction = {torch.isfinite(out).float().mean().item():.4f})"
            )
        return out

    @torch.no_grad()
    def select(
        self,
        qheads_fn: QHeadsFn,
        critic_obs: torch.Tensor,
        student_action: torch.Tensor,
        src_actions: torch.Tensor,
        dones: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """返回 (selected_action[B,A], choice[B] long, diagnostics)。

        choice==0 表示 student,choice==i+1 表示源 i。
        """
        sc = self.scores(qheads_fn, critic_obs, student_action, src_actions)
        choice = sc.argmax(dim=1)  # 并列 → 最小索引 → student
        cand = torch.cat([student_action.unsqueeze(1), src_actions], dim=1)
        selected = cand[torch.arange(cand.shape[0], device=cand.device), choice]
        diag = self._update_diagnostics(sc, choice, dones)
        return selected, choice, diag

    def _update_diagnostics(
        self,
        sc: torch.Tensor,
        choice: torch.Tensor,
        dones: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        # episode 边界必然开启新段,与"选择改变"同等对待
        if dones is not None:
            boundary = dones.view(-1).bool()
        else:
            boundary = torch.zeros_like(choice, dtype=torch.bool)
        changed = (choice != self.prev_choice) | boundary
        if not self._started:
            changed = torch.ones_like(changed)
            self._started = True

        self.run_sum += torch.bincount(choice, minlength=self.num_candidates).to(self.run_sum)
        if changed.any():
            self.run_count += torch.bincount(
                choice[changed], minlength=self.num_candidates
            ).to(self.run_count)

        score_gap = (sc.max(dim=1).values - sc[:, 0]).mean()
        diag: dict[str, torch.Tensor] = {
            "qmp/source_share": (choice != 0).float().mean(),
            "qmp/score_gap": score_gap,
            "qmp/switch_rate": changed.float().mean(),
        }
        share = torch.bincount(choice, minlength=self.num_candidates).float() / choice.numel()
        diag["qmp/share_student"] = share[0]
        for i in range(self.num_sources):
            diag[f"qmp/share_src{i}"] = share[i + 1]
        self.prev_choice = choice.clone()
        return diag

    def mean_run_lengths(self) -> torch.Tensor:
        """[1+S] 每个候选的平均连续执行长度(段数为 0 时给 0)。"""
        return torch.where(
            self.run_count > 0, self.run_sum / self.run_count.clamp_min(1.0),
            torch.zeros_like(self.run_sum),
        ).float()
