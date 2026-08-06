from __future__ import annotations

import torch


class OptionSelector:
    """Call-and-return option selector.

    ``min_duration``:重选后的冷却步数,期内 β 终止被抑制(episode 结束的
    强制重选不受限)。execute 模式必需:β 冻结在 ~0.16 时选项平均只持续
    ~6 步,源策略(如 approach 行走)的闭环行为会被频繁换老师打断。
    """

    def __init__(
        self,
        num_envs: int,
        num_options: int,
        device: torch.device,
        epsilon: float = 0.1,
        initial_option: int | None = None,
        seed: int | None = None,
        min_duration: int = 1,
        select_on_reset: bool = False,
        sample_choices_only_when_needed: bool = False,
    ):
        self.num_envs = int(num_envs)
        self.num_options = int(num_options)
        self.device = device
        self.epsilon = float(epsilon)
        self.min_duration = max(1, int(min_duration))
        self.select_on_reset = bool(select_on_reset)
        self.sample_choices_only_when_needed = bool(
            sample_choices_only_when_needed
        )
        self.generator_device = device
        try:
            self.generator = torch.Generator(device=device)
        except (RuntimeError, TypeError):
            self.generator_device = torch.device("cpu")
            self.generator = torch.Generator()
        if seed is not None:
            self.generator.manual_seed(int(seed))
        if initial_option is None:
            initial_option = self.num_options - 1
        self.current_options = torch.full((self.num_envs,), int(initial_option), device=device, dtype=torch.long)
        self.steps_in_option = torch.zeros(self.num_envs, device=device, dtype=torch.long)
        # Author-released PTF calls choose_o immediately after every reset.
        # Legacy project behavior keeps the configured initial option until
        # beta/done triggers a choice, so this is opt-in.
        self.needs_reselection = torch.full(
            (self.num_envs,),
            self.select_on_reset,
            device=device,
            dtype=torch.bool,
        )
        # Cumulative observability only. Keeping counters on-device avoids a
        # per-step synchronization and does not consume RNG or alter choices.
        self.total_option_opportunities = torch.zeros((), device=device, dtype=torch.long)
        self.beta_termination_events = torch.zeros((), device=device, dtype=torch.long)
        self.done_reselection_events = torch.zeros((), device=device, dtype=torch.long)
        self.option_change_events = torch.zeros((), device=device, dtype=torch.long)

    def reset(self, dones: torch.Tensor | None = None) -> None:
        if dones is None:
            self.current_options.fill_(self.num_options - 1)
            self.steps_in_option.zero_()
            self.needs_reselection.fill_(self.select_on_reset)
        else:
            mask = dones.bool()
            self.current_options[mask] = self.num_options - 1
            self.steps_in_option[mask] = 0
            if self.select_on_reset:
                self.needs_reselection[mask] = True

    @torch.no_grad()
    def step(self, obs_norm: torch.Tensor, option_module, dones: torch.Tensor | None = None, epsilon: float | None = None) -> torch.Tensor:
        q, beta = option_module(obs_norm)
        eps = self.epsilon if epsilon is None else float(epsilon)
        cur = self.current_options
        cur_beta = beta.gather(1, cur.view(-1, 1)).squeeze(1)
        self.steps_in_option += 1
        in_cooldown = self.steps_in_option < self.min_duration
        done_mask = (
            dones.to(self.device).bool().view(-1)
            if dones is not None
            else torch.zeros_like(cur, dtype=torch.bool)
        )
        reset_choice_mask = self.needs_reselection
        forced_reselection = done_mask | reset_choice_mask
        beta_opportunity = ~in_cooldown & ~forced_reselection
        if self.sample_choices_only_when_needed:
            beta_terminate = torch.zeros_like(beta_opportunity)
            beta_idx = beta_opportunity.nonzero(as_tuple=False).view(-1)
            if beta_idx.numel() > 0:
                beta_terminate[beta_idx] = (
                    self._rand((beta_idx.numel(),)) < cur_beta[beta_idx]
                )
        else:
            beta_terminate = (self._rand(cur_beta.shape) < cur_beta) & beta_opportunity
        terminate = beta_terminate | forced_reselection
        greedy = q.argmax(dim=-1)
        if self.sample_choices_only_when_needed:
            new_option = cur.clone()
            choice_idx = terminate.nonzero(as_tuple=False).view(-1)
            if choice_idx.numel() > 0:
                choice = greedy[choice_idx].clone()
                explore = self._rand((choice_idx.numel(),)) < eps
                explore_idx = explore.nonzero(as_tuple=False).view(-1)
                if explore_idx.numel() > 0:
                    choice[explore_idx] = self._randint(
                        0,
                        self.num_options,
                        (explore_idx.numel(),),
                    )
                new_option[choice_idx] = choice
        else:
            random = self._randint(0, self.num_options, greedy.shape)
            explore = self._rand(cur_beta.shape) < eps
            new_option = torch.where(explore, random, greedy)
        changed = terminate & (new_option != cur)
        self.total_option_opportunities += beta_opportunity.sum()
        self.beta_termination_events += beta_terminate.sum()
        self.done_reselection_events += done_mask.sum()
        self.option_change_events += changed.sum()
        self.current_options = torch.where(terminate, new_option, cur)
        self.steps_in_option = torch.where(terminate, torch.zeros_like(self.steps_in_option), self.steps_in_option)
        self.needs_reselection = torch.where(
            terminate,
            torch.zeros_like(self.needs_reselection),
            self.needs_reselection,
        )
        return self.current_options.clone()

    def cumulative_diagnostics(self) -> dict[str, torch.Tensor]:
        opportunities = self.total_option_opportunities.clamp_min(1)
        reselections = (self.beta_termination_events + self.done_reselection_events).clamp_min(1)
        return {
            "beta_termination_rate": self.beta_termination_events.float() / opportunities,
            "beta_termination_events": self.beta_termination_events.float(),
            "done_reselection_events": self.done_reselection_events.float(),
            "option_change_rate_per_reselection": self.option_change_events.float() / reselections,
        }

    def _rand(self, shape: torch.Size | tuple[int, ...]) -> torch.Tensor:
        return torch.rand(shape, device=self.generator_device, generator=self.generator).to(self.device)

    def _randint(self, low: int, high: int, shape: torch.Size | tuple[int, ...]) -> torch.Tensor:
        return torch.randint(low, high, shape, device=self.generator_device, generator=self.generator).to(self.device)
