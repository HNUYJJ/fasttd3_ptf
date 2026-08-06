"""OptionSelector 的 call-and-return 语义测试,重点覆盖 min_duration 冷却。"""
import torch

from fasttd3_ptf.ptf.option_selector import OptionSelector


class _StubOptionModule:
    """固定 Q/β 的桩:β=1 必终止(无冷却时),Q 偏好 option 0。"""

    def __init__(self, num_options: int, beta_value: float = 1.0):
        self.num_options = num_options
        self.beta_value = beta_value

    def __call__(self, obs):
        n = obs.shape[0]
        q = torch.zeros(n, self.num_options)
        q[:, 0] = 1.0
        beta = torch.full((n, self.num_options), self.beta_value)
        return q, beta


def test_min_duration_suppresses_termination():
    device = torch.device("cpu")
    sel = OptionSelector(num_envs=4, num_options=3, device=device, epsilon=0.0, initial_option=2, seed=0, min_duration=5)
    stub = _StubOptionModule(3, beta_value=1.0)
    obs = torch.zeros(4, 8)
    # β=1 本应每步终止;冷却期内(前 4 步)必须保持 initial_option=2
    for _ in range(4):
        ids = sel.step(obs, stub)
        assert torch.all(ids == 2), ids
    # 第 5 步冷却结束,greedy 切到 option 0
    ids = sel.step(obs, stub)
    assert torch.all(ids == 0), ids
    # 切换后计数清零:接下来 4 步又保持
    for _ in range(4):
        ids = sel.step(obs, stub)
        assert torch.all(ids == 0), ids


def test_done_forces_reselect_despite_cooldown():
    device = torch.device("cpu")
    sel = OptionSelector(num_envs=2, num_options=3, device=device, epsilon=0.0, initial_option=2, seed=0, min_duration=100)
    stub = _StubOptionModule(3, beta_value=0.0)
    obs = torch.zeros(2, 8)
    dones = torch.tensor([True, False])
    ids = sel.step(obs, stub, dones=dones)
    # done 的 env 强制重选(greedy=0),未 done 的 env 留在冷却中的 option 2
    assert ids[0].item() == 0 and ids[1].item() == 2, ids


def test_min_duration_default_keeps_legacy_behavior():
    device = torch.device("cpu")
    sel = OptionSelector(num_envs=3, num_options=2, device=device, epsilon=0.0, initial_option=1, seed=0)
    stub = _StubOptionModule(2, beta_value=1.0)
    obs = torch.zeros(3, 8)
    # min_duration=1(默认):β=1 第一步即可终止并切 greedy
    ids = sel.step(obs, stub)
    assert torch.all(ids == 0), ids


def test_cumulative_diagnostics_count_beta_termination():
    device = torch.device("cpu")
    sel = OptionSelector(
        num_envs=4,
        num_options=2,
        device=device,
        epsilon=0.0,
        initial_option=1,
        seed=3,
    )
    stub = _StubOptionModule(2, beta_value=1.0)
    obs = torch.zeros(4, 8)

    ids = sel.step(obs, stub, dones=torch.zeros(4, dtype=torch.bool))
    diagnostics = sel.cumulative_diagnostics()

    assert torch.equal(ids, torch.zeros(4, dtype=torch.long))
    assert diagnostics["beta_termination_events"].item() == 4
    assert diagnostics["beta_termination_rate"].item() == 1.0


def test_released_code_selector_chooses_immediately_after_reset() -> None:
    device = torch.device("cpu")
    selector = OptionSelector(
        num_envs=3,
        num_options=3,
        device=device,
        epsilon=0.0,
        initial_option=2,
        seed=11,
        select_on_reset=True,
        sample_choices_only_when_needed=True,
    )
    stub = _StubOptionModule(3, beta_value=0.0)
    obs = torch.zeros(3, 8)

    first = selector.step(obs, stub)
    diagnostics = selector.cumulative_diagnostics()

    assert torch.equal(first, torch.zeros(3, dtype=torch.long))
    assert diagnostics["beta_termination_events"].item() == 0
    assert torch.equal(
        selector.needs_reselection,
        torch.zeros(3, dtype=torch.bool),
    )
