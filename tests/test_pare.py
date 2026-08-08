#!/usr/bin/env python3
"""PARE 核心数学的单元测试（docs/PARE_ALGORITHM_SPEC_v1.md §6）。

重点验两条实现正确性依赖的性质：
  Lemma 1        ⟨g_Q, g_PARE⟩ ≥ ‖g_Q‖²
  AMP 齐次性      两路梯度同乘 c>0，合成结果恰为 c·g_PARE
后者是"unscale 一次即可"的依据，写错会让 GradScaler 下的 PARE 静默失真。

运行：python tests/test_pare.py
"""

import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from fasttd3_ptf.official_fasttd3_ptf.pare import (  # noqa: E402
    PARERuntime,
    SourceOccupancyDiscriminator,
    SourceTransitionReservoir,
    compose_pare_gradient,
)

FAILS = []


def check(cond, msg):
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    if not cond:
        FAILS.append(msg)


def _dot(a, b):
    return float(sum((x * y).sum() for x, y in zip(a, b)))


def _norm(a):
    return float(sum((x * x).sum() for x in a)) ** 0.5


torch.manual_seed(0)

print("T1  Lemma 1：⟨g_Q, g_PARE⟩ ≥ ‖g_Q‖²（含冲突与非冲突两种情形）")
for trial in range(200):
    g_Q = [torch.randn(7), torch.randn(3, 4)]
    g_E = [torch.randn(7) * 5, torch.randn(3, 4) * 5]
    g, conflict, ratio = compose_pare_gradient(g_Q, g_E)
    lhs, rhs = _dot(g_Q, g), _dot(g_Q, g_Q)
    if lhs < rhs - 1e-4 * max(1.0, abs(rhs)):
        check(False, f"trial {trial}: ⟨g_Q,g_PARE⟩={lhs:.6f} < ‖g_Q‖²={rhs:.6f}")
        break
else:
    check(True, "200 次随机试验全部满足 Lemma 1")

# 构造一个必然冲突的例子，确认冲突分支真的被走到。
g_Q = [torch.tensor([1.0, 0.0])]
g_E = [torch.tensor([-3.0, 2.0])]
g, conflict, ratio = compose_pare_gradient(g_Q, g_E)
check(conflict, "⟨g_Q,g_E⟩<0 时 conflict=True")
# 投影后 g_E 的 g_Q 分量应被完全消去
check(abs(float(g[0][0]) - 1.0) < 1e-6,
      f"投影后 g_PARE 的 g_Q 方向分量 = ‖g_Q‖ 分量本身（实得 {float(g[0][0]):.6f}）")

g_Q = [torch.tensor([1.0, 0.0])]
g_E = [torch.tensor([2.0, 0.0])]
g, conflict, ratio = compose_pare_gradient(g_Q, g_E)
check(not conflict, "⟨g_Q,g_E⟩>0 时 conflict=False（不投影）")

print("\nT2  norm cap：‖ḡ_E‖ ≤ ‖g_Q‖")
for scale_e in (0.01, 1.0, 100.0):
    g_Q = [torch.randn(20)]
    g_E = [torch.randn(20) * scale_e]
    g, _, ratio = compose_pare_gradient(g_Q, g_E)
    # ḡ_E = g_PARE - g_Q
    bar_e = [a - b for a, b in zip(g, g_Q)]
    check(_norm(bar_e) <= _norm(g_Q) * (1 + 1e-5),
          f"scale_e={scale_e:<6}: ‖ḡ_E‖={_norm(bar_e):.6f} ≤ ‖g_Q‖={_norm(g_Q):.6f}")
    check(float(ratio) <= 1.0 + 1e-5, f"  expansion_norm_ratio={float(ratio):.4f} ≤ 1")

print("\nT3  60° 锥推论：cos∠(g_Q, g_PARE) ≥ 1/2")
worst = 1.0
for _ in range(300):
    g_Q = [torch.randn(11)]
    g_E = [torch.randn(11) * torch.rand(1).item() * 50]
    g, _, _ = compose_pare_gradient(g_Q, g_E)
    cos = _dot(g_Q, g) / (_norm(g_Q) * _norm(g) + 1e-12)
    worst = min(worst, cos)
check(worst >= 0.5 - 1e-5, f"300 次试验最小 cos = {worst:.6f} ≥ 0.5")

print("\nT4  AMP 齐次性：两路同乘 c>0 → 结果恰为 c 倍")
for c in (2.0, 65536.0, 0.03125):
    g_Q = [torch.randn(9), torch.randn(2, 5)]
    g_E = [torch.randn(9) * 3, torch.randn(2, 5) * 3]
    base, conflict_a, ratio_a = compose_pare_gradient(g_Q, g_E)
    scaled, conflict_b, ratio_b = compose_pare_gradient(
        [x * c for x in g_Q], [x * c for x in g_E]
    )
    err = max(float((s - b * c).abs().max()) for s, b in zip(scaled, base))
    rel = err / max(1e-12, max(float((b * c).abs().max()) for b in base))
    check(rel < 1e-5, f"c={c:<10}: 相对误差 {rel:.3e} < 1e-5")
    check(conflict_a == conflict_b, f"c={c:<10}: conflict 判定不随缩放改变")
    check(abs(float(ratio_a) - float(ratio_b)) < 1e-5,
          f"c={c:<10}: norm ratio 不随缩放改变")

print("\nT5  g_E=0 时 PARE 退化为纯 base RL")
g_Q = [torch.randn(6)]
g_E = [torch.zeros(6)]
g, conflict, ratio = compose_pare_gradient(g_Q, g_E)
check(max(float((a - b).abs().max()) for a, b in zip(g, g_Q)) < 1e-6,
      "g_E=0 → g_PARE == g_Q（不引入任何扰动）")
check(not conflict and float(ratio) < 1e-6, "conflict=False 且 ratio=0")

print("\nT6  reservoir：截断可见、不静默")
raw = torch.randn(1000, 4)
act = torch.randn(1000, 2)


class _FakeRB:
    def source_provenance_samples(self):
        return raw, act


r = SourceTransitionReservoir.from_replay(_FakeRB(), capacity=100)
check(len(r) == 100, f"超容量时截到 capacity（实得 {len(r)}）")
check(r.n_candidates == 1000, f"候选总数如实记录（实得 {r.n_candidates}）")
check(r.truncated is True, "truncated=True")

r2 = SourceTransitionReservoir.from_replay(_FakeRB(), capacity=5000)
check(len(r2) == 1000 and r2.truncated is False, "未超容量时全留且 truncated=False")

try:
    SourceTransitionReservoir(torch.zeros(0, 4), torch.zeros(0, 2), n_candidates=0)
    check(False, "空 reservoir 应当报错")
except ValueError:
    check(True, "空 reservoir 直接报错（fail-closed，不静默继续）")

s_obs, s_act = r.sample(32)
check(s_obs.shape == (32, 4) and s_act.shape == (32, 2), "sample 形状正确")

print("\nT7  discriminator：形状与有限性")
d = SourceOccupancyDiscriminator(obs_dim=4, act_dim=2)
out = d.logit(torch.randn(16, 4), torch.randn(16, 2))
check(out.shape == (16,), f"logit 形状 [B]（实得 {tuple(out.shape)}）")
check(bool(torch.isfinite(out).all()), "logit 有限")

print("\nT8  expansion 梯度在 D 饱和时**不得**消失（clamp 死区回归）")
# 2k smoke 实测：release 后 D 迅速达到 d_acc=1.0，72% 样本的 |logit| > 10。
# 首版对 logit 做 clamp(±10)，把这 72% 的梯度置零——恰好静音了最像 source、
# 最该被推离的样本。这里锁死正确行为：dJ_E/dz = -sigmoid(z)，z 越大越接近 -1。


class _SaturatingD(torch.nn.Module):
    """强制输出大 logit，模拟 release 后 D 完美判别的情形。"""

    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def logit(self, obs, act):
        return self.scale * act.sum(dim=-1)


class _Anchor:
    def __call__(self, obs):
        return torch.zeros(obs.shape[0], 3)


class _Shell:
    """只借 PARERuntime 的 expansion_objective，不构造完整训练栈。"""

    def __init__(self, scale):
        self.D = _SaturatingD(scale)
        self.pi_B = _Anchor()

    expansion_objective = PARERuntime.expansion_objective


for scale, tag in ((1.0, "未饱和"), (50.0, "深度饱和")):
    shell = _Shell(scale)
    obs = torch.ones(8, 5)
    act = torch.ones(8, 3, requires_grad=True)
    j_e, metrics = shell.expansion_objective(
        obs, obs, act, lambda o, a: torch.ones(o.shape[0])
    )
    (g,) = torch.autograd.grad(j_e, act)
    gn = float(g.abs().max())
    check(gn > 1e-6, f"{tag}(scale={scale}): |dJ_E/da|max={gn:.6e} > 0")
    check(bool(torch.isfinite(j_e)), f"{tag}: J_E 有限（{float(j_e):.4f}）")

# 直接验梯度恒等式，避免以后有人重新引入截断
z = torch.tensor([-30.0, -5.0, 0.0, 5.0, 30.0], requires_grad=True)
(gz,) = torch.autograd.grad((-torch.nn.functional.softplus(z)).sum(), z)
expect = -torch.sigmoid(z.detach())
check(float((gz - expect).abs().max()) < 1e-6,
      "d/dz[-softplus(z)] == -sigmoid(z) 全域成立")
check(float(gz[-1]) < -0.99, f"z=+30（最像 source）处梯度={float(gz[-1]):.4f} ≈ -1，未被静音")

print()
print("全部通过" if not FAILS else f"{len(FAILS)} 项失败")
sys.exit(0 if not FAILS else 1)
