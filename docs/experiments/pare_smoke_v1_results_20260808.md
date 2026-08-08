# PARE 2k smoke 结果 —— SMOKE_PASS

日期：2026-08-08 · 判据：`docs/PARE_ALGORITHM_SPEC_v1.md` §12 第 3 项
数据：`docs/data/pare_smoke_v1/smoke_verdict.json`

**定位：工程冒烟，不构成任何科学结论。** 32 env / 2k 步的规模只够验代码路径，
不用于任何性能主张。通过即停止工程验证，不再扩建。

---

## 判决

| 判据 | 结果 | 实测 |
|---|---|---|
| S1 provenance 两类都存在 | PASS | z=1 共 16861 条，z=0 共 15139 条（source share 0.527） |
| S2 D 输出有限 | PASS | `d_loss` + `source_affinity` 共 20 个采样点，非有限 0 个，范围 [0.000193, 1.379] |
| S3 actor 梯度有限 | PASS | `base_grad_norm_scaled` + `expansion_norm_ratio` 共 20 点，非有限 0 个 |
| S4 PARE 生效且可关 | PASS | PARE-on 与 PARE-off 从同一 anchor 出发，actor 参数最大差 **0.3372** |

`verdict = SMOKE_PASS`，退出码 0。

S4 的方向说明：这里证的是"PARE 不是空操作"。
`pare_runtime is None` 时 `update_pol` 逐行走原路径，该等价性由 357 项既有回归测试覆盖，
不在此重复。

---

## 首轮抓到的真实缺陷（已修，本结果为 hotfix 后的独立重跑）

首轮 smoke 实测 `d_logit_clamp_rate` 达到 **0.722**。

根因：spec v1.0 对 `log(1-D)` 的 logit 做了 `clamp(±10)`，**方向是反的**。

```
log(1 - σ(z)) = -softplus(z),   d/dz = -σ(z) ∈ (-1, 0)
```

`z→+∞` 即样本**最像 source、最该被推离**处，梯度趋于 −1；
clamp 恰好把这部分置零，静音了 72% 的样本。
与仓库里记录的 β-clamp logit 死区属同一类错误：护住值域、杀死梯度。
`softplus` 自身已数值稳定，clamp 纯属多余。

处置按 CLAUDE.md §4.1：首轮结果未作为通过证据提交 → hotfix 只含代码
（`598135d` 之后的实现提交）→ spec 修订为 v1.1 → **删除首轮全部产物后独立重跑**。
本文件报告的是重跑结果。判据 §8 F1–F7 全程未动。

回归测试 `tests/test_pare.py` T8 锁死正确行为：深度饱和（scale=50）时
`|dJ_E/da|max = 6.25` 而非 0，并直接验 `d/dz[-softplus(z)] == -sigmoid(z)` 全域成立。

---

## 机制量观测（诊断，不构成结论）

重跑末次采样点：

| 量 | 值 |
|---|---|
| `pare/d_acc` | 1.000 |
| `pare/source_affinity` | 0.000381 |
| `pare/anchor_adv_positive_frac` | 0.899 |
| `pare/anchor_adv_mean` | 2.150 |
| `pare/grad_conflict` | 0.0 |
| `pare/expansion_norm_ratio` | 0.00475 |
| `pare/d_logit_saturated_rate` | 0.952 |
| `d_skip_count` | 0 |

两点值得在正式实验里盯住，**现在不据此加任何机制**：

1. **`d_acc` 迅速到 1.0，`d_logit_saturated_rate` 0.95。** release 后 student 很快
   离开 source occupancy，D 几乎完美判别。这不再造成梯度死区（已去 clamp），
   且 `J_E` 对 logit 的梯度有界于 (−1, 0)，不会爆炸。但它意味着
   `ρ_src = logit D` 的**数值**在饱和区不可当作精确 density ratio 读，
   只有符号与序是可信的。
2. **`expansion_norm_ratio` 仅 0.005–0.078**，norm cap 几乎从不触发。
   expansion 梯度天然远小于 base RL 梯度。若正式实验里它始终这么小，
   spec §8 的 F7（"投影从未生效 → PARE 退化为纯加性 bonus"）的近亲情形出现，
   届时须如实报告 expansion 实际影响有限，而不是邀功。

`d_skip_count = 0` 说明 discriminator 的负样本从未为空——
release 后新写入的 transition 恒为 z=0，过滤后仍充足。
