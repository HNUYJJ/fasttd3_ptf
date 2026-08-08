# PARE Gate A —— POST-HOC INTERPRETATION（不修改冻结 verdict）

日期：2026-08-08 · 对应结果：`pare_gate_a_results_20260808.md`
数据：`docs/data/pare_gate_a_v1/gate_a_verdict.json`（**原样保留，不重算、不改判据**）

本文件**只**收窄原报告中被我写过头的解释，不产生任何新裁决。
`F1_TRIGGERED_CLOSE_PARE` 与两任务的 `NO_SCAFFOLD_EFFECT` 均维持不变。

---

## 1. G1 只证明了 `NO_IMMEDIATE_RELEASE_GAIN`

原报告的措辞暗示"没有 source-shaped occupancy"，这是**概念错误**，我收回。

source 在 10k–20k 期间**确实在 target 环境里执行了动作**，其 `(s,a,r,s')` 连同
provenance 一起写进了 target replay（`train_ptf.py:3037-3039`）。
所以 replay 与 state visitation 已经被改变了，这一点与 student 表现无关。

准确的表述是：

$$
\text{source-shaped experience} \;\not\Rightarrow\; \text{immediate student improvement}
$$

而**不是** `¬immediate improvement ⇒ ¬occupancy change`。

在 off-policy RL 里这两件事本来就不必同向：采集某条 transition 的 behavior policy
当场表现好不好，与这条 transition 对 Bellman 学习有没有价值，是两个问题。

因此 G1 的正确读法是：**在 10k exposure、source mass 0.5 下，
release 时刻不存在稳健的即时 student 收益。** 仅此而已。

---

## 2. **50k / 100k 不是 matched continuation**（本轮最重要的自我更正）

我在 `run_pare_gate_a_v1.sh` 的注释和结果报告里都写了
"唯一差别是 10k–20k 有无 scaffold"。**这句话对 20k 点成立，对 50k/100k 不成立。**

实际的两臂结构：

| 臂 | 20k 之后 |
|---|---|
| exit | 从 A1 **重新 anchor-resume**：新进程、fresh env reset、`resume_noise_seed = 93000+seed` |
| scratch | 从 A0 一路连续跑到 100k，**20k 处没有任何 restart** |

`train_ptf.py:2544-2552` 明确会在 resume 时重采 `noise_scales`：

```python
noise_generator.manual_seed(int(ptf_cfg["resume_noise_seed"]))
resampled_scales = torch.rand(args.num_envs, 1, generator=noise_generator, ...) * (std_max - std_min) + std_min
actor.noise_scales.copy_(resampled_scales)
```

（注释原文："noise_scales 是 episode 级状态……fresh reset 后不得沿用 anchor 中
上一 episode 的值，按配对 seed 重采样。"）

所以 20k 之后两臂至少相差三项：

$$
\text{source history} \;+\; \text{20k 处的 env/process restart} \;+\; \text{noise\_scales 重采样}
$$

**受影响的结论（一律降级为 descriptive，不得作因果解读）**：

- stair `100k Δ = −226.4, t = −3.18` —— 原报告写的"后期显著有害"**不成立为因果结论**；
- truck `50k Δ = +155.5` / `100k Δ = +148.5` —— 原报告写的"延迟收益"同样**未确认**
  是 source history 造成的。

**不受影响的结论**：20k 的 G1。scaf 臂与 scratch 臂在 10k 都经历了**同一次**
matched resume（同从 A0、同用 `92000+seed`），10k–20k 期间唯一的主要干预就是
source 有无。故"没有即时收益"仍然成立，两任务的 `NO_SCAFFOLD_EFFECT` 维持。

**今后任何 branch 对照，所有臂都必须在分叉点保存并重启**，
使 restart / reset / noise 干预在各臂之间完全匹配。

---

## 3. G2 与 G3 不再作为未来的机制判据

- **G2**：`theory_max = 1000` 的论证只对 locomotion 成立，对 truck 是错的
  （详见原报告 §3，已核实 `humanoid_bench/envs/truck.py`）。
- **G3**：$r = (J_{100}-J_{50})/(J_{50}-J_{20})$ 无法区分 optimization-limited 与
  occupancy-limited。深度 RL 的学习曲线非单调，三个 checkpoint 的斜率比不是
  learner adequacy test。

两者都是"看起来聪明、实际没有辨别力"的判据。**今后不再使用。**
真要排除 learner bottleneck，就直接跑 `num_updates=2` vs `4`。

原判据的输出仍原样保留在 `gate_a_verdict.json` 中，不重算——
在同一批数据上换门即使如实披露也不恢复确认性地位（CLAUDE.md §8.7）。

---

## 4. 关于 stair 旧结果不复现的定性

不归因于"旧实验错了"，也不归因于单纯的随机性。旧 Slide→Stair 实验与 fresh 之间
至少相差：source dose（sibling behavior share 系统性偏高）、训练代码版本、
resume noise seed（`91000+seed` vs `92000+seed`）。

合适的结论是：**Slide→Stair 的 early transfer effect 对 learner trajectory 与
runtime 版本不够稳健，不适合作为新算法成立的基础现象。**
这正是弃用 stair 作开发场的理由。

---

## 5. PARE 保持关闭；若日后复活需先解决的实现—理论不一致

PARE v1 维持关闭，不抢救。以下问题记录在案，供日后参考，**当前不修**：

1. **discriminator 的负样本不是 $d_{\pi_t}$。** 实现取 replay 中所有 `z=0`
   transition，混合了 pre-release student、scaffold 期 student arm、post-release
   student 三种分布，所以实际估计的是
   $\log \frac{d_{\text{source reservoir}}}{d_{\text{historical } z=0}}$，
   而非 spec §3 写的当前 student occupancy ratio。
2. **$\pi_B$ 未真正冻结为 raw-obs→action 的映射。** 它共享持续更新的 obs
   normalizer（spec §10 D1）。我当时的理由（两者须在同一坐标系被同一 Q 评估）
   仍然成立，但若主张"保留 release competence"，normalizer 本应属于 policy 的一部分。
   这是一个真实的张力，不是笔误。
3. **`z` 的粒度很粗。** 任一身体 group 用过 source 即记 `z=1`，
   估的是"发生过 source authority"的分布，不是纯 source policy occupancy。
4. **Lemma 1 的范围。** $g_Q^\top g_{\mathrm{PARE}} \ge \lVert g_Q\rVert^2$ 只说明
   expansion 不反对**当前 learned critic** 的梯度方向，**不蕴含**
   $J(\pi_{\text{new}}) \ge J(\pi)$。critic 错时两者可以一起朝错误方向走。
   原 spec §6 已声明不作 monotonic-return 保证，此处进一步写明边界。
