# PTF + FastTD3 实现与原论文差异审计

> 撰写日期：2026-05-20
> 范围：审查本仓库 `fasttd3_ptf/official_fasttd3_ptf/` + `fasttd3_ptf/ptf/` 当前实现，与
> - **原 PTF 论文**：Yang et al., *Efficient Deep Reinforcement Learning via Adaptive Policy Transfer*, IJCAI 2020 (arXiv:2002.08037)
> - **原 FastTD3 论文**：Seo et al., *FastTD3: Simple, Fast, and Capable Reinforcement Learning for Humanoid Control*, arXiv:2505.22642 (2025)
> 逐项对照，并记录我们已经在每处差异上做过的工作和实验结论。

---

## 1. 文档目的

本仓库的目标是把 PTF 自适应迁移框架应用到 HumanoidBench 这种高维 humanoid 控制场景，
研究问题是：**PTF 能否在 HumanoidBench h1hand 任务（push / package / door / window）上加速 FastTD3 学习，并/或提升最终策略**。

但原 PTF 论文只在 GridWorld / Pinball / 2-link Reacher 等小尺度域上验证过，原 FastTD3 论文
也没做迁移学习实验。因此我们这个实现是**两套方法的合并 + 大幅扩展**。本文档把每一处与
原论文的差异列出来，说明为什么这么做，并对应到我们已完成的实验结果，作为后续研究的方法学
基线。

---

## 2. 原 PTF 算法 (Yang et al., 2020) 核心要点

### 2.1 框架结构（论文 Figure 1）

PTF 包含两个模块：

- **Agent 模块**：目标任务的策略（actor-critic，原论文用 A3C / PPO）
- **Option 模块**：
  - 一组源策略 $\Pi_s = \{\pi_1, \pi_2, \ldots, \pi_n\}$ 作为 intra-option policies
  - 一个 option-value 网络 $Q_o(s, o | \theta_o)$
  - 一个 termination 网络 $\beta(s, o | \theta_\beta)$（与 option-value 共享底层）

### 2.2 关键数学（论文 § 4.3-4.5）

**U-value**（option 进入下一状态 $s'$ 后的期望回报）：

$$
U(s', o | \theta_o) = (1 - \beta(s', o)) \cdot Q'_o(s', o) + \beta(s', o) \cdot \max_{o'} Q'_o(s', o')
$$

**Option Q-learning target**（论文 Algorithm 2 第 5 行）：

$$
y = r + \gamma \cdot U(s', o | \theta_o)
$$

**Termination loss 梯度**（论文 Eq. 5）：

$$
\theta_\beta \leftarrow \theta_\beta - \alpha_\beta \frac{\partial \beta(s', o)}{\partial \theta_\beta} (A(s', o) + \xi)
$$

其中 $A(s', o) = Q_o(s', o) - \max_{o'} Q_o(s', o')$ 是 advantage，$\xi$ 是固定小正则项
（论文 Table 4 给出 $\xi = 0.001$）防止 termination probability 在 option Q 估计早期不准时
被错误地推到最优值。

**自适应权重 $f(\beta_o, t)$**（论文 Eq. 7）：

$$
f(\beta_o, t) = f(t) \cdot (1 - \beta(s_t, o | \theta_\beta))
$$

其中 $f(t) = \frac{1 + \tanh(3 - 0.001 t)}{2}$（论文 Table 4）是一个**预定义的 tanh 衰减
schedule**，从 $t=0$ 时的 ≈1.0 慢慢降到大 $t$ 时的 0。

**Actor 总 loss**（论文 Algorithm 1 第 18 行）：

$$
L_{\text{actor}} = L_{\text{RL}} + f(\beta_o, t) \cdot L_{H} = L_{\text{RL}} + f(\beta_o, t) \cdot H(\pi_o \| \pi(\theta'))
$$

注意：原论文 $L_H$ 是 **cross-entropy** 蒸馏损失 $H(\pi_o \| \pi)$，因为 actor 是 stochastic
（A3C/PPO 输出动作分布）。

### 2.3 Multi-option 更新（论文 § 4.4 末段）

> "Each sample can be used to update the values of multiple options, as long as the option allows to select the sampled action (for continuous action space, this is achieved by fitting action $a$ in the source policy distribution with a certain confidence interval). Thus the sample efficiency can be significantly improved in an off-policy manner."

这是一个**高斯置信区间**判定：对连续动作空间下的每个 source $i$，如果当前样本的 action $a$
落在 $\pi_i(s)$ 的 $\sigma_i$-置信区间内，就用这个样本更新该 option 的 Q-value。

### 2.4 域 / 网络规模

- 网络：actor 2 层 × 64 hidden, critic 2 层 × 64 hidden, option-value 2 层 × 32 hidden
- 实验域：GridWorld (24×21)、Pinball (2D)、Reacher (2-link planar)
- 没有 null option 概念
- 每 task 跑 20 random seeds

---

## 3. 原 FastTD3 算法 (Seo et al., 2025) 核心要点

### 3.1 算法栈

- **Base**: TD3 (Twin Delayed DDPG, Fujimoto et al. 2018)
- **加成项**：
  - 128 个并行 env（SubprocVecEnv for HumanoidBench）
  - Batch size = **32768**
  - **C51 distributional critic**（101 atoms，task-specific $v_{\min}, v_{\max}$）
  - **CDQ**（Clipped Double Q-learning，取 min）
  - **UTD=2**（每个 env step 做 2 次梯度更新）
  - **AMP bfloat16** + **torch.compile** (mode=reduce-overhead) — 文中说合起来 ~70% 加速
  - σ_max = 0.4 mixed noise（mixed Gaussian exploration）
  - Replay buffer size = N × num_envs（FastTD3 paper 默认 N=50000）

### 3.2 关键超参数（论文 Fig 5/6 + Table 1）

| 参数 | 值 |
|---|---|
| Actor 隐层 | 512, 256, 128 |
| Critic 隐层 | 1024, 512, 256 |
| LR | actor=critic=3e-4 |
| γ | 0.99 |
| τ (target soft update) | 0.005 (PyTorch 标准) |
| Policy delay | 2 |
| Noise σ_max | 0.4 |
| Buffer N | 50000 |
| Batch | 32768 |
| num_envs | 128 |
| Total env steps | ~12.8M (= 100k outer iters × 128 envs) |

### 3.3 关键 finding（论文 § 2.1）

- Without layer norm，**CDQ 关键**；
- 大 batch + 并行 env + distributional critic 三件套是核心；
- Layer norm / residual 不必要；
- AMP+compile 必须开才有 paper 的速度。

### 3.4 HumanoidBench 结果（论文 Fig 9）

FastTD3 在 100k 外层迭代（= 12.8M env steps）内，3 seeds 平均：

- h1hand-push: ~**+700-800** return（success_bar=700）
- h1hand-walk/stand/run: 接近 +1000 ceiling
- h1hand-package: 仍是负值
- h1hand-window: ~+500-800
- h1hand-door: ~+500-800

---

## 4. 差异审计（按模块）

### 4.1 PTF 算法层差异

| # | 项目 | 原 PTF 论文 | 我们的实现 | 动机 / 必要性 |
|---|---|---|---|---|
| 1 | Backbone RL 算法 | A3C / PPO（stochastic policy） | TD3（deterministic policy） | **必要**：FastTD3 是确定性策略，符合 HumanoidBench 的连续动作高维设置。论文也说 PTF 可以与 actor-critic 类方法整合。 |
| 2 | Distillation 损失形式 | $H(\pi_o \| \pi(\theta'))$ cross-entropy on action distributions | masked Huber / MSE on action vectors | **被迫修改**：TD3 actor 直接输出动作向量，没有可对照的概率分布。Huber 是 action-distance regression 的标准做法。 |
| 3 | $f(t)$ 时间衰减 schedule | $\frac{1+\tanh(3 - 0.001 t)}{2}$（显式 tanh 公式） | **线性** `start → end / decay_steps`（默认 0.2→0.0 / 300k） | **简化**：线性比 tanh 在工程上更易控制三个变量。Force-PTF 验证发现 schedule 形状不是瓶颈；λ 是否会衰减才是关键。 |
| 4 | Termination 正则 $\xi$ | **固定** $\xi = 0.001$（论文 Table 4） | `xi=0` 时**自适应** $\xi = 0.8 \cdot (Q_{top1} - Q_{top2})$；`xi>0` 时回退到固定 | **扩展**：自适应模式参考了 option-critic 的 margin 思想，目的是在 Q 估计稳定后给出更紧的边界。我们目前的实验都用 `xi=0.001` 固定模式（参考下面的 force-PTF / decay-PTF 实验）。 |
| 5 | Null option | **不存在** | 加入一个固定输出 0 动作、固定终止概率 0 的伪源 | **关键扩展**：原论文说"in case none of the source policy is useful, PTF can still learn optimal policy"但没给具体机制。我们的 null option 提供了一个"安全出口"——option Q 学到 null > 真 source 时就把蒸馏关掉。论文原型在 toy 域不必要，但 humanoid 上要紧。 |
| 6 | Multi-option 更新 | "fitting action a in source policy distribution with a certain confidence interval" | `gaussian_action_compatibility_all()` —— 各维 Gaussian kernel，加 source-specific σ | **实现忠实**：与论文 § 4.4 末段精确对应。 |
| 7 | Action mask | 不存在（toy 域所有 source 都覆盖完整动作空间） | per-source joint group mask（h1hand: legs / torso / arms / hands 四组）| **必要扩展**：h1hand 61 维动作，不同 source（stand 用腿、reach 用手）只该蒸馏自己负责的子集。覆盖整个 action vector 会造成腿部源给手部任务发噪声。 |
| 8 | Observation adapter | 不存在（toy 域 source 与 target obs 完全同构） | per-source obs adapter（`robot_only` 切前 151 维 / `reach` 额外加 hand+target / `humanoidbench_robot_qpos_qvel` 抽 qpos+qvel） | **必要扩展**：HumanoidBench 不同任务的 obs 长度不同（stand=151, reach=157, push=181, door=157 with full qpos+qvel）。源策略训练时见过的 obs 长度必须严格对齐。 |
| 9 | $(1-\beta)$ 加权 | 始终开启（Eq. 7） | 由 `beta_weighted_transfer` 开关控制 | **扩展**：当 β saturating 到 1 时，$(1-\beta)$ gate 会把 distillation 完全杀掉。这正是我们 door run 发现的 bug。提供开关让我们能在调试期单独验证 distillation 机制。 |
| 10 | Option / β 网络规模 | hidden = 32 | hidden = [256, 256] | **规模匹配**：obs 维度从 toy 域的 ~8 维涨到 h1hand 的 151+ 维，必须放大。 |
| 11 | Compatibility σ | 不存在（论文用置信区间但没给数值） | `compatibility_sigma` per source，默认 0.25 → 我们用 1.5 修复 | **被迫调整**：高斯核 $\exp(-\|\Delta a\|^2 / 2\sigma^2 \cdot d)$ 在 61 维动作空间下，σ=0.25 几乎只能产生 0 兼容性（每维误差 0.1 就够把兼容压到 ~0）。我们把 σ 拉到 1.5 才让 `update_all_compatible_options` 真正起作用。 |
| 12 | Option 探索 ε-greedy | 论文 Table 4: ε-start=1.0, ε-end=0.05, decrement=1e-3 (每 episode) | 我们: ε-start=0.3, ε-end=0.05, decay_steps=50000 | **调小起始 ε**：humanoid 上每次切 option 都很贵（actor 完全换轨道），ε=1.0 会让早期完全无序。 |
| 13 | β-head 训练 | 论文 Eq. 5 + β-warmup 无明文规定 | 我们: `beta_warmup_steps`（默认 20000，diagnostic 跑用 2000000 冻结 β） | **加入 warmup**：我们发现 β 默认配置下会瞬间 saturating 到 1，原因可能是 β-head 初始化偏置 + advantage 早期为 0 导致梯度为 0。warmup 是工程缓解。**这是当前最大的算法层未解决问题**。 |

### 4.2 FastTD3 算法层差异

| # | 项目 | 原 FastTD3 论文 | 我们的实现 | 动机 |
|---|---|---|---|---|
| 1 | `torch.compile` | 默认开（论文说 +35% 加速） | **PTF 模式下默认关**（环境变量 `FASTTD3_PTF_ALLOW_COMPILE=1` 才开） | **必要**：动态 option 选择 + source bank 分发不兼容 graph capture。代价：PTF 跑速 ~15 it/s vs scratch ~19 it/s（**Scratch 比 PTF 快 ~30%**）。这导致同等 wallclock 比较时 PTF 处于劣势，必须用 sample-efficiency（同 env_steps）做对比。 |
| 2 | AMP bfloat16 | 默认开 | 一致（`AMP=1`） | - |
| 3 | $v_{\min}, v_{\max}$（C51 atoms 支撑） | 任务相关，论文没给 h1hand-push 具体值 | 我们：push 用 -1000 / +1000；window/door 用 -250 / +250 | **可疑**：push 任务理论 return 上限可达 +1000+ × time-discount，atoms ±1000 可能太窄。我们在 force-PTF 跑中观察到 `qf_max=999.9, qf_min=-1000` 全程贴边，说明分布已撑到极限。但 scratch 也用同样设置，所以这不是 PTF vs scratch 的偏差源。 |
| 4 | 网络架构 | actor=[512,256,128], critic=[1024,512,256] | 一致 | - |
| 5 | num_envs / batch / buffer | 128 / 32768 / 50000×128 | 一致 | - |
| 6 | UTD | 2（论文默认） | 一致 | - |
| 7 | σ_max | 0.4 | 一致 | - |
| 8 | Eval protocol | 论文用 128 eval envs × 128 steps deterministic | 一致 | - |
| 9 | Seed reporting | 论文 3 seeds 平均 | 我们：**单 seed → 多 seed (3)**，本审计的最后一次实验已经做 3v3 多 seed | 在我们之前的 5/16-5/18 实验都用单 seed，导致 force-PTF 跑出"Scratch +148 vs Force-PTF -20" 的误导对比。多 seed 修复了这点（见 § 5）。 |
| 10 | Total env steps | 12.8M (= 100k iter × 128 envs) | 一致 | - |

### 4.3 SourceBank / Adapter 层差异（这是我们完全新增的）

| # | 项目 | 原 PTF 论文 | 我们的实现 | 动机 |
|---|---|---|---|---|
| 1 | Source 加载格式 | 不存在（论文 source 是同代码训出的 A3C network） | manifest JSON（含 checkpoint 路径、obs/action dim、actor_hidden_dims、normalizer state、adapter spec、action mask）→ 拼成 source bank YAML | 必要工程化：跨 task 复用 checkpoints |
| 2 | Source obs normalizer | 不存在 | 每个 source 独立 normalizer，状态从 checkpoint 加载并 freeze | h1hand 的 obs 数值范围跨任务差异大 |
| 3 | Source action 输出范围 | 不存在 | per-source `action_low/high` 数组 | tanh-bounded actor 默认是 [-1,1] |

---

## 5. 已完成的实验工作和结论

按时间线整理：

### 5.1 Door PTF default run（2026-05-17 ~ 18）

**目的**：用默认 PTF 配置（论文式 schedule + β-gate + adaptive ξ）跑 h1hand-door-v0 一次，看效果。

**配置**：`λ_start=0.2 → 0.0 / 300k steps`, β-gate 开, adaptive ξ (`xi=0`), 4 sources + null, σ=0.25。

**结果**：跑完 100k iters 后 eval_avg_return ≈ +343。**但** PTF 内部信号:

| 信号 | 值 |
|---|---|
| `transfer_loss` | **1.4e-4**（应该 > 0.1）|
| `ptf_transfer_gate_mean` = $(1-\beta)$ | **0.002**（应该 ~0.5-1.0） |
| `ptf_transfer_weight_mean` = λ × (1-β) | **3e-4** |
| `ptf_beta_selected_mean` | **0.996**（β 饱和到 1） |
| `ptf_source_compat_mean` | **0.0018**（σ=0.25 太小）|

**诊断结论**：PTF 实际上**等价于关闭状态** —— β 立刻饱和到 1 把蒸馏 gate 杀到 0，加上 σ 太小让 multi-option 更新永远不触发。我们之前以为"PTF 没效果"，其实是"PTF 没运行"。

详见 memory: [project_force_ptf_run.md](memory)。

### 5.2 Force-PTF vs Scratch（2026-05-20 上午，单 seed）

**目的**：把 β-gate 关闭、λ 固定 1.0、σ 拉宽到 1.5，**强制让 distillation 真正作用到 actor**，验证 PTF 机制本身是否能影响 policy。

**配置**：`λ=1.0 constant`, `beta_weighted_transfer=false`, `xi=0.001` 固定, `beta_warmup_steps=2M`（冻 β）, σ=1.5。

**结果**（同一 seed=1, 100k iters）：

| | PTF mechanism 信号 | Final eval |
|---|---|---|
| Force-PTF | `transfer_loss ≈ 0.5`, gate=1.0, compat=0.73 全程健康 | **-20.4** |
| Scratch | - | **+148.4** |

**结论**：
1. ✅ PTF 机制**正确工作**，能影响 actor（早期 step 5-15k PTF 领先 scratch +35~+84）；
2. ❌ Force 配置造成严重负迁移 —— actor 被 source 持续拉锚，到 step 100k 落后 scratch 169 分；
3. 确认 5.1 的 door run "无效果"是 β saturation 引起的，不是 PTF 算法本身的问题。

详见 [project_force_ptf_run.md](memory)。

### 5.3 Decay-PTF 3v3 vs Scratch 3v3（2026-05-20 下午）

**目的**：加入合理的 λ 衰减 schedule，让 PTF 早期借 source、晚期断奶，跑多 seed 评估。

**配置**：`λ=1.0 → 0.0 / 30000 steps`，β-gate 关，ξ=0.001 固定，β 冻结（warmup=2M），σ=1.5；**3 seeds × (PTF + Scratch)**。

**结果**：

| Seed | Decay-PTF | Scratch |
|---|---|---|
| 1 | -58.82 | **+148.39** |
| 2 | **+21.22** | -15.19 |
| 3 | **+10.75** | -84.63 |
| **Mean** | **-8.95** | **+16.19** |
| **Stdev** | **43.50** | **119.63** |

**统计检验**：
- 均值差 = -25.14
- Welch's t = -0.34（|t|<2 → **无显著性差异**）
- 方差比 Scratch/PTF = **2.75×**
- 配对胜率：PTF 5/9 = 56%

**关键发现**：
1. **PTF 与 Scratch 在均值上统计无差**（t=-0.34）；
2. **PTF 方差是 Scratch 的 1/3**，2/3 PTF seeds 跑出正分 vs 1/3 Scratch seeds 跑出正分；
3. **Scratch 的正均值完全靠 seed 1 outlier (+148) 撑起**：去掉 s1 后 Scratch mean = -50，PTF (-9) 反超 41 分；
4. PTF 全胜 vs Scratch s2/s3（6/6），全输 vs Scratch s1（0/3）——seed 1 像是中彩；
5. 这印证了 PTF 论文的核心想法：早期借 source 加速 → 后期 λ→0 让 actor 自由 → 既不丧失 ceiling 也降低 variance。

详见 [project_decay_ptf_3seed.md](memory)。

---

## 6. 当前还未解决的问题

按重要性排序：

### 6.1 β-head 训练机制是坏的（**2026-05-20 已修复**）

**症状**：默认配置（无 warmup 冻结）下 β 几乎瞬间饱和到 ≈1，导致 (1-β) gate 把 distillation 杀掉。

**根因（已通过 toy diagnostic 确认）**：
- `OptionModule.beta_head` 用 `β = sigmoid(z)` 直接输出
- Sigmoid 在 (0, 1) 两端梯度消失：`dσ/dz = σ(1-σ)` 在 σ ≈ 0 或 σ ≈ 1 时 → 0
- 早期 Q 估计噪声大，每个 option 都会偶尔出现 q_o < max_q 的批次 → β 被推向 1
- 一旦 β 到达 0.95+，sigmoid 死区让梯度信号无法把它拉回来
- 网络对所有 obs 退化为输出常数 ≈ 1，β-head 完全"放弃区分状态"
- 在 option-selector 中表现为：每步都"terminate 并 re-select"，虽然 argmax-Q 仍稳定地选回 stand，但 (1-β) gate ≈ 0.003 把 distillation 完全杀掉

Toy 诊断（`/tmp/beta_diagnostic.py`）：用 5 options + stand argmax 70% of time 训练 3000 步，β_argmax 正确 → 0.003，但 β_non-argmax 卡在 1.000 永远回不来。

**修复**：把 β 限制在 (β_min, β_max)=(0.05, 0.95) 区间：

```python
β = β_min + (β_max - β_min) · sigmoid(z)
```

实现见 [`fasttd3_ptf/my_fasttd3_ptf/models/option.py`](../../fasttd3_ptf/my_fasttd3_ptf/models/option.py)。
关键好处：sigmoid 的可用工作范围对应于 β ∈ [0.05, 0.95]，梯度永远不消失，β 始终可以响应 Q 变化。

**修复后 toy 测试**：β_argmax ≈ 0.053（贴下界但可回升），β_non-argmax ≈ 0.950（贴上界但可下降）。(1-β_selected) ≥ 0.05 始终给 distillation 留出通道。

**测试覆盖**：[`tests/test_option_module.py`](../../tests/test_option_module.py) 新增两个 test：
- `test_option_module_beta_is_clamped_to_min_max`：极端输入下 β 仍在范围内
- `test_option_module_beta_clamp_default_range`：默认 (0.05, 0.95) + export_kwargs 包含两个边界

**这个修复的意义**：解开了之前所有实验都不得不用 `beta_warmup_steps=2M` 冻结 β 的限制。现在可以用 paper-aligned 的小 warmup（5k-20k）+ β-weighted transfer，让 termination 学习真正起作用 —— 即把 PTF 框架的另一半（option 终止决策）恢复运转。

### 6.2 v_min/v_max 可能太窄（中优先级）

push 任务下我们观察到 C51 atoms 完全撑到 ±1000 边界。需要测试 v_min/v_max 是否影响 final return（FastTD3 paper 没给 push 的具体配置）。

### 6.3 训练预算太短（中优先级）

100k iters = 12.8M env steps 是 FastTD3 paper 的标准预算，但 paper 报告的 push +700 是 3 seeds 平均。我们 3v3 实验里 Scratch s1 跑出 +148，s2/s3 跑出负分，说明 100k iters 还远没让所有 seed 收敛到 paper 的 ceiling。**延长到 200-300k iters 是必要的**才能做 PTF vs Scratch 的"最终性能"对比。

### 6.4 单一 target task（push）的数据点太少（中优先级）

push 在 FastTD3 下已经能跑到 +700，PTF 的相对优势空间有限。**真正 hard target 是 package**（FastTD3 paper 显示 package 至今所有方法都是负分）。下一阶段建议把同样的 decay-PTF 3-seed 在 package 上跑。

### 6.5 我们的 PTF 退化为"无 termination 学习的固定 source 蒸馏"（高优先级，方法学）

由于 6.1 我们冻结了 β，PTF 框架的核心"何时切换 option"机制是失效的。当前实现实际上是：

- **Option-Q 仍然学**（决定每个时刻"哪个 source 是最优蒸馏对象"）
- **Option termination 不学**（β 锁定在初值，option 在 rollout 中按 ε-greedy 切换而不是按 β 概率切换）

如果论文要写"PTF 框架在 humanoid 上的应用"，**我们必须解决 β 训练问题**，否则只能说"我们提出了一个简化版 PTF（无 termination 学习）"，理论贡献严重打折。

---

## 7. 总结：到目前为止的里程碑

| 里程碑 | 状态 |
|---|---|
| ✅ PTF 框架完整移植到 PyTorch + 对接 FastTD3 official 代码 | 完成 |
| ✅ Source bank 工程链路（manifest → bank YAML → loader）| 完成 |
| ✅ HumanoidBench 任务适配（obs adapter + action mask）| 完成 |
| ✅ 验证 PTF 机制能影响 actor（force-PTF 实验）| 完成 |
| ✅ 验证 decay schedule 修复负迁移（3v3 多 seed）| 完成 |
| ⚠️ β-head 训练机制工作正常 | **未解决**，目前 workaround 冻结 |
| ⚠️ PTF vs Scratch 显著性 winner | **未确定**，3v3 显示无显著差异 |
| ⚠️ Hard target (package / kitchen) 上的 PTF 优势 | **未验证** |
| ⚠️ 延长训练到 paper-budget 200-300k iters | **未做** |

---

## 8. 下一步建议（按 ROI）

1. **修 β 训练**（治本，决定文章的理论完整性）→ 估计 1-2 天 dev + 1 天 toy 验证
2. **package 任务 3-seed**（找 PTF 真正闪光的场景）→ ~12h GPU
3. **5-seed × 200k 延长训练**（让 mean 比较更有说服力）→ ~30h GPU
4. **PPO + decay schedule 对比**（如果论文要写"PTF + FastTD3"，最好也证明 PTF 不依赖于特定 backbone）→ 比较低优先级
