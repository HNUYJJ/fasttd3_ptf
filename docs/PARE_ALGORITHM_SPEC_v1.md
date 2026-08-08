# PARE 算法规格 v1

**Provenance-Aware Release and Expansion**

冻结日期：2026-08-07 · 状态：实现前冻结（先于任何 PARE 运行结果）

---

## 0. 问题、贡献边界、与近邻工作的区分

**问题**：跨任务 source policy 在 target 环境中临时代管行为，能把 student 早期带入有用区域；
但它同时把学习数据限制在 source 支持的状态—动作分布附近。source 退出（release）之后，
student 保有 release 时刻的能力，却仍在 source 塑造的 occupancy 里打转。
**PARE 要解决的是：release 之后，如何在不丢失既有能力的前提下，主动解除 source 曾施加的 occupancy 约束。**

**贡献边界（必须写进论文，不得越界）**：

| 组件 | 是否本文贡献 |
|---|---|
| source 在 target 环境执行动作 + target reward transition + off-policy replay | 否，是既有 scaffold 路径 |
| 固定 B 步后 hard exit | **否，是 baseline**。"是个人都想得到" |
| 用 discriminator 估 density ratio | 否，标准 density-ratio estimation |
| 梯度投影本身 | 否，PCGrad (Yu et al., NeurIPS 2020) 已研究 conflicting gradient projection |
| **把 source-provenance occupancy repulsion 定义为 post-release RL 的第二目标，并以 target-value gradient 为优先目标对其做保护** | **是** |

**与三条近邻工作的区分**：

- **PTGOOD**（offline-to-online，寻找 behavior policy 少访问但 reward 高的 OOD 区域）：
  它从 *dataset support* 向外探索。PARE 的排斥对象不是 dataset support，
  而是**一个临时介入 target 环境、随后被撤销的跨任务行为策略所留下的 occupancy**。
  provenance 标签在 PTGOOD 里不存在，在 PARE 里不可缺。
  → 因此论文**不得**写成"一种新的 OOD exploration"。
- **OMPO**（policy/dynamics shift 下的 transition occupancy *matching*）：方向相反。
  OMPO 要对齐分布，PARE 要在恢复自主性后**有控制地反向偏离** source occupancy。
- **OPT**（offline-to-online 后 value function 本身成为瓶颈）：提醒我们 `num_updates` 充分性
  必须作为 falsification control（见 §8），但不是首先要跑的。

论文切入点定为：**Autonomy recovery after temporary cross-task policy scaffolding.**

---

## 1. provenance 标签 $z$ 的定义

训练循环已逐 transition 记录 provenance（`ptf_replay.py:71-80`，
写入点 `train_ptf.py:3037-3039`）。字段 `source_by_group` 形状 `[n_env, buffer, n_groups]`，
`-1` 表示该身体组由 student 驱动；`executed_group_mask = (source_by_group >= 0)`。

定义：

$$
z_t = \mathbf{1}\left[\exists\,g:\ \texttt{executed\_group\_mask}[t, g] = \text{True}\right]
$$

即**任一身体组实际由 source 执行**，则该 transition 记为 source-provenance（$z=1$）；
全部身体组均为 student 时 $z=0$。

采用 `executed_group_mask` 而非 `behavior_source`，因为后者是 option id，
null option 的取值语义与"student 执行"不可靠地重合。
provenance 仅在 `rb.enable_provenance()` 被调用后可用（`train_ptf.py:1390`，MCG 路径），
PARE 启动时必须断言其已启用且 `provenance_written` 完整，否则 fail-closed 退出。

---

## 2. Release 时刻 $B$ 保存什么

release 时执行四件事，全部一次性，之后**不再需要 source policy**：

1. **冻结 competence anchor** $\pi_B \leftarrow \text{deepcopy}(\pi_\theta)$，置 `eval()`，
   全部参数 `requires_grad_(False)`，不进任何 optimizer。
2. **source 永久退出**：`admission_mode=none`，行为策略恒为 $\pi_\theta$。
3. **构建固定 source reservoir** $\mathcal{D}_S$：扫描 replay 中当前仍存在的全部 $z=1$ 的
   $(s, a)$，存 **raw obs**（与 replay 一致，`train_ptf.py:3235` 显示 replay 存 raw、
   采样后才 normalize），容量上限 `pare_reservoir_capacity`（默认 262144）。
   超出上限时**均匀无放回子采样**，并把实际候选数与保留数写进日志与 checkpoint。
4. **记录 release 快照**：obs normalizer 的 state_dict、$B$、reservoir 实际大小。

> **实现决定 D1（偏离 ChatGPT 表述，须 PI 知晓）**：release 时**记录**但**不使用**冻结的
> obs normalizer。$\pi_B$ 与 $\pi_\theta$ 在 PARE 计算中共用**当前**的 normalizer。
> 理由：anchor advantage 的语义是"当前 critic 认为 $\pi_\theta$ 是否不比 $\pi_B$ 差"，
> 这要求两者在**同一个 normalized $s$** 上被同一个 $Q$ 评估。若给 $\pi_B$ 配一个冻结的
> normalizer，就人为制造了 $\pi_B$ 与 $\pi_\theta$ 输入坐标系不一致，
> 使 $\hat A_B$ 的符号混入 normalizer 漂移而非策略差异。
> 快照仅用于可复现与事后诊断。

---

## 3. Discriminator

小 MLP $D_\phi(s,a) \in (0,1)$：输入 `concat(normalize_obs(s), a)`，隐层 `[256, 256]`，
ReLU，输出单 logit。独立 Adam，`lr=3e-4`。**不用 Transformer / VAE / flow / ensemble。**

balanced BCE，正负 batch 等大：

$$
\mathcal{L}_D = -\mathbb{E}_{(s,a)\sim \mathcal{D}_S}\log D_\phi(s,a)
\;-\;\mathbb{E}_{(s,a)\sim d_{\pi_t}}\log\bigl(1 - D_\phi(s,a)\bigr)
$$

- **正样本**：从固定 reservoir $\mathcal{D}_S$ 均匀采样，raw obs 经**当前** normalizer 转换。
- **负样本**：从主 replay 采样，**必须过滤掉残留的 $z=1$ 条目**。
  若不过滤，同一批数据既作正例又作负例，$D$ 无法收敛。
  release 后新写入的 transition 恒为 $z=0$，残留的 $z=1$ 会随覆盖逐步消失。
  若某步过滤后负样本数为 0，跳过该步 $D$ 更新并计数（日志字段 `pare/d_skip_count`）。

**最优解**（先验平衡下）：

$$
D^*(s,a) = \frac{d_S(s,a)}{d_S(s,a) + d_{\pi_t}(s,a)}
\quad\Longrightarrow\quad
\operatorname{logit} D^*(s,a) = \log\frac{d_S(s,a)}{d_{\pi_t}(s,a)}
$$

**$D$ 在线更新，不冻结。** $\pi_t$ 改变导致 $D$ 改变不是 estimator failure：
$D$ 一直在重新回答"当前 student 相对于**固定的**过去 source occupancy 已经走出去多少"。
分母随 student 演化，分子是 release 时刻冻结的 $d_S$——这正是我们要的物理量。

---

## 4. Anchor advantage

FastTD3 的 critic 是 distributional 双头。conservative value 取

$$
Q_L(s,a) = \min\bigl(Q_1(s,a),\, Q_2(s,a)\bigr)
$$

（`train_ptf.py:1905-1907` 已有 `get_value(softmax(qf))` 两路，直接取 min；
**不引入新的 uncertainty 系数 $\beta$**——v0 草案里的 $Q^- = \min - \beta|Q_1-Q_2|$ 已删除。）

$$
\hat A_B(s) = Q_L\bigl(s, \pi_\theta(s)\bigr) - Q_L\bigl(s, \pi_B(s)\bigr),
\qquad
m(s) = \operatorname{stopgrad}\Bigl(\mathbf{1}\bigl[\hat A_B(s) \ge 0\bigr]\Bigr)
$$

$m$ 是 competence gate：**只有当前 target critic 认为 $\pi_\theta$ 在该状态上不比 release
policy 差时，才允许 expansion 目标对 actor 产生影响。**

---

## 5. Expansion objective

$$
J_E(\theta) = \mathbb{E}_{s\sim \text{replay}}
\Bigl[\, m(s)\,\log\bigl(1 - D_\phi(s, \pi_\theta(s))\bigr) \Bigr]
$$

最大化 $J_E$ 即压低 $\pi_\theta$ 所选动作被判为 source-provenance 的概率——
在 §3 的 density-ratio 意义下，等价于把 $\pi_\theta$ 的 occupancy 推离 $d_S$。
$D_\phi$ 在此处**不接收梯度**（其参数 `requires_grad` 在 actor 更新中被临时关闭，
或对 $\phi$ 侧 detach）。

实现恒等式：令 $z$ 为 logit，则

$$
\log\bigl(1 - \sigma(z)\bigr) = -\operatorname{softplus}(z),
\qquad
\frac{\partial}{\partial z}\bigl[-\operatorname{softplus}(z)\bigr] = -\sigma(z) \in (-1, 0)
$$

**不得对 $z$ 做 clamp。** 梯度本身有界且方向正确：$z\to+\infty$（样本最像 source，
正是最该推离处）时梯度趋于 $-1$；$z\to-\infty$（已不像 source）时趋于 $0$。
`softplus` 自身数值稳定，截断只会制造死区。

> **v1.1 修订（2026-08-07，smoke 实测后）**：v1.0 曾写 `clamp(-10,10)`。
> 2k smoke 实测 `d_logit_clamp_rate` 冲到 **0.722**——D 在 release 后迅速达到
> `d_acc=1.0`，logit 大量越过 ±10，被截断的样本梯度归零，
> **恰好把最需要 expansion 的那部分样本全部静音**。这与
> `docs` 里记录的 β-clamp logit 死区是同一类错误：护住值域、杀死梯度。
> 现改为不截断，仅保留 `pare/d_logit_saturated_rate` 作诊断。
> 该修订发生在任何 PARE 科学结果产出**之前**，不涉及判据（§8 F1–F7 未动）。

---

## 6. Gradient projection（无超参数的核心）

**不使用** $\kappa$、decay schedule、support threshold、$\lambda$、$\epsilon_Q$。
两个目标的合成完全由几何决定。

令 $\theta$ 为 actor 参数，ascent 梯度：

$$
g_Q = \nabla_\theta J_Q,\qquad J_Q = -\bigl(\texttt{rl\_actor\_loss} + \texttt{transfer\_loss}\bigr)
$$
$$
g_E = \nabla_\theta J_E
$$

**步骤 1 — 冲突投影**：

$$
\tilde g_E =
\begin{cases}
g_E - \dfrac{\langle g_E, g_Q\rangle}{\lVert g_Q\rVert^2}\, g_Q, & \langle g_Q, g_E\rangle < 0\\[2mm]
g_E, & \text{否则}
\end{cases}
$$

**步骤 2 — norm cap**（expansion 不得盖过 base RL）：

$$
\bar g_E = \tilde g_E \cdot \min\!\left(1, \frac{\lVert g_Q\rVert}{\lVert \tilde g_E\rVert + \epsilon}\right)
$$

**最终**：

$$
\boxed{\,g_{\mathrm{PARE}} = g_Q + \bar g_E\,}
$$

写回 `p.grad = -g_PARE`（optimizer 做 descent），再走既有的 `clip_grad_norm_`。

### Lemma 1（一阶不冲突）

若 $g_Q \neq 0$，则 $\langle g_Q,\, g_{\mathrm{PARE}}\rangle \ge \lVert g_Q\rVert^2 > 0$。

*证明*：$\langle g_Q, g_{\mathrm{PARE}}\rangle = \lVert g_Q\rVert^2 + \langle g_Q, \bar g_E\rangle$。
若 $\langle g_Q,g_E\rangle \ge 0$，则 $\bar g_E = c\,g_E$（$c\in(0,1]$），故 $\langle g_Q,\bar g_E\rangle = c\langle g_Q,g_E\rangle \ge 0$。
若 $\langle g_Q,g_E\rangle < 0$，则 $\langle g_Q, \tilde g_E\rangle = \langle g_Q,g_E\rangle - \langle g_E,g_Q\rangle = 0$，
乘正标量后仍为 0。两种情形均有 $\langle g_Q,\bar g_E\rangle \ge 0$。∎

**推论（45° 锥）**：记 $t = \langle g_Q, \bar g_E\rangle \ge 0$，$e = \lVert\bar g_E\rVert \le \lVert g_Q\rVert$。
取 $\lVert g_Q\rVert = 1$，则

$$
\cos\angle(g_Q, g_{\mathrm{PARE}}) = \frac{1+t}{\sqrt{1 + 2t + e^2}}
\;\ge\; \frac{1}{\sqrt{2}}
$$

（右端在 $t=0,\ e=1$ 取到）。即 **PARE 的实际更新方向恒落在 base RL 更新方向的
45° 锥内**。`tests/test_pare.py` T3 以 300 次随机试验实测最小 $\cos = 0.707107$，
与该界吻合。

这取代了 v0 草案里依赖未知 critic 误差 $\epsilon_Q$ 的 Proposition 2。
Performance Difference Lemma 此后**仅用于解释** anchor advantage 为何是合理的 competence
reference，**不声称** deep critic 下的 monotonic-return 保证。

### AMP 齐次性（实现正确性关键）

现有代码用 `GradScaler`。若用同一 scale 因子 $c>0$ 求两个梯度，则：
$\operatorname{sign}\langle cg_Q, cg_E\rangle = \operatorname{sign}\langle g_Q,g_E\rangle$；
投影式对 $c$ **一次齐次**；norm cap 的比值 $\frac{c\lVert g_Q\rVert}{c\lVert\tilde g_E\rVert}$ 与 $c$ **无关**。
故 $c\,g_{\mathrm{PARE}}$ 恰为 scaled 结果，**unscale 一次即可，无需对两路分别处理**。

---

## 7. 伪代码

```
# ---- release（一次性，step == B）----
pi_B      = frozen_copy(actor)
D_S       = subsample({(s_raw, a) in replay if z == 1}, cap=262144)
D_phi     = MLP(obs_dim + act_dim -> 1);  opt_D = Adam(D_phi, 3e-4)
admission_mode = "none"                     # source 永久退出

# ---- 每个 actor 更新步（step > B）----
def pare_actor_update(batch):
    # (a) discriminator
    pos = normalize_obs(sample(D_S))                     # 固定 source occupancy
    neg = batch[batch.z == 0]                            # 必须过滤残留 z=1
    if len(neg) > 0:
        L_D = BCE(D_phi(pos), 1) + BCE(D_phi(neg), 0)
        opt_D.zero_grad(); L_D.backward(); opt_D.step()
    else:
        pare_d_skip_count += 1

    # (b) 两个目标
    s   = batch.observations                             # 已 normalize
    a_t = actor(s)
    J_Q = -(rl_actor_loss(s, a_t) + transfer_loss)       # ascent 目标
    with torch.no_grad():
        Q_theta = Q_L(s, a_t);  Q_B = Q_L(s, pi_B(s))
        m       = (Q_theta - Q_B >= 0).float()           # stopgrad competence gate
    J_E = ( m * log(1 - D_phi(s, a_t).detach_phi()) ).mean()

    # (c) 梯度合成（同一 GradScaler 因子，见 §6 齐次性）
    g_Q = autograd.grad(scaler.scale(J_Q), actor.params, retain_graph=True)
    g_E = autograd.grad(scaler.scale(J_E), actor.params)
    if dot(g_Q, g_E) < 0:
        g_E = g_E - dot(g_E, g_Q) / dot(g_Q, g_Q) * g_Q
    g_E = g_E * min(1, norm(g_Q) / (norm(g_E) + 1e-12))
    write_grads(actor, -(g_Q + g_E))                     # optimizer 做 descent
    scaler.unscale_(actor_opt); clip_grad_norm_(actor);  scaler.step(actor_opt)
```

`pare_enabled == False` 时**完全不进入以上任何分支**，代码路径与现有 FastTD3 hard-exit 逐位一致。

---

## 8. 证伪条件（先于结果冻结）

**PARE 唯一核心成功标准：相同 target interactions 下显著优于 fixed hard exit。
只优于 continuous source 不算成功。**

以下任一成立即判 PARE 失败或解释错误：

| # | 现象 | 结论 |
|---|---|---|
| F1 | Experiment A 显示两个任务都**没有** early scaffold gain + post-exit residual headroom | 不存在待解决的 post-release 问题，**诚实关闭 PARE，不硬造算法** |
| F2 | 相同 target interactions 下 PARE $\not>$ fixed hard exit（learner 间方差尺度，非 episode SE） | 主张失败 |
| F3 | return 提升了，但 source affinity $\mathbb{E}[D_\phi(s,\pi(s))]$ **没有**下降 | 机制解释错误——收益不来自 occupancy 扩展 |
| F4 | 消融 "PARE without provenance"（把 $D_\phi$ 换成 generic novelty，不用 $z$）同样好 | provenance 不是必要成分，贡献不成立 |
| F5 | 消融 "source-repulsion without value protection"（去掉 $m$ 与投影）同样好 | value protection 不是必要成分 |
| F6 | hard exit 只把 `num_updates` 从 2 提到 4 就追平 PARE | 贡献受严重挑战（plateau 只是更新不足，见 OPT） |
| F7 | gradient conflict rate $P(\langle g_Q,g_E\rangle<0) \approx 0$ 全程 | 投影从未生效，PARE 退化为纯加性 bonus，须重新解释而非邀功 |

**记录的机制量**（除 return 外，只记这四个）：

- `pare/source_affinity` $= \mathbb{E}[D_\phi(s,\pi_\theta(s))]$
- `pare/anchor_adv_positive_frac` $= P(\hat A_B \ge 0)$
- `pare/grad_conflict_rate` $= P(\langle g_Q, g_E\rangle < 0)$
- `pare/expansion_norm_ratio` $= \lVert\bar g_E\rVert / \lVert g_Q\rVert$

目标是能画出 **source affinity 下降 → policy expansion → return 提高** 的因果链，
而不是只有一条 reward 曲线。

---

## 9. 与现有代码的接口

| 需求 | 现有支撑 | 状态 |
|---|---|---|
| provenance 逐 transition 标签 | `ptf_replay.py:71-80`、`train_ptf.py:3037-3039` | 已具备，无需改 schema |
| source 永久退出 | `PTF_ADMISSION_MODE=none`（`run_slide_hard_exit_v1.sh:65`） | 已具备 |
| release 处分叉 | `PTF_BRANCH_ANCHOR_STEP` / `PTF_ANCHOR_RESUME` | 已具备 |
| 定点停止 | `PTF_RUN_STOP_STEP` | 已具备 |
| $Q_L=\min(Q_1,Q_2)$ | `train_ptf.py:1905-1907` | 已具备 |
| actor 更新插入点 | `update_pol()`，`train_ptf.py:1899` | 新增分支 |

**新增模块（只此四个）**：`SourceTransitionReservoir`、`SourceOccupancyDiscriminator`、
`FrozenReleaseActor`、`pare_actor_update`。
**不新增**：selector、threshold scheduler、VAE/flow/ensemble、新 proxy、新 inventory。

---

## 10. 实现决定（须 PI 知晓）

- **D1**：$\pi_B$ 与 $\pi_\theta$ 共用当前 obs normalizer，release 快照只作记录。理由见 §2。
- **D2**：reservoir 有容量上限（默认 262144）并均匀子采样。
  30k prefix × 128 env 下 $z=1$ 候选可达数十万条，全量常驻显存不现实。
  实际候选数与保留数必须落盘，**不得静默截断**。
- **D3**：discriminator 的负样本过滤掉残留 $z=1$；负样本为空时跳过该步 $D$ 更新并计数。

---

## 11. 设计层自检（CLAUDE.md §8）

- **8.1 辨别力**：平凡解释是"任何朝低密度区的扰动都等价于额外探索噪声"。
  由 F4（generic novelty 消融）与 F5（去 value protection 消融）排除；
  若二者任一同样好，PARE 的特定形式即无必要。
- **8.2 混淆变量**：PARE 臂与 hard-exit 臂从**同一 release anchor** 分叉，
  共享 actor / critic / replay / source history / release state，
  唯一差别是 post-release 的 expansion 更新。source 剂量固定为 10k，**不调**。
- **8.3 独立重复**：pilot 用 1 seed 定方向，成立后扩到 **3 个新 learner seeds**（非同 seed 重跑）。
- **8.4 前提是否蕴含结论**：Experiment A 的通过条件是
  "early scaffold gain **且** post-exit residual headroom"。
  它**不**蕴含 "PARE > hard exit"——headroom 存在与 PARE 能否吃到 headroom 是两个独立命题。
  A 通过只解锁 B，不预判 B 的方向。
- **8.5 site selection**：stair 的选场理由是旧 Slide→Stair 20k 三 seed 全正
  （+21.91 / +13.23 / +11.05），但**旧结果存在 source-dose confound**，
  故它只作**选场理由**，不作论文证据；实验须 fresh 跑。
  truck 用此前已冻结、已有正迁移证据的 hurdle-enhanced bank，**不重新选源**。
  开发任务只允许一个，另一个冻结为 holdout，**不得依结果调 PARE**。
- **8.6 是否重演本轮教训**：本轮教训是"审计膨胀"与"estimand 没变就是换皮"。
  PARE 的 estimand **已变**：从"预测/测量反事实延迟学习效用 $U$"
  变为"改变 post-release 的 occupancy 与可达 return"。
  它不选源、不预测迁移性、不给剂量打分——不属于被禁的十二族。
- **8.7 判据切换红线**：§8 的 F1–F7 在任何 PARE 结果产出**之前**冻结于本文件；
  之后只允许改路径参数，不得动判据逻辑。

---

## 12. 执行顺序

1. 本 spec 冻结（本文件，先于实现提交）。
2. 实现四个模块 + `--ptf_pare` flag。
3. **2k smoke**，只验四项：provenance 两类都存在、$D$ 输出有限、actor 梯度有限、
   PARE-off 与 baseline 一致。通过即停止工程验证，**不扩建**。
4. 与 2–3 并行：Experiment A（stair / truck × {scratch, hard-exit} × 3 seeds = 12 条）。
5. A 通过者进入 branch-at-release pilot（1 seed），方向正确后扩 3 seeds。

**当前禁止**：任何新 inventory、全库 checkpoint 扫描、批量旧 checkpoint 重评、
selector、exit threshold 搜索、B×R 实验、新 proxy。
