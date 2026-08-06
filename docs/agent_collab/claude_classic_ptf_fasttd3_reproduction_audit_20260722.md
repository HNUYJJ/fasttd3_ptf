# Classic PTF × FastTD3 独立复现审计(Claude,2026-07-22)

> 审计范围:`fasttd3_ptf/official_fasttd3_ptf/train_ptf.py` 经典 PTF 主路径 +
> `classic_ptf_hurdle_single_source_v1` 实验(learned/fixed/scratch × 3 seeds)。
> 方法:先从一手论文(`papers/PTF-arxiv.pdf`、`papers/FastTD3.pdf`)与官方源码
> (`reference_source_code/PTF_code/`、`fasttd3_ptf/official_code/FastTD3/fast_td3/`)
> 独立重建算法语义,再逐模块对照本项目实现,最后独立重算实验数字。
> 本轮为只读审查:未修改任何代码,未启动任何训练。

---

## 1. Executive verdict

1. **是否正确复现了 PTF 论文?** 结构层面忠实(student 执行环境、option 只作蒸馏
   调度、call-and-return、intra-option Q_o、β·(A+ξ) termination、(1−β) 蒸馏权重),
   但存在多处已声明的适配与两处**实测已使机制失效的参数/动力学问题**(见 H1/H2);
   应称为"结构适配版复现",不是逐式复现——项目文档自己的定位与此一致。
2. **是否正确复现了 PTF 官方代码?** 官方代码本身就与论文不一致(官方 transfer loss
   **没有** (1−β) 项;ξ 是自适应 0.8·(top1−top2) 而非论文的 0.001)。本项目是
   混合体:(1−β) 跟论文、ξ 跟官方代码。混合选择在代码注释与结果文档中均有声明,
   不构成隐藏偏差。
3. **PTF→确定性 FastTD3 的适配是否合理?** 蒸馏(cross-entropy→masked Huber)、
   兼容性(μ±1σ 区间→高斯核)、off-policy 化(on-policy rollout→replay batch)三项
   适配方向均可辩护;但 `compatibility_sigma=1.5` 的取值使兼容度实测恒 ≈0.8(官方
   语义下同样动作距离的兼容度 ≈e⁻⁸≈0),**Q_o 的 intra-option 选择性被实际取消**;
   termination 在"adaptive-ξ + ε-greedy argmax + 2 options"组合下被系统性下压至
   β 下轨 0.05,**learned termination 全程无信号**。这两点使"learned 调度"臂实际
   测的不是设计中的机制。
4. **当前实验结果是否可信?** 数字可信:我对 19 点 eval 曲线、梯形 AUC、paired
   delta、32-episode source-free 终点全部独立重算,与
   `classic_ptf_hurdle_single_source_v1_results.md` **逐位一致**(误差 <0.01)。
   结论边界总体谨慎;需收窄/加注的主张见 §7。

---

## 2. 算法基准(从一手资料独立重建)

### 2.1 PTF(论文,PTF-arxiv.pdf)

- **目标/假设**:多个冻结 source policy(与 target 同状态-动作空间,或至少部分共享)
  加速 target 任务学习;不要求任何 source 在目标任务上最优。
- **环境执行者**:**永远是 student**(Algorithm 1 line 8:`a ~ π(s|θ')`)。source
  action 不进环境、不进 replay;这是 PTF 与 CAPS 类方法的核心区别(论文 §5.1 明确)。
- **Option 模块**:Q_o(s,o) + β(s,o) 两个 head 共享输入与隐藏层。ε-greedy 选 option;
  call-and-return:option 保持到 β(s′,o) 采样触发终止,再重选;episode 开始重选。
- **Q_o 更新(Alg.2 + Eq.3)**:intra-option learning——对每个"在 s 会选出动作 a"
  的 option o(连续动作:a 落在 π_o 分布的置信区间内),用
  `y = r + γ[(1−β(s′,o))Q′(s′,o) + β(s′,o)·max_o′ Q′(s′,o′)]` 做 TD;replay batch,
  off-policy;target 网络周期硬拷贝。
- **Termination 更新(Eq.4/5)**:`θ_β ← θ_β − α_β ∇β(s′,o)·(A(s′,o)+ξ)`,
  `A = Q(s′,o) − max_o′ Q(s′,o′) ≤ 0`,在 **next state s′** 上;A 视为常数(detach)。
  语义:β 是**终止概率**;非最优 option β↑,最优 option β 被 ξ 轻微下压。
- **蒸馏(Eq.6/7)**:policy-based 用 cross-entropy `H(π_o(·|x) ‖ π_s(·|x,θ))`,
  value-based 用带温度的 softmax-Q KL;权重 `f(β_o,t) = f(t)·(1−β(s_t,o))`,
  `f(t) = (1+tanh(3−0.001t))/2`;teacher = **当前激活 option** 的 source,输入是
  **同一 target 状态**;loss 加到 actor 目标上,只影响 policy 参数。
- **β 的双重使用**(切换 + 蒸馏权重)是论文机制(Eq.7),非本项目发明。

### 2.2 PTF(官方代码,与论文的实测差异)

| # | 论文 | 官方代码(PTF_A3C.py / PTF_PPO.py) |
|---|---|---|
| 1 | transfer 权重含 (1−β)(Eq.7) | **无 (1−β)**:`entropyTS = Σ·weight·c1`;`(1−t)` 版本留在注释里(PTF_A3C.py:78) |
| 2 | ξ = 0.001 固定(Table 4) | `xi==0` 时 **adaptive ξ = 0.8·(top1Q−top2Q)**(PTF_A3C.py:270-271);发布配置 xi=0 |
| 3 | ε-greedy:探索率 1.0→0.05 衰减 | `epsilon` 是 **greedy 概率** 0→0.9 递增(语义等价,参数表述相反) |
| 4 | 未提 Q_ω 值域 | Q_ω 输出层 **tanh(值域 [−1,1])** + reward 归一化 r/done_reward |
| 5 | U 的 max 用 target 网络 | **double-Q**:online argmax → target 取值(PTF_A3C.py:260-261) |
| 6 | termination 与 Q_o 同步更新 | termination 只用**当前 on-policy transition 单样本**且 `if not done`;Q_o 用 replay batch |
| 7 | — | 连续动作兼容性 = 逐维 `μ±1σ` 硬区间(全维都在区间内→兼容,source_actor.py:181-198);官方超参 c1=0.001、lr_o=lr_t=5e-4、option_batch=32 |

### 2.3 FastTD3(论文 + 官方代码,二者一致)

TD3 + 并行环境(HB:SubprocVecEnv×128)+ 大 batch(32768)+ C51 distributional
critic(hurdle:v_min/v_max=±250、101 atoms)+ CDQ(min)+ per-env 混合探索噪声
σ∈[0.001,0.4](done 时重采样)+ EmpiricalNormalization 观测归一化(HB 无 reward
归一化)+ AdamW(wd=0.1)+ 余弦 LR + AMP bf16 + torch.compile + replay
N×num_envs(HB:400×128=51200)+ num_updates=2/policy_frequency=2(每 env 步
2 次 critic、1 次 actor 更新)+ soft target τ=0.1。`Actor.forward` = 确定性 tanh μ。
本项目 vendored 快照 `fasttd3_ptf/official_code/FastTD3/fast_td3/` 与
`reference_source_code/FastTD3` 逐文件一致,确认为官方固定快照。

---

## 3. 论文—官方代码—本项目三方对照表

判定:MATCH / PRINCIPLED_ADAPTATION(PA)/ INTENTIONAL_EXTENSION(IE)/
SEMANTIC_MISMATCH(SM)/ BUG / UNVERIFIED。

| 机制 | 论文 | 官方代码 | 本项目(文件:行) | 判定 |
|---|---|---|---|---|
| 环境执行者 = student | ✓ | ✓ | `train_ptf.py:2004`(execute_sources=false 时唯一动作源) | **MATCH** |
| source 动作仅用于蒸馏/兼容性 | ✓ | ✓ | `act_all/act_selected` 全部 `@torch.no_grad`,不进 env/replay | **MATCH** |
| source 冻结 | ✓ | ✓ | `source_policy.py:158-160`(eval + requires_grad_(False),不入任何 optimizer) | **MATCH** |
| source 输入链 | 同一 target 状态 | ✓ | raw obs → obs_adapter → **source 自带冻结 normalizer** → actor(`source_policy.py:178-183`);target normalizer 不污染 source 输入(`train_ptf.py:2333` 先 clone raw) | **MATCH** |
| call-and-return 锁存 | β(s′) 采样终止→ε-greedy 重选 | ✓ | `option_selector.py:52-68`;在下一步开始时用当步 obs 判断,与官方"步末用 s′ 判断"数学等价;done 强制重选 | **MATCH** |
| episode 初始 option | ε-greedy 选择 | ✓ | 初始/reset 置 null,等 β 触发再选(`option_selector.py:37-49`) | PA(轻微,影响前 ~10 步) |
| Q_o 结构 | 共享干+双 head | 共享层+tanh Q | `option_module.py:62-64` 共享 trunk,**Q 无界线性 head** | PA(HB reward 未归一化,连带 M3) |
| β 值域 | sigmoid [0,1] | sigmoid [0,1] | **rescale 到 [0.05,0.95] + bias init −2**(`option_module.py:76-77`) | **IE**(防饱和;副作用见 H2) |
| Q_o intra-option 更新 | 兼容样本更新多 option | μ±1σ 硬区间 opa mask | 高斯软核 `exp(−d̄²/2σ²)`(`compatibility.py:38-50`)+ executed option 强制 1 + null 恒 1 | PA 结构 / **SM 取值**(σ=1.5 实测使 compat≈0.8 恒真,见 H1) |
| Q_o 目标 U | (1−β)Q′+β·maxQ′,β=online | β=online,max=double-Q | β 与 max **均用 target 网络**(`train_ptf.py:1469-1474`) | SM(轻微,M2) |
| Q_o 损失归一化 | — | batch mean(mask 内 sum) | `/compat.sum()`(`train_ptf.py:1508`) | PA(有效 lr 随 compat 波动,M4) |
| termination 目标 | β(s′,o)·(A+ξ),on-policy | 同,单 transition | β(s′,o)·(A+ξ) 在 **replay batch + 存储 option** 上(`option_update.py:36-83`,`train_ptf.py:1521`) | PA/SM(off-policy 化已在 docstring 声明,M1) |
| termination 对称 clamp | 无 | 无 | advantage clamp 到 [−margin,+margin](`option_update.py:66-68`) | **IE**(2-option 时数学上不激活,见 H2 分析) |
| adaptive ξ | 0.001 固定 | 0.8·(top1−top2) | 同官方(`option_update.py:30-33`),xi=0 默认 | MATCH(对官方)/ SM(对论文) |
| β warmup | 无 | 无 | `beta_warmup_steps`(本实验=0,未生效) | IE(本实验关闭) |
| (1−β) 蒸馏权重 | ✓(Eq.7) | **✗(注释掉)** | ✓ `transfer_gate = 1−β_o`(`train_ptf.py:1277`),β detach,在 s_t 上取 | MATCH(对论文)/ 比官方多 |
| 时间衰减 f(t) | (1+tanh(3−0.001t))/2 | 同 ×c1=0.001 | **线性 λ:1→0/100k**(`LinearScheduler`,`train_ptf.py:833-837`) | PA(形状不同;λ_start=1.0 vs c1=0.001 不可直接比较——损失量纲不同,见 UNVERIFIED) |
| 蒸馏损失形式 | cross-entropy(随机策略) | Gaussian cross-entropy | **masked Huber 动作匹配**(`distillation.py:18-26`) | PA(确定性策略下 KL 于同方差高斯 ∝ MSE;Huber 为稳健版) |
| 蒸馏数据 | on-policy rollout 状态 | ✓ | **replay batch(off-policy)+ 存储时的 option id**(`train_ptf.py:1258-1262`) | PA/SM(FastTD3 无 on-policy batch,必要适配;引入教师-状态错位) |
| transfer loss 位置 | 加进 actor 梯度 | ✓ | 仅 actor 更新步,`actor_loss = rl + transfer`(`train_ptf.py:1400`);梯度只达 actor(source no_grad、β detach、critic 不在 actor optimizer) | **MATCH** |
| no-transfer/null option | **无** | **无** | 有(`source_bank.py:20-21`,β/Q 照常学,兼容度恒 1) | **IE**(scratch-arm/不迁移所需) |
| FastTD3 critic/actor/target/AMP/scheduler | — | 基准 | `update_main`/`update_pol` 与官方 train.py **逐行一致**(仅 actor loss 加 transfer 项) | **MATCH** |
| replay 语义 | — | randint+gather, n=1 | `PTFReplayWrapper` 镜像官方 n=1 分支,加挂 option id;经典路径 `draw_indices` = 同一 `torch.randint` 原语(`ptf_replay.py:512-519`) | **MATCH** |
| 环境 wrapper | — | 基准 | 仅改种子链(worker 双播种修复),step/reset/truncation 语义一致 | PA |
| scratch = 官方 FastTD3 | — | 基准 | 空 bank → target_only 快路径 + RNG capture/restore(`train_ptf.py:808,1104-1110`);Gate A 等价测试在 `tests/test_p1_gate_a.py` | **MATCH**(结构+RNG 级对齐) |
| optimizer 隔离 | — | 双 optimizer 均持共享层 | option/beta 两个 Adam 均注册全部 option 参数(`train_ptf.py:831-832`),梯度经 zero_grad(set_to_none) 隔离;**与官方同构** | MATCH |
| option 目标网络更新 | 周期硬拷贝(1000 步) | 同 | soft update τ=0.05(`train_ptf.py:1550-1551`) | PA |
| 更新频率 | 每 env 步 1 次 | 同 | 每 env 步 num_updates=2 次(跟随 FastTD3 UTD) | PA |

**结论:未发现会让"环境中执行的动作、replay 内容、FastTD3 基线语义、source-free
评估"出错的 BUG。** 语义问题集中在 option/termination 的信号质量(H1/H2),属
"机制被参数与动力学静默失效",不属于实现错误。

---

## 4. 当前完整执行流程(经典 PTF,execute_sources=false)

```
每个 outer step(128 env 并行):
  norm_obs = obs_normalizer(obs)                      # 更新统计
  option_ids = OptionSelector.step(norm_obs, Q_o/β)   # β(s_t,o_cur) 采样终止→ε-greedy 重选;done 强制重选
  actions = actor_detach.explore(norm_obs)            # student μ + per-env 噪声 —— 唯一进入环境的动作
  env.step(actions) → r, done, trunc
  rb.extend({obs, actions, r, done, trunc, true_next_obs}, option_ids)   # option 与 transition 同槽存储
  if step > learning_starts:
    for i in 0..num_updates-1(=2):
      data = rb.sample(256/env)                        # randint 均匀;raw_observations 先 clone 再归一化
      update_main:   官方 C51 投影 + CDQ + critic AdamW      (与官方逐行一致)
      if i % 2 == 1: update_pol:
          pi = actor(obs_norm)
          rl_loss = −min(Q1,Q2)(obs, pi).mean()
          teacher = source_bank.act_selected(raw_obs, options)   # 冻结 walk;null→inactive
          gate    = (1−β(obs_norm)[option]).detach()             # learned 臂;fixed 臂=1
          transfer = λ(t)·Σ_active gate·Huber_masked(pi, teacher) / N_active
          actor_loss = rl_loss + transfer → 仅 actor AdamW
      update_option:                                   # learned/fixed 臂;scratch 跳过
          y_o = r + γ·boot·[(1−β̄′)Q̄′_o + β̄′·max Q̄′]   (target 网络)
          q_loss = Σ compat_o·(Q_o(s)−y_o)² / Σcompat   compat: walk=exp(−d̄²/4.5)≈0.8, null=1, executed=1
          β_loss = mean β(s′,o_stored)·clamp(A+ξ_adapt) → beta Adam
          soft_update(option_target, 0.05)
      soft_update(qnet_target, 0.1)
  每 5k:evaluate()(确定性,与训练 env 共享→eval 后 envs.reset(),官方同款 hack)
最终:保存 actor+normalizer(+option/critic);source-free 评估只重建 actor+normalizer
     (scripts/p0_evaluator.py,结构上不构建 bank/option/admission → 无伪 source-free 风险)
```

---

## 5. 逐函数审计(经典路径关键函数)

| 函数 | 位置 | 输入→输出 | 梯度路径 | 发现 |
|---|---|---|---|---|
| `OptionSelector.step` | option_selector.py:52 | norm_obs, Q_o/β → option_ids | no_grad | 正确的 call-and-return;独立 generator(option_seed),不扰动全局 RNG;eval 打断后状态残留(L2) |
| `SourcePolicy.act` | source_policy.py:178 | raw target obs → teacher action | no_grad,冻结 | 链路正确:adapter→source 冻结 normalizer→确定性 μ→action adapter |
| `gaussian_action_compatibility_all` | compatibility.py:6 | (a, teacher_a, mask, σ) → [B,K] | no_grad | 结构合理;**σ=1.5 生效值(bank yaml 覆盖 manifest 0.25)使 compat 恒≈0.8 → H1** |
| `masked_action_distillation_loss` | distillation.py:7 | (pi, teacher, mask) → per-sample | grad→actor only | mask 分母正确;Huber 语义正确 |
| `compute_transfer_loss` | train_ptf.py:1248 | replay batch, pi, step → loss | actor only(β detach、teacher no_grad) | 权重 λ(t)·(1−β)·active/N_active 正确实现;option-样本对应正确(同槽存储) |
| `update_option` (Q_o) | train_ptf.py:1440-1511 | batch → q_loss | option trunk+q_head | y 的 β/max 用 target 网络(M2);/compat.sum() 归一化(M4);obs detach 防蒸馏梯度回流 ✓ |
| `termination_loss_at_next_state` | option_update.py:75 | s′, 存储 option → β_loss | trunk+β_head | 方向正确(β=终止概率;差 option β↑);**2-option 时 clamp 不激活,动力学=官方 adaptive-ξ → β 系统性下压(H2)**;off-policy 化(M1) |
| `update_main` | train_ptf.py:1173 | batch → critic 更新 | critic only | 与官方逐行一致;admission 分支经典路径不触发 |
| `update_pol` | train_ptf.py:1378 | batch → actor 更新 | actor only | rl+transfer 相加;transfer 只在 actor 步 ✓ |
| `PTFReplayWrapper.extend/sample` | ptf_replay.py:317/721 | — | no_grad | ptr 语义与官方对齐;经典路径 randint 同官方;option 同槽 gather ✓ |
| `evaluate` | train_ptf.py:1115 | — | no_grad | 确定性 eval;与训练 env 共享并 reset(官方同款,三臂对称,M5) |
| `p0_evaluator` | scripts/p0_evaluator.py | ckpt → 32 episodes | no_grad | 结构性 source-free 成立;`terminated_success` 在 hurdle 语义 = 摔倒而非成功(L1,未被结论引用) |

**用户清单逐项回答**(§三):
1. 环境行为:默认执行 student action;source 只用于蒸馏与兼容性;`execute_sources=false`
   在所有经典路径严格成立(唯一注入点 `train_ptf.py:2005-2013` 有显式开关守卫);
   无隐蔽分支让 source 动作进 env 或 replay。
2. Source:冻结、无 optimizer、eval 固定;接收当前 target state(raw);normalizer
   链正确;walk→hurdle 为同 151 维同构 obs + 同 61 维动作,identity/passthrough
   合法(维度不符会显式报错)。
3. Option selection:符合 PTF;null option 是**本项目扩展**(论文/官方均无);
   ε-greedy(ε:0.3→0.05/50k,与论文 1.0→0.05 意图同向)、call-and-return 正确;
   done/truncation 后强制重选正确;单教师+null=2 options 无索引错误;
   fixed 臂(单 option 无 null)退化行为正确(见 §6 实验部分)。
4. Termination:β=终止概率,方向正确;loss 在 s′ 上、detach 正确;β 同时驱动切换
   与 (1−β) 蒸馏权重——**双重使用来自论文 Eq.7**(官方代码只有切换);"双重撤退"
   风险在本实验未发生(β 贴 0.05 下轨,两个通道都几乎不撤退,H2);
   β warmup 本实验=0,未改变语义。
5. Distillation:原 PTF 确为蒸馏正则(非教师执行);cross-entropy→Huber 的退化
   对应"同方差高斯策略 KL ∝ 均方差"假设,合理;当前用 Huber(δ=1);teacher/student
   同一状态、同一 [−1,1] 动作尺度;mask 分母正确;batch-option-teacher 三者对齐;
   λ(t)、(1−β)、active 的乘法与 /N_active 归一化正确;**transfer 与 RL loss 的梯度
   主导关系不能从 loss 标量判断(rl_loss 量级~百,transfer~10⁻¹,但两者梯度尺度
   无标量可比性)——日志只有合并的 actor_grad_norm,分项梯度范数 UNVERIFIED,
   建议加监控(§8)**。
6. Option/termination 更新:replay 的 option_id 与 transition 同槽、正确对应;
   compatible-option 更新结构忠实官方、取值失真(H1);off-policy Q_o 的未声明假设
   已在 `update_option` docstring 中声明(值得肯定);null 的 Q/β 更新构造正确
   (compat 恒 1 有设计依据:null 不施加任何教师正则,与所有行为兼容)。
7. FastTD3 主干:关闭 PTF(空 bank)后与官方语义等价且 RNG 流对齐(Gate A);
   replay wrapper 不改变官方张量的形状/dtype/采样分布;critic/CDQ/投影/AMP/调度
   逐行一致;PTF 梯度不流入 critic/source/option 交叉路径;四个 optimizer
   (actor/critic/option/beta)彼此隔离(option 与 beta 共享参数集属官方同构设计)。
8. Source-free eval:只加载 actor+obs normalizer;不构建 bank/option/termination/
   admission;独立进程离线执行,无训练期内存残留风险。

---

## 6. 问题分级

### BLOCKER
无。实验的因果对照结构(learned/fixed/scratch,同 seed、同超参、同评估协议)成立,
数字可复算。

### HIGH

- **H1|compatibility σ=1.5 使 intra-option 更新失去选择性**(SM,取值层)。
  日志实测 `ptf_source_compat/walk` 全程 0.78–0.87;由 compat=exp(−d̄²/4.5) 反推
  per-dim 均方动作距离 ≈1.0——在 [−1,1] 动作空间中这是"接近不相关"的距离,官方
  μ±1σ 语义或 manifest 默认 σ=0.25 下同样距离的兼容度 ≈e⁻⁸≈0。后果:walk 在
  几乎所有 transition 上以 ~0.8 权重更新(null 恒 1.0),Q_o(walk) 与 Q_o(null)
  用几乎相同的数据与几乎相同的 target 训练,option-value 的区分信号先天塌缩,
  Q_o 排序接近噪声(rollout walk 占比 0.44–0.96 随机漂移与此一致)。
  bank yaml 中 σ=1.5 覆盖 manifest 0.25 的决策无任何记录依据。
- **H2|learned termination 全程无信号:β 被系统性压至下轨 0.05**(动力学层)。
  日志实测 `beta_selected` 三个 seed 绝大多数时间 =0.050(clamp 下限),偶发
  0.1–0.6 后回落。机制:adaptive ξ=0.8·(top1−top2) 下,被选 option(ε-greedy,
  70–95% 是 argmax)的 advantage+ξ=+0.8·gap → β 持续下压;非 argmax 仅得
  −0.2·gap 的四分之一强度上推。注意**这是官方 adaptive-ξ 公式的固有性质**
  (2-option 时本项目的对称 clamp 数学上不激活,|−0.2·gap|<0.8·gap),叠加
  Q_o 无区分(H1)使 gap 噪声化。后果:(1−β)≈0.95 恒定 → "β-weighted transfer"
  与"learned termination 退场"两个机制在本实验中都没有实际发生;教师退场完全由
  预设 λ(t) 承担(结果文档 §4 已承认此点,但未指出 β 贴轨与 compat 的根因)。
  [0.05,0.95] clamp(IE)让失效更隐蔽:β 永远显示"合法值"而非饱和告警。
- **H3|learned−fixed 差异的归因仍有未排除的 confound**(实验设计层)。
  由于 H1/H2,learned 臂实际 = "近随机的 option 漂移 + 恒定 0.95 gate 的间歇蒸馏",
  与 fixed 的差别不是"学到的调度质量",而是"蒸馏作用于非平稳的样本子集
  (walk-active 44–96% 漂移)"。因此"当前 Q_o/β 自适应没有超过 fixed"成立,但
  "learned 调度做出了坏决策"或"option/termination 机制本身在 HB 上不可行"都
  **不能**由本实验得出——机制根本没有获得可用输入信号。需要 yoked 对照
  (随机调度、匹配 walk 占比)才能分离"调度存在的噪声成本"与"调度决策的质量"。

### MEDIUM

- **M1**|termination 在 off-policy replay batch + 存储 option 上更新(官方:当前
  on-policy transition)。进一步稀释 β 信号(旧状态上的 argmax 早已漂移),与 H2
  叠加;已在 docstring 声明,但其对 β 动力学的定量影响未验证。
- **M2**|U 目标中 β 与 max 均取 target 网络(论文/官方:β 用 online,官方 max 用
  double-Q)。方向性影响小;在 Q_o 本就无区分的现状下不是主要矛盾。
- **M3**|Q_o 无界线性 head + raw reward(官方:tanh 有界 + reward 归一化)。Q 量级
  ~数百,直接放大 adaptive ξ 与 termination 梯度量级(与 H2 交互);β_lr=1e-4 下
  未见发散,但尺度敏感性未表征。
- **M4**|q_loss 用 /compat.sum() 归一化(官方按 batch 均值)。有效学习率随 compat
  总量波动;经典 2-option 下 compat 总量稳定(≈1.8B),影响小。
- **M5**|evaluate() 与训练共享 env 并在 eval 后 reset(官方 FastTD3 同款 hack);
  OptionSelector 状态跨 eval-reset 残留 ~一个 option 段。三臂对称、不构成对照
  confound,但使"episode 边界重选"语义与官方 PTF 略有出入。

### LOW

- **L1**|`p0_evaluator.py` 的 `terminated_success`:hurdle 中 terminated=摔倒,
  JSON 里 `success_count` 实为提前终止计数(ptf_s2 的 "succ=8" 是最差臂)。未被
  结果文档引用,但字段名有误导性,建议改名或按任务语义化。
- **L2**|初始/episode option 固定为 null(官方:ε-greedy 立即选)。影响 ~前 10 步。
- **L3**|`torch.compile` 在 PTF 路径强制禁用(三臂一致,已知 trade-off,见
  memory `project_compile_tradeoff`);与官方 FastTD3 的吞吐差异不影响语义。
- **L4**|官方 PTF 的 ε 参数语义(greedy 概率递增)与本项目(探索率递减)相反但
  等价;审阅他人配置时易误读,建议注释。

### UNVERIFIED

- transfer 梯度与 RL 梯度的相对范数(判断蒸馏"剂量"必须看分项梯度范数,不能用
  loss 标量;当前日志只有合并 actor_grad_norm)。
- walk 教师在 hurdle 上的 zero-shot 回报(manifest `eval_return: null`);fixed 臂
  3/3 早期加速间接证明教师有用,但教师质量基线未记录。
- M1/M3 对 β 动力学的定量贡献(需对照实验)。

---

## 7. 对当前实验结果的独立裁决

**数字复核**:19 点 eval 曲线、5k–30k/60k/95k 梯形归一化 AUC、paired delta、
32-episode source-free 终点全部独立重算,与 results.md **逐位一致**(例:learned
AUC 463.052/99.598/143.940;final panel 875.189/177.690/439.759 vs scratch
629.035/695.585/608.354;fixed−scratch 终点 [−71.3,+37.6,+3.8] mean −9.9)。
相关单元测试 39 passed(文档写 25,应为当时数;非问题)。

**已支持的主张**:
1. 冻结 walk 的确定性 Huber 动作蒸馏在 walk→hurdle 上稳定加速前中期学习
   (fixed−scratch:5k–30k/60k/95k AUC 均 3/3 为正);
2. 在 λ→0 后,固定蒸馏臂终点回到 scratch 水平(−9.9,2/3 为正)——蒸馏不伤害
   终点,也未稳定提升终点;
3. 当前 learned 臂平均不如 fixed,且方差远大(终点 seed SD 352 vs 90),不能把
   收益归因于自动选择或 learned termination;
4. scratch 基线可信(95k≈618,与官方 FastTD3 100k hurdle 量级吻合),不是弱基线。

**尚未支持的主张**:
- 任何"PTF 的 option-value/termination 机制在 HB 上被测试过并失败"的强读法。
  H1/H2 表明该机制的输入信号(compat 区分度、β 梯度方向多样性)在本配置下
  不存在,实验测的是"无信号调度的成本",不是"学习调度的能力上限"。
- "learned PTF 早期 3/3 加速"作为 learned 机制的功劳(§3.1 的表述)——fixed 同样
  3/3 且更强,加速功劳属于蒸馏通道;results.md §6 已作此修正,§3.1 单独引用时
  需带上此限定。

**必须收窄/加注的主张**:
- results.md §4 "option 和 termination 网络不是死代码,能够改变调度"——字面成立
  (option 占比确实漂移),但应加注:漂移主要由 Q_o 噪声排序+ε 探索驱动,β 全程
  贴 0.05 下轨,不构成"机制在工作"的证据。
- 结果文档把 learned−fixed 的蒸馏剂量差异定义为 treatment 本身(§设计文档),
  在 H1/H2 失效背景下应降格为"treatment=调度机制存在与否(含其噪声)",
  而非"treatment=学到的调度"。

---

## 8. 最小修复建议(按优先级)

1. **校准 compatibility σ(修 H1)**。
   - 修什么:恢复 intra-option 更新的选择性——Q_o 的信息来源。
   - 不修后果:任何 learned option 实验都在无信号条件下运行,多教师选择无从谈起。
   - 最小改动:离线用现有 checkpoint + replay 快照测 walk 动作距离分布,选 σ 使
     compat 中位数落在 ~0.3–0.5(或直接改回官方硬区间语义,一个函数);配置层面
     改 `configs/source_banks/pure_ptf/*.yaml` 一行。
   - 最小回归:单元测试(compat 分布断言)+ 3k smoke 看 `ptf_source_compat/walk`
     是否出现跨状态方差。
   - 需重跑:learned 臂需要重跑才能重新评估调度;fixed/scratch 结果不受影响。
2. **恢复 termination 的梯度信号(修 H2)**。
   - 修什么:β 不再单调压向下轨,termination 才有被检验的资格。
   - 最小改动(三选一,建议按序尝试):a) 2-option 场景把 ξ 显式设为小常数
     (论文 0.001 的意图,避免 adaptive 0.8·gap 全额下压 argmax);b) β 更新只用
     近期(on-policy 近端)窗口样本;c) 监控层先行——β 距下轨/上轨的占比进日志,
     失效可见化。
   - 最小回归:3k smoke 中 β 应随 Q_o 排序变化出现双向移动。
   - 需重跑:learned 臂。
3. **加分项梯度范数监控(UNVERIFIED→可验证)**:update_pol 中分别记录 rl_loss 与
   transfer_loss 对 actor 的梯度范数(两次 backward 或 grad 采样),一行级改动,
   不改变训练语义;此后"蒸馏剂量"讨论才有依据。
4. **yoked 随机调度对照(修 H3,下一项实验)**:随机 option 序列匹配 learned 臂的
   walk 占比经验分布、gate 固定 0.95。learned−yoked 才是"调度决策质量"的无偏估计。
   一个臂 ×3 seeds,100k,复用现有脚本。
5. **L1 改名**(`terminated_success`→`terminated_early` 或按任务映射),防止未来
   报告误用。

## 9. 最终建议

- **是否值得继续多教师 PTF?** 以当前实现直接上多教师:**不值得**。多教师的全部
  增量价值都经由 Q_o 排序与 β 终止表达,而这两者在单教师+null 的最简配置下已被
  证实无信号(H1/H2)。先修信号,再谈选择。
- **是否应先修单教师 option/termination?** 是。修复成本低(σ 一行 + ξ 一行 +
  监控两行),且单教师 hurdle 场地已有 9 条基线曲线可直接配对复用。
- **下一项最小、可证伪、能改变科研决策的实验**:修复 σ+ξ 后重跑 learned 臂
  (3 seeds,同协议),并加 yoked 随机调度臂。可证伪判据:若修复后
  learned ≥ fixed 且方差恢复到 fixed 量级 → option/termination 机制值得保留并
  推进多教师;若 learned ≈ yoked ≈ fixed → 调度机制在该任务族无增量价值,
  资源应转向蒸馏通道本身(何时退场,即 λ 调度/T 度量线),这将直接改变
  RESEARCH_ROADMAP 中"组件①迁移性指标"与 PTF 结构线的优先级关系。

---

## 10. 勘误与更新(2026-07-22,ChatGPT 裁决后)

ChatGPT 对本审计的裁决(接受 H1、修正 H2 归因强度、否决"改 ξ + 直接跑正式臂")
经我独立复核后,以下修正成立并纳入本报告;原文对应段落以本节为准。

1. **off-policy 归属更正(修正 §3 表、§5 第 6 条、M1 表述)**。PTF 官方代码的
   Q_ω 本来就用 replay batch 更新(`PTF_A3C.py:344` `replay_buffer.get_batch`),
   论文也明确说 "in an off-policy manner"。因此"Q_o 的 replay/off-policy 更新"
   不是 FastTD3 适配,而是官方行为;真正的适配点只有两个:
   (a) termination 从"当前 on-policy transition 单样本"变为 replay batch(M1 保留);
   (b) 蒸馏数据从 on-policy rollout 变为 FastTD3 replay。
   连带新发现一处文档级错误:`train_ptf.py` `update_option` docstring 声称
   "PTF's original Q_o (Alg.2) is on-policy",与论文文本和官方代码都冲突。
   代码行为不受影响,但该 docstring 曾误导本审计初稿,建议将来修正。
2. **H2 表述降格**。"β 全程贴 0.05 下轨"是事实;"termination 完全无信号"超出
   现有证据。若被选 option 确为最优,β 低本是正确行为——该替代解释依赖
   "Q_o 排序有意义"这一前提(被 H1 削弱),但裁决它需要条件统计:argmax /
   non-argmax 分组的 β、termination advantage 正负比例、β 实际引发的切换率。
   在拿到这些统计前,H2 的准确表述是:**β 缺少已证实的状态条件区分能力,且
   adaptive-ξ 的 0.8/0.2 不对称提供了一个与观测一致的下压机制假设**。
3. **撤回修复建议 2a(ξ→0.001)**。在现行对称 clamp 实现下
   (`option_update.py:66-68`),固定 ξ=0.001 会使 margin_safe=0.001,正负
   advantage 全部被 clamp 到 ±0.001,termination 学习速率整体被压扁 3–4 个
   数量级(对比 adaptive 下 0.8·gap,gap~1–50)——"改一行 ξ 恢复信号"不成立。
   ξ 与 clamp 必须联动设计,且应先由诊断数据驱动。
4. **修复建议 4 的"无偏"降格**。yoked 随机调度只匹配全局 walk 占比,不匹配
   状态条件选择,且两臂 occupancy 随训练分叉;learned−yoked 是有参考价值的
   对照,不是"调度决策质量"的无偏估计。
5. **H1 的历史依据补强(独立核验)**。
   `docs/archive/design/PTF_FastTD3_implementation_audit.md` §4.1 #11 记录当年
   把 σ 从 0.25 调到 1.5 的理由是 61 维累积公式 `exp(−‖Δa‖²/(2σ²)·d)`
   ("每维误差 0.1 就够把兼容压到 ~0");现行 `compatibility.py:38-41` 用
   masked **mean** squared distance,同样每维误差 0.1 在 σ=0.25 下
   compat≈exp(−0.08)≈0.92。**σ=1.5 是为一个已不存在的距离公式标定的**,
   在现行代码下既无依据也与实测(compat 恒≈0.8)后果相符。H1 维持并加强。
6. **机制洞察采纳(影响 §9 定位)**。"确定性教师的当前动作相似度 ≠ 教师对
   student 的未来学习价值":walk 在 σ=0.25 语义下与 student 动作"不兼容",
   却在 fixed 臂中 3/3 加速学习。因此 action compatibility 只能作为纯 PTF
   复现内部的支持域估计,不能被包装为项目所需的迁移性指标 T——与
   roadmap"组件①须换非行为信号族"的既有裁定一致。
7. **下一步方案更新(替换 §8.1/8.2 的执行顺序)**。采纳 ChatGPT 的
   signal smoke 方向(五类诊断 + 离线 σ∈{0.25,0.5,1.0,1.5} 覆盖扫描 +
   不改训练机制 + 预注册停止条件),并补充降本改进:诊断的大部分可先
   **纯离线**完成——9 条正式 run 均存有 25k/50k/75k/final 四个 checkpoint
   (ptf 臂含 option_state_dict/actor/normalizer),配合冻结 walk source 即可在
   四个训练断面上计算 action-distance/compat 分位数、Q_source−Q_null、
   argmax 占比与分组 β;切换率可由现有日志的 `ptf_rollout_option_age_mean`
   直接反推。仅当离线断面证据不足以裁决时,再跑 3k–5k 在线 smoke 补充
   训练期动态。停止条件维持 ChatGPT 版:若"0.25 几乎无覆盖、1.5 几乎全
   覆盖、各 σ 下 Q_o 均无可靠区分",则停止把标量 σ 当修复方向。

---

## 11. current_transition 停止裁决复核(2026-07-22 晚,零训练探针)

针对 `classic_ptf_signal_diagnostic_20260722.md` §6.4 的裁决("β 失效是原 PTF
termination 动力学的结构性下压,不是实现 bug;按停止规则终止修复"),我用
current_beta 5k checkpoint 做了零训练前向探针
(`scratchpad/beta_saturation_probe.py`,1207 个冻结 rollout 状态),结果要求
**更正归因**,但**基本维持决策**:

| 量 | walk | null |
|---|---:|---:|
| β_head logit(mean [p1,p99]) | −14.6 [−19.1,−10.4] | −17.3 [−22.1,−13.2] |
| sigmoid′ 梯度因子 | 2.2e-6 | 2.1e-7 |
| non-argmax 状态 clamped advantage | mean −0.0059,**100% 为负** | mean −0.0055,**100% 为负** |
| E[logit 梯度 \| β 健康(logit≈0)] | argmax +4.9e-3 / non-argmax **−1.3e-3** | +5.3e-3 / **−1.2e-3** |
| E[logit 梯度 \| 实际饱和] | ~1e-8 | ~1e-9 |

1. **状态条件上推信号是存在的**:non-argmax 状态上 termination advantage 100%
   为负(要求 β↑),量级在健康参数化下完全可学习。§6.4"termination 动力学
   不产生可用信号"的读法不成立。
2. **失败的直接原因是表示层死区**:β_head logit 深度饱和(−15~−17),上推
   梯度被 sigmoid′ 乘 ~1e-6 吞掉;解饱和需 ≈13 个 logit 单位 ≳12.7 万次
   update,5k gate(1 万次)物理上不可能显现任何差异。
   **[0.05,0.95] 输出 rescale 只保护值域、不保护梯度**——2026-05-20 β-clamp
   fix 的原始意图(防 sigmoid death)并未真正实现。
3. **完整因果链**:adaptive-ξ 的 +0.8/−0.2 不对称在聚合上净下压
   (E[healthy grad]≈+2.8e-3)→ logit 单调坠入深度饱和 → 死区锁死 →
   之后任何状态条件信号都无法表达。ChatGPT 的"净下压"作为**入坑动力**成立,
   "非实现问题"不成立——入坑后出不来是明确的实现层缺陷。
4. **current_transition gate 无诊断力**:replay 与 current_transition 两臂的 β
   都在死区,该实验注定观察不到差异;"replay 假设被否证"应降格为
   "该实验对此假设无判别力"(结合本探针,replay 时序确实不再是首要嫌疑,
   方向性结论保留,证据链更换为本探针)。
5. **对停止决策的意见**:预注册停止规则在程序上被触发,但触发它的实验无
   诊断力。存在最后一个便宜的可证伪 gate:防饱和参数化(如 β_head logit
   clamp 到 ±4,或等效梯度保护),5k 单 seed,判据 = argmax/non-argmax 的
   **条件 β 分离**(非平均 β 上浮——净下压下平均 β 本应下降)。是否执行由
   PI 决定;若不执行,对外结论必须写成"learned termination 未在本设置恢复
   (根因 = 净下压动力 + β 头表示层饱和)",**不可**写成"原 PTF termination
   机制在 HB 上结构性无效"。

---

## 12. released_code_fidelity 模式复核(2026-07-23)

对 ChatGPT 的多教师实验、官方忠实模式实现与 30k gate 的独立复核。

### 12.1 核验通过项

- **adaptive ξ 归属**:确为作者公开代码行为(`PTF_A3C.py:270-271` + 官方配置
  `xi: 0`),本审计 §2.2 第一轮已独立确认;PI 的怀疑虽不成立,但"2022 公开
  代码 ≠ 2020 论文私有代码"的 caveat 正确。
- **fidelity 实现忠实度**:逐项对照 diff 核实——source-only bank、reset 立即
  ε-greedy(needs_reselection)、20-unit ReLU6 trunk + tanh Q + bare sigmoid β
  (normal(0,0.01) 初始化、零 bias)、β 先于 Q 且用 current transition、每
  outer step 仅 1 次 option 更新(绑定 i==0)、独立 option_batch_size=32、
  online-β/online-argmax + target-Q 的 U、1000 次硬拷贝、无 advantage clamp、
  无 (1−β)、released tanh 衰减、按需 RNG 消耗。与官方语义一致;混合防护
  (`_validate_released_code_fidelity_config`)合理。37 项相关测试通过。
- **报告措辞**:与历史曲线的比较明确标注"描述性、不能写成因果增益",合格。
- **"Q_ω 学相似度"的撤回**正确。补一层机制表述:教师不执行时 o 不影响环境
  转移,各 Q_o 共享同一 reward 流,区分只来自 compat 数据选择与各自 β;故
  Q_ω 语义 = "教师 o 动作支持域内的学生 return-to-go"。

### 12.2 新发现

- **F1(HIGH,更新 2026-07-23:已发生,非前瞻)|tanh Q × HB 未归一化
  reward 的量纲冲突**。官方 tanh Q 头的隐含前提是回报落在 [−1,1]
  (grid/pinball 靠 `reward_normalize: r/done_reward`,reacher 为稀疏 +1);
  fidelity 恢复了 tanh 头但 HB 路径 `normalize_reward = nn.Identity()`,
  raw reward 直接进 TD target。初稿判断为"100k 才会触发的前瞻风险",
  **ChatGPT 复测证明 30k 已大面积发生,我的独立复算确认且更严重**
  (两个独立冻结面板,7245 vs 5792 状态):

  | 量 | ChatGPT 面板 | 本审计独立面板 |
  |---|---:|---:|
  | 全 option \|Q\|>0.95 状态占比 | 38.4% | **49.6%** |
  | top1−top2 gap 中位数 | 0.00226 | 0.00527 |
  | gap<0.01 状态占比 | 63.9% | 60.9% |
  | mean\|Q\|(全局) | 0.458 | 0.597 |

  **判据勘误**:初稿建议的全局 `mean|Q|>0.95` 会漏检(饱和是状态条件的,
  全局均值被未饱和状态稀释)——接受 ChatGPT 修正,监控指标应为
  "全 option 同时饱和且 gap 小的状态比例"。**新佐证**:stand/run 的 argmax
  占比在两个独立面板间从 53%/44% 翻转为 37%/60%——饱和区的教师排序
  连评估面板之间都不稳定,排序退化是直接观察而非推断。修复方案统一为
  `option_reward_scale=0.01`(=(1−γ)/R_max,只作用于 Q_ω 的 TD reward,
  默认 1.0 保旧实验不变),属官方 r/done_reward 的等价 HB 适配。
- **F1b(观察,官方语义内)|bare sigmoid β 已整体二值化**。独立复算显示
  三个 option 的 β logit 全部深入 ±16~20:stand +19.9(单向,collapse)、
  walk +16.6、run −18.9(状态条件的双向饱和)。run 的"+0.908 条件分化"
  实为近二值 termination 规则(β≈0 或 ≈1),不是软概率。官方 bare sigmoid
  本就如此,不必修;但两点解读约束:(a) 正式实验中 β 应按二值规则解读;
  (b) 修 F1 后检验 stand 是否仍坍缩的 gate **必须是 fresh 训练**(β logit
  从 0 出发)——从旧 checkpoint 续训时 ±20 logit 的死区惯性会使任何 5k
  观察窗都无诊断力(同 §11 的死区机理)。
- **F2(MEDIUM)|stand 的 β≡1.000(双向饱和重现)**。bare sigmoid 下
  stand 在 argmax/non-argmax 状态 β 均为 1.000——2026-05-20 最初发现的
  β→1 collapse 在 fidelity 模式按官方语义重现(termination collapse 是
  PTF/option-critic 已知现象)。当前无 (1−β) gate 故不杀蒸馏,但 stand
  实际上单步即弃、退出有效调度。"termination trainability 已恢复"应表述
  为 **2/3 source 恢复状态条件终止**(run +0.908、walk +0.347),非全面健康。
- **F3(归因替代解释)|"familiarity bias" 未被证实**。stand 高 argmax 占比
  (53%)有一个更简单的竞争解释:hurdle reward 以"活着+前进"为主,早期
  学生在站稳状态的 return-to-go 确实最高——Q_ω 可能是**短视地正确**
  (它排的就是 return,而 return 与学习增量不同),而非 compat 支持域偏置。
  区分三个假设(支持域偏置/短视正确/估计噪声)恰好就是 ChatGPT 提的
  stand-only/run-only fixed 蒸馏排序实验,应保留该设计。

### 12.3 对三臂计划的意见

同意"当前 checkout、seed1、scratch/fidelity/fixed-walk 严格匹配"的方向,
两点修正建议:

1. **先修 F1 再跑 100k**(倾向),或至少预注册 Q 饱和判据(mean|Q|>0.95 且
   gap<0.01 → 判定量纲失效),否则 fidelity 臂后半程在测一个已知将自毁的
   机制,浪费一次正式预算。
2. **蒸馏日程不统一是 confound**:fidelity 臂用 released tanh 衰减,现有
   fixed-walk 配置是 linear λ——"learned vs fixed"的差异会混入日程形状。
   严格对照应让 fixed-walk 同用 tanh 日程,或在预注册中显式声明此差异。

定位不变:fidelity 线做实"PTF 复现 baseline";Q_ω 仍是 option-conditioned
return estimator,不是迁移性指标——教师价值需要 source-specific evidence
(reward-bearing 线),这是 PI 主线的既有裁定。

---

## 13. option_reward_scale 终局复核与整线收束(2026-07-23 晚)

对 `classic_ptf_option_reward_scale_gate_20260723.md` 的停止裁决做最后一轮
独立复核(scaled 30k checkpoint,独立面板 5861 状态)。

### 13.1 数字复核:全部吻合

| 量 | ChatGPT | 本审计独立面板 |
|---|---:|---:|
| TD target 越界 / 全 option 饱和 | 0 / 0% | 0 / **0%** |
| gap 中位数 | 0.000428 | 0.00048 |
| gap<0.01 占比 | 100% | 100% |
| walk argmax 占比 | 89.6% | 91.1% |
| walk β(argmax/non) | 0.896/0.838 | 0.921/0.835 |

`option_reward_scale` 实现确认只进 Q_ω TD target;42 项聚焦测试通过。
F1 修复成功、机制未恢复、停止 scale/ξ/β_lr 搜索——**均维持**。

### 13.2 两处科学措辞修正(写论文/结论时必须采用)

1. **"Q_ω 没有学到教师信息"过强**。去饱和后 Q_ω 的 argmax 在 ~90% 状态
   一致指向 walk——正是 fixed 对照已证明的最佳教师(且未缩放版的
   stand/run 面板间翻转消失)。准确表述:**Q_ω 含有方向正确但量级微小的
   教师信号(gap ≈ Q 量级的 ~2%),其量级不足以驱动 β/termination 机制**。
   这与"零信息"是不同的结论,且为"教师排序标定实验"(backlog)提供了
   一个可检验预测:Q_ω argmax 排序可能与 fixed 蒸馏收益排序一致。
2. **termination 失效的微观机制是"信号淹没"而非"β 学到了坏策略"**。
   独立面板显示三个 option 的 β logit 全部落在 +1.0~+2.7 的温和区
   (β≈0.7–0.9,条件差 ≈0 或反向:最优教师 walk 的 β 反而最高)。
   termination 梯度量级 ∝ gap(≈5e-4),已低于 trunk 被 Q 学习驱动的
   漂移量级——β 的取值由非语义因素决定,call-and-return 退化为
   每步重抽(age 0.24 步)。

### 13.3 整线终局结论(经典 PTF 复现审计收束)

两条独立证据链在此收束,共同指向同一结构性失配:

- **termination 线**(§11):β 的状态条件信号存在,但被 sigmoid 表示层
  死区吞噬(clamp 版)或被净下压推入死区(bare 版);
- **option-value 线**(本节):Q_ω 的教师排序信号存在且方向正确,但其
  量级(共享 reward 流下的支持域二阶差异,~2%)低于机制的驱动阈值。

因此经典 PTF 在 HB+FastTD3 上的准确终局判词是:
**"两级调度机制(Q_ω 选择、β 终止)的输入信号并非不存在,而是其量级与
机制灵敏度在此域系统性失配;教师价值必须由 source-specific、
student-relative 的证据承载"**——这直接支持回归 transferability 主线,
且比"PTF 无效/失败"既准确又可辩护。

### 13.4 遗留标注

- 三臂(scratch/fidelity/fixed-walk)是**暂停而非永久否决**:fidelity 的
  30k nAUC 22.63 仍高于历史 scratch 11.04(描述性)。若论文需要严格匹配的
  "PTF baseline"曲线,该三臂仍需回补;当前证据链(gate + 机制诊断)可
  支撑定性叙述,不可支撑定量 baseline 数字。
- 教师排序标定(stand/run/walk-only fixed 短跑)保留在 backlog,双重价值:
  transferability 组件①的 ground-truth 标定 + 检验 §13.2-1 的预测。

---

### 附:审计中使用的一手证据索引

- PTF 论文:`papers/PTF-arxiv.pdf`(Alg.1/2,Eq.3–7,Table 4)
- PTF 官方:`reference_source_code/PTF_code/alg/{PTF_A3C,PTF_PPO,source_actor}.py`、
  `run/run_ptf_ppo.py`、`config/{ptf_a3c_conf,reacher_conf}.yaml`
- FastTD3 论文:`papers/FastTD3.pdf`;官方快照:`fasttd3_ptf/official_code/FastTD3/`
  (与 `reference_source_code/FastTD3` 逐文件一致)
- 本项目:`fasttd3_ptf/official_fasttd3_ptf/{train_ptf,ptf_replay,humanoid_bench_env}.py`、
  `fasttd3_ptf/ptf/{option_module,option_update,option_selector,compatibility,distillation,source_bank,source_policy,adapters}.py`
- 实验:`configs/experiments/classic_*.yaml`、`configs/source_banks/pure_ptf/*.yaml`、
  `logs/train/classic_ptf_hurdle_single_source_v1/*.log`(独立重算脚本见本审计
  会话记录)、`docs/data/classic_ptf_hurdle_single_source_v1/final_eval/*.json`
- 测试:`tests/`(本轮复跑 39 passed:option_module/option_selector/replay_snapshot/
  source_policy/adapters/core)
