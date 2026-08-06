# PTF-FastTD3 分模块实现逐函数审计（第一阶段）

> 日期：2026-07-21  
> 审计者：ChatGPT / Codex  
> 范围：历史分模块实现 `fasttd3_ptf/my_fasttd3_ptf/`，以提交 `40b04cc` 为主要代码锚点；同时审查它所调用、后来迁移到 `fasttd3_ptf/ptf/` 的共享 PTF 模块。  
> 对照基准：`papers/PTF-arxiv.pdf`、`reference_source_code/PTF_code/`、`papers/FastTD3.pdf`、`fasttd3_ptf/official_code/FastTD3/fast_td3/`。  
> 本阶段只做静态代码与既有测试审计，不修改训练逻辑、不启动正式实验。下一阶段再审查 `official_fasttd3_ptf/` 集成版本。

## 1. 先给结论

旧分模块版本不是一个“完全错误的 PTF/FastTD3 实现”。它已经形成了闭合训练链，而且以下核心语义是成立的：

1. FastTD3 主体包含确定性 actor、双分布式 critic、C51 投影、CDQ、target-policy smoothing、delayed actor update、GPU per-env replay 和 episode-level mixed Gaussian exploration。
2. PTF 路径遵循论文最重要的边界：**环境动作由目标 student actor 产生；被选择的 source 是互补模仿目标，而不是直接替换环境动作。** 因此旧分模块路径不是后来发现有效的 reward-bearing source bootstrap。
3. option selector 具有 call-and-return 结构；option-value 使用论文的 `U(s',o)` 形式；termination 在下一状态更新；actor loss 由 RL loss 与 source action distillation 组成。
4. 为 HumanoidBench 增加 observation adapter、action adapter、body-group mask、source-specific normalizer，是跨任务迁移所必需的工程扩展。
5. 目标评估调用 `agent.act(..., explore=False)`，而不是 `act_with_options()`；因此评估的是 source-free student，边界清楚。

但它不能被表述为“与 PTF 论文和 FastTD3 官方源码严格等价”。主要原因有四类：

- **可复现性断点**：历史提交漏收整个 `my_fasttd3_ptf/models/`，无法仅凭 Git 重建；目前只能用迁移后的副本与历史终端记录辅助恢复。
- **FastTD3 基线不完全等价**：没有官方 AMP、LR scheduler 和完整 checkpoint-resume 语义；normalizer、环境封装和计步契约也不同。
- **PTF→off-policy TD3 的实质改写**：actor 从长期 replay 中使用历史 option 标签蒸馏；selected option 即使与样本动作不兼容仍被强制更新；option target、β 训练和 compatibility 都不是论文原式。
- **若干稳定化说明强于代码事实**：β 输出限幅不能防止内部 sigmoid 饱和；adaptive-margin clamp 也不总能产生所谓“1:1”终止推力。

因此，本阶段总裁决是：

> **算法骨架可运行、主要方程大体对齐，但它是一个研究性适配原型，不是可作为正式 FastTD3 基线的严格复现，也没有证明 PTF selector/termination 在 HumanoidBench 上被正确识别。后续正式实验应走官方 FastTD3 集成路径；旧分模块版本只适合用于理解、toy 调试和历史机制考古。**

## 2. 证据与版本边界

### 2.1 代码锚点

- 历史分模块主体：Git `40b04cc`。
- 删除旧路径的提交：`2e7aded`。
- 当前共享 PTF 模块：`fasttd3_ptf/ptf/`。
- 当前审计时 HEAD：`a5de1a3f55c3088ed250a68c9e5dc4a45cea459f`。

当前工作树含用户/GLM 的未提交改动，本审计没有覆盖、回退或提交这些改动。

### 2.2 历史模型文件缺失

`40b04cc` 中 `agents/fasttd3_agent.py` 明确导入：

```python
from fasttd3_ptf.my_fasttd3_ptf.models import (
    Actor, DistributionalCritic,
    UpstreamFastTD3Actor, UpstreamFastTD3Critic,
)
```

但该提交的 tree 不含 `my_fasttd3_ptf/models/`。根因是 `.gitignore` 使用了无根路径限定的 `models/`，从而把任意层级同名目录都忽略了。可用辅助证据包括：

- 迁移到当前 `ptf/legacy_actors.py` 和 `ptf/option_module.py` 的副本；
- 历史 Codex 会话 `/home/yjj/.codex/sessions/2026/06/01/rollout-2026-06-01T07-29-24-019e8216-5ea1-71e2-a970-32033e472693.jsonl` 中对当时文件的原样终端输出。

本审计会使用这些证据理解逻辑，但不把它们冒充为 Git 可重构源码。这一缺口本身记为 `B-01`。

### 2.3 “论文”与“发布代码”不是同一个基准

PTF 论文和其发布代码存在差异：

- 论文 Eq.7 明确给出 `f(t) * (1-beta)`；发布的 A3C/PPO 实现计算了 termination 信息，但 transfer cross-entropy 实际只乘时间 schedule，没有真正乘 `(1-beta)`。
- 论文 Algorithm 2 的连续动作条件是“source policy 能选择该动作”；发布代码却把“当前 selected option”无条件置为可更新，即 `selected OR compatible`。
- 论文表格给固定 `xi=0.001`；发布代码在 `xi==0` 时使用 `0.8*(top1-top2)`。

所以每项裁决都区分：论文一致、发布代码一致、必要适配、项目自定义扩展。

## 3. 端到端执行流程

```text
train_target.train
  ├─ make_vec_env -> TorchVecEnv
  ├─ SourcePolicyBank.from_config
  │    └─ SourcePolicy.from_spec -> load frozen actor + normalizer + adapters
  ├─ PTFFastTD3Agent
  │    ├─ FastTD3 actor / twin distributional critic
  │    ├─ OptionModule(Q_o, beta)
  │    ├─ OptionSelector(call-and-return)
  │    └─ source bank
  ├─ rollout
  │    ├─ option_selector.step(obs)
  │    ├─ student actor produces environment action
  │    └─ replay stores (s, a_student, r_target, s', option_id)
  └─ off-policy update
       ├─ critic: C51 + CDQ target
       ├─ actor: -Q(s,pi(s)) + lambda(t)(1-beta)D(pi,source)
       ├─ option Q: Bellman U target + compatibility-weighted multi-option update
       ├─ termination beta: next-state option advantage
       └─ Polyak target updates
```

这里最容易误解的一点是：旧分模块实现中 replay 没有 source 行为轨迹。`option_id` 只是“当时选择了哪个蒸馏教师”，存储的 `action` 仍来自 student。后来的 SAFE/WFix/admission bootstrap 才让 source 在目标环境直接执行并写入 reward-bearing transition。

## 4. FastTD3 模型与损失逐函数审查

### 4.1 模型文件（历史记录恢复）

| 函数/类 | 实现逻辑 | 对照结论 | 风险 |
|---|---|---|---|
| `build_mlp()` | 按 hidden dims 堆叠 Linear+ReLU，末层线性输出 | 通用组件 | 无明显问题 |
| `init_last_layer_small()` | 找到最后一个 Linear，以小均匀分布初始化 | 与 TD3 小输出层初始化意图一致 | 非官方 FastTD3 精确初始化，但 modular actor 才使用 |
| `Actor.__init__()` | `[512,256,128]` MLP，tanh 后映射到 action bounds | 架构规模对齐 FastTD3；额外支持非 `[-1,1]` 范围 | 对 HumanoidBench 无实质差异 |
| `Actor.forward()` | `tanh(net(obs))*scale+bias` | 正确 | 无 |
| `Actor.clamp_action()` | 元素级裁剪 | 正确 | 官方 rollout 没在 actor 内裁剪 exploration action，属于小差异 |
| `Actor.export_kwargs()` | 导出维度、隐层、动作范围 | 支持 checkpoint 重建 | 与严格 checkpoint schema 配套不足 |
| `_action_bounds()` | 构造/检查上下界 | 正确 | 无 |
| `UpstreamFastTD3Actor.__init__()` | 复刻官方 512→256→128 actor、末层正态初始化、per-env noise scale | 主体与官方一致 | noise buffer 改为 non-persistent；这是合理 checkpoint 适配 |
| `UpstreamFastTD3Actor.forward()` | MLP+tanh，再做 bounds scale/bias | 主体一致 | 官方默认直接输出 `[-1,1]` |
| `_ensure_noise_shape()` | batch/env 数变化时重采样 noise scale | 工程扩展 | 会改变运行时 RNG；正常固定 num_envs 不触发 |
| `explore()` | done env 重采 noise scale，再加 Gaussian noise | 对齐 FastTD3 mixed exploration | 最后由 agent clamp，和官方 env 侧行为略不同 |
| `UpstreamDistributionalQNetwork.forward()` | 拼接 obs/action，输出 C51 logits | 与官方一致 | 无 |
| `UpstreamFastTD3Critic.forward()` | twin logits | 与官方一致 | 无 |
| `q_values()` | softmax 后按 support 求期望 | 与官方 `get_value()` 一致 | 无 |
| `DistributionalCritic` | 自定义 twin MLP C51 critic | 算法正确，结构规模对齐 | 不是官方类，不能据此声称源码等价 |
| `OptionModule` | 共享 trunk，独立 Q 与 beta head | 结构符合 PTF | 具体输出与训练目标有项目改写，见第 6 节 |

### 4.2 `agents/losses.py`

| 函数 | 审查 |
|---|---|
| `project_c51_distribution()` | 正确构造 `r + gamma*bootstrap*z`、截断到 support、向上下 atom 分配质量；对 `l==u` 单独把全部质量留在同一 atom，逻辑比官方代码的整数边界改写更直观。最后 `clamp_min(1e-8)` 并重归一化是额外数值稳定化，不改变主方程。只支持 1-step scalar gamma。 |
| `distributional_ce_loss()` | 对 target distribution 和预测 logits 做交叉熵，正确。 |

### 4.3 `FastTD3Agent`

| 函数 | 实现逻辑 | 裁决 |
|---|---|---|
| `__init__()` | 读取网络、C51、CDQ、noise、optimizer、normalizer 配置；创建 actor/critic targets | upstream core 的网络与核心超参接近官方；但缺官方 AMP/scaler、cosine LR scheduler、actor_detach 和完整 checkpoint-resume |
| `act()` | raw obs→冻结 normalizer→actor/explore→clamp | 正确；目标 student 行为路径清楚 |
| `add_exploration_noise()` | 固定 sigma 或 batch 内逐样本 uniform sigma | modular core 可用；不是官方按 episode 固定 noise scale 的同一过程 |
| `normalize_batch()` | 用 replay batch 更新 obs/critic RMS，再归一化 next obs；可选 reward norm | 可运行；统计更新时机与官方不同，见 `H-06` |
| `update()` | critic→按 delay actor→target update→counter | 顺序合理；upstream core 对 critic 每次软更新、actor target 默认不用，接近官方 |
| `_should_update_actor()` | upstream core 从第二次 critic update 开始每两次更新 actor | 对齐官方 `num_updates=2, policy_frequency=2` 的第二个 inner update |
| `update_critic()` | online actor 生成 target action、加 clipped noise、target twin C51 projection、按较低期望分布做 CDQ、双 CE 优化 | FastTD3 核心数学正确，是本实现最扎实的一部分 |
| `compute_actor_extra_loss()` | base class 返回 0 | 合理扩展钩子 |
| `update_actor()` | 对 twin distribution expectation 取 min，最小化 `-Q`，叠加子类 transfer loss | 正确的 DPG 适配 |
| `soft_update_targets()` | actor/critic Polyak | critic 与官方一致；upstream 默认不使用 actor target |
| `state_dict()` | 保存网络、target、optimizer、normalizer | 保存字段较完整，但加载不对称 |
| `load_state_dict()` | 恢复网络/target/normalizer | **没有恢复已保存的 actor/critic optimizer，也不恢复 update counters/RNG/scheduler（本来就没有 scheduler）**；不具备可靠训练续跑语义 |
| `save()` / `load()` | torch checkpoint 封装 | 推理加载可用；训练 resume 不能视为完整 |

`agents/schedulers.py` 本身没有新的调度逻辑，只是重新导出共享
`fasttd3_ptf.utils.schedules.LinearScheduler`。因此 transfer lambda、option epsilon
和 beta warmup 的真正定义来自共享 scheduler；分模块训练入口传给它的 `step` 单位才是
需要审计的关键，而不是这个转发模块本身。

### 4.4 与官方 FastTD3 的逐项结论

| 项目 | 分模块版 | 官方 FastTD3 | 结论 |
|---|---|---|---|
| actor/critic 隐层 | 512/256/128 与 1024/512/256 | 相同 | 对齐 |
| distributional critic | 101 atom twin C51 | 相同 | 对齐 |
| CDQ | 选择期望值较低的 target distribution | 相同 | 对齐 |
| target action | online actor + clipped policy noise | 相同 | 对齐 |
| actor delay | 每 2 critic updates | 相同默认语义 | 对齐 |
| target critic | 每 critic update Polyak，默认 tau 0.1 | 相同 | 对齐 |
| mixed exploration | upstream actor 按 env/episode保持 noise scale | 相同思想 | 基本对齐 |
| per-env replay | 有 | 有 | 基本对齐 |
| AMP | 无 | 有 | 未对齐；主要影响速度/数值路径 |
| `torch.compile` | 只 compile actor/critic module | 官方 compile update functions、policy、normalizer | 未对齐 |
| LR scheduler | 无 | cosine scheduler | 未对齐；若端点 LR 相同则影响可能有限，但不能假定所有任务配置相同 |
| normalizer | replay batch更新、自定义 RMS/clip | online和replay调用均可能更新，官方 EmpiricalNormalization | 未对齐 |
| n-step | 只 1-step | 支持，HB 默认 1 | 默认配置对齐 |
| asymmetric critic obs | buffer接口支持，但 HB wrapper主路同 obs | 官方完整支持 | 部分对齐 |
| checkpoint resume | 不完整 | 官方也以训练脚本保存字段为准；当前集成另有 anchor-resume | 旧版本不可用于严谨续跑 |

## 5. Replay、训练入口与评估逐函数审查

### 5.1 Replay

| 函数/类 | 审查 |
|---|---|
| `ReplayBatch.as_dict()` | 简单复制 dataclass 字段；主训练未依赖。 |
| `ReplayBuffer.__init__()` | 全局 flat GPU ring；容量单位是 transition。 |
| `ReplayBuffer.add()` | 规范 dtype/shape，支持环绕与一次插入多 env transition；逻辑正确。`store_options` 只保存为属性，实际上无论开关都分配并写 `options`。 |
| `_assign()` | 对各 storage 做 `copy_`；正确。 |
| `can_sample()` | 要求全局 size≥batch；合理。 |
| `sample()` | 全局 uniform sampling；不是 per-env 平衡采样。 |
| `PerEnvReplayBuffer.__init__()` | `[num_envs, buffer_size_per_env,...]`，符合 FastTD3 replay 设计。 |
| `size` | 返回 `_size*num_envs`；正确。 |
| `add()` | 所有 env 同一 ptr 写入；符合同步 vector step。 |
| `add_ptf()` | 强制提供 option id，委托 `add()`。 |
| `can_sample()` | 只要求 `_size>0`；正式训练另有 learning-start gate，能用，但接口语义和 flat buffer 不一致。 |
| `_gather_last_dim()` / `_gather_flat()` | 按 env 各采 `per_env_batch`，再 flatten；正确。 |
| `sample(batch_size)` | `per_env_batch=max(1,batch_size//num_envs)`；当 batch 不可整除时实际 batch 变成下取整倍数，当 batch<num_envs 时反而返回 num_envs。默认 32768/128 无问题，但接口没有验证。 |
| `PTFReplayBuffer.add_ptf()` | 强制 option id，正确。 |

旧 replay 只有 `option_id`，没有 source provenance、行为执行者、segment、learner step 或 active eligibility。因此它不能表达后来老师意见 1 所需的 source lifecycle replay。

### 5.2 `train_source.py`

| 函数 | 审查 |
|---|---|
| `_buffer_capacity()` / `_buffer_size_per_env()` | 在 global capacity 与 per-env length 之间转换；默认合理。 |
| `_make_replay_buffer()` | 按 config 选择 flat/per-env；正式 upstream 应选 per-env。 |
| `train()` | seed→env→agent→buffer→rollout→update→eval/render/save→manifest，流程闭合。`global_step` 以所有 env transitions 计，而 `log_step` 才除以 num_envs；这是后续 schedule 单位风险源。 |
| `main()` | CLI config 入口；正常。 |

### 5.3 `train_target.py`

| 函数 | 审查 |
|---|---|
| `_buffer_capacity()` / `_buffer_size_per_env()` | 同 source 路径。 |
| `_make_replay_buffer()` | PTF 时存 option id；正确。 |
| `train()` | 构造 source bank 与 PTF agent；每一步先选 option、再由 student actor 行动、写入目标 reward transition、执行 off-policy 更新；旧 PTF 主链正确。 |
| `main()` | CLI 入口。 |

`train()` 有两个需明确的边界：

1. `global_step += env.num_envs`，并把这个值传给 transfer、epsilon、beta-warmup scheduler；而官方 FastTD3 的 `global_step` 是 vector step。旧配置名 `total_env_steps` 暗示它可能故意使用 transition 数，但文档和后续实验经常用“30k/300k steps”指 vector step。这一契约没有在 config schema 中显式声明，迁移时可产生 128 倍差异。
2. 若 `random_warmup=True` 且启用 PTF，代码先选择 option，再把 student action替换成随机动作，最后仍以该 option id 入 replay。由于 option update 强制 `selected_oh=1`，会把随机动作 transition 归因给所选 source。upstream core 默认 `random_warmup=False`，所以主配置不触发；但 modular core 会触发。

### 5.4 环境与评估支持函数

| 函数/类 | 审查 |
|---|---|
| `TorchVecEnv.__init__()` | 只支持 Box，展平维度，记录 action bounds；适用于旧 HB 路径。 |
| `reset()` / `step()` | NumPy↔Torch 转换；同步语义正确。 |
| `sample_actions()` | 每 env 独立从 action space 采样；正确。 |
| `render()` / `close()` | 简单代理。 |
| `_obs_to_tensor()` | 直接 `np.asarray(...).reshape`，不支持 dict obs；旧 HB state obs可用。 |
| `_infer_max_episode_steps()` | 优先 env spec，否则任务名硬编码；回退可用但易随 HB 版本漂移。 |
| `_make_one()` | 每 rank 单独 seed；合理。 |
| `make_vec_env()` | Sync/AsyncVectorEnv；思想接近官方 HB SubprocVecEnv，但不是相同 wrapper。 |
| `get_true_next_obs()` | 尝试从 `final_observation`/`terminal_observation` 恢复 terminal state；方向正确；异常被静默吞掉，失败时会用 autoreset obs。 |
| `flatten_obs()` | dict key 排序后拼接；主 wrapper 没使用。 |
| `info_has_time_limit()` | 兼容多个 timeout key；主训练没使用。 |
| `clip_tensor_action()` | 正确裁剪；主 agent 使用自身 clamp。 |
| `ObservationSlice.select()` / `ObservationSchema.get/select()` / `robot_only_schema()` | 简单 schema；正确，功能有限。 |
| `ActionSlice.mask()` / `ActionSchema.get/mask()` / `h1hand_default_action_schema()` | 61维 h1hand 动作分组；现有测试对 XML 顺序做过核对。 |
| `register_humanoidbench()` / `make_humanoidbench_env()` | 注册并构造 HB；正确。 |
| IsaacLab/Playground maker | 仅最小可选包装，未达到官方 FastTD3 对这些 suite 的完整接口。 |
| `ToyPointEnv` | 只用于 smoke；环境实现闭合。 |
| `evaluate_agent()` | 固定 horizon、deterministic student；source-free 边界正确。 |
| `render_rollout()` | deterministic student，首个 env done 即结束；可视化用途。 |
| `render._build_agent()` | 根据 checkpoint type 重建 PTF/base agent；依赖外部 bank config。 |
| `render()` | 调用 `agent.act`，所以即使 PTF checkpoint 也不运行 option selector/source；source-free。 |
| `evaluate.main()` | 只能构建 `FastTD3Agent`，不能完整重建 PTF option 模块，但评 student actor 足够。 |
| `CheckpointCallback.__init__()` / `maybe_save()` | 建目录并按 `step % interval == 0` 保存 step/latest；功能正确，但不解决 agent 自身缺失的完整 resume 状态。 |
| `EvalCallback.__init__()` / `maybe_eval()` | 使用 `agent.act(..., explore=False)` 收集指定数量 episode；因此是 source-free student 评估。它会重置传入 env，不能直接嵌入共享训练 env 而不改变 occupancy。 |
| `PTFDiagnostics.__init__()` / `collect()` | 记录 Q/beta 均值、Q 最大值和 greedy option 直方图；是诊断，不证明 option 判断正确。 |
| `RenderCallback.__init__()` / `maybe_render()` | 仅返回 `render/requested=1`，没有真正生成视频。 |
| `update_fns.py` | 四个薄代理；无算法逻辑，主训练也直接调 agent 方法。 |

## 6. PTF 模块逐函数审查

### 6.1 `OptionModule`

| 函数 | 审查 |
|---|---|
| `build_mlp()` | 正确的共享 MLP 构造。 |
| `OptionModule.__init__()` | shared trunk + Q head + beta head，符合 PTF 网络共享结构；隐藏层由 32 扩到 256×2 是 HB 扩容。Q head 从发布代码的 tanh 改为 unbounded linear，适合大回报量级，但不是精确复刻。beta bias=-2 使初始 beta约0.16（限幅后约0.157），属自定义初始化。 |
| `forward()` | `beta=beta_min+(beta_max-beta_min)*sigmoid(logit)` 保证输出在 `[.05,.95]`，因此 `(1-beta)` 不会变成0。但**它不能阻止内部 sigmoid logit饱和，也不能保证 beta 梯度可恢复**；当前注释和旧测试对此表述过强。 |
| `export_kwargs()` | 支持重建；正确。 |

### 6.2 `OptionSelector`

| 函数 | 审查 |
|---|---|
| `__init__()` | per-env current option、独立 RNG；旧版默认 null option。当前新增 `min_duration` 是后续扩展，`40b04cc` 版本实际相当于1。 |
| `reset()` | done/全体回 null 并清时长；主训练主要通过 `step(dones=...)` 处理。 |
| `step()` | 先在当前状态求 Q/beta，再按 beta 判断终止；终止时 epsilon-greedy 选新 option；done 强制重选。call-and-return 语义正确。项目 epsilon 表示探索概率并递减，发布代码 epsilon 表示 exploitation 概率并递增，两者数值不完全相同但概念可换算。 |
| `_rand()` / `_randint()` | 使用 named generator，不污染全局 torch RNG；是良好工程实践。 |

### 6.3 option/termination 更新辅助函数

| 函数 | 审查 |
|---|---|
| `option_u_value()` | 精确实现论文 Eq.3 的矩阵形式。 |
| `termination_margin()` | `xi!=0` 固定；`xi==0` 使用发布代码的 `0.8*(top1-top2)`。这不是论文 Table 4 的固定 xi，属于 code-faithful 选择。 |
| `termination_loss()` | 在下一状态取 selected option beta，并对 `Q_o-maxQ+margin` 做 stop-gradient；基本方向与 Eq.5 一致。但对 advantage 做额外 clamp，是自定义 objective。注释声称“所有样本1:1推力”不准确：adaptive margin时，best option约 `+0.8g`，second约 `-0.2g`，只有足够差的 option 才被截到 `-0.8g`。 |
| `termination_loss_at_next_state()` | 确保 beta loss 在 `s'` 重新前向，修复了早期可能误在 `s` 更新的问题；正确。 |

### 6.4 source compatibility 与 distillation

| 函数 | 审查 |
|---|---|
| `gaussian_action_compatibility_all()` | 对 masked mean squared action distance 使用 Gaussian kernel。它是确定性 actor 无 density 时的合理 surrogate，但不是发布代码的逐维 `mu±sigma` hard interval。soft weight 可以提高平滑性，也改变了 Q_o estimand。 |
| zero-mask guard | GLM 新增 guard 能让未被 selected 的退化 source compatibility=0；但调用方随后 `compat=max(compat,selected_oh)`，所以一旦 zero-mask source 被 selector 选中，它仍以1更新 Q。现有 bank 没有 zero mask，因此不是当前实验 bug；更完整做法是在 bank 加载时 fail-fast。 |
| `masked_action_distillation_loss()` | 对 active action dims 求 Huber/MSE/L1 并按 active dim 数归一，正确。zero mask 返回0，不报错。 |

### 6.5 adapters / schema

| 函数 | 审查 |
|---|---|
| `IdentityObsAdapter` | 默认严格要求维度相等；显式 opt-in 才截断/补零，设计正确。 |
| `SliceObsAdapter` | 支持 indices 或区间，输出维度严格检查；正确。 |
| `RobotOnlyObsAdapter` / `ReachObsAdapter` | 类型别名，本身不增加逻辑；语义完全依赖配置的 indices/start/end。`ReachObsAdapter` 默认取前缀只适合同任务 smoke，不能自动理解任务实体。 |
| `HumanoidBenchRobotQposQvelAdapter` | 正确处理“full qpos + full qvel”中 robot qvel 不紧接 robot qpos 的情况；是重要修复。 |
| `ActionPassthroughAdapter` | 默认严格维度；正确。 |
| `ActionPadAdapter` | 支持 source→target index 映射；没有显式检查两组索引长度、重复和越界，错误通常在 tensor assignment 时才暴露。 |
| `build_obs_adapter()` / `build_action_adapter()` | 配置工厂；类型路由清楚。 |
| `build_action_mask()` | 支持全量、indices、ranges、named groups；没有拒绝 zero mask 或非法空组。 |
| `ActionSlice.mask()` / `ActionSchema.mask()` | 简单、正确。 |

### 6.6 `SourcePolicy` 与 `SourcePolicyBank`

| 函数 | 审查 |
|---|---|
| `_strip_compile_prefix()` | 去掉 `_orig_mod.`；用于加载 compiled checkpoint，正确。 |
| `_load_matching_state()` | 仅加载名称和 shape 匹配的 tensor，其余静默跳过并打印摘要。**风险较高**：只要任意层匹配就接受，可能留下随机初始化的输出层/中间层，产生看似能运行的错误教师。对 source actor 应要求完整核心权重匹配。 |
| `_IdentitySourceNormalizer.normalize()` | 真 identity。 |
| `_FrozenOfficialEmpiricalNormalizer` | 用官方 `_mean/_std` 和 eps=1e-2 推理，基本对齐。应额外验证 shape/count/schema。 |
| `SourcePolicy.__init__()` | 识别官方、upstream-copy、legacy actor；加载冻结权重、normalizer、adapter、mask。整体工程链合理；主要问题是 partial load 准入过宽。 |
| `act()` | target raw obs→source adapter→source normalizer→frozen actor→target action adapter；链路正确。 |
| `from_spec()` | manifest 与 YAML merge，显式 YAML 覆盖 manifest；合理，但缺 schema/version/冲突字段的强验证。 |
| `SourcePolicyBank.__init__()` | ModuleList + masks/sigmas + optional null；正确。未检查 source name 唯一性、mask非空、至少一个 option。 |
| `from_config()` | 加载全部 source；正确。 |
| `act_all()` | 每个 source 对同批 raw obs独立推理，返回 `[B,S,A]`；正确但计算开销线性。 |
| `act_selected()` | 只对被选 source 的子批推理；null返回全0 action/mask且 active=false；蒸馏语义正确。 |
| `names()` | 返回 source名并追加null。 |

## 7. `PTFFastTD3Agent` 逐函数审查

### 7.1 `__init__()`

正确建立：

- FastTD3 base；
- online/target option module；
- 独立 option/beta optimizer；
- transfer lambda 和 option epsilon scheduler；
- per-env call-and-return selector。

与论文/发布代码的差异：

- option target 每步 Polyak (`tau=.05`)，发布代码每 `replace_target_iter` hard copy；
- Q target、beta target和max action选择都来自 target module；发布代码用 online beta、online argmax 选择 target Q；
- option 与 beta optimizer 都包含共享 trunk和两头的全部参数。由于各 loss 的依赖图，未参与 loss 的 head通常无梯度，共享 trunk会被两类更新依次修改；这与发布代码两个 optimizer都作用于 `q_net` 的结构近似。

### 7.2 `act_with_options()`

流程为：归一化当前 obs→option selector决定教师标签→调用 `self.act()` 让 student行动。这个实现**忠实于 PTF 论文 Algorithm 1 Line 8**：动作由目标策略 `pi(s|theta)` 产生，source仅提供 transfer loss。

它不应被描述为“selected source executed”。这个措辞在当前 GLM 加入的 `train_ptf.py` 注释中出现，和 PTF 原论文/发布代码不一致。

### 7.3 `compute_actor_extra_loss()`

| 步骤 | 裁决 |
|---|---|
| 从 replay 读取历史 `option_ids` | 可运行，但比 PTF A3C 使用近期 trajectory 更 off-policy；旧 option 决策可能已过时 |
| `source_bank.act_selected(raw_obs, option_ids)` | 正确使用 source 自己的 normalizer/adapter |
| current `pi_action` 对历史 state蒸馏 | 符合 off-policy actor训练形式 |
| Huber/MSE 替代 cross-entropy | 确定性 TD3 的必要适配 |
| `lambda(t)*(1-beta_o)` | 对齐论文 Eq.7；比发布代码更接近论文 |
| `beta_o.detach()` | 正确，actor distillation不应借机改 beta |
| active-count normalization | 保持 null占比变化时 loss scale稳定，合理 |
| scalar loss clip | 自定义稳定化；会截断过强 transfer gradient |

最重要的理论张力是：**replay 中的 option id 是采集时 option network 的决策，actor 更新时却使用当前 source动作与当前 beta。** 当 selector已改变或 source在当前阶段已无益时，旧标签仍可继续蒸馏。这正是 off-policy PTF 缺少 source lifecycle 的早期形式。

### 7.4 `update()`

顺序为 critic→actor→FastTD3 targets→option Q/beta→option target。逻辑闭合。option每个 critic update都更新一次；在默认 UTD=2 时每个 vector step更新两次，需把 beta warmup 的 step单位与此区分。

### 7.5 `update_option()`

正确部分：

- 使用 next observation构造 `U(s',o)`；
- terminal不bootstrap、time-limit truncation仍bootstrap；
- 支持一条 transition更新多个 compatible options；
- termination重新在 `next_obs` 前向；
- 记录 option/beta/compatibility诊断。

关键差异/问题：

1. **selected option被强制 compatibility=1**：`compat=max(compat,selected_oh)`。这与发布代码一致，但比论文 Algorithm 2 更宽。因为实际 action来自 student，不是 source，selected source完全可能不兼容；把该 transition的回报全权归给 source会污染 Q_o。
2. **null option对所有 transition compatibility=1**：在旧分模块版本里全部行为都是 student，所以这大体可解释为 student/self option value；在后来含 source行为数据的官方集成路径里，该含义会变成混合 behavior value，需在下一阶段单独审查。
3. **termination对 terminal transition也更新**：`valid`只检查 option id，不排除 `dones & ~truncations`。发布代码在 `not done` 时才更新 termination；terminal处不再有下一 option决策，训练 beta没有明确语义。
4. **target定义不同**：全 target Q/beta虽然稳定，但不是论文/发布代码的同一 estimator；代码中没有明确记录这一差异。
5. **compatibility是soft kernel**：这改变了 loss权重，不只是布尔准入。

### 7.6 `soft_update_option_target()`

标准 Polyak更新，工程上正确；是相对 PTF hard-copy target的稳定化改写。

### 7.7 `state_dict()` / `load_state_dict()`

PTF 特有 option/beta optimizer能够恢复；但 base class没有恢复 actor/critic optimizer，整个 agent仍不具备完整训练 resume。selector current option、selector RNG、scheduler progress、update counters同样未保存。

## 8. 现有测试覆盖与缺口

历史测试覆盖了：

- actor/critic/option shape；
- upstream actor done后重采 noise；
- C51/agent update可以运行；
- policy delay时序；
- option state保存/恢复；
- termination使用 next observation；
- adapter shape与 action schema；
- replay包含 option id；
- source checkpoint可加载；
- selector不污染全局 RNG；
- beta输出落在配置区间。

但它们多数是 smoke/shape测试，没有覆盖以下科学关键点：

- 与官方 FastTD3同初值同 batch 的数值等价性；
- AMP、LR schedule、normalizer轨迹差异；
- optimizer/counter/RNG resume；
- `global_step` 的 transition/vector/update单位；
- terminal transition不应训练 beta；
- incompatible selected source是否应强制更新；
- 历史 option标签对当前 actor的长期影响；
- strict source checkpoint加载；
- partial state load随机层；
- zero mask在 selector选中后的端到端行为；
- student/null option的真实 estimand。

## 9. 问题清单与优先级

### 9.1 Blocker：阻止“严格复现”声明

| ID | 问题 | 影响 | 建议 |
|---|---|---|---|
| B-01 | `40b04cc` 缺失整个 `my_fasttd3_ptf/models/` | 旧版本无法从 Git重建、测试或复现实验 | 保留本审计的证据说明；不要再修复/复活旧路径，正式工作走官方集成路径 |

### 9.2 High：会改变科学结论或训练语义

| ID | 问题 | 影响 | 建议 |
|---|---|---|---|
| H-01 | base `load_state_dict()`不恢复actor/critic optimizer和计数器 | latest checkpoint续跑不等价 | 旧版不再维护；在官方路径确认完整resume边界 |
| H-02 | schedule传入的是总transition数，官方global step是vector step | lambda/epsilon/beta warmup可能相差num_envs倍 | 下一阶段逐CLI/配置冻结单位；文档禁止只写“steps” |
| H-03 | actor从长期replay使用历史option标签蒸馏 | 旧教师决策过期后仍施加梯度，可能产生持续负迁移 | 正式方法需provenance/lifecycle或只用当前准入一致数据 |
| H-04 | selected source即使与student action不兼容也被强制更新Q_o | Q_o不再估计“source可产生该动作时的价值” | 把它标为发布代码选择而非论文必需；官方路径做消融前先明确estimand |
| H-05 | source checkpoint只要求任意匹配tensor即可接受 | 可静默加载半随机教师 | 未来修复应要求所有actor核心参数完整匹配，adapter只解决输入/输出语义，不应掩盖网络缺层 |
| H-06 | 旧版缺AMP/LR scheduler且normalizer路径不同 | 不能把旧版结果当官方FastTD3同等baseline | 正式实验只用官方集成路径 |

### 9.3 Medium：局部公式、边界或解释需收窄

| ID | 问题 | 影响 | 建议 |
|---|---|---|---|
| M-01 | terminal transition仍更新beta | 无意义terminal状态可扰动termination | 官方路径检查并考虑用 `valid & bootstrap`/nonterminal mask；先补单测再改 |
| M-02 | beta限幅被描述为“防sigmoid饱和/梯度永不消失” | 科学解释错误 | 改为“保证输出gate floor，不保证logit梯度恢复” |
| M-03 | adaptive advantage clamp被描述为普遍1:1 | 与实际梯度比例不符 | 精确写成“限制最坏不对称”，不要声称全样本平衡 |
| M-04 | option target estimator与PTF发布代码不同 | 复现边界不清 | 下一阶段把online-beta/double-selection/Polyak差异独立列出 |
| M-05 | random warmup仍存selected option标签 | modular core配置下Q_o归因错误 | 若保留旧版应随机warmup存option=-1；当前正式路径另审 |
| M-06 | PerEnvReplayBuffer未强制batch可整除num_envs | 非默认配置实际batch变化 | 加参数验证即可；旧版不必优先修 |
| M-07 | adapter/mask config缺强schema校验 | 错配置可能静默运行 | source bank加载时验证唯一name、非空mask、索引映射与完整权重 |

### 9.4 Low/说明性

- callbacks大多不是实际训练入口使用的路径；`RenderCallback`只是占位。
- flat replay可以用于toy，但不是FastTD3正式语义。
- source-free eval是刻意设计，不是遗漏source option。
- actor target在upstream模式默认不参与target action，保留对象略冗余但不改变结果。

## 10. 对 GLM 2026-07-21 审计的修正

GLM 的大量模块定位是有价值的，但有三处需要纠正：

1. **“原始 PTF 的 Q_o 是 on-policy、只更新实际执行option”不准确。** 论文 Algorithm 2明确从 replay采样，并允许一条样本更新所有能选择该动作的options，原文直接称其为 off-policy sample reuse。更准确的张力不是“从on-policy改为off-policy”，而是“在TD3长期replay、确定性source surrogate和历史option标签下，off-policy程度远高于原PTF短期A3C经验”。
2. **“selected option actually controlled/executed behavior”不准确。** PTF论文和发布A3C代码正常分支都由student actor执行环境动作；source只提供互补loss。发布代码的`selected OR compatible`是实现选择，不能用“source执行了该动作”解释。
3. **zero-mask修复不是完整闭环。** compatibility函数返回0后，调用方仍可能用`selected_oh`强制抬回1。现有bank无zero mask，所以这是防御性边界，不是影响既有结果的bug修复。

此外，GLM新增的beta/termination注释重复了两个过强表述：输出限幅不能阻止内部sigmoid饱和，adaptive clamp也不普遍1:1。这些属于解释问题，不应为了修注释而拖延实验，但正式论文/审计文档必须收窄。

## 11. 第一阶段最终裁决与下一阶段问题

### 11.1 已支持

- 分模块版拥有可运行的FastTD3核心和PTF辅助蒸馏链。
- student而非source执行环境动作，符合原PTF设计。
- `U(s',o)`、call-and-return、next-state beta、multi-option replay更新的核心方向存在。
- deterministic action distillation、obs/action adapter、body mask是合理且必要的HB适配。

### 11.2 不支持

- 不支持“旧分模块版是官方FastTD3严格复现”。
- 不支持“它已经实现reward-bearing source trajectory injection”。
- 不支持“beta限幅解决了termination学习”。它只保证transfer gate不彻底为0。
- 不支持“option Q/termination已经可靠衡量HB迁移性”。代码可运行不等于判断有效。
- 不支持把GLM的on-policy解释或“source实际执行”解释写进论文。

### 11.3 第二阶段必须回答

审查 `official_fasttd3_ptf/` 时，重点不再重复网络shape，而要检查：

1. 是否真正保持官方FastTD3 actor/critic/replay/AMP/scheduler/update顺序；
2. reward-bearing source behavior如何与student行为组合；
3. option id、source provenance、admission snapshot和replay eligibility是否一致；
4. actor/critic是否共享同一采样分布；source撤销后旧数据是否退出；
5. null/student option在混合行为replay中的estimand是否仍成立；
6. MCG是否绕过经典Q_o/beta路径，以及哪些模块是活跃、可选或dead path；
7. GLM本轮改动是否仅修改注释/防御边界，还是改变正式实验语义。

本文件是第一阶段的审计基线；后续发现应追加版本，不覆盖历史裁决。
