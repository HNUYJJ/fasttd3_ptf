# PTF-FastTD3 官方 FastTD3 集成版逐函数审计（第二阶段）

> 日期：2026-07-21  
> 审计者：ChatGPT / Codex  
> 主入口：`scripts/official_fasttd3_train_target_ptf.sh` → `python -m fasttd3_ptf.official_fasttd3_ptf.train_ptf`  
> 当前代码锚点：HEAD `a5de1a3f55c3088ed250a68c9e5dc4a45cea459f` 加审计时工作树中的用户/GLM未提交改动  
> 对照基准：`papers/FastTD3.pdf`、`fasttd3_ptf/official_code/FastTD3/fast_td3/`、`papers/PTF-arxiv.pdf`、`reference_source_code/PTF_code/`  
> 前置审计：`docs/agent_collab/chatgpt_ptf_fasttd3_modular_function_audit_20260721.md`  
> 本轮只做代码、论文、官方源码和既有测试审计；没有修改训练机制，没有启动新训练。

## 1. 先给总裁决

官方集成版本的性质应拆成两层判断：

1. **FastTD3 learner 核心基本忠实保留。** Actor/Critic 直接使用仓库内 vendored 官方类；critic 仍是 twin C51，target projection、target-policy smoothing、CDQ、AMP/scaler、AdamW、cosine learning-rate schedule、delayed actor update 与 Polyak target update 的主体顺序均与官方训练代码一致。`PTFReplayWrapper` 包装官方 `SimpleReplayBuffer`，没有重写其 transition storage。
2. **PTF/MCG/RBO/admission 是叠加在官方 learner 周围的研究扩展，不是 PTF 原论文的逐字复刻。** 它同时保留了经典 PTF 蒸馏、可选 source 行为执行、MCG、warmup bootstrap、student-inclusive admission、provenance、replay lifecycle、anchor fork 等多套机制。不同 CLI 组合激活的是不同算法，不能笼统称为一个“PTF-FastTD3”。

相较历史分模块版，当前版本显著更适合作为正式实验基础：它保留了官方 FastTD3 核心训练配方，具有更完整的 source provenance、准入状态、replay 生命周期和实验锚点工具。然而本轮发现一个新的实质缺陷：

> **当 authority quota 仍有效、只撤销部分 source、且一条 MCG transition 混用多个 source 时，`uniform_mix>0` 会把已判定为 forbidden 的 mixed-source slot 重新赋予正采样权重。** 因而“任一贡献 source 被撤销后，该 transition 立即退出 active replay”这一通用声明目前不成立。

这个缺陷有明确适用边界：它不推翻已经完成的 SAFE/WFix、P0 或 Phase-1 bounded bank lease 结果，因为这些正式路径在 source bootstrap 中是组同步的单-source完整动作，Phase-1 又在 30k 整库 hard-exit 后切到 physical-allowed sampling；它会阻塞未来“post-warmup模块化混合动作 + 部分source在线撤销 + quota sampling”的一般算法声明。

因此第二阶段最终结论是：

> **官方集成版是当前唯一应继续维护的正式训练路径；FastTD3核心可视为高可信官方派生实现，但经典 PTF、MCG、admission和replay lifecycle必须按运行模式分别声明。当前最优先的代码修复不是重新设计迁移性指标，而是修复 mixed-source exact-revoke 的一行权重掩码并补覆盖其真实反例的单测。其余发现大多是历史/可选路径或声明边界，不应再次演变成长期细节审查。**

## 2. 证据、版本与工作树边界

### 2.1 审计对象

- 官方集成训练入口：`fasttd3_ptf/official_fasttd3_ptf/train_ptf.py`。
- 官方核心副本：`fasttd3_ptf/official_code/FastTD3/fast_td3/`。
- PTF共享模块：`fasttd3_ptf/ptf/`。
- replay/admission/anchor扩展：`fasttd3_ptf/official_fasttd3_ptf/`。
- launcher：`scripts/official_fasttd3_train_target_ptf.sh`。
- 当前正式机制结果边界：`docs/phase1_bounded_bank_lease_results_20260720.md`。

审计时工作树不是干净状态。以下是用户/GLM已有改动，本轮没有覆盖或回退：

- `fasttd3_ptf/official_fasttd3_ptf/train_ptf.py`：新增解释性注释；
- `fasttd3_ptf/official_fasttd3_ptf/ptf_replay.py`：扩写 n-step 限制说明；
- `fasttd3_ptf/ptf/compatibility.py`：新增 zero-mask compatibility guard；
- `fasttd3_ptf/ptf/option_module.py`、`option_update.py`：新增解释性注释；
- roadmap、guardrail和GLM审计文档。

### 2.2 论文给出的关键基准

PTF论文 Algorithm 1/2 的事实边界是：

- 环境动作由目标策略 `pi(s|theta)` 产生；source policy是辅助优化目标，不是默认行为控制器；
- option按 `Q_o` 与 termination beta 形成 call-and-return；
- `Q_o` 从 replay采样更新；
- 一条 transition可更新多个能够选择该动作的option，论文明确将其称为 off-policy sample reuse；
- transfer loss由时间schedule与 `(1-beta)` 调节。

FastTD3论文与官方代码的关键配方是：

- 大规模并行环境与32768量级大batch；
- twin distributional critic（C51）；
- CDQ；
- target-policy smoothing；
- delayed deterministic actor update；
- per-environment GPU replay；
- mixed exploration-noise schedule；
- AMP、AdamW和cosine LR schedule。

这两组基准意味着：不能把“PTF本来是on-policy”或“selected source本来就控制环境”作为当前改写的依据；也不能因为引入了source行为轨迹，就把所有路径都称为原始PTF。

## 3. 当前代码实际包含的算法模式

| 模式 | 关键开关 | 环境动作 | actor额外loss | Q_o/beta | replay策略 |
|---|---|---|---|---|---|
| 纯FastTD3 / empty bank | 无source，或静态exact abstention | student | 无 | 不更新 | 官方uniform |
| 经典PTF蒸馏 | `mcg=false, execute_sources=false` | student | `lambda(1-beta)` source蒸馏 | 更新 | 官方uniform或历史source权重 |
| 经典PTF行为执行 | `mcg=false, execute_sources=true` | selected source或student | 同上 | 更新 | 写入实际source/student动作 |
| MCG warmup random/SAFE | `mcg=true`且warmup active | 按segment组同步选择source或student | warmup期关闭 | 不更新 | 可为uniform/legacy重权/准入quota |
| student-inclusive admission bootstrap | `warmup_mode=admission_bootstrap` | admitted sources与student统一categorical；无source时100% student | warmup期关闭 | 不更新 | provenance-stratified quota |
| post-warmup MCG gate | `ablation=full/no_bootstrap` | 每个body group可由不同source或student控制 | critic-gated group distillation | 不更新 | 可出现mixed-source provenance |
| bootstrap-only | `ablation=bootstrap_only` | warmup source/student；之后纯student | 始终无MCG蒸馏 | 不更新 | source尾部按配置保留或退出 |

正式实验解读必须先说明是哪一行。尤其：

- SAFE/WFix、admission handoff、P0和Phase-1大量实验走的是 `MCG + bootstrap_only`，并不训练经典 `Q_o/beta`；
- 经典PTF路径不是这些reward-bearing bootstrap结果的产生机制；
- MCG gate也是可选模块，`bootstrap_only≈full` 的既有证据不支持把它写成主性能来源。

## 4. 端到端调用链

```text
official_fasttd3_train_target_ptf.sh
  -> train_ptf._parse_ptf_cli + official hyperparams.get_args
  -> _make_envs
  -> official Actor / Critic / SimpleReplayBuffer
  -> PTFReplayWrapper(SimpleReplayBuffer)
  -> SourcePolicyBank.from_config
       -> source checkpoint + frozen normalizer
       -> observation adapter
       -> action adapter + body mask
  -> OptionModule / OptionSelector                       [classic path]
  -> ModularGating / McgBehaviorController              [MCG path]
  -> AdmissionSnapshot / Schedule / AdaptiveController  [admission path]
  -> rollout
       -> student actor action
       -> optional classic full-source execution
          OR optional MCG warmup/gated action composition
       -> target env reward
       -> transition + option id + provenance -> replay
  -> replay sample
       -> official C51 critic update
       -> delayed official actor RL update
       -> optional classic PTF distillation
          OR optional MCG group distillation
       -> classic Q_o/beta update only when MCG is off
       -> target critic Polyak update
  -> source-free deterministic student evaluation
  -> standard checkpoint OR paper-grade anchor bundle
```

## 5. 入口、环境、RNG与保存函数逐项审查

### 5.1 `train_ptf.py`

| 函数 | 实现逻辑 | 裁决 |
|---|---|---|
| `_parse_ptf_cli()` | 单独解析全部PTF/MCG/admission/anchor开关，再与官方FastTD3参数并存；支持下划线与短横线别名 | 功能完整，但运行模式很多；正式run card必须冻结完整PTF配置而非只写方法名 |
| `_make_envs(args,device)` | 延用官方各suite构造路径；HumanoidBench走本地wrapper | 与官方结构一致；HB增加确定性播种修复 |
| `_get_ddp_state_dict()` | DDP包装兼容 | 正确 |
| `save_ptf_params()` | 保存actor、双critic、normalizer、option/target、option与beta optimizer、PTF配置、admission审计 | 推理/结果保存足够；不包含actor/critic optimizer、scheduler、scaler、replay和RNG，不是完整训练resume |
| `main()` | 完整构建、rollout、update、eval、save和anchor流程 | 主流程闭合；各子路径见后续章节 |
| nested `make_actor/make_critic` | 直接实例化官方类或SimbaV2类 | 正确；FastTD3默认直接用官方类 |
| nested `evaluate()` | deterministic actor、`update=False` normalizer、只评student | source-free边界正确；但共享训练env问题见 `M-03` |
| nested `render_with_rollout()` | deterministic student render | 正确 |
| nested `update_main()` | 官方critic update加可选priority统计 | 核心数学保持，见第6节 |
| nested `compute_transfer_loss()` | 经典PTF selected-source action distillation | 是PTF→deterministic TD3适配，不是MCG路径 |
| nested `compute_mcg_transfer_loss()` | replay state上重算source actions和critic gate，group-wise distill | 逻辑闭合；在线critic与表述不一致，见第9节 |
| nested `update_pol()` | `-Q(s,pi(s)) + transfer_loss` | FastTD3 RL loss保持，transfer是加项 |
| nested `update_option()` | 经典Q_o/beta update | MCG开启时完全跳过；公式差异见第8节 |
| nested `soft_update()` | `torch._foreach` Polyak update | 与官方一致 |

### 5.2 HumanoidBench wrapper

| 函数/类 | 审查 |
|---|---|
| `GlobalNumpySeedOnReset.reset()` | 同时设置env自己的seed与worker进程global NumPy seed，修复HB任务内部直接调用`np.random`导致的跨seed污染 | 是必要修复，测试覆盖两次同seed/不同seed行为 |
| `max_episode_steps()` | 读取任务类的episode长度 | 正确 |
| `make_env()` | 关闭renderer、按rank播种、包装Monitor | 与官方HB并行路径相容 |
| `HumanoidBenchEnv.__init__()` | 构造SB3 `SubprocVecEnv`并暴露FastTD3接口 | 正常 |
| `reset()` | 返回device tensor | 正确 |
| `step()` | NumPy action→vec env→device tensor，并构造`time_outs`与真实terminal obs信息 | 与官方FastTD3 replay需要的接口一致 |
| `render()/close()` | 代理env | 正常 |

一个继承自官方代码的边界：`eval_envs = envs`。评估会reset训练env，评估后虽然重新获得了`obs`，但主循环的`dones`仍来自评估前一步，因此下一次actor exploration的episode-noise reset状态与严格不间断训练不同。这不是PTF特有偏差，scratch/PTF同配置会共同受到影响；P0/anchor协议已用`eval_interval=0`和离线evaluator规避。论文若强调精确learner trajectory，需使用独立eval env或离线评估。

### 5.3 RNG isolation

| 函数/类 | 审查 |
|---|---|
| `GlobalRngState.capture()` | 捕获Python、NumPy、Torch CPU和指定CUDA device RNG | 正确 |
| `restore()` | 恢复上述全局RNG | 正确 |
| `capture_rng_after_reference_construction()` | 在不污染调用方RNG的前提下构造参照对象并捕获其后状态 | 测试存在，但当前主训练没有调用 |

静态exact abstention在source/option/MCG构造后恢复`target_only_rng`，使其与empty-bank scratch共享目标学习RNG。一个未覆盖的边界是：**schedule若在step 0为空、以后才准入source，不会走静态fast path，也不会恢复构造前RNG。** 因此未来“延迟介入”实验在介入前就可能因无用模块构造消耗了global RNG而偏离scratch。当前Phase-1 schedule从step 0全源准入，不受此问题影响。

## 6. FastTD3核心逐项对照

### 6.1 模型与replay

- `Actor`、`Critic`直接从`official_code/FastTD3/fast_td3/fast_td3.py`导入；不是重新抄写的近似版本。
- `actor_detach`通过`TensorDict.from_module(actor).data.to_module(actor_detach)`共享参数storage，只保留独立runtime buffer。实测参数data pointer一致，修改/加载actor参数会立即反映到`actor_detach`；checkpoint load后不需要额外copy参数。
- `qnet_target`由online critic完整初始化。
- base replay是官方`SimpleReplayBuffer`；wrapper只增加option/provenance和自定义index selection。
- 集成版明确锁定`num_steps=1`。HB官方默认就是1，因此当前正式配置不损失官方功能；若启用n-step，option/provenance的时序语义确实需要重新定义。

### 6.2 `update_main()`

执行顺序与官方代码一致：

1. 从replay读取state/action/reward/done/truncation；
2. 计算terminal bootstrap mask；
3. online actor产生next action并加入clipped target-policy noise；
4. target twin C51分别投影；
5. 若启用CDQ，按较小期望值选择整条target distribution；
6. online twin critic logits对target distribution做cross entropy；
7. AMP scale/backward/unscale、可选grad clip、AdamW step、scaler update；
8. 可选计算TD magnitude供admission replay priority。

新增的priority只在`admission_replay_priority_alpha>0`时启用，默认正式配置为0，不改变FastTD3核心loss。

### 6.3 `update_pol()`

1. online actor在replay state上产生`pi_action`；
2. online twin distributional critic求期望；
3. CDQ配置下取较小Q，否则取平均；
4. RL actor loss=`-mean Q`；
5. 叠加classic或MCG transfer loss；
6. AMP/scaler、grad clip、AdamW step。

因此target-only或transfer loss为0时，actor更新主体与官方一致。

### 6.4 update频率、target与scheduler

- actor delay公式与官方一致：`num_updates>1`时在inner update满足`i % policy_frequency == 1`时更新；默认2 updates/2 frequency时每vector step更新一次actor。
- 每个critic inner update后均对`qnet_target`做Polyak更新，与官方一致。
- actor和critic scheduler每个vector step调用一次，`T_max=args.total_timesteps`，与官方一致。
- `run_stop_step`只提前停止fork，不压缩scheduler；这是P0分支正确性所需。
- 动态PTF路径默认关闭`torch.compile`，除非显式环境变量允许。它影响吞吐和数值执行路径，不改变算法方程；因此不能宣称和官方相同速度，但可以称核心更新公式保持。

### 6.5 标准checkpoint与anchor不是一回事

普通`--checkpoint-path`只恢复：

- actor、qnet、qnet_target；
- obs/critic normalizer；
- option与option_target（若存在）；
- global step。

它**不恢复**actor/critic optimizer、cosine scheduler、AMP scaler、replay、exploration/RNG、selector/MCG/admission状态，也没有加载保存过的option/beta optimizer。这个限制继承自官方FastTD3普通checkpoint设计，并在集成版中继续存在。因此普通checkpoint可用于推理、评估或近似续训，不能称为严格resume。

`anchor_io`则是专门为反事实fork设计的另一套paper-grade状态包，包含learner、optimizer、scheduler、scaler、replay、RNG、manifest和checksum。两者不能混用叙述。

## 7. Source policy、adapter与bank逐函数审查

### 7.1 `source_policy.py`

| 函数/类 | 审查 |
|---|---|
| `_strip_compile_prefix()` | 去除compiled模型的`_orig_mod.`前缀，正确 |
| `_load_matching_state()` | 只加载key和shape匹配tensor；对runtime `noise_scales`不匹配很实用，但只要求“至少一项匹配”过宽，可能接受部分随机网络 |
| `_IdentitySourceNormalizer.normalize()` | identity，正确 |
| `_FrozenOfficialEmpiricalNormalizer` | 用官方checkpoint中的mean/std做冻结推理，公式对齐官方normalizer |
| `SourcePolicy.__init__()` | 重建actor、加载checkpoint和normalizer、构造obs/action adapter与mask、冻结参数 | 主链正确；严格性缺口是partial load |
| `act()` | target raw obs→obs adapter→source normalizer→frozen actor→action adapter | 跨任务source执行的正确顺序 |
| `from_spec()` | 合并manifest和bank YAML，YAML覆盖manifest | 可用；缺schema/version/冲突字段强校验 |

对当前`official/h1hand_basic_sources.yaml`的实际CPU加载验证：stand/walk/run/reach均加载10个actor parameter tensor，只跳过`noise_scales`这一runtime buffer；当前官方source权重完整，没有证据表明既有实验用了半随机网络。风险面向未来错误checkpoint，建议把准入规则收紧为“所有actor核心parameter必须匹配，仅允许明确白名单runtime buffer不匹配”。

### 7.2 adapter函数

| 函数/类 | 审查 |
|---|---|
| `IdentityObsAdapter` | 维度不等默认报错，只有显式opt-in才截断/补零；正确 |
| `SliceObsAdapter` | indices或区间选择并检查输出维度；正确 |
| `RobotOnlyObsAdapter` | 默认取前`source_obs_dim`；只在目标obs前缀语义与source一致时成立 |
| `ReachObsAdapter` | 默认也是前缀；代码注释已承认跨任务manipulation需显式indices |
| `HumanoidBenchRobotQposQvelAdapter` | 从full qpos+full qvel中抽robot qpos/qvel，避免“前151维”在带task DoF任务上错位 | 是重要的HB适配 |
| `ActionPassthroughAdapter` | 维度严格identity；正确 |
| `ActionPadAdapter` | 将source indices映射到target indices | 缺长度、重复和越界的预检查 |
| `build_obs_adapter/build_action_adapter` | 配置工厂 | 正常 |
| `build_action_mask` | 支持full、indices、range、named body groups | 没有拒绝empty/zero mask；GLM在compatibility层加guard但没有从配置源头fail-fast |

`h1hand_basic_sources.yaml`中的reach默认前缀adapter只是模板/同任务smoke级安全，不应被当作任意目标任务的语义对齐保证。正式WFix/任务bank普遍使用显式adapter，需以每个bank为准。

### 7.3 `SourcePolicyBank`

| 函数 | 审查 |
|---|---|
| `__init__()` | ModuleList保存sources，缓存mask/sigma，附加可选null | 正常；应补source name唯一和mask非空验证 |
| `from_config()` | 从YAML/dict逐source构建 | 正确 |
| `act_all()` | 对所有source在同批raw obs推理，返回`[B,S,A]` | 正确，计算量随source线性增长 |
| `act_selected()` | 只为被选source子批推理，null返回inactive | 正确 |
| `names()` | source names加null | 正常 |

## 8. 经典PTF路径逐函数审查

### 8.1 `OptionModule`与`OptionSelector`

这些共享函数在第一阶段已逐函数审查；在官方集成运行时需强调：

- `OptionModule`是shared trunk + `Q_o` head + beta head；beta输出被映射到`[beta_min,beta_max]`；
- 这保证`1-beta`存在输出floor，但**不阻止内部sigmoid logit饱和，也不保证termination学习有效**；
- `OptionSelector.step()`实现per-env call-and-return、epsilon-greedy reselection和minimum duration；
- default `execute_sources=false`时，selector标签只决定蒸馏教师，student仍控制环境；
- `execute_sources=true`是本项目额外的reward-bearing behavior extension，不是PTF Algorithm 1的默认语义。

### 8.2 `compute_transfer_loss()`

- 从replay读取历史option id；
- 当前source对该历史state重新产生动作；
- current beta提供`1-beta`权重；
- deterministic actor使用masked Huber/MSE/L1代替策略cross entropy；
- loss按active sample数归一并乘时间lambda。

这是合理的TD3适配，但历史option标签可能已经过时，旧source仍能通过长期replay继续向actor施加梯度。经典路径没有admission provenance lifecycle解决这一点。

### 8.3 `update_option()`

正确部分：

- 从replay离线采样，符合PTF Algorithm 2的off-policy reuse方向；
- 使用`U(s',o)=(1-beta)Q_o+beta max Q_o`；
- transition可更新多个compatible options；
- beta在next state重新前向；
- option target做稳定化Polyak更新。

实质差异/风险：

1. action compatibility由确定性action距离的Gaussian surrogate定义，不是原发布代码的区间density判据；
2. `compat=max(compat,selected_one_hot)`会强制历史selected option以权重1更新，即使实际动作并非该source产生；
3. null option对所有transition compatibility=1。在`execute_sources=true`的混合behavior replay里，它不再是纯student value，而是混合行为数据上的self/no-transfer标签；
4. terminal transition仍训练beta，因为`valid`只检查option id，没有排除真正terminal；
5. target全部使用target Q和target beta，且用Polyak而非发布代码的hard copy，是稳定化改写；
6. beta warmup只是延迟梯度，不证明20k后`Q_o`已能判断迁移性。

最重要的GLM注释错误也位于这里：PTF Algorithm 2本来就从replay采样、可更新多个compatible options，并非“on-policy Q_o”；Algorithm 1默认也是student执行环境动作，并非“selected source actually controlled behavior”。这些错误只在注释中，没有改变数值，但不能进入论文解释。

### 8.4 zero-mask guard

GLM在`gaussian_action_compatibility_all()`中把zero-mask source的compatibility强制为0，局部逻辑正确，也不改变现有非zero-mask banks。然而调用方随后仍用`selected_one_hot`取max；若zero-mask source已被selector选中，最终compatibility仍会回到1。因此它是防御性改进，不是端到端闭环。更简单可靠的处理是bank加载时直接拒绝zero mask。

## 9. MCG逐函数审查

### 9.1 `ModularGating`

| 函数 | 审查 |
|---|---|
| `__init__()` | 构造指定body-group mask并拒绝组重叠；只支持h1hand 61维action schema | 当前HB主路正确；不是通用机器人实现 |
| `num_groups` | 返回group数 | 正确 |
| `deltas()` | 每个`(source,group)`只替换student action的该组；分别计算twin head paired difference，再取较小head delta | paired计算逻辑清楚，能降低min-head切换噪声 |
| `null_margins()` | 打乱state-action配对形成null delta并按group取quantile | 是经验校准heuristic，不是统计显著性保证 |
| `select()` | source mask→每组argmax→减margin→hard gate与sigmoid confidence | 正确执行定义；source全被mask时会得到`-inf`，上层exact abstention另行短路 |

重大声明边界：训练中的`qheads_value()`调用的是**online `qnet`**，不是`qnet_target`。源码顶部和`train_ptf.py`多处写“target critic gating”，与事实不符。它未必是算法bug，但论文与文档必须称“current online target-task critic”；若要改成target critic，需要重新实验，不能只改代码名。

此外，rollout行为gate把`norm_obs`直接传给critic。若`envs.asymmetric_obs=True`，critic期望privileged critic observation而这里给actor observation，维度/语义会错误。当前正式HumanoidBench是symmetric obs，因此既有HB实验不受影响；泛化到IsaacLab或其他asymmetric任务前应fail-fast或提供正确critic obs。

### 9.2 `mcg_distillation_loss()`

- 每个group只向当前best source的该组动作蒸馏；
- hard gate乘confidence作为权重；
- total是各group loss之和；
- 最终actor transfer loss除以“至少一个active group的sample数”，不是除以active group权重和。

这意味着同一sample有多个active group时transfer梯度自然更大。代码docstring已明确“不按组数归一”，因此属于设计选择，不是隐藏bug；论文必须如实说明。

### 9.3 `AdmissionSegmentTracker`

| 函数 | 审查 |
|---|---|
| `__init__()` | per-env记录active candidate、reward sum、length | 正确 |
| `observe()` | 累积segment reward，horizon/done自然结束时输出candidate与mean reward | 时序与behavior latch一致 |
| `discard_sources()` | source被撤销时丢弃未完成partial segment | 防止跨window复用残缺return；正确 |
| `_reset()` | 清相应env状态 | 正确 |

### 9.4 `McgBehaviorController`

| 函数 | 审查 |
|---|---|
| `__init__()` | 初始化random/SAFE/online/admission四种warmup、named CPU RNG、per-group latch、online arm统计 | 功能很多；必须按mode解读 |
| `_rand()` | named RNG，不污染global torch RNG | 正确 |
| `set_admitted_sources()` | 原子替换mask，并立即释放被撤销source的当前latch | exact behavior revoke成立 |
| `set_admission_policy()` | 同时更新mask、source logits与student logit | 正确 |
| `admission_probabilities()` | admitted sources + student统一softmax；无source时one-hot student且不抽source RNG | exact abstention核心执行正确 |
| `update_arm_reward()` | online mode按当前arm归属本步reward，count-based running mean/EMA | 是即时行为收益反馈，不是data utility或迁移性指标 |
| `step()` | done reset→expired latch重选→按mode构造source/student选择→组合group action→递减latch→返回审计指标 | 主体逻辑闭合 |

四种warmup语义：

- `random`：外层teacher概率，teacher内uniform source；
- `safe_bootstrap`：外层teacher概率，teacher内按静态bootstrap weight softmax；
- `online_bootstrap`：student作为arm、reward EMA在线竞争，但历史已证明不能普遍避免crawl/stair负迁移；
- `admission_bootstrap`：没有固定0.5 teacher floor，student与admitted sources统一竞争；无source时100% student。

warmup中所有group同步选择同一个source，若groups覆盖完整action，则等价于source完整闭环动作；post-warmup gate才可能每个group来自不同source。这一差异决定了mixed-source replay bug只影响后一类一般路径。

## 10. Admission control逐函数审查

### 10.1 immutable decision与schedule

| 函数/类 | 审查 |
|---|---|
| `_sha256()` | 文件hash | 正确 |
| `AdmissionSnapshot.__post_init__()` | 检查source名字唯一、长度一致、logit有限 | 正确 |
| `exact_abstain` | 无source admitted | 正确 |
| `admitted_names/admitted_tensor` | 便捷表示 | 正确 |
| `candidate_probabilities()` | source mask后与student一起softmax；exact abstain直接one-hot student | 核心语义正确 |
| `as_dict()` | 持久化决策、artifact/hash | 正确 |
| `AdmissionSchedule.__post_init__()` | 要求step 0起、严格递增 | 正确 |
| `snapshot_at()` | 返回不晚于step的最后决策 | 正确 |
| `_parse_names/_load_manifest` | 解析配置和quarantine manifest | 正常 |
| `build_admission_snapshot()` | 支持legacy/all/none/manifest等模式 | 外部或静态决策构造器，不是自动迁移性判断器 |
| `build_admission_schedule()` | 从YAML构建step-indexed immutable snapshots | 能表达延迟介入、逐source撤销和整库hard exit |

schedule应用在每个rollout iteration最前面。`step:30000`意味着transition 0..29999可用source，生成第30000个completed step前就应用新决策；Phase-1的窗口语义正确。

### 10.2 adaptive controller

| 函数/类 | 审查 |
|---|---|
| `CandidateWindowStatistics.as_dict()` | 统计序列化 | 正确 |
| `AdaptiveAdmissionWindowResult` | 记录window统计与可选snapshot | 正确 |
| `_RunningMoments.update/snapshot()` | Welford mean/variance与normal interval | 数值正确；interval只是heuristic |
| `AdaptiveAdmissionController.__init__()` | 固定window、z、min segments、persistence | 清楚 |
| `record_segments()` | 收集自然完成segment mean reward | 正确 |
| `maybe_close_window()` | `UCB(source)<LCB(student)`连续若干window后不可逆撤销 | 状态机正确，但信号已被实验否定为可靠迁移性指标 |
| `desired_admission_source_authority()` | warmup/postwarmup authority与exact abstain组合 | 用于replay quota→physical handoff；正确 |

adaptive controller能执行撤销，但不能因为代码有置信界就称其为“保守可靠的迁移性估计器”。项目既有adaptive实验已判失败，保留它的正确定位是审计复现/负结果基线。

一个措辞边界：`desired_admission_source_authority()` docstring称exact abstention non-revivable，但显式schedule以后仍可重新admit source。对adaptive不可逆状态机该说法成立，对schedule全局语义不成立。

## 11. Replay wrapper逐函数审查

### 11.1 状态与写入

| 函数 | 审查 |
|---|---|
| `__init__()` | 包装官方buffer，新增option、legacy source weights、admission policy、priority/write step、audit与provenance storage | 结构清楚；锁定n=1 |
| `enable_provenance()` | 分配8字段schema：behavior source、source per group、executed mask、segment id/step、anchor id、env rank、learner step | 足以支持source lifecycle审计 |
| `set_source_weights()` | legacy OBRW按actor/critic/both设per-source降权 | 没有admission exact zero语义；历史消融路径 |
| `set_admission_policy()` | 安装admitted mask、candidate masses、recency/uniform/priority参数，并记录policy event | source mask先把rejected mass置0；正确 |
| `set_admission_source_authority()` | quota与physical-allowed sampling切换 | 解决source物理尾部被固定quota过采样问题 |
| `clear_admission_policy()` | 清admission sampling state | 简单正确 |
| `update_priorities()` | 按显式replay index写priority | 正确；无importance correction见`M-05` |
| `ptr/valid_size/chronological_slot_indices()` | ring状态与时间顺序 | 正确 |
| `extend()` | 同一physical slot写option/provenance/priority metadata，再委托官方buffer写transition | option与transition索引对齐；正确 |

### 11.2 eligibility与quota

`_admission_allowed_slots()`的意图是正确的：

- 有完整provenance时，若`source_by_group`中任何贡献source已被拒绝，整条transition不允许；
- 无provenance时按canonical option id过滤；
- 未写provenance的slot也不允许进入active replay。

问题出在`_admission_slot_weights()`的uniform混合项。当前代码：

```python
quality = allowed.float()
...
within = (
    (1-uniform_mix) * quality / stratum_quality
    + uniform_mix / stratum_count
)
```

第二项没有乘`allowed`。如果forbidden mixed transition的canonical stratum里还有一个allowed transition，`stratum_count`由后者保持为正，forbidden slot就得到正uniform权重。

当前代码上的最小CPU反例：

- slot 0：`source_by_group=[0,1]`，canonical option=0；source 1已撤销，所以slot forbidden；
- slot 1：`[0,0]`，source 0仍准入，所以allowed；
- slot 2：student；
- candidate masses=`[0.8,0,0.2]`，`uniform_mix=1`。

实际结果：

```text
allowed = [[False, True, True]]
weights = [[0.8, 0.8, 0.2]]
10k draws = [4402, 4429, 1169]
```

forbidden slot反而获得与allowed source slot相同的0.8未归一权重，采样约44%。这不是浮点容差，而是逻辑mask遗漏。

最小修复应是让uniform项也受`allowed`约束，例如：

```python
within = (
    (1-uniform_mix) * quality / stratum_quality
    + uniform_mix * one / stratum_count
)
```

或在返回前对整个`within`乘`one`，并新增“同一canonical stratum同时含forbidden mixed slot和allowed pure slot”的回归测试。

### 11.3 sampling、audit与snapshot

| 函数 | 审查 |
|---|---|
| `_record_admission_samples()` | 按canonical option累计actor/critic sample counts | 可审计quota，但mixed transition只归一个canonical source |
| `draw_indices()` | exact-none走官方`randint`；authority active走quota；authority off走physical allowed；legacy权重走multinomial | 分支设计合理；mixed uniform bug位于quota权重函数 |
| `admission_audit()` | 报告main/active counts、effective masses、sample counts与events | 有用；counts是canonical strata，不是所有contributing source计数 |
| `gather()` | 用同一per-env indices收集官方全部tensor、option与provenance | 正确，支持asymmetric replay字段 |
| `sample()` | draw+gather | 正确 |
| `_base_tensor_names()` | 列出基础buffer字段 | 正确 |
| `export_valid()` | 按chronological ring顺序导出compact snapshot | 正确 |
| `import_valid()` | 验证schema/shape并恢复ring、admission状态和provenance | anchor fork所需；测试覆盖roundtrip |
| `provenance_enabled/max_provenance_segment_id/assert_complete_provenance()` | 状态查询、segment namespace续接、完整性gate | 正确 |

两个额外边界：

1. legacy非admission MCG只把`current_arm[:,0]`写入option。post-warmup某source若只控制arms/hands而不控制group 0，legacy OBRW根本看不到它。这是历史路径的provenance不足；admission路径已用`source_by_group`补全。
2. priority sampling没有importance-sampling correction。`priority_alpha=0`的正式实验不受影响；若以后开启，它应被解释为有意改变训练分布的heuristic，而不是无偏PER。

## 12. Anchor与paper-grade fork逐函数审查

| 函数 | 审查 |
|---|---|
| `_cpu_tree/_module_state/_jsonable/_sha256` | 序列化与hash辅助 | 正常 |
| `_git_state()` | 保存HEAD、dirty状态hash | 正确，但dirty hash不能重构源码内容 |
| `capture_rng_state()` | 保存global RNG与named generators | 单GPU可用；见active device边界 |
| `restore_global_rng_state/restore_rng_state()` | 最后恢复RNG并验证named generator集合 | 正确 |
| `save_anchor_bundle()` | 原子写learner/replay/RNG/checksum/manifest，验证replay ptr与completed step | paper-grade完整状态明显优于普通checkpoint |
| `verify_anchor_bundle()` | 校验hash/size与manifest一致 | 正确 |
| `load_anchor_bundle()` | strict恢复全部声明组件、replay与RNG | 正确 |
| `load_anchor_core()` | 只恢复FastTD3核心白名单，option/MCG按fork重建；验证scheduler step与reward normalizer | 为P0设计，语义清楚 |

两项一般化边界：

- `capture_rng_state()`使用`torch.cuda.current_device()`，而`main()`只构造`cuda:{device_rank}`，没有显式`torch.cuda.set_device(device_rank)`。在多GPU都可见且训练device rank不是current device时，可能保存错CUDA RNG。P0通过单卡可见/设备0运行，既有结果不受影响；通用多GPU运行前应修。
- anchor manifest的`code_paths`覆盖main、anchor、replay、HB wrapper和官方FastTD3核心，但不包含全部source/MCG/admission共享文件。正式P0 orchestrator另行冻结plan inputs，因此P0证据闭合；单独使用anchor manifest时不能声称完整branch源码溯源。

## 13. Rollout与provenance时序核查

### 13.1 schedule与authority

每轮先应用到期schedule，再计算desired replay authority，再产生行为。这保证：

- step 30000 hard-exit不会再产生第30000步source transition；
- behavior latch会被`set_admitted_sources()`立即释放；
- replay policy在下一次sample前已更新；
- exact abstain时source行为和source active replay都能在同一边界停止。

### 13.2 行为动作

- target-only：student actor；
- MCG warmup：按segment从source/student选择，group同步；
- post-warmup MCG：online critic delta决定每个group候选；
- classic：默认student，只有`execute_sources`才用source完整动作。

所有写入replay的reward都来自target task环境，因而reward-bearing bootstrap定义成立。

### 13.3 provenance

admission路径为每条transition记录完整`source_by_group`。`behavior_source/option_id`只是canonical source，作用是quota stratum和简化audit，不足以单独表示mixed action。代码在eligibility过滤时正确使用了full group provenance；本轮bug是随后uniform weight没有继续mask，而不是provenance没记录。

`admission_execution_counts`也按canonical source计数。因此它能精确审计“source vs student总行为占比”和warmup组同步单source情形，不能在post-warmup mixed MCG中当作每个source真实body-group贡献次数。

## 14. 现有测试、实测验证与覆盖缺口

### 14.1 本轮执行

```bash
PYTHONPATH=. conda run -n FastTD3 pytest -q
```

结果：`211 passed, 11 warnings`。warning均来自matplotlib/pyparsing弃用提示。

定向核心组也独立通过：option、selector、MCG、source、admission、replay snapshot、anchor与RNG共`68 passed`。

附加只读验证：

- 当前official source bank四个source均完整加载10个actor parameter tensor，只跳过runtime `noise_scales`；
- `actor`/`actor_detach` parameter storage确实共享；
- mixed-source revoke反例在当前代码稳定复现为forbidden slot约44%采样。

### 14.2 已覆盖的关键语义

- exact no-source fallback；
- admitted source mask与即时latch release；
- schedule决策更新；
- provenance roundtrip与anchor import/export；
- quota不依赖历史stratum count；
- authority handoff回到physical uniform；
- mixed transition在“被撤销canonical stratum没有其他allowed slot”时退出；
- anchor core resume组件、scheduler和RNG；
- Phase-1 Gate A和机制checkpoint审计。

### 14.3 关键缺口

1. mixed forbidden slot与同canonical stratum allowed slot共存；
2. MCG + asymmetric critic obs fail-fast/正确forward；
3. delayed schedule在source介入前与scratch的RNG等价；
4. ordinary checkpoint严格resume应明确拒绝或标注；
5. source actor核心parameter必须全匹配；
6. split actor replay时checkpoint `actor_sampling`审计字段正确性；
7. full MCG mixed动作下execution audit按contributing source统计；
8. classic PTF terminal beta mask与selected-source强制compatibility的科学消融。

## 15. 问题分级与适用范围

### 15.1 Blocker/High

| ID | 问题 | 影响范围 | 是否影响既有主结果 | 建议 |
|---|---|---|---|---|
| B-01 | quota sampling的uniform项未乘allowed mask，mixed-source revoked transition可被采样 | full/post-warmup MCG + partial revoke + authority active + `uniform_mix>0` | 不影响组同步bootstrap、P0、Phase-1；阻塞未来通用exact revoke声明 | 一行mask修复+真实反例单测，最高优先级 |
| H-01 | 普通checkpoint并非完整resume | 所有从`--checkpoint-path`继续训练的run | 若只用于eval无影响；严格续训声明无效 | 明确标为weights resume，严格fork只用anchor |
| H-02 | source checkpoint只需任意tensor匹配 | 未来错误/异构checkpoint | 当前official sources实测完整，不影响既有结果 | 核心parameter全匹配，runtime buffer白名单跳过 |
| H-03 | 经典Q_o强制selected compat=1、terminal beta仍更新、null在混合行为中的estimand漂移 | classic PTF，MCG关闭 | SAFE/WFix/admission bootstrap均跳过Q_o/beta | 若重启经典PTF研究，先冻结estimand和修/消融；当前不优先 |
| H-04 | legacy OBRW只记录group 0 arm | 非admission post-warmup mixed MCG重权路径 | 不影响admission provenance正式路径 | 历史路径只保留复现，不再作为主方法 |

### 15.2 Medium

| ID | 问题 | 边界/建议 |
|---|---|---|
| M-01 | MCG文档称target critic，代码用online critic | 修正文档；若改算法需新实验 |
| M-02 | MCG rollout不支持asymmetric critic obs | HB当前不受影响；其他suite前fail-fast |
| M-03 | eval复用训练env并reset occupancy | 继承官方；严谨fork用offline eval/独立env |
| M-04 | checkpoint audit字段把所有actor sampling写成`shared_critic_batch` | split历史消融metadata不准确；根据mode动态写 |
| M-05 | priority sampling无IS correction | 默认alpha=0不影响；启用时称biased heuristic |
| M-06 | audit counts按canonical source，不是全部contributors | warmup单source准确；full MCG需另报group contributor counts |
| M-07 | delayed schedule初始none不恢复target-only RNG | 当前Phase-1不受影响；延迟介入实验前隔离构造RNG |
| M-08 | anchor CUDA RNG按`current_device`而非显式training device | 单卡rank0不受影响；多GPU可见时修复 |
| M-09 | adapter/mask/source names缺强schema检查 | 增强bank load fail-fast即可 |

### 15.3 说明性而非bug

- `torch.compile`默认关闭：速度差异，不是算法数学错误。
- MCG loss多group不按group数归一：代码明确的loss-scale选择。
- replay `uniform_mix`是在每个provenance stratum内部提供coverage，不是全buffer global uniform。
- `physical_after_authority`不会删除物理数据，只改变active sampling；这是审计保留设计。
- exact abstention是执行能力，何时应abstain仍需要外部决策；当前没有可靠在线迁移性oracle。

## 16. 对既有实验结论的影响审计

| 实验族 | 典型机制 | B-01是否适用 | 结论 |
|---|---|---|---|
| SAFE/WFix | warmup组同步单source完整动作；之后student | 否 | 不推翻已报告加速/AUC结果 |
| admission core/handoff | `bootstrap_only`，warmup单source | 否 | exact behavior fallback和单source replay lifecycle证据保留 |
| adaptive admission v1 | warmup组同步单source，partial revoke，gate后关闭 | mixed条件不成立 | 其“迁移性判断失败”结论不变 |
| P0 lease oracle | `bootstrap_only`，source bank行为组同步 | 否 | ENGINEERING_INVALID+conditional F-a结论不变 |
| Phase-1 bounded bank lease | 0–30k全源准入但每transition单source；30k整库hard exit并authority off | 否 | hard-exit工程有效、性能HETEROGENEOUS结论不变 |
| future full MCG + partial admission | postwarmup每group可不同source | 是 | 修复前不能声称exact replay revoke |

这一区分很重要：发现真实bug不等于把所有历史实验作废。应按执行条件判断，而不是重新跑整个项目。

## 17. 对GLM改动的最终核查

### 17.1 实际改变行为的部分

只有`compatibility.py`的zero-mask guard改变数值。它对现有所有非zero-maskbank无影响；当前official bank实际mask均非空。局部实现正确，但selected-one-hot仍可绕过，因此不是完整解决。

### 17.2 只改注释的部分

- replay n-step限制说明：总体方向合理，但“n-step必然跨多个options”是当前实现风险，不是理论上无法设计；保持fail-fast合理。
- beta clamp说明：声称可防止sigmoid饱和过强；实际只保证输出gate floor。
- termination margin说明：应区分论文固定xi和发布代码adaptive branch，但不能把adaptive branch自动称为HB正确尺度。
- `update_option()`说明：关于PTF原始Q_o是on-policy、selected source控制行为的叙述错误，必须纠正。

这些解释错误不会改变已跑实验数值，所以无需为注释反复重跑；只需要在论文和正式文档中改正。

## 18. 最小修复顺序

遵守项目“科学主线优先、不为边缘细节无限审查”的执行准则，建议只做以下分级：

### P0：在任何full MCG partial-revoke实验前必须完成

1. 修`_admission_slot_weights()` uniform项的allowed mask；
2. 增加同canonical stratum中“mixed forbidden + pure allowed + student”回归测试；
3. 运行现有全仓测试；
4. 不需要为SAFE/WFix/P0/Phase-1重跑训练。

### P1：下次触碰相应功能时顺手完成

1. `actor_sampling`按真实replay mode写checkpoint；
2. source actor严格完整parameter加载；
3. MCG+asymmetric obs显式拒绝或正确接critic obs；
4. MCG文档由“target critic”改为“online target-task critic”；
5. 延迟schedule隔离source/option构造的global RNG。

### P2：仅在重新研究经典PTF时处理

1. terminal beta mask；
2. selected compatibility强制更新的estimand/消融；
3. null option在source behavior replay中的定义；
4. beta clamp、warmup与target estimator；
5. 普通checkpoint若需严格resume则扩展完整状态，否则保持weights-only边界。

不要为了P2阻塞当前科研主线。当前论文性能证据主要来自reward-bearing bootstrap，不来自经典Q_o/beta。

## 19. 两套实现的合并总评

| 维度 | 历史分模块版 | 官方FastTD3集成版 |
|---|---|---|
| Git可重构性 | 历史`models/`缺失，存在断点 | 当前主体可重构，实验资产更完整 |
| FastTD3核心 | 方程大体对齐但自实现，缺AMP/LR schedule等 | 直接复用官方类和训练配方，可信度高 |
| 原始PTF语义 | student执行、option蒸馏、Q_o/beta存在 | 保留同一路径，同时加入可选source执行/MCG |
| reward-bearing bootstrap | 没有 | 有，是当前主要性能通道 |
| source provenance | 仅option id | 完整body-group provenance |
| exact abstention | 无自动/严格执行机制 | 静态/调度层可以100% student |
| replay lifecycle | 无 | quota、physical handoff、hard exit均可表达；mixed revoke有待修复 |
| paper-grade fork | 无 | anchor bundle与P0工具完整 |
| 正式维护建议 | 只做历史考古/toy | 唯一正式主路径 |

最终应把当前研究框架表述为：

> **以官方FastTD3为目标learner，在HumanoidBench跨任务source bank上研究reward-bearing behavior/data transfer；经典PTF的option/termination是历史起点和可选基线，RBO/admission/provenance/replay lifecycle是后续扩展。**

这比把所有模块统称为“复现PTF”更准确，也更能解释为什么主要收益来自bootstrap而不是Q_o/beta或MCG。

## 20. 当前支持与不支持的论文主张

### 20.1 可以支持

- 目标learner保持官方FastTD3的主要算法和优化配方；
- frozen source经obs/action adapter可在目标HB环境产生target-reward transition；
- source/student warmup bootstrap可把这些transition写入同一off-policy replay；
- admission模式支持student成为一等候选和exact no-source fallback；
- provenance与bank-level hard exit能可验证地切断未来source behavior/replay暴露；
- Phase-1证明hard exit工程语义成立，但其性能影响seed-dependent、非稳定改善。

### 20.2 不能支持

- 不能称当前系统已拥有可靠迁移性指标；
- 不能称exact abstention能自动判断何时弃权；
- 不能称经典Q_o/beta已经在HB上可靠选teacher；
- 不能称MCG使用target critic或已独立证明提升；
- 不能称所有partial source撤销都已严格清除mixed replay，修复B-01前不成立；
- 不能称普通checkpoint是严格resume；
- 不能把PTF描述为原本on-policy或source直接控制环境。

## 21. 审计结束条件

本轮已经完成：

- 两篇论文与两套官方/参考源码对照；
- 历史分模块版逐函数审计；
- 官方集成版入口、FastTD3核心、classic PTF、MCG、admission、replay、anchor、source adapter逐函数审计；
- 当前全仓测试；
- 现有source bank实际加载；
- mixed-source revoke反例复现；
- 既有实验适用性分析；
- GLM改动边界与最小修复优先级。

审计不应继续扩张为无穷的小数精度或底层确定性检查。下一步若用户批准代码修复，只需处理`B-01`及其单测；随后回到论文核心机制和新的科学问题，而不是重跑所有历史实验。
