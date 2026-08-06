# PTF 官方代码基准下的 FastTD3 复现忠实性复审

> 日期：2026-07-23  
> 基准优先级：**作者公开源代码 > 论文说明 > 本项目历史文档**  
> 审计对象：作者 PTF-A3C/PTF-PPO 公开实现，与本项目 classic PTF × FastTD3
> 路径。RBO、MCG、admission、replay lifecycle 不属于本轮复现基准。  
> 本轮目的：回答“原始 PTF 与 FastTD3 结合在 HumanoidBench 上是否有效”之前，
> 先保证被测试的确实是作者代码的关键机制，而不是本项目多个后续改写的混合体。

## 1. 立项门

1. 核心问题：教师价值信号——`Q_omega` 是否能选择合适 source，`beta` 是否能在
   正确状态终止并切换 source。
2. 唯一主要假设：现有 classic 实验未充分保持作者公开代码语义；恢复非必要偏差后，
   才能有效裁决“PTF 与 FastTD3/HumanoidBench 本身不兼容”。
3. 决策影响：若忠实模式仍失败，才停止原始 PTF 复现；若恢复选择/终止信号，则再
   比较 fixed teacher 与 learned scheduling。
4. 已有实验只验证了当前适配混合体，尤其包含 null option、软 target、beta 输出
   重参数化等，不等于作者代码本体。
5. 最小方案：先做代码对照与一个独立 `released-code fidelity` 模式；只通过单 seed
   5k signal gate 后才跑 30k，禁止直接扩三种子。

## 2. 官方源代码身份核验

作者组织仓库 `PTF-transfer/Code_PTF` 的 README 指向
`https://github.com/tianpeiyang/PTF_code`。2026-07-23 重新拉取作者仓库：

- 作者仓库 HEAD：`143dbfe41a0af4fa55ff20640e46392a171fe204`；
- 本地 `reference_source_code/PTF_code` 与作者仓库除 `.git` 外递归 `diff -qr`
  无差异；
- `PTF_A3C.py` SHA256：
  `52c27e60469bdb69fadefaaa016c9c0c3b2e0d685956d26c9018b1ef0f0bce93`；
- `PTF_PPO.py` SHA256：
  `7cde74a9e4f83edf48baab39cd40a06b6830a6e14fff751bea1e9540d3b3d0fe`；
- 两个 SHA 在本地副本与作者仓库逐位相同。

因此本轮比较的 `reference_source_code/PTF_code` 可以作为作者公开代码的可信镜像，
不是本项目改写后的伪“官方代码”。

## 3. 两个争议的直接裁决

### 3.1 自适应 xi 确实来自作者代码

作者 A3C 和 PPO 的 option 模块都有相同分支：

```python
if args["xi"] == 0:
    xi = 0.8 * (max_q - top2_q)
else:
    xi = args["xi"]
advantage = q_current_option - max_q + xi
termination_loss = beta_current_option * stop_gradient(advantage)
```

公开配置 `ptf_a3c_conf.yaml` 和 `ptf_ppo_conf.yaml` 均设 `xi: 0`，所以默认实际走
`0.8 * (top1 - top2)`。`git blame` 也显示该分支来自作者 2022-01-02 的提交
`183c36c5`，早于本项目。

准确归因应写成：

- **机制来源**：作者公开 PTF 代码；
- **本项目决定**：FastTD3 适配配置继续设 `xi=0`，因而主动选择沿用该官方分支；
- **不是**：本项目新提出 adaptive xi；
- 论文中的固定 xi 与作者代码不同，只作为 paper-vs-code 差异记录，不覆盖本轮
  官方代码基准。

### 3.2 Q_omega 不是动作相似度网络

作者代码中 `Q_omega` 的 loss 是 target reward 的 TD loss：

```text
y_o = r + gamma * [(1-beta_o) Q'_o + beta_o Q'_{argmax online Q}]
L_Q = mean_o compatible(o|s,a) * (y_o - Q_o(s))^2
```

option 选择时直接 `argmax_o Q_omega(s,o)`；termination 使用
`A(s,o)=Q_omega(s,o)-max_o' Q_omega(s,o')`。所以用户的理解正确：

> `Q_omega` 的定义是 option value / target return estimator，选择价值最大的
> source；不是学习“哪个教师动作最像 student”。

动作 compatibility 的角色只是**训练支持域/更新资格**：

- 当前已选 option 无条件标记为兼容；
- 其他 option 只有其 source policy 也允许 student 实际动作时才共享该 transition；
- 因而同一 student transition 可以更新多个 option。

在本项目确定性高维动作适配中，compatibility 会改变每个 Q 看到的数据分布，因此
可能让 Q 的估计与“动作支持域/当前 occupancy”高度相关；但这是一种可能的估计偏置，
不能把 Q 的数学目标改写成相似度。此前文档中“Q 更接近学习哪个教师像 student”的
表述过强，已在实验记录中更正。

## 4. 作者公开代码的真实核心流程

以下以作者 PTF-A3C 为主；PTF-PPO 的 option/termination 主体几乎逐行相同。

1. 载入多个冻结 source policy；option 数量就是 source 数量，**没有 null/student
   option**。
2. 每个 episode 开始，用 option Q 的 epsilon-greedy 立即选择一个 source。
3. 环境动作始终由 target/student policy 采样；source 不直接控制环境。
4. 记录 student transition `(s,a,r,s')`，并记录每个 source 是否兼容该 student
   action；当前已选 source 强制兼容。
5. Q 从 replay 均匀采样，所有兼容 source 共享 transition，做 intra-option TD。
6. beta 在刚产生的 `s'` 和当前 option 上更新，而不是从 replay 随机抽历史 option。
7. 到达下一状态后，按当前 option 的 beta 采样是否终止；终止后 epsilon-greedy
   选择新 source。
8. 当前 source 的 policy distribution 作为 target actor 的辅助模仿目标。
9. 作者实际代码的 transfer weight 只有时间衰减 `f(t)*c1`；`(1-beta)` 版本被注释，
   没有进入执行图。
10. Q target 网络按固定间隔硬拷贝；next option 使用 online-Q argmax、target-Q
    取值，beta 使用 online termination head。

## 5. 逐机制对照

分类：

- `MATCH`：当前实现与作者代码关键语义一致；
- `NECESSARY`：确定性 FastTD3 或 HumanoidBench 所必需的适配；
- `NONESSENTIAL`：并非结合 FastTD3/HB 所必需，改变了作者机制；
- `EXTENSION`：本项目后续研究机制，不能计入“纯 PTF”；
- `CONFIG_DEPENDENT`：代码可表达官方语义，但正式配置未使用。

| 机制 | 作者公开代码 | 当前 classic 路径 | 分类与结论 |
|---|---|---|---|
| 环境行为策略 | student action | `execute_sources=false` 时 student action | `MATCH` |
| source 冻结 | 冻结 teacher | eval + no-grad + 不进 optimizer | `MATCH` |
| source/target obs/action | 同空间 | 显式 adapter + source normalizer | `NECESSARY` |
| option 集 | 只有 source | 默认 source + null/student | `NONESSENTIAL`，且正式 classic 实验使用了 null |
| episode 初始选择 | 立即 epsilon-greedy | 初始固定 null/首 option，等 beta 或 done 才重选 | `NONESSENTIAL` |
| epsilon 语义 | greedy 概率 0 -> 0.9 | random 概率 0.30 -> 0.05 | 方向类似但不是同一调度；`NONESSENTIAL` |
| call-and-return | beta 触发重选 | beta 触发重选，`min_steps=1` 时无额外锁存 | 核心 `MATCH` |
| min duration | 无 | 可选 `option_min_steps` | `EXTENSION`，classic=1 时不改变 |
| Q/beta 架构 | 一层 ReLU6，共享干；Q=tanh，beta=sigmoid | 默认两层 256 仿射 trunk，但只在第一层后有 ReLU；Q 线性；beta 仿射到 `[.05,.95]` | Q 尺度/容量部分可辩护；网络非线性位置与 beta 重参数化均为 `NONESSENTIAL` |
| beta 初始化 | dense 默认 bias 0，初始约 .5 | beta bias=-2，输出约 .157 | `NONESSENTIAL`，强改初始 termination hazard |
| Q replay | 均匀 replay | FastTD3 replay batch | 核心 `MATCH`；batch/UTD 是 `NECESSARY` |
| compatible options 共享样本 | 是 | 是 | `MATCH`，不是本项目新增 |
| 当前 option 强制兼容 | 是 | `max(compat, selected_one_hot)` | `MATCH` |
| continuous compatibility | source `mu +/- 1 sigma` 全维硬判断 | masked mean-MSE Gaussian soft weight | `NECESSARY`：FastTD3 teacher 无 learned state-dependent sigma；具体带宽仍是适配假设 |
| null compatibility | 无 null | null 对所有 transition 为 1 | `EXTENSION`，会让 null Q 获得全数据 |
| Q target | online beta；online argmax、target 取值；周期硬拷贝 | target beta；target max；每步 Polyak | `NONESSENTIAL`，当前不是作者更新 |
| Q loss reduction | batch mean of compatible squared losses | 除以 compatibility weight 总和 | `NONESSENTIAL`，改变有效 source 权重/学习率 |
| beta batch | 当前 transition/current option，非 terminal | 历史默认 replay；`current_transition` 可恢复 | `CONFIG_DEPENDENT`；最新多教师 gate 已用官方语义 |
| beta 更新次序 | 先 beta、后 replay Q | 先 Q、后 beta | `NONESSENTIAL`，相差一个 option 更新 |
| beta 更新频率 | 每环境步一次 | 每 FastTD3 `num_updates` 重复一次，HB 默认两次 | `NONESSENTIAL`，beta 有效更新剂量翻倍 |
| termination loss | `beta*(A+xi)`，只 detach A | 额外把 advantage clamp 到 `[-margin,+margin]` | `NONESSENTIAL`；多 option 下会截断差 source 的上推信号 |
| adaptive xi | `xi=0 -> .8*(top1-top2)` | 相同 | `MATCH` |
| beta warmup | 无 | 可选，classic=0 | classic 配置下 `MATCH` |
| beta logit clip | 无 | 可选 STE clip | `EXTENSION`；普通多教师配置关闭 |
| transfer teacher | 当前 selected source | replay 中存储 selected source | 教师身份 `MATCH`；off-policy actor batch 为 `NECESSARY` |
| transfer loss | stochastic distribution cross-entropy | deterministic masked Huber | `NECESSARY` |
| transfer 的 beta 权重 | 作者执行代码没有 | classic 配置为 `(1-beta)` | `NONESSENTIAL`，跟了论文而非作者代码 |
| transfer 时间调度 | tanh `f(t)*c1` | 线性 1 -> 0 | `NONESSENTIAL`，改变剂量曲线 |
| target actor 更新 | RL loss + transfer loss | FastTD3 actor loss + transfer loss | `NECESSARY` 且结构一致 |
| source 直接执行 | 不执行 | 可选 `execute_sources` | `EXTENSION`；pure classic 必须关闭 |
| MCG/RBO/admission/provenance | 无 | 项目后续机制 | `EXTENSION`；pure classic 必须全部关闭 |

## 6. 当前 classic 实验究竟测试了什么

### 6.1 单教师 + null

这不是作者官方结构。作者 option 数等于 source 数；单 source 时选择退化为唯一 option，
而 adaptive xi 的 `top_k(...,2)` 甚至要求至少两个 option。我们加入 null 后才能构造
“walk vs no-transfer”，但同时创造了官方代码没有的：

- null Q/beta；
- null 对所有 student transition 恒兼容；
- 初始 null；
- 两 option 下 adaptive-xi 的 `+0.8 gap / -0.2 gap` 聚合动力学。

因此单教师实验适合研究本项目的“student-inclusive option extension”，不适合裁决
原始 PTF 多教师机制是否能在 HB 上工作。

### 6.2 stand/walk/run + null

它比单教师更接近 PTF 的多 source 场景，且已证明 beta 可以恢复状态条件分离；但仍
不是作者官方复现，因为它含 null、beta 输出重参数化、clamped advantage、soft target、
beta-weighted transfer 和线性 schedule。

30k 结果只能说明：

- 当前混合适配版的多 source termination 可训练；
- 当前 option scheduling 远弱于 fixed-walk；
- 不能把失败简单归因于 beta；
- 不能说 Q 的定义是动作相似度。

尚未回答：

- 去掉 null 并恢复作者 update semantics 后，官方 PTF 的 teacher selection 是否有效；
- Q 排序失败来自 compatibility/support bias、即时 return 与学习增益目标错位，还是
  其余适配偏差。

## 7. Q_omega 的正确解释与真正局限

### 7.1 正确解释

`Q_omega(s,o)` 试图估计在 call-and-return 结构中继续/切换 option 后的 target-task
discounted return。policy over options 选最大 Q；beta 看当前 Q 相对 max Q 的优势。

### 7.2 compatibility 为什么会影响 Q，但不等于 Q

compatibility 决定 off-policy transition 是否能安全用于 source `o` 的 TD 更新。
在作者低维随机策略中，它是 source distribution 的支持域判断；在 HB 的 61 维
确定性策略中，我们改成 soft action kernel。

因此当前 Q 更准确地说是在拟合：

> 当前 student occupancy 下、由 selected-option 强制样本与 teacher-compatible
> 样本共同支撑的 option-conditioned target return。

它不是相似度分数，但估计误差可能受相似性定义强烈控制。

### 7.3 它为何仍不等于项目要找的迁移性指标

项目需要的是“现在让 source 介入，是否会让 student **之后学得更快/更好**”。
作者 Q 的监督是当前 target reward TD target，不直接比较“接受该 teacher 更新”和
“student 自学更新”的反事实未来差值。尤其在 FastTD3 中：

- 当前 reward 来自 student action；
- selected source 只通过后续 replay actor update 的辅助 loss 影响 learner；
- teacher 的因果效果延迟进入 student 参数与 occupancy；
- Q 却把当前 transition reward 归给当前 option。

所以即使完全忠实复现，`Q_omega` 也首先是原 PTF 的 option-return 信号，而不是本
项目最终希望提出的 counterfactual transferability metric。这是科学边界，不是把
Q 错叫成相似度的理由。

## 8. 官方忠实模式应恢复什么

建议新增独立、显式持久化到 checkpoint 的 `released_code_fidelity` 模式；不得覆盖
旧配置或改变已有实验复现性。

### 8.1 必须恢复的作者语义

1. source-only option bank；至少两个 source；禁用 null；
2. episode/reset 时立即 epsilon-greedy 选择；
3. random/explore 概率从 1.0 降至 0.1，对应官方 greedy 概率 0 -> 0.9；
4. 恢复一层 ReLU6 shared trunk、tanh Q head、bare sigmoid beta 和 zero beta
   bias；不做 `[.05,.95]` rescale；
5. 不做 advantage clamp，不做 beta logit clip，不做 beta warmup；
6. beta 使用 current transition/current option，每个 outer step 只更新一次；
7. beta 先更新、Q 后更新；
8. Q target 使用 online beta、online argmax + target value；
9. option target 每 1000 option updates 硬拷贝；
10. transfer 不乘 `(1-beta)`，因为作者执行代码没有该项；
11. 时间权重恢复 tanh 形状；仅把 episode 时间轴映射成 HB 训练进度；
12. 保留 selected option 强制兼容和多 compatible option 共享 transition。

### 8.2 必须保留的 FastTD3/HB 适配

1. FastTD3 actor/critic/replay/normalization 主干；
2. target-to-source obs adapter、action adapter、冻结 source normalizer；
3. deterministic teacher/student 的 masked Huber 蒸馏；
4. 高维 deterministic action 的 soft compatibility，但必须明确它是支持域适配，
   不是 transferability；
5. vectorized current-transition batch 与 time-limit bootstrap；
6. checkpoint/source-free evaluator。

### 8.3 需要作为显式消融而非悄悄混入的项目改写

- null/student option；
- `(1-beta)` transfer gate；
- beta rescale/logit clip；
- advantage clamp；
- soft option target；
- replay-beta；
- linear schedule；
- execute-source、MCG、RBO、admission。

## 9. 最小后续验证

不应再用单教师+null 裁决原始 PTF。正确的最低成本路径是：

1. 实现独立 fidelity mode，并做公式级单元测试：
   source-only、初始重选、official epsilon 语义、online-beta double-Q target、
   current-transition beta、无 clamp、hard target copy、无 beta transfer gate。
2. 一次 200-step HumanoidBench wiring smoke，只验证数据流。
3. `stand/walk/run -> hurdle`，seed 1，5k signal gate：
   - Q 不能恒等/数值崩溃；
   - beta(non-argmax) 应高于 beta(argmax)；
   - option 必须发生真实切换；
   - return 不参与 signal gate。
4. 只有 signal gate 通过，才续跑同 seed 到 30k，与 scratch 和 fixed-walk 的既有
   同 seed 曲线比较。
5. 若 30k 不超过 scratch，结论是“作者代码忠实适配下，PTF scheduling 未把机制
   信号转化为迁移收益”，停止这条复现线；不得再调 xi/beta 阈值。
6. 若超过 scratch，再申请 3 seeds，并单独比较当前-return Q 与未来-learning-value
   指标；不能直接把 Q 宣称为已解决迁移性。

## 10. 本轮已做的非功能性纠正

1. `option_update.py`：把“paper dynamic margin”改为“released-code dynamic
   margin”，避免混淆论文与作者代码。
2. `compatibility.py`：把错误的“替代 PTF on-policy density”改为准确的“替代作者
   代码 `mu +/- 1 sigma` 二值支持域测试”。
3. `train_ptf.py`：明确 compatible-option sharing 与 selected-option 强制兼容本来
   就是作者代码；FastTD3 改的是二值支持测试到 soft deterministic compatibility。
4. `classic_ptf_signal_diagnostic_20260722.md`：撤回“Q 学动作相似度”的过强表述，
   改为“Q 是 return estimator，但其数据支持和 credit assignment 可能偏离未来
   learning utility”。

本轮没有改训练公式、没有启动新实验。功能性 fidelity mode 应在本审计确认后作为
下一步单独实现，避免在脏工作树中悄悄改变旧实验语义。
