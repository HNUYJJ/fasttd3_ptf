# 与 GLM-5.2 讨论纪要:PTF×FastTD3 复现核验与课题方向(2026-07-21)

> 本文档由 GLM-5.2(我,ZCode 助手)整理,记录 2026-07-21 与 PI 的一次长讨论。PI 先要求
> 检查"自己分模块实现的 PTF 结合 FastTD3"复现是否正确(对照 PTF 论文 TF 原版
> 与 FastTD3 论文),讨论延伸到课题真实脉络、当前断点、以及未来可走的迁移
> RL 方向。本文档供 ChatGPT 参考,定位为"第二位顾问的独立判断",不要求认同,
> 请按严格审稿人视角反驳。

讨论覆盖的文件与论文:
- 论文:`FastTD3.pdf`(Seo et al. 2025)、`PTF-arxiv.pdf`(Yang et al. 2020,AAAI)
- 官方 backbone:`fasttd3_ptf/official_code/FastTD3/fast_td3/fast_td3.py`
- 自己的 PTF 实现:`fasttd3_ptf/ptf/{option_module,option_selector,option_update,distillation,compatibility,source_policy,legacy_actors,mcg}.py`
- 缝合主循环:`fasttd3_ptf/official_fasttd3_ptf/train_ptf.py`
- replay 适配:`fasttd3_ptf/official_fasttd3_ptf/ptf_replay.py`

---

## 0. 讨论起点

PI 希望确认:在不直接用官方 FastTD3 源码的那部分(即 PTF 框架本身),自己分模块
实现的版本是否正确。原 PTF 是 TensorFlow + A3C/PPO(on-policy,actor 输出动作概率
分布),我们改写成 PyTorch 并与 FastTD3(off-policy TD3,actor 输出确定性动作)缝合。
讨论因此横跨两层:(1) PTF 论文公式 vs 实现的对照核验;(2) A3C/PPO→TD3 适配是否
语义正确。本文档先给课题脉络(否则核验结论失去上下文),再给核验,再给方向建议。

---

## 1. 课题真实脉络的重建(PI 澄清后我的理解)

### 1.1 课题创立之初:PTF on HumanoidBench

课题最初设想来自 PTF(Yang et al. 2020):训练多个源任务策略 → 冻结 → 把每个源策略
视为一个 Option → Option 网络 μ(o|s) 根据 student 当前状态选教师 → 被选教师连续指导
student 一段时间 → Termination 网络 β(s) 决定教师何时失效 → 终止并重选 → student
逐渐学会目标任务。

最初真正想研究的是:**PTF 能否在 HumanoidBench 中根据 student 状态,自动决定"当前
用哪个教师、用多久、什么时候切换",从而加速目标策略学习?** 当时没有单独设计显式
迁移性指标——option value Q_o(s,o) 和 termination β 本身承担了隐式的"教师价值判断"。

### 1.2 PTF 没成功与无法区分的两种可能

搬到 HB 后效果未达预期,当时存在两个无法区分的可能:
1. 工程实现问题(obs/action 适配、option/termination 更新、source 执行衔接、selector
   不稳定、并行环境锁存/终止逻辑有误等);
2. 方法本身不适合 HB(source-target 差异大、高维连续动作下 option credit assignment
   困难、target reward 短期内无法正确评价教师、source 指导效果延迟很长、termination
   难从稀疏高噪声 reward 学、selector 塌缩或频繁错误切换)。

没有得到可信结论:"到底是 PTF 没实现正确,还是 PTF 即使实现正确也不适合 HB"。
这是后续所有转向的起点。

### 1.3 发现 reward-bearing bootstrap 是主通道

后续实验发现:即使没有成功学出可靠的 option selector 和 termination,只要让冻结 source
进入目标环境执行,并把产生的目标任务经验写入 replay,就能明显加速 student 前期学习。
注入的不是 source 原任务的旧数据,而是 `(s_t, a_t^source, r_T, s_{t+1})`——source 在
**目标任务环境**执行后由**目标任务 reward 函数**重新产生的 transition。

关键实证发现(写入 RESEARCH_ROADMAP 与 advisor_feedback_analysis):
- 收益集中在训练前期;
- bootstrap 是主要性能通道(`bootstrap_only ≈ full` 消融,gate 蒸馏只占 ~10%);
- 部分任务不仅前期加速,后期上限也提高(hurdle 有迹象);
- source 即使不会完成完整目标任务,其站立/平衡/行走能力也能改变 student 早期状态分布;
- 不合适 source 会通过行为和 replay 产生负迁移(crawl/basketball)。

研究路线因此从"学习一个 PTF option selector 和 termination"转变为"如何选择 source、
如何注入 source 经验、如何控制 source 经验在 off-policy replay 中的生命周期"。

### 1.4 SAFE/WFix 阶段性方案

因为 PTF 动态 option/termination 没成功,先用训练前静态 probe 替代它:Source Bank →
在目标环境短 Probe → Effect Map/T⁰ → 静态 source 权重 → warmup 期按权重选 source →
锁存 25 或 50 步 → source 轨迹注入 replay → FastTD3 更新 student。

SAFE/WFix 解决了"哪些 source 获得更多执行机会、每次执行多长、source 经验如何进入
bootstrap replay",但**没有真正恢复 PTF 核心能力**:不根据 student 阶段动态选教师、
没学出可靠 termination、T⁰ 只是静态 allocation prior、不能在 crawl/basketball 等任务
自动拒绝有害 source。

### 1.5 老师三点意见 = 原始 PTF 的三个缺口(核心 reframing)

这是 PI 澄清后我得到的最重要的对应关系,把散乱的机制重新收束成统一问题:

- **意见 1(replay 采样)**:原始 PTF 只关心"当前由哪个 option 控制行为"(行为控制
  平面 B_i)。但在 FastTD3 这种 off-policy 算法中,即使 source 已停止控制环境,它的
  旧数据仍可能继续影响 student。因此必须增加第二条控制通道:"当前哪些 source 数据
  仍有资格训练 actor 和 critic"(数据控制平面 R_i)。这产生 source provenance、
  admission-consistent replay、recency 控制、source 撤销后退出 active replay、
  authority/replay 双通道生命周期。**原始 PTF 缺数据平面。**

- **意见 2(student 作为候选策略)**:原始框架主要在 source options 间选择,但跨
  任务负迁移场景中 student 必须是一等候选。选择空间应是 `{π_student, π_1^S, …, π_M^S}`;
  若无 source 优于 student 则 `p(π_student)=1`。这就是 student-inclusive admission、
  exact abstention、删除固定 0.5 teacher floor。**修复的是原 PTF 控制平面的安全缺口:
  selector 不仅要决定"选哪个教师",还必须能决定"一个教师都不用"。**

- **意见 3(可迁移性指标)**:原始 PTF 通过 option value 和 termination **隐式**学习
  教师价值,但在 HB 上这一信号没有可靠工作。老师希望显式回答"某个 source 在 student
  当前阶段到底是否值得迁移"。这个指标应同时支持 source 选择、student 与 source 比较、
  source termination、replay 数据准入或衰减。**修复的是原 PTF 隐式信号不可靠。**

而 reward-bearing bootstrap、WFix、admission、MCG、provenance replay 全部成为"修复
原始 PTF 在 HB 上三个缺口"的子任务,不再是散乱的新机制。统一框架即"双通道 PTF":
Behavior-option selection(B_i)+ Replay-data lifecycle control(R_i)。

### 1.6 当前活跃机制:admission control

当前活跃主线是 admission control(我之前误判它"简单不够 novel",后修正):
- 用不可变 `AdmissionSnapshot` 显式决定哪些 source 被"准入";
- admitted 源 + student 组成单一 categorical(`candidate_probabilities`),被拒源概率
  严格 0;全拒绝(exact_abstain)时行为确定性回退纯 student,不消耗抽样 RNG(为让
  "无迁移"基线可证明等价于纯 scratch);
- 训练循环只消费快照、不做 utility 推断(可复现性关键设计);
- 两种决策来源:静态 schedule(yaml 声明,如 step 0 准入全源、step 30k 全驱逐=bounded
  bank lease)、adaptive controller(stage-window UCB/LCB 保守撤销,但此版 7-15 预注册
  FAIL,实现保留供审计)。

### 1.7 围绕 admission 建的严格验证基础设施

这部分占了 7-16 后绝大部分工作:预注册 + 双 Gate(A 等价性 / B 容差)+ Hoeffding 统计
ε(`ε=sqrt(ln(2M/α)/(2N))`)+ SESOI 外部锚定(`delta_task=0.5×cross-seed SD of historical
scratch 35k-80k nAUC`,basketball δ_SESOI=45.79 / truck 36.52)+ SHA256 provenance 链
(每 checkpoint 逐项 env/seed/step/bank/mode/warmup/handoff/masses 哈希)+ anchor-resume
可复现训练 + 对抗审查(23 轮复核,run card v0.1→v0.6)。

### 1.8 Phase-1 bounded lease 阴性 + 核心断点未解决

Phase-1 实测(`final_result.json`):
- basketball `hard_exit_minus_scratch`:mean −127.4,CI90 [−207.0, −47.7],classification
  **HARM**(越过 SESOI 45.79);
- basketball `retention_minus_scratch`:mean −122.3,CI90 [−255.6, +11.0],classification
  **UNCERTAIN**(跨 0);
- persistence_delta(retention−hard_exit,35k-80k 窗口):basketball −39.2、truck −24.4,
  **全负**(保留源反而比强制驱逐源更差)。

**核心断点 = 课题最初那个问题**:如何可靠判断选哪个教师、何时终止、何时让学生自学。
组件②③④⑤(执行与生命周期机制)全部建成并验证,组件①(stage-conditioned 迁移性
指标)三次尝试(SIV 2×2 机制门 / SHU gate / adaptive revocation)**全部 FAIL,且都是
行为 reward 信号族**。共同病灶:引导型好源执行段做"脏活"(爬坡跨障即时 reward 低),
与劣源在行为 reward 信号下不可区分;判据只能识别"明显无关源"。

纪律规定:不再做行为 reward 信号第四种变体。重攻组件①只能换**非行为信号族**:
学生侧 learning progress(更新前后性能/损失变化)、replay/update 通道 TD 统计(source
数据对 learner 更新的直接价值)、半交互 T^critic 公式。

---

## 2. 几个关键技术讨论的详细结论

### 2.1 safe_bootstrap warmup 机制详解

设计:不是人手调,由离线探针 `probe_transfer_map_v2.py` 实测两个 per-source 参数:
- weight(抽谁)= per-task 各源的 vs-zero reward-bearing score,`max_h[reward_gain_vs_zero
  − fall]` clamp≥0;
- horizon(执行多久)= safe_horizon,`max{h : fall_prob(h)<0.5}`。

`build_safe_bootstrap_banks.py` 把两者写进 bank yaml 的 `bootstrap:{weight,horizon}`。
训练循环 warmup 期每个 env 锁存到期时走 `McgBehaviorController.step` 的 safe_bootstrap
分支:以 warmup_exec_prob 决定抽不抽教师;按 bootstrap_weights 做
`softmax(weight/τ)` 多项分布采样抽哪个教师;锁存步数=抽到教师的 env 用该源
safe_horizon,纯学生 env 用默认 warmup_min_steps。整动作执行(非步级混合),锁存期
保持当前选择(temporal consistency,PTF β/termination 的模块化对应物),done 的 env
立即回学生并清零。gate 阶段(step≥15k)切换到 critic Δ gating。

作用:解决 random warmup 在脆弱 OOD 任务上注入片段质量 seed 敏感、高方差的问题
(window +76/−27)。

如何起作用:① 把"哪个源有用"从"事后训练出的 Q_o"换成"事前实测出来的"——绕开
Q_o 鸡生蛋;② 用 vs-zero(非绝对回报)作收益基准,避开 HumanoidBench dense reward
shaping 偏置;③ safe_horizon 把"注入成功片段"和"注入失败片段"切开——只执行教师
能稳定站立的安全前缀,在它开始摔之前交还给学生;④ 为后续 critic gating 铺好路——
注入的片段质量决定 critic 15k 步时的可靠度,进而决定 gate 成败。

为什么能起作用:本质是把"无差别注入固定长度的随机教师片段"升级成"按收益加权、按
安全边界截断的定向注入",优化"warmup 注入的质量";底层产生收益的动作仍是同一个:
**教师数据注入 replay 让 critic 预热**。

### 2.2 哪些机制真正起作用

证据链:
1. 消融(commit `00825d0`/`4c6599e`):`bootstrap_only` 独自拿下 full 方案 65–112%、
   均值 ~90% 收益。gate 蒸馏只贡献约 10%,且 commit 原话"gate is safety not
   clean-task performance"。
2. 多任务聚合(`aggregate_multitask.py:43` 原文):"safe/rand 均 bootstrap_only 口径,
   只差 warmup 方式"——当前正式对比只围绕 warmup bootstrap 两个变体,gate 蒸馏不进
   主对比。
3. 最新 commit(b183f40, wfix 消融)仍只围绕 warmup bootstrap 的源选择/执行时长两变量。

判断:
- **真正在用、且产生收益的机制 = warmup 期把教师整动作注入 replay buffer**,具体到
  当前主线是 safe_bootstrap(按收益抽源 + 按摔倒时间截断);产生收益的**本质动作**
  就是"教师演示注入 replay"。
- **FastTD3** 是底层 RL 引擎,不可选,所有 target 训练都跑它。
- **MCG critic gating**:代码活跃、默认开,但**被消融否定**(只占 10%),实际角色退化
  为"防负迁移的安全阀",从"增益来源"降级成"保险丝"。
- **经典 PTF(option Q_o/β/λ 蒸馏)**:MCG 模式下整段被跳过
  (`train_ptf.py:1536` `if not mcg_enabled` 跳过 `update_option`),是 dead path,
  MCG 用 critic Δ 替代 Q_o。
- **ED-SF / z-native**:ED-SF 只在 push 单任务跑过未进主表;z-native 只在 push 试过,
  且与 MCG 显式互斥(`train_ptf.py:781`)。两者均搁置。

### 2.3 critic 与 actor 学习的问题(PI 问到的核心)

FastTD3 是 actor-critic:
- **critic 学 Q(s,a)**:信号来自环境真实 reward(TD/Bellman 更新),只依赖 buffer 数据
  质量,不在乎动作谁产生。类比"考官根据学生答卷(reward)学会打分"。
- **actor 学 π(s)→a**:通过最大化 critic 给的 Q(DDPG/TDG:`rl_actor_loss=-Q(s,π(s))`),
  信号来自 critic。类比"考生根据考官打分调整答题"。actor 好坏**完全取决于 critic 好坏**。

鸡生蛋:actor 要学好→需 critic 准;critic 要学准→需 buffer 有足够多/多样/高质量真实数据;
但早期学生 actor 随机→产出的数据质量差(humanoid 开局即摔)→ critic 学不准→ actor
学不好→恶性循环。warmup bootstrap 灌入教师好数据打破此循环:critic 在含教师轨迹的
buffer 上 Bellman 更新→Q 学得准;actor 的 `-Q(s,π(s))` 梯度因此指向正确方向→actor 也
靠 RL 梯度持续变好(warmup 期就在学,不是"warmup 后才开始学")。蒸馏(`transfer_loss`)
是和 RL 学习正交的另一条通路,直接规定"actor 动作向教师动作靠",不经 critic;warmup
期被关(`compute_mcg_transfer_loss` 里 `step<warmup` return 0),因为此时 critic 还不可信、
modular 蒸馏的"向谁学"要 critic Δ 决定、噪声 Δ 会乱模仿。gate 期才开。

### 2.4 PTF 蒸馏 vs bootstrap 注入:为何后者更有效

五条原因(每条对应代码注释里的实测教训):
1. **蒸馏是原地监督,无法改变状态分布——而 humanoid 瓶颈在状态覆盖**。蒸馏在 buffer
   已采样状态上算 `‖π_s−π_t‖²`,只在"学生已能到达的状态"纠正动作。humanoid 失败模式
   是学生到不了"站立靠近目标"的状态(开局即摔,buffer 全是摔倒状态),在这些状态上
   蒸馏噪声动作。注入式教师亲自走过去,沿途"走向目标物"的 transition 全进 buffer。
2. **蒸馏只更新 actor,critic 不受益;但 actor 又依赖 critic——鸡生蛋**。critic 只在
   学生(摔倒)数据上训→学不准→actor 的 RL 梯度错→两个信号打架。注入让 critic 在好数据
   上训→变准→actor 的 RL 梯度自动正确。
3. **教师的"动作"在目标任务是 OOD 的,但教师"走过的轨迹"不是**。教师动作 OOD,
   蒸馏模仿越像越糟(负迁移);但教师执行后环境给的 reward 是真实的,transition 对 critic
   是高质量真实标注。蒸馏迁移"教师动作输出"(噪声);注入迁移"教师与环境真实交互的
   结果"(真信号)。
4. **off-policy 注入是免费午餐,蒸馏是额外税**。注入 transition 进 buffer 后被 critic
   和 actor 当正常 off-policy 数据复用,无额外损失项,不干扰训练动力学,教师退场后数据
   继续在 buffer 多轮采样供能。蒸馏给 actor 额外加 `λ(t)·‖…‖²` 梯度干扰,λ 大压过 RL、
   λ 小没用,衰减即失效无持续复用。
5. **humanoid 教师是半成品专家,只配当数据源不配当模仿目标**。stand/walk/run 只会站走
   不会做目标任务。当模仿目标→学生学"只会走"→任务完不成;当数据源→教师走到目标物旁
   维持站立,这段 transition 对"学会在目标物附近稳定"有用,至于任务怎么做交给 RL+task
   reward。`bootstrap_only` 拿 90%、`full` 只多 10% 正因此。

### 2.5 为什么 PTF 分阶段选教师行不通(五环全断)

PTF 的 option selector μ(o|s)+β 想做的正是"分阶段选合适教师",理论上完美(前期选 stand、
站稳后转 walk、到目标物旁转 reach)。实际在 humanoid 上五环全断:
1. **Q_o 鸡生蛋**:Q_o 用 TD target 学,要成功 transition 才学得出;humanoid 开局即摔没
   成功样本→Q_o 学不出→选不出好教师→还是摔。
2. **β 在大量级 Q 下饱和**:PTF 终止损失 `β·(A+ξ)`,A=Q_o−maxQ。toy domain reward
   ∈[0,5] 平衡;HB Q~100(v_min/v_max=±1000),β 1000 步内饱和到上限→"换教师"开关焊死
   在"一直换",教师来不及教。
3. **教师是部位专家非阶段专家,整教师切换粒度太粗**:stand 教师任何状态都输出站立动作,
   包括该推箱子时。Q(s,o) 是全局标量,无法表达"该教师只对身体某部位有用"。Q 选 stand
   覆盖掉学生该做的推箱子动作。
4. **蒸馏式迁移不改变状态分布**:即使调度完美,蒸馏治不了 humanoid 真病因(状态覆盖)。
5. **off-policy 调度的时序错配**:call-and-return 在线按 step 切换,但 off-policy replay
   同 batch 混着不同时刻不同教师采集的 transition,option_id 标注的是采集时教师但当前
   策略已变。compatibility 是对此的修补,但是 patch over patch。

任一环断整条链废;humanoid 上五环全断。这不是 PTF 实现不好,是 PTF 数学为 toy domain
(小量级 reward、教师完整专家、状态覆盖不难)设计,搬到 humanoid 全身任务结构性失配。
MCG 只修了①(Q_o→critic Δ)和③(option 推广为(教师,身体组)),没修④(蒸馏不改变状态
分布),所以 gate 蒸馏只贡献 10%;真治④的是 safe_bootstrap 注入。

### 2.6 PTF 是否真在 HB 不起作用

我的判断:**不是"不起作用",是"PTF 的执行-注入半身 work,调度-终止半身的信号机制
学不出来"**。证据:bootstrap(PTF 的 option 执行后注入 replay 子组件)明显 work(扛 90%);
option selector + β(智能调度半身)不可靠。失败根因:Q_o 鸡生蛋、β 大量级饱和、教师半
成品、critic 不可信时 Δ 噪声。所以"PTF 死路"说法不对,是**信号源需要被替换**:用显式、
更可靠的"教师价值信号"替代 PTF 的隐式 option-value,PTF 的 option/termination 可复活。
这正是老师意见 3 的指向,也是当前工作的核心断点。但显式信号也三连失败(SIV/SHU/adaptive
revocation),全是行为信号族;纪律指明换非行为信号族,本质是"用更可靠信号重建 PTF
原始的 option-value 等价物"。

---

## 3. 未来方向建议(供 ChatGPT 审视)

### 3.1 判断 insight 的三条标准

一个好的迁移研究方向必须同时满足:(1) 对准 HumanoidBench 真实病因(非 toy 伪需求);
(2) 利用 humanoid 独特结构(全身耦合+摔倒终止+身体分组+多任务共享身体);(3) 在现有
基础设施上可落地(FastTD3+源技能库+off-policy replay+critic gating,不需重写训练栈)。

### 3.2 方向 A:攻 SOTA-难任务(最高 impact)

HumanoidBench 论文自己承认 flat RL 在多数任务 struggle,hierarchical baseline 只解决
部分。已有迹象:hurdle 100k 上限提升迹象;truck 加 hurdle 源后 +229.9(技能互补);
stair safe(h50)超 scr(279 vs 252),静态短 horizon 全负但 horizon 对了就赢。
**建议**:停止在 cabinet/crawl 这种已判负的边界任务上纠缠,集中火力攻 1-2 个 SOTA
做不好的复杂任务(stair/hurdle/truck),用源注入做出 SOTA 都达不到的成绩。"攻下别人
攻不下的任务"比"在已解决任务上加速"强得多,是顶会级 contribution 最短路径。

### 3.3 方向 ①:可达状态迁移 Reachability Transfer(我最看好)

核心 reframing:源策略的真正价值不是"会什么动作",是"能到达哪些状态"。stand 源价值
不是会站,是"把学生带到站立状态,在那里学生才能学到操作"。能统一解释所有正负结果:
有用源(stand→hurdle、walk→truck)= 把学生带到目标 bottleneck 附近状态;有害源
(stand→crawl)= 把学生卡在无关状态(站住不动没匍匐);stair horizon 敏感 = h50 才能把
学生带到"登上第一级台阶"的状态,h25 到不了。文献里这个 framing 几乎没有,而已有全部
实验数据恰好是它的实证。能成为论文理论支柱:把"迁移什么"从策略/表示/reward 扩展到
"可达状态包络"——第四种迁移对象。**且 PTF 复活点可能就在这:用"可达状态"作为可靠
教师价值信号**(替代失败的隐式 option-value 和行为 reward T)。一个源"能把学生带到哪些
有用状态"是可测的、非行为的、不依赖即时 reward 的。

### 3.4 方向 ②:课程迁移(最实用)

不只"简单源→复杂目标",而是"按难度阶梯学一串任务,每个学好的当下站的源"。Transfer
Map 已能算 (源,目标) snippet-level 可迁移性,把它从"选源工具"升级为"自动排课程":
若任务 A 策略能零样本迁移到 B 并拿正分,A 是 B 前置课程。HB 有天然难度梯度
(stand<walk<hurdle<stair+crawl+manip),是做课程最好场景。且课程学习"任务越多自动课程
越值"是单任务迁移讲不出的 scaling 故事。

### 3.5 方向 ③:奖励函数迁移(与教师注入正交)

用源策略 value function V_source(s) 做 potential,给学生 reward 加 `γV_source(s')−V_source(s)`
(Ng et al. 1999 理论保底最优策略不变,但加速探索)。HB 稀疏 reward 是 RL 失败根因之一,
源 value 是现成密集信号;与教师注入正交(一个改数据、一个改 reward),可同时用,消融
能干净分离。风险:源 value 在 OOD 状态不可信,可用可达状态 gating 配合。

### 3.6 方向 ④:critic 表示迁移(直接攻瓶颈)

不迁策略,迁 critic 表示。HB 各任务共享具身,价值函数拓扑相似(站立状态都值高,摔倒
都值低)。源任务 critic 学到的"什么状态好"的表示可热启动目标 critic——直接攻已诊断的
真病因(critic 预热),而非像 bootstrap 绕开。风险:易被审稿人质疑"和预训练何区别"——
要强调 critic-only + 价值拓扑共享这个特定动机。搁置的 entity encoder(z-native)其实在这条
路上,但当时迁了 actor 表示失败;这次只迁 critic 表示可能成。

### 3.7 优先级建议

最 impact 组合:**方向 A(攻 SOTA-难任务)+ 方向①(可达状态迁移 framing)**。A 给最硬实证
结果(攻下别人攻不下的任务),① 给理论 framing 统一解释所有正负结果;两者天然耦合:
可达状态迁移的 framing 正好解释"为什么源能攻下复杂任务"——因为源把学生带到目标
bottleneck 附近状态,而 flat RL 自己到不了。PTF 复活点也在①:用可达状态作为可靠教师
价值信号。

---

## 4. PTF×FastTD3 复现核验

### 4.1 核验方法

对照三处:PTF 论文 §4.3-4.5 的核心公式(Eq.3 U-value、Eq.5 termination、Alg.2 option-Q
更新、§4.4 compatibility、§4.5 蒸馏)、FastTD3 论文 §2(distributional C51 critic + CDQ
+ 大 batch + 降序架构 + mixed noise)、官方 FastTD3 源码 `fast_td3.py`。逐项比对
`fasttd3_ptf/ptf/` 与 `train_ptf.py` 的实现。

### 4.2 FastTD3 backbone 核验:✅ 完全正确

`fasttd3_ptf/official_code/FastTD3/` 是上游官方源码只读拷贝,通过 `paths.py` 注入
`sys.path`,**对官方代码零修改**。逐项:Algorithm=TD3 变体;distributional critic=C51
(`DistributionalQNetwork.projection` 实现完整 C51 投影);Critic 架构
`hidden_dim→//2→//4`(hidden_dim=1024 即 1024/512/256);Actor 同结构 hidden_dim=512;
CDQ=`torch.minimum(qf1,qf2)`;mixed noise=`Actor.noise_scales` buffer + done 时重采样;
target policy smoothing=`clipped_noise=randn×policy_noise.clamp(-noise_clip,noise_clip)`;
critic loss=`-Σ(target_dist·log_softmax(qf))`;replay=N×num_envs 存 GPU。全对。
**方法上最稳妥——直接用官方实现不重写。**

### 4.3 PTF 机制核验:✅ 基本正确,有几处有据工程化修改

精确对应论文的:
- **Option-value Q_o(s,o)**:论文 §4.3 ↔ `OptionModule.q_head` 输出 num_options 维 Q。
- **Termination β(s,o)**:论文 §4.3 sigmoid ↔ `OptionModule.beta_head`+sigmoid,[beta_min,
  beta_max] 区间(clamp 是合理工程修改,见 4.5)。
- **U-value**:论文 Eq.3 `U(s',o)=(1−β)Q_o(s',o)+β·max_o' Q_o(s',o')` ↔ `option_u_value`
  `(1−beta_next)*q_next + beta_next*max_q`。**精确对应**。
- **Option-Q TD target**:论文 Alg.2 `y=r+γ·U(s',o)` ↔ `y_all=rewards+γ·bootstrap·u_next`。
- **Option-Q loss**:论文 Alg.2 Line 7 `L=(1/N)Σ(y−Q_o)²` ↔ `weighted_option_q_loss`
  `(weights·(q−targets)²).sum()/weights.sum()`(加权版,见 4.4)。
- **Termination loss**:论文 Eq.5 `∂L/∂θ_β=α_β·(A+ξ)`,A=Q_o−maxQ ↔ `termination_loss`
  `β_o·(q_o−max_q+margin)`(有 advantage clamp,见 4.5)。
- **Call-and-return 调度**:论文 §4.3 选 o→执行 π_o→β 终止→重选 ↔ `OptionSelector.step`
  `terminate=rand<β` 后 `greedy=argmax Q`。
- **ε-greedy**:论文 Alg.2 Line 6 ↔ `OptionSelector.step` `explore=rand<ε→random else greedy`。
- **Transfer loss 蒸馏**:论文 §4.5 跨熵 H(π_o,π_θ)加权 f(β,t) ↔ `compute_transfer_loss`
  `λ·(1−β_o)·masked_distill_loss`(回归,见 4.4 适配)。
- **f(β,t)**:论文 Eq.7 `f(t)·(1−β)` ↔ `transfer_gate=1−beta_o`(λ 当 f(t) 线性衰减版)。
- **λ 线性衰减**:论文 f(t) 随训练衰减 ↔ `LinearScheduler(0.2→0, 300k)`。
- **Compatibility**:论文 §4.4 ↔ `gaussian_action_compatibility_all` 高斯核 + 
  `update_all_compatible_options`(见 4.4 适配)。

### 4.4 五大 A3C/PPO→TD3 适配核验:✅ 正确

架构层:概率分布 → 确定性动作转换彻底无残留。`legacy_actors.py` 虽叫 legacy,但两种
actor 都是确定性(legacy `Actor` 用 `tanh(net)`;`UpstreamFastTD3Actor` 复刻 FastTD3
`fc_mu+Tanh`),**无任何输出分布参数(均值+方差)的 A3C/PPO 式 actor 残留**。`SourcePolicy.act`
直接 `source_action=self.actor(source_obs)` 返回确定性动作。

五大适配点:
1. **蒸馏损失:跨熵→回归**。论文 §4.5 跨熵 H(π_o,π_θ)(π_o,π_θ 都是分布);TD3 确定性
   无分布跨熵无定义。`masked_action_distillation_loss` 支持 huber/mse/l1 默认 huber,标准
   policy distillation 在确定性策略间的形式。✅
2. **兼容性:概率密度→高斯核**。论文 §4.4 compatibility 是"源策略 π_o(a|s) 在学生实际
   动作 a 上的概率密度"(度量 transition 对源 o 的 on-policy 程度);TD3 源策略确定性无
   概率密度。`gaussian_action_compatibility_all` 用 `exp(-‖a−a_source‖²·mask/(2σ²))`(度量
   学生动作和源动作接近度)。语义偏移:原版是源策略概率密度(取决于源策略方差),你的
   是两个确定性动作欧氏距离高斯衰减(取决于 σ 超参默认 0.25)。可接受——TD3 下唯一可行
   选择,保留"学生动作越接近源动作,transition 越适合更新该源 Q_o"的核心意图。风险:σ 是
   新超参标定敏感(bank yaml 常见 1.5),σ 太小退化为 0/1 硬阈值,太大所有 option 兼容。
3. **option-Q 更新:on-policy→off-policy 兼容加权**。论文 Alg.2 Line 3 on-policy 只更新
   选中且实际执行的 option;TD3 off-policy replay 来自不同 behavior policy,只更新选中会浪费
   数据。`update_all_compatible_options=True`(默认)对所有动作兼容 option 更新 Q_o,compat
   加权 `q_loss=(compat·(q−y)²).sum()/compat.sum()`。null_option 兼容性 `null_col=ones`
   (null 不施加教师约束,对所有 transition 兼容)✅;`torch.maximum(compat,selected_oh)`
   保证选中 option 兼容性至少 1(强制更新,防退化)✅。
4. **actor 损失:policy gradient→DPG+蒸馏**。论文 actor=A3C `∂(R−V)²/∂θ` 或 PPO clipped
   ratio + `λ·f(β,t)·H`;你的 `actor_loss=rl_actor_loss+transfer_loss` 其中
   `rl_actor_loss=-qf_value.mean()`(DPG)、`transfer_loss=λ·(1−β_o)·distill_loss`。f(β,t)
   简化 `transfer_gate=1−beta_o`(λ 当 f(t) 线性衰减)✅;`beta_o.detach()`(行 1263)蒸馏
   只更新 actor 不顺便更新 β ✅;`pi_action=actor(pol_obs)` 当前 forward 动作非 buffer 旧动作
   ✅(蒸馏往当前策略拉)。
5. **termination β**:见 4.5(偏离但必要)。

### 4.5 偏离论文但属 HB 必要稳定化修复(不是 bug,改回会让 β 崩)

1. **Termination loss advantage 对称 clamp**:`advantage=(q_o−max_q+margin).clamp(-margin,
   +margin).detach()`。论文 Eq.5 无 clamp。原因:HB Q~100(v_min/v_max=±1000),原公式
   asymmetry 1000×,β 1000 步内饱和到上限,silencing (1−β) transfer gate。`option_update.py`
   docstring 已充分说明(论文公式+HB Q 量级+1000×不对称+β饱和+(1−β)门失效均点名)。
2. **β 的 [beta_min,beta_max] clamp**(默认 [0.05,0.95]):论文裸 sigmoid。原因防 sigmoid
   饱和。`option_module.py` 类 docstring 有机制说明。
3. **β warmup**(beta_warmup_steps 默认 20k):论文 β 与 Q_o 同步训练。原因早期 Q_o 不可信
   时学 β 会乱,warmup 期只前向不更新。
4. **xi 自适应 margin**(xi=0 时走 `0.8·(Q_top1−Q_top2)`):论文表 4 用固定 xi=0.001。
   是 PTF 原代码 convention,非论文公式字面。

### 4.6 固有张力(非 bug)

- **off-policy Q_o 学的是混合分布的 option-value,非严格 on-policy**:replay buffer 状态
  分布 ≠ 当前学生策略 occupancy(warmup 期还混入教师数据)。compatibility 加权是张力修补,
  但不是严格等价。这是 PTF×off-policy 组合的固有难题,可能是 PTF 在 HB 调度不可靠根因之一。
- **num_steps=1 硬约束**:`PTFReplayWrapper` 强制 `n_steps=1`(否则 raise)。PTF 可 n-step
  return,TD3 这里锁死成 1-step。option 的 U-value 只用 1-step bootstrap,长 horizon option
  credit assignment 更难——恰好是"教师执行段 credit 延迟"问题的实现侧根源之一。

---

## 5. 代码修正(本轮已执行)

### 5.1 修正原则

识别的"问题"分两类,处理方式不同:
- **真 bug**(mask 全空 edge case):改代码逻辑 + 加注释;
- **偏离论文但属 HB 必要稳定化修复 / 理论张力 / 设计约束**:**不改代码逻辑**(改回论文
  原版会让 β 饱和崩、Q_o 学不出),只补充注释说明偏离理由。

### 5.2 真 bug 修正(改代码,1 处)

**`fasttd3_ptf/ptf/compatibility.py` 的 `gaussian_action_compatibility_all` mask 全空 edge case**

现状:某 source action mask 全 0 时,`denom=masks.sum(-1).clamp_min(1.0)` 把分母锁 1,
`diff2=0→dist=0→compat=exp(0)=1`——把"mask 全空退化 source"静默判为"完全兼容",会参与
所有 Q_o 更新并学到噪声。修正:新增 `mask_sum=masks.sum(dim=-1)`,最后用
`torch.where(mask_sum>0, compat, torch.zeros_like(compat))` 强制全空 source compat=0(不参与
更新);docstring 详述 edge case 与 TD3 确定性 actor 适配语义。验证:数值测试通过(mask 全空
source compat=0,正常 source action 重合 compat=1.0);不改现有实验行为(bank 配置里 mask
不会全空,防御性修正)。

### 5.3 注释补充(不改逻辑,6 处)

| 处 | 文件:位置 | 补充 |
|---|---|---|
| B1 | `option_module.py:67` forward β clamp | 行内注释点名"偏离 PTF 论文 Eq.5 裸 sigmoid;HB 大量级 Q(~100,v_min/v_max=±1000)会让 β 饱和到 rail、(1−β) 传输门失效" |
| B2 | `option_update.py:13` `termination_margin` docstring | 点明"默认 xi=0 走自适应 0.8*(top1−top2) 是 PTF 原代码 convention;论文 Table 4 用固定 xi=0.001,两者不等价",HB Q 量级下差异 |
| B3 | `train_ptf.py:1434` `update_option` 函数顶 | docstring 说明"off-policy replay 下 Q_o 学的是混合分布(behavior policy 含教师/学生混合)的 option-value,非严格 on-policy;这是 PTF×off-policy(TD3)组合的固有张力,是 PTF 在 HB 调度不可靠潜在根因之一" |
| B4 | `train_ptf.py:1454` `update_all_compatible_options` 分支 | 点明"这是论文 Alg.2 Line 3 on-policy '只更新选中 option' 的 off-policy 扩展——对所有动作兼容 option 更新 Q_o 以复用 off-policy replay 数据" |
| B5 | `train_ptf.py:359` 配置默认 + `1481` 使用处 | 说明"β 延迟训练(偏离论文同步训练):早期 Q_o 不可信时学 β 会乱,故 warmup 期只前向不更新 β" |
| B6 | `ptf_replay.py:8` 类 docstring + `18` raise message | 显式说明"PTF 的 Q_o/β 按 per-transition 单步定义,n-step return 跨越多 option、破坏 per-step termination 语义;故锁死 n=1" |

### 5.4 验证结果

- ✅ 全部 5 个修改文件 `py_compile` 语法检查通过;
- ✅ A1 数值测试通过(mask 全空→compat=0,正常 source 行为不变);
- ✅ 没改动任何稳定化修复逻辑(β clamp / termination advantage clamp / β warmup / xi 自适应
  保持原状),不破坏现有实验;
- ✅ 没改动 FastTD3 backbone 或 PTF 五大适配核心逻辑。

---

## 6. 给 ChatGPT 的核心问题与我的立场

### 6.1 三点待 ChatGPT 裁定

1. **当前是否锁定 v2 完整六组件路线**(继续攻组件①找非行为信号 T),还是已实质退到
   v3(静态 RBO 保底收缩)?文档有张力:RESEARCH_ROADMAP(7-16)说 v2 active、v3 保底,
   但 `paper_core_contribution_reconstruction_v2.md` 文件头标 "2026-07-12 superseded by v3"。
   7-16 的代码清理+admission_control+P0/Phase-1 验证基础设施,到底是为"再攻组件①
   (用非行为信号族)做准备",还是已实质退到 v3 静态路线?
2. **Phase-1 bounded lease 阴性结果**在框架里定位为"机制失败"还是"验证基础设施暴露的
   问题"?从我读到的 `final_result.json` 看(basketball HARM、retention 全负),bounded
   lease 本身是阴性——它是在测一个候选①机制(admission lifecycle 本身当作 transferability
   体现),还是只是验证基础设施?
3. **机制新颖性 vs 验证严谨性的张力**:当前重心几乎全在验证工程上(23 轮复核、双 Gate、
   provenance 链),机制本身新颖性可能被验证严谨性盖过。`AdaptiveAdmissionController`
   自己的预注册 FAIL(行为 reward 第三次否定),当前有效的是静态 bounded lease——它
   能扛过严格验证,但"静态准入+到期驱逐"作为 contribution 可能不够 novel。建议把论文主
   贡献钉死在"humanoid 摔倒-负迁移 → safe/online bootstrap 注入"这条线,MCG gate 蒸馏
   降级为"已验证但非主线"配角(否则 10% 数字会反噬)。

### 6.2 我对齐的大方向(请 ChatGPT 校验)

在 HumanoidBench 做迁移 RL:简单任务(stand/walk/run/hurdle)为源,复杂任务
(stair/truck/package 等)为目标,目标是效率/上限/攻下 SOTA 攻不下的任务。当前有效方法
=教师轨迹注入(reward-bearing bootstrap)。最有 insight 的下一步:
**可达状态迁移 framing(方向①)+ 攻 SOTA-难任务(方向 A)+ 用非行为信号族重建 PTF
调度半身(换信号族攻组件①)**。

### 6.3 我的担忧

当前陷入的 admission 验证工程可能偏离了"HB 迁移 RL 让简单任务帮复杂任务"这个简单清晰
的大方向。建议:
- 不再围绕 cabinet 单点剂量/阈值/floor/horizon 档位无边界扩展;
- 把重心从"验证 bounded lease 是否过 Gate"转回"找一个能 reliably 衡量源对 student 学习
  增量的非行为信号 T",并据此闭环分配 source/student/replay budget;
- 诚实接受组件①行为信号族三连失败,换信号族(learning progress / TD 残差 / influence /
  可达状态)再攻,而非继续在 admission lifecycle 的工程细节上加深验证。

---

## 附录:本次讨论的关键证据文件索引

- 课题脉络:`docs/RESEARCH_ROADMAP.md`、`docs/archive/advisor_feedback_analysis_20260702.md`
  (§0-§19,导师三点意见+三信号失败模式+SHU/SIV/adaptive 三连失败)
- 当前验证:`docs/run_card_phase1_bounded_bank_lease.md`、
  `docs/data/p1_bounded_bank_lease/{final_result,delta_frozen,gate_a_report}.json`
- 核验对照论文:`papers/FastTD3.pdf`、`papers/PTF-arxiv.pdf`
- 官方 backbone:`fasttd3_ptf/official_code/FastTD3/fast_td3/fast_td3.py`
- PTF 实现:`fasttd3_ptf/ptf/{option_module,option_update,distillation,compatibility,
  source_policy,legacy_actors,option_selector,mcg}.py`
- 缝合主循环:`fasttd3_ptf/official_fasttd3_ptf/train_ptf.py`
- replay 适配:`fasttd3_ptf/official_fasttd3_ptf/ptf_replay.py`
- 本轮修正:见第 5 节,5 个文件已改并通过 py_compile + 数值测试
