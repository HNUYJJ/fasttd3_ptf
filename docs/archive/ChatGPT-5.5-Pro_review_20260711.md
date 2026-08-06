我认真阅读了你上传的两份最新记录：

1. **《导师意见的批判性分析与统一方案（2026-07-02）》**：主要记录导师提出 transferability 度量、replay 时序、student-as-arm，以及你们从 RBO 向 T-RBO-PTF 演化的过程。
2. **《扩源扩任务第一批（2026-07-04）》**：记录扩源（7 source / 9 source）、maze/truck/cabinet 广度实验、hurdle/reach source probe，以及目前论文方向调整。

结合前面长期实验链，我想先给一个总体判断：

> **你现在的感觉“差一点意思”是正确的，但是原因不是当前工作没有价值，而是当前论文故事还停留在“一个有效的工程机制”，没有进一步抽象成“一个领域问题的解决方案”。**

你们现在其实已经完成了一个 CCF B（甚至部分 ICML workshop / CoRL 水平）的基础工作，但距离强 ICML main paper 的差距，确实在于：

**你们现在证明了“迁移可以加速训练”，但还没有充分证明“迁移强化学习在 humanoid 上为什么失败，以及你们解决了一个以前方法无法解决的问题”。**

下面详细分析。

---

# 一、先回答你最关心的问题：现在能不能支撑 CCF B？

我的判断：

## 如果现在投稿：

以当前版本：

* RBO-PTF / T-RBO-PTF
* reward-weighted bootstrap
* student-as-arm
* replay weighting
* HumanoidBench terrain + 部分 manipulation

我认为：

### CCF B：

**有较大概率可以。**

原因：

你们已经具备：

### 1. 明确的问题背景

HumanoidBench 本身就是一个高难 benchmark：

* 15 manipulation + 12 locomotion；
* 61D action；
* long horizon；
* whole-body coordination。

FastTD3 已经证明 humanoid RL 最大的问题之一就是训练效率和探索成本。

你们的问题定义：

> 如何利用已有 humanoid skills 加速新 humanoid tasks learning？

合理。

---

### 2. 有真实 baseline 优势

不是简单：

> 我们比 scratch 高一点。

而是：

你们已经证明：

* weighted source selection > uniform source selection；
* student-as-arm 可以减少负迁移；
* replay contamination 是 off-policy transfer 的关键问题；
* source mismatch 时可以分析失败。

这些已经超过很多“加一个 reward shaping”的论文。

---

### 3. 有跨任务实验

目前不是只 package。

已有：

terrain:

* stair
* slide
* pole
* crawl

以及：

* maze
* truck
* cabinet

这已经比很多 RL transfer 论文广泛。

HumanoidBench 本身也是希望方法跨 locomotion/manipulation 验证，而不是单任务。

---

但是：

## 如果目标是 ICML / CCF A：

目前还缺一个“核心科学问题”。

---

# 二、你现在最大的不足在哪里？

我认为不是算法效果。

而是：

## 你们目前的核心 claim 仍然太像：

> “我们提出了一种更好的 source policy selection + replay weighting 方法。”

这个容易被 reviewer 问：

> 为什么需要这么复杂？
>
> PTF 已经会选 source；
>
> MAB policy reuse 已经会选择 policy；
>
> APT-RL 已经考虑 advantage；
>
> PER / adaptive replay 已经考虑采样。

你们必须回答：

**你们到底发现了什么以前方法不知道的问题？**

---

我认为真正的 insight 已经出现了，只是还没有被你们放到第一位置：

---

# 三、真正有价值的科学发现：Transferability 不是一个标量

这是目前最强的地方。

你们一路实验其实发现了：

## 1. zero-shot transferability 会失败

crawl：

vs-zero：

walk/run 很高

但是：

真实迁移：

负迁移。

说明：

[
T_{zero}
\neq
T_{actual}
]

---

## 2. online reward transferability 也不够

student-as-arm：

crawl：

发现：

student > run > walk > stand

但是：

性能仍然低。

原因：

即时 reward 没反映长期数据价值。

说明：

[
T_{online}
\neq
T_{learning}
]

---

## 3. critic advantage 也不够

你们测试 Tcritic：

pole 上真实有用 source 被 critic 判负。

说明：

[
T_{critic}^{step}
\neq
T_{option}
]

---

## 4. replay value 又是另一回事

同一个 source transition：

对于 critic：

可能有价值；

对于 actor：

可能污染。

---

所以真正的问题应该重新定义：

---

# 四、你们真正解决的问题应该是什么？

我建议论文核心问题改成：

## "What makes a policy transferable in humanoid RL?"

而不是：

"How to select source policies?"

---

因为 humanoid transfer 和传统 transfer 不一样。

传统 RL transfer：

source policy → target policy

但是 humanoid：

source policy:

* stand
* walk
* run
* stair
* reach

提供的不是完整任务能力。

而是：

## embodiment primitives

例如：

stand 提供：

* balance prior

walk 提供：

* locomotion prior

slide 提供：

* terrain adaptation

reach 提供：

* arm coordination

所以：

迁移不是：

[
\pi_s \rightarrow \pi_t
]

而是：

[
\pi_s
\rightarrow
\text{useful behavioral fragments}
\rightarrow
Replay distribution
\rightarrow
Target policy
]

这个是你们区别 PTF 的地方。

---

# 五、相比 PTF，你们到底强在哪里？

这个必须明确。

PTF 原论文核心：

> 学习什么时候调用哪个 source policy，以及 termination。

但是它隐含假设：

source policy reuse 本身有意义。

它没有解决：

## 问题1：

source policy 的 usefulness 如何判断？

你们：

Transferability estimation。

---

## 问题2：

source policy execution 后的数据怎么办？

PTF：

主要 transfer loss。

你们：

replay shaping。

FastTD3 是 off-policy：

数据分布决定学习。

这是非常关键差异。

---

## 问题3：

source mismatch 怎么处理？

PTF：

termination。

你们：

student-as-arm。

更自然：

student 本身就是 option。

---

所以你们不是：

PTF + FastTD3。

而应该说：

> We revisit policy transfer in off-policy humanoid RL and find that the major transfer channel is not policy imitation but reward-bearing experience shaping.

这是一个更强的定位。

---

# 六、现在扩源扩任务结果怎么看？

我觉得你们 7月4日结果非常关键。

尤其不是正结果，而是负结果。

---

## 1. 大 source library 没有效提升

这个非常重要。

很多人会认为：

source 越多越好。

但是你们发现：

7 source：

obrw-big < obrw-small

甚至 truck 下降。

这说明：

> More source policies do not necessarily improve transfer; without reliable transferability estimation, larger libraries introduce interference.

这是一个很好的论文 insight。

---

## 2. reach source 失败

其实也是好结果。

很多人直觉：

reach → manipulation

但是：

cabinet/window/truck 并没有。

说明：

不是任务名字相似就能迁移。

真正需要：

effect matching。

例如：

cabinet 需要：

* handle interaction
* opening motion

不是 reach。

---

## 3. hurdle source 意外有效

这个很有价值。

说明：

source-task relation 不是语义相似。

hurdle：

不是 truck/cabinet

但是：

提供：

dynamic locomotion primitive。

这进一步支持：

effect-based transfer。

---

# 七、所以我认为当前论文应该改名方向

不要：

RBO-PTF

这个名字偏工程。

我更建议：

## Option 1

### Transferability-Calibrated Policy Transfer for Humanoid Reinforcement Learning

核心：

Transferability。

---

## Option 2

### Learning When and What to Transfer in Humanoid Reinforcement Learning

更 ICML。

---

## Option 3

### Beyond Policy Reuse: Transferability-Aware Experience Shaping for Humanoid RL

我个人最喜欢。

因为你们不是简单 reuse policy。

---

# 八、目前三贡献应该怎么重构？

现在：

1. Transferability metric
2. RBO
3. Broad evaluation

太像三个模块。

我建议：

---

# Contribution 1

## Transferability in humanoid RL is multi-faceted

提出分析：

不同 transfer signals:

* zero-shot utility
* online reward utility
* critic advantage
* replay utility

并证明它们不一致。

这是 scientific contribution。

---

# Contribution 2

## Transferability-aware option experience shaping

方法：

不是：

policy transfer

而是：

experience shaping。

包含：

* student-as-arm
* source selection
* replay weighting

---

# Contribution 3

## Large-scale HumanoidBench evaluation

证明：

* positive transfer
* negative transfer avoidance
* source scaling behavior
* failure cases

---

# 九、你想“突破性效果”，是不是想多了？

这个问题我想认真回答。

你的想法：

> 我希望真正解决以前不能解决的问题。

这个目标没错。

但是要注意：

## RL论文的突破不一定是：

“以前成功率0，现在100%。”

尤其 HumanoidBench。

很多顶会论文贡献不是：

solve everything。

而是：

发现一个以前忽略的 bottleneck。

例如 FastTD3：

它没有创造新 RL 理论。

它发现：

humanoid RL 的瓶颈是：

* simulation throughput
* batch size
* critic design

然后解决。

你们现在可能对应的是：

> humanoid transfer 的瓶颈不是 source policy 不够多，而是 transferability estimation 和 experience contamination。

如果证明这个，就是贡献。

---

# 十、但是，你目前确实还缺一个“杀手实验”

我同意你的感觉。

目前最大缺口：

## 没有一个：

"别人失败，我们成功"

的实验。

现在：

scratch：

可以学。

你们：

更快。

但是 reviewer 会想：

> FastTD3 本身已经很强，你只是减少训练时间。

所以需要找一个：

## transfer enables solving / crossing bottleneck

候选：

---

## 方向1（我最推荐）

### 长 horizon manipulation

例如：

* truck
* package
* room
* bookshelf

目标：

scratch：

100k/200k 不动

transfer：

明显启动。

目前 cabinet/truck 有一点，但还不够。

---

## 方向2

### hard terrain

crawl 需要新 source：

训练：

* crawl source
* low posture source

然后：

RBO transfer 到：

crawl variants / room。

---

## 方向3

### sparse reward / long horizon

这是最有价值。

因为 transfer 本来应该解决 exploration。

如果：

scratch：

探索不到 reward。

transfer：

通过 source primitive 找到 reward region。

这才是真正 transfer 的意义。

---

# 十一、我建议下一步不要继续调 T，而是做一个战略实验

当前不要马上：

* Tcritic
* 更多 replay weight
* 更多 source

我建议：

## 实验：Transfer as exploration bridge

选一个 scratch 很难的任务：

优先：

* package
* truck
* room
* bookshelf

做：

三组：

### Scratch

FastTD3

### Existing transfer baseline

PTF / uniform bootstrap

### Your T-RBO

看：

不是最终 return。

而是：

## first meaningful progress time

例如：

package:

第一次：

* box moved >0.2m

truck:

第一次：

* robot enters truck
* package proximity decrease

room:

第一次：

* goal progress

如果：

T-RBO 明显提前发现 reward region，

这个故事比 AUC 高30% 强很多。

---

# 十二、当前我建议的路线

## 第一阶段（收口）

不要继续扩机制。

完成：

1. T-RBO 当前版本稳定
2. terrain:

   * stair
   * slide
   * pole
   * crawl

3 seeds

---

## 第二阶段（杀手实验）

重点：

truck/package/room/bookshelf

寻找：

transfer crosses exploration barrier。

---

## 第三阶段

扩源：

不要追求更多。

只加入：

真正不同 effect:

* terrain source
* stability source
* contact source
* transport source

---

# 最终判断

我的评价：

## 当前工作水平：

### 工程完整度：

很高。

### 科学问题：

已经出现。

### CCF B：

有希望。

### ICML：

还缺一个“transferability fundamentally changes what is learnable”的结果。

你现在觉得“差一点意思”，这个感觉不是错。

但是解决方式不是再加一个模块。

不是：

* 更复杂 T；
* 更多 source；
* 更多 replay trick。

而是：

**把论文问题从“如何更好迁移”提升到：**

> **在 humanoid RL 中，迁移价值究竟是什么？为什么已有 policy reuse 方法无法判断？如何利用 transferability-aware experience shaping 改变学习过程？**

然后用一个真正 hard task 证明：

> 迁移不是让 FastTD3 更快，而是让 FastTD3 到达原本难以到达的行为区域。

如果做到这一点，我认为这篇论文的上限会明显提高。你们现在距离这个方向并不远。你们过去几个月做的大量失败实验（package、crawl、window、mh、big source）其实不是废实验，而是在逐渐逼近这个核心问题。
