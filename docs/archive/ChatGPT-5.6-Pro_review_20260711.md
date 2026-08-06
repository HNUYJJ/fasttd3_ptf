我认真阅读了你上传的两份最新记录，以及前面的完整实验演化过程。先给一个总判断：

> **你们目前的工作已经绝对超过“简单把 PTF 搬到 HumanoidBench 做验证”的水平，支撑 CCF-B 会议投稿是没有问题的；但距离 ICML 级别的方法论文，目前最大的差距不是实验量，而是“核心科学问题还没有被压缩成一个足够尖锐的贡献”。**

换句话说：

* **工程完整度：已经很强。**
* **实验严谨性：明显超过普通 CCF-B 论文要求。**
* **当前最大风险：论文故事容易变成“很多 transfer trick + 大量消融分析”，而不是一个清晰的新问题、新方法。**

我认为你现在最应该做的不是继续增加实验，而是重新定义论文核心。

下面详细分析。

---

# 1. 当前工作到底有什么价值？

先不要妄自菲薄。你们现在已经解决了几个真实存在的问题。

目前很多 policy transfer / policy reuse 工作默认一个假设：

> 一个 source policy 如果在 target task 上表现好，那么它就是一个好的 transferable policy。

但你们的大量实验实际上反驳了这个假设。

你们发现：

---

## Insight 1：Zero-shot performance ≠ transferability

这是你们目前最强的科学发现。

例如：

### crawl

source：

* walk
* run
* stand

zero-shot reward 很高。

但是：

最终 transfer：

负迁移。

原因：

walk/run 能让 humanoid：

* 站起来；
* 移动；
* 获得 alive reward；

但是不能提供 crawl 所需：

* 低姿态；
* 身体压低；
* 地面接触模式。

所以：

[
\text{policy competence}
\neq
\text{transfer value}
]

这是一个非常重要的问题。

---

## Insight 2：reward-bearing transition 比 teacher skill 更重要

这是你们 RBO 最核心的地方。

以前：

> teacher policy → student policy

你们现在发现：

更准确：

> teacher policy → target environment data distribution → learner improvement

也就是说 source policy 不是 teacher。

它是：

**data generator / intervention policy**

这个观点比传统 PTF 强。

---

比如 cabinet：

你们 P0/P1 审计非常关键。

发现：

stand：

* 稳定；
* 不摔；

但是：

run：

* hard progress 更强。

说明：

不是：

> humanoid 先站稳，所以所有 manipulation 都提升。

而是：

不同 source 产生不同的训练数据分布。

run source 产生：

* 更丰富 root motion；
* 更好的 approach state；
* 更接近 interaction state。

所以真正有价值的是：

[
D_{\pi_s}
]

而不是：

[
\pi_s
]

---

## Insight 3：transferability 不是 scalar，而是条件化对象

这是你们最有潜力的理论贡献。

现在已有证据：

同一个 source：

### run

对：

cabinet：

有效；

crawl：

无效；

truck：

可能有效。

所以：

transferability 应该写成：

[
T(\pi_s,T,t,e)
]

而不是：

[
T(\pi_s)
]

其中：

* (\pi_s)：source policy
* (T)：target task
* (t)：training stage
* (e)：target effect

例如：

run:

cabinet:
[
T>0
]

crawl:
[
T<0
]

这是一个非常好的论文核心。

---

# 2. 但是目前最大的问题在哪里？

我认为有三个。

---

# 问题1：RBO 本身创新程度还不够高

目前 RBO：

> reward-weighted bootstrap

如果单独拿出去：

审稿人可能会说：

“这是不是一种 informed replay / prioritized replay？”

因为：

你做的事情：

* source rollout；
* 根据 reward；
* 调 source probability；
* 修改 replay distribution。

容易被归类：

* curriculum learning；
* prioritized replay；
* policy reuse。

所以 RBO 不能作为唯一贡献。

---

# 问题2：T 目前还没有成为真正的方法

导师最初要求：

> 一个 transferability metric

但是目前：

你们实际上证明：

很多 metric 都失败：

* T0 failed
* Tcritic failed
* Tonline limited

这是好事。

但现在论文容易变成：

> 我们试了很多 T，最后发现 reward-weighted bootstrap 有用。

需要进一步抽象：

你们真正提出的不是：

一个 metric。

而是：

## Transferability evaluation problem

即：

> In humanoid policy transfer, what should a transferability measure predict?

然后你们回答：

不是：

* source return
* zero-shot reward
* single-step advantage

而是：

**future learner improvement caused by source-induced data.**

这个才是核心。

---

# 3. 我认为目前最有价值的论文主题应该改成：

## 不要：

> Reward-weighted Policy Transfer

太像技巧。

---

## 建议：

# "Learning What to Transfer: Data-Centric Transferability Estimation for Humanoid Reinforcement Learning"

中文：

> 学习什么值得迁移：面向人形强化学习的数据中心迁移性估计

核心问题：

已有 transfer 方法问：

> Which policy should be reused?

你们问：

> Which source-generated experience should influence the learner?

这是更高层的问题。

---

# 4. 重新组织论文贡献

我建议不要写：

Contribution 1:
Transfer Map

Contribution 2:
RBO

Contribution 3:
Benchmark

太弱。

应该：

---

# Contribution 1

## Transferability should be measured by induced learning value, not source performance

这是科学发现。

证据：

### Table：

| metric           | failure                  |
| ---------------- | ------------------------ |
| zero-shot return | crawl false positive     |
| critic advantage | pole false negative      |
| immediate reward | powerlift false negative |
| hard progress    | better aligned           |

这就是论文 motivation。

---

# Contribution 2

## Reward-bearing policy bootstrap

提出：

source policy as data intervention

形式：

source:

[
\pi_s
]

induces:

[
D_s={(s,a,r,s')}
]

不是蒸馏：

[
\pi_s \rightarrow \pi
]

而是：

[
D_s \rightarrow \mathcal{L}_{target}
]

这是方法核心。

---

# Contribution 3

## Effect-aware transfer analysis on HumanoidBench

不是简单 benchmark。

而是：

建立：

Source × Target × Effect matrix

例如：

| source | cabinet | crawl | truck |
| ------ | ------- | ----- | ----- |
| run    | ✓       | ×     | ✓     |
| stand  | △       | ×     | △     |
| hurdle | -       | -     | ✓     |

证明：

transferability is structured.

---

# 5. 那 baseline 怎么体现优势？

这是你最关心的问题。

现在不要问：

> 我的方法比 baseline 高多少？

应该问：

> baseline 失败在哪里？

---

## Baseline 1：scratch FastTD3

问题：

没有 prior。

你优势：

更少 interaction。

指标：

time-to-progress。

不是 final return。

---

## Baseline 2：PTF original

问题：

假设 source policy 可以作为 teacher。

你的优势：

PTF：

[
\pi_s \rightarrow imitation
]

你：

[
\pi_s\rightarrow data\rightarrow learner
]

实验：

bootstrap_only vs fixed distill。

你们已经有：

full ≈ bootstrap only。

这是非常关键。

说明：

传统 PTF 的 imitation 不是主要原因。

---

## Baseline 3：uniform source reuse

例如 rand。

问题：

不知道什么 source 有价值。

你的优势：

effect-aware source weighting。

Terrain 已经证明：

stair/slide/pole。

---

## Baseline 4：zero-shot transfer metric

例如：

APT-RL style advantage。

问题：

不能预测 delayed learning value。

你：

hard-progress evaluation。

---

# 6. 目前实验是否足够 CCF-B？

我的判断：

## 足够。

甚至偏强。

原因：

CCF-B 通常要求：

* 方法；
* baseline；
* 实验。

你们已有：

## 方法：

* PTF integration
* RBO
* source selection
* replay intervention

## 实验：

任务：

terrain:

* stair
* slide
* pole
* crawl

manipulation:

* cabinet
* truck
* spoon
* door
* basketball
* powerlift
* bookshelf

已经超过很多 CCF-B。

---

但是：

如果目标：

ICML/NeurIPS。

目前还缺：

一个“不可替代的新概念”。

---

# 7. 下一步我不建议继续加实验

我认为你现在最该做的是：

## 实验1（最高优先级）

完成：

## Data Value Probe

也就是你文件里提出的：

[
DV_s
====

P(L+D_s)-P(L)
]

这是最重要的。

因为它直接回答：

> source-generated data 有没有真正价值？

如果成立：

论文直接升级。

---

## 具体设计

不要大规模。

只做：

cabinet:

stage:

10k

sources:

* run
* stand
* student

流程：

保存：

[
L_t
]

包括：

* actor
* critic
* optimizer
* replay
* RNG

然后：

三个 intervention：

### A

加入 run data

### B

加入 stand data

### C

加入 student data

固定：

* transition 数；
* update steps；
* batch schedule。

测：

30k hard progress。

如果：

run > stand > student

三次 seed 稳定。

那么：

你的核心理论成立。

---

# 8. 第二个实验：预测性

如果 DV 成功：

证明：

早期 DV 能预测未来收益。

例如：

10k DV ranking：

预测：

100k performance。

比较：

* zero-shot return
* T0
* critic advantage
* immediate reward
* DV

如果 DV 最好：

这是论文 killer experiment。

---

# 9. 第三个实验：闭环

最后：

用 DV 做 source allocation。

例如：

不用：

reward weight

改：

DV weight。

比较：

RBO

vs

DV-RBO

如果：

DV-RBO > RBO

论文升级。

---

# 10. MCG 是否保留？

我的建议：

降级。

现在证据：

bootstrap_only ≈ full。

所以不要说：

MCG 是核心。

否则 reviewer 一问：

> remove MCG?

你不好回答。

定位：

MCG：

* option interface；
* safety module；
* future extension。

主线：

RBO + DV。

---

# 11. 当前最大风险

不是实验不足。

而是：

## claim 太散。

现在你们有：

* T
* RBO
* MCG
* replay
* abstain
* source map
* horizon

太多。

最终必须砍。

---

# 我建议最终论文只保留：

## 核心：

### Data-centric transferability

三个组件：

---

## 1. Source intervention

source policy generates target data.

---

## 2. Data value estimation

measure:

[
DV(\pi_s|T,t,e)
]

---

## 3. Data-aware bootstrap

use DV:

* select source
* allocate replay
* decide transfer budget

---

其他：

| 模块         | 处理                    |
| ---------- | --------------------- |
| MCG        | support               |
| Tcritic    | baseline              |
| horizon    | analysis              |
| abstain    | implementation detail |
| Effect Map | analysis tool         |

---

# 12. 最终论文一句话

我认为最有潜力的是：

> We show that policy transfer in humanoid reinforcement learning is fundamentally a data intervention problem rather than a policy imitation problem. We propose a data-centric transferability framework that estimates the learning value of source-induced experiences and uses them to bootstrap FastTD3, enabling efficient transfer across diverse HumanoidBench tasks while avoiding harmful reuse.

中文：

> 我们发现人形强化学习中的策略迁移本质不是“哪个教师更强”，而是“哪个源策略产生的数据能够改善目标学习器”。因此提出数据中心迁移性估计与基于价值的数据引导 bootstrap。

---

# 13. 我的最终建议排序

## 现在：

不要：

❌ 扩更多任务
❌ 调更多 T
❌ 加更多 source
❌ 修 MCG

应该：

---

## 第一：

做 DV pilot。

这是决定论文等级的实验。

---

## 第二：

如果 DV 成功：

做 DV prediction。

---

## 第三：

DV guided RBO。

---

如果 DV 失败：

退回：

RBO + effect analysis

仍然可以投 CCF-B。

---

# 最终评价

我的真实评价：

### 当前状态：

CCF-B：
**可以投。**

ICML：
**还差一个核心概念。**

这个核心不是更多实验，而是：

> 从“policy transfer”提升到“data intervention transfer”。

你们过去两个月的所有实验，其实已经在不断证明这一点：

* crawl 证明 policy reward ≠ transfer value；
* cabinet 证明 source identity matters；
* powerlift 证明 stability ≠ skill；
* basketball 证明错误 prior 会冲突；
* hurdle/truck 证明 source diversity matters；
* replay 实验说明数据组成影响 learner。

这些结果已经不是简单实验，而是在逼近一个新的问题定义。

现在最值得做的事情，不是继续修 RBO，而是证明：

**source-generated data 的价值，可以被度量，并可以指导迁移。**

如果这个闭环成立，这篇论文的层次会明显提升。
