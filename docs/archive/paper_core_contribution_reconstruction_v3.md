# 整篇论文核心贡献重构 v3：RBO 主方法与条件化迁移规律

日期：2026-07-12

状态：**当前主路线；已按最新实验总账校正；不启动新算法变体**

结果注册表：[`rbo_core_result_registry_v1.yaml`](../configs/experiments/rbo_core_result_registry_v1.yaml)

机制证据：[`dual_channel_transfer_evidence_matrix_v1.md`](dual_channel_transfer_evidence_matrix_v1.md)

下一阶段：[`core_mechanism_polishing_v4_plan.md`](core_mechanism_polishing_v4_plan.md)

## 1. 战略裁决

论文仍然研究 **HumanoidBench 上的跨任务迁移强化学习**。最终默认方法不是 OBRW，也不是被
SHU 否证的 closed-loop admission，而是现有证据支持的最简单版本：

> **RBO（Reward-weighted Bootstrap）= 目标环境短 probe 提供静态 source allocation prior，
> warmup 期间由冻结 source policy 产生带目标奖励的行为片段，与 student 自身片段共同进入
> replay，随后由标准 off-policy FastTD3 学出 source-free target policy。**

代码标签仍为 `wfix / safe_bootstrap + h25`。OBRW（student-as-arm + symmetric replay
attenuation）降级为**在线安全扩展与双通道机制消融**。原因不是追求简单，而是已有九任务裁决中
OBRW 相对静态 RBO 只有 slide 一个决定性净胜场，而第二批 basketball 上二者均 3/3 负迁移；
在线层没有形成 universal safety guarantee。

论文的 insight 不再寄托于一个“万能可迁移性分数”，而是四条有正反例的条件规律：

1. source 的主要作用是注入 **target-reward-bearing experience**，不是直接把完整目标技能蒸馏给
   student；
2. 加权选源的收益取决于 **source bank 内部的价值分化**，并非每个任务都优于 uniform；
3. 扩充 source bank 的价值同时受 **技能互补性与目标学习饱和**约束，probe 增量是必要非充分条件；
4. 负迁移沿 **execution/occupancy** 与 **replay/update** 两个通道传播，行为分数不能直接充当
   replay data utility。

这比“bootstrap 在一些任务有效”更强，也比“我们已经自动解决负迁移”更诚实。

## 2. 主方法 RBO

### 2.1 输入与接口兼容

给定目标 MDP、冻结 source bank `{π_i}` 与正在训练的 student `π_θ`。source checkpoint 不要求与
目标任务具有相同 task observation；bank 中的 `obs_adapter/action_adapter` 把目标 observation 映射到
source policy 所需输入，并把 source action 映回目标动作空间。它解决工程接口兼容，不等于学习了
通用跨任务表示，因此不作为主要算法贡献。

### 2.2 静态目标环境 probe 与相对分配

每个 source 先在目标环境执行短片段，得到静态 reward-bearing score `T_i^0`。训练开始后该分数不随
新轨迹实时更新，只用于 source bank 内的相对分配：

`p(i | teacher) = softmax(T_i^0 / τ)`。

必须收窄其含义：`T_i^0` 是 **cold-start allocation heuristic**，不是校准的 transfer ROI，也不是
go/no-go 判别器。terrain 中 source 分化约 20 倍时，加权相对 uniform 有明显收益；第一批 breadth
只有 2--3 倍分化时，二者基本打平；第二批低分区同时出现 powerlift 大正例与 basketball 大负例，
进一步否证“低分=负迁移”。

### 2.3 Warmup reward-bearing bootstrap

前 30k 环境步内，每个已到期的并行环境独立作一次 segment 决策：

- 以 `0.5` 概率选择 student；student 连续执行默认 25 步；
- 以 `0.5` 概率进入 teacher 分支，再按上述 softmax 只选择**一个** source；该 source 连续执行
  固定 25 步；
- segment 内不把 25 步再按多个 source 权重切碎。

无论行为来自 source 还是 student，buffer 都记录目标环境的真实 `(s,a,r,s',done)`；奖励始终是
target reward。warmup 后 source 全部撤出，student 独立控制并接受 source-free 评估。

### 2.4 Learner 与 replay

主方法使用 `bootstrap_only`：关闭 warmup 后的 MCG 行为 gate 与蒸馏，以隔离 bootstrap 本身。
RBO 不按 `T_i^0` 重加权 replay；critic/actor 沿用标准 FastTD3 的 off-policy replay sampling。
“reward-weighted”指 **行为数据生成阶段的 source allocation**，不是 prioritized replay。

保留历史数据而不是贪心只抽最新最高 return 轨迹，是因为单条 return 同时混合初始状态、轨迹长度、
随机性和策略质量；贪心会缩窄状态动作支持、重复相关样本，并让 critic 对少量高回报 occupancy
过拟合。RBO 的证据支持改变“谁来产生 warmup 数据”，不支持把 replay 改成 top-return buffer。

### 2.5 论文中的source与target范围

论文不是用一个固定source bank覆盖所有表，必须按实验阶段透明报告：

- 核心locomotion bank：官方 `stand / walk / run`；
- source-library实验：在locomotion bank上增加自训 `hurdle`；
- 第二批standard-9 bank：官方 `stand / walk / run / reach` + 自训
  `hurdle / stair / slide / crawl / pole`；
- 14个target：`stair / slide / pole / crawl / maze / truck / cabinet / door / spoon /
  bookshelf_simple / basketball / window / powerlift / balance_hard`。

不同表的结论必须绑定对应bank。尤其truck的`+229.9`是“加入hurdle后的source-library gain”，
不能和三locomotion bank结果混写成同一算法配置；第二批也不能被用来声称standard-9对所有任务安全。

## 3. 为什么该机制是迁移强化学习，而不只是初始化

冻结 source policy 实际控制目标环境，从而把跨任务先验转换成 target-MDP transition。student 没有
复制 source 参数，最终也不调用 source；迁移发生在 **behavior-induced data acquisition**：source
改变有限预算内 learner 能看到的状态、动作与奖励，再由目标 RL 自己完成 value fitting 和 policy
improvement。

因此正确主张是：

> source option 为 off-policy learner 提供暂时的行为支架与有目标奖励的数据覆盖；收益取决于这些
> 片段是否覆盖目标学习当前缺少的、可被 learner 利用的区域。

这也解释了为什么 locomotion source 能帮助 hurdle、slide、pole、maze、cabinet 或 powerlift，却不
代表它已经会完成开柜、投篮或举重；迁移的是有限预算学习条件，不是完整目标策略本身。

## 4. 三个机制层次与实证裁决

### 4.1 第一层：有奖励的数据注入

第一批 breadth 中 `rand≈RBO/OBRW` 而三者常高于 scratch，说明主要收益首先来自 frozen source 在
目标环境中产生 reward-bearing data。`bootstrap_only≈full`、`no_bootstrap`弱/负也把主要性能通道
定位到 warmup bootstrap，而不是 MCG 蒸馏。

边界同样明确：basketball 中 random、RBO、OBRW 全部低于 scratch，说明“有 source 数据”不是天然
有益；错误 occupancy 与 replay 支持可以系统性拖慢 target learner。

### 4.2 第二层：source bank 分化决定加权价值

terrain 的干净 3-seed 解耦为：`RBO(wfix)-uniform(rand)=+77.9`，12 个 task×seed pair 中
11 个为正，paired `t=3.08`；固定 source 权重、只改变 horizon 的总体效应则为 `−11.4, t=−0.46`。
这支持“source selection 是 terrain 主增益，horizon 不是总体混淆因子”。

但 maze/truck/cabinet 中 RBO 对 uniform 基本打平，说明加权不是无条件贡献。更一般的规律是：

> 当 source bank 内有明显强弱分化时，静态权重减少有限 teacher budget 的浪费；当全员近似可用时，
> uniform 已经足够，加权层的边际价值很小。

因此可把 `T_i^0` 的**分布形状/分化程度**作为“是否值得精细选源”的预训练诊断，而不能把分数绝对值
当作任务可迁移性的概率。

### 4.3 第三层：source bank 覆盖与任务饱和

在 truck 中加入 probe 更高、技能更互补的 hurdle source 后，静态 RBO 从 `1191.4` 提高到
`1421.3`，配对 `+229.9, t=3.47`，超过旧 source bank 的 OBRW；同样扩源在 maze 只有
`+0.3, t=0.16`。

由此得到可写入 discussion 的条件命题：

`downstream source-addition gain ≈ complementarity × remaining learning headroom`。

probe 没有新增相对优势时，新增 source 通常无益；probe 有增量仍不保证收益，因为目标可能已在现有
方法下饱和。truck 是“互补且有 headroom”，maze 是“有 probe 增量但已聚拢饱和”。

### 4.4 Return改善不自动等于目标技能迁移

论文必须把AUC证据与task-specific hard progress分层：

- **hurdle**：50k `move≈0.731 vs 0.356`，说明有真实早期跨障进展；100k约
  `0.922 vs 0.923`持平，只支持加速、不支持ceiling；
- **cabinet**：run优于stand的30k/100k hard progress在3 seed同向，且source-free评估长度相同，
  排除了“只是评估时活得更久”的单一解释；
- **maze**：更早获得checkpoint/导航进展，但后期趋于饱和；
- **powerlift**：虽然AUC `+77.6`稳定，现有stability-deconfounded审计没有举重skill完成证据，
  只能写return/sample-efficiency正例；
- **basketball**：姿态/viability信号不能挽救任务表现，三种子return也确认负迁移。

因此正文可用hurdle、cabinet、maze反驳“所有提升都只是站立平衡”，但不能把这一反驳推广到
powerlift等所有任务。return主表与hard-progress表必须并列，且结论使用各自证据粒度。

## 5. 双通道负迁移与在线扩展

### 5.1 两种暴露

source 影响 learner 的路径应分成：

1. **execution/occupancy exposure**：source 动作改变即时 reward、termination 与后续访问状态；
2. **replay/update exposure**：source-conditioned transition 在 source 撤出后仍被重复采样并改变
   actor/critic update。

SHU formal regression 中 behavior/handoff 为正却对应 downstream update effect 为负，说明不能用一个
短期行为分数同时批准 source 执行和 replay admission。

### 5.2 OBRW 的限定角色

OBRW 把 student 加入在线 arm，并用 target reward EMA 同时衰减坏 source 的执行份额与 replay
采样权重；actor/critic使用一致权重。它提供两条有价值的机制证据：

- crawl 中 execution-only onlineb 已提高 student share，却仍弱且高方差；对称 replay control 相对
  onlineb 恢复 `+94.7`，说明 replay 是持久通道；
- actor-only、critic-only、split 都弱于 both，说明 transfer replay intervention 需要维持
  actor--critic sampling coherence。

但 OBRW 不进入默认主方法：九任务 RBO/OBRW 裁决中，OBRW在slide决定性大胜、在spoon小幅胜，
静态RBO在cabinet胜，其余六任务打平；basketball 上 OBRW 仍比 scratch 低 `−74.0` 且 3/3 负。
在线收益集中在少数regime而非普遍出现。它能缓解部分
execution-harm/replay-harm regime，不能提供 exact student-only fallback。

## 6. 重构后的论文贡献

### Contribution 1：Reward-bearing Option Bootstrap

提出一个参数不迁移、最终 source-free 的跨任务 off-policy 迁移机制：用冻结 source option 暂时改变
target-MDP 数据获取，再由目标 learner 从统一 replay 中学习。通过 `full/bootstrap_only/no_bootstrap`
与 `scratch/random/RBO` 解耦，证明当前增益主要来自 reward-bearing bootstrap。

### Contribution 2：Source-bank-conditioned allocation law

提出并实证“选源价值取决于 source bank 分化、扩源价值取决于互补性与剩余 headroom”的条件规律。
terrain 强分化、breadth 弱分化、truck/maze 扩源对照分别提供正反例。`T^0` 被严格定位为相对分配与
source-library management signal，而非夸大的 universal transferability metric。

### Contribution 3：Dual-channel negative-transfer diagnosis

区分瞬时 execution/occupancy exposure 与持久 replay/update exposure，并通过 crawl onlineb/OBRW、
split replay、cabinet 2×2 和 SHU contradiction 给出机制证据。得到 actor--critic replay sampling
coherence 的实证设计原则，同时明确现有在线控制仍无法保证消除负迁移。

### Contribution 4：Broad humanoid transfer regime map

在 terrain、navigation、whole-body 与 manipulation 任务上报告 strong positive、weak positive、null、
negative、horizon-sensitive 与 saturation regimes。headline 正例包括 slide、pole、maze/cabinet、
truck+hurdle source；任务进展证据最强的是hurdle、cabinet与maze。powerlift虽是稳定AUC正例，但因
缺少hard-skill佐证而归入metric-boundary分析；边界还包括stair、crawl、basketball、window，以及
balance_hard的完整3-seed null regime（WFix≈rand）。论文报告有限预算sample efficiency/AUC，并单列hard progress，不把稳定性、站立时间或
return自动等同为目标技能完成。

## 7. 旧机制的去留

| 组件 | 当前位置 | 裁决 |
|---|---|---|
| source bank + obs/action adapters | 系统输入与兼容层 | 保留 |
| `T^0` source score | 静态相对 allocation prior | 保留，禁止称 calibrated ROI |
| reward-bearing bootstrap | RBO 主方法 | 核心保留 |
| student 0.5 branch | warmup 自主探索与数据混合 | 保留 |
| standard replay sampling | RBO learner | 保留，不做 top-return 贪心 |
| student-as-arm + symmetric replay weighting | OBRW 在线扩展 | 保留为机制/安全扩展，不作默认方法 |
| MCG gate/distillation | supporting/appendix | 代码保留，主方法用 `bootstrap_only` 关闭 |
| SHU closed-loop admission | failed measurement study | 封存 |
| hard abstain / T-gated | threshold-fragile ablation | 封存 |
| actor-only/critic-only/split | causal ablation | 保留为 coherence 证据 |
| multi-horizon arm | failed global extension | appendix limitation |
| EPS/DV/SIV/ROI predictor | 未证或高成本方向 | 暂停，不恢复 |

## 8. 论文主张边界

### 可以写

- RBO 在多个 HumanoidBench 任务提高有限预算性能，但效果具有 regime 条件；
- source weighting 在 source bank 强分化时显著优于 uniform；
- 扩源收益受 source complementarity 与 target headroom 联合约束；
- source execution 与 replay persistence 是不同负迁移通道；
- replay 干预需要 actor/critic sampling coherence；
- 最终目标策略 source-free，obs adapter 允许使用 observation layout 不同的 source。

### 不能写

- `T^0/T^online/SHU` 精确衡量 source 或 trajectory 的迁移 ROI；
- RBO 或 OBRW 对所有任务不弱于 scratch；
- student-as-arm 能严格退化为全程 student-only；
- 学到了通用完整目标任务技能；
- 普遍提高 asymptotic ceiling；
- MCG 蒸馏已具有与 bootstrap 同等扎实的独立贡献；
- 自动 temporal extent 或 replay data-value estimation 已解决；
- `balance_hard`全零probe下加权稳定优于uniform或实现了student-only fallback。

## 9. 成稿主表与叙事顺序

1. **主方法图**：static probe → weighted source segment/student segment → target-reward replay →
   source withdrawal → source-free FastTD3 student；
2. **terrain 因果表**：scratch / random / RBO / safe-h50，突出 selection 与 horizon 解耦；
3. **breadth 表**：正例、null 与 negative regime 同表，不只展示成功任务；
4. **source-library 表**：truck 大增益 vs maze 饱和；
5. **双通道机制表**：onlineb / OBRW / actor-only / critic-only / split + SHU contradiction；
6. **边界表**：stair、crawl、basketball、window，以及balance_hard的all-zero-probe null regime。

方法章节先讲 RBO，再讲条件规律；OBRW 放在“negative-transfer diagnosis and optional control”小节，
而不是把整篇论文命名为 CE-RBO。

## 10. 下一步与停止规则

经用户授权，`balance_hard`两个历史OOM cell已按原配置恢复完成；当前不再启动其他100k实验。先完成：

1. 用结果注册表逐项追溯 headline run 配置、seed、AUC 窗口和 source bank；
2. 完成v4机制冻结门：RBO形式化、bank separability离线审计、complementarity/headroom操作化；
3. 生成论文abstract、method figure、主表与claim ledger，作为机制压力测试而非直接定稿；
4. 只有在机制/成稿审计发现一个会阻止核心主张成立的唯一缺口时，才讨论一个最小补证实验。

目前真正未解决的问题是**如何在低成本下识别 replay/update data utility 并实现可信 exact fallback**。
SHU 与 OBRW 都没有解决它；论文应把这一点作为明确限制与 future work，而不是继续用小实验包装成已完成
机制。
