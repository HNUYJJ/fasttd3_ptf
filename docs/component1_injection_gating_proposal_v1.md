# 组件①重攻方案 v1：源数据注入的判定、时机与事后无害化

> 日期：2026-07-16
> 作者：Claude（PI 指示全权负责，本文档同时作为完整工作记录，供 ChatGPT 后续对抗性审查）
> 状态：**v1 已经 ChatGPT 对抗性审查（Major revision）+ Claude 反向对抗审查——
> 双方裁决与融合后的 v2 修订见文末 §8（§1-§7 保留 v1 原文不改动，供审计对照；
> 以 §8 为当前有效版本）**
> 回答 PI 的问题：(1) 如何在前期准确判断源是否有害？(2) 如何保证注入后的轨迹后期
> 不再对 student 造成毒害？(3) 如何界定"注入窗口"——什么时间注入有益、什么时间有害？

---

## 0. 一句话答案

三重否定的证据与文献共同指向同一个结论：**"事前一次性准确判断"在 crawl 型
隐性毒源上很可能是不可达的目标；正确的设计是把"判断"从一次性决策改造成
持续重估，并让误判的代价有界、可逆**。具体 = 四层机制：
**低剂量注入（R2D3 启发）+ 逐样本 critic 门（Q-filter 推广）+ 窗口级非行为判据
（T^critic 族）+ 撤销后 critic reset（primacy-bias 启发，把不可逆变可逆）**。
任何信号进入在线闭环前，必须先通过**离线"考古"判别力 gate**（用已有 checkpoint
库和已知好/坏源标签测判别力，零训练成本）。

---

## 1. 问题形式化

### 1.1 毒害的精确定义（从 crawl 证据链提炼）

设 t 时刻 student 为 π_t、critic 为 Q_t、buffer 分布为 D_t。源 π_i 执行 segment
产生轨迹 τ_i 注入 buffer。毒害不是 τ_i 的固有属性，而是三元耦合
**(τ_i 的状态-动作分布, 任务最优路径, 当前学习动力学)**：

- **有益**：τ_i 覆盖了 student 欠缺、且任务最优路径必经的流形，为 Bellman 备份
  链提供 reward-bearing 支撑（hurdle/cabinet/truck 的 stabilization bootstrap）；
- **无关**：τ_i 的流形与最优路径无交，仅有 displacement 代价（powerlift 的 crawl/reach 源）；
- **毒害**：τ_i 把 critic 的 argmax 锚定在**非最优路径必经**的中等回报盆地
  （crawl 上的 walk/run：即时 reward 体面、状态分布是死路）。

关键机制（tandem effect 给出理论基础）：教师数据里 student 自身动作 π_t(s) 从未
被执行 → critic 在这些状态上对 π_t(s) 的估值靠外推；而教师动作有真实 reward
支撑 → argmax 系统性偏向教师流形。**毒害的载体是 critic 参数，不是 buffer 槽位**
——这就是撤销/降权只能"止血"不能"解毒"的原因（crawl obrw +94.7 但不翻正）。

### 1.2 注入窗口的形式化

对每个 (source i, 时刻 t) 定义注入决策。理想判据 U_i(t) 应满足：
- **相对性**：相对当前 π_t/Q_t（v2 组件①定义），非静态 source return；
- **前瞻性**：覆盖驻留期外部性——τ_i 注入后要驻留 ~T_res（circular buffer
  51200 步，warmup 数据驻留到 ~81k 步=全程 80%），其价值随 student 演化会**变号**；
- **可操作口径**：
  - `t_open(i)`：student 在源覆盖流形上自给不足（far-policy 且源动作占优）；
  - `t_close(i)`：源动作在 student 自己的价值观里不再占优（advantage 过零）。

crawl 型的诚实结论：`t_open` 从未应该打开，但**零交互事前判定它很可能 ill-posed**
（zero-shot 探针系统性低估 dynamic 源的教训 + tandem effect 均支持"必须交互
才能知道"）。因此方案的重心不是找完美先知，而是**让打开错误窗口的代价有界**。

### 1.3 剂量算术（为什么现状对误判零容忍）

当前 admission_bootstrap：教师 mass ≈50% → warmup 30k × 128 env × 0.5 ≈
**192 万条教师 transition ≈ buffer 的 29%，驻留至 ~81k 步**。在这个剂量下，任何
判据的反应延迟（×128 env/步的注入速率）都会让误判变成致命剂量。
对照 R2D3 的发现（见 §2）：最优 demo 比例是 **1/256**——我们比它高约两个数量级。

---

## 2. 文献调研结果（2026-07-16，检索过程见 §6 工作日志）

### 2.1 直接可用的机制先例

| 文献 | 机制 | 对我们的启发 |
|---|---|---|
| **R2D3**（Paine et al. 2019, [arXiv:1909.01387](https://arxiv.org/abs/1909.01387)） | demo ratio 消融：**最优比例极小但非零（~1/256）**，过高则过拟合 demo 损害泛化；demo 的作用机制是"引导探索"而非直接提供学习主料 | **低剂量注入假说 H1**：教师数据的 bootstrap 价值可能在低剂量就饱和，而毒害随剂量线性增长——剂量本身就是最便宜的第一道防线 |
| **Q-filter**（Nair et al. 2018, [arXiv:1709.10089](https://arxiv.org/abs/1709.10089)） | BC loss 仅在 Q(s,a_demo) > Q(s,π(s)) 时启用；"critic 批准"式逐样本门，**自动退火**（student 超过 demo 后门自动关闭） | 推广到 replay 采样权重：**逐样本 advantage 门**（§3.B），实现"注入后价值持续重估" |
| **ReF-ER**（Novati & Koumoutsakos, ICML 2019, [arXiv:1807.05827](https://arxiv.org/abs/1807.05827)） | importance ratio ρ∉[1/c_max, c_max] 的 far-policy 样本**跳过梯度**；c_max 随训练退火 | off-policyness 是独立于"好坏"的第二维度；确定性策略可用**动作距离** ‖a_src−π_t(s)‖ 作代理（便宜、不依赖 critic 打分正确性） |
| **Balanced Replay**（Lee et al. 2021, [arXiv:2107.00591](https://arxiv.org/abs/2107.00591)） | 训练 density-ratio 估计器测样本"on-policyness"作采样优先级，offline 数据随 online 进程自动降权 | 教师数据 = 我们的"offline 数据"；on-policyness 优先级是 TTL 的连续版 |
| **JSRL**（Uchendu et al. 2023, [arXiv:2204.02372](https://arxiv.org/abs/2204.02372)） | guide horizon 按 student 移动平均表现收缩；tolerance 过大（过早退出 guide）性能下降 | 注入窗口的**行为侧**先例；"按 student 能力退火教师暴露"有理论支撑（Phase B baseline 之一） |
| **Primacy bias / resets**（Nikishin et al., ICML 2022, [arXiv:2205.07802](https://arxiv.org/abs/2205.07802)） | 深度 RL 系统性过拟合早期数据；**周期性重置网络后几层、保留 buffer 重训**可消除 | **撤销后 critic reset**：唯一能"解毒已写进参数的偏差"的现成手段——把不可逆决策改造成可逆（§3.C） |
| **Tandem effect**（Ostrovski et al. 2021, [arXiv:2110.14020](https://arxiv.org/abs/2110.14020)） | 相同数据流+相同算法，被动学习者灾难性掉队；根因=欠覆盖动作上的外推误差 | crawl 毒害机制的理论基础；**警示**：教师数据对 student 是"被动数据"，其危害内生于函数逼近+分布固定，不是实现 bug |
| **PCGrad 族**（Yu et al. 2020） | 梯度余弦为负 = 干扰冲突，投影消除 | 梯度余弦可作**诊断信号**（教师层 batch vs 学生层 batch 的梯度对齐度，§3.D3） |

### 2.2 同题工作的定位

- **Discarding Erroneous Knowledge Online**（Notsu et al., PRIMA 2024）：agent 在线
  自测 transferred knowledge 对学习的效应、有害即丢弃——**问题意识与我们相同**，
  但在 tabular/多智能体小环境，其"效应度量"仍属行为反馈族，粒度粗；证明该问题
  在社区活跃但**无 humanoid 尺度 off-policy 的成熟方案**——我们的空间还在。
- **bandit 式在线选源**（Li & Zhang 2017 等）：把选源当 bandit——我们的 onlineb
  正是其行为信号实现，教训已入 [EXPERIMENT_LOG](EXPERIMENT_LOG.md)。
- DQfD/R2D3 的 demo-ratio 线：**固定比例是已知反模式**（我们的 repetition
  divergence 是其极端形式的独立再发现）。

### 2.3 最新工作（2025-2026，第二轮检索补充）

| 文献 | 机制 | 对方案的影响 |
|---|---|---|
| **A Snapshot of Influence / IIF**（**NeurIPS 2025 Oral**, [arXiv:2505.19281](https://arxiv.org/abs/2505.19281)，代码 [LDAORL/LDA-ORL](https://github.com/LDAORL/LDA-ORL)） | 在线 RL 首个**局部数据归因**框架：对"最近训练 buffer 窗口"做归因（信号=样本训练损失梯度与目标函数梯度的**梯度相似度**，目标含动作一致性与累计回报）；IIF 迭代过滤有害样本，提升样本效率与最终回报 | **直接强化 D3（梯度对齐）**：顶会 Oral 证明梯度相似度归因在在线 RL 可行有效；其"局部窗口归因"与我们 stage-window 结构同构。限制=PPO/on-policy——**迁移到 off-policy TD3 是空白，正是我们的位置** |
| **WSRL**（Zhou et al., **ICLR 2025**, [arXiv:2412.07762](https://arxiv.org/abs/2412.07762)） | offline-to-online 微调**不需要保留 offline 数据**：warmup 期用少量 rollout"重校准"Q 函数后完全丢弃 offline 数据 + 高 UTD 在线训练，比保留数据的方法更快更高 | **冲击性启发**：教师数据的正确用法可能是"短期校准、用完即弃"而非长期驻留——支持比降权更激进的选项：**warmup 结束后教师数据主动退役**（TTL=warmup 长度），见 §3.A+ |
| **Adaptive Replay Buffer**（[arXiv:2512.10510](https://arxiv.org/abs/2512.10510)，2025-12） | "on-policyness"轻量指标（轨迹与当前策略的对齐度）动态加权采样，learning-free；D4RL 上缓解 offline→online 早期退化并提升终点 | 支持 **D4 动作距离 staleness** 作为免学习的 on-policyness 代理（比 Balanced Replay 的 density-ratio 网络更轻） |
| **QoQ / influence 式 demo curation**（[arXiv:2603.09056](https://arxiv.org/abs/2603.09056)，2026） | influence functions 定义数据质量=对验证集损失下降的贡献；轨迹级聚合降噪 | influence 路线在机器人数据 curation 已实用化（imitation 侧）；轨迹级聚合的降噪技巧可借给 D2/D3 |
| **Plasticity 工具箱 2025-26**（ReDO、ReGraMa、plasticity injection、[soft parameter reset](https://arxiv.org/abs/2411.04034)；综述见 [Lyle 2025](https://clarelyle.com/posts/2025-09-06-plasticity-survey.html)） | 从"重置后几层"进化为**定向重置**（按 dormant 神经元/梯度幅值选择性重置），性能坑更小 | **降低 C 层成本**：撤销后的 critic 解毒可用定向 reset 而非全层 reset |
| 反面确认 | 截至 2026-07 检索，**off-policy actor-critic 上"有害教师/源数据识别"的专门工作仍是空白**（现有 attribution 都在 on-policy 或 imitation 侧） | 贡献空间确认 |

### 2.4 文献没有的（我们的潜在贡献点）

- 没有工作显式研究**"引导型好源 vs 隐性毒源"在行为信号下不可区分**这个判别难题
  （我们有三重否定的干预证据）；
- 没有工作把"注入决策的驻留期外部性"（数据在 buffer 里存活期间价值变号）作为
  一等公民建模；
- reset 文献治疗的是"自产数据的 primacy bias"，没有人把它用作**源撤销后的解毒**。

---

## 3. 方案：四层防线（Dose → Gate → Verdict → Undo）

> 设计原则：每一层独立可消融、独立可验证；任何一层失效不放大其他层的风险。
> 全部机制生长在现有 admission lifecycle 基础设施上（PTF 框架内，零新网络）。

### A. 低剂量注入（Dose）——把误判的暴露压低一个数量级

教师 categorical mass 从 ~0.5 降到 **η∈[0.05, 0.15]**（消融定），可选配
**JSRL 式退火**（按 student eval 移动平均收缩 η）。
- 依据：R2D3（1/256 最优）+ 我们自己的证据（bootstrap 收益来自"引导探索出
  reward 事件"，不需要教师数据当学习主料）；
- **可证伪假说 H1**：低剂量保留大部分正迁移（truck/powerlift 加速），毒害按剂量
  缩小（crawl 伤害显著变浅）；
- 风险：正迁移可能确实需要剂量（hurdle 型"拉出卡死区"）——消融矩阵直接回答。

**A+ 变体（WSRL 启发，2025 新证据）**：教师数据**主动短驻留**——warmup 结束
（authority 释放）时不只切换配额，而是把教师层数据从可采样集中退役（TTL=warmup
长度），此后纯 student 数据 + 高 UTD。WSRL 证明"offline 数据用于早期 Q 校准后
完全丢弃"不但可行还更优——若成立，可从根上消灭驻留期外部性（毒数据最多活到
30k，而非 81k）。与 lifecycle 的 provenance 基础设施天然兼容（按 provenance 退役
即可）。作为消融臂加入。

### B. 逐样本 critic 门（Gate）——"注入后不再毒害"的主机制

replay 采样时对教师层样本按当前 critic 重估加权：

```
A_t(s, a_src) = minQ_t(s, a_src) − minQ_t(s, π_t(s))      # 双 head min,天然悲观
w(s, a_src)   = σ(A_t / T) 或 clip(A_t,0,·)/Z              # 形式消融定
```

- 语义：教师 transition 的采样权重 = "它的动作在 student 当前价值观里还有没有
  信息量"。student 超过教师后权重自动衰减到底（Q-filter 的自动退火性质）——
  **这直接实现"注入时有益、后期自动退场"，注入决策不再一锤定音**；
- 与 T^critic 的关系：同一公式的逐样本版（导师 07-02"三处使用"中 replay 加权的
  critic 实现），源级聚合 E_s[A_t] 即 T^critic_i(t)，供 D 层窗口判据复用；
- 计算成本：每 batch 多一次 critic 前向（教师层样本），可接受；
- **已知风险 R1（鸡生蛋）**：crawl 型毒害恰恰使 critic 高估教师流形 → 门失灵。
  这是本方案最大的单点风险，**用考古实验 P2 直接测**（§4），并由 A 层（低剂量下
  critic 塑形弱）与 D4（不依赖 critic 的 staleness 维度）对冲。

### C. 撤销后 critic reset（Undo）——把不可逆变可逆

当 D 层判据撤销某源（或 exact abstain）时，除现有的 lifecycle 清理外，增加：
**重置 critic（及可选 actor）的后几层 → 保留治理后的 buffer → 高 replay 比重训**
（Nikishin et al. 的 reset 配方）。
- 语义：数据层的毒已由 lifecycle 排掉，参数层的毒靠 reset 洗掉——回答 PI 问题
  的第二半："注入之后，后期不会再造成害处"；
- 成本：reset 后短暂性能坑 + 重训 wallclock；仅在撤销事件触发，不是周期性；
- **可证伪假说 H2**：crawl 上"撤销+reset" 的终点 ≥ scratch（把负迁移清零），
  而无 reset 的撤销（adaptive v1 已测）做不到。

### D. 窗口级非行为判据（Verdict）——换信号族重攻组件①

四个候选信号，**全部先过离线考古 gate（§4）再谈在线闭环**：

| 信号 | 定义 | 性质 |
|---|---|---|
| D1 源级 critic advantage | T^critic_i(t)=E_{s~D_t}[A_t(s,π_i(s))] | 半交互、零额外 rollout；与 B 共享计算 |
| D2 学生侧 learning progress | 按 provenance 分层归因 eval/TD 进步 | 不依赖 critic 打分教师动作；慢、粗粒度 |
| D3 梯度对齐 | cos(∇L(batch_teacher层), ∇L(batch_student层)) | 诊断性；干扰的直接度量（PCGrad 检测端） |
| D4 动作距离 staleness | E[‖a_src−π_t(s)‖] | 不依赖 critic 正确性；测 far-policy 度而非好坏（ReF-ER 代理） |

判据组合语义（草案）：`t_close(i)` = D1 zero-crossing 持续 K 窗 **且** D4 高
（far-policy 且无 advantage = 纯锚定风险）→ 撤销 + 触发 C。
`t_open` 用 quarantine probe（组件③现成）：源 probe 轨迹先进隔离区，D1 在隔离
数据上首评通过才放行主 buffer——判断期主 buffer 零污染。

---

## 4. 验证设计：离线"考古"判别力 gate（第一步，零训练成本）

**吸取 SIV/SHU/adaptive 的教训：先在离线已知答案的数据上测判别力，再上在线闭环。**

- **材料**（全部现成）：checkpoint 库（crawl/truck/powerlift/basketball ×
  wfix/adaptive × 30k/60k/90k/final）+ 源 bank + 少量 eval rollout 采状态样本；
- **ground truth 标签**（来自已有实验裁决）：
  - crawl: walk/run = **毒**（terrain 翻转 + adaptive 干预证据）；
  - truck: hurdle/walk/run = **好**（bank 级 +227.8，且 hurdle 即时 reward 低）；
  - powerlift: crawl/reach = **无关**，整包 = 好；
  - basketball: 行走类 = **毒**（std9 3/3 负）；
- **预注册式 gate**：候选信号（D1-D4，各时间断面）必须分开
  **{crawl:walk/run 毒} vs {truck:hurdle 好但即时 reward 低}** ——这正是行为信号
  族全军覆没的判别对。分不开的信号不进入任何在线实验；
- **R1 风险直测（P2 探针）**：用 crawl-wfix-30k（已中毒 critic）算 walk/run 的
  D1——若已中毒 critic 仍给毒源正分，则 B 门的鸡生蛋坐实，方案降级为
  A+C+D2/D4（不依赖 critic 的子集）；
- 成本估计：纯离线前向+梯度，单卡数小时。

**只有考古 gate 通过后**，才设计在线闭环 run card（届时按流程报 PI 审批）。

## 5. 与既有结论的一致性检查

- 不与三重否定冲突：D1-D4 全部是非行为信号（不测执行段即时 reward）；
- 不违反"勿做行为信号第四种变体"纪律；
- 不动 FastTD3 基座与 PTF 框架；全部机制挂在 admission lifecycle 既有接口上
  （candidate mass、per-sample 权重、revocation、quarantine）；
- 与 v3 保底路线兼容：A/B/C 即使 D 失败也各自成立（A 是消融、B 是 Q-filter 推广、
  C 是 reset 应用），可作为 lifecycle 的增强件单独发表价值有限但组合有故事。

## 6. 工作日志（供对抗性审查）

- 2026-07-16：PI 指示由 Claude 全权负责本方案，暂不与 ChatGPT 协作，全程记录。
- 起点 = 当日对 crawl 负迁移的机制分析（四道防线分解 + 剂量算术 + "毒害载体是
  critic 参数"论断），已向 PI 口头汇报，本文档 §1 为其正式化。
- 文献检索（WebSearch/WebFetch，检索词与结果均可复查）：Q-filter、ReF-ER、JSRL、
  tandem effect、primacy bias resets、gradient conflict（PCGrad 族）、suboptimal
  demonstrations、replay data valuation/pruning 2024-25、DQfD/R2D3 demo ratio、
  balanced replay off2on、online negative transfer detection（PRIMA 2024）。
  精读：R2D3（ar5iv 全文，demo-ratio 消融）；其余以摘要/综述层为准——
  **审查提示**：引用的具体数字（如 1/256）建议 ChatGPT 复核原文。
- 第二轮检索（同日，PI 要求补最新文献）：2025-26 的 replay curation、warm-start/
  hybrid RL、plasticity/reset 进展、RL 数据归因/influence。关键新发现=
  IIF（NeurIPS 2025 Oral，梯度相似度归因+迭代过滤，D3 的直接先例）、
  WSRL（ICLR 2025，"教师数据用完即弃"，催生 A+ 变体）、ARB（2512.10510，
  免学习 on-policyness 加权）、QoQ（2603.09056，influence 式 demo curation）；
  并确认 off-policy actor-critic 上的有害源数据识别仍是空白。
  **审查提示**：IIF 是 PPO/on-policy，迁移到 TD3 的 gap 是否被我低估，请重点攻击。
- 主要设计判断（欢迎攻击）：
  1. "事前完美判据不可达 → 误判代价有界化"的 reframing 是否放弃太早？
  2. R1（B 门鸡生蛋）是否使 B 层整体不可用？（考古 P2 可证伪）
  3. 低剂量 H1 在 dense-shaped humanoid 上是否成立（R2D3 是 sparse/离散域，
     外推有风险）？
  4. reset 重训的成本与 100k 预算的相容性；
  5. 考古 gate 的标签是否有循环论证风险（标签来自行为层裁决，但信号是非行为的，
     应无循环——请审查）。

## 7. 下一步（等 PI 批准的动作）

1. **P1 考古判别力探针**（零训练、单卡数小时）：D1/D3/D4 × 4 任务 × 已知标签
   → 判别力表；
2. **P2 鸡生蛋直测**：中毒 critic 上的 D1 打分；
3. P1/P2 结果出来后，若有信号过 gate → 写在线闭环 run card（A+B+C+D 组合与
   消融矩阵）报 PI；若全军覆没 → 如实报告，组件①降级为 open problem，转 v3。

---

## 8. 对抗审查裁决与 v2 修订（2026-07-16，当前有效版本）

> ChatGPT 对 v1 的审查裁决为 Major revision（核心批评：B 层 critic 门撞已有负结果、
> 考古标签越权、reset 外推过度）。本节是我对其审查的**反向对抗审查**：逐条核实其
> 事实主张后给出裁决，并指出其修订方案自身的两个问题。融合结果构成 v2。

### 8.1 事实核实（全部完成，带出处）

| ChatGPT 的事实主张 | 核实结果 |
|---|---|
| T^critic 符号已在本项目失败：pole 上真实有用的 walk 全程被判负 | **属实**（[archive/advisor_feedback_analysis_20260702.md](archive/advisor_feedback_analysis_20260702.md) §Step C 离线 logging，2026-07-02）：系统性负偏，根因=学生 actor 本就在最大化学生 critic 的 Q，任何 off-policy 单步动作在成熟 critic 眼里都≈略差；当时已裁决"符号不可用作 transfer/abstain 判据，降级为源间相对排序辅助" |
| 考古标签越权（bank 级被写成 source 级） | **属实且是我的自我矛盾**：我在 adaptive results §3.1 亲手写过"不能声称 walk/run/hurdle 单独是已证好源"，v1 §4 却违反了它 |
| Nikishin reset 是全网络+optimizer 重置，非"critic 后几层" | **属实**（SAC 实验重置整个 agent；"后几层"是其离散域变体，我混淆了 setting） |
| 驻留期"至 81k"不准确 | **属实**：51.2k 起逐步覆盖，81.2k 完全退出 |
| 正式配置下 admitted 时 source 总暴露≈50%、9k 内≈57.6 万条 | **属实**（student_logit 校准是实验配置所致，与 v1 §1.3 的 30k 全量算术相容） |

### 8.2 对 ChatGPT 各项裁决的接受/反驳

**接受（v2 生效）**：
1. **B 层（critic advantage 逐样本门）撤出主机制**——撞 2026-07-02 已有负结果
   （这是 v1 最严重的错误：设计前未检索自己项目的既有证据）。残余价值仅剩：
   跨源 z-score 排序化后作剂量分配辅助（"排序有信息"有既有证据），以及作为
   P2 鸡生蛋诊断的对象。绝对符号/幅度门死刑成立；
2. **考古标签修正为 bank/intervention 级**：crawl-loco=negative、pole-loco=
   positive（同时是 T^critic 已知 false-negative 测试例）、truck-h4=positive、
   basketball-std9=negative、powerlift-std9=compatibility/null；
3. **C 层 reset 暂缓**，替换为 anchor + shadow branch + rollback 路线；reset 降为
   后置 ablation（与 revoke-only、matched-time placebo reset 对照，先证明参数
   污染真实存在）；
4. η 的 5-15% 无文献依据，定位改为"预注册消融的风险预算"；
5. D4 用 exploration-noise 归一化的 Mahalanobis 距离替代裸 L2，且仅作
   staleness/风险特征，不作 utility 符号；
6. "毒害载体是 critic 参数"收窄为："learner 状态（critic/actor/target/optimizer）
   与后续 occupancy 都是候选持久化通道，当前证据未分离"；
7. **A+ lease 化**：从"warmup 结束统一清除"升级为 per-batch lease
   （collection_stage/source_id/admission_epoch 记账、到期自动退出、revoke 即失效）。

**反驳/修正 ChatGPT 的两点（我的反向审查发现）**：

**R-1：ChatGPT 的核心修订方案与我的 D3 都违反 v2 §9 预注册禁令，它未声明。**
[archive/paper_core_contribution_reconstruction_v2.md](archive/paper_core_contribution_reconstruction_v2.md) §9
字面写明："不恢复 DV/SIV、**gradient influence** 或 **K-step learner-value
estimator**"。ChatGPT 提议的 counterfactual delayed-learning test
U_i(t,K)（paired forks + K 次更新 + 下游评估）**正是 K-step learner-value
estimator**（SIV 的端到端变体：SIV=六分支 400/200 updates from cabinet-10k
anchor，信号 B0=−0.051/R0=−0.030/I=+0.033 均未过 0.10 门 → STOP）；
其推荐的 influence/gradient-alignment proxy（含我的 D3）属于被禁的
gradient influence 家族。**这不等于方案错**——"这次不同"有实质理由：
(a) SIV 死于在弱对价任务（cabinet，效应 ±30 级）单 anchor、超短 K=400 的
设置下测微弱信号；新方案应在已知强效应 cell（crawl −100~−200、truck +228）
校准；(b) 端到端总效应替代 B0/R0/I 三路分解，方差更小；(c) 多时间尺度 K +
保守下界替代单点判断。**但按预注册纪律，重启被禁家族必须由 PI 显式解除
禁令**，且解禁条件应含一个**正例校准 gate（新 P0）**：U_i(t,K) 若连 crawl
的已知 −100 级毒害都分辨不出（3 seed 保守下界不显著为负），则该估计器
家族第二次死刑，不得再调参重试。

**R-2：ChatGPT 漏掉了项目里已有的合法单源标签，其考古集可以更强。**
`configs/source_banks/audit/` 下存在 cabinet 与 basketball 的**单源** bank
（sd_run/sd_stand/sd_walk），stability audit P1 已跑过 cabinet 单源训练：
run-only > stand-only 3/3 seed 同向（[archive/stability_deconfounded_audit_p1_cabinet_s123_findings.md](archive/stability_deconfounded_audit_p1_cabinet_s123_findings.md)）。
因此修正后的考古集 = bank/intervention 级标签 + **cabinet 单源序标签
（run≻stand，序标签而非绝对毒/好标签）**——比 ChatGPT 给的清单多一个
真 source-level 判别测试例。

**R-3（成本告知义务）**：ChatGPT 的 lease"到期须新证据才续期"在线实现需要
shadow learner fork（双份 learner 显存+算力+调度），它未标价。v2 首版把续期
机制退化为**纯 TTL 不续期（保守）**，shadow-branch 证据机制列为二期。

### 8.3 v2 融合框架（双方一致部分）

**estimand 正式采纳 ChatGPT 的定义**（这就是组件①要估的量）：

```
U_i(t,K) = E[ J_sf(A^K(Θ_t, D_t ∪ Q_i)) − J_sf(A^K(Θ_t, D_t ∪ Q_student)) ]
```

同一 checkpoint 分叉、等量数据、等更新数、source-free 评估（return + hard
progress + fall 三口径）。机制链：

```
Quarantine probe → 反事实延迟效用检验(离线校准后的 proxy)
→ 保守准入(LCB>δ)/exact abstention → capped-dose lease(η 消融)
→ provenance+TTL active replay → 到期退出/撤销 → anchor+rollback
```

**验证顺序（全部先离线，逐级过门）**：
- **P0 正例校准**（新增，解禁前置条件）：U_i(t,K) 在已知强效应 cell
  （crawl-loco 毒 / truck-h4 好）上的分辨率与功效检验；连已知效应都测不出
  则本方向终止；
- **P1' 修正标签考古**：proxy 族 = D2（U 的低成本近似）、D3'（对 source-free
  held-out objective 的梯度对齐，非任意 batch 对）、D4'（Mahalanobis staleness，
  仅作风险特征）、D1（作为**已知失败 negative baseline** 陪跑）；标签 =
  8.2 修正集 + cabinet 单源序标签；
- **leave-one-task-out**：proxy 只能拟合 crawl/truck 而不能预测持出任务者不上线；
- 全部过门后才写在线闭环 run card（capped dose + TTL lease + exact revoke +
  student-only anchor），报 PI 审批。

### 8.4 需要 PI 裁决的三件事

1. **是否有条件解除 v2 §9 对 K-step learner-value estimator / gradient influence
   的禁令**（条件=P0 正例校准 gate；不解禁则组件①只剩 D4' 风险控制+纯 TTL，
   判据侧实质封存）；
2. **是否批准 P0+P1' 离线实验包**（P1' 纯前向打分为主，GPU 小时级；P0 是
   分支训练，量级更大——每 cell 2×K 次更新×3 seed，预算待细算后单独报）；
3. **贡献叙事确认**：双方一致的定位=
   "stage-conditioned counterfactual learning utility + lease-based provenance
   replay lifecycle"（先隔离估计延迟学习增量，保守下界为正才授予有限期注入权，
   到期重证明，失败即 exact abstain/revoke + 回滚未污染分支）。

### 8.5 v2 工作日志补记

- 2026-07-16：收到 ChatGPT 审查（Major revision）。核实其三项关键事实主张
  （全部属实，见 8.1）；确认 v1 两处真实错误（B 层撞 2026-07-02 已有负结果、
  考古标签越权）系我未检索自己项目既有证据所致——此教训已计入
  [ISSUES_AND_LESSONS](ISSUES_AND_LESSONS.md)（M15）。
- 反向审查发现 R-1~R-3（见 8.2）：其中 R-1（双方方案均触碰 v2 §9 禁令）
  为流程级发现，提交 PI 裁决。
- ChatGPT 新增文献（Pessimism Principle 2505.18447、Hybrid Transfer RL
  qu25a、OPT shin25c）我未独立复核，标注待验。

---

## 9. 第二轮对抗审查裁决（2026-07-16 晚，双方收敛，当前有效版本）

> ChatGPT 对 v2 的再审查承认了 R-1/R-2/R-3 的有效性，但对我的"这次不同"论证
> 提出四点反驳并新发现 commit 混入问题。本节为我的核实与最终裁决。
> **结论：双方在全部可行动项上已收敛，唯一残余分歧是 P0 的认识论定位表述，
> 且不影响行动。**

### 9.1 事实核实（全部属实）

| ChatGPT 主张 | 核实 |
|---|---|
| SIV 已定义并计算总干预效应 T=μ11−μ00，非只有 B0/R0/I 分解 | **属实**（[archive/source_intervention_mechanism_gate_v1.md](archive/source_intervention_mechanism_gate_v1.md) §2.3 定义 SIV_s(L_t;d,h,f,K,U,P)，§14.3 给出四 cell 均值）→ 我"端到端总效应是新设计"的理由**作废** |
| SIV 已查 K=0/100/200/400 grid | **属实**（:341）→ 我"多时间尺度是新设计"的措辞**作废**（正确表述见 9.3 K-2） |
| v2 §9 已 superseded，正确说法是"推翻 v3 停止决定"而非"违反预注册禁令" | **属实**，措辞修正接受（治理决定 ≠ 冻结 run card 协议；实质不变：需 PI 显式授权） |
| commit 8a2d441 混入四个未披露删除 | **属实**。文件已恢复（commit 81c8c9d，带披露；删除来源不明，若系 PI 有意请告知）。教训入 [ISSUES_AND_LESSONS](ISSUES_AND_LESSONS.md) E14 |
| cabinet 单源标签的三个边界（序标签/仅 ranking/seeding 债务） | 接受（与 R-2 原意一致，无分歧） |
| 离线 P0 顺序分支共享冻结 bank，无需双份显存；K=400 量级约 3-6 GPU 小时 | 接受成本修正（但 K 拉长到 lease 尺度则线性增长，见 9.3 K-2，run card 精确标价） |

### 9.2 接受的修正（v2.1 生效）

1. 跨任务原始 return/AUC 绝对值不作"效应强弱"比较，改用相对 scratch 的
   标准化效应量与 t 值；
2. **Treatment matching**：P0 的 treatment（dose、segment h、follow-up f、K、
   stage）必须匹配未来 lease 机制的意图参数，完整 30k exposure 的总效应标签
   不能直接充当任意局部 U_i(t,K) 的 ground truth；
3. D1 彻底退为 diagnostic/negative baseline——P1' held-out 验证通过前，
   不进入任何在线环节（含剂量分配）；
4. P1' 不称"纯前向"，run card 逐信号列算力/显存（D3 需 backward+梯度存储，
   D2 需 learner update）；
5. P0 gate 双侧化：UCB(U_crawl) < −δ **且** LCB(U_truck) > +δ，并加
   duplicate/control 分支的噪声地板；两处措辞不一致已统一；
6. **Phase 拆分消除内部不一致**：Phase-1 = 一次性 admission + capped dose +
   固定 TTL + 到期无条件退出 + exact revoke（无续期、无 rollback 声明）；
   Phase-2 = shadow renewal + anchor/rollback（P0/P1' 过门且预算另批后才设计）；
7. 贡献叙事降级为**候选研究方向/工作假设**："We propose to investigate
   stage-conditioned counterfactual learning utility on top of an already
   validated provenance-consistent source-data lifecycle."——全部过门后才可
   写"introduce"；
8. 顺序化：**P0 通过后才批准 P1'**（proxy 验证需要 U 作 oracle label 的部分
   尤其如此；历史 bank 标签只够 P1' 的粗筛面）。

### 9.3 坚持的两点（最终立场）

**K-1：P0 是双方融合方案的共同必要条件，不是可选校准。**
ChatGPT 把 P0 框成"用可能失配的 ground truth 审判 estimator"，但 lease 框架
（它自己的核心修订）的根基假设恰恰是：**源数据的延迟学习效用可以在 lease
周期尺度上被局部测量**——"到期须新证据续期"的"证据"正是局部 U 或其 proxy。
若 P0 在 treatment-matched 的设置下测不出任何方向信息，死掉的不只是
estimator，是 lease-renewal 框架本身（届时组件①合法退化为纯剂量/TTL 控制 +
open problem）。因此 P0 的失败结论表述精确化为："局部延迟效用在 lease
treatment 下不可测 → lease 式判据封存"，而非模糊的"第二次死刑"。
防调参寻门机制维持：预注册冻结 K grid/dose/双侧 gate/停止规则，失败恢复
v3 停止决定，不得改参重试。

**K-2：K 尺度失配是 SIV null 与新测量之间的实质区别（修正后表述）。**
SIV 的 K grid 全部 ≤400 updates，而真实机制的决策周期是 3000 步窗口/30k
warmup——primary endpoint 比目标机制的时间尺度小一个量级以上。SIV 文档
§2.2 自己写明 source value 依赖 (L_t,d,h,f,K,U,P)：**treatment-matching 论点
双向适用——SIV 在 K≤400 的 null 同样不能外推宣判 lease 尺度的 U 无信息**。
故 P0 不是"换参数追结果"（同一 treatment 下调参数直到显著），而是**在目标
机制定义的新 treatment 点上的首次测量**；接受 ChatGPT 的框定：这构成对 v3
停止决定的显式推翻，须 PI 授权，并在文档中如实称为路线重启。

### 9.4 收敛后的统一建议（提交 PI）

1. **授权一次严格限界的 SIV-v2/P0 最终可行性审计**（显式承认推翻 v3 停止
   决定）：只解禁 counterfactual oracle；D3/gradient-influence 暂不解禁；
   不启动在线 controller；预注册冻结全部参数与双侧 gate + 噪声地板；
   失败即恢复停止决定，不得改参重试；
2. **下一步交付物 = P0 run card**（含：承认 U 是 SIV 总效应 T 的延续、
   lease-matched treatment 定义、K grid（含 lease 尺度点）、3 seeds、
   duplicate control、source-free hard progress 口径、GPU 小时/VRAM/RAM/
   artifact 预算）——run card 经 ChatGPT 复核后报 PI 批准，才执行；
3. P0 过门 → P1' run card；P0 失败 → 组件①判据侧封存，转 Phase-1 纯
   剂量/TTL 机制（其价值独立于判据）+ v3 保底路线；
4. 贡献叙事以"候选研究方向"入档。

### 9.5 工作日志补记（第二轮）

- 核实四项事实主张（9.1 全部属实）；恢复被混入删除的四文件（81c8c9d）；
  新增教训 E14；
- 承认第一轮"这次不同"三理由中两个不成立（端到端/多时间尺度），保留并
  精确化一个（K 尺度失配，treatment-matching 双向适用）；
- 残余分歧仅为 P0 的认识论定位表述（"仪器校准" vs "可能失配的审判"），
  对行动清单无影响——双方行动建议一致。

---

## 10. 第三轮对抗审查：正式收敛（2026-07-16 深夜，最终版本）

> ChatGPT 第三轮接受了 K-1/K-2（并给出比我更精确的 K 尺度换算），新增两个
> 强制条件。我全盘接受，仅有一处事实错位需 PI 直接澄清。**双方联合建议正式
> 定稿；最终授权属于 PI。**

### 10.1 接受的新增条件（全部并入 P0 run card 要求）

1. **失败类型分解**：P0 失败不得统一写成"局部效用不可测"，run card 预先规定
   四类失败模式的裁决语句——
   (a) 统计不可测（CI 宽/duplicate 噪声大）；(b) 效应真≈0（CI 窄，lease
   treatment 太弱或无价值）；(c) 局部 lease 效应与完整 30k 干预方向不一致
   （两种 estimand 分离）；(d) 跨 seed/stage 不稳定（不存在可部署统一判据）。
   四者都阻止 Phase-2，但科学结论不同；
2. **部署四 gate**（决定"算法贡献 vs 昂贵 oracle"的分界）：
   measurement（crawl 负+truck 正+超噪声地板）、latency（证据产出不晚于下次
   续期决策）、cost（probe 成本相对训练预算足够低）、scalability
   （C_renewal≈(S+1)×C_branch(K) 扩展后不退化为并行全训练）；
3. **lease 长度 L 必须在看到 P0 结果之前独立确定**（由未来机制的运行方式
   推导，不得事后把显著的 K 点定义成"正确 lease"）；旧 adaptive 的 3000 步
   窗口不自动等于新 lease，run card 须独立论证 L；
4. **K 尺度换算采纳**（已核实 num_updates=2）：旧 SIV K=400 critic updates
   ≈ 200 在线 vector steps；3000 步窗口=6000 critic updates（15×）、
   30k warmup=60000（150×）——K-2 的尺度失配得到精确量化；
   成本按线性外推 K=6000 约 45-90 GPU 小时、K=18000 约 135-270 GPU 小时，
   run card 须用无结果窥视的 throughput smoke 精算。

### 10.2 一处事实错位（需 PI 直接澄清）

ChatGPT 第三轮写"既然你已确认四个文件是本人有意删除"——**我（Claude）从未
做此确认**（我的原话是"来源不明，非本人删除操作"并向 PI 提问）。若 PI 在与
ChatGPT 的对话中确认了删除意图，请 PI 向我直接确认一次；确认后我将以单独、
明确说明用途的 commit 重新执行删除。**在得到 PI 一手确认前，文件维持恢复
状态（81c8c9d），我不基于二手转述删除 PI 的文件。**

### 10.3 联合建议定稿（双方一致；授权权在 PI）

1. **有条件推翻 v3 停止决定**：只解禁 counterfactual SIV-v2 oracle（明确承认
   U_i(t,K) 是旧 SIV 总效应 T 的 lease-scale 延伸）；D3/gradient influence
   继续封存；不启动在线 controller；P0 失败后不得改 task/K/dose/threshold
   再救，恢复停止决定；
2. **现阶段只授权编写 P0 run card**（不授权执行）：内容清单=intended lease
   的来源与固定长度、stage/dose/horizon/follow-up/K grid、双侧 gate+噪声
   地板、三口径 outcome（source-free return/hard progress/fall）、四类失败
   模式裁决语句、四个部署 gate、GPU-hours/VRAM/RAM/artifact 预算、执行前
   冻结 SHA；run card 由 ChatGPT 交叉复核后报 PI 批准；
3. P0 过门 → 讨论 P1'；P0 失败 → lease-based utility judgment 封存，
   Phase-1（固定剂量/TTL，不依赖 U）独立推进 + v3 保底；
4. 贡献叙事维持候选表述（"We propose to investigate..."）。

### 10.4 工作日志补记（第三轮）

- 核实 num_updates=2（K 换算成立）；接受失败类型分解与部署四 gate；
- 指出 ChatGPT 的角色措辞（"授权/批准"）应读作联合建议，最终裁决属 PI；
- 文件删除事宜维持"待 PI 一手确认"状态。
