# SC-MCG 执行报告与求诊 v2（发 ChatGPT-5.5-Pro，2026-06-12）

你上次给出的 SC-MCG-PTF 方案（significance-calibrated gate + initiation/competence + budgeted execution，任务排序 window→push→package→door）已完整落地并跑完一个战役周期。本文是自上次交接（MCG v1 三任务 pilot）以来的全部执行情况：你的方案修好了 window 负迁移（核心承诺兑现），package 上我们沿你的"补源库+initiation"路线打了四轮，每轮都把根因下钻了一层，现在卡在一个清晰的学习端瓶颈上。请你评判我们的诊断与候选下一步。

## 一、你的 v2 方案落地结果（window/push）

实现与你的设计的对应关系：paired head delta（每 head 内 paired diff 再 min_h）✓；null 校准 margin（教师动作×打乱状态的 paired delta 取 q95，EMA 跨 batch 平滑，行为端复用蒸馏端 margin）✓；confidence c=σ((Δ−m_g)/τ) ✓，但 v1.2 改为 **gate.float()×conf**（硬门控×软权重）——纯软权重在 sig≈0 处 σ≈0.5 漏蒸馏，conf_mean 只有 0.16-0.31。

| window (1 seed, 100k) | v1 (sign gate) | v1.1 (null+conf) | v1.2 (gate×conf) | scratch |
|---|---|---|---|---|
| 末段 eval | 336 | 365 | ~407 | 489 |
| 15-30k 段 | <scratch | — | **121 vs 66（反超）** | — |

负迁移从 −153 压到 ~−80（仍单 seed，方差大，可能已是噪声级）；gate_rate 末段 4.5%≈q95 理论假阳性率，机制按设计工作。**你的核心论断被验证：显著性校准是 critic-gated transfer 负迁移免疫的必要条件**。

push sanity：不变差，但 v1 里稳定的 arms gate 偏好消失——显著性门槛把 ~0.2 量级的弱真信号一并压掉了。**门槛的代价是弱信号灵敏度**，这是 calibration-sensitivity trade-off 的实证，我们准备写进论文。

door 维持暂缓（P3"转把手+推门"探索瓶颈，教师库无对应技能，与你判断一致）。

## 二、package 战役：四轮实验，根因逐层下钻

约束回顾：package = 走到箱边→搬起→运到 2-4m 外的 dest→放下。reward 主项 −3·dist(box,dest)，**机器人走向箱子本身零即时回报**。HB/FastTD3 论文里此任务接近 0 分（hardest 级）。所有实验 1 seed、128 envs、100k steps、paired scratch 对照（参数 diff 验证过）。

### R1：MCG v1 + 随机 warmup（库：approach+contact+push_s1+reach）
双 0%，hand_min 1.4（站桩）。诊断：**状态覆盖≠回报事件**——warmup 教师执行注入了"走近/贴箱"状态，但 −3·dist(box,dest) 只在箱子动时才有梯度，库里没有教师会移动贴地箱（push_s1 的技能是推 ~1m 高桌面箱，蹲姿贴地箱分布外）。

### 接口链建设（你的"补源库"路线，三环各两版迭代）
按"每环从上一环终态分布训练"构造：approach（走近）→ contact（贴箱）→ nearcarry（贴箱+推到近距 dest）。验收用 coverage matrix（near/contact/moved/lifted/to_dest/fall 列+零动作对照行，near-reset=接口分布）：

| 源（验收版） | near | contact | to_dest | fall |
|---|---|---|---|---|
| zero-action | 0% | 0% | 0% | 0% |
| approach v2 | 31% | 0% | 25%* | 25% |
| contact v2 | 100% | 100% | 6% | 6% |
| nearcarry v2 | 100% | 88% | **81%** | 6% |

（*approach 的 to_dest 是撞运气：身体撞箱碰巧滚近。）方法论发现：三环的 aux reward **全部需要两版**，缺陷模式同款（满分区位置错/约束与物理冲突/梯度太平），但审计矩阵每次都能一轮定位到具体 reward 项。

### R2：bank v3（三环+initiation 区间）+ initiation-aware 随机 warmup
warmup 抽教师只在规则版 I_o(s) 内抽（robot-box 距离区间：approach>0.9m，contact/nearcarry<0.9m）。**仍双 0%**。wandb exec_share 揭示新根因：**contact/nearcarry 合计只拿到 1.4% 执行份额**（approach 49%）——随机 warmup 的"50% 步级学生混合+25 步锁存"让行走不断被站桩学生段打断，机器人很少真正进入近区；探针证实纯 approach 连续执行其实 75% episode 能把 root 带进 0.9m。**随机抽样无法展开技能链**。

### Oracle 链探针（决定性实验）
规则调度（远区 approach/近区 nearcarry、离区即切，无随机无锁存）：success 0%，但失败模式从"站桩"变为"**推到半路**"——5/16 episode 箱子被向 dest 推进 1-2m（dest_min 3.5→0.87-1.2）。残余问题：(a) nearcarry v2 只见过 0.25-0.5m 近距 dest，真实 dest 2-4m 是分布外输入；(b) 箱子推远后 root-box>0.9 切回 approach，approach 不推箱。

### R3：chain warmup（单变量 vs R2）
把 oracle 链装进 warmup：episode 级 demo 抽签（done 时按 warmup_exec_prob）+ demo env 每步选 priority 最高的 eligible 教师 + 离区即切。这同时是 PTF initiation+termination 的规则版（learned 版留作 ablation）。

结果：**数据端修复证实，学习端瓶颈暴露**。warmup env_rewards −9.26 vs R2 的 −10.74（每步 +1.5 ≈ 箱子平均被推近 0.5m 的回报差进了 buffer），nearcarry 份额 0.7%→8%。但 eval 曲线与 R2 完全重合，终评仍 0%，hand_min 1.12。

### 学习端瓶颈的机制分析
buffer 里有了回报事件，为什么学生不学？(1) demo transition 只占 ~8%，"走近箱子"零即时回报，价值要从"推箱段"经 100+ 步 TD bootstrap 传播回初始状态的"迈步动作"；(2) 中间状态（行走中）只有 demo 轨迹覆盖，密度低，CDQ+distributional critic 对低密度区悲观；(3) actor 的 DPG 梯度在 demo 状态上学到的局部动作，从初始状态自己走不出第一步就复合误差偏离（演示学习的经典 distribution shift）。

### 正在跑的两个验证（截至发稿）
- **300k 延长对照**（chain vs paired scratch，检验"只是慢"）：跑到 ~150k，**两条全平**（−6000~−7500 徘徊，无任何起飞迹象）。"只是慢"初步偏向否定。
- **nearcarry v3 长程化**（dest 0.25-2.0m + 双尺度 progress + 1000 步 episode）：已完成，**混合信号**——oracle 链上**首次出现完整 success（1/16=6%，best dest_min 0.10）**，但近距接口分布上 to_dest 从 81% 退化到 0%（长程牵引 vs 近距收尾精度的 trade-off；far 项在近距 spawn 时近满分，稀释了近端梯度）。v2 与 v3 互补：v2 近距精，v3 长程拉。

## 三、机制发现汇总（候选论文素材）

1. 显著性校准（null 分布 q95）是 critic-gated transfer 负迁移免疫的必要条件；sign gate 在 Δ 噪声同量级时 20-30% 假阳性（window −153 实证）。
2. 校准的代价是弱信号灵敏度（push arms 偏好消失）——calibration-sensitivity trade-off。
3. 状态覆盖≠回报事件：教师执行注入的状态必须携带 reward 增量才对 off-policy 学习有用。
4. 随机 warmup 无法展开技能链；initiation-as-scheduler（chain warmup）可以——执行份额 ×10，回报事件入 buffer。
5. 接口源的 aux reward 设计有可复现的缺陷模式与一轮定位的审计方法。
6. **开放问题：demo 数据在 buffer 里但 TD3 不消费**——长程 credit assignment + 演示 distribution shift。

## 四、候选下一步（请你评判/排序/补充）

**A. demo-BC（DDPGfD 风格，攻学习端）**：buffer 标记教师执行的 transition（及教师 id），actor update 对 demo 样本加 BC 项（向 buffer 里实际执行的教师动作回归），带 Q-filter（仅当 Q(s,a_teacher)>Q(s,a_θ) 时生效）+权重衰减。与 MCG gate 兼容：BC 管 warmup 期 bootstrap，gate 管中后期模块化精修。注意：你曾否决"无差别保底蒸馏"（window 教训），demo-BC 语义不同——只在教师真正执行过且 critic 不反对的状态上模仿。我们认为这是对"actor 不消费 demo"最直接的攻击。

**B. 双 carry 库 + 子任务级 initiation（攻 demo 质量）**：bank v5 = approach + contact + nearcarry_v2 + nearcarry_v3，initiation 规则加 box_dest_dist 类型：dest 远（>0.6m）→v3 长程推，dest 近→v2 精确收尾。先用 oracle 链探针验证（预期 success>6%）。这是"option 分工按子任务进度划分"的进一步细化，规则版竞争力图谱的自然延伸。

**C. n-step return（学习端算法侧）**：当前 num_steps=1；n-step（3-5）能把 TD 传播加速 n 倍，是 FastTD3 论文在难任务上的标准配置。修改在我们自己的 train_ptf.py 副本内（不动官方代码),与 A 正交可叠加。

**D. 论文策略调整**：接受 package=hard case 的定位（HB 上专门算法也接近 0），把它写成 negative result + 诊断方法论（四轮下钻链本身是贡献）；资源转 window 巩固（2-seed paired 确认负迁移消除）+ door P3 专项源。headline 从"package 成功"降级为"安全迁移机制 + 诊断框架"。

我们的倾向：A+C 同跑（学习端双管），B 的 oracle 探针先行（半小时），D 作为论文保底并行推进。但 demo-BC 是机制级新增（PTF 框架内的合法延伸：相当于把 PTF 的蒸馏项从"on-policy 状态分布"扩展到"demo 状态分布"），希望你确认它不会重蹈 window 覆辙，以及 Q-filter/权重/衰减的具体设计。

## 五、约束提醒（不变）

- 必须基于 PTF 框架 + FastTD3 算法 + HumanoidBench，创新长在 PTF 内；FastTD3 官方代码不可改（train_ptf.py 是我们的外挂副本，可改）；目标 ICML 水平。
- 所有结论目前 1 seed，方差 >±100；正式声明前会跑 paired 2-seed 起步。
- 可用算力：8×V100 32G，单跑 100k≈2h。
