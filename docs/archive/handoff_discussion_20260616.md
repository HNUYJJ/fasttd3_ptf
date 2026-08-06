# RBO-PTF 研究讨论 handoff（2026-06-16，供 ChatGPT 分身参与）

本文件汇总 2026-06-16 与 PI 讨论中浮现的研究问题与假说，便于 ChatGPT 分身一起参与。
原则：**区分"有数据支撑的事实 / 有机制解释的假说 / 需多 seed 验证的猜想"**，不粉饰。

---

## 0. 当前进展速览

- **核心论点 terrain seed1 成立**（docs/terrain_core_result_v1.md）：stair/slide/pole/crawl ×
  {safe(reward-weighted 源选择) / rand(uniform) / scr(scratch)}，共同窗口 AUC（到 95k）。
  **safe>rand 4/4 一致（平均 +88.1，符号无翻转）**；uniform rand ROI≈0(−2.4) 还时常负迁移；
  safe>scr 3/4（crawl 反例：整体 loco→crawl 不划算 scratch 最强，但 safe 仍>rand）。
- **terrain 3-seed 加固完成**（详见 A.3）：**safe>rand 10/12（mean +66.5，paired t≈2.58 显著）**，
  stair/slide/pole 上 9/9 完美一致；**唯一翻转全在 crawl（safe 反而三者最差）= abstain（Open Q5）黄金动机**。
- **现有源库 = stand/walk/run/reach**（4 个 official source），全是平地移动+伸手，
  **无抬腿/跨越/攀爬 motor primitive**。

---

## A. 本阶段工作汇总与核心结果（06-14 ChatGPT v3 之后 → 06-16，供通览）

### A.1 采纳 v3 两项消融（已实现，待跑）
- **wfix 解耦**：reviewer 会质疑 safe(weighted源+长horizon) vs rand(uniform+短horizon) 纠缠了
  "源选择"与"执行时长"。新增 wfix bank(weighted源 + horizon=25 同 rand)：**wfix−rand=纯源选择
  增益；safe−wfix=纯执行时长增益**(commit b183f40)。
- **negctrl 边界**：选 door/spoon(Effect Map+Day1 audit 均判无对价，loco adapter 够不到门把手/
  勺子 bottleneck)，跑 {safe,scr} 验证 safe≈scr，把"无对价"从 teacher-level 升级为训练级证据。

### A.2 运维事故 + 方法论转向（PI 纠正，已固化为纪律）
- **OOM 事故**：无人值守全链 + 并行 breadth(第5/6进程)触发节点 RAM OOM，杀掉正推进的
  terrain safe×4。根因=节点瓶颈是 system RAM 非 GPU，稳定上限~4 训练进程。
- **PI 纠正**：稳扎稳打、一项一项做扎实，别为省时把一堆对照盲跑。重构为逐项串行 4-slot 队列
  (核心论点→解耦→边界→广度)，每项做完汇报再进下一项。

### A.3 terrain 核心论点 3-seed 结果（第①项已完成，scripts/analyze_terrain.py）

设置：stair/slide/pole/crawl × {safe/rand/scr} × seed 1/2/3，128 env×100k，共同窗口 AUC(到 95k)。
三方超参一字不差(只差 warmup 源选择方式)，seed1/2/3 完全可比。

| task | scr | rand | safe | safe−rand |
|------|-----|------|------|-----------|
| stair | 252.5±37 | 169.1±41 | 279.2±20 | +110±54 |
| slide | 271.1±46 | 450.2±20 | 504.7±14 | +54±31 |
| pole | 603.3±48 | 573.2±13 | 717.9±25 | +145±29 |
| crawl | 812.0±25 | 699.6±32 | 656.3±35 | **−43±65** |

判读(12 个 task×seed 组合)：
- **safe>rand 10/12**(mean +66.5±85.5，paired t≈2.58 显著) ← 核心卖点(源选择价值)
- safe>scr 8/12(mean +54.8)；rand>scr 仅 4/12(mean −11.7) = uniform 无净价值

**最重要发现(诚实)：2 个翻转全在 crawl**——
- stair/slide/pole(loco 有对价) 上 **safe>rand = 9/9 完美一致**；
- crawl 上 **safe 反而三者最差(656 < rand 700 < scr 812)**，safe<rand(mean −43)。
- 解读：源(loco)对 crawl 系统性负迁移时，**reward-weighted(safe)比 uniform(rand)更糟**——safe
  更"自信"地集中抽系统性有害的 walk/run(vs-zero 高 weight)，rand 均匀分散(还抽到站桩 stand)反而
  少受伤。**crawl 从"反例"升级为 abstain(Open Q5)的黄金动机**：无合适源时越聪明选源越糟。
- ⇒ **建议分层呈现**：主结果"loco 有对价的地形任务(stair/slide/pole) safe>rand>≈scr 一致显著"；
  crawl 作 negative-transfer 边界直接 motivate abstain。**勿把 crawl 负值混进总平均**(会拉低 mean/
  放大 std：safe−scr mean+55 但 std150 几乎全是 crawl 拉的)。

---

## Open Q1：源库从 loco 扩到 skill-diverse（hurdle→stair 抬腿）

**动机**：stair 失败疑似因源库缺"抬腿越障"primitive。hurdle(跨栏)策略含抬腿动作，
若加入 stair 源能否教会抬腿登台阶？若成立，把贡献①从"loco 内部源选择"升级为
"**跨技能库源选择**"（论文最亮正例）。

**zero-shot 探针结果（scripts/probe_hurdle_to_stair.py，五 policy 在 stair zero-shot）**：

| source | return | fall% | ep_len | move |
|--------|--------|-------|--------|------|
| hurdle | 22.3 | 100% | 59.6 | 0.836 |
| walk | 30.7 | 100% | 61.5 | 0.737 |
| run | 26.3 | 100% | 55.8 | 0.772 |
| stand | 160.9 | 12% | 955.8 | 0.179 |
| stair_scr | 516.1 | 19% | 921.6 | 0.677 |

**诚实结论**：zero-shot 层面 hurdle **没有**救场——和 walk/run 一样 ~60 步 100% 摔。
"整条 hurdle 策略搬到 stair 会抬腿登台阶"的乐观版被证伪。机制：hurdle 抬腿被"平地见栏"
触发，stair 无此信号+几何不同 → 按冲刺-跨栏节奏失稳。

**但不能停在"想法1 失败"**，三个细节改变判断：
1. **zero-shot ≠ 我们的 bootstrap**：safe_bootstrap 只在 warmup 前 ~25 步 safe horizon
   执行源、注入 reward-bearing 片段，不是整条 rollout 到摔。hurdle 前 25 步抬腿仍可能有价值。
   **zero-shot 对 dynamic motor 源（hurdle/walk/run）系统性低估 bootstrap 价值**，对
   stabilization 源（stand）则预测准——这是 zero-shot 探针预测力的**源类型不对称**
   （补强 Transfer Map v2"zero-shot 不预测 bootstrap ROI"的发现）。
2. **stair 卡点需重新核实**：stair_scr 不是"一直摔上不去"，而是 fall 19%/ep_len 921/
   move 0.677/return 516——学会站稳往前挪。视频"上不去台阶"可能是"move 在台阶前水平蹭
   也能拿到，但登高没学会"。**先弄清 stair 卡在"登高失败"还是"上得慢"**，决定 hurdle 是否对症。
3. hurdle move=0.836 五者最高=它最想往前冲，符合跨栏冲刺特性，policy 本身不废。

**待办**：(a) 查 HB stair reward 结构是否含显式登高分量；(b) 想法1 正确检验在**训练层**
（stair × {safe+hurdle 源 vs safe(loco) vs scr}），等 seed 加固腾卡再做。

---

## Open Q2：行为模式差异 — fall-recovery & risky-skill（本轮新发现，重点）

PI 观察（部分有硬数据支撑，部分需验证）：
1. **rand 的 stair 学会了上台阶，但摔倒后站不起来**；safe/scr 学不会上台阶，但**摔倒能站起来**；
   **safe 摔倒后站起来更快更稳**。
2. 跨任务长期观察：**我们的方法(PTF/safe)训出的 h1 摔倒能自己站起来，scr 训出的起不来**。

**硬数据侧证（seed3 stair eval 曲线）**：
- eval_avg_return: safe~670 > scr~450 > rand~220（与 seed1 同序）。
- **eval_avg_length: safe~900 ≈ scr~850 >> rand~300-470**。episode 长度=存活时长，
  **rand ep_len 显著低 = 容易摔且起不来（episode 早终止）；safe/scr 高 = 不摔或摔了能恢复**。
  ⇒ "rand 脆弱 / safe-scr 鲁棒"在 ep_len 上**有量化支撑，不是纯偶然**。

**机制假说（待多 seed 验证）**：warmup 源选择不只影响"学多快"，还塑造"学成什么行为"——
- **rand**（uniform 含 walk/run，horizon 短=25，注入杂乱激进先验）→ 早期激进探索**偶然撞对
  上台阶的 risky 动作序列**，但没锚定稳定恢复 → 脆弱（摔了起不来，net return 最低）。
- **safe**（reward-weighted 主抽 walk/run，horizon 长=50，注入稳定 reward-bearing 先验）→
  学到**保守鲁棒**策略（优先平衡+可恢复，ep_len/return 最高），代价是"不敢"做 risky 上台阶。
- 这是 **exploration diversity（rand 发现难技能）vs reward-anchored stability（safe 鲁棒）的 tradeoff**。

**对核心论点的影响（张力处理）**：
- "rand 学会上台阶"**不威胁** safe>rand：rand 的上台阶没转化成 return（摔了起不来，net 最低 220）。
  AUC/return 上 safe>rand 稳固。
- 但它暴露**reward-weighted 可能过保守，牺牲探索难技能的机会**（呼应 crawl 反例）。
  属方法局限/未来方向，值得讨论。

**潜在论文价值**：把核心论点从"AUC 更高"深化到"**行为质量更好（鲁棒性/fall-recovery）**"——
RBO bootstrap 注入的 loco 站立/平衡先验 → 更鲁棒、可恢复的策略。fall-recovery 是 humanoid
控制受关注的能力，若系统成立可作独立卖点。

**诚实风险**：n=1 视频帧 + 单 seed 曲线，强烈可能含偶然。**必须多 seed + 定量探针验证**。

**验证方案（cheap，不训练）— fall-recovery 量化探针**：
- load safe/rand/scr 的 final policy，rollout 记录 per-step upright/standing 时序。
- 检测事件：upright/root-height 掉到阈值下（摔）后是否在 N 步内回升到站立阈值（恢复）。
- 指标：fall 次数 / recovery 率 / recovery 用时 / 净站立比例，跨 seed + 跨任务。
- 口径需先认可（用 HB info 的 upright/standing 还是 sim root height？阈值？恢复窗口 N？）。

---

## Open Q3：源库迭代扩张（target 学好→加入源库→提升他任务，PI 想法2-A）

本质 = lifelong / source-library expansion。**Open Q1 的 hurdle→stair 就是它的一个实例**。
- 价值：自然源库增长 + Effect-Map 指导分配。
- 风险：撞车（policy-reuse libraries / progressive nets / CLEAR / lifelong RL 成熟领域），
  差异化必须锁在 RBO 机制（reward-bearing bootstrap + skill-matched 选择）而非"迭代"框架；
  多轮闭环组合爆炸难做扎实。
- **建议定位**：作 Open Q1 的自然延伸，做**一轮**演示（hurdle→stair）即可，多轮留 future work。

---

## Open Q4：self-bootstrap 反衬 ablation（A 教 A，PI 想法2-B"左脚踩右脚")

**作为研究主张不成立**：reward-bearing bootstrap 增益来自"教师携带当前策略未学到的
reward-bearing transition"；若教师=A 已收敛策略，只能加速收敛（warm start），**不能突破上界**
（自举无外部信息注入，期望性能≤教师）。对应 self-distillation/SIL：能加速不能超越。
唯一窄缝（次优→继续探索跳出局部最优）增益来自算力/探索，非"迭代魔法"，易被 reviewer 打掉。

**但有对照价值**：改造为 ablation `self-bootstrap(A 教 A) vs cross-task skill-matched(hurdle 教 stair)`。
若后者显著>前者，证明增益来自"跨任务携带新技能/新 reward-bearing transition"而非
"bootstrap 这个动作本身"——补强 Open Q1 与核心论点。

---

## Open Q5：warmup bootstrap 的自适应 abstain（PI 洞察，crawl negative-transfer，可能是方法增量）

**PI 观察**：crawl 上 safe/rand 的 return 都不如 scr（seed1: scr 832.8 > safe 688.6 > rand 658.4；
3-seed 坐实: scr 812 > rand 700 > **safe 656(三者最差)**；safe<rand mean−43——
负迁移任务上 reward-weighted 比 uniform 更危险）。符合直觉——源库只有 loco(stand/walk/run/reach)，
都倾向站立移动，crawl 需匍匐爬行 → 站立平衡是负迁移。

**PI 质疑（精确命中机制缺陷）**：safe 机制为何不能避免从不合适教师学坏经验?所有教师都不合适时，
warmup 还硬抽教师执行、注入 replay?能否像 option 的 null_option 那样，当所有教师都不好时，
warmup 阶段回归学生自行 RL 探索(=scr)?

**机制核实（读 mcg.py step() safe_bootstrap 分支 + crawl bank）**：
- crawl bank weight(vs-zero reward-bearing): stand=0.0, **walk=13.9, run=13.7(都正且大)**。
- safe_bootstrap warmup: 每源 expired 时以固定 `warmup_exec_prob=0.5` 决定 teacher/student，
  用教师则按 softmax(weight/tau) 抽源。**即已有 50% 盲目弃权(student)，但三个缺陷**:
  1. **gating 信号是 vs-zero**(源执行 vs 完全不动 zero-action)，**不是 vs-student**。crawl 上
     walk/run 让机器人站起来移动，比躺平不动(zero) reward 高 → vs-zero 正(13.9)，softmax 几乎
     必抽 walk/run；但远不如学生学爬行 → **vs-zero 系统性误判"源整体有害"的任务**。
  2. **弃权是固定 50% 盲目比例**，不随源质量自适应；源都烂时不会把 exec_prob 降到 0。
  3. **null_option: true 只在 warmup 后的 critic gating 阶段生效(best is not None 分支)**，warmup
     safe_bootstrap 抽源的 softmax **不含 null 竞争项**。
- ⇒ PI 批评成立：当前**无 warmup 阶段的自适应 negative-transfer safety**。

**张力(为何不能简单换静态信号)**：Transfer Map v2 试过 vs-student/opportunity 信号，但它在
stabilization 任务(hurdle/cabinet)上**全负无法区分源**，才退回 vs-zero。**没有单一静态信号
同时解决 stabilization(需 vs-zero) 和 crawl(需 vs-student)**。

**建议方向 = 动态/在线 abstain**(不是换静态 weight)：warmup 阶段在线监控"源执行 env 的实际
return/TD 价值"vs"学生探索 env"，源持续更差则自适应降其 exec_prob→0(弃权)。等价于把 critic
gating 的 negative-transfer 检测**提前到 warmup**。crawl=完美 motivating + 验证 case。

**诚实预期**：abstain 卖点是 **no-harm robustness**(把 crawl safe 从略低于 scr 拉回≈scr)，
**不是性能提升**(源对 crawl 无正价值，最好也就回归 scratch 水平)。这补全三贡献的 safety 维度:
①Effect-Map(诊断 go/no-go) ②RBO method(reward-weighted 选好源 + **abstain 无好源时回归学生**)
③Broad eval。与"扩大源库/目标范围"互补:宽源库→更可能有合适源;abstain→没有时 no-harm。

---

## 给 ChatGPT 的具体问题清单

1. **fall-recovery（Open Q2）是否值得提升为论文的一个机制 insight / 独立卖点**？把核心论点从
   "AUC 更高"深化到"行为更鲁棒可恢复"。验证探针口径建议？
2. **rand 学会上台阶但脆弱 vs safe 鲁棒但保守的 tradeoff** 怎么在论文里诚实处理——是 safe 的
   局限（过保守），还是正好佐证 reward-anchored stability 的价值？两者如何兼顾？
3. **Open Q1 源库扩展（skill-diverse source）**值不值得作为论文亮点正例？zero-shot 负结果下，
   是否直接上训练层验证（hurdle 源 bootstrap），还是先做更细的 prefix-horizon 探针？
4. **zero-shot 探针的源类型不对称**（stabilization vs dynamic motor）是否要写进 Transfer Map
   方法论的边界讨论？
5. self-bootstrap 反衬 ablation（Open Q4）是否纳入主表 baseline？
6. **seed 加固已完成**。下一步逐项串行(节点~4 进程)该先做哪个：wfix 解耦 / negctrl 边界 /
   **abstain 实现(Open Q5)** / fall-recovery 探针(Open Q2) / hurdle 训练层验证(Open Q1)？
7. **crawl 3-seed 坐实 safe<rand(不只<scr)**：reward-weighted 在负迁移任务上比 uniform 更危险。
   **abstain(Open Q5)是否从"补全 safety"提升为方法核心增量**，优先级排到 wfix/negctrl 之前？
8. 核心论点**分层呈现**(stair/slide/pole 一致显著 + crawl 作 abstain 动机)是否站得住，
   还是 reviewer 会因 crawl 质疑方法鲁棒性？
