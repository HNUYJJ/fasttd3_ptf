# Dual-Channel Transfer Evidence Matrix v1

日期：2026-07-12

状态：**现有证据整合完成；静态RBO为主方法，双通道为机制解释与可选扩展**

上位裁决：SHU v1 `STOP_CLOSED_LOOP`

## 1. 核心问题

当前结果不能支持“用一个transferability标量决定source执行与replay注入”。更符合代码与实验的
因果对象是两个不同通道：

1. **Execution / occupancy channel**：source在目标环境中控制动作，改变即时reward、termination与
   后续访问状态；该影响在source撤出后原则上停止继续产生。
2. **Replay / update channel**：source-conditioned transition驻留buffer，被actor/critic反复采样，
   即使source已撤出仍持续改变learner；该影响具有记忆和放大效应。

因此“少执行坏教师”不等于“消除坏教师影响”，而“局部handoff reward为正”也不等于“这些数据
经过off-policy update后有正学习价值”。

## 2. 因果形式化

令冻结stage learner为`L_t`，既有student replay为`D_0`。对同一组anchor，定义：

- `Z^S/Z^i`：student/source `i`产生的prefix transition；
- `F^S/F^i`：从student/source prefix终点继续由student产生的follow-up transition；
- `U(L,D,q)`：FastTD3在sampling distribution `q`下用数据`D`进行固定预算更新；
- `Y(L)`：source-free target hard progress或有限预算性能。

对应旧formal 2×2：

| cell | occupancy/reachability data | prefix replay data | learner candidate data |
|---|---|---|---|
| `00` | student | student | `D0 + Z^S + F^S` |
| `10` | source | student | `D0 + Z^S + F^i` |
| `01` | student | source | `D0 + Z^i + F^S` |
| `11` | source | source | `D0 + Z^i + F^i` |

定义：

- occupancy transport effect：`OTE = Y_10 - Y_00`；
- update transport effect：`UTE = Y_01 - Y_00`；
- interaction：`INT = Y_11 - Y_10 - Y_01 + Y_00`；
- total transfer：`TOT = Y_11 - Y_00 = OTE + UTE + INT`。

这不是要重新训练`DV/SIV`预测器，而是给论文机制与消融建立正确的因果语言。

## 3. 现有证据矩阵

| 证据 | 结果 | 对双通道的含义 | claim状态 |
|---|---|---|---|
| `bootstrap_only ≈ full`，`no_bootstrap`弱/负 | 当前性能主要来自warmup source行为和其reward-bearing data | bootstrap是主通道，MCG/蒸馏不是已证独立主因 | 支持 |
| crawl onlineb | student执行share升至76%，3seed AUC仍仅`634.8±77.5` | 只控制execution无法清除buffer中的持久影响 | 支持persistent replay exposure |
| crawl OBRW vs onlineb | `729.5±27.6` vs `634.8±77.5`；配对`+94.7`，3/3正 | 对称replay attenuation恢复均值并降低方差 | 支持replay控制必要性 |
| slide OBRW vs onlineb/wfix | `614.8±16` vs `551.3±5 / 522.7±20` | execution adaptation与replay控制可累积为正收益 | 支持正迁移通道 |
| pole OBRW vs onlineb/wfix | `755.0±15.5`，与`761.3/767.4`打平 | replay控制没有稳定误伤有效source配置 | 支持non-degradation，非提升 |
| actor-only/critic-only/split | crawl `676.2/629.6/715.9`，均低于both `767.5` | actor/critic不同source-state分布造成AC mismatch | 支持对称采样一致性 |
| door OBRW vs scratch | `295.8±8.5` vs `295.0±5.4`；student share 86–88% | 无对价source的影响软坍缩且近零代价 | 支持soft abstention |
| spoon OBRW vs scratch | `354.2±1.4` vs `315.4±13.9` | 同一机制保留弱但真实的source对价 | 支持自动利用 |
| stair | h25 OBRW `184.2±23.9`，safe h50 `279.2±20.3` | source identity以外还有temporal-extent维度 | 已知limitation |
| global horizon-arm `mh` | slide/crawl显著恶化，arm方差上升 | 增加所有horizon arm有探索税、毒害扩大和统计代价 | 不支持免费扩arm |
| cabinet formal 2×2 | `OTE=-0.0511, UTE=-0.0304, INT=+0.0331, TOT=-0.0484` | 两通道及交互均未形成学习收益 | 支持通道可分、非通用正值 |
| cabinet SHU v1 | direct/handoff为正却错误accept；null scale塌floor | behavior/handoff utility不能代理update utility | 否证单分数admission |
| run vs stand/cabinet | 相同eval长度，run 30k/100k hard progress更高 | source身份与动态覆盖重要，不只是站立稳定 | 支持source identity matters |
| run24 dose control | 低剂量run对scratch为3/3正；run24−WFix混合 | 高价值source剂量有用；主动多源毒性未确认 | 支持机会成本候选，不能写毒性定论 |
| terrain WFix−rand | 12个task×seed中11个为正；mean `+77.9`, `t=3.08` | source bank强分化时静态加权显著节省teacher budget | 支持RBO主方法 |
| breadth maze/truck/cabinet | RBO/OBRW与uniform总体打平，但常高于scratch | source bank弱分化时注入数据是主收益，精细加权边际小 | 支持“分化度条件” |
| hurdle扩源 truck/maze | truck `+229.9,t=3.47`；maze `+0.3,t=0.16` | source互补性还需target learning headroom才能转化 | 支持扩源必要非充分规律 |
| powerlift batch2 | RBO−scratch `+77.6,t=14.78`，3/3正；所有短probe均<1 | 短期执行信号可漏掉长期可利用的数据价值 | 否证低probe=无迁移 |
| basketball batch2 | RBO−scratch `−101.5`、OBRW−scratch `−74.0`，均3/3负 | static与online双通道控制均未提供exact fallback | 硬负迁移边界 |
| window batch2 | RBO−scratch `−31.9,t=-0.99`，高方差 | pilot单seed正例不稳定 | 不作正例 |
| balance_hard batch2 | 首次2个cell OOM后原配置恢复；3seed WFix−scr `+6.1,t=0.75`，WFix−rand `−0.6,t=-0.09` | 全零T⁰权重无选源区分力，softmax退化uniform | 支持null regime，不支持exact fallback |
| task-progress/stability audit | hurdle、cabinet、maze有任务进展；powerlift无hard-skill证据；basketball负迁移 | return、viability与目标skill是不同结果层 | 支持分层claim，禁止以return代替完成度 |

## 4. 四类失败指标的统一解释

| 指标 | 能测什么 | 已观察失败 | 不能承担的角色 |
|---|---|---|---|
| static `T^0` | target环境中的source先验表现 | crawl false positive、source排序与hard progress错位 | calibrated transfer ROI |
| `T^online` reward EMA | 当前segment的reward-bearing执行质量 | pole/crawl读数接近但数据后果不同；stair对horizon盲 | replay data utility |
| critic one-step advantage | student critic下的相对动作排序 | pole上所有source系统性负偏 | transfer/abstain符号判据 |
| SHU direct/handoff | 短期行为与交权状态后果 | cabinet downstream-negative被accept | update transport admission |

统一结论不是“所有信号无用”，而是它们各自只测部分通道。论文不能再把任一信号命名为完整
transferability。

## 5. RBO主方法与OBRW的正确层级

九任务静态裁决与第二批basketball反例要求方法层级固定为：

1. **主方法RBO/WFix**：`T^0`静态相对权重、teacher/student各0.5、h25 segment、
   `bootstrap_only`、standard replay；
2. **OBRW在线扩展**：用于研究execution/replay双通道并在slide型regime提供额外控制；
3. **MCG/蒸馏**：supporting/appendix，不进入主方法性能归因。

OBRW不应解释为“已经估准source data value”，也不应替代RBO成为默认算法。更准确的定义是：

> 基于reward-bearing operational feedback，对source执行暴露与其持久replay暴露进行连续、耦合、
> provenance-aware的风险控制。

其当前实现语义：

- student是`S+1`个arm中的一等arm；
- source/student segment在目标环境产生真实target reward；
- `T^online`只作为operational control signal，而非ROI估计；
- student transition采样权重恒为1，source transition只降不升；
- actor/critic使用同一source权重；
- warmup结束后source执行停止，但最后的replay attenuation继续作用于残留轨迹；
- 最终评估为source-free student。

它在slide相对静态RBO有唯一决定性净胜，也在crawl相对execution-only恢复性能；但在basketball仍
3/3低于scratch。因此“连续耦合风险控制”是局部有效机制，不是普遍安全或exact fallback。

## 6. 支持、不支持与待补证

### 已支持

1. execution与replay是两个具有不同持续时间的干预通道；
2. 只修execution不足以关闭crawl负迁移，replay控制带来稳定改善；
3. actor/critic replay干预必须分布一致；
4. 静态RBO在source bank强分化时显著优于uniform，而弱分化时两者可打平；
5. 扩源收益同时受source互补性和target headroom约束；
6. OBRW在slide有显著额外收益，并在crawl揭示replay persistence。

### 不支持

1. OBRW准确估计每个source transition的边际数据价值；
2. `T^online`是通用transferability metric；
3. 所有负迁移都能关闭到scratch（basketball直接否证）；
4. 自动horizon selection已经解决；
5. 稳定性改善等于目标skill transfer；
6. 已证实普遍的asymptotic ceiling gain；
7. OBRW应作为论文默认主方法；
8. `balance_hard`全零probe能产生优于uniform的稳定加权收益或student-only fallback。

### 论文前仍需审计但不立即跑实验

1. RBO与OBRW所有headline run的配置、checkpoint、AUC窗口与seed是否可由registry逐项追溯；
2. OBRW机制表是否全部使用`bootstrap_only + replay_mode=both`，排除MCG gate混入；
3. actor/critic是否在OBRW中确实调用相同weight snapshot和相同sampling语义；
4. return/AUC正例中哪些有source-free hard-progress佐证，哪些只能写sample efficiency；
5. 不补跑balance_hard或新矩阵，除非成稿审计证明其是唯一阻断性缺口。
