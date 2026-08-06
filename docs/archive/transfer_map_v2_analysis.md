# Transfer Map v2 分析：一个重要负结果 + 机制 insight（RBO-PTF Day2-3，2026-06-14）

承接 ChatGPT v3 的 Day2-3:把 Transfer Map 从 episode-level(v1)升级到
snippet-level + safe-horizon,并用 Spearman 验证其对 transfer ROI 的预测力。
**结论分两半:safe-horizon/源选择部分成立且有用;跨任务 ROI 预测部分是干净的
负结果——这反而逼出了一个更深的机制 insight。**

## 一、v2 探针（[scripts/probe_transfer_map_v2.py](../scripts/probe_transfer_map_v2.py)）

对每个 (source o, target T) 执行教师整动作 prefix,从一条 max_h=50 轨迹同时读出
所有 horizon h∈{5,10,15,25,50} 的:prefix reward / progress / fall。score:

  S(o,T,h) = [R_h(o) − R_h(base)] + λ·[Φ_h(o) − Φ_h(base)] − α·P_fall(o,T,h)

baseline 用每任务 **scratch@10k**(opportunity 基线,而非 zero):教师 prefix
相对 scratch 早期 policy 的增量,才是 scratch 学不到的"真对价"(ChatGPT 的
Opportunity = TeacherSnippetScore − ScratchEarlyScore)。safe_horizon =
max{h : P_fall < 0.5}。

## 二、成立的部分:safe-horizon + 源选择（服务 Day4 bootstrap）

snippet-level 修正了 v1 的 episode-level 误判,在脆弱任务上给出正确的 safe horizon:

| 任务 | full-episode fall | **safe_horizon** | 解读 |
|---|---|---|---|
| window | 92-97% | **25** | 长 episode 摔,但前 25 步站立片段可用(v1 误判全员 OOD)|
| balance_hard | 100% | **10-15** | 更脆弱,只前 10-15 步 |
| cabinet/hurdle/maze/powerlift/truck | 0% | 50 | 稳定,可长执行 |

这正是 v1 episode-level 漏掉、Day4 safe-horizon bootstrap 要用的信号:**脆弱 OOD
任务(window/balance)不应让教师长执行,按 time-to-fall 限制到 safe prefix**。这部分
是对的,直接喂 bootstrap 的"执行多久"。

## 三、负结果:snippet teacher score 不预测跨任务 transfer ROI

把每任务"最佳源的 best_score"与训练 eval 的 transfer ROI 做 Spearman(n=9):

| 候选预测信号 | Spearman ρ (vs 相对ROI) | p |
|---|---|---|
| A. v2 snippet opportunity score | **−0.22** | 0.58 |
| B. 教师整段 zero-shot return | −0.22 | 0.58 |
| C. 教师整段 − scratch@50k（v1 判据）| +0.47 | 0.21 |
| D. scratch@50k 能力低（越卡越高）| +0.57~0.68 | 0.04~0.11 |
| E. scratch 早期学习增量小（越卡越高）| +0.70~0.73 | 0.025 |

**最尖锐的矛盾**:cabinet(ROI +70)、window(ROI +76)是收益最高的两个任务,但它们
的 opportunity score **是负的**(−0.024, −0.117);truck/door(ROI +10/+1)score 最高
(16.9, 9.2)。教师 prefix reward 越高的任务,迁移 ROI 反而越低。

**根因**:cabinet/window 的迁移价值是 **stabilization**——教师让机器人站稳,critic
学到"站稳"的长期价值。这个价值在前 50 步的 immediate prefix reward 里几乎看不出来
(scratch@10k 前 50 步也能短暂站着,prefix 差异≈0),但它对 downstream learning 价值
巨大。**snippet-level 的 immediate reward 探针无法捕捉 bootstrap 对后续学习的价值**
——这是 credit assignment 的根本困难,任何"教师 prefix 质量"探针都绕不过。

## 四、连负结果也是 artifact:跨任务 ROI 预测本身 ill-posed

信号 E(scratch 卡住度)看似显著预测相对 ROI(ρ=+0.73),但这是 **分母 confound**:
ROI=(AUC_mcg−AUC_scr)/AUC_scr,scratch 差的任务分母小、ROI 天然偏高。用**绝对
AUC 增益**(AUC_mcg−AUC_scr,无分母)重测:

| 信号 | vs 相对 ROI | vs **绝对增益** |
|---|---|---|
| E scratch 卡住度 | +0.73 (p=0.025) | **+0.10 (p=0.80)** |
| D scratch@50k 低 | +0.57 (p=0.11) | **−0.12 (p=0.77)** |

**scratch 卡住度不预测绝对增益**——之前的相关性几乎全是分母效应。绝对增益排序
(window 199 / truck 187 / hurdle 144 / powerlift 85 / cabinet 82 …)与 scratch 难度
无对应(truck scratch 强=1187 但绝对增益第二)。

**结论:跨任务比较 transfer 收益本质 ill-posed**——绝对增益被各任务 reward 尺度
主导,相对 ROI 被 scratch 分母 confound,n=9 又太小。**不存在一个干净的单探针信号
能跨任务排序 transfer ROI。**

## 五、Transfer Map 的诚实重新定位

不能再声称 Transfer Map 是"跨任务 ROI 预测器"(已证伪)。它真正能做且有价值的是:

1. **per-task 的 bootstrap 配置**:safe_horizon(执行多久)+ 该任务上不立刻摔的源
   (执行哪个)——v2 已产出,直接服务 Day4 safe-horizon bootstrap;
2. **定性 go/no-go 筛选**:score≤0 或 full-episode 高 fall 的 (源,任务) 跳过
   (door 主门、spoon 入杯=教师够不到瓶颈;无对价),而非精确排序。

## 六、被逼出的更深 insight（与 Day1 audit 闭合，可能是更强的论文 message）

把 Day1(增益归因)+ Day2-3(预测失败)合起来,得到一个**定性但机制清晰**的论断:

> **在 FastTD3 + HumanoidBench 上,loco→manipulation 的迁移对价主要来自
> whole-body stability bootstrap,且由 target 任务"scratch 是否卡在 stabilization"
> 决定,而非 source 教师的质量。**

证据链:
- Day1:cabinet/window/powerlift 的增益是 stabilization(开门子任务/存活/站稳),
  非单步任务进度;
- Day2-3:教师质量(prefix reward/整段 return)**反向**于 ROI;真正高对价的任务
  (cabinet/window)恰是 scratch 卡在"站不稳/摔"的任务;
- 合:**scratch 卡在 stabilization 的任务 = 高对价;几乎任何能维持站立 reward-bearing
  状态的 loco 教师都能提供这个 bootstrap**(故"哪个教师"不是主因,"哪个任务"才是)。

这比"Transfer Map 精确预测 ROI"更诚实,也更有 insight:它解释了**为什么** RBO-PTF
有效(off-policy critic 消费教师注入的 reward-bearing stabilization transitions),
以及**在哪些任务**有效(scratch 自身难以稳定的任务)。

## 七、对 ChatGPT v3 方案的影响（待评判）

1. **贡献 1(Transfer Map predictive diagnosis)需重新定位**:从"Spearman 预测 ROI"
   (已证伪)收缩为"per-task safe-horizon/源配置 + 定性筛选";预测性主张若保留,只能
   是"scratch 早期探针预测哪些任务会卡"(target-difficulty diagnostic,且仅相对 ROI,
   带分母 caveat),不是 teacher-quality map。
2. **Day4 safe-horizon bootstrap 的 weight 来源存疑**:"TransferMap-weighted"原计划
   用 opportunity score 加权抽源,但 score 在 stabilization 任务(cabinet)上全负/≈0,
   无法区分源。safe-horizon(执行步数)部分不受影响、仍可做;weight 部分需改用别的
   信号(如 per-task fall-free reward 维持能力,即"哪个源在该任务站得最稳最久")。
3. **可能的论文 message 升级**:核心 insight 从"预测对价"转向"对价 = scratch
   stabilization 难度 × 教师 stabilization 能力",更机制、更诚实。

数据:[logs/probe/transfer_map_v2.jsonl](../logs/probe/transfer_map_v2.jsonl)。
