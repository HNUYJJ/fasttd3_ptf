# RBO-PTF 执行报告与求诊 v3（发 ChatGPT-5.5-Pro，2026-06-14）

承接你上轮的七天计划(Day1 task-progress audit → Day2-3 Transfer Map v2 →
Day4 safe-horizon weighted bootstrap → Day5-6 重跑 → Day7 baseline)和论文重命名
RBO-PTF。我们跑完 Day1-4,得到**一个挑战你方案核心的负结果**(Transfer Map 跨任务
ROI 预测 ill-posed)和**一个被它逼出的、可能更强的机制 insight**,需要你评判贡献 1
的重新定位与论文 message。Day5-6 的 window 三方实验正在跑,结果稍后补。

---

## 一、Day1 Task-Progress Audit:增益归因四层(诚实,且比预期有 insight)

工具 `scripts/task_progress_audit.py`:load pilot checkpoint(复现 obs normalizer)
→ 确定性 rollout → 采集 HB info 细粒度 reward 分量 → 分离 task-progress vs
alive/stand → 沿训练算 AUC,full vs scratch(seed1)。**核心:每任务用最硬的完成度
字段(HB 官方 success_subtasks / object-goal proximity / 前进量),不是黑名单和。**

| 归因层 | 任务 | 硬完成度证据(full vs scratch AUC) |
|---|---|---|
| **A. task-completion 增益** | cabinet | success_subtasks **+169%**(完成子任务数近 3 倍)|
| | truck | robot-package-to-truck proximity **+60%**(partial,未真上车)|
| | maze | success_subtasks **+27%**(到达 checkpoint 数)|
| **B. time-to-progress 加速** | hurdle | move 50k 时 2×(0.73 vs 0.36),末端持平 |
| **C. stabilization 主导**(prerequisite subskill)| window | 擦窗 per-step 仅 +11%,return +117% 真因=**fall AUC −0.40**(81%→31% 存活)|
| | powerlift | dumbbell_lifted **全程恒定 0.189**(从未举起),return +49% 纯站稳 |
| | balance_hard | 任务=平衡,站得更久 +36% |
| **D. 无对价/瓶颈未触及** | spoon | spoon_in_cup=0 两者(control);door 主门 openness=0(loco 够不到)|

**诚实 claim**:RBO-PTF 在 cabinet/maze/truck 提升 task-specific completion,在
hurdle 加速 progress 获得,在 powerlift/window/balance bootstrap whole-body stability
(manipulation 前置技能,印证 HB 论文"先 locomotion 后 manipulation"),在 spoon/door
正确地不迁移(near-zero regret)。比"全是 task-progress"可信,比"只是站得更久"有 insight。

---

## 二、Day2-3 Transfer Map v2:safe-horizon 成立,但跨任务 ROI 预测是负结果

`scripts/probe_transfer_map_v2.py`:对每个 (source, target) 执行教师 prefix,从一条
max_h=50 轨迹读出所有 h∈{5,10,15,25,50} 的 prefix reward / progress / fall。

### 成立的部分(服务 Day4 bootstrap):safe-horizon + per-task 源选择
snippet-level 修正了 v1 episode-level 的误判:window full-episode fall 92-97%(v1 据此
判全员 OOD),但 v2 给出 **safe_horizon=25**(前 25 步 fall<0.5),正确捕捉"长 episode 摔、
前 25 步站立片段可用"。balance safe_h=10-15,稳定任务=50。这直接喂 bootstrap 的"执行多久"。

### 负结果(挑战你的贡献 1):snippet teacher score 不预测跨任务 transfer ROI
把每任务"最佳源 best_score" vs 训练 eval transfer ROI 做 Spearman(n=9):

| 候选预测信号 | Spearman ρ |
|---|---|
| v2 snippet opportunity score(教师 prefix 相对 scratch@10k)| **−0.22** |
| 教师整段 zero-shot return | −0.22 |
| scratch 卡住度(scratch 早期学习增量小)| +0.70~0.73 |

最尖锐矛盾:**cabinet(ROI+70)、window(ROI+76)是收益最高的两个任务,但 opportunity
score 是负的**;truck/door(ROI+10/+1)score 最高。**教师 prefix reward 越高的任务,ROI
反而越低**(truck/door 的 loco 走路得分高,但那恰是 scratch 也容易的任务)。

根因:cabinet/window 的迁移价值是 stabilization(教师让 critic 学到"站稳"的长期价值),
这在前 50 步 immediate prefix reward 里几乎看不出(scratch@10k 前 50 步也能短暂站着),
但对 downstream learning 价值巨大。**snippet-level immediate reward 探针无法捕捉
bootstrap 对后续学习的价值——credit assignment 的根本困难,任何"教师质量"探针都绕不过。**

### 连"scratch 难度预测"也是 artifact:跨任务 ROI 预测本身 ill-posed
scratch 卡住度看似预测相对 ROI(ρ=+0.73),但 ROI=(AUC_mcg−AUC_scr)/AUC_scr,scratch
差的任务分母小、ROI 天然偏高。用**绝对增益**(无分母)重测:

| 信号 | vs 相对 ROI | vs **绝对增益** |
|---|---|---|
| scratch 卡住度 | +0.73 (p=.025) | **+0.10 (p=.80)** |

**scratch 难度不预测绝对增益**——之前相关性几乎全是分母 confound。绝对增益排序
(window 199/truck 187/hurdle 144/…)与 scratch 难度无对应。**结论:跨任务比较 transfer
收益本质 ill-posed(绝对被各任务 reward 尺度主导,相对被 scratch 分母 confound,n=9 太小),
不存在干净的单探针信号跨任务排序 ROI。**

### 被逼出的机制 insight(与 Day1 闭合,可能是更强的论文 message)
> **在 FastTD3 + HB 上,loco→manipulation 的迁移对价主要来自 whole-body stability
> bootstrap,且由 target 任务"scratch 是否卡在 stabilization"决定,而非 source 教师
> 质量。** scratch 卡在 stabilization 的任务(cabinet/window/powerlift/balance)对价大;
> 几乎任何能维持站立 reward-bearing 状态的 loco 教师都能提供这个 bootstrap(故"哪个
> 教师"不是主因,"哪个任务"才是)。

---

## 三、Day4 safe-horizon weighted bootstrap:已实现+三层验证

`McgBehaviorController` 加 `warmup_mode=safe_bootstrap`:warmup 期到期 env 按
**Transfer Map v2 的 per-source reward-bearing weight 的 softmax 抽源**(替代 random)
+ 按 **per-source safe_horizon 锁存**(替代固定 25 步)。脆弱任务(window 25/balance
10-15)的教师只执行 safe prefix,不注入摔倒片段。

- **weight 来源的关键修正**:用 **vs-zero reward-bearing score**(教师能维持的 reward
  状态质量),不是失效的 opportunity score(opportunity 在 stabilization 任务上全负,
  无法区分源——因为 scratch 早期也短暂站着)。这区分了两件事:**跨任务 go/no-go**
  (ill-posed)与 **per-task 源选择**(vs-zero score 有效:window stand/walk>run,
  cabinet run/stand>walk,hurdle run≫)。
- safe bank 自动生成(`build_safe_bootstrap_banks.py`,per-source bootstrap.{weight,
  horizon} 写进 yaml,不破坏原 bank;random vs safe 对比=切 SOURCE_BANK)。
- 三层验证全过:单元(抽源分布匹配 softmax、horizon 锁存正确)、bank 读取、端到端
  smoke(window safe_bootstrap 跨 warmup 边界 1500 步无 crash)。

---

## 四、Day5-6 多任务三方对比(safe vs random vs scratch, 全 bootstrap_only, ×3 seed)

balance/hurdle/cabinet × 3 方法 × 3 seed + window。**绝对 AUC**(避开 ROI 分母
confound;safe/rand 均 bootstrap_only,只差 warmup 方式——隔离了 gate):

| task | scratch | rand | **safe** |
|---|---|---|---|
| **hurdle** | 155±13 | 466±3 | **538±4** |
| **cabinet** | 116±19 | 199±7 | **205±4** |
| balance_hard | 86±3 | 88±5 | 92±11 |
| window | 309±**88** [240,432,254] | 280±51 | 234±27 |

**诚实解读(含对我们 Day5 单任务结论的撤回)**:
1. **hurdle/cabinet: RBO bootstrap 强正迁移, 且 safe ≥ random**。hurdle safe 538 >
   rand 466 ≫ scr 155(safe 比 random **高 +15% AUC**、方差最小 ±4);cabinet safe
   205 ≳ rand 199 > scr 116。**safe 的优势来自 weighted 源选择**(hurdle run 被优先
   抽,horizon=50 未限制)。这是 safe_bootstrap 的扎实正面证据。
2. **撤回 Day5 window +15%(重要)**:Day5 用 2-seed 旧 scratch(AUC=204,恰好低)算出
   safe +15%;3-seed scratch 实为 **309±88**([240,**432**,254],432 是 lucky seed)。
   去各方 outlier 后三方都 ~240、完全重叠 → **window transfer 无净对价,其高方差来自
   scratch baseline 本身,而非 bootstrap**。与你我 Day2-3 的"window scratch 不弱、
   教师无对价"完全一致。Day5 的 +15% 是 baseline 抽样波动的产物,不成立。
3. **safe-horizon 步数限制的独立价值仍未证**:它本为 window/balance 这类脆弱任务
   设计,但这两个任务恰恰无对价(scratch 自己够好/都差),没有"有对价+脆弱"的场景来
   体现"限制脆弱执行"的价值。safe_bootstrap 当前全部正面证据来自 weighted 源选择
   (机制1),不是 safe-horizon(机制2)。

**净结论**:RBO bootstrap 在强对价任务(hurdle/cabinet)强正迁移、safe≥random(机制1
=weighted 源选择有效);但"safe-horizon 解决脆弱任务高方差"的卖点**未立**(window
无对价是 confound,Day5 乐观数字已撤回)。33 runs 无报错、全 3 seed。

---

## 五、待你评判的决策点

**[多任务结果引出的最关键新问题] safe-horizon 这个名字/卖点还保不保得住?**
safe_bootstrap 有两机制:(1)reward-weighted 源选择、(2)safe-horizon 步数限制。多任务
证明只有(1)有价值(hurdle/cabinet safe≥rand);(2)的独立价值无场景体现——它为脆弱任务
设计,但脆弱任务(window/balance)恰恰无对价(scratch 自己够好/都差)。三条路:
(a) **方法改名** "Reward-Weighted Option Bootstrap",去掉未证的 safe-horizon,主张
    =按 Transfer Map reward-bearing weight 选源 bootstrap(hurdle/cabinet 扎实);
(b) **补"有对价+脆弱"任务**证 safe-horizon——但 Transfer Map 显示这类任务难找(脆弱⇒
    scratch 也难⇒往往无对价),可能本就稀少;
(c) **safe-horizon 保留为 robustness 而非 performance**:即使无性能增益,它防止脆弱
    任务负迁移(window safe ±27 比 rand ±51 稳),作为"安全保险"写,不声称提升。
你倾向哪条?还是有别的framing?

1. **贡献 1(Transfer Map predictive diagnosis)怎么重新定位?** 跨任务 ROI 预测已证伪
   (ill-posed)。是否接受把 Transfer Map 从"ROI 预测器"收缩为 **(a) per-task bootstrap
   配置器(safe-horizon + 源选择,已用于 Day4)+ (b) 定性 go/no-go 筛选**?Day7 的
   selected-vs-uniform/shuffled baseline 在 per-task 内部仍能证"源选择有用",但不能
   证"跨任务选址"——这个降级你认可吗?

2. **机制 insight 提为核心 message?** "迁移对价 = scratch stabilization 难度 × 教师
   stabilization 能力,非教师质量"——是否取代"Transfer Map 预测对价"成为论文核心 insight?
   它更诚实、更机制,但放弃了"我们能预测哪些任务值得迁移"的卖点。

3. **safe_bootstrap 的 weight 用 vs-zero reward-bearing score**(per-task 源能维持的
   reward 状态质量),你认可吗?还是有更好的 per-source weight(如 fall-free reward
   维持时长)?

4. **如果跨任务预测 ill-posed,RBO-PTF 怎么决定"在哪些任务上 bootstrap"?** 目前靠定性
   (教师不立刻摔 + 有正 reward),但这不是 principled。是否需要一个 target-side 的
   "scratch 早期卡住"探针(10k 步预算,比训练便宜 10×)作为 go/no-go,即使它只对相对 ROI
   有(带 confound 的)预测力?还是干脆不声称选址、只声称"在给定任务上正确配置 bootstrap"?

5. **三贡献结构**现在是:(1)Transfer Map=bootstrap configurator(降级);(2)Reward-Bearing
   Option Bootstrap=主性能(ablation 坐实 boot≈full,最强);(3)safety gate(降级 appendix)。
   核心 insight=stabilization-difficulty-driven transfer。这个结构够 ICML 吗?贡献 1
   降级后,会不会显得单薄(只剩主性能 + 一个机制观察)?

## 六、约束(不变)
- PTF + FastTD3 + HB,创新长在 PTF 内,官方代码不可改(train_ptf.py 外挂副本可改);
  方法通用、多任务,无任务名分支(bank yaml 的 per-source bootstrap 字段=环境元数据,合法)。
- 算力:8×V100 32G,单跑 100k≈2h,最多 4 并行。
