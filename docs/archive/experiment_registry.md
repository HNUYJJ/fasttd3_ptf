# RBO-PTF 实验与机制总览（experiment registry，更新至2026-07-13）

把"加了哪些机制、几种同时作用、解耦了什么、每个实验代号是什么"一次讲清。
口径以代码为准（`fasttd3_ptf/ptf/mcg.py`、`configs/source_banks/*`、本会话验证）。

> **当前裁决**：性能主通道仍为静态RBO/WFix（`T⁰`相对allocation prior + 0.5 teacher
> branch + h25 + target-reward replay + `bootstrap_only`）；其外增加 student-inclusive admission
> lifecycle：exact-none/quarantine/revocation + authority-coupled physical replay handoff。OBRW是局部
> 在线扩展，MCG蒸馏不进入主方法。自动 transferability/admission estimator 尚未解决。较早的“最终
> 方法”措辞由后续 2026-07-13 行、[`paper_core_contribution_reconstruction_v3.md`](paper_core_contribution_reconstruction_v3.md)
> 和 [`admission_handoff_v1_results.md`](admission_handoff_v1_results.md) 覆盖。

---

## 1. 机制栈：我们一共叠了哪些机制

从底到顶，RBO-PTF = FastTD3 + 一层层加上去的迁移机制：

| 层 | 机制 | 说明 | 谁有 |
|----|------|------|------|
| **L0** | FastTD3 backbone | off-policy, 128 并行 env, distributional critic, replay buffer | 所有实验 |
| **L1** | PTF source options | source policy 作 call-and-return option；原始 PTF 的 transfer loss(模仿) + β termination + λ 衰减 | 所有迁移实验 |
| **L2** | MCG modular + gate | option=(教师, body-group∈{legs_torso,arms,hands})；warmup 后用 critic significance gate(null_option/quantile) 决定是否执行/蒸馏 | full / no_bootstrap |
| **L3** | **RBO warmup bootstrap（核心创新）** | warmup 阶段执行 source option，把 reward-bearing transitions 注入 replay，重塑早期 replay distribution | bootstrap_only / full |
| **L4** | transfer decay | transfer_lambda 线性衰减到 0 | 所有迁移实验 |

**L3（核心）内部还有 4 个子机制（变量）**，这才是我们真正在消融的东西：

| 子机制 | 选项 | 当前实现 |
|--------|------|---------|
| **3a 源选择** | uniform 随机 / reward-weighted | weighted=按目标环境短probe的相对weight作softmax；不是校准ROI |
| **3b 执行时长 horizon** | 固定短(25) / per-source safe-horizon(50 或按 time-to-fall 截断) | terrain fall 低→safe-horizon 退化成固定 50 |
| **3c teacher/student 混合** | warmup_exec_prob | 固定0.5；在segment到期时独立抽teacher/student，长期期望各半，并非逐25步精确平分 |
| **3d admission / exact abstain** | student是一等候选；空集时回归student-only | exact-none/quarantine/revocation基础设施已验证；自动决定何时弃权的utility estimator仍未解决 |
| **3e replay lifecycle handoff** | source behavior authority结束后释放固定replay quota | 30k后回归allowed physical share；powerlift消除80k repetition collapse，truck保留95k正迁移 |

公共超参：warmup_steps=30000，warmup_min_steps=25，decay_steps=80000，total=100000，128 env。

---

## 2. 几种机制同时作用？要解耦哪些？

**问题根源**：我们的"主方法 safe"同时改了 **3a(weighted) + 3b(horizon=50)** 两个变量，相对 rand(uniform+25) 纠缠。reviewer 会问"safe 好是因为选源好，还是执行更久？"

**已解耦的两组**：

1. **3a 源选择 vs 3b horizon**（本会话 wfix 做的）：
   - rand = uniform + horizon 25
   - wfix = weighted + horizon 25（与 rand 只差 **3a 源选择**）
   - safe = weighted + horizon 50（与 wfix 只差 **3b horizon**）
   - ⇒ `wfix−rand` = 纯源选择增益；`safe−wfix` = 纯 horizon 增益。
   - **结论（seed1）**：源选择 +127.8(4/4 正)是主因；horizon −39.7(3/4 负)反而有害。

2. **L3 bootstrap vs L2 gate**（前序会话做的）：
   - bootstrap_only = 只 L3 warmup 执行
   - no_bootstrap = 只 L2 gate/蒸馏，无 warmup 执行
   - full = L3 + L2
   - ⇒ 证明 **bootstrap(L3) 是主性能通道**(boot≈full，no_bootstrap≈0 甚至负)。

**仍待解决**：3d 的自动 admission 判据。严格弃权和 replay 生命周期已实现，但当前 `all/none/static/manifest/schedule` 的决策来源仍是外部快照，不应冒充自动迁移性估计。

---

## 3. 方法代号字典（每个代号到底是什么配置）

### 本会话 terrain 系列（全部 ABLATION=bootstrap_only，只差 warmup）

| 代号 | bank | warmup_mode | 3a 源选择 | 3b horizon | 含义 |
|------|------|-------------|----------|-----------|------|
| **scr** | empty | — | 无源 | — | scratch，纯 FastTD3 从零，迁移基线 |
| **rand** | sources | random | uniform(含无用的 stand) | 25 固定 | 随机 warmup（PTF 朴素版） |
| **safe** | safe | safe_bootstrap | weighted(vs-zero) | **50** per-source | 当前"主方法"（reward-weighted + safe-horizon） |
| **wfix** | wfix | safe_bootstrap | weighted(vs-zero) | **25** 固定 | 解耦用：weighted 源 + 短 horizon |

> `safe` 与 `wfix` 唯一差别 = horizon(50 vs 25)；`wfix` 与 `rand` 唯一差别 = 源选择(weighted vs uniform)。

### 历史代号（前序会话，memory 记录）

| 代号 | 含义 |
|------|------|
| `mt_{safe,rand,scr}` | multitask 三方（balance_hard/hurdle/cabinet/window） |
| `d56_safe` | window 的 safe 变体 |
| `bootstrap_only / no_bootstrap / full` | L3/L2 ablation（前序证 bootstrap 是主通道） |
| `admission-all+handoff` | student-inclusive categorical bootstrap；source authority结束后replay由固定quota切至allowed physical share |
| `pilot_{mcg,boot,nobo,scr}` | 宽 pilot 的 SC-MCG / bootstrap_only / no_bootstrap / scratch |
| `nc_{safe,scr}` | negctrl door/spoon（**待跑**，验证无对价任务 safe≈scr） |
| `br_{safe,rand,scr}` | breadth maze/truck（OOM 杀，**待跑**，验证广度） |

---

## 4. 实验总表（做过的所有实验）

### A. 本会话 terrain 主线（核心，有结果）

| 实验 | 任务 | 方法 | seed | 状态 | 关键结果 |
|------|------|------|------|------|---------|
| 核心三方 | stair/slide/pole/crawl | scr/rand/safe | 1,2,3 | ✅完成 | safe>rand 10/12(t≈2.58)；crawl 翻转(safe<rand) |
| wfix 解耦 | stair/slide/pole/crawl | wfix | 1 | ✅完成 | 源选择 +127.8(4/4)；horizon −39.7(3/4 有害) |
| wfix 加固 | stair/slide/pole/crawl | wfix | 2,3 | ✅完成 | **3-seed 定论：源选择 +77.9(11/12, t=+3.08)；horizon −11.4(t=−0.46 中性、任务依赖，修正 seed1"有害")** |
| fall-recovery probe | stair/slide/pole/crawl | safe/rand/scr | 1 | ✅完成 | safe 摔倒终止率最低(0-3%)；一摔即done测不到恢复 |
| hurdle→stair probe | stair | hurdle/walk/run/stand/scr | — | ✅完成 | zero-shot hurdle 不救场(100%摔)；dynamic 源 zero-shot 不可靠 |

### B. 前序会话（memory 记录，背景）

| 实验 | 任务 | 方法 | 状态 | 关键结果 |
|------|------|------|------|---------|
| multitask 三方 | balance_hard/hurdle/cabinet/window | mt_{safe,rand,scr} ×3seed | ✅ | hurdle/cabinet safe≥rand≫scr；window 高方差；balance 无对价 |
| bootstrap ablation | hurdle/cabinet/maze/powerlift | scratch/bootstrap_only/no_bootstrap/full ×2seed | ✅ | boot≈full；no_bootstrap≈0 → bootstrap 是主通道 |
| 边界 pilot | truck/spoon/door | SC-MCG vs scratch | ✅ | +10%/+8%/+1%，regret 0（弱对价，符合 Effect Map 判级） |
| zero-shot transfer map | 17 任务 × 6 源 | zero-shot | ✅ | 迁移=源-任务-分量三元相关；跨任务 ROI 预测 ill-posed |
| task-progress audit | 9 任务 | full vs scratch | ✅ | 增益分层(真完成/加速/稳定/无对价)，非"只是站更久" |

### C. 进行中/待跑（2026-07-02 按导师意见重排，见 advisor_feedback_analysis_20260702.md）

| 实验 | 目的 | 状态 |
|------|------|------|
| **Step A: online_bootstrap（student-as-arm）** | 替代 abstain：学生=平等 arm+在线 EMA | ✅代码+双向自测+冒烟全过 |
| Step A 验证 crawl+pole ×1seed | crawl ≥scr？pole ≈wfix？ | ✅**pole PASS(−4%)；crawl 726<scr 833 未达标但机制正常(share 50→76%,排序学对)——诊断:残留 buffer 毒害是主因** |
| **Step B: replay 按 T 重加权（意见1）** | 驻留坏轨迹降权（PTFReplayWrapper.sample） | ✅**crawl +41.6(726→767.5,gap收窄39%,超wfix成crawl迁移最好成绩)——残留毒害假设证实；pole −30.8(教师有用任务降权有代价→下轮:自适应激活)** |
| T-gated abstain（ChatGPT 一优,合并硬abstain+自适应激活） | 全源<stu−δ 持续K→abstain | ✗**crawl abstain 从未触发**(arm分离度0.002-0.009<δ0.0095)→退化 onlineb(727);pole 749.5 PASS。教训:弱信号下二值阈值脆弱,连续降权(obrw)稳健。核心缺口:执行reward区分不了"pole型略差(数据有用)"vs"crawl型略差(数据有毒)" |
| split 归因矩阵 v2 | aonly/conly/split × crawl/pole | ✅**对称性主导**:单/非对称降权全伤(crawl aonly−50/conly−96/split−10),仅对称both获益;AC数据分布一致性=独立finding;首轮因漏normalize_obs作废重跑 |
| seed 加固 onlineb/obrw ×crawl/pole×s2/3 | 裁决 pole−31 真伪 | ✅pole−31=噪声(t=−0.50),obrw≥0.95wfix PASS;crawl obrw+94.7(t=2.54)稳健;onlineb 无replay保护在crawl不稳(±77) |
| **主表补全 stair/slide×onlineb/obrw×3seed** | 4任务×6方法×3seed 主表成型 | ✅**slide: obrw 614.8 全场第一,vs wfix +92.1(t=14.7)="在线vs静态"决定性胜利**;stair=horizon边界(safe h50=279唯一超scr,T^online盲于时长);pole/crawl 打平最佳 |
| **horizon-arm 扩展(mh, S×H+1 arms)** | 修 stair 边界,四任务全绿 | ✗**3-seed 否决(STAMP 071959Z+091439Z)**:stair 224.6±52(vs obrw +40 t=0.86 不稳,未过 scr 线)、slide **−158(t=−2.87)**、crawl **−63(t=−2.63)** 显著恶化;归因=探索税(先验档均匀)+毒害扩大(h50 翻倍执行)+统计代价(7 arm 样本减半,std 全面翻倍)。**正面保留**:在线竞争自动辨识 horizon 偏好且四任务方向全对(stair→h50,slide/pole→h25,双seed复现,无任务名分支);stair 上唯一 vs scr 不显著负的迁移法。mh 进 appendix；后续全局裁决已把静态RBO定为默认主方法，OBRW仅为在线扩展 |
| **negctrl door/spoon({obrw,scr}×3seed)** | 在线扩展的soft退让验证 | ✅(STAMP 124531Z)door:+0.7(t=0.10)零伤害,student share升86-88%,replay权重压0.10-0.25；spoon:+38.8(t=+3.69,3/3正),方差显著降低。该结果支持局部soft control，但basketball后续3/3负已否证“无对价必自动关闭/普遍安全”；bank=wfix_{door,spoon}(safe改h25) |
| Step C: T^critic 半交互度量 + 度量对比表 | 导师核心要求："不实际交互"的 transferability 公式 | 设计定稿（文档§3），待 Step A 落地 |
| negctrl door/spoon | 验证无对价任务 safe≈scr（解释边界） | 排 Step A 后 |
| hurdle→stair training-level | 验证 skill-diverse 源 | 排后 |
| **breadth 第一批 maze/truck/cabinet ×{scr,randb,obrwb,obrws}** | 广度+scalability(贡献3);PI 定向 2026-07-04 | ✅pilot 12 runs(STAMP 015912Z):**广度全绿**(obrws vs scr: maze+14%/truck+16%/cabinet+168%);**扩源反向**(obrws≥obrwb 三任务,truck 大源库−200,arm value 整体下移反馈环+执行reward/数据价值分离放大)=与 mh 同构的第二个"arm 空间扩张有结构性代价"证据,诚实收口为负结果小节;详见 docs/breadth_expansion_20260704.md §5 |
| **breadth 3-seed 补齐 {scr,rands,obrws}×3任务** | 广度表成型(与 terrain 主表同口径 3 loco 源) | ✅(STAMP 135105Z)**广度 3/3 显著**:obrw vs scr maze+74.0(t=3.63)/truck+148.5(t=3.25)/cabinet+98.5(t=4.78);**vs rand 全打平**(选源增益取决于源分化度:terrain 20倍权重差 vs breadth 2-3倍;T⁰ 权重 CV 训练前可预测);三层收益拆解=注入数据主收益/选源看分化/在线只在 slide 型;详见 breadth_expansion_20260704.md §7 |
| **wfix 裁决实验 5任务×wfix×3seed=15runs** | PI 对账质疑(2026-07-04 btw):在线层复杂度是否值得;预注册=全线打平且 door/spoon 无伤害→主方法简化为静态版 | ✅(STAMP 152010Z)**裁决=主方法简化成立**:wfix vs obrw 在maze/truck/door打平，cabinet由wfix显著赢(+15.7,t=11.4)，spoon由obrw小幅赢(+7.3,t=2.66)；9任务账本另有obrw在slide决定性大胜，其余6任务打平。在线收益集中而非普遍，**默认主方法=静态RBO，OBRW=局部扩展**；详见 breadth_expansion §8 |
| **扩源 v2 正交源 probe(9源×5目标)** | 归因盲点修正:7源实验只证"冗余源无收益";正交源(hurdle 跨障/reach 伸手)翻盘假设=cabinet 弱对价是覆盖缺口 | ✅probe:reach 全线≤2.1(非 adapter 问题,"站稳伸手"无对价,cabinet 翻盘假设证伪);**hurdle maze 16.4/truck 13.6 双冠**;window≤3.1 弱对价/powerlift≤0.9 极端 negctrl 候选 |
| **stability-deconfounded audit P0/P1** | 检验增益是否只是站立/生存，及 cabinet 源身份差异 | ✅四任务三 seed + cabinet 单源加固：maze 有真实早期进展；powerlift 无技能证据；basketball 100k 负迁移；cabinet run>stand 在 30k/100k 均 3/3 seed 同向，评估 survival 全相同，排除“只是评估期活得久”；详见 `docs/stability_deconfounded_transfer_audit_v1_findings.md` 与 `docs/stability_deconfounded_audit_p1_cabinet_s123_findings.md` |
| **cabinet run-dose-matched P2** | 区分多源 active dilution 与 run dose/opportunity cost | ✅run24×3 seed；实际 run share 23.829% vs WFix 23.894%，匹配成立。30k run24−WFix=+0.031±0.113(−/0/+)，run50−run24=+0.229±0.416(+/+/−)：**严格预注册裁决=方向混合、机制不确定**；active toxicity 未支持，均值轮廓仅与机会成本一致；return 与 hard progress 方向错位；详见 `docs/stability_deconfounded_audit_p2_cabinet_run24_findings.md` |
| **hurdle 增量实验 maze/truck×wfix-4src×3seed** | 检验"T⁰ 增量预判扩源收益"(扩源三部曲收官) | ✅(STAMP 113556Z)**truck +229.9(t=+3.47,1421.3 全场新高,超 obrw,方差减半)**;maze +0.3 零转化(饱和解释:四方法挤 340-355);定稿="T⁰ 增量=必要非充分,饱和判据=AUC 聚拢度" |
| **第二批扩展 5任务×{scr,rand,wfix(std9)} + basketball OBRW** | PI 定向(2026-07-05):更多任务+源库扩至标准9源(4官方+5自训) | ✅48/48逻辑run slot有效（50次尝试；balance_hard rand-s3/WFix-s2首次OOM后于2026-07-12原配置恢复）：powerlift WFix−scr **+77.6(t=14.78)**；basketball WFix **−101.5**、OBRW **−74.0**且均3/3负；window负趋势、bookshelf正趋势；balance_hard WFix−scr `+6.1(t=0.75)`且WFix≈rand。T⁰低分无方向判别力，全零权重无加权优势，OBRW不构成普遍安全保证；详见breadth_expansion §10与`artifacts/breadth_batch2_local_audit/` |
| **Admission Core v1** | student-inclusive exact-none、quarantine、runtime revoke、admission-consistent replay | ✅basketball exact-none 3seed source execution/replay/distillation全零且性能回到scratch分布；powerlift all-source 发现fixed quota在退役期过采样旧source，旧100k retention gate因headroom耗尽作废；详见`admission_core_v1_results.md` |
| **Authority-coupled replay handoff v1** | 修复behavior authority结束后fixed replay quota的repetition divergence；truck裁决late retention | ✅6/6 formal runs。powerlift 35–80k fix−fixed=`+20.1`(3/3正)、fix−WFix=`−1.46`，80k mean `192.2→319.4`；实际critic source share `50.0%→33.65%`(30–60k)、`34.87%→7.19%`(60–90k)，命中预结果预测；truck 95k fix−scratch=`+227.8`, t=4.74, 3/3正，保留legacy gap 84.3%。详见`admission_handoff_v1_results.md` |
| **Adaptive behavioral-source revocation v1（Phase A）** | 时间维弃权自动化：stage-window(3000) segment 级 UCB/LCB 单向撤销判据能否截断负迁移且不误伤正迁移 bank | ❌**预注册裁决 FAIL**（18/18 runs 完成，18/18 lifecycle audit PASS）。crawl `+41.5/−66.8/+53.9`，但 s2 无撤销仍有 `−66.8` placebo，故收益与机制归因均不成立；truck 撤 walk/hurdle 3/3、run 2/3，s2/s3 代价 `−119.7/−204.9`（禁撤 gate FAIL）；powerlift 保持 PASS（9k 撤 crawl/reach 3/3，Δ≥−4.9）但只支持兼容性，不构成单源 learning-utility 标签；basketball 大量撤销仍无系统改善，不能作行走源因果归因。**结论：目前试过的行为 reward proxy 不能可靠驱动 automatic admission；behavior utility 与 delayed learning utility 必须分离。** 详见`adaptive_admission_v1_results.md`与`adaptive_admission_v1_codex_independent_audit.md` |

> active perturbation probe 已完成并撞环境硬限制（HB 摔倒即终止），结论=fall-avoidance，从待跑移除。
> abstain(LCB) 被 student-as-arm 取代（导师意见2，语义等价更简洁），LCB 留作可选消融。

---

## 5. 一句话小结

- **机制**：性能核心是 L3 **reward-bearing option bootstrap**；3a源选择在强分化bank中是已证主因，
  3b horizon总体中性但任务依赖；exact fallback基础设施和authority-coupled replay lifecycle已解决，自动admission判据未解决
  （行为 reward 信号已被 SIV/SHU/adaptive-revocation 三重独立否定——后续只能换信号族，勿做行为信号第四种变体）。
- **解耦**：已完成[source selection vs horizon]、[bootstrap vs gate]、[execution vs replay]和
  [actor vs critic replay distribution]；SHU证明behavior utility不能替代update utility。
- **当前主方法**：静态`reward_weighted_bootstrap`（WFix，weighted+h25）+ provenance-consistent
  admission/replay lifecycle；OBRW仅作局部在线扩展，MCG/蒸馏与失败gate进入supporting/appendix。
