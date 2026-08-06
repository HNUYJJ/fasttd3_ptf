# 扩源扩任务第一批（2026-07-04，PI 定向：进入 ⑤，我独立掌舵）

方向由 PI 直接裁定（不再经 ChatGPT）：预注册序列 ①-④ 已收口（见
`advisor_feedback_analysis_20260702.md` §6.10/§6.11），进入扩源扩任务。

## 1. 战略设计（独立判断）

扩展要回答两个审稿人问题，各配一条证据链：

1. **任务广度**：terrain 之外（manipulation/whole-body）方法是否成立
   → 新任务上 obrw vs scr/rand。
2. **源库 scalability**：源多了、坏源多了，在线竞争还稳吗
   → 同任务 obrw-big(7源) vs obrw-small(3源) vs rand-big(7源 uniform)。
   预期："源库越大 uniform 越差、obrw 靠 T⁰ 先验+在线竞争持平或更好"
   ——源库越大，自动选源的价值越凸显，这是 T-RBO 的主场。

**mh 教训的直接应用**：7源+学生=8 arms（与 mh 的 9 相近），但结构性差异是——
mh 的 h50 档间先验 uniform（纯税）；大源库的先验按 T⁰ weight 抽，先验预算自动
集中在 probe 分数高的源上。**T⁰ 先验 + T^online 修正的两级结构在大源库场景才
充分发挥**——把 mh 的负结果转化为设计论证。

## 2. 第一批配置

- **任务**：maze（loco 导航，151/76）、truck（whole-body 搬运，216/111）、
  cabinet（manipulation，213/109）。三者跨类别且各有旧数据锚点（bootstrap
  ablation / 边界 pilot +10% / mt 三方 safe≫scr）。basketball/powerlift 等
  全新任务放第二批。
- **扩源（7源）**：stand/walk/run（官方 loco）+ crawl/pole/slide/stair
  （terrain scr s1 final，STAMP 20260615T044012Z）。terrain 任务 obs=151/nq=76
  与 loco 源完全同布局（hb_task_layouts 核实）→ identity 直连零 adapter 风险；
  manifest 在 `checkpoints/terrain_sources/`，SourcePolicy 加载+act 已验证。
  reach（157 布局特殊、REWRITTEN_PROPRIO_PREFIX）留第二批。
- **方法列瘦身**：scr / rand-big / obrw-big / obrw-small。wfix/safe 不重复
  （"静态 vs 在线"已在 terrain 说清）。
- **口径**：全部与主表一致（100k 步、warmup 30k、h25、eval 5k、95k AUC）。

## 3. probe 结果（transfer_map_v2_bigsrc.jsonl，7源同批 vs-zero）

| 源 | maze | truck | cabinet |
|----|------|-------|---------|
| stand | 4.6 | 5.2 | 1.5 |
| walk | **12.0** | **12.7** | 0.9 |
| run | 10.7 | **12.6** | 2.3 |
| crawl | 1.0 | 0(-0.2) | 0(-0.1) |
| pole | 4.7 | 6.9 | 2.6 |
| slide | 9.5 | 10.5 | **4.6** |
| stair | 5.2 | 0.3 | 0.1 |

判读：maze/truck=loco 主导（walk/run≈12）；cabinet=弱对价（全员≤4.6，与旧
3源 probe 一致）；crawl 在 truck/cabinet 被正确判 0；terrain 源经 proprio 视图
adapter 在 216/213 布局任务上行为正常（不摔、分数量级合理）——adapter 无恙。

## 4. bank 与 run

- bank：`h1hand_big_wfix_{t}.yaml`（7源+weight+h25，obrw-big）、
  `h1hand_big_sources_{t}.yaml`（7源 uniform，rand-big）、
  `h1hand_loco_wfix_{t}.yaml`（3源，safe 改 h25，obrw-small）；
  生成脚本 `scripts/build_bigsrc_banks.py`。
- pilot：12 runs（3任务×4方法×s1），STAMP `20260704T015912Z`，
  exp 名 `h1hand_{t}_br_{scr,randb,obrwb,obrws}_s1_*`。
- 通过后 3-seed 加固（+24 runs），再视结果第二批（basketball/powerlift/reach 源）。

## 5. pilot 结果（2026-07-04，1-seed，95k AUC）

| task | scr | rand-big | obrw-big | obrw-small |
|------|-----|----------|----------|------------|
| maze | 297.7 | 337.4 | 339.1 | **340.5** |
| truck | 1191.1 | 1293.2 | 1182.7 | **1382.2** |
| cabinet | 83.1 | 192.5 | 208.7 | **222.6** |

### 判读

1. **广度链全绿（进 3-seed）**：obrw-small vs scr 三任务全正——maze +14%、
   truck +16%、cabinet **+168%**（83→223）。manipulation（cabinet）与
   whole-body（truck）任务上拿到正迁移，广度主张有了跨类别支撑。
2. **scalability 链反向（负结果，诚实收口）**：obrw-small ≥ obrw-big 三任务
   全部成立，truck 上大源库 −200（obrw-big 1183 甚至 < scr 1191）。机制读数：
   - truck/obrw-big 所有 arm value 整体下移（run 0.789 vs 小源库同源 0.966）
     ——**反馈环**：大源库 warmup 注入更多杂源数据 → 学生早期更弱 →
     rollout 状态分布更差 → 所有 arm 执行 reward 被拉低；
   - maze/obrw-big：stair 源以执行 reward 霸占 79% 份额（T^online 1.274 遥遥
     领先，推翻 T⁰ 排序 walk>run>slide），但 AUC 与小源库打平——**执行期
     reward 与数据价值分离在大源库被放大**（T-gated 时代核心缺口的重演）;
   - cabinet：两版行为一致（都主要执行 stand，60-70% 份额），差异在噪声带。
3. **与 mh 否决同构**：这是"扩 arm 空间有结构性代价"的第二个证据（mh=横向加
   horizon 档，big=纵向加源）。共同教训：**arm 空间的扩张必须由数据价值信号
   而非执行 reward 信号来 justify**——当前 T^online（执行 reward EMA）对
   "多 arm 大空间"的分辨力不足以覆盖其统计与数据质量成本。

### 决策（2026-07-04）

- 广度主线进 3-seed：广度表口径与 terrain 主表对齐（3 loco 源），三行
  scr / rand(3源 uniform) / obrw(3源)，补 21 runs（STAMP `20260704T135105Z`）。
- 扩源初探以本节 1-seed 数据诚实收口为负结果小节，不再花 3-seed 预算确认；
  第二批扩源（basketball/powerlift 任务、reach 源）暂缓，优先成稿。
- 论文叙事获得意外增强：mh + big 两个负结果共同指向"reward-bearing 信号的
  边界"，与三盲区拼图（T⁰/T^critic/T^online）合并成完整的 transferability
  度量论证——什么信号能支撑什么粒度的决策。

## 6. 扩源 v2（正交源）probe 结果（2026-07-04，9 源 × 5 目标，transfer_map_v2_orthosrc.jsonl）

| task | stand | walk | run | crawl | pole | slide | stair | **hurdle** | **reach** |
|------|-------|------|-----|-------|------|-------|-------|-----------|-----------|
| maze | 4.6 | 11.3 | 10.0 | 1.6 | 4.7 | 10.7 | 5.6 | **16.4** | 0.1 |
| truck | 5.2 | 12.8 | 12.6 | 0 | 6.7 | 10.8 | 0.3 | **13.6** | 2.1 |
| cabinet | 1.4 | 0.2 | 2.5 | 0 | **5.9** | 5.4 | 0 | 1.3 | 0 |
| window | 2.4 | 2.3 | 0.7 | 0.7 | 0.8 | 2.4 | 0.9 | **3.1** | 0.9 |
| powerlift | **0.9** | 0.8 | 0 | 0 | 0.7 | 0 | 0.7 | 0.4 | 0 |

### 判读

1. **翻盘假设部分证伪**：reach（唯一 manipulation 候选源）全线低分（≤2.1），
   且诊断确认非 adapter 问题——fall 0-5%（除 window），行为稳定但不产 reward，
   即"站稳伸手"型技能在这些任务前期窗口无对价。cabinet 的弱对价（上限 5.9）
   不是现有源库能解决的——需要真正的开门/抓取技能源（当前不存在）。
2. **意外收获——hurdle 源两处夺冠**：maze 16.4（超 walk 11.3 达 +45%）、
   truck 13.6（第一）。跨障技能的"稳健快速前进"是比 walk 更好的导航/搬运先验。
3. **叙事升级——T⁰ 作为扩源决策工具**：冗余源实验（7源，全无 probe 增量）
   无收益 vs hurdle（probe +45% 增量）待验证——若 hurdle 入库能提升 maze/truck
   的最终 AUC，则"扩源的价值可由 T⁰ probe 增量预判"成立，度量从"选源"升级为
   "源库管理"工具。
4. **第二批目标定位**（probe 数据支持）：window=弱对价广度任务（全员≤3.1）；
   powerlift=比 door 更极端的 negctrl（全员≤0.9）。

### 修订后的实验队列（在 wfix 裁决 15 runs 之后）

1. **hurdle 增量实验**：maze/truck × obrw-4src(3loco+hurdle) vs 已有 obrw-3src
   ×3seed = 12 runs——单源增量、干净归因，检验"T⁰ 预判扩源收益"；
2. cabinet 不再做扩源实验（probe 已证无新对价，保持弱对价角色诚实呈现）；
3. window/powerlift 进第二批目标（用 wfix 裁决后胜出的方法跑）。

## 7. 广度表 3-seed 终裁（2026-07-05 凌晨，95k AUC，{scr,rand,obrw}×3seed）

| task | scr | rand(uniform) | obrw | obrw vs scr 配对 | obrw vs rand 配对 |
|------|-----|---------------|------|-----------------|-------------------|
| maze | 267.7±28.5 | 346.1±9.3 | 341.7±0.9 | **+74.0 (t=+3.63)** | −4.4 (t=−0.72) |
| truck | 1156.3±29.1 | 1296.6±33.2 | 1304.8±91.0 | **+148.5 (t=+3.25)** | +8.2 (t=+0.13) |
| cabinet | 121.0±27.5 | 210.9±3.6 | 219.6±3.6 | **+98.5 (t=+4.78)** | +8.7 (t=+1.82) |

### 判读

1. **广度主张成立（贡献 3 核心表成型）**：三个跨类别任务（导航/whole-body/
   manipulation）上 obrw vs scr 全部显著正迁移（t=3.25~4.78，9/9 seed 正）。
2. **诚实发现——vs rand 全部打平**：加权选源在这三个任务上无增益。机制解释：
   **选源增益取决于源库内部的好坏分化程度**——terrain 的权重高度不均
   （stair: stand 0.9 vs walk 22.0，20 倍）且含接近无用的源 → 加权把 1/3 的
   浪费预算集中到好源（+78~+182）；breadth 三任务权重仅 2-3 倍差且全员可用
   （probe 全正、fall 全 0，maze: 4.6/12.0/10.7）→ uniform 的浪费本来就小。
3. **三层收益拆解（论文叙事主干）**：
   - 第一层"注入 reward-bearing 数据"= 主收益（breadth +74~+148 全来自它）；
   - 第二层"选哪个源"只在源分化大时 matter（terrain 是、breadth 否）——
     且 T⁰ 权重的分布（如变异系数）**训练前就能预测这一点**；
   - 第三层"在线 vs 静态"只在执行伤害型任务 matter（slide 是唯一决定性证据）。
4. **对 wfix 裁决的预期更新**：breadth 上 rand≈obrw ⇒ wfix（复杂度介于两者）
   大概率同样打平；裁决的真正关键在 door/spoon——若 wfix 在无对价任务也无
   伤害，在线层的独特价值将只剩 slide 一例，主方法简化概率上升。

## 8. wfix 裁决终裁（2026-07-05 午，STAMP `20260704T152010Z`，5任务×3seed）——**主方法简化成立**

| task | scr | rand | wfix | obrw | wfix−obrw 配对 |
|------|-----|------|------|------|----------------|
| maze | 267.7±28.5 | 346.1±9.3 | **351.0±7.8** | 341.7±0.9 | +9.3 (t=+1.83) |
| truck | 1156.3±29.1 | 1296.6±33.2 | 1191.4±65.3 | 1304.8±91.0 | −113.4 (t=−1.03) |
| cabinet | 121.0±27.5 | 210.9±3.6 | **235.3±5.5** | 219.6±3.6 | **+15.7 (t=+11.44)** |
| door | 295.0±5.4 | — | 285.9±8.8 | 295.8±8.5 | −9.9 (t=−0.99) |
| spoon | 315.4±13.9 | — | 346.9±4.2 | **354.2±1.4** | −7.3 (t=−2.66) |

### 按预注册规则的裁决

判据"wfix 全线打平且 door/spoon 无伤害"**成立**：door 无显著伤害（wfix−scr
−9.1, t=−0.93；s1 的 −28.5 被 s2/s3 稀释）；spoon wfix 仍显著正迁移（+31.5,
t=+3.46）；例外微小且双向（cabinet wfix 显著赢 +15.7，spoon obrw 小幅赢 −7.3，
truck 方向偏 obrw 但 t=−1.03 不显著且双方方差都大）。

**9 任务全局账本（obrw vs wfix）**：obrw 唯一决定性净胜场 = slide（+92.1,
t=+14.7），另在spoon有小幅显著净胜（+7.3）；wfix显著胜场=cabinet(+15.7)；其余6任务
（maze/pole/crawl/stair/door/truck）统计打平。

### 方法章定稿（裁决产物）

1. **主方法 = RBO（Reward-weighted Bootstrap）**：T⁰ probe 排序 + softmax 加权
   注入 + h25 锁存。九任务与OBRW直接对账中，RBO在cabinet胜、OBRW在slide大胜且在spoon
   小胜、其余6个打平；在线收益高度集中而非普遍，支持把更简单的静态版作为默认算法、OBRW
   作为局部扩展。这不等于RBO九任务都优于scratch。
2. **在线层（student-as-arm + 对称 replay 降权）降级为安全扩展章节**，
   适用场景由数据划定：执行伤害型任务（slide 型，+92 决定性）；另有 door 上
   "自动关闭 vs 静态的轻微负向趋势（−9.9 不显著）"作为安全价值的弱证据。
3. **三层收益拆解为最终叙事**：①注入 reward-bearing 数据 = 多个breadth任务的主收益；
   ②选源加权只在源库分化大时有益（terrain 20 倍权重差 vs breadth 2-3 倍，
   T⁰ 分布训练前可预判）；③在线自适应只在执行伤害型任务必要（slide 唯一例）。
   每层适用条件都有正反例与预判指标——这就是"什么信号支撑什么粒度决策"贡献的
   完整落地。

## 9. hurdle 增量实验终裁（2026-07-05 午后，STAMP `20260705T113556Z`，wfix-4src vs wfix-3src × 3seed）

| task | wfix-4src(+hurdle) | wfix-3src | 配对 | per-seed Δ |
|------|--------------------|-----------|------|------------|
| truck | **1421.3±29.3** | 1191.4±65.3 | **+229.9 (t=+3.47)** | +307.5/+98.1/+283.9 |
| maze | 351.3±5.7 | 351.0±7.8 | +0.3 (t=+0.16) | +4.2/−0.8/−2.4 |

### 判读

1. **truck 大结果**：加入 hurdle 源（probe 13.6 略超 walk 12.8）带来 +19.3% 的
   显著提升，且 **wfix-4src 1421.3 成为 truck 全场最佳**（超 obrw-3src 的
   1304.8）——"主方法（静态）+ 更好的源库"胜过"在线机制 + 旧源库"，进一步支持
   方法简化定稿；同时方差减半（29.3 vs 65.3）。
2. **maze 未兑现**：probe 增量最大（hurdle 16.4 vs walk 11.3，+45%）却零转化。
   候选解释（诚实标注为事后假设）：maze 已饱和——所有 bootstrap 方法挤在
   340-355 平台（rand 346/obrw 342/wfix 351/wfix-4src 351），瓶颈已不在
   warmup 数据质量（导航探索本身），再好的源也推不动平台。
3. **扩源三部曲定稿**："T⁰ 增量是扩源收益的必要非充分条件"——
   (a) probe 无增量 → 必无收益（冗余源 7src 实验，三任务全平/负）；
   (b) probe 有增量 + 任务未饱和 → 大收益（truck +229.9）；
   (c) probe 有增量 + 任务已饱和 → 零收益（maze）。
   饱和判据候选：现有方法们的 AUC 聚拢度（maze 四方法带宽 <5% vs truck 带宽
   ~10%）——训练前可从已有 run 读出，闭环了"T⁰ 作为源库管理工具"的叙事。

## 10. 第二批扩展（2026-07-05，PI 定向：更多任务 + 源库扩大）

PI 裁定继续扩：目标任务加 basketball/bookshelf/balance 等，源库至少含
stand/walk/run/reach 四官方源，训练好的自训策略（hurdle/stair/slide 等）也入库。

- **标准 9 源库定稿**：4 官方（stand/walk/run/reach）+ 5 自训 terrain
  （hurdle/stair/slide/crawl/pole）。哲学：**库尽量大，选择交给 T⁰**——静态
  softmax 加权下低分源自动边缘化、无在线探索税（hurdle 增量实验已验证：truck
  上 stair probe 0.3 被自动边缘化）。reach 的跨布局 adapter =
  slice(proprio indices)+pad6（task 维置零），bank 生成器
  `scripts/build_std9_banks.py` 按 obs_dim/nq metadata 自动配置。
- **第二批 5 目标 = 对价递减谱系**（probe 定位）：bookshelf_simple 中对价
  （stair 源 7.78 意外居首）→ basketball/window 弱（≤2.7/3.1）→ powerlift
  极弱（≤0.9）→ balance_hard **全员 0.00**（完美极端 negctrl；wfix 权重全 0
  时 softmax 退化 uniform="无信号时不装懂"）。与第一批（maze/truck 强对价）
  互补，直接检验方法在对价递减下的安全性谱系。
- pilot：5 任务 × {scr, rand(std9), wfix(std9)} × s1 = 15 runs
  （STAMP `20260705T153732Z`），判读后 3-seed 补齐。

### 第二批 pilot 结果（2026-07-05 夜，1-seed，95k AUC）

| task | scr | rand | wfix | wfix−scr | rand−scr |
|------|-----|------|------|----------|----------|
| bookshelf_simple | 606.3 | 658.4 | 659.8 | +53.5 | +52.1 |
| basketball | 141.6 | 90.2 | 102.4 | **−39.2** | **−51.4** |
| window | 226.7 | 276.1 | 242.1 | +15.4 | +49.3 |
| powerlift | 175.8 | 249.6 | **255.8** | **+80.0** | +73.8 |
| balance_hard | 79.7 | 93.9 | 93.3 | +13.6 | +14.2 |

**判读**：

> 以下为pilot时的中间判断；最终3-seed裁决见本节后文。特别是“安全谱系成立”和
> “OBRW可能自动关闭basketball”均已被后续数据否证。

1. **basketball = 14 任务中首个"任何注入都负"案例**：rand（−51.4）比 wfix
   （−39.2）更差 → 归因为注入本身（投掷精细技能与 loco 数据冲突），非加权之过。
2. **powerlift = probe 假阴性大正例**：T⁰ 全员 ≤0.92（预测极端 negctrl），
   实际 +80.0（+45.5%）——50 步窗口读不到的价值。与 basketball（T⁰ 2.7 弱正
   却 −39）合并：**T⁰ <3 的低分区间无方向判别力，两侧反例俱在**（度量边界的
   干净刻画，替代"低分=negctrl"的旧假设）。
3. 安全谱系整体成立：4/5 任务正迁移或无伤害；balance_hard（权重全 0→uniform）
   +13.6 无伤害；bookshelf 中对价兑现 +53.5。
4. basketball 为在线安全层提供第二个（潜在硬）价值场景：若 obrw 自动关闭
   避免 −39，安全层证据从 door 弱信号升级为硬案例——已加入 3-seed 矩阵。

3-seed 补齐原计划（STAMP `20260705T224905Z`）：5 任务 × 3 方法 × s2/s3
+ **basketball × obrw × s1-s3**（裁决在线层安全价值）。2026-07-12 从本地 W&B
binary stream 完成离线审计，口径为严格 5k--95k 评估网格归一化 AUC；不依赖 W&B 网络 API。

### 第二批终裁（2026-07-12 本地审计）

数据质量先行：48个逻辑run slot全部可追溯。`balance_hard`的`rand-s3`和`wfix-s2`
首次尝试在训练前分配replay buffer时CUDA OOM；2026-07-12在空闲独占GPU上按原配置恢复成功。
因此审计账本为**50次attempt、2次历史失败、48个有效逻辑slot**，五个任务现在均为完整3-seed。
失败日志不删除，recovery run作为同一逻辑slot的有效替代。可复核产物：
[`artifacts/breadth_batch2_local_audit/report.md`](../artifacts/breadth_batch2_local_audit/report.md)
与 `analysis.json`；重算脚本为 `scripts/analyze_breadth_batch2_local.py`。

| task | scr | rand | wfix | obrw | wfix−scr 配对 |
|------|-----|------|------|------|----------------|
| bookshelf_simple | 679.4±67.4 | 654.7±19.9 | 711.6±50.1 | — | +32.1 (t=+1.13) |
| basketball | 188.9±70.6 | 116.0±23.7 | 87.4±20.6 | 114.8±21.0 | **−101.5 (t=−2.58, 3/3负)** |
| window | 269.0±41.8 | 206.4±71.0 | 237.2±58.1 | — | −31.9 (t=−0.99) |
| powerlift | 177.6±9.6 | 254.2±6.0 | **255.2±0.8** | — | **+77.6 (t=+14.78, 3/3正)** |
| balance_hard | 84.4±4.9 | 91.1±4.2 | 90.5±15.4 | — | +6.1 (t=+0.75) |

本表的 `±` 为本次离线审计重算的 sample SD；配对只使用同 seed。

**终裁修正**：

1. **powerlift 是强正例，也是 T⁰ 低分假阴性**：wfix 3/3 正、方差极低，尽管所有
   50-step probe 分数都小于 1。短 probe 看不到的身体组织/稳定数据仍可显著帮助长期学习。
2. **basketball 是硬负迁移边界**：wfix 与 obrw 均 3/3 低于 scratch。OBRW 只把
   wfix 的损失从 −101.5 缩到 −74.0，不能恢复 scratch；因此 student-as-arm + 对称 replay
   降权不是 universal safety guarantee，也不能升格为默认主方法。
3. **bookshelf_simple 仅为正趋势，window 为高方差负趋势**：两者都不能作为 headline
   正例。pilot 的单 seed 正负方向均不足以裁决。
4. **T⁰ 的可靠边界进一步收窄**：高分差可用于 source bank 内相对分配，但低分区既有
   powerlift 大正例，也有 basketball 大负例，不能承担 go/no-go 或 ROI 符号预测。
5. **主方法简化仍成立，但安全主张必须撤回**：静态 RBO/WFix 仍是默认算法；在线 OBRW
   只保留为双通道机制与 slide 型安全扩展，不能宣称自动避免任意负迁移。
6. **balance_hard完整终裁=null**：WFix−rand `−0.6,t=−0.09`，二者均相对scratch仅小幅
   正趋势。该task所有T⁰权重为0，softmax自然退化为uniform；结果验证“无区分信号时不应声称
   选源增益”，但不构成exact fallback，因为teacher exposure仍存在。

### 证据链完整性判定（成稿转入建议）

核心块现在均有完整3-seed逻辑slot；但审计必须保留balance_hard两次历史OOM及recovery provenance。
论文仍必须同时保留basketball硬负迁移、stair horizon limitation、crawl负迁移与T-gated/mh/SHU等负结果，
把贡献写成**有条件的有限预算迁移规律与机制**，而不是全任务安全保证。
