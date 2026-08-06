# 问题记录与教训（ISSUES AND LESSONS）

> 更新：2026-07-16。整合项目全程的工程事故、方法论教训与审计缺口，防止重蹈覆辙。
> 每条注明发生时间与出处；新问题请追加到对应小节。

---

## 1. 工程执行纪律（违反过、修复过的）

| # | 教训 | 事故与出处 |
|---|------|-----------|
| E1 | **长训练必须 tmux + `PYTHONUNBUFFERED=1` + `tee`**；裸后台（nohup/run_in_background）会因连接断开被 SIGHUP 杀 | reach 源训练 80k 被杀（2026-06-15） |
| E2 | **节点内存上限 ≈4 个训练进程**；瓶颈是 RAM 不是 GPU，>4 并发 OOM 会连坐杀掉核心实验；默认双并发 | breadth 首批 OOM（2026-06-15）；balance_hard 两 run OOM 重跑（2026-07-12） |
| E3 | **禁用 `pkill -f`**（模式会匹配到自身/无关进程）；用 `kill <PID>` 或 `tmux kill-session` | 自杀事故（2026-06） |
| E4 | **EXIT=0 ≠ 训练健康**：tee 管道可掩盖 Python Traceback；收尾必须看日志尾部与 eval 曲线 | 多次 |
| E5 | **改数据流后冒烟必须验证学习信号**：新路径的后处理须与主路径逐行对齐（normalize 链），冒烟须过 eval 点对比基线 | split 矩阵首轮因漏 `normalize_obs` 全废（2026-07-02） |
| E6 | **中断/截断后的工具输出不可信**：output-token-limit 下"已完成"叙述可能是幻觉；动手前 `git reflog` / 重新 Read 核实 | 两次幻觉事故（"5 commit 成功"为假；2026-06-15） |
| E7 | **wandb 断连后用 `wandb sync` 补传，勿重启训练**；启动时 `env -u http_proxy -u https_proxy` 清代理 | 2026-06/07 多次 |
| E8 | tmux 命令必须显式 `cd`（新 pane 的 cwd 不继承预期目录） | 2026-06 |
| E9 | **PTF 激活时 `torch.compile` 自动关闭**（动态 option 控制流不兼容）；scratch vs PTF 对比必须按 env steps 对齐 | 设计约束（长期有效） |
| E10 | **CUDA learner 跨进程非确定性是本底**：同配置同 seed 也非逐 bit counterfactual；机制归因必须靠干预设计与中介预测，不能靠单对 run 差值 | crawl s2 placebo −66.8（2026-07-15）；SIV gate v0 因非确定性作废重跑 |
| E11 | HumanoidBench legacy `seed` 只播 NumPy 全局 RNG，基座 reset 噪声走 Gymnasium `np_random`；已由项目本地 wrapper 修复双播种。**旧 P0/P1/P2 eval 的 episode-paired delta 不能直接进论文**（seeding 债务） | `humanoid_bench_env.py`；[archive/paper_core_contribution_reconstruction_v1.md](archive/paper_core_contribution_reconstruction_v1.md) §证据债务 |
| E12 | **跨任务 obs 适配必须显式 adapter**；维度合法但语义错误的隐式截断/补零会**静默**腐蚀蒸馏 | 设计约束（`ptf/adapters.py`） |
| E13 | 磁盘管理：中间 checkpoint 定期清理需 PI 批准；`checkpoints/sources/` 旧格式 ckpt 仍被 bank 引用**勿删**（加载依赖 `ptf/legacy_actors.py`） | 2026-07-06 清理 47.5GB；memory 长期条款 |

## 2. 方法论教训（负结果沉淀的规律）

| # | 教训 | 证据 |
|---|------|------|
| M1 | **行为 reward 信号不能驱动自动源准入/退出（三重否定）**：引导型好源执行段即时 reward 低（做"脏活"），与劣源不可区分。**勿做第四种行为信号变体**；重攻只能换信号族（learning progress / TD 统计 / T^critic） | SIV（07-11）、SHU（07-12）、adaptive revocation（07-15） |
| M2 | **behavior utility ≠ replay/update data utility**：source 的行为表现与其数据对 learner 更新的价值是两个 estimand | SHU 失败病理；crawl"排序学对但结果差" |
| M3 | **表征统一 ≠ 技能可迁移**：迁移瓶颈在源-目标技能重叠（L1）与表征可适配性（L2），不在池化方式（L3） | entity/anchored readout/EDSF 全 null（06-08 复盘） |
| M4 | **状态覆盖 ≠ 回报事件**：demo 能把状态送到位，但回报事件不进 buffer / credit assignment 不通，学习端仍然失败 | package chain warmup（06-12） |
| M5 | **critic-gated transfer 的安全性取决于 Δ 显著性而非符号**；早期 critic（<10k）不可信 | MCG v1→SC-MCG（06-11/12） |
| M6 | **跨任务 scalar ROI 预测 ill-posed**："scratch 卡住度"相关性是分母 confound | Transfer Map v1/v2（06-14） |
| M7 | **zero-shot 探针系统性低估 dynamic motor 源的 bootstrap 价值**（源类型不对称） | hurdle→stair（06-16） |
| M8 | **arm 空间扩张有结构性代价**：探索税+毒害扩大+统计代价（样本减半），即使方向判断全对 | mh horizon-arm 3-seed 否决（07-02）；breadth 扩源反向 |
| M9 | **fixed replay quota 在源退役期产生 repetition divergence**（物理残留 1.2% 拿 50% 配额 → oversample 43× → 崩点）；修复=authority-coupled physical handoff | 80k 崩点诊断与修复（07-13） |
| M10 | **retention 裁决要先查 headroom**：任务收敛后 wfix−scr 差距自然坍缩，不是机制失败。powerlift 不是 retention 场地，truck 是 | T0002 归因修正（07-13） |
| M11 | **弱信号下二值阈值脆弱，连续降权稳健** | T-gated abstain vs obrw（07-02） |
| M12 | **执行期 return 不可作在线选源信号**：return 与 hard progress 方向错位 | stability audit P2（07-08） |
| M13 | **选源增益取决于 bank 权重分化度**（terrain 20× 权重差 vs breadth 2-3×；T⁰ 权重 CV 训练前可预测） | breadth 3-seed（07-04） |
| M14 | **预注册纪律有效**：gate 字面执行 + 同批 same-launch 对照 + 干预因果设计 + 中介预测，是本项目所有可信结论的来源；负结果按预注册裁决如实写 | handoff 6/6（正例）；adaptive FAIL（负例） |
| M15 | **方案设计前必须先检索本项目的既有证据**（EXPERIMENT_LOG + archive 负结果）：本项目已积累大量"试过且失败"的判据/机制，新方案撞旧负结果是最容易犯也最伤公信力的错误；同理，声称 ground truth 标签时不得超过原实验的归因分辨率（bank 级证据 ≠ source 级标签） | 组件①方案 v1 的 critic-advantage 门撞 2026-07-02 T^critic 符号负偏结果、考古标签越权（ChatGPT 审查发现，2026-07-16） |

| M16 | **episode-level SE 不能代替 learner-seed 不确定性**：前者只证明单个 checkpoint 的评价均值稳定，后者才决定机制结论能否复现。Door 通道分解上，若按单 seed 的 128-episode SE 裁决，seed 1 会"证明"behavior 通道主导、seed 2 会"证明"replay 通道主导，**两个互相矛盾的结论各自都有 ~12σ 显著性**。同理"总效应显著而两分量各自不显著"只能裁 UNRESOLVED，**不得称纯交互**——那也正是功效不足的样子 | Door channel decomposition（07-28；PI 事前纠正我的判据建议，随后被数据直接验证） |
| M17 | **learner-path dependence：迁移效用的通道归因不是 (source, target, stage) 的稳定函数**。即使 source/target/stage/剂量/anchor/噪声种子全部固定，行为通道与 replay 通道的作用归因仍随 learner trajectory 翻转（s1/s3 行为致害 −59/−102 而 replay 补偿 +41/+68；s2 行为无害 −1.3 而 replay 致害 −40），且非评估噪声（episode 层面每 seed 内比值 10–20）。**推论**：迁移效用须写成分布 `U ~ p(U | source,target,θ_t,D_t,occupancy_t,channel,d,K)`；安全迁移必须处理效用不确定性，不能把 transferability 当固定属性 | Door channel decomposition（07-28） |
| M18 | **标签可测性必须先于标签解释**：投入源标定前先用无源臂数据判断标签在该 stage 能否被分辨（判据 `U/trend`，锚点 crawl 0.83 可测 / cabinet 10.31 不可测）。Cabinet@10k 整轮 12 臂因罕见事件主导（median 11–28 vs max 33–706）而无法判定，事后才诊断出不可测 | Cabinet gate UNCERTAIN → 零训练可识别性审计（07-27） |
| M19 | **行为即时效果 ≠ 延迟学习价值（同任务内直接证据）**：door-run zero-shot 行为 +58% 却 harmful，walk 行为 −61%（62% 摔）却是三源中最不负的。此前该论断只有跨任务证据（hurdle 全正/crawl 全负，行为量无法区分），Door 首次在固定 target/stage/剂量下取得直接反例。**后果：zero-shot 行为探针作为廉价迁移性指标关闭**，与 T⁰/T^critic sign/SIV/SHU/adaptive revocation/P0 lease/update-space influence 同族同因（全在测即时量） | Door@10k gate（07-27） |

补充工程条目：

| # | 教训 | 事故与出处 |
|---|------|-----------|
| E14 | **commit 前必须审查 `git status` 的暂存全集**：`git add -A` 会把工作树中来源不明的文件消失静默打包成删除；commit message 未披露的变更破坏审计链 | commit 8a2d441 混入四个未披露文件删除（ChatGPT 审计发现；81c8c9d 恢复，2026-07-16） |
| E15 | **同进程顺序两次运行也非逐位确定（E10 的扩展）**：第二次运行复用第一次建立的 CUDA 库内部状态（cublasLt 启发式/分配器布局），大 GEMM（critic 1024×101 atoms）选不同归约顺序；`cudnn.deterministic`、`CUBLAS_WORKSPACE_CONFIG`、fp32 均不免疫。**逐位等价对照只在 CPU 上可达**；GPU 只能做幅度对照（与"完全重复运行"的噪声地板比值判定），判定性证据=B-B 对照复现 A-B 失败模式 | P0 等价性测试 4 首跑 FAIL 诊断（2026-07-17；run card A.3 诊断记录） |
| E16 | **多通道 treatment 审计的每个通道验收带必须分别由机制推导校准（含暂态），不能把名义参数带统一套到下游累计观测上**：P0 把 exec 名义剂量带 `0.10±0.02` 同时用于累计 critic 采样占比，未建模冷启动暂态（分支从纯 student buffer 起步，空源桶配额被让渡，每有效源亏损 ≈h/L）与有效源数依赖——crawl(2 有效源,~0.085)压线过、truck(3 有效源,~0.078)压线挂，一次完整 P0 被判 `ENGINEERING_INVALID`，而分段占比末 750 步已收敛 0.096–0.100，与机制正常运行一致。带定错的代价=正式结论只能保持无效（事后修带改变证据地位，只能作 post hoc 敏感性分析） | P0 正式裁决（2026-07-18；`docs/p0_posthoc_engineering_sensitivity_20260718.md`） |
| E17 | **正式实验必须用干净、已提交的代码；确需脏树运行则必须保存完整源码快照，而非只存文件哈希**：`git_head` 在长期脏树下无锚定价值（admission 全族 meta 记 `git_head=b183f40`，实为 6-15 wfix commit，6-15→7-16 整月开发全在未提交工作树）。Phase-1 复用审计中 basketball 因 `implementation_sha256` 恰好逐位命中某 git 快照（a5cec9d）而 REUSE_PASS；**truck 则 REUSE_FAIL——train_ptf/mcg/admission_control 的 7-13 内容只有哈希、无内容，实验实现不可重构，0–30k 行为/学习路径可比性无法验证，一族现成 3-seed 基线因此不能进正式配对裁决**。今后每个正式 run 的 meta 必记：per-file implementation SHA 清单 + `base_git_head + dirty 状态`；关键实验直接在干净 commit 上跑 | Phase-1 兼容性审计（2026-07-19；`docs/phase1_reuse_compatibility_audit_20260719.md`） |
| E18 | **否定性断言（"字段/证据不存在"）必须穷尽检索所有可能载体后才能下，只查一处就断言"不存在"会污染下游决策**：2026-07-19 一天内犯两次——(1) 断言 `execution_counts_at_apply` "根本不存在"，实际只查了 `policy_events`（replay 侧），该字段在 `decision_history`（train_ptf 侧），差点让门 B 放弃更精确的取证源；(2) 断言 scratch "无任何 provenance"，实际只查了 `logs/train/`，未查 `wandb/`（494 个 run 档案含完整 config/metadata/**历史入口代码副本**/diff.patch），差点让一族可用的 δ 外部尺度被误废。**规则**：写"不存在/无"前，先列出所有可能存放位置（代码侧 vs 数据侧、目录 vs 归档、本地 vs W&B）并逐一验证；宁可写"在 X 中未找到"也不写"不存在" | Phase-1 二十/二十一次复核（2026-07-19） |

| M20 | **规范写进 docs 无效，必须有强制加载的执行点**：E3（禁用 `pkill -f`）、M15（先检索既有证据）、M16（learner 方差而非 episode SE）三条**都已在本文件中**，2026-07-30 当天仍被逐条违反。根因不是缺规范，是缺强制检查点。**修复：项目根 `CLAUDE.md`**，内容为"必须实际运行的命令"而非道理 | 2026-07-30 一日内违反 E3/M15×2/M16 |
| M21 | **grep 到一处用法 ≠ 读懂参数影响**：断言某配置有影响前必须读完整链（定义→传递→消费）并用运行时实际值验证。反例：由 `T_max=args.total_timesteps` 断言 LR 日程被压缩，实际 `eta_min == base_lr` 使余弦退火恒为常数，多跑一条 `grep learning_rate_end` 即可避免 | RACING_K 批间差异误判（07-30，已更正 `11ff656`） |
| M22 | **判决脚本的缺失数据分支必须是 `INCOMPLETE` 且非零退出**：绝不能让"评估没跑完"落进 `REFUTED`/`PASS` 分支——那会把工程未完成读成科学结论。同理缺失统计须独立扫描全部组合，不得在前置项缺失时 `continue` | RACING_K 裁决脚本自查（07-30，数据到位前修复） |
| M23 | **并行任务的输出文件集合必须两两不相交**：`[[ -f "$OUT" ]] && skip` 只在启动瞬间检查，不是原子锁。两组 `STEPS_LIST` 重叠导致同一 json 被两进程并发写（现象：log 为空、json 迟迟不出、同 ckpt 两个 evaluator） | hurdle_speedup_v1 评估（07-30） |
| M24 | **单批 n=3 的 3/3 不足以定论，需独立重复**：RACING_K 批1 在 K=5000 显示 run 领先 8.4–14.8 个 **episode**-SE（看似无可争议），独立重复仅 1/3；按 learner 间方差 `t=1.57` 不显著。同批的 `progress_dx` 观察项（批1 3/3 → 批2 0/3）是同一教训第二例。**M16 的实操推论：效应量接近 learner SE 时必须重复** | RACING_K v1（07-30，`a744adb`） |

| M25 | **"正确答案不同" ≠ "排序反转"**：真正的 crossover 必须是**同一候选集合**上赢家反转；候选集合不同时，一个**全局固定排序**即可解释全部结果，辨别力为零。设计判决场时必须写出至少一个平凡解释（含"选 |U| 最大者"这类捷径）并说明如何排除 | RACING_MULTI 作废（07-30，未执行；Codex 决策前 review 判 FATAL） |
| M26 | **与处理共变的剂量差是混淆，放宽容差不是解决方案**：sibling 臂 behavior share 系统性高 2.4–3.3pp 且与效应同向，则"源更有用"与"源被用得更多"不可区分。正确做法是按步/配额强制匹配实际剂量并在该控制器下重建参照，或把 estimand 显式改写为"源+控制器"整体 | 同上（我识别出该差异却只沿用宽容差，等于把混淆写进协议） |

| M27 | **per-seed `U` 标签的 run-to-run 漂移与效应量同量级，此前从未被刻画**：同 `(source,target,stage,dose,anchor,noise seed)` 重跑 door，`|ΔU|` 中位 **24.23**、最大 **43.78**，而这些标签的效应量本身只有 −7~−43；hurdle 同 seed 两批亦有约 15。**后果**：(a) 符号/排序可用，per-seed 数值不可当作可复现真值（`EQD30K`/`sibling gate`/`door gate` 的点值均只有单次运行支撑）；(b) 凡"与已发表值比对"的复制检查，容差必须基于 run-to-run 漂移而非评估噪声，否则系统性误杀；(c) 为 M17 补上"同协议重跑"这一层证据 | RACING_REJECT v2 揭盲（07-31，`REPLICATION_DIVERGED`） |
| M28 | **核实必须到"本场景实际值"这一层**：外部 review 建议校验 `source_names == [arm,"null"]` 并引用了 `source_bank.py` 的保存逻辑；我核实了那段代码却没核实它在 door 上的实际输出（door 的 bank 是 `null_option:false` → `['stand']`，hurdle 才是 `['run','null']`），导致首轮裁决 54 条假缺陷。更难看的是我在数小时前的剂量验收里**自己打印过** `names=['stand']` | RACING_REJECT v2 首轮 `VOID_ENGINEERING`（07-31，R8 已修） |

| M29 | **自查抓住的假模式（记录一次未遂）**：我一度归纳出"racing 决策的可重复性 ≈ 源间间隔 / 不确定性尺度"，并列了 5 个事后数据点。按 `CLAUDE.md` §8.1 逐条检验时否掉了自己：(a) **不单调**——比值 1.25 可重复而 1.57 不可重复；(b) **量纲不统一**——排序判据的"间隔"是两源之差，符号判据的"间隔"是距零点距离，二者被我放进同一张比值表；(c) 5 个事后点不足以支持任何模式。**该观察未被写成假设，未投入实验** | 2026-07-31，RACING_REJECT v2 揭盲后的探索 |

| M30 | **写下教训 ≠ 内化教训：86 秒内重演了自己刚否决的推理**。`dd63187`(00:51:43) 记 M29 否决"可重复性 ≈ 源间间隔/不确定性尺度"；其直接子提交 `01dd93e`(00:53:09) 的 v3 §3.3 又用"源间差与漂移同量级 → 排序本就不该指望稳健"。**修复**：`CLAUDE.md` §8 增设"写预注册前先读本轮新增的 M 条目，逐条检查是否重演"。另一并记：**outcome-contingent gate switching**——在同一数据上把已知失败的门换成已知通过的门，即使主终点数据仍盲、即使如实披露，也不能恢复确认性地位；合法替代是 split-sample（旧 seeds 降级为 design data，未揭盲批作 holdout，另加新 seeds） | RACING_REJECT v3 撤回（07-31，未执行；Codex 裁定不合法） |

| M31 | **标签可推广性：U 的符号本身可跨 learner 显著反转**。door 上 gate(`s1-3`) 与 holdout(`s4-6`) 共 18/18 per-seed 为负，而新批 `s7-9` 出现 2/9 为正，其中 `s9` 的 `run = +36.32 ± 3.95` **显著正**（gate 对 run 的结论是 −30.63，跨度 67）。**后果**：(a) `door_at10k_gate_v1` 的"三源一致有害"须限制为"在 seeds 1–6 上"，那是 learner 子总体的性质而非 target 的性质；(b) `M18` 的"标签可测性"审计不够，还需**可推广性**审计——现有全部 A 级标签都只在 3 个 learner 上测过且从未做过跨 learner 符号稳定性检验；(c) 这是 `M17` 迄今最强证据：不是效应大小不稳，是**有害/有益的方向不稳**；(d) door 因此**不是合适的判决场**，racing 拒绝能力的检验须另找 `U` 符号稳定的全负 target | RACING_REJECT v4（07-31，`PARTICIPANT_DIVERGED`，预注册分支） |

| M32 | **源间差消去 student 基线漂移——这是 racing 决策稳健的机制原因**：slide 上换一批 learner 后，三个源的 `U` 同步下移 `−12.6~−14.0`（因 `U_i = J_i − J_student` 共用基线），但 **argmax 与次优的间隔几乎不变**（40.05 → 41.45，差 1.4）。故 racing 的决策量（源间差）天然比绝对 `U` 稳健。**推论**：判断某 target 能否用 racing，应看**源间差 vs run-to-run 漂移**，而非绝对 `U` 的稳定性。三个 target 方向一致（hurdle 间隔~275、slide ~40 → argmax 各 6/6 稳定；door ~8.4 < 漂移 24 → 不稳），但**仅 3 个事后点，按 M29/M30 只作待前瞻检验的假设，不得写成判据** | slide 可推广性审计（07-31，`GEN_OK`）；机制判断源自 Codex v3 review |
| M33 | **降低用法强度救不了已失败的信号空间；而"是否存在任何阈值"应先于任何阈值取法去验算**。族 12 把要求降到最弱——只做单向排除（不排序）、把测量量从 `return` 换成 `task progress`——两处降级都没用：crawl 上有害的 `run` 位移 `14.302`，slide 上有用的 `walk` 位移 `1.814`，**反向 7.9×**，故阈值须满足 `14.302 < θ < 1.814` = **空集**。**方法论后果**：这个空集论证只用两个数、与阈值取法无关，比预注册的 `HOLDOUT_FAILED` 强得多；设计任何阈值型判据时，都应先检查"目标区间是否非空"，而不是先设计取法。**附带的执行层教训**：初稿把"run 穿过了大半条隧道 / 摔倒后翻滚"写成肯定句，实际**同批探针已采集 `in_tunnel` 却没看**——`run` 有 58% 时间步在隧道横向范围之外（`in_tunnel` 均值 0.4175，硬门控），是绕开任务而非穿过。**采了数据却用推理代替查询**，是 M21 的新变体 | 族 12 `progress_screen_v1`（08-04，`HOLDOUT_FAILED`）；机制更正由 PI 质疑触发（`5ae3d3b`）|

## 3. 审计缺口与修复记录

| 缺口 | 状态 |
|------|------|
| admission_history 持久化：撤销链与窗口统计初版只能从 replay `policy_events` 重建 | **已修复**：checkpoint `admission_audit.decision_history` 完整持久化（含每窗 LCB/UCB/persistence），finalizer 可离线复算 |
| 旧 stability eval 的 env seeding 未正确播种（E11） | wrapper 已修复；历史 episode-paired 数字标记为不可直接引用 |
| W&B 曲线在磁盘清理后是部分历史实验的唯一数据源 | backfill 组 `admission_core_v1_20260712TFINALV2Z_backfill`；本地 log `[eval]` 行为兜底 |
| 2026-07-16 代码整理删除了死线代码 | 全量恢复点：git 快照 `a5cec9d`（删除前）；死线结论存档于 `docs/archive/` |

## 4. 实验启动 checklist（沉淀为流程）

1. run card → PI 批准（高成本实验）；预注册 config 写入 `configs/experiments/` 并 SHA256 冻结；
2. tmux + `PYTHONUNBUFFERED=1` + `tee`，显式 `cd`，清代理，**wandb 在线**（PI 通过 dashboard 监控）；
3. 默认双并发；补跑走同样的并发+内存守护；
4. 每 run 落 meta（bank/protocol/implementation SHA）；
5. 收尾：exit code + 日志尾 Traceback + eval 曲线 + checkpoint `global_step` 四重确认；
6. 裁决：按预注册 gate 字面执行，超出 gate 的解读明确标注为探索性；
7. **决策前与实现后各交 Codex review 一次**（PI 要求 2026-07-30）；其引用的证据逐条独立核实后再接受；
8. **动手前先读项目根 `CLAUDE.md` 并实际执行其中的检查命令**（M20）。
