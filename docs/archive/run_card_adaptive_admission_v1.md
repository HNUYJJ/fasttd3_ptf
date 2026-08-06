# Run Card：adaptive_admission_v1（自适应源撤销 + 外部迁移 baseline）

> 状态：**Phase A 已获 PI 批准执行（2026-07-14）；Phase B 暂缓**——PI 指示"暂时先不要跑 baseline，先把我们自己的工作实验效果跑出来"。Phase B 的实现与训练均暂停，待 Phase A 裁决后由 PI 另行决定。
> （v1 草案 Claude 2026-07-14；ChatGPT T0008 对抗审查五点修订全部采纳后定稿 v2；实现前 ChatGPT 做可行性核对。）
> 前置批准：PI 已批方向（基座固定 FastTD3，不做任何第二 backbone 实验）。
> Phase A 预算：18 runs × 100k；默认 2 并发 ≈ 18h 墙钟；实现+单测+冒烟 0.5-1 天。

---

## Phase A：Adaptive Behavioral-Source Revocation（18 runs，优先）

### A.1 假设与 claim 边界（预先声明）

**假设**：用"同期随机化 segment 干预统计"的保守单向撤销判据，可以在训练中自动驱逐行为层有害的源（截断负迁移），且不误伤正迁移源。

**claim 边界（T0008 对齐后收紧）**：本机制估计的是 **behavior-authority utility**（该源现在执行是否比学生差），不是 replay learning utility，不是完整迁移性指标；机制名固定为 *adaptive behavioral-source revocation*；对导师意见③只主张"时间维、行为通道的部分兑现"。basketball 型任务（源凭存活 reward 高于弱探索学生、但把数据分布锁死在错误区域）是该判据的已知边界，如实预登记。

### A.2 机制（预注册公式，v2 = segment 级置信判据）

现有 admission_bootstrap 的调度本身就是"128 个并行 env 在 segment 边界对候选（admitted 源 + student）做随机化分配"——这天然构成 mixed occupancy 上持续运行的随机化短干预实验。据此：

1. **结算单位 = segment**：每个 segment 结束（25 步到期或 done 截断）时，在 `envs.step` 之后结算该 segment 的 per-step mean reward（对长度稳健），归属到执行它的候选；`steps_left==0` 或本步 done 时闭合一次，避免下一轮调度重置造成丢失或重复结算。
2. **Stage window 统计（T0011 修正：非全历史累计）**：以固定、互不重叠的 `stage_window_steps=3000`（vector steps）为统计窗口；窗口内 per-candidate 用 Welford 汇总已完成 segment 的 per-step mean reward，**窗口结束后统计全部清空**——比较的是"该源在当前 student stage 是否仍值得介入"，而非从 step 0 至今的历史平均。全部 CPU、零额外交互、零额外 RNG。
3. **撤销判据（单向置信比较，窗口粒度）**：窗口结束时，若源 i 与 student 在**本窗口内**的 segment count 都 ≥ `min_segments=20`，该源获得一次判定：`UCB(source_i) < LCB(student)`（`UCB/LCB = mean ± z·se`，`z=1.645`，语义为 normal-approximate 单侧 95% 界，**不主张**有限样本/序贯多重检验下的严格保证）。证据不足的窗口按 false 处理**并清零该源的 persistence**，禁止跨窗口携带 stale vote。
4. **Persistence 以窗口为单位**：连续 3 个完整窗口判定成立才撤销 → 理论最早撤销点 = 9k（3×3000），与原设计意图（约 10k 前不动手）一致。每源每窗口至多一票——杜绝 h25 同步结算批次在单个 vector step 内重复投票。
5. **多源同窗触发的原子性**：同一窗口判定出的全部撤销合并为**一个** immutable snapshot，由 train loop 一次性原子应用，不逐源更新候选分布。
6. **撤销执行**（全部复用已验证组件）：`AdaptiveAdmissionController`（admission_control.py 纯状态机，输入窗口统计、输出 immutable decision + event 快照）产出决策；train loop 原子完成四件事——更新 `admission_snapshot` 本身（rollout 端 `exact_abstain` 读它）、`mcg_behavior.set_admission_policy`（latch 立即释放）、`rb.set_admission_policy`（active mass 归零）、若全撤则立即 `set_admission_source_authority(False)` 且**后续固定 warmup authority 同步逻辑必须尊重 adaptive exact-abstain，不得复活 authority**。撤销不可逆。
7. **warmup 上限 30k 不变**（只做提前退出，保持与全部历史对照的总预算可比）。
8. **明确不做**：在线 logit 重排序 / per-source 效用打分 / 阈值挽救（SIV/SHU 禁令继续有效）。

新增参数 4 个且全部预注册冻结：`stage_window_steps=3000`（=10 窗/warmup，锚定"最早撤销 9k"的原设计意图）、`z=1.645`、`min_segments=20`（每窗口，主要源每窗远超此数、近零份额源如 truck-stand 永不达标 → 永不撤，保守语义）、`persistence=3` 窗。

### A.3 实验矩阵（18 runs）

**adaptive-on**（4 任务 × 3 seeds = 12 runs）+ **补静态对照**（crawl/basketball 的 admission-all + handoff、adaptive-off × 3 seeds = 6 runs；powerlift/truck 的该对照即 20260713THANDOFFV1Z 的 fix runs，直接复用）：

| 任务 | bank | 角色 | 主对照（单变量：仅差 adaptive 开关） | 预注册预期 |
|---|---|---|---|---|
| crawl | loco 3 源 wfix h25 | **主收益场** | crawl admission-all+handoff（本批新增） | 源 segment return 低于学生（T-gated 预演方向），预期撤销发生、伤害截断 |
| truck | hurdle4 wfix | **负控制** | truck fix（复用） | hurdle/walk/run 不被撤，+270 量级保持 |
| powerlift | std9 wfix | **保持 + 选择性** | powerlift fix（复用） | 整体保持；观察低份额源是否被选择性撤销 |
| basketball | std9 wfix | **压力测试** | basketball admission-all+handoff（本批新增） | **如实预登记：判据可能不触发**（存活 reward 掩护），触发与否都是判据边界信息 |

legacy WFix / scratch 曲线作次级参照（非因果对照）。

### A.4 预注册 gate（统计量窗口全部显式；"evaluation-grid mean return" = 5k 评估网格 10k–95k 共 18 点的均值，与本机制的 behavior segment return 是两种量，不得混淆）

1. **crawl 收益 gate**：adaptive − 静态对照的 10k–95k evaluation-grid mean return ≥ +30 且 3/3 seed 为正，且 audit 存在撤销事件（时点入档）。
2. **truck 无伤害 gate**：|adaptive − fix| 10k–95k evaluation-grid mean return ≤ 60，且 hurdle/walk/run 无一被撤。
3. **powerlift 保持 gate**：adaptive ≥ fix − 20（10k–95k evaluation-grid mean return）。
4. **basketball**：描述性（不设 PASS/FAIL）——判据是否触发、时点、触发后 10k–95k evaluation-grid mean return 相对静态对照的变化。
5. **机制 audit**：撤销事件带统计快照（mean/var/count/置信界/step/execution counts/replay sample counts）；被撤源 execution / active replay mass / critic 增量采样立即归零。
6. **回归 gate**：判据从未触发的 run 与静态对照同 seed 的候选选择轨迹逐位一致（记账与判定不消耗 RNG；触发后 RNG 分叉为预期行为并明确标注）。

### A.5 单测清单（发车前全绿，T0008 §Q3 八条 + T0011 补充）

no-trigger 双控制器同 seed 逐步一致（selection/current/current_arm/steps_left/actions/generator state）；segment 归属边界（done/truncation/重置/h25 结算，`envs.step` 后闭合、无丢失无重复）；**stage window 边界**（窗口清零、证据不足清零 persistence、每源每窗至多一票、h25 同步批不重复投票）；synthetic trigger 精确撤销+latch 释放；**多源同窗撤销的单 snapshot 原子应用**（含 `admission_snapshot` 对象本身更新）；behavior/replay 双侧 mask 原子一致+三通道归零；all-revoked → exact student + 立即 authority release + **固定 warmup authority 同步逻辑不复活**；低 count 源不得基于 stale 统计撤销；decision event 完整快照（窗口统计/count/置信界/step/execution 与 replay sample counts）；未触发 integration smoke 与静态对照同 trace。

### A.6 风险与止损

- basketball 判据被骗 → 如实报告为判据边界，不调阈值；
- truck 误撤 → Phase A 判 FAIL，机制降级为"crawl 型任务的可选安全层"；
- 一切按预注册字面裁决，禁止事后改窗口/阈值。

---

## Phase B：外部迁移 baseline（36 runs）——**暂缓（PI 2026-07-14 指示），实现与训练均不启动，仅存档设计**

### B.1 方法（命名忠实性按 T0008 修订）

| 方法（主表名） | 实现 | 备注 |
|---|---|---|
| **JSRL (single-guide, curriculum)** | 新 `warmup_mode=jsrl`（不复用 chain）：guide = 各任务 T⁰ 最高单源；每 episode 前 h 步 guide 执行后交学生；h 线性退火（h_max=500 → 0 @30k，与 RBO 共用 30k warmup 窗口） | 忠实于原版 single guide policy；curriculum 形式在实现对齐时冻结一种 |
| **PTF distillation** | legacy PTF 路径（mcg=false、execute_sources=false、λ(t)(1−β_o) 加权蒸馏、λ 沿用历史默认退火）——这是我们框架的原方法，最应比较的蒸馏类对照 | 不标注为 canonical Kickstarting（多了 option selector/β gate）；发车前真实 smoke（路径长期未跑，防 bit rot） |
| **Best-single-source** | admission_mode=static + admitted_sources = T⁰ 最高单源 | 零改动；检验多源混合净价值 |

（v1 草案的"Kickstarting-style"取消：PTF distillation 已是同家族中最相关的对照，另实现一个 canonical 版边际价值低。FastDSAC 按"同 benchmark non-transfer reference + 互补探索层级"写入 related work，不进主表、不跑其代码。）

### B.2 矩阵与解读框架

同 4 任务 × 3 方法 × 3 seeds = 36 runs。主表 = {scr, RBO(wfix), admission+handoff, adaptive(Phase A), JSRL, PTF-distill, best-single}，前四列复用已有数据。预登记解读框架（不设硬 gate）：truck 预期数据注入类 ≥ 蒸馏类；basketball 预期所有注入类 ≤ scratch（负迁移的 regime 普遍性）；best-single vs RBO 检验 bank 分化规律；反预期结果如实进 regime map。

---

## 工程约束

- 并发默认 **2**；升 4-slot 须先过吞吐健康检查（sps ≥ 2-slot 基线的 80% 且 avail RAM ≥ 60GB 硬门控；本节点 320 逻辑核，4×128 env workers 存在 CPU 超订阅风险——T0008 实测提示）。
- 成本按 ~2h/run 保守估计（近期 formal 实测 1h40–2h20）。
- tmux + PYTHONUNBUFFERED + tee + 显式 cd；W&B 在线；实现文件 SHA256 冻结；每 run verify；全批 adjudication。
- 分工：ChatGPT 实现+单测+Phase A no-trigger 回归 smoke（Phase B 相关 smoke 随暂缓决策一并取消），Claude 只读复审后发车；Phase A 完整裁决并汇报 PI 后，Phase B 是否启动由 PI 另行决定。
