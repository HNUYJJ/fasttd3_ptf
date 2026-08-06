# Phase-1 机制现状审计

> 日期：2026-07-19
> 版本：**v2**（十四次复核 Needs revision 后修订）。
> **v1 的中心结论"per-source TTL 是唯一需要新增代码的缺口"错误**：v1 漏审了
> `AdmissionSchedule` 路径——固定时钟驱动的 per-source 撤销、到期硬退出与
> 延迟注入窗口**已可由通用 schedule 表达**（本版 §2 逐项核实）。v1 的其余
> 错误：未区分三种 TTL 语义（§1）；证据表混合工程正确性与性能有效性（§4）；
> "毒害载体是 critic 参数"越界（§5，已收窄回项目标准表述）。
> 依据：ChatGPT 十三次复核（P0 关闭）指示的机制现状审计；本审计不跑实验、
> 不写新代码。
> Phase-1 核心定位（冻结措辞）：**在没有可靠迁移性 oracle 时，通过可弃权、
> 有限剂量和经验生命周期控制限制负迁移；不声称能够准确识别有害 source**
> （P0 最终结论：局部 counterfactual delayed utility 在 3 seeds/L=3000/当前
> 评估噪声下不可稳定测量，Phase-2 oracle 封存）。

## 1. 三种"TTL"语义（必须区分，Phase-1 第一决策在此）

| # | 语义 | 定义 | 现状 |
|---|---|---|---|
| T1 | behavior authority lease | source 只在 [t_on, t_off) 内控制行为 | 可由 schedule 表达 |
| T2 | source-level replay eligibility | source 到期时其**全部历史轨迹**立即退出 active replay | 可由 scheduled revocation 表达（短 smoke 验证） |
| T3 | transition-age TTL | 每条 source transition 写入后只活跃固定年龄（如 3000 步），逐条退出 | **未实现** |

T2 与 T3 的原理、代码与科学问题完全不同：T2 的退出单位是 source（时钟到
→整体排除，复用 revocation 排除机制）；T3 的退出单位是 transition（需要
per-slot 年龄判定进入采样权重路径）。**Phase-1 统一规格必须先冻结采用
T2 还是 T3（或 T1+T2 组合），此决策未定前不写新代码、不展开实验矩阵。**

## 2. 既有能力核实（v1 遗漏的 schedule 路径，逐项代码验证）

1. **step 索引的准入决策序列**：`AdmissionSchedule`
   （admission_control.py:118-137）——decisions=((step, snapshot), …)，
   必须从 step 0 起，`snapshot_at(step)` 给出当前准入集合。可表达
   "0 admit A,B → 10k 撤 A → 20k 撤 B → 30k exact abstention"。
2. **变更的原子应用**：训练循环游标推进时同步调用
   `mcg_behavior.set_admission_policy`（行为）与 `rb.set_admission_policy`
   （replay 配额+排除）（train_ptf.py:1787-1812）；admission history 同录。
3. **撤销立即生效**：MCG `set_admitted_sources` 对被撤销源
   `current[revoked]=-1, steps_left=0`——锁存 segment 立即释放，不等
   25 步自然结束（fasttd3_ptf/ptf/mcg.py:440-446）。
4. **撤销后历史数据硬退出 active replay**：`set_admission_policy` rejected
   mass 强制零 + `_admission_allowed_slots` 按 provenance 精确排除
   （物理轨迹保留供审计）；main replay 入口断言 rejected 源 transition
   不得进入（train_ptf.py:2149）。
5. **延迟注入窗口**：schedule 模式初始为空**不会**进入不可逆的静态
   target-only fast path（train_ptf.py:1089-1096——只有
   `admission_schedule is None` 且不可变 exact-empty 才走 fast path），
   故"0 全 student → 10k 准入 → 13k 撤销"可表达。约束：窗口须与 MCG
   warmup/gate 阶段兼容（source 执行仅在 warmup authority 活跃期发生）。
6. 测试覆盖：`tests/test_admission_control.py`
   （`test_explicit_schedule_changes_from_source_to_exact_abstention`、
   adaptive 撤销族、`test_exact_abstention_cannot_be_revived_by_warmup_authority`）；
   runtime smoke（十四次复核**重新核查既有 smoke**，本轮未重新执行实验）：
   step 0 准入 stand→step 20 定时撤销→撤销后 execution 与 critic sample
   新增量均=0，物理轨迹保留。

## 3. 修正后的机制现状清单

- exact abstention：已实现（`admission_mode=none`；replay 侧与 scratch 同
  `randint` 原语 RNG 流逐位一致，ptf_replay.py:456-467）。**弃权决定由外部
  配置给出，系统不自主判断何时应弃权。**
- source provenance：已实现（8 字段 schema，ptf_replay.py:62-71；缺字段
  即拒；`assert_complete_provenance`）。
- 有限注入剂量：已实现（softmax 配额+float32 运行时断言；P0 交付
  0.098–0.103）。
- authority 结束后的 physical handoff：已实现并有因果证据
  （`physical_after_authority`，80k repetition divergence 修复）。
- scheduled per-source authority lease（T1）：已实现（§2.1-2.3）。
- source 到期后全历史数据硬退出 active replay（T2）：已可由 scheduled
  revocation 实现；短 smoke 通过。
- 任意固定注入起止窗口：已可表达（§2.5）。
- **transition-age TTL（T3）：未实现。**
- **自动判断 source 何时有益/有害：未解决**（P0 裁定该判据不可测）。
- **TTL/窗口对长期正负迁移的因果作用：未验证**（无同任务同剂量只变
  窗口/TTL 的受控长程实验）。

## 4. 证据表（工程正确性与科学/性能证据分列）

| 机制 | 工程正确性 | 科学/性能证据 |
|---|---|---|
| exact abstention | 强（P0 8 分支审计双零+CPU 逐位等价） | basketball 上回到 scratch 统计分布（十四次复核引证）；**不等于系统能自动判断何时弃权** |
| provenance | 强（round-trip+全链审计） | 未独立证明能提升性能（它是审计基础设施） |
| 剂量控制 | 强（10% 交付准确；E16 多通道审计带教训） | 未证明某剂量水平能普遍限制负迁移 |
| scheduled source 退出（T1/T2） | 短 smoke 强（撤销后双计数增量=0） | **尚无长程因果性能验证** |
| physical handoff | 强 | powerlift 因果证据（避免固定配额重复采样发散）；不等于 TTL 有效性证据 |
| 窗口位置 | 可配置（schedule 表达） | 无直接受控对比（历史 +227.8 从 0 注入 vs P0 中期负值为不同 estimand，非受控窗口对比） |

## 5. 效果边界的标准表述（v1 越界处修正）

撤销/退出机制的既知边界：**learner state（critic、actor、target、
optimizer）与后续 occupancy/数据分布均为负迁移的候选持久化通道，现有
证据未完成通道分离**；OBRW 相对 onlineb 的 +94.7 证明 replay exposure
控制重要且缓解不完整，不能单独证明 critic 参数是唯一载体，也不是
scheduled revocation 的直接效果证据。删数据不保证解毒。adaptive
revocation 的因果归因未成立（crawl 上不得引用）。

## 6. 下一步（十四次复核指定顺序）

1. 本审计 v2 交复核确认——**已确认（十五次复核"v2 基本通过"）并作出
   TTL 科学裁决：Phase-1 主机制冻结为 T1+T2 的 bank 级共享 lease
   （bounded bank lease），暂不采用 T3**。注意：不是 per-source TTL——
   现有效果证据主要是 bank/intervention 级，且无 oracle 时为不同 source
   设不同 TTL 缺乏依据；第一版所有 source 同时到期（错开到期会被
   softmax 重归一化抬高剩余源概率，破坏剂量解释）。
2. 统一规格见 `docs/phase1_bounded_bank_lease_spec_20260719.md`；
3. 最小验证=单因素因果实验（30k 后旧 source 数据命运：physical
   retention vs hard bank exit），不做"起点×长度×剂量"网格；
4. 工程注记：分析产物 provenance 今后记
   `base_git_head + generator_sha256 + dirty` 三元组。
