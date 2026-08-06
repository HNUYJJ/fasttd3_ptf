# Phase-1 Bounded Bank Lease 最终结果

> 日期：2026-07-20  
> 冻结协议：`docs/run_card_phase1_bounded_bank_lease.md` v0.6  
> 正式矩阵：`20260719T130500Z`，9 × 100k，seeds 1–3  
> 训练实现：git `e0c9d4b07df8e8fd7f519d32ee78de55fa8fa158`  
> 机器可读结果：`docs/data/p1_bounded_bank_lease/final_result.json`

## 1. 一句话结论

**T2 hard-exit 的工程语义完全成立，但在 basketball 与 truck 上均没有形成
方向稳定的性能效果。** 两个任务的冻结主判序都是 `HETEROGENEOUS`：同一
机制在 seed 2 有实质改善、在 seed 3 有实质伤害。因此，bounded bank lease
可以作为“限制后续 source 暴露、保证旧数据退出 active replay”的安全/正确性
机制，但不能声称能稳定缓解负迁移或提升性能。

## 2. 实验到底比较什么

两臂在 0–30k 完全共享约 0.5 的 source 剂量与相同 source bank；30k 后行为
权限都关闭。唯一干预是旧 source 数据的 replay 生命周期：

- retention：`B=0, R=physical tail`，旧 source 数据按 ring buffer 物理占比
  自然衰减，约 81.2k 后耗尽；
- hard-exit：`B=0, R=0`，30k 时 source provenance 立即退出 active replay，
  物理轨迹只保留作审计。

主 estimand 为每 seed 的
`Δ_exit = nAUC_35k–80k(hard-exit) − nAUC_35k–80k(retention)`。

## 3. 工程有效性

- 9/9 新训练均正常完成，退出码全为 0；每条均有
  30k/60k/80k/90k/100k completed-step checkpoint 与 final checkpoint。
- Gate A PASS：basketball 的历史 `admission_mode=all` 与 HEAD 的
  `schedule step0 admit-all` 在 30k 前的 admission、MCG 行为、provenance、
  actor/critic replay 抽样 trace 等价。
- hard-exit 6/6 runs：30k 后 source behavior 增量严格为 0、source critic
  采样增量严格为 0、active source slots 严格为 0。
- truck retention 3/3 runs：所有 checkpoint `active==main`，source physical
  share 单调衰减，区间 critic share 全部落在预冻结端点包络内，90k/100k
  source share 为 0。
- 0–30k behavior/critic source share 全部落在预冻结剂量风险带内。

因此性能结果可解释为有效的 `R=0` 对 `R=physical tail` 干预，而不是配置或
执行失败。

## 4. 冻结主结果：35k–80k nAUC

| 任务 | hard-exit nAUC（s1/s2/s3） | retention nAUC（s1/s2/s3） | Δ_exit（s1/s2/s3） | 均值 | 90% paired CI | SESOI δ | 判序 |
|---|---|---|---|---:|---|---:|---|
| basketball | 82.958 / 117.239 / 87.712 | 81.083 / 63.587 / 158.457 | +1.875 / +53.652 / −70.745 | −5.073 | [−73.101, +62.956] | 45.786 | **HETEROGENEOUS** |
| truck | 1566.741 / 1536.432 / 1479.265 | 1575.831 / 1486.636 / 1591.940 | −9.091 / +49.796 / −112.674 | −23.990 | [−113.535, +65.556] | 36.518 | **HETEROGENEOUS** |

冻结定义中，`HETEROGENEOUS` 要求至少一个 seed `d_s>δ` 且另一个
`d_s<−δ`。两个任务都由 seed 2（改善）和 seed 3（伤害）触发，而 seed 1
接近零。故不能用均值接近零宣称 `EQUIVALENT`，也不能宣称总体改善或总体
伤害。

## 5. 次级结果

80k–95k persistence：

- basketball：均值 −39.157，90% CI [−132.470, +54.157]，`UNCERTAIN`；
- truck：均值 −24.434，90% CI [−123.694, +74.825]，
  `HETEROGENEOUS`。

95k endpoint 的跨 seed 均值差很小：basketball hard−retention = −3.14，
truck = +11.06。它们不改变主判序。

## 6. 与 scratch 的描述性背景（不进因果主判序）

历史 scratch 只具有 `METRIC_SCALE_REUSE_PASS`，不具有严格
`CAUSAL_COMPARATOR_REUSE_PASS`，因此以下只作背景：

- basketball hard-exit − scratch：均值 −127.373，90% CI
  [−207.014, −47.731]；描述上仍明显低于 scratch。30k 后清除旧 source
  数据没有恢复 scratch，负迁移不能归因于持续 replay 拖尾这一条通道。
- truck hard-exit − scratch：均值 +292.433，90% CI
  [+219.613, +365.252]；retention − scratch：+316.423，90% CI
  [+260.888, +371.958]。两臂都保留了明显的早期迁移收益，说明 truck 的正
  收益主要不依赖 30k 后继续保留旧 source replay tail。

## 7. 科学解释与论文边界

1. **支持的贡献**：exact abstention、source provenance、有限 source 剂量、
   authority/replay 双通道解耦、撤销/到期后旧 source 数据严格退出 active
   replay。这些机制可验证地限制未来暴露。
2. **不支持的贡献**：固定 30k T2 hard-exit 不是稳定的性能提升器，也不能
   保证恢复 scratch 或避免负迁移。
3. **机制洞见**：basketball 在 hard-exit 后仍低于 scratch，而 truck 的正
   收益在两臂都保留，说明关键影响主要已在 0–30k 的 learner state、occupancy、
   early data coverage 或其组合中形成；30k 后的物理 replay tail 不是这两个
   场地上稳定的主要因果杠杆。
4. **跨任务一致的 seed 模式**：两个任务均为 seed 2 改善、seed 3 恶化、
   seed 1 近零，提示该干预的效果高度依赖学习轨迹/初始化，而非形成稳定的
   task-level 机制效应；这是来源于结果的诊断性推断，不是已分离的因果结论。

## 8. 处置建议

- 关闭本轮 Phase-1，不做 TTL/ε 小数级追调或同矩阵补跑；预注册问题已经回答。
- 论文中把 replay lifecycle 写成**风险控制与语义正确性机制**，主张“切断后续
  暴露”，不要写成“自动避免负迁移”或“提高回报”。
- 若继续追求性能，干预点必须前移到 source admission/早期窗口/剂量，而不是
  继续优化 30k 后旧数据清理；但 P0 已表明当前 delayed-utility oracle 不稳定，
  因此在出现新的、低成本且可验证的早期准入信号前，不建议立刻启动大规模网格。

## 9. 复现

```bash
python scripts/analysis/analyze_phase1_bounded_bank_lease.py \
  --root /home/yjj/fasttd3_ptf \
  --out /tmp/phase1_bounded_bank_lease_result.json
```

分析程序强制验证 5k–95k 的 19 点 eval 网格、9 条新跑的五个机制 checkpoint、
Gate B、冻结 SESOI 和冻结历史 basketball retention 证据。
