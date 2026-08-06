# Adaptive Admission v1：Codex 独立复核

> 复核时间：2026-07-15 UTC  
> stamp：`20260714T110054Z`  
> 复核顺序：先从冻结协议、训练日志、final checkpoints 与机器可读 analysis 独立复算，再读取 Claude T0017 与结果文档。

## 1. 总结性裁决

**工程证据 Ready to share；科学方法 Needs revision。** 18/18 训练完成且 W&B 为 `finished`，18/18 final checkpoint 生命周期审计通过，完整性 finalizer 为 PASS；但预注册科学状态为 **FAIL**。失败不在 exact revoke/handoff 的执行正确性，而在用“source 自己 25-step segment 的即时 reward”代理“该 source 对后续 student learning 的价值”。

## 2. 独立数字复算

| task | per-seed 10k–95k mean Δ | mean ± seed SD | 预注册裁决 |
|---|---|---:|---|
| crawl | `+41.493 / −66.786 / +53.892` | `+9.533 ± 66.384` | FAIL：未达 +30，且非 3/3 positive |
| truck | `−6.034 / −119.693 / −204.887` | `−110.204 ± 99.766` | FAIL：超过 ±60 且出现 forbidden revocations |
| powerlift | `−4.747 / −1.980 / −4.898` | `−3.875 ± 1.643` | PASS：全部高于 −20 保持线 |
| basketball | `−23.695 / +36.207 / −34.661` | `−7.383 ± 38.146` | descriptive：无系统改善 |

这里的 18 个 evaluation points 是同一 learner seed 的相关时间点，不能当成 18 个独立样本。表中的 SD 只跨 3 个 learner seeds；预注册 gate 是幅度/方向门，不依赖把时间点伪装成独立重复。

## 3. 机制正确性与因果边界

支持的机制结论：

1. 被撤 source 的 behavior execution、active replay、effective replay mass 与后续 critic exposure 均严格冻结/归零；actor 使用 shared critic batch，没有额外 actor sampling。
2. 12 个 adaptive 与 6 个 static final checkpoints 全部审计通过；crawl_s2 的四个 checkpoint provenance strata 与 static 对照完全一致，no-trigger 回归通过。
3. exact abstention、原子多源撤销、quarantine/admission-consistent replay、authority-coupled handoff 仍是可保留的 provenance lifecycle 贡献。

不支持的科学结论：

1. crawl 不能证明撤销造成收益。2/3 triggered seeds 的事件对齐改善只是探索性信号；no-trigger s2 在机制计数完全一致时仍有 `−66.786` placebo AUC，说明 CUDA 跨进程 learner 分叉足以污染单 seed 归因。
2. truck 证明的是规则破坏了一个已知整体正迁移的 source bank，而不是 hurdle/walk/run 每个单独都已被证明有益。现有 bank-level 训练不能做 individual-source attribution。
3. powerlift 只证明选择性删源与性能兼容；没有单源 learning-utility ground truth，不能声称已正确识别“无价值源”。
4. basketball 没有单源因果标签，不能把负迁移主体归于 locomotion source。大量撤销但无稳定改善，只能说明即时 reward 排序没有找到可复现的修复。

## 4. 最重要的新机制洞见

truck 的撤销发生于 12k–21k；30k 后 adaptive 与 fix 都已退出 source behavior authority。尽管如此，adaptive−fix 从 10–30k 的三 seed 均值 `−62.419` 扩大为 35–95k 的 `−128.583`。这说明早期 source 介入的价值可以在 source 离场后通过 learner state、occupancy 与 replay 更新继续实现。

因此当前 controller 估计的是：

`B_i(t) = source i 在当前混合 occupancy 上执行一个短段时的即时 reward`

论文真正需要的是：

`L_i(t) = 允许 source i 产生数据后，student 在后续 source-free 学习中的增量`

`B_i(t)` 与 `L_i(t)` 不是同一个 estimand。更窄的置信区间、更多 persistence window 或更保守的 z 只会改变触发频率，无法消除二者的结构性错位。

## 5. 与 Claude T0017 的交叉审计

一致之处：Phase A 必须 FAIL；adaptive revocation 不进入主方法；不应继续调行为 reward 阈值；lifecycle 机制本身通过；powerlift 是兼容性边界，truck 是关键反例。

需要收窄/修正之处：

1. “crawl 正向机制真实存在”超出证据，机器分析器已明确 `mechanism_attribution_supported=false`。
2. “truck 三个源均为已证好源”把 bank-level 正迁移误写成 source-level 因果归因。
3. “basketball 伤害主体是行走类源”没有单源训练对照；FastDSAC 的 body-rebound 机制不能替代本项目的 source attribution。
4. checkpoint 确实持久化了 `admission_audit.decision_history`；窗口 mean/LCB/UCB 可以离线重建，不是仅存 W&B。
5. powerlift 用未舍入数值复算的 t 为 `−4.085`，不是 `−4.13`；t 非预注册 gate，不改变 PASS。

## 6. 下一步建议

1. **立即停止 adaptive behavioral reward 路线**：不调 z、window、persistence，不启动“第四种行为 reward proxy”。
2. **保留并写实 lifecycle 主张**：exact abstention、quarantine、atomic revoke、behavior/replay/critic 三通道同步退出、authority-coupled replay handoff；把它们表述为 correctness/safety infrastructure，而不是已解决 automatic admission。
3. **不要把本轮负结果包装成 selector 贡献**：可作为关键机制消融/失败分析，支持“behavior utility 与 learning utility 分离”，但不能替代正向算法贡献。
4. **自动选择若继续，只允许换 estimand**：候选必须评价 student-side delayed learning outcome，而不是 source-own reward。由于既有 SIV/SHU 已表明短 micro-branch 也未稳定复现完整训练价值，下一阶段应先做设计与可识别性论证，不直接发起新一轮大矩阵训练。
5. **Phase B 继续暂停**：先由 PI 决定论文采取“当前可守住的 static RBO + lifecycle + honest boundary”路线，还是承担更高风险重新攻 automatic learning-utility admission；在这个选择前，外部 baseline 扩张不会修复核心创新缺口。

## 7. 可复核证据

- 冻结协议：`configs/experiments/adaptive_admission_v1.yaml`
- 自动裁决：`artifacts/adaptive_admission_v1/20260714T110054Z/analysis_20260714T110054Z.json`
- finalizer：`artifacts/adaptive_admission_v1/20260714T110054Z/finalization_summary.json`
- 逐 checkpoint 审计：`artifacts/adaptive_admission_v1/20260714T110054Z/training_verification/`
- 原始日志：`logs/train/adaptive_admission_v1_20260714T110054Z/`
- Claude 原分析：`docs/adaptive_admission_v1_results.md`、协作记录 T0017
