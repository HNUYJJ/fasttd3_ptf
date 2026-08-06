# Hurdle 等剂量单源 RBO 标定：seed-1 feasibility screen

日期：2026-07-23  
任务：`h1hand-hurdle-v0`

## 研究问题与边界

本实验不是迁移性指标实验，而是为指标构造 source-specific 的因果标定标签：
固定目标任务、学生训练配置、source/student 剂量、25 步锁存、reward-bearing
bootstrap 与 replay 规则，仅替换 source 身份（stand、walk、run），测量完整 RBO
干预包对学生学习的影响。

seed 1 只用于低成本 feasibility screen。它可以决定是否值得补种子，但不能证明
source 排序在训练随机性下稳定。

## 冻结处理

- source 与 student 处于同一个 categorical admission 分布；
- 单源 logit 与 student logit 均为 0，因此候选概率为 0.5/0.5；
- 不存在额外的 `teacher Bernoulli(0.5)`；
- source 每次锁存 25 步，warmup 为 0–30k；
- `bootstrap_only`，不启用动作蒸馏；
- source 与 student transition 均携带目标任务 reward 进入 replay；
- replay 在 source/student strata 间各分配 0.5，stratum 内 uniform；
- recency 与 TD priority 均关闭；
- 30k checkpoint 使用不构造 source bank 的冻结 evaluator 做 32 episode
  source-free 面板。

这里的 0.5/0.5 是单源等剂量标定所需的受控 treatment，不是最终框架的固定
teacher floor，也不是自动选源机制。

## 结果

| arm | 5k–25k nAUC | ΔAUC vs scratch | 30k source-free return | Δ30k vs scratch | 最大前进位移均值 | source behavior share | source critic share |
|---|---:|---:|---:|---:|---:|---:|---:|
| scratch | 14.264 | — | 48.536 | — | 4.480 | — | — |
| stand | 45.535 | +31.271 | 99.814 | +51.278 | 9.209 | 0.500 | 0.500 |
| walk | 75.099 | +60.835 | 141.879 | +93.343 | 13.217 | 0.501 | 0.500 |
| run | 175.950 | +161.686 | 357.071 | +308.535 | 41.519 | 0.500 | 0.500 |

训练期 5k–25k AUC 与 30k source-free 终点给出相同排序：

`run > walk > stand > scratch`

在共享的 32 个冻结评估 episode seeds 上：

- stand、walk、run 相对 scratch 的 return 分别为 31/32、32/32、32/32 胜；
- run 相对 walk 的 return 为 29/32 胜；
- run 相对 walk 的最大前进位移为 32/32 胜。

这些 episode 配对结果只说明 endpoint 排序不太可能是评估面板抽样偶然性，不能
替代独立训练 seeds。

## 结论

1. Hurdle 适合作为正向 source-specific RBO 标定任务：三个 source 在 seed 1
   均同时改善早期 AUC 与 30k source-free endpoint。
2. source 身份产生了很大的干预效应差异；run 是当前 provisional top，walk 为
   runner-up。
3. classic PTF 多教师实验中，option selector 观测上最常选择 walk；但此前没有
   walk-only、run-only、stand-only 相对 scratch 的独立干预实验，因此该现象不能
   表述为“`Q_omega` 已预测真实最佳教师为 walk”。本轮真实 RBO 标签 top 是 run，
   只说明该多教师调度器的 walk 选择偏好没有对齐当前 RBO 干预排序；原因可能包括
   option value/advantage 估计不准、termination 训练失效、共享 reward 信号量级
   不足，以及 PTF 调度 estimand 与 RBO 干预 estimand 不同。
4. 本实验测得的是完整干预包效应，混合了轨迹内容、实际状态覆盖、episode
   生存长度、occupancy 改变和 replay 暴露；不能将其缩写成纯数据质量分数。

## 决策

seed-1 gate 判定为 `ADVANCE_PROVISIONAL_TOP_AND_RUNNER_UP`。下一步应只给
run 与 walk 补 seed 2/3，并保留 scratch 对照；是否还需 stand 的补种子取决于
论文是否要估计完整三源排序，而不是判断 top source。

在补种子之前，不根据本轮结果修改 source 剂量、horizon、warmup 或任何候选指标。
补种子验证通过后，才进入“用低成本 source-specific probe 特征预测 RBO 干预
标签”的迁移性指标设计。

## 工程记录

- 四条训练均正常退出并生成 30k checkpoint；
- behavior source share 与 critic source share 均实测约为 0.5；
- 自动分析首次因 tqdm 回车进度片段导致 eval 正则只识别行首标记而失败；
  已将解析从行首匹配改为行内搜索并用原始产物重跑，未修改训练数据、评估数据或
  科学判据。

原始结果：
`logs/train/hurdle_equal_dose_source_calibration_v1_20260723T132917Z/`
