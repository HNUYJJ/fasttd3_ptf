# Hurdle 等剂量单源 RBO 标定：3-seed 确认结果

日期：2026-07-23  
任务：`h1hand-hurdle-v0`  
裁决：`FULL_ORDER_REPLICATED`

## 证据边界修正

classic PTF 多教师实验此前只观察到 option selector 最常选择 walk。项目没有在该
classic PTF 实验中分别运行 walk-only、run-only、stand-only 和 scratch 来建立
独立教师价值标签。因此，准确说法是“多教师 PTF 调度器表现出 walk 选择偏好”，
而不是“`Q_omega` 已正确预测 walk 是最佳教师”。

该选择偏好可能同时受到 option value/advantage 估计、探索、termination 训练、
共享 reward 信号量级以及被 selector 访问到的状态分布影响。

## 冻结实验

seed-1 feasibility screen 之后，只补：

- scratch：seeds 2/3；
- walk：seeds 2/3；
- run：seeds 2/3。

stand 不补种子，因为当前最小问题只是确认 provisional top、runner-up 与 scratch。
所有 source 臂继续使用 source/student 同一 categorical 分布中的 0.5/0.5 等剂量、
25 步锁存、0–30k `bootstrap_only` 和相同 replay 规则。

## 结果

| arm | 5k–25k nAUC（seeds 1/2/3） | mean | 30k source-free return（seeds 1/2/3） | mean |
|---|---|---:|---|---:|
| scratch | 14.264 / 12.841 / 14.036 | 13.713 | 48.536 / 26.922 / 32.353 | 35.937 |
| walk | 75.099 / 88.942 / 82.153 | 82.065 | 141.879 / 123.282 / 157.304 | 140.822 |
| run | 175.950 / 129.368 / 175.016 | 160.111 | 357.071 / 424.121 / 465.594 | 415.596 |

配对干预增量均值：

| contrast | ΔnAUC mean | 90% CI | Δsource-free endpoint mean | 90% CI |
|---|---:|---:|---:|---:|
| walk − scratch | +68.351 | [55.477, 81.225] | +104.885 | [75.479, 134.291] |
| run − scratch | +146.398 | [102.783, 190.013] | +379.659 | [271.466, 487.851] |
| run − walk | +78.046 | [22.708, 133.385] | +274.774 | [187.559, 361.989] |

每个 seed 的在线 AUC 和 source-free endpoint 都分别满足：

`run > walk > scratch`

## Treatment audit

| arm/seed | candidate masses | behavior source share | critic source share |
|---|---|---:|---:|
| run/s2 | [0.5, 0.5] | 0.502153 | 0.500135 |
| run/s3 | [0.5, 0.5] | 0.500304 | 0.500020 |
| walk/s2 | [0.5, 0.5] | 0.501084 | 0.500092 |
| walk/s3 | [0.5, 0.5] | 0.499220 | 0.500014 |

seed 1 的相应 share 同样约为 0.5。source 排序不能由不同臂的注入剂量失配解释。

## 科学结论

1. Hurdle 上存在稳定、可重复的 source-specific RBO 干预差异。
2. run 与 walk 都能明显提高学生早期学习效率和撤除 source 后的能力，run 在三个
   seeds、两个视角中均优于 walk。
3. 这些数值可以作为 Hurdle/RBO 通道的干预标签：
   `G_run > G_walk > G_scratch`。
4. 它们仍不是迁移性指标。标签包含轨迹内容、状态覆盖、episode 生存长度、
   occupancy 改变和 replay 暴露等完整 intervention-package 效应。
5. classic PTF 的 walk 选择偏好与 RBO 标签 top=run 不一致，但该事实单独不能证明
   `Q_omega` 网络训练错误；PTF selector 与 RBO intervention 衡量的对象不同，
   且前者还受 option/termination 机制质量影响。

## 下一步

下一阶段不再补 Hurdle 的相似曲线，而应：

1. 对现有 stand/walk/run 的低成本 target probe 特征做只读提取，检查哪些特征能
   排出本轮冻结的 RBO 标签；
2. 在一个已知负迁移任务上用同一等剂量协议建立 source-specific 负标签；
3. 只有候选信号同时通过正任务排序和负任务拒绝，才升级为迁移性指标候选。

原始确认结果：
`logs/train/hurdle_equal_dose_source_calibration_confirm_v1_20260723T151810Z/`

