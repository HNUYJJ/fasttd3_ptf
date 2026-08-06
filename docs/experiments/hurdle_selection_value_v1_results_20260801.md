# Hurdle 选源价值实验结果

> 日期：2026-08-01
> 预注册：`docs/experiments/hurdle_selection_value_v1_prereg_20260731.md`
> 冻结裁决输出：`docs/data/hurdle_selection_value_v1/results.json`
> 裁决：**`SELECTION_VALUABLE`**

## Material Passport

- `material_type`: experiment result
- `claim_status`: pilot-supported
- `engineering_status`: PASS
- `scientific_status`: SELECTION_VALUABLE
- `scope`: Hurdle，source bank `{stand, walk, run}`，单批 3 learner seeds

## 1. 实验回答的问题

此前已经知道：短期 racing 在 Hurdle 上选择 `run`，且固定 `run` bootstrap
相对 scratch 能明显加速学习。但这仍可被“source identity 不重要”解释。

本实验从头训练与 `run` 完全同协议、同剂量的 `stand` 臂，检验 racing 选择的
`run` 是否真的比候选集合中排名最低的 `stand` 更快达到任务阈值。

## 2. 工程身份

- stand 行为占比：`0.4975 / 0.4965 / 0.4975`
- run 行为占比：`0.4994 / 0.4983 / 0.4995`
- 同 seed 绝对差：`0.0019 / 0.0018 / 0.0020`
- 18 份 stand checkpoint、18 份 128-episode source-free 评估均通过身份、
  seed 面板、global step 与协议检查。

因此主比较没有由 source 剂量差解释。

## 3. 主结果

| 阈值 | seed | run 达阈步数 | stand 达阈步数 | stand/run 时间比 |
|---:|---:|---:|---:|---:|
| 200 | 1 | 18,781 | 72,246 | 3.85× |
| 200 | 2 | 16,602 | 100,000（右删失） | ≥6.02× |
| 200 | 3 | 17,557 | 86,899 | 4.95× |
| 300 | 1 | 27,115 | 100,000（右删失） | ≥3.69× |
| 300 | 2 | 22,403 | 100,000（右删失） | ≥4.46× |
| 300 | 3 | 23,054 | 100,000（右删失） | ≥4.34× |

- `θ=200`：run 比 stand 快，`3/3`；中位时间比 `4.95×`。
- `θ=300`：run 比 stand 快，`3/3`；中位时间比下界 `≥4.34×`。
- 两个预注册阈值都满足 `3/3`，故输出 `SELECTION_VALUABLE`。

次级比较显示：按同一首次达阈口径，stand 在 `θ=200` 仅 `1/3` 快于
scratch，在 `θ=300` 为 `0/3`。这支持“run 与 stand 不能任意替代”，但不能
推出 stand 在所有阶段都无益：stand 在 30k 的三条曲线均高于 scratch，而持续
50% 剂量下的后期表现又低于 scratch，早期 bootstrap 与长期暴露效应没有被本实验
分离。

预注册主判据采用首次达阈，合法但不等于“此后稳定保持”。作为不改变裁决的敏感性
分析，若改为“从该点起所有后续 checkpoint 均不低于阈值”，run 相对 stand 在
`θ=200` 和 `θ=300` 均为 `2/3`，失效的都是最终回落的 seed 2。因此主张必须限定为
**更早首次达阈**，不能写成更早稳定学会。

## 4. 科学解释

本结果支持一个窄而重要的结论：**在 Hurdle 这个已知存在可迁移 source 的场地，
source identity 对长期训练的早期样本效率有实质影响；racing 选择的 run 不是
可以被任意 source 替代的。**

这里排除的是“bank 中任意 source 都同样快”的平凡解释；它没有排除“任意真正
有益的 source 都同样快”，因为本轮没有做 `run` 与中间候选 `walk` 的同协议长程
比较，也不能据此称 `run` 已是全部候选中的长期最优源。

它不等于“已经得到通用迁移性指标”。racing 是有成本的短期干预协议；本实验也
只有一个 target、一个新 learner 批次。按照预注册 M24，结果只能称 pilot，
仍需第二个 argmax 不同的 target 与独立批次支持。

## 5. 同时暴露的边界

`run` seed 2 的 source-free return 从 75k 的 `698.5` 回落到 100k 的 `111.3`；
`stand` 三个 seed 在 100k 仅为 `227.5 / 158.5 / 229.8`。这说明：

1. 选对 source 能解决早期“谁来教”的问题；
2. 全程约 50% 的恒定 source 剂量仍会产生长期不稳定；
3. 后续方法必须把 **source selection** 与 **有限 bootstrap / lifecycle exit**
   组合，而不能把选源成功误写成调度问题也已解决。

## 6. 成本与部署口径（冻结定位）

后续若把 racing 组成完整算法，**允许继续训练获胜 arm 的 learner state**，否则会无谓
丢弃已经支付的 `K=10k` 学习。相应地，正式评估必须加入相同候选数、相同总环境交互
预算的 scratch population，并比较其 best-of-N；不能把“挑多个 learner 后取最好”的
order-statistic 收益误归因于选源机制。

在本轮 Hurdle 数据中，run 相对单条 scratch 的**首次**达阈中位节省约 `59.6k`
（`θ=200`）和 `67.0k`（`θ=300`）vector steps。这个口径不能直接变成候选规模上限：
若改用“从该点起所有后续已观测 checkpoint 都不再跌破阈值”，中位节省的观测上界
降为约 `27.2k / 33.1k`；其中 run seed 2 在 100k 又跌破两个阈值，真实持续达阈
时间仍右删失。

成本还必须区分三种算法：丢弃 pilot 后重训选中 source 的开销为 `n×K`；续用实际
winner learner state 的开销为 `(n-1)×K`；若另含 student abstention arm，则三源
设置的开销又回到 `3×K=30k`。128-episode 选择面板也消耗 target simulator
interaction；当前 evaluator 没有记录实际 episode length，只能把每臂成本界在
`1–1000` 个 128-env vector-step equivalents，不能精确扣账。

更根本的是，现有 racing 分支在 K=10k 后没有被 faithful continuation；长期 run
曲线来自独立的 source-identity 验证，且没有同候选数、同 anchor 的 scratch
population best-of-N 对照。因此本轮**不能再声称 `n≤6/n≤7`**。机器可读复算见
`docs/data/hurdle_portfolio_economics_v1/results.json`；当前严格裁决是
`END_TO_END_ECONOMICS_NOT_IDENTIFIED`。候选规模与净收益须由真正的 winner-state
continuation + compute-matched scratch population 实验决定。

Exact abstention 目前仍是工程能力而不是已验证的自动判断贡献：相对排序在 Hurdle 有
证据，但“最佳 arm 相对 student arm 的短窗绝对符号能否预测长程正负”尚未通过独立
正负任务验证。

## 7. 主张边界

可以声称：

- Hurdle 上，自动短期 racing 选出的 `run` 相比同 bank 中的 `stand` 带来约
  4–5 倍的**首次**达阈时间优势（含右删失下界）；
- “任何源都一样”的平凡解释在该场地被排除。

不能声称：

- 跨 HumanoidBench 普遍成立；
- racing 是零成本 transferability metric；
- 100k 终点稳定改善；
- `run` 已胜过 bank 中全部候选的长程训练；
- learned termination 或自动退出已经解决。
