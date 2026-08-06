# Classic PTF：bounded Q 的 option-only reward-scale 修复与否证结果

> 日期：2026-07-23  
> 干预：只把 `Q_omega` 的 TD reward 乘固定比例 0.01；FastTD3 critic、
> 环境 return、蒸馏损失和 source-free evaluator 不变。  
> 结论：**数值量纲修复成功，但原始 PTF 教师排序/termination 机制 gate 失败；
> 停止扩展三臂、100k和多种子。**

## 1. 假设与停止规则

Hurdle 单步 reward 位于 `[0,1]`，`gamma=0.99`，作者公开代码的 option Q 使用
`tanh` 输出。未缩放时

```text
y = r + gamma * U
Q_omega in [-1, 1]
```

大量 TD target 超出 Q 的表达范围。30k 未缩放 checkpoint 已有 38.4% 的状态
出现“所有 option 同时 `|Q|>0.95`”，37.2% 同时满足低 gap。

本轮只检验：

> 使用 `r_option=(1-gamma)/R_max*r=0.01*r` 消除量纲冲突后，
> `Q_omega` 是否恢复可辨识的 source 排序，beta 是否恢复状态条件切换。

若30k时 Q 虽不饱和但 gap 仍退化、beta 仍不能区分最优/非最优 source，则停止，
不更换 reward scale、xi 或 beta 参数挽救结果。

## 2. 实现

新增 `option_reward_scale`，默认1.0以保持旧路径。它只参与：

```python
y_option = option_reward_scale * r + gamma * bootstrap * U_next
```

新增在线中介量：

- option TD target min/max/越界率；
- replay batch 与 rollout state 的 top1-top2 Q gap；
- 所有 option 同时饱和比例；
- 所有 option 同时饱和且 gap<0.01 的比例。

正式 gate 配置：
`configs/experiments/classic_ptf_hurdle_released_fidelity_scaled_v1.yaml`。

## 3. 验证链

- 聚焦测试：40项通过；
- 200-step fresh smoke：option/beta 各更新189次，参数有限；
- fresh 5k：TD target 越界率0、Q饱和率0、三源均执行过选择与切换；
- fresh 30k：完整结束，option/beta 各更新29,989次，参数有限。

所有训练均从头开始，没有从旧 beta logit 已饱和的 checkpoint 续训。

## 4. 30k训练结果

### 4.1 source-free在线曲线

| step | scaled return | 未缩放 fidelity return |
|---:|---:|---:|
| 5k | 8.06 | 5.21 |
| 10k | 9.97 | 15.41 |
| 15k | 19.57 | 25.32 |
| 20k | 29.91 | 67.31 |
| 25k | 54.08 | 94.97 |
| 5k--25k nAUC | **22.63** | **39.53** |

历史 scratch nAUC=11.04只能作尺度参考，不能与本次不同checkout的run构成严格因果比较。

### 4.2 数值量纲

修复后的训练全程：

- option TD target 越界率：0；
- rollout“所有 option 同时饱和且低gap”比例：0；
- 25k option TD target 最大值：0.0464；
- 29.9k replay batch TD target 最大值：0.0790。

因此 F1 的 `tanh Q × raw HB reward` 量纲冲突确实被消除。

### 4.3 教师排序与termination

30k固定面板共6,447个状态：

| 量 | 未缩放 | scaled |
|---|---:|---:|
| mean `abs(Q)` | 0.458 | 0.0209 |
| 所有 option `abs(Q)>0.95` | 38.4% | 0% |
| Q gap中位数 | 0.00226 | 0.000428 |
| gap<0.01 | 63.9% | **100%** |

scaled checkpoint 的 source 条件统计：

| source | Q argmax | beta(argmax) | beta(non-argmax) | non−argmax |
|---|---:|---:|---:|---:|
| stand | 8.1% | 0.760 | 0.784 | +0.024 |
| walk | 89.6% | 0.896 | 0.838 | −0.058 |
| run | 2.3% | 0.695 | 0.691 | −0.003 |

Q 并非完全没有教师信息：去饱和后，两个独立评估面板都在约90%的状态上把 walk
排为第一。walk 已有3-seed fixed-transfer正收益，因此这个方向与“walk 是已知有效
教师”一致；但项目尚未完成Hurdle上的stand-only/run-only匹配实验，**不能写成
walk已被证明是三者中的最优教师**。

真正失败的是信号向termination的传递：Q gap中位数只有Q平均量级的约2%，没有一个
source表现出旧未缩放run中 `run +0.908 / walk +0.347` 的正确状态条件termination。
walk虽几乎总被Q排第一，其beta在argmax状态反而更高。准确地说，这是微小排序信号
未能驱动beta，而不是beta学会了有意义但错误的终止策略。

在线25k：

- termination rate=0.897；
- option age=0.24；
- stand/walk/run rollout占比=0.320/0.219/0.461。

这不是稳定的 call-and-return 教师调度，而是高概率终止后，在仍然很高的官方
epsilon探索率下频繁重新抽取 source。

### 4.4 独立固定面板性能

30k scaled student 的8-episode return：

```text
mean = 54.34
median = 61.04
population SD = 28.16
```

未缩放 student 在同一评估协议下为121.17。两者均为单训练seed，不升级为正式性能结论。

## 5. 科学裁决

### 支持

1. 未缩放 fidelity run 的 Q/beta信号部分来自超出 `tanh` 表达范围后的饱和动力学；
2. option-only scale=0.01可以在不改FastTD3 critic的情况下完全消除该数值越界；
3. `tanh Q`、adaptive xi、bare sigmoid beta与高探索率是耦合系统，不能分别看作
   独立可修的小问题；
4. 去饱和后的Q排序稳定指向已知有效的walk教师，说明Q含有方向一致但量级微小的
   教师信息；stand/run/walk真实收益全排序仍待fixed-source标定。

### 否证

1. “只修reward量纲即可恢复原始PTF自动教师调度”被否证；
2. 去掉饱和后，各source Q保留了walk优先的微小排序信号，但极小gap不足以驱动
   beta，call-and-return仍退化为高频终止；
3. 旧未缩放run中漂亮的run/walk条件beta不能作为健康PTF调度的稳健证据；
4. 当前不应通过选择0.02/0.05等中间scale或调整xi/beta继续搜索折中点。

更根本的限制是：环境行为由student产生，所有option共享同一student reward；
`Q_omega` 的差异主要来自compatibility数据选择和beta bootstrap，而不是
“接受source蒸馏后student未来学习增量”的直接监督。量纲修复无法改变这个estimand。

## 6. 后续定位

- classic PTF fidelity作为严格复现baseline保留；
- 当前暂停原计划的scratch/fidelity/fixed-walk三臂正式扩展；若论文最终需要严格
  匹配的PTF定量baseline，再以独立baseline目的回补，而不是用于抢救termination；
- stand/walk/run fixed蒸馏收益排序保留为transferability主线的ground-truth backlog，
  但它服务于检验“source真实学习收益排序”，不再用于抢救原始PTF termination；
- 主线应回到source-specific、student-relative、面向后续学习增量的教师价值证据。

## 7. 产物

- W&B 5k：`xarefkms`
- W&B 30k：`mp0k86wm`
- 训练日志：`logs/classic_ptf_released_fidelity_scaled/`
- checkpoint：
  `models/h1hand-hurdle-v0__classic_ptf_hurdle_released_fidelity_scaled_30k_s1_20260723__1_final.pt`
- 固定面板：
  `docs/data/classic_ptf_released_fidelity_scaled/feasibility_s1_30k_signal_v2.json`
