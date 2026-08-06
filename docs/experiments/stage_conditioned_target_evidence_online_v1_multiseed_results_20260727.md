# Stage-conditioned target evidence online v1：多种子终局裁决

> 日期：2026-07-27  
> 主裁决：
> **Hurdle 上 3/3 正迁移成立，但 Crawl 出现 1/3 false admission 并造成明确负迁移。**
> 因此当前方法可保留为 **target-semantic feasibility / immediate intervention
> evidence**，不能宣称为可靠的一般迁移性指标或独立在线 admission oracle。

## 1. 本实验回答什么

唯一主要假设是：

> 在当前 student occupancy 上，从匹配状态分别执行 source 和 student 的短干预；
> 若 target return 增量和 target-achievement progress 增量的 90% 保守下界都为正，
> 则该 source 当前值得进入 reward-bearing bootstrap。

核心算法不包含 `crawl`、`hurdle` 或任何 source 名称分支。任务差异只由
target-evidence contract 声明：

- Hurdle 的有效进度是满足直立/越障语义的前进；
- Crawl 的有效进度是满足匍匐姿态和隧道占用语义的前进。

这遵守“机制一般化、目标完成语义由 target MDP 提供”的设计边界。实验的外部验证
不是看 probe 本身是否按规则工作，而是看其 admission 是否真的改善后续
source-free student 学习。

## 2. 冻结机制

- 决策时点：10k、20k；0–10k 为 exact abstention；
- matched-state panel：固定 reset seeds × episode ages，共 32 个 student states；
- source/student intervention horizon：25；
- confidence：固定 5000 次 bootstrap，90% LCB；
- admission：`LCB90(ΔR)>0` 且 `LCB90(ΔP)>0`；
- selection：在 admitted sources 内按 `LCB90(ΔP)` 选 top-1；
- behavior/replay dose：top-1 与 student 各 0.5；
- probe 数据只进入 quarantine，不写主 replay；
- source 撤销后旧数据保留在 physical/main buffer 供审计，但立即退出 active replay；
- 训练长度：30k；评估曲线：5k、10k、15k、20k、25k；
- endpoint：30k checkpoint 上结构性 source-free、deterministic、32 episodes。

## 3. 跨种子 admission 决策

| Task | Seed | 10k selected / admitted | 20k selected / admitted |
|---|---:|---|---|
| Hurdle | 1 | walk / walk, run, stand | run / run, walk |
| Hurdle | 2 | run / run, walk | **NONE / exact abstention** |
| Hurdle | 3 | walk / walk, run | walk / walk |
| Crawl | 1 | **NONE** | **NONE** |
| Crawl | 2 | **NONE** | **NONE** |
| Crawl | 3 | **run / run, walk** | **NONE** |

### 稳定部分

- Hurdle 10k 的三种子都接纳 run 和 walk；
- Hurdle 的精确 top-1 为 walk/run/walk，说明 admission 集合比近似并列教师的细排序
  更稳定；
- Hurdle 20k 会随 student 能力发生收缩、切换或完全弃权；
- Crawl seed1/2 在 10k、20k 均 exact abstention。

### 关键反例

Crawl seed3 在 10k 的 matched-state probe 中：

| Source | LCB90 ΔR | LCB90 ΔP | Decision |
|---|---:|---:|---|
| run | +2.560 | +0.068 | admit, selected |
| walk | +1.980 | +0.031 | admit |
| stand | -0.235 | -0.010 | reject |

这不是“只看 root-x”造成的明显语义错误：run/walk 在该 student occupancy 和 25-step
窗口内，确实同时提高了 target return 与受 Crawl posture/tunnel gate 约束的 progress。
但它们是否提高后续 student 学习仍需外部结果裁决。

20k 时三个 source 的 `ΔR/ΔP` LCB 全转负，机制撤销全部 source。

## 4. Behavior / replay 生命周期审计

计数顺序为 `[stand, walk, run, student]`。

| Task/seed | execution counts | final active replay counts | final masses |
|---|---|---|---|
| Hurdle/s1 | `[0, 634725, 641003, 2564272]` | `[0, 0, 641003, 2564272]` | `[0,0,0.5,0.5]` |
| Hurdle/s2 | `[0, 0, 638401, 3201599]` | `[0, 0, 0, 3201599]` | `[0,0,0,1]` |
| Hurdle/s3 | `[0, 1274959, 0, 2565041]` | `[0, 1274959, 0, 2565041]` | `[0,0.5,0,0.5]` |
| Crawl/s1 | `[0, 0, 0, 3840000]` | `[0, 0, 0, 3840000]` | `[0,0,0,1]` |
| Crawl/s2 | `[0, 0, 0, 3840000]` | `[0, 0, 0, 3840000]` | `[0,0,0,1]` |
| Crawl/s3 | `[0, 0, 638706, 3201294]` | `[0, 0, 0, 3201294]` | `[0,0,0,1]` |

结论：

- exact abstention 是严格的：Crawl s1/s2 source behavior、source replay 和 source
  critic sampling 全部为零；
- Crawl s3 只在 10k–20k 暴露 run，20k 后 active replay 中 run 立即归零；
- 撤销/清除机制正确，但只能切断后续暴露，不能回滚 source 已经改变的 learner
  parameters 和 occupancy。

## 5. Hurdle：3/3 正迁移

### 5k–25k normalized AUC

| Seed | Online | Scratch | Paired Δ |
|---:|---:|---:|---:|
| 1 | 25.833 | 12.807 | **+13.027** |
| 2 | 57.784 | 14.808 | **+42.975** |
| 3 | 39.540 | 21.765 | **+17.775** |

- paired mean Δ：**+24.592**；
- paired SD：16.096；
- 90% t interval（df=2）：**[+7.069, +42.115]**。

### 30k frozen source-free endpoint

| Seed | Online return | Scratch return | Paired Δ |
|---:|---:|---:|---:|
| 1 | 115.07 | 25.36 | **+89.71** |
| 2 | 242.83 | 43.34 | **+199.49** |
| 3 | 104.49 | 36.53 | **+67.96** |

- paired mean Δ：**+119.05**；
- 90% t interval（df=2）：**[+42.30, +195.81]**。

Hurdle 在 AUC 与 source-free endpoint 上均 3/3 正向，且当前三种子的 90% paired
interval 不跨零。这支持：

1. reward-bearing bootstrap 通道真实有效；
2. 当前 stage-conditioned evidence 在 Hurdle 上能找到有用 source；
3. student 能在 source 撤销后保留并继续发展收益。

它不单独证明该 evidence 是跨任务可靠的迁移性指标。

## 6. Crawl：false admission 导致负迁移

### Seed1/2

- s1/s2 全程 exact abstention，工程上严格零 source 暴露；
- s1 online 与 scratch 的曲线差属于独立 GPU 训练噪声，不能归因于迁移；
- s2 不需要 scratch 来证明零暴露事实。

### Seed3 matched counterexample

10k–20k 注入 run，20k 后 exact abstention。

| Metric | Online | Same-checkout scratch | Paired Δ |
|---|---:|---:|---:|
| 5k–25k normalized AUC | 393.848 | 550.841 | **-156.992** |
| 30k source-free return | 577.95 | 720.86 | **-142.91** |
| 30k source-free max-dx | 0.60 | 8.67 | **-8.07** |
| 30k posture mean | 0.753 | 0.748 | +0.005 |

online policy 的 posture 并未明显更差，但实际有效位移几乎消失；因此不能把结果解释为
“仅仅没有匍匐”。更准确的因果链是：

1. run 在 10k 当前 occupancy 的 25-step 局部干预中产生了正 target evidence；
2. 该短期行为证据没有预测 replay 数据对后续 learner update 的价值；
3. 约 639k 条 run transitions 进入 replay 并改变 student；
4. 20k 撤销能够止血，却不能保证恢复到 scratch learner state；
5. 最终 AUC 与 source-free endpoint 均明显低于 matched scratch。

## 7. 终局科学裁决

### 已支持

- **通用接口成立**：核心算法与 task 名称解耦，任务只提供 target-achievement 语义；
- **正任务有效**：Hurdle AUC 与 endpoint 均 3/3 提升；
- **阶段条件 handoff 存在**：source 集合会随 student stage 收缩、切换或清空；
- **exact abstention / lifecycle 正确**：无 source 时严格零暴露；撤销后旧 source 立即
  退出 active replay。

### 被否证

- “target return + target-achievement progress 的短干预 LCB 足以构成可靠的一般
  迁移性指标”——**被 Crawl seed3 否证**；
- “加入 Crawl posture/tunnel 语义就能稳定避免 locomotion source 负迁移”——
  **只在 2/3 seed 成立**；
- “后续撤销可消除此前错误 admission 的伤害”——**不成立**。

### 正确定位

当前量更适合命名为：

> **Stage-conditioned target-semantic intervention evidence**

它回答“source 现在能否在目标语义下短期做得比 student 好”，而不是：

> “把 source 数据写入 replay 后，student 的未来 source-free 学习是否变好。”

因此它可以保留为迁移性机制中的 **feasibility/safety pre-filter**，但不能单独决定
admission。可靠 transferability 仍需要第二层、student-relative 的 learning-value /
handoff 证据；该层必须专门解释 Crawl s3 这种“短期行为有益、长期学习有害”的反例，
同时不能退回任务特定规则或已被否证的即时 return / critic-sign 判据。

## 8. 实验完整性

正式纳入统计的 run 均从 step 0 完整训练到 30k，并有 checkpoint、quarantine JSON、
W&B 曲线和 frozen evaluator JSON。

运行期间出现过三类纯工程中断：外部运行环境终止、受限 sandbox 下 W&B socket
初始化失败、复用时间戳造成 quarantine 拒绝覆盖。所有不完整产物均隔离保存，未进入
任何统计；最终 seed2/3 使用全新时间戳/目录从零完成。没有因为结果修改 threshold、
horizon、evidence contract、source bank 或 seed。

主要 artifact roots：

- seed1：`logs/train/stage_target_evidence_online_v1_20260726T104500Z/`
- seed2 Hurdle：`logs/train/stage_target_evidence_online_v1_confirm_20260726T114555Z/`
- fresh seed2/3 + frozen eval：
  `logs/train/stage_target_evidence_online_v1_confirm_retry_20260726T162834Z/`
