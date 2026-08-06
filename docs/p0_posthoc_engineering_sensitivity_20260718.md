# P0 Post-hoc Engineering Sensitivity Analysis（事后机制审计与条件敏感性分析）

> 日期：2026-07-18
> 状态：**FINAL post-hoc analysis（v1.0；十二次复核 `Minor revision` 五项
> 修复+两处措辞收窄后升级；科学结论三层获认可）**
> 证据地位声明（按 ChatGPT 十一次意见的两层结论架构）：
> **本文档是 post hoc 分析，不具有预注册裁决地位。P0 的预注册正式结论保持
> `ENGINEERING_INVALID`（`logs/p0_lease_oracle/p0_adjudication_result.json`，
> 不可覆盖、不可修改）。** 本分析仅回答两个问题：(1) truck 触发工程门的
> critic 采样占比下偏是否为已解释的结构性机制效应；(2) 在接受该解释的
> **条件下**，P0 数据给出什么科学信息。

---

## 1. 预注册正式结论（不变）

- 联合裁决：`ENGINEERING_INVALID`。
- 触发原因：truck 三个 lease seed 的累计 critic 源采样占比
  （0.0787 / 0.0793 / 0.0773）低于冻结审计带 `[0.08, 0.12]` 下限。
- 该结论按预注册协议字面成立，本文档不推翻它。

## 2. 机制诊断：critic 累计占比下偏是冷启动结构效应

### 2.1 代码根因

`fasttd3_ptf/official_fasttd3_ptf/ptf_replay.py`（`_admission_slot_weights`）：

```python
available = counts > 0                    # per-env、per-桶可用性
masses = candidate_masses * available     # 空桶配额被掩码
masses = masses / mass_sum                # 归一化 → 空桶份额让渡(主要给 student)
```

lease 分支从 anchor 的**纯 student buffer**（空 bank scratch）起步。某
(env, source) 桶在该源第一个 segment 写入前为空，其 replay 配额被让渡；
随源数据流入，占比逐步收敛到名义值。因此**累计** critic 占比必然低于
名义 0.10，且有效源数越多、单源越稀疏，下偏越大。

### 2.2 冻结先验参数（仿真与解析的唯一机制输入）

分支起点（`policy_events[0]`，`replay_ptr=0`，先于任何训练数据）安装的
candidate_masses（机制参数，非结果观测）：

| 任务 | stand | walk | run | hurdle | student |
|---|---|---|---|---|---|
| crawl | 5.12e-08 | 0.05609 | 0.04391 | — | 0.90 |
| truck | 1.21e-05 | 0.02464 | 0.02012 | 0.05523 | 0.90 |

有效源数（mass 不可忽略的源）：crawl=2、truck=3；stand 两任务均 ≈0。
这些 masses 可从冻结配置独立重算：`softmax([bank yaml 各源
bootstrap.weight…, student_logit])`（τ=1，与 `admission_control.
AdmissionSnapshot.masses` 同一公式；student_logit=冻结 CLI 值
16.6823567039/16.4139012941）——重算值与 checkpoint `policy_events[0]`
逐位一致（atol 1e-6 断言，见 §2.5 脚本），checkpoint 事件仅作一致性
校验，不是分析输入。

### 2.3 实测复算（checkpoint 审计计数，独立复现 ChatGPT 十一次意见的两个经验声明）

行为端（exec 占比，累计@13000，3 seeds）：crawl walk 0.0563–0.0587 /
run 0.0421–0.0449；truck hurdle 0.0535–0.0561 / walk 0.0243–0.0258 /
run 0.0205–0.0224；两任务 stand≈0——**与先验 masses 一致，行为剂量按协议
精确实现（源合计 0.098–0.103 ∈ [0.08,0.12] 全过）**。

critic 端 750 步分段占比（四 checkpoint 计数差分）：

| 任务 | 10k–10.75k | –11.5k | –12.25k | –13k（末段） |
|---|---|---|---|---|
| crawl s1/s2/s3 | 0.050/0.054/0.051 | 0.091/0.092/0.088 | 0.099/0.099/0.097 | **0.100/0.100/0.099** |
| truck s1/s2/s3 | 0.042/0.045/0.044 | 0.082/0.082/0.079 | 0.094/0.093/0.091 | **0.098/0.097/0.096** |

单调爬升、末段收敛到名义 0.10——与 replay 通道按机制预期运行一致；
累计口径的下偏主要由冷启动暂态充分解释（不排除采样波动等次要因素）。

### 2.4 解析近似

每个非零源桶的冷启动亏损 =（1/L)∫₀^L m_i·e^{−λ_i t}dt = h(1−e^{−λ_i L})/L，
其中 λ_i=m_i/h（segment 到达率）。λ_iL≫1 时每源亏损→h/L=25/3000≈0.0083，
即 ChatGPT 十一次意见的一阶式 q≈η−N_eff·h/L 的来源。代入冻结 masses：

- crawl 预测 ≈ 0.10 − 0.00832 − 0.00830 = **0.0834**（实测 0.0836–0.0862）；
- truck 预测 ≈ 0.10 − 0.00790 − 0.00759 − 0.00832 = **0.0762**（实测 0.0773–0.0793）。

实测略高于一阶预测 +0.001–0.002，方向与被忽略的"归一化返还"二阶项
（空桶份额按比例部分返还给非空源桶）一致。

### 2.5 CPU 微仿真（复用真实权重函数计算条件期望，冻结配置输入）

方法：**复用真实 `PTFReplayWrapper._admission_slot_weights`（被审计机制的
核心权重函数）计算条件期望占比** `E[frac] = mean_env(Σw[source slots]/Σw)`
——multinomial(replacement=True) 的样本占比是该期望的无偏估计，期望法给出
同一数值的零方差版本（本仿真不调用 `draw_indices` 抽样）。行为流（segment
边界 option 选择）为随机，由 3 个仿真 seed 覆盖；权重每 5 步评估（冷启动
时间尺度数百步，稀疏化误差可忽略）。

输入：candidate_masses **从冻结配置重算**（§2.2 公式；checkpoint 事件仅作
atol 1e-6 一致性断言）；其余为冻结机制参数（n_env=128、h=25、anchor 10000
步纯 student、branch 3000 步、buffer 51200、uniform_mix=1.0/recency=0/
priority=0/authority active）；不读取任何 return、U 值、eval 结果或实测
critic 计数。脚本=`scripts/analysis/p0_critic_fraction_expectation.py`，
输出=`docs/data/p0_posthoc/critic_fraction_expectation.json`（含 bank yaml
SHA256 与重算 masses）。

| 任务 | 指标 | 3 个仿真实现的范围 | P0 实测（3 seeds） |
|---|---|---|---|
| crawl | 累计@3000 | **0.0845–0.0858** | 0.0836–0.0862 |
| crawl | 首段 750 步 | 0.0518–0.0562 | 0.050–0.054 |
| crawl | 末段 750 步 | 0.0992–0.0995 | 0.099–0.100 |
| truck | 累计@3000 | **0.0775–0.0791** | 0.0773–0.0793 |
| truck | 首段 750 步 | 0.0446–0.0469 | 0.042–0.045 |
| truck | 末段 750 步 | 0.0958–0.0964 | 0.096–0.098 |

累计值、分段爬升曲线、任务排序全部复现；3 个仿真实现的范围与实测范围
几乎重合（注：此为随机行为流的实现范围，非具有覆盖率定义的统计预测
区间）。

### 2.6 机制结论

行为剂量正确实现（exec 通道三 seed 全过），观测与 replay 通道按代码
预期运行一致（末段收敛 0.096–0.100）；解析与微仿真在不读实测 critic
数据的前提下复现实测累计占比的量级与任务排序，累计下偏主要由冷启动
暂态充分解释。**原冻结审计带把 exec 名义剂量带
`0.10±0.02` 统一套到累计 critic 观测上，未建模冷启动暂态与有效源数
依赖——属验收带校准失配，不是注入实施失败。**该判定为 post hoc：
修改验收标准会改变 truck 数据的证据地位，因此正式结论仍为
`ENGINEERING_INVALID`，本节仅提供机制解释。

## 3. 条件科学结论（条件=接受 §2 机制解释；程序化计算，裁决器纯函数）

用 `p0_adjudicate._classify_task/_joint`（冻结判序，未修改）对正式
manifest 计算，跳过的唯一环节是 treatment 审计（即"条件"本身）。
可复现入口=`scripts/analysis/p0_conditional_classification.py`，输出
=`docs/data/p0_posthoc/conditional_classification.json`（记录 git HEAD、
裁决器 SHA256 与全部输入文件 SHA256）：

| 任务 | 平均 U | 90% CI | d_dup vs δ | 条件分类 |
|---|---|---|---|---|
| crawl | −60.77 | [−152.91, +31.37] | 50.50 ≥ 33.56 | `UNCERTAIN_NUMERIC` |
| truck | −66.47 | [−149.07, +16.13] | 28.37 < 28.85 | `UNCERTAIN` |
| 联合 | — | — | — | **`F-a`：统计不可测（数值地板）；判据封存** |

即使接受机制修正，P0 也不是 PASS、不是 SURROGATE_FAIL。条件结论=
预写 F-a 语句：**在 3 seeds、L=3000、当前评估噪声下，局部延迟学习效用
不足以被稳定测量，不能承担在线准入或续租判据。**

## 4. 描述性趋势（无裁决地位）

- **truck U 均值为负（−66.47）**：s1/s2 大幅为负、s3 端点 +20.7；四个
  checkpoint 的配对均值均为负。与历史 matched handoff 正收益（≈+227.8）
  **不构成逻辑矛盾**——两者 estimand 不同（历史=从训练早期开始的完整
  bootstrap 累计收益；P0=10k 步后一次 3000 步、10% 剂量 lease 的边际
  效用）。支持"source utility 具有阶段依赖性、有效注入窗口可能更靠前"
  的假设；因工程门未过+CI 跨零+一个 seed 为正，只能记为描述性趋势，
  **不能写成"10k 后注入已被证明有害"**。
- **crawl d_dup=50.50 的正确语义**："固定 seed 条件下完整 3000 步分支
  训练的重复运行差异"（含独立进程启动、CUDA 非确定性、3000 步训练混沌
  放大、评估差异），不是纯 CUDA 算子噪声。它与 abstain 臂 seed 间极差
  （≈73）同量级——单个 counterfactual fork 在 crawl 上分辨率不足，
  此类 oracle 即使作为离线分析工具也不稳定，更不能每 3000 步在线运行。

## 5. 处置（按 ChatGPT 十一次意见第 6 节）

1. 不修改原裁决器与原结果；不重训 P0；不通过调剂量/强制 critic 达标
   "挽救"。
2. 最终记录三层并存：预注册 `ENGINEERING_INVALID` + 机制诊断（验收带
   失配）+ 条件科学结论（crawl `UNCERTAIN_NUMERIC` / truck `UNCERTAIN` /
   联合 `F-a`）。
3. **Phase-2 counterfactual lease oracle 作为在线算法贡献封存**。
4. 回到 Phase-1：exact abstention、source provenance、有限剂量+TTL、
   source 撤销后旧数据退出 active replay、注入集中于早期阶段。
   明确边界：这些机制**限制伤害、清除旧数据，不解决"如何判断当前
   source 有益还是有害"**。

## 6. 教训（拟入 ISSUES_AND_LESSONS）

E16（拟）：多通道 treatment 审计的每个通道验收带必须**分别**由机制推导
校准（含暂态），不能把名义参数带统一套到下游累计观测上；带定错的代价
= 一次完整实验的正式无效。P0 的 exec 带正确（直接度量），critic 带
失配（下游累计观测，冷启动暂态 ~N_eff·h/L 未建模）。
