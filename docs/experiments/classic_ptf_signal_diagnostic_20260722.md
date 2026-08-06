# Classic PTF × FastTD3 信号诊断与 σ=0.5 在线 smoke

> 日期：2026-07-22  
> 任务：`walk -> h1hand-hurdle-v0`，单 source + null option  
> 定位：只检验 Q_o/β 的学习信号，不将短程 return 当作性能结论。

## 1. 科学问题与唯一主要假设

问题是当前 learned PTF 不如 fixed-transfer，究竟是动作蒸馏无效，还是
option/termination 的输入信号失真。已有 fixed-transfer 3/3 seed 正收益已经说明
蒸馏通道有效；本轮唯一假设为：`compatibility_sigma=1.5` 使 source 几乎更新所有
student transition，破坏了 Q_o 的 option-specific 选择性。

执行顺序预先冻结为：先用已有正式 checkpoint 做无梯度 frozen-rollout 断面诊断；
只有存在兼具覆盖和状态离散度的预声明 sigma 候选时，才允许一次只改 sigma 的在线
smoke。`xi`、β 学习率、蒸馏、FastTD3 和其他机制均不改变。

## 2. 冻结断面诊断

### 2.1 协议边界

- 工件：learned PTF `3 seeds × {25k,50k,75k,100k}`，共 12 个 checkpoint；
- occupancy：每个 checkpoint 用同一固定 seed 面板重新采集 deterministic student
  rollout；这是冻结策略断面，不是训练 replay/current-option 重建；
- 反事实 sigma：`{0.25,0.5,1.0,1.5}` 只重算 compatibility 权重，不代表相应 sigma
  重新训练后的 Q_o；
- Q/β 统计只描述实际使用 `sigma=1.5` 训练出的 checkpoint。

### 2.2 Compatibility 结果

下表为每阶段跨 3 seeds 的均值：

| step | action MSE p50 | compat mean σ=.25 | σ=.5 | σ=1.0 | σ=1.5 |
|---:|---:|---:|---:|---:|---:|
| 25k | 0.531 | 0.038 | 0.363 | 0.762 | 0.885 |
| 50k | 0.579 | 0.030 | 0.330 | 0.744 | 0.875 |
| 75k | 0.813 | 0.011 | 0.236 | 0.675 | 0.838 |
| 100k | 0.744 | 0.006 | 0.231 | 0.681 | 0.842 |

- `sigma=0.25`：compatibility `<0.05` 的状态比例从 77.8% 增至 99.1%，source Q_o
  基本数据饥饿；
- `sigma=1.5`：effective-sample fraction 为 0.997--0.998，几乎所有状态等权更新，
  选择性很弱；
- `sigma=0.5`：compatibility 均值 0.23--0.36，84%--99% 状态处于 0.1--0.9，
  是四个预声明值中唯一同时避免数据饥饿和近似全覆盖的候选。

因此离线 gate 通过，允许一次 `sigma: 1.5 -> 0.5` 在线 smoke。该选择完全依据
冻结支持域分布，未观察任何 sigma=0.5 训练 return。

### 2.3 原 sigma=1.5 的 Q/β 状态

`walk` argmax 比例跨阶段均值为 `0.467 / 0.400 / 0.886 / 0.677`，且同阶段跨 seed
变化很大（例如 25k 为 0.170--0.668，100k 为 0.220--0.933）。大多数断面中两个 β
仍贴近 0.05；只有少数 seed/stage 出现 non-argmax β 高于 argmax β，缺少稳定的状态
条件 termination 信号。

## 3. σ=0.5 在线 signal smoke

### 3.1 配置与产物

- 单一干预：bank 中 `compatibility_sigma=0.5`；
- seed 1，完整 100k 学习率日程，独立 `run_stop_step=5000`；
- W&B：project `ptf_fasttd3_classic_revisit`，run id `k5eyohxo`；
- checkpoint：
  `models/h1hand-hurdle-v0__classic_ptf_hurdle_walk_sigma05_s1_signal_smoke_20260722T160948Z__1_final.pt`；
- 训练速度约 16.27 vector steps/s，正常完成，无残留进程。

为避免从平均 option age 反推 hazard，本轮增加了只读累计计数：β termination
事件率、done 重选、实际 option change，以及当前 option 是否为 Q argmax。计数不
消费 RNG、不改变动作或更新。

### 3.2 在线训练期中介量（约 step 4900）

| metric | value |
|---|---:|
| source compatibility mean | 0.1727 |
| rollout walk fraction | 0.5156 |
| current option is argmax | 0.6719 |
| β when current is argmax | 0.050002 |
| β when current is non-argmax | 0.050097 |
| cumulative β termination rate | 0.05164 |
| option age mean | 18.16 |
| option change / reselection | 0.4279 |

真实 β termination rate 几乎等于网络硬下限 0.05；argmax 与 non-argmax 条件下的 β
也没有实际差异。因此 call-and-return 的终止节奏仍近似“固定 5% hazard”，而不是
学到的状态条件 termination。

### 3.3 冻结 checkpoint 断面

在同一 frozen-rollout 协议下：

| metric | 原 σ=1.5 3k smoke | 新 σ=0.5 5k smoke |
|---|---:|---:|
| trained-sigma compat mean | 0.8486 | 0.1462 |
| Q_walk − Q_null mean | -0.0027 | -0.0352 |
| walk argmax fraction | 0.2406 | 0.0615 |
| β_walk mean | 0.05030 | 0.05000 |
| β_null median | 0.05004 | 0.05000 |

两个 checkpoint 步数不同，因此该表只用于 signal gate，不是性能因果估计。新
sigma 确实让 Q_o 从近似平局变成更明确的排序；但 walk 在约 94% 冻结状态中不是
argmax 时，其 β 仍为 0.05000，没有把 Q 排序转化为终止。

## 4. 裁决

1. **H1 得到支持并被局部修复**：`sigma=1.5` 过宽；`sigma=0.5` 恢复了非饥饿、
   非全覆盖的 option-specific 更新权重。
2. **Compatibility 不是最终迁移性指标**：在 fixed walk 蒸馏已知有益的 hurdle 上，
   新 Q_o 在冻结 5k occupancy 中却 94% 选择 null，再次说明“当前动作支持域”不等于
   “未来学习价值”。
3. **H2 现在有更直接的在线证据**：即使 Q 排序已出现分离，真实 β hazard 和条件 β
   仍贴下轨，termination 没有表达这种差异。
4. **本次 gate 不支持直接跑 3-seed 正式实验或多教师**：只修 sigma 不足以恢复
   learned PTF 调度；继续正式训练只会增加曲线，不能回答新问题。

若继续最原始 PTF 复现线，下一个单因素问题应是：把 β 更新从 FastTD3 replay batch
恢复为 PTF 官方的“当前 transition/current option”更新，能否让 non-argmax option
的 β 在真实 rollout 中升高。该实验应继续保持 `sigma=0.5` 与 `xi=0` 不变，先做一次
5k 单 seed gate；若仍贴轨，则停止纯 PTF termination 修复，不再调 sigma/xi。

## 5. 产物

- 离线诊断脚本：`scripts/analyze_classic_ptf_signal_offline.py`
- 12 断面原始结果：
  `docs/data/classic_ptf_signal_diagnostic/frozen_cross_sections_v1.json`
- 在线 smoke 对照断面：
  `docs/data/classic_ptf_signal_diagnostic/sigma05_online_smoke_v2.json`
- 在线日志：
  `logs/train/classic_ptf_signal_smoke/classic_ptf_hurdle_walk_sigma05_s1_signal_smoke_20260722T160948Z.log`
- W&B 本地记录：`wandb/run-20260722_160956-k5eyohxo/`

## 6. 官方 current-transition termination 修复与停止裁决

### 6.1 单因素修复

官方 PTF 的 `Q_omega` 本来就从 replay batch 更新；真正的时序差异是 termination：
官方代码用刚发生的 `(s', current option)`，旧 FastTD3 适配则从 replay 随机抽取
历史 option。为此新增可审计的 `beta_update_mode`：

- `replay`：保留历史行为，保证旧实验可复现；
- `current_transition`：Q 更新仍用 replay，β 改用本步到达状态与本步实际锁存 option，
  done transition 不更新；
- current batch 的观测归一化使用 `update=False`，不会额外改变 FastTD3 normalizer；
- 相比上一条 σ=0.5 smoke，其余配置（σ、ξ、学习率、蒸馏、FastTD3）全部不变。

定向单元测试共 14 项通过；另有 200 步真实 HumanoidBench wiring smoke 通过。正式
signal gate 为 seed 1、5k steps、完整 100k 学习率日程。

### 6.2 在线 gate 结果

| metric @ 4.9k | replay β（上一条） | current-transition β |
|---|---:|---:|
| source compatibility mean | 0.1727 | 0.1892 |
| rollout walk fraction | 0.5156 | 0.4297 |
| current option is Q argmax | 0.6719 | 0.6719 |
| β when current is argmax | 0.0500024 | 0.0500016 |
| β when current is non-argmax | 0.0500975 | 0.0500018 |
| cumulative β termination rate | 0.05164 | 0.05108 |

`current_transition` 已由日志中的 `ptf_beta_update_current_transition=1` 和
`ptf_beta_training_valid_fraction=1` 确认真实生效，但状态条件 β 信号没有恢复；
termination hazard 仍等于约 5% 的暴露下限。

### 6.3 冻结断面复核

新 5k checkpoint 上 `Q_walk-Q_null=-0.0399`，walk argmax 比例为 0.228，说明
Q 排序并未重新塌成完全平局；然而 walk 在非最优的 2679 个状态上
`beta_walk=0.05000013`，null 在非最优的 791 个状态上
`beta_null=0.05000008`，两者仍贴下轨。因此失败不能归因于“本次 Q 没有任何排序”。

### 6.4 科学裁决

两条 gate 都从随机初始化开始，因此 current-transition 若能阻止 β 进入死区，5k
内就应与 replay 臂产生差异；结果没有。因此“replay 中的历史 option 稀释是主要
致因”在**预防入坑**意义上被否证。该实验不能回答一个已经饱和的 checkpoint 需要
多久才能恢复，但它不是从饱和 checkpoint 启动，不能据此称为无诊断力。

后续前向探针同时修正了直接失败机制：non-argmax advantage 的上推方向存在，但
raw β logits 已下降到 `-18/-21`，sigmoid 导数约 `1e-7`，有效梯度被表示层死区
吞没。因此准确因果链是：两 option 与 adaptive `xi=0.8*gap` 形成聚合净下压，
把 raw logits 推入 sigmoid 死区；死区随后阻断状态条件上推。前半段是 loss 动力学，
后半段是当前 HB 适配没有真正防止梯度饱和的表示层缺陷，不能笼统写成“完全不是
实现问题”。

按预注册停止规则：**不跑 3 seeds、不上多教师、不继续调 σ/ξ 抢救 termination**。
当前能够保留的结论是 fixed teacher 蒸馏通道在 hurdle 有效；自动 option/termination
尚未恢复，不能作为已验证贡献。若后续继续自动教师选择，应转向新的阶段条件
teacher-value/transferability 信号，而不是把 action compatibility 或 β 小修包装成
迁移性指标。

新增产物：

- W&B run：`d0dsnmf9`（project `ptf_fasttd3_classic_revisit`）；
- checkpoint：`models/h1hand-hurdle-v0__classic_ptf_hurdle_current_beta_s1_signal_gate_20260722T165814Z__1_final.pt`；
- frozen cross-section：`docs/data/classic_ptf_signal_diagnostic/current_beta_online_smoke_v3.json`；
- training log：`logs/train/classic_ptf_signal_smoke/classic_ptf_hurdle_current_beta_s1_signal_gate_20260722T165814Z.log`。

## 7. β 表示层防饱和最终 gate（预注册）

### 7.1 重启理由与唯一干预

后续零训练前向探针发现，current-transition 5k checkpoint 的 β-head raw logits 已
下降到约 `-18/-21`，sigmoid 导数只有 `1e-7` 量级；non-argmax advantage 虽然方向
正确，但传到 logit 的有效梯度约为 `1e-9`。这要求把“净下压动力”和“进入死区后
无法表达上推证据”分开。

唯一干预是在 current-transition + σ=0.5 配置上增加 `beta_logit_clip=4`。实现采用
**forward clamp + straight-through gradient**；普通 `torch.clamp` 在区间外梯度为
零，会制造新的 hard dead zone，禁止使用。该机制是 HumanoidBench 适配实验，不冒充
PTF 论文原式。

### 7.2 冻结判据

- seed 1、5k steps；其他训练参数完全不变；
- 主判据只看冻结 rollout 上的状态条件 termination：对 walk 和 null 分别要求
  `mean(beta | non-argmax) > mean(beta | argmax)`，且两项差值的平均至少为 `0.05`；
- 在线 `beta(current non-argmax)-beta(current argmax)` 只作一致性检查；
- 5k return 不参与裁决；
- 通过才允许讨论正/负任务可行性；失败则永久停止 classic-PTF termination 修复，
  不再更换 clip、ξ、学习率或 gate 阈值。

### 7.3 实验结果

200 步真实环境 wiring smoke 与 16 个定向测试全部通过。正式 gate 正常完成，W&B
run 为 `vhz57xza`。训练末期在线统计为：

| metric @ 4.9k | value |
|---|---:|
| rollout walk fraction | 0.8906 |
| current option is Q argmax | 0.5625 |
| β when current is argmax | 0.06618760 |
| β when current is non-argmax | 0.06618759 |
| cumulative β termination rate | 0.3313 |

累计 termination rate 较高来自训练早期 β 尚未完全下压时的历史累计，不能替代
训练末期条件 β 判据；末期两个条件已经相同并贴在 forward clip 对应的下边界。

固定 rollout 断面给出：

| option | mean β when argmax | mean β when non-argmax | difference |
|---|---:|---:|---:|
| walk | 0.06618760 | 0.06618761 | +1.49e-8 |
| null | 0.06618761 | 0.06618760 | -1.49e-8 |

两项 difference 的平均恰为 `0`，未达到预注册的 `0.05`，且 null 的方向错误。
与此同时 `Q_walk-Q_null=-0.1380`、walk argmax 比例 0.214，说明 Q 排序非常明确，
失败不能归因于 Q 平局。

STE 确实保留了区间外反向梯度，但 raw logits 仍被聚合目标持续推到约
`-45k/-43k`。这说明单纯消除 sigmoid 的局部导数死区并不足以战胜当前 objective
的全局净下压；普通 hard clamp 更会因区间外梯度为零而直接锁死。

### 7.4 最终裁决

最终 gate **失败**。classic PTF termination 在“单 walk source + null、FastTD3、
HumanoidBench hurdle”设置下没有恢复状态条件区分能力。证据支持的失败链是：

`adaptive-xi 的聚合净下压 → raw logit 下坠 → sigmoid/边界表示失效 → β 退化为
近常数 hazard`。

这不证明 PTF 在所有任务、所有 option 数量下普遍无效；可以对外声称的是：当前
单教师+null 复现中的 learned termination 未恢复，且 replay 时序修复与一次明确的
防饱和参数化都不足以解决。按照停止规则，本线不再调 clip、ξ、β 学习率或继续跑
正式 seeds；fixed-teacher 蒸馏的正结果继续保留，自动教师选择应转向新的阶段条件
teacher-value/transferability 机制。

新增产物：

- checkpoint：`models/h1hand-hurdle-v0__classic_ptf_hurdle_beta_ste_clip_s1_signal_gate_20260723T050338Z__1_final.pt`；
- frozen cross-section：`docs/data/classic_ptf_signal_diagnostic/beta_ste_clip_online_smoke_v4.json`；
- training log：`logs/train/classic_ptf_signal_smoke/classic_ptf_hurdle_beta_ste_clip_s1_signal_gate_20260723T050338Z.log`；
- W&B：project `ptf_fasttd3_classic_revisit`，run `vhz57xza`。

## 8. 多教师 termination 信号 gate（预注册）

### 8.1 重启理由与唯一假设

上一节只否证了“单 walk 教师 + null”设置下的两种局部修补，不能外推到 PTF 原本
面向的多 option 设置。两 option 时，adaptive
`xi=0.8*(Q_top1-Q_top2)` 使最优 option 的 clamped advantage 为
`+0.8 gap`，另一 option 仅为 `-0.2 gap`；实际 current option 又经常是 argmax，
因而 aggregate termination objective 强烈偏向降低 beta。增加 stand、walk、run
三个教师后，低于第二名的 option 可以得到幅度更大的负 advantage，可能自然增加
“当前 option 已非最优时终止”的训练信号。

本 gate 的唯一主要假设是：

> **多教师 option 集可以恢复 beta 的状态条件区分，使 current option 在非 argmax
> 状态下的 termination probability 明显高于 argmax 状态，而不是所有 option
> 一起退化到下界。**

### 8.2 固定设置与判据

- 任务：`h1hand-hurdle-v0`；
- option：stand、walk、run、null；
- compatibility sigma：沿用已校准的 `0.5`；
- termination：`current_transition`，不使用 STE clip，不再调 xi、学习率；
- seed 1、5k steps；5k return 不参与裁决；
- 主判据：训练末期在线
  `mean(beta_current | non-argmax) - mean(beta_current | argmax) >= 0.05`，
  且两项不能同时贴在 `0.05` 下界；
- 支持性判据：冻结 student rollout 上，至少两个同时具有 greedy/non-greedy
  样本的 option 满足 `mean(beta | non-greedy) > mean(beta | greedy)`，raw beta
  logits 不能全部坠入明显饱和区；
- 通过只表示 termination signal gate 通过，后续仍需与 fixed/yoked 调度比较才能
  判断调度是否改善学习；失败则停止 classic-PTF termination 修复，不跑正式 seeds。

### 8.3 5k gate 结果

200 步真实环境 wiring smoke 与 16 个定向测试通过。5k seed-1 gate 完成，W&B
run 为 `g2gqbzd0`。训练末期在线统计为：

| metric @ 4.9k | value |
|---|---:|
| `beta(current | argmax)` | 0.16145 |
| `beta(current | non-argmax)` | 0.21436 |
| 条件差值 | **+0.05292** |
| current option 为 argmax | 0.625 |
| 累计 beta termination rate | 0.14287 |
| 重选时真实换 option 比例 | 0.65468 |
| 当前 option 平均 age | 7.34 |

末点条件差值达到预注册的 `0.05`，且两组均未贴在 `0.05` 下界。4.0k--4.9k
十个断面的差值均值为 `+0.0145`、6/10 为正，说明末点通过但在线信号仍有明显
mini-batch/occupancy 波动，不能把单个末点包装成稳定性能证据。

固定 student rollout（2930 状态）进一步给出：

| option | argmax fraction | beta(greedy) | beta(non-greedy) | difference | raw logit mean |
|---|---:|---:|---:|---:|---:|
| stand | 0.126 | 0.227 | 0.556 | **+0.329** | +0.716 |
| walk | 0.118 | 0.357 | 0.412 | **+0.055** | -0.321 |
| run | 0.621 | 0.085 | 0.315 | **+0.230** | -4.301 |
| null | 0.134 | 0.545 | 0.418 | **-0.127** | -0.173 |

三个 source option 均恢复了正确方向的状态条件 termination，且 raw logits 没有
像单教师实验一样全部坠入数值死区；支持性判据通过。null 在冻结 source-free
occupancy 上方向相反，表明恢复并不完整，也提示 beta 的状态泛化仍可能依赖训练时
“哪个 option 当前正在执行”的 occupancy。

因此本 gate 的严格结论是：**多 option 确实修复了单教师+null 场景最严重的
termination 聚合净下压，原 PTF termination 不是在 HumanoidBench 上必然不可训练。**
它尚未证明 `Q_omega` 排序对应教师的未来学习价值，也尚未证明 learned scheduling
改善 return。

新增产物：

- W&B：project `ptf_fasttd3_classic_revisit`，run `g2gqbzd0`；
- checkpoint：`models/h1hand-hurdle-v0__classic_ptf_hurdle_loco3_current_beta_s1_signal_gate_20260723T053156Z__1_final.pt`；
- frozen cross-section：
  `docs/data/classic_ptf_signal_diagnostic/loco3_current_beta_online_smoke_v5_full.json`；
- training log：
  `logs/train/classic_ptf_signal_smoke/classic_ptf_hurdle_loco3_current_beta_s1_signal_gate_20260723T053156Z.log`。

## 9. 多教师 30k 性能可行性 gate（预注册）

5k 只回答“termination 是否获得训练信号”。下一项最小问题是：这个恢复的信号是否
至少没有把已知有效的 hurdle 早期迁移收益破坏。只跑一个 seed-1、0--30k 训练，
每 5k source-free eval；不改四 option、sigma、xi、学习率或 selector。

判序为：

1. 若 30k 时状态条件 beta 再次整体贴下界，判为 termination 信号不持久；
2. 若 beta 信号仍在但 5k--30k AUC 不高于同 seed scratch，判为“可训练但选择信号
   没有转化成正迁移”；
3. 若 AUC 高于 scratch，只称单 seed feasibility；与既有 fixed-walk 作描述性比较，
   通过后才考虑 matched control 和 3 seeds，不能直接形成性能贡献。

### 9.1 执行与评估口径

fresh seed-1 训练正常完成，W&B run 为 `m5wpw1fo`。`run_stop_step=30000`
与当前训练循环的常规 eval/save 时机组合，只产生 5k--25k 五个在线 eval 点；
30k final checkpoint 正常保存，但没有同协议的内置 endpoint eval。为避免把不同
reset 面板混入 AUC，主曲线据实改为各臂同口径的 **5k--25k normalized AUC**。
另用完全相同的 32-episode source-free 冻结面板评估四个 25k checkpoint，并单独
报告 30k multi-teacher final，不把它们混进在线 AUC。

### 9.2 性能结果

在线 return：

| step | multi-teacher | scratch | fixed-walk | old single-source learned |
|---:|---:|---:|---:|---:|
| 5k | 5.49 | 7.35 | 15.40 | 8.06 |
| 10k | 9.44 | 7.79 | 32.19 | 19.79 |
| 15k | 14.77 | 8.86 | 37.74 | 39.21 |
| 20k | 18.48 | 16.50 | 96.57 | 108.50 |
| 25k | 27.57 | 14.64 | 100.85 | 121.25 |
| 5k--25k nAUC | **14.81** | 11.04 | 56.16 | 58.04 |

multi-teacher 相对 scratch 为 `+3.77`，通过最低单 seed feasibility；但只达到
fixed-walk AUC 的约 26%，不支持“自动多教师调度优于已知好教师”。

同一 32-episode source-free panel 的 25k checkpoint：

| arm | return mean ± episode SD | root-x progress |
|---|---:|---:|
| multi-teacher | **31.08 ± 8.77** | 2.12 |
| scratch | 16.96 ± 10.49 | 0.89 |
| fixed-walk | 100.07 ± 18.60 | 8.25 |
| old single-source learned | 129.85 ± 38.59 | 7.97 |

30k multi-teacher final panel 为 `39.50 ± 13.28`，root-x progress 为 `2.73`。
这些是单 train-seed 的 episode 面板统计，不是跨 train-seed 显著性证据。
Hurdle 的 `terminated` 语义不能安全当作 success，本表不使用 evaluator 输出的
`success_count`。

### 9.3 机制结果与根因收窄

30k 末期在线量：

- `beta(current | argmax)=0.0686`，
  `beta(current | non-argmax)=0.2102`，条件差 `+0.1417`；
- current option 是 Q argmax 的比例 `0.805`；
- 累计 beta termination rate `0.181`，重选时真实换 option 比例 `0.647`；
- option 占比：stand `0.515`、run `0.296`、walk `0.132`、null `0.057`；
- action compatibility：stand `0.198`、run `0.109`、walk `0.083`。

固定 final-student rollout 上，stand/walk/run 的
`beta(non-greedy)-beta(greedy)` 分别为 `+0.645/+0.359/+0.547`；termination
信号不但没有再次贴轨，反而已经很强。与此同时，Q argmax 仍主要是 stand
（54.9%），walk 仅 14.9%。已知 fixed-walk 在同 seed 上远好于 multi-teacher，
但当前 student 与 stand 动作最相似、与 walk 最不相似。因而当前 evidence 支持：

1. 多 option 修复了两 option termination objective 的聚合净下压；
2. beta 已能按 Q 排序结束非最优 option，termination 不是这条 run 的主要失败点；
3. `Q_omega` 的优化目标仍是 target-reward TD return，不能表述成“直接学习动作
   相似度”；action compatibility 只决定每个 source 的 Q 能从哪些 student
   transition 获得更新。这里观测到 stand 同时具有较高 compatibility 和较高
   Q/占比，只能说明当前支持域构造可能使 Q 的估计受到 student occupancy 与动作
   支持域的强烈影响，不能据此把 Q 的定义改写为相似度；
4. 更准确的失败点是 estimand/credit assignment：官方式 Q 试图估计当前
   call-and-return option 下的 target return，却不直接衡量“现在接受该教师的
   蒸馏更新后，student 的未来学习速度会增加多少”。本 run 最终主要选择
   stand/run，并稀释了已知有效的 fixed-walk 蒸馏，但这既可能来自支持域偏置，
   也可能来自“当前稳定回报”和“未来学习增益”的目标错位，现有单 run 尚未分离。

严格裁决：**classic PTF 多教师 termination trainability gate 通过，单 seed
正迁移 feasibility 也勉强通过，但 teacher-value / automatic selection gate
失败。** 当前不应再修 beta，也不应直接扩到三种子；下一项机制工作必须针对
`Q_omega` 的教师价值 estimand，而不是继续调 termination 超参数。

新增产物：

- W&B：`m5wpw1fo`；
- final checkpoint：
  `models/h1hand-hurdle-v0__classic_ptf_hurdle_loco3_current_beta_s1_30k_gate_20260723T054211Z__1_final.pt`；
- final option diagnostic：
  `docs/data/classic_ptf_signal_diagnostic/loco3_30k_gate_final_v7.json`；
- 30k source-free panel：
  `docs/data/classic_ptf_signal_diagnostic/loco3_30k_gate_source_free_panel_v8.json`；
- matched 25k panels：
  `docs/data/classic_ptf_signal_diagnostic/{multi,scratch,fixed,old_learned}_step25000_source_free_panel_v9.json`；
- training log：
  `logs/train/classic_ptf_signal_smoke/classic_ptf_hurdle_loco3_current_beta_s1_30k_gate_20260723T054211Z.log`。
