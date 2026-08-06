# 标签可识别性审计 v1 — 结果与裁决

> 日期：2026-07-27  
> 性质：**一次性、零训练、零 GPU、只读**审计。承接 `CABINET_UNCERTAIN` 的收尾指令。  
> 问题：已有 student/scratch 数据里，是否存在一个**比 Cabinet 更可测**的 task × stage？  
> **裁决：`CANDIDATE_FOUND` — `h1hand-door-v0` @ 10k→20k**（附明确风险，见 §6）

## 1. 为什么必须先做这一步

Cabinet gate 的失败不是"三种 source 效应相同"，而是**在 3 seed × 32 episode 的预算下
标签不可分辨**。因此在投入下一次 source 标定之前，必须先确认场地本身是否可测；
否则再做一次就是重复同一个测量失败。

关键约束：这个筛选**只能用无源臂数据**。若先看各 source 的效果再挑任务，
就成了结果导向的任务筛选，后续任何标签都失去可信度。

## 2. 盲化如何被强制（不依赖实验名猜测）

判据取自训练代码自己打印的事实，而非命名约定：

```
fasttd3_ptf/official_fasttd3_ptf/train_ptf.py:989
    print(f"Loaded source bank options: {source_bank.names()}")
```

| bank 内容 | 判定 | 处理 |
|---|---|---|
| `['null']` | 无源臂 | 解析回报曲线 |
| 含任何非 null 条目 | 有源臂 | **立即中止读取该文件**，只计数 |
| 该行缺失（旧代码路径） | unknown | 排除，不用于任何结论 |

扫描 563 个 wandb run：**source 350（跳过，回报数值从未被解析）/ nosrc 103 /
unknown 42 / 无 env 68**。

> **执行中修正的一处自身缺陷（必须记录）**：本审计第一版用实验名关键词分类，
> 关键词表漏了 `stand`/`walk`/`run`/`obrw`/`dm_`/`sd_`，导致 `cabinet_sd_stand` 等
> **有源臂被误判为 unknown 并混入统计**（cabinet 样本从 9 被虚增到 25，pole 从 3 到 18）。
> 改用上述硬判据后全部重算。本文所有数字来自硬判据版本。

## 3. 指标链

训练中的 `eval_avg_return` 是 `num_eval_envs=128` 的**面板均值**，而 Cabinet gate 的
评估面板是 32 episodes，故

$$\mathrm{SE}_{32}=\sigma_{\text{episode}}/\sqrt{32}=2\,\sigma_{\text{panel}}(128)$$

gate 判据是 3 个配对 seed 的 90% t 区间（df=2）不跨 0，故**最小可判定效应**

$$U_{\min}=t_{90}\sqrt{2}\,\mathrm{SE}_{32}/\sqrt{3}=2.384\,\mathrm{SE}_{32}$$

这是**乐观下界**——真实 `sd(U)` 还含 learner-seed 异质性，只会更大。

`sigma_panel` 用两个互补估计量，**取较差者**作保守值：

- **一阶差分 MAD**：对平滑上升曲线会把学习增量误当噪声 → 高估
- **10k–30k 去趋势残差 MAD**：对曲率大的曲线会把曲率误当噪声 → 高估

两者都只会高估噪声，故 `max(·,·)` 是保守方向。

**校准**：由训练曲线预测 cabinet 的 `SE_32 = 19.25`，实测 12 臂 32-episode 面板
SE 中位 `15.62` → 比值 1.23，同量级且偏保守，校准通过。

### 两个已实测的锚点

| 锚点 | 保守 U/\|r\| | 保守 U/trend | stage 相对学习率 | 实测结局 |
|---|---:|---:|---:|---|
| **crawl** | 0.273 | **0.83** | 32.7% | 干净标签（成功） |
| **cabinet** | 1.191 | **10.31** | 11.6% | `CABINET_UNCERTAIN`（失败） |

`U/trend` = 效应需达到"10k→20k 自然学习增量"的多少倍才能被判定。**这是主判据**，
因为它同时吸收了噪声和学习速率两个维度。

## 4. 全表（仅无源臂数据）

| env | n | 保守 U/\|r\| | **保守 U/trend** | seedCV | r@20k | r@end |
|---|--:|--:|--:|--:|--:|--:|
| crawl *(成功锚点)* | 5 | 0.273 | **0.83** | 0.046 | 641.8 | 949.4 |
| balance_hard | 8 | 0.156 | **1.05** | 0.044 | 67.0 | 105.0 |
| pole | 3 | 0.726 | 1.17 | 0.266 | 298.5 | 845.7 |
| maze | 6 | 0.252 | 1.20 | **0.349** | 220.6 | 348.7 |
| **door** | 5 | 0.316 | **1.42** | 0.101 | 247.2 | 332.1 |
| truck | 5 | 0.250 | 1.61 | 0.050 | 1025.9 | 1356.7 |
| slide | 3 | 0.971 | 2.09 | 0.184 | 43.4 | 749.7 |
| window | 9 | 1.522 | 2.27 | 1.398 | 38.3 | 517.8 |
| hurdle | 16 | 0.990 | 2.40 | 0.213 | 16.5 | 597.5 |
| powerlift | 6 | 0.211 | 2.67 | 0.109 | 132.8 | 296.6 |
| spoon | 5 | 0.630 | 3.29 | 0.184 | 258.8 | 358.1 |
| basketball | 3 | 1.868 | 3.44 | 0.435 | 35.9 | 256.8 |
| stair | 3 | 1.234 | 3.57 | 0.050 | 47.6 | 513.1 |
| bookshelf_simple | 3 | 0.899 | 3.95 | 0.375 | 685.2 | 810.4 |
| cabinet *(失败锚点)* | 9 | 1.191 | **10.31** | 0.242 | 50.3 | 227.2 |

## 5. 逐项淘汰理由

| env | 淘汰理由 |
|---|---|
| **balance_hard** | 统计上最优（U/trend 1.05，仅次于 crawl），但见 §5.1 —— **因已有效果先验而排除** |
| **maze** | `seedCV=0.349` 是**真双峰**，非批次效应：3 条 seed 在 20k 达 ~300，另 3 条卡在 ~145 直到 95k 才追上。双模态会直接主导 `sd(U)`，对 3-seed 配对是致命的 |
| **powerlift** | 10k–30k **完全停滞**（118→141、113→135、128→136，20k–30k 基本平坦，直到 95k 才跳到 ~300）。相对学习率 7.9%，**低于 cabinet 的 11.6%**——与 cabinet 同病 |
| **pole / spoon / slide / window / stair / bookshelf / basketball** | 保守 U/trend ≥ 1.17 且噪声或 seedCV 明显劣于 door；window 的 `seedCV=1.398` 尤其失控 |
| **truck** | U/trend 1.61 劣于 door；且有源臂暴露 34 个，为全表最高之一 |
| **hurdle / crawl / cabinet** | 已归档，非候选 |
| **basketball** | 保留为外部 abstention 测试，**任何情况下不得选** |

### 5.1 balance_hard：一处必须披露的先验污染

balance_hard 在两个归一化上都排第一（U/\|r\| 0.156、U/trend 1.05），seedCV 0.044 最低，
8 条曲线在 20k 处密集落在 60.0–70.0，是全表最干净的数据。**但审计过程中在其 bank
配置的注释里读到了一条已有的效果先验**：

```yaml
# configs/source_banks/h1hand_loco_sources_balance_hard.yaml
# balance_hard 在 Transfer Map 上是"全员 OOD"任务(所有 loco 源 zero-shot 全摔/负迁移)。
# SC-MCG 的正确行为=null gate 全关, 退化到≈scratch(不伤害)。这是 negative
# transfer control 的实证场。
```

**排除理由（PI 2026-07-27 更正后的准确表述）**：balance_hard **已被项目保留为负迁移
控制场**，而本轮需要的是可能具有**任务内异质性**的场地；把控制场挪作标签采集会破坏
该用途。此外我在审计中已经看到了这条先验，继续选它会引入结果导向筛选的嫌疑。

> **一处必须更正的推理错误（我原先写错了）**：不能由"zero-shot 全负"推出
> "RBO delayed learning utility 全负"。本项目已反复证明**行为即时效果 ≠ 后续学习价值**
> （这正是 T⁰、T^critic sign、influence gate 等一系列信号族被封存的共同原因）。
> 因此"source 会摔倒"不能断言三个 RBO 标签必然同号为负。排除 balance_hard 的正当理由
> 只有上述"它是保留的负迁移控制场"，不是效应方向的预测。

door 的两个 bank 注释仅含 obs adapter 的技术说明，无效果先验。**但 door 并非完全盲态候选**
——见 §6.0。

## 6. 推荐：`h1hand-door-v0` @ 10k→20k

### 6.0 定位更正：Door 不是完全盲态候选（PI 2026-07-27）

我原先写"door 盲化干净"是**不准确的**。项目旧记录已载有 door 的**行为层**迁移先验：

| 来源 | door zero | stand | walk | run |
|---|--:|--:|--:|--:|
| `docs/archive/transfer_map_v1_analysis.md:20` | 64 (0% 摔) | 59 | **25（62% 摔，负迁移）** | **101（+56%）** |

该文档同时把 door 列为"推进型对价"任务：*run 的前倾步态推进 approach 段*。
因此本实验必须定位为：

> **有历史行为先验的定向 RBO 学习效用标定**，**不是**盲测，**不是**外部验证。

之所以仍然值得做，是因为旧证据与新标签测的**不是同一件事**：

- 旧证据 = source 在 target 上的**即时行为效果**（zero-shot 执行回报）；
- 历史 RBO 结果 = 多源 WFix/OBRW，**不是**标准化的单源等剂量标签；
- 新标签 = $U_i(10k,10k)=J^{sf}_{@20k}(\text{source}_i)-J^{sf}_{@20k}(\text{student})$，
  即 reward-bearing replay 的**延迟学习效用**。

本项目已反复证明**行为即时效果 ≠ 后续学习价值**（T⁰、T^critic sign、influence gate
等信号族被封存的共同原因就是把两者混同）。所以这次实验回答的是一个尚未被回答的问题：

> **door 上已知的行为层 run/walk 异质性，是否会转化为 reward-bearing replay 的学习效用异质性？**

这也意味着：若结果与行为先验方向一致，那是一个**正面但非独立**的证据；若方向相反或
全同号，同样是有信息量的结果——两种情况都不能当作对"迁移性指标"的外部验证。

### 为何比 Cabinet 更可测（量化）

| 维度 | cabinet | **door** | 改善 |
|---|--:|--:|--:|
| 保守 U/trend（主判据） | 10.31 | **1.42** | **7.3×** |
| 保守 U/\|r\| | 1.191 | **0.316** | **3.8×** |
| stage 相对学习率 | 11.6% | **22.2%** | 1.9× |
| 跨 seed 变异系数 | 0.242 | **0.101** | 2.4× |
| 保守 SE_32 | 25.16 | 32.71 | —（绝对尺度不同，看相对量） |

机制上的直接对照——两者各 5/9 条无源曲线在 10k–30k 的表现：

```
door  s1 216 → 270 → 287 → 303 → 309     单调、平滑
      s2 191 → 237 → 247 → 268 → 289
      s3 161 → 253 → 282 → 284 → 284
cabinet s1  37 →  35 →  65 →  18 →  53   非单调、剧烈跳变
        s2  90 →  62 →  49 →  73 →  79
        s3  23 →  60 →  35 →  29 →  95
```

cabinet 在 10k–30k 的自然学习增量只有 **5.82/10k 步**（其学习实际发生在 50k 之后：
50k→95k 从 ~120 到 ~230），door 是 **54.92/10k 步**。cabinet 的 stage 选在了一个
**学习停滞窗口**里——这是它失败的第二个机制，独立于重尾。

### 资产缺口与成本

| 资产 | 状态 |
|---|---|
| obs adapter | **已验证**：`hb_robot_qpos_qvel, qpos_dim=78, output_dim=151`（door 已跑过 12 个有源 run） |
| 合并 source bank | **已存在**两版：`h1hand_door_loco_sources.yaml`、`h1hand_loco_sources_door.yaml` |
| 三个独立单源 bank | 需新建 —— **纯复制粘贴**，照搬 `calibration/h1hand_cabinet_rbo_*.yaml` 模板改 `qpos_dim=78` |
| 10k anchor × 3 seed | 需新建（cabinet 同规模实测约 4.5 min/个） |
| target-evidence 契约 | 需新建（**仅用于 10k 决策点的离线建模数据，不参与本实验的源选择**） |
| 训练矩阵 | 3 anchor + 12 臂（stand/walk/run/student × 3 seed）+ 12 次冻结评估 |
| 估计成本 | **与 Cabinet gate 同量级，约 3–5 GPU 小时**；双卡并行 wall-clock 约 1.5–2 h |

## 7. 必须一并说清的局限

1. **door 未达到成功锚点的水平。** 保守 U/trend 1.42 vs crawl 的 0.83——door 需要的
   相对效应是 crawl 的 1.7 倍。它只能保证**比 cabinet 好 7.3 倍**，不能保证成功。
2. **没有任何候选达到 crawl 的水平。** 全表最好的非归档候选是被排除的 balance_hard(1.05)。
3. **`U_min` 是乐观下界。** 它假设 `sd(U)` 仅由面板噪声构成；真实的 learner-seed
   异质性会把它推高。door 的 seedCV 0.101 说明这一项不大，但不为零。
4. **重尾性未被直接测量。** 128-env 面板均值经中心极限定理已抹平单 episode 尾部，
   落盘数据在**原理上**无法直接测单 episode 回报分布的重尾程度。本审计绕开的方式是
   直接估 `sigma_panel` 并按 √(128/32) 换算到 32-episode 面板——该换算已用 cabinet
   实测校准（比值 1.23）。
5. **无天然盲化的候选存在。** 所有有 scratch 数据的 target 都跑过有源臂
   （door 12 个，为全表最少之一）；而从未跑过的任务（`bookshelf_hard`、`cube` 等）
   **没有任何 scratch 数据**，无法预判可测性。这是一个结构性的两难。本审计的筛选
   本身只用了无源臂数据，方法上干净，但**残留风险是项目历史文档中可能已含 door 的
   迁移结论**——我在本次审计中未查阅，PI 应自行判断这是否构成污染。
6. **任务内异质性能否建立，本审计无法预知**，也不允许作为筛选依据。door 通过的是
   **可测性**筛选，不是"会出现异质性"的预测。若 door 也返回全同号，那将是关于
   loco 源普适性的结论，而不是又一次测量失败。
7. **door 仍不增加任务族多样性之外的保证。** 它是 loco-manipulation，与现有
   hurdle/crawl（纯 locomotion）不同族，这对"模型是否只在识别 target task"有帮助，
   但单个新任务不足以消除任务混杂。

## 8. 产物

- 审计脚本（可复现，只读）：`scripts/analysis/label_identifiability_audit_v1.py`
- 数据：`docs/data/label_identifiability_audit_v1/label_identifiability_audit_20260727.json`
- 数据源：563 个 wandb run 的 `files/{config.yaml,output.log}`（未修改）

## 9. 裁决

**`CANDIDATE_FOUND`** — 存在一个明确优于 Cabinet 的候选场地：
**`h1hand-door-v0`，stage 10k→20k**，保守 U/trend 优于 cabinet **7.3 倍**。

按预注册的决策规则，这不触发"停止补任务、退回已证主线"的分支。但 §7.1 的风险是
实质性的：door 只是**比失败锚点好得多**，并未达到成功锚点。是否投入这 3–5 GPU 小时，
以及是否接受 §7.5 的盲化残留风险，由 PI 决定。**本审计不启动任何训练。**
