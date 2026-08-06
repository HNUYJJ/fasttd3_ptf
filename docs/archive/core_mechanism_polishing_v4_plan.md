# 核心机制继续打磨 v4：论文成稿前的机制冻结门

日期：2026-07-12

状态：**下一阶段工作协议；尚未进入完整论文写作**

上位证据：[`paper_core_contribution_reconstruction_v3.md`](paper_core_contribution_reconstruction_v3.md)
与[`rbo_core_result_registry_v1.yaml`](../configs/experiments/rbo_core_result_registry_v1.yaml)

## 1. “成稿四件套”不等于现在直接写完整论文

Abstract、Method Figure、主结果表、Claim--Evidence Ledger的作用，是把方法压缩到审稿人能检查的
形式，反向暴露以下问题：

- 一句话是否说得清算法新在哪里；
- 方法图中的每个模块是否真在默认算法里；
- 每项贡献是否有直接消融，而不是依靠事后解释；
- 正例是否有负例和适用条件；
- return改善是否有task-specific progress支撑。

如果四件套无法自洽，应该返回机制层继续重构，而不是开始铺写Introduction和Related Work。

## 2. 当前机制仍需打磨的四个核心问题

### 2.1 RBO为什么不只是“把teacher数据塞进buffer”

需要把科学对象正式写成 **source-conditioned experience acquisition**：冻结source改变目标MDP中的
有限预算behavior occupancy，所有transition重新获得target reward，再由off-policy learner完成目标策略
改进。需明确：

- source不提供目标任务标签或参数初始化；
- 迁移变量是有限预算数据分布，而非最终策略依赖；
- student/source mixture同时保证自主探索与外部occupancy覆盖；
- source withdrawal后仍可能通过replay persistence继续影响更新。

这应形成主方法的数学定义、伪代码和与scratch/uniform bootstrap的最小算法差异。

### 2.2 `T⁰`究竟是什么指标

不再尝试把`T⁰`包装成通用transferability。下一版只允许它回答：

> 在已经决定使用某个source bank的前提下，有限teacher budget应如何在bank内部相对分配？

需要利用已有数据、无需新训练，完成两项离线分析：

1. 定义bank separability，如score dispersion、softmax entropy或effective source count；
2. 检验separability与`RBO−uniform`收益是否在terrain、breadth和第二批任务上同向。

若相关性不足，就把“分化度预判加权价值”降为经验观察；若稳定，再把它提升为source-bank allocation
principle。低分绝对值不再承担go/no-go，powerlift与basketball已经给出双向反例。

### 2.3 source library管理如何成为solid贡献

truck `+229.9`与maze `+0.3`说明probe增量不是充分条件。需要给两个条件可操作定义：

- **complementarity**：新source相对现有bank提供多少新增行为/score覆盖；
- **remaining headroom**：目标是否已被现有bootstrap方法推入近似饱和区。

然后用已有冗余扩源、hurdle→truck和hurdle→maze三组结果构成正反例链。若headroom只能事后从结果表
读取，就诚实定位为selection-after-existing-runs的管理准则，而不声称训练前预测器。

### 2.4 负迁移安全仍未解决

execution/occupancy与replay/update双通道是可靠机制发现，但不是已完成的universal safety algorithm：

- crawl支持replay persistence；
- split支持actor--critic sampling coherence；
- slide支持OBRW局部价值；
- basketball否证OBRW exact fallback；
- SHU否证behavior score直接决定replay admission。

因此双通道应作为“为什么负迁移难、现有控制何时有效”的机制贡献；可信data utility与exact fallback
仍是限制/future work，不能靠重新命名掩盖。

## 3. 最终贡献层级必须固定

| 层级 | 内容 | 论文角色 |
|---|---|---|
| 主算法 | static RBO/WFix | 唯一默认method |
| 方法规律 | bank separability决定加权边际价值 | 待离线审计后定强度 |
| source-library规律 | complementarity × remaining headroom | 有正反例的管理原则 |
| 机制分析 | execution/replay双通道 + AC coherence | explanation与optional OBRW |
| supporting | MCG gate/distillation | appendix，不进入默认算法 |
| limitation | exact fallback、data utility、automatic horizon | 明确未解决 |

不能再把每个历史实现都列成并列核心贡献，也不能让论文标题暗示默认方法包含OBRW或MCG。

## 4. 机制冻结门

进入完整论文写作前必须同时通过：

1. **Algorithm gate**：主方法公式、伪代码和真实运行配置一一对应；
2. **Novelty gate**：能清楚区分RBO与普通offline demonstration injection、behavior cloning和uniform
   bootstrap；
3. **Metric gate**：`T⁰`只承担被数据支持的相对allocation决策；
4. **Causal gate**：selection、horizon、bootstrap、execution/replay与actor/critic分布已有对应消融；
5. **Skill gate**：return主表与hard-progress证据分开，hurdle/cabinet/maze作技能锚点，powerlift不冒充
   举重成功；
6. **Boundary gate**：stair、crawl、basketball、window与balance_hard null均进入主文边界；
7. **Reproducibility gate**：所有headline数字可由result registry追溯。

任何一项失败，先修机制表达或复用已有数据做离线分析；只有出现唯一且阻断性的证据缺口，才讨论一个
最小新增实验。

## 5. 下一步顺序

1. 形式化RBO的behavior-mixture、occupancy与replay作用路径；
2. 用现有probe/result registry完成bank separability离线审计；
3. 把complementarity/headroom从事后描述收窄为可操作判据；
4. 生成四件套作为机制压力测试；
5. 通过机制冻结门后，才写完整Abstract/Introduction/Method/Experiments。

因此当前不是“停止打磨、直接写论文”，而是从漫无边界的算法试错转入**以论文可证伪性为约束的机制
打磨**。
