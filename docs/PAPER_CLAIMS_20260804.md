# 论文主张 — 证据 — 边界 逐条映射

> 2026-08-04。P3 的基础件：**每一条打算写进论文的话，都必须在此表中有一行**，
> 且该行给出可复现的裁决输出与不可省略的边界。
> 定位见 `/goal`：**已测试代理信号的经验性失败刻画（主）+ 直接测量方法（次）**。
> **2026-08-06 表述降级**：原写「不可能性刻画」与「最小充分方法」均过强——
> 前者无形式证明，后者未排除 best-of-N 与总交互成本优势。
>
> 状态标记：`READY`=证据完备可写 / `PENDING`=实验进行中 / `WEAK`=证据不足，须降级或删

## 0. 论文的一句话

> 在已检验的十二类代理信号上，迁移效用无法由 `(source, target)` 的即时量预测；
> 而准入与选源可以用**可量化的短跑交互**直接测出来。

两半互为因果：经验性失败刻画给出"为什么要付代价"，直接测量给出"代价是多少"。
**注意**：这不是不可能性定理——未证明不存在任何可行的预测器，
只证明已测试的十二类都失败了。

---

## 1. 支柱 I：已测试代理信号的经验性失败刻画

| # | 主张 | 证据 | 必须同时写出的边界 | 状态 |
|---|---|---|---|---|
| I-1 | **十二类已测试代理信号的经验性系统失败刻画**（2026-08-06 表述降级） | `impossibility_characterization_of_transfer_prediction_20260730.md` §2 | **不是不可能性定理**，无形式证明。除非有形式证明，一律不使用一般意义的 "impossibility" 措辞；主张限于"已检验的十二类信号、只依赖 `(source,target)` 的即时量" | `READY` |
| I-2 | **仅读 reward specification 的预测器**无法区分 slide 与 stair（2026-08-06 收窄） | slide 与 stair 共用同一份 `ClimbingUpwards.get_reward`、常量逐字节相同，walk 的效用为 **+56.95** vs **+0.19 [−5.35,+5.72]** | **该反例只反驳"只读 reward 规格"的预测器**。slide 与 stair 的地形几何、transition dynamics、初始状态分布、MJCF 均不同，故**不能**反驳读取完整静态 task specification 的预测器。原表述"静态规格无法区分"过强，已更正 | `READY` |
| I-3 | 第二条原理性反例：位移阈值的可行区间为**空集** | `progress_screen_v1_results_20260804.md` §3.1：需 `14.302 < θ < 1.814` | 只覆盖 locomotion target（进度=位移）；未测其他 progress 定义 | `READY` |
| I-4 | 失败不限于"排序"，连最弱的**单向排除**也不成立 | 族 12 `HOLDOUT_FAILED`，反向 **7.9×** | 单批、三个 target 真值已知 | `READY` |
| I-5 | 相似性是对称的，迁移效用是非对称的 | sibling gate：**slide 源→stair target** `+15.40 [+5.72,+25.08]` 3/3 正；**stair 源→slide target** 反向 `−20.79 [−31.61,−9.97]` 0/3 正（原文误写为 "stair→stair"，2026-08-06 更正；出处 `sibling_source_gate_v1_results_20260729.md` §方向 1/2） | **两个方向的稳健性不对称，必须分别陈述**：负方向（`−20.79`）稳健——sibling 臂有 2.5–3.3pp 的**剂量优势**却仍然输，剂量无法解释；正方向（`+15.40`）**须打折**——剂量优势与胜出同向，本实验不能排除它贡献了一部分。单一任务对 | `READY` |
| I-6 | 行为量与学习价值可**反号**，非仅弱相关 | door：run 位移 +58% 却 harmful，walk −61% 却最不负（M19）；族 12：7.9× 反向 | door 的标签本身跨 learner 不稳（M31），引用时须并列说明 | `READY` |
| I-7 | 正确的形式是条件分布而非点函数 | `U ~ p(U \| source, target, θ_t, D_t, occupancy_t, channel, dose, K)` | 这是刻画不是定理；由 §3.3 的三类证据支撑 | `READY` |

## 2. 支柱 II：直接测量（原称「最小充分测量」，2026-08-06 降级）

> `K*=10000` 只能称 **已测试预算中最小的稳健 horizon**（tested budgets = {2k, 5k, 10k}），
> **不是**数学意义上的 minimum sufficient measurement——未证明更小的 K 一定不可行，
> 也未排除 best-of-N 与总交互成本优势（后者需 P8 的完整对照）。

| # | 主张 | 证据 | 必须同时写出的边界 | 状态 |
|---|---|---|---|---|
| II-1 | 选源可用 `K=10000` 的短跑测出；该值是**已测试预算 {2k,5k,10k} 中最小的稳健者** | `RACING_VIABLE`，两批独立各 3/3；`K=5000` 不稳健（批1 3/3 → 批2 1/3） | 单 target（hurdle）、单候选集合 | `READY` |
| II-2 | racing 测的是延迟学习效用，不是行为质量 | K≥5000 的 **12/12** 运行排出 `walk > stand`，与 zero-shot 行为排序**相反** | 仅 hurdle | `READY` |
| II-3 | 选对源确实比选错源快 | `SELECTION_VALUABLE`：θ=200 中位 **4.95×**（3/3）；θ=300 **3/3 右删失** | 单 target、单批 3 seeds（M24） | `READY` |
| II-4 | 同一次测量还能判**准入**（是否全部拒绝） | `ADMISSION_VIABLE`：`false_admit=0`、`false_reject=0`；crawl 9/9 显著负 | 三个 target 真值均已知 → 是**在已知场地检验判据**，非发现新事实 | `READY` |
| II-5 | 准入不可被选源替代 | crawl 上 argmax 跨 seed 完全不一致（s1 stand / s2 run / s3 walk）——**全负时 argmax 是噪声** | 与 M32 不矛盾：M32 说的是存在有用源时源间差稳健 | `READY` |
| II-6 | 决策裕度远离阈值，结论不靠边界情形 | 最小裕度 crawl s1 `−59.75`、slide s1 `+45.06`，为阈值的 **3.8×–12.7×** | 判据不控制多重比较，靠裕度而非 FWER | `READY` |
| II-7 | 代价可量化 | **理论最小** `N_src×K = 30k`（并行墙钟 10k）vs 选对源节省 67k → 净 +37k | 见下方 §2.1 的口径澄清；收益取自 hurdle 已测曲线，不跨 target 外推 | `READY` |
| II-8 | 三零件组合成的自动系统，其 9/9 自动决策与已知真值一致，且在 learner 步数对齐下优于 scratch 与盲目用源（各 3/3） | `ENDTOEND_SUPPORTED`（`endtoend_v1_results_20260806.md`） | **必须并列口径 2**：全额计入 racing 成本后 slide 输 0/3、crawl 输 2/3，仅 hurdle 仍赢（+313.3）。见 II-9 | `READY` |
| II-9 | **正确表述是"最终性能更高且更稳"，不是"提升样本效率"** | 口径 2：hurdle +313.3 (3/3)、slide −252.1 (0/3)、crawl −20.9 (1/3)；方差 hurdle C sd=11.4 vs B sd=331.8 | 方差比较未预注册，仅描述性；slide 的 `C−A` 裕度不均（s1 +15.6 / s3 +46.0 相对 scratch 自身 sd 167.3 很小，换批有翻转风险） | `READY` |
| II-10 | 准入在坏 target 上**必然**略劣于纯 scratch，其价值是避免灾难性负迁移 | crawl：C≡A 但多花 40k → 口径 2 净 −20.9；同时 C−B = +66.3/+149.3/+237.4 | 事先在预注册 §8 声明，非事后解释。**须并列 II-11 的饱和度背景** | `READY` |
| II-11 | crawl 作为准入例证的边界 | **饱和度数字已降级**：`984.9` 来自单个 checkpoint（`endtoend_v1/.../scratch_s3_step100000`），是 `best_observed_return` 而非固定预算多 seed 均值，**不能证明稳定接近上限**。在 evaluator v2 重评前，`98.5% of theory max` 一律标 `PROVISIONAL`，不得写入论文 | **可保留的部分**：源在**早期**（K=10k，9 个 source–learner cells 方向全负，`−44.06`~`−258.77`）与**终点**（100k，`−151.0`）都有害，故 source mismatch 解释成立。**但 9 个 cells 共享同一 student 基线，独立单位是 3 个 learner seeds，不得写成"9/9 独立显著"**（M16） | `PROVISIONAL` |

### 2.1 racing 成本的两个口径（自查发现的不一致，必须同时写出）

`racing_min_horizon_v1` 预注册的公式是 `racing 成本 = N_sources × K* = 3 × 10k = 30k`，
**不含 student 臂**；而 `endtoend_v1` 记的实现成本是 `4 × 10k = 40k`。二者都对，
但含义不同，论文里**不得混用**：

```
理论最小成本 = 3 × K = 30k
    前提：racing 的 4 条臂本身就是主训练的前 K 步，最终选中的那条可直接续训。
    admit  时丢弃 2 条源臂 + 1 条 student 臂 → 净额外 3K
    reject 时丢弃 3 条源臂，student 臂续训   → 净额外 3K
    两种情况都是 3K。

本实验的实现成本 = 4 × K = 40k
    endtoend_v1 未实现"从 racing 臂续训"，主训练是从 t=0 重跑的，
    故 4 条臂全部成为额外开销。这是**工程实现的简化，不是方法的固有代价**。
```

**论文写法**：主文报理论最小 `3K` 并说明其前提；实验章节如实报本实验用了 `4K`，
并注明差额来自未实现续训。**不得**只报 `3K` 而用 `4K` 的实验结果，反之亦然。

## 3. 必须同时报告的负面/限制结果（不得省略）

| # | 内容 | 证据 |
|---|---|---|
| N-1 | **跨任务加速不成立**：hurdle 4.38×/3.59×，slide `SPEEDUP_REFUTED`（100k 被反超 2.7×） | `slide_speedup_v1_results_20260804.md` |
| N-2 | hurdle 的加速在 100k 衰减到 **1.24×**，且训练不稳定为 source 臂独有 | `hurdle_speedup_v1_results_20260730.md` |
| N-3 | 全程恒定剂量是**本项目自设**的配置缺陷，不是 PTF 原文用法；修它属工程基线 | `HARD_EXIT_SUPPORTED`（+631.8±9.3）；PTF 2020 的 λ 线性衰减 |
| N-4 | `U` 的**符号**可跨 learner 反转 | door：18/18 负 → 新批 2/9 正，`s9` run `+36.32±3.95` 显著正（M31） |
| N-5 | per-seed `U` 数值不可当可复现真值 | 同协议重跑 `\|ΔU\|` 中位 24.23、最大 43.78，而效应量本身仅 −7~−43（M27） |
| N-6 | 准入**不会**在坏 target 上提升性能；它花 `N×K` 步换取避免灾难性负迁移 | `racing_admission_v1_results_20260804.md` §7 |
| N-7 | 未解决"前人解不了的任务"——hurdle 虽是 TD-MPC2 解不了的，但 FastTD3 本身能解 | `EVIDENCE_STATE` §5 |

## 4. 全局适用范围（每次投稿都要写全）

```
机器人      仅 h1hand（76 DoF / 61 执行器），未测其他形态
target     hurdle / slide / crawl（+ door、cabinet 等作为负例或已关闭场地）
           **跨任务正面加速目前只有 hurdle 一个**
           **饱和度全部 PROVISIONAL（2026-08-06 降级，不得写入论文正文）**：
           此前的 crawl 98.5% / slide 95.1% / hurdle 85.1% 取自
           `best_observed_return` —— 跨 method/seed/step 的单点最大值，
           构成 winner's curse。它**只能证明"曾有 checkpoint 达到过"，
           不能证明固定预算、多 seed、固定方法下稳定接近上限**。
           用同 100k、3-seed 均值重算为 96.0% / 92.9% / 84.0%，
           但该口径同样待 evaluator v2 重评确认后才可引用。
源          stand / walk / run 三个 locomotion 源，均由 FastTD3 在同 benchmark 内训练
算法        FastTD3（分布式 critic + CDQ），PTF 的 bootstrap 通道
seeds      每个裁决 3 个 learner seed（M24：单批 3/3 不足以定论）
评估        128-episode 冻结面板，source-free、deterministic
真值        三个判决场的源标签真值均**已知**→ 判据检验而非前瞻发现（§8.5）
```

## 5. 明令不得写进论文的话

0. **不得称端到端系统"提升样本效率"**——口径 2 上 slide 输 0/3、crawl 输 2/3
   （II-9）。只能说"最终性能更高且更稳"，样本效率优势仅在 hurdle 成立；
1. 不得称提出了"新的迁移性指标"——十二类信号经验性失败，racing 是**测量**；
   **也不得称已证明其不可得**——无形式证明，见 I-1；
2. 不得把剂量衰减 / hard exit 称为贡献（PTF 原文已有，且本项目自陈为工程基线）；
3. 不得把 hurdle 的加速表述为跨任务现象（N-1）；
4. 不得省略 racing 的交互成本，也不得只报对己有利的成本口径；
5. 不得声称超越 scratch 而不给出配对不确定性（如 slide hard-exit 的 `lcb90 = −63.6`）；
6. 不得用 episode 面板 SE 支撑跨 learner 的结论（M16）；
7. 不得把"在已知真值场地上检验判据"写成"发现了新规律"（§8.5）；
8. **不得使用一般意义的 "impossibility" 措辞**（标题、摘要、正文皆然）——
   除非给出形式证明。正确表述是 "empirical systematic failure of tested proxy families"；
9. **不得称 `K*=10000` 是 minimum sufficient measurement**——
   只能称已测试预算 {2k,5k,10k} 中最小的稳健 horizon；
10. **不得引用任何饱和度百分比**（98.5% / 95.1% / 85.1% 及其 3-seed 重算版），
    直到 evaluator v2 重评完成（§4）。

## 6. 待补的空缺（诚实清单）

| 空缺 | 影响 | 现状 |
|---|---|---|
| 端到端系统整体效果 | 决定支柱 II 是"零件集合"还是"系统" | `endtoend_v1` 进行中 |
| racing 在真值**未知**的新 target 上前瞻验证 | 决定能否声称跨任务推广 | **未做**；当前全部判决场真值已知 |
| 独立重复（新 learner 批） | M24 要求 | 仅 racing K 有两批；其余单批 |
| 更大源库下的 racing 成本 | `N×K` 随 N 线性增长 | 仅测过 N=3 |
