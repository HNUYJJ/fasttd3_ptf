# Bottleneck-Aligned Coverage v1：指标定义、回溯检验与前瞻预测冻结

> 2026-07-28。本文件在任何新训练启动**之前**提交。
> 回溯部分是事后分析（不算证据），前瞻预测部分是可证伪的科学声明（算证据）。

## 1. 出发点

本项目已封存八个迁移性信号族，全部失败。它们的共同点是**都把 target 的 reward
聚合成一个标量**。但 HumanoidBench 的 reward 不是标量，是带组合算子的分量结构，
而标量 return 把三样东西混成一个数：

```
return  =  每步分量质量  ×  分量组合算子  ×  生存时长
```

四个独立机制会让 return 系统性反向（均为本仓库实测数据，非构造）：

| 机制 | 实例 | 现象 |
|---|---|---|
| `min` 瓶颈算子 | crawl `0.25·min(crawling, crawling_head)` | stand 把 crawling 抬到 0.845 却把 crawling_head 压到 0.343，min 取小者反比 zero(0.526) 差，return 却最高 |
| 乘性归零 | sit_hard `sc × sit_reward × dont_move` | stand 把 sit_reward 打到 0.005（zero 为 0.113），乘性下总 reward 必崩 |
| 生存时长 | slide `sr × sc × move` | stand 每步 reward 0.178 最低，却因不摔跑满 episode 而 return 88.5 最高（walk 每步 0.589、return 仅 45.7） |
| 权重错配 | door `0.45·door_openness + 0.35·passage` | run 推进 passage(0→0.205)，占最大权重的 door_openness 三源零覆盖 |

## 2. 指标

实现见 `scripts/analysis/bottleneck_aligned_coverage_v1.py`，
reward 结构规格见 `configs/reward_structure/humanoidbench_v1.py`
（17 个 target 逐个从 `humanoid_bench/envs/*.py` 的 `get_reward()` 核准，
含组合算子、min 组、门控因子、分量方向，以及 reward 用到但 info dict 未导出的盲区）。

以 zero-action 基线作为 student 起点代理：

```
m_c        = ∂R/∂x_c 在 x[zero] 处
             加性项  m_c = w_c（若被门控再乘 gate[zero]）
             乘性因子 m_c = Π_{c'≠c} x_{c'}[zero]

B          = 按 m_c·(1 − x_c[zero]) 降序累计到 ≥50% 的分量集合（瓶颈集）

Coverage_i = Σ_{c∈B}   m_c · max(0, x_c[i] − x_c[zero])
Damage_i   = Σ_{c∈all} m_c · min(0, x_c[i] − x_c[zero])
NET_i      = Coverage_i + Damage_i
```

正负不对称是有语义的：正向只算瓶颈分量（非瓶颈已近饱和，推高它只让 return 好看）；
负向算全部分量（乘性/门控下任一因子被压垮是结构性破坏，不可由其他分量补偿）。

无界任务（package / push / truck / cabinet）直接判 `UNMEASURABLE`——
return 被稀疏事件主导。这与已观测的 `CABINET_UNCERTAIN` 和 package「return 不可分辨」一致，
**是该指标的一个独立的事后一致点**。

## 3. 回溯检验（事后，不作为证据）

三个已有配对学习效用真值的 target：

| target | NET (stand / walk / run) | NET 排序 | 实测 U 排序 | return 排序 | 判定 |
|---|---|---|---|---|---|
| hurdle | +0.103 / +0.131 / **+0.157** | run>walk>stand | run(+380) > walk(+105) | run>walk>stand | **命中** |
| crawl | **−0.044** / −0.004 / −0.011 | walk>run>stand | run(−208) ≈ walk(−217) >> stand(−448) | stand>walk>run | **主判别命中**（stand 最差），walk/run 细序不命中 |
| door | −0.018 / −0.026 / −0.001 | run>stand>walk | walk(−22) > run(−31) > stand(−33) | run>stand>walk | **不命中** |

诚实计分：**排序 1/3 完全命中；主判别 2/3**（hurdle 全序、crawl 的 stand 最差）。
return 在 crawl 上完全反向、在 door 上同样不命中，故 NET 不劣于 return，但也没有压倒性优势。

### 两处必须记录的事后调整

1. **v1 初稿只用 Coverage 作主量**，它在 crawl 上给 walk/run 打 0.104/0.113（正），
   与实测（−217/−208，全负）矛盾。加入 Damage 后 NET 全负且 stand 最负，才与实测一致。
   **此调整发生在看到 crawl 结果之后，故 crawl 不构成对 NET 的独立验证。**
2. door 的 NET 极差 0.0255 勉强超过 `SEPARATION_MIN=0.02` 而给出了排序，
   但三源 NET 全部接近 0（−0.0008 ~ −0.026）。在"全部接近零"时给排序本身是过度解读。

## 4. 由回溯暴露出的缺口：交互预算机会成本

door 的失败暴露了模型缺一项。三源 NET 全部 ≈ 0，实测 U 却全部 ≈ −22 ~ −33，
且**三源之间几乎无差别**。这不是"源有害"，是"源无用，而预算被占"：
所有源臂都让出 50% 的环境交互给 source。

因此正确的模型形式应是：

```
U_i  ≈  α · NET_i  −  C(dose)
```

`C(dose)` 与源身份无关，只与剂量有关。这一项定性解释了全部三个 target：

- door：NET ≈ 0 → U ≈ −C ≈ −30，且三源接近 ✓（也解释了为何 door 三源差异如此小）
- crawl：stand 的 NET 显著负 → 在 −C 之上再叠加破坏 → 最差 ✓
- hurdle：NET ≈ +0.10~0.16 → α·NET 超过 C → 转正 ✓

**`C(dose)` 是可独立测量的**：一个不注入任何源、但同样按 dose 削减 student 有效交互的
对照臂即可测出，且该臂完全不涉及迁移。这是本框架产出的第一个新实验设计。

α 的标定是 target-specific 的（各 target 的 return 尺度差一个量级），
当前三个点不足以标定，**故本轮不作任何定量预测，只冻结符号与排序**。

## 5. 冻结的前瞻预测（可证伪，本节是本文件的证据部分）

以下 target 尚无配对学习效用真值。预测在任何相关训练启动前冻结。
协议须与 door/hurdle/crawl 系列一致：10k anchor、0.5/0.5 等剂量、h=25、
`bootstrap_only`、跑到 20k、128-episode 冻结 source-free 面板、3 learner seeds。

| target | 结构 | NET (stand/walk/run) | **NET 预测排序** | return 预测排序 | 判决力 |
|---|---|---|---|---|---|
| **slide** | 乘性 | 0.0129 / **0.5153** / 0.5086 | **walk ≈ run >> stand**（全正） | stand>walk>run | **★最强**：两指标极端反向，NET 差 **40 倍** |
| stair | 乘性 | 0.0046 / 0.4617 / 0.4573 | walk ≈ run >> stand | stand>walk>run | ★强反向，NET 差 100 倍 |
| pole | 门控 | 0.0074 / 0.2678 / 0.1853 | walk > run >> stand（全正） | stand>walk>run | ★反向 |
| sit_hard | 乘性 | −0.0933 / −0.1116 / −0.1267 | **ALL_NEGATIVE**（三源均有害） | run>walk>stand | ★反向，且预测符号 |
| maze | 门控 | 0.1105 / 0.0760 / 0.0652 | stand > walk > run（全正） | walk>run>stand | 不一致 |
| powerlift | 加性 | 0.000 / 0.000 / 0.000 | 三源均无实质效应 | stand>walk>run | 符号预测 |
| room | 加性 | −0.0007 / −0.0020 / +0.0020 | 三源均无实质效应 | stand>run>walk | 符号预测 |

### 首选判决场：slide

- NET：stand 0.0129，walk 0.5153，run 0.5086 —— 相差 **40 倍**
- return：stand **88.5**（最高），walk 45.7，run 27.8 —— **顺序完全相反**
- 机制透明：slide 是纯乘性 `stand_reward × small_control × move`，
  stand 的 move 仅 0.188（几乎不动），靠不摔活满 episode 刷出高 return；
  walk/run 的 move 达 0.82/0.84，每步 reward 是 stand 的 3 倍
- 成本低：slide 已有 scratch 基线与配置（wfix 系列用过）

### 预注册裁决

```
BAC_SUPPORTED      slide 上 U(walk) 与 U(run) 均显著高于 U(stand)
                   （3 learner seed 配对 90% t 区间下界 > 0）
                   → NET 在 return 反向的场合胜出，指标进入论文主线

BAC_PARTIAL        stand 为三者最差但区间跨零，或 walk/run 仅一者显著高于 stand
                   → 报告，不外推；须在 stair 上重复一次才决定去留

BAC_REFUTED        U(stand) 不低于 walk/run 中的任何一个
                   → 指标失败。不调 BOTTLENECK_MASS、不调 SIGN_EPS、
                     不调 SEPARATION_MIN、不换瓶颈定义来抢救。
                     按项目纪律，gate 失败后不得调参重跑。
```

裁决只用 slide。stair/pole/sit_hard 的预测同时冻结在上表，
但本轮不跑——它们是 slide 通过后的第二轮重复验证场，
预先写下是为了防止事后挑选有利 target。

## 6. 本框架已产生的、与主线无关的一个推论

Door 的 fixed-horizon prefix handoff 实验（`docs/run_card_door_prefix_handoff_v1.md`）
可直接停止。door 的瓶颈分量 `door_openness_reward` 三源零覆盖，
NET ≈ 0，`Δ_placement` 的两个臂都只是在推进 `passage_reward`——
改变 source 的时间放置方式不改变它覆盖哪个分量。
框架预测 `Δ_placement ≈ 0`，这比"协议无法实例化（share 0.3896 未达门槛）"
是强得多的停止理由。
