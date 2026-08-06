# Zero-shot 可迁移性探针:源策略 → h1hand-package-v0(2026-06-10)

## 目的(L1 验证,预注册决策规则)

在启动 push→package PTF pilot 之前,先测量"push 源的技能能否 zero-shot 搬到
package",决策规则**事先**定为:

- push 显著优于 reach/locomotion 源 → L1 成立,启动 pilot;
- push 与所有源一样贴着 zero/random 基线 → 高重叠配对也无信号,
  冻结源 + 行为蒸馏的 PTF 范式归档,转向技能组合(HRL)方向。

脚本:`scripts/probe_zero_shot_package.py`(slice obs 适配的语义映射见脚本 docstring)。
每源 32 episodes(32 并行 env 各 1 episode),deterministic 动作。

## 有效性对照:push 源回 push 环境(identity adapter)

| source | return | success | goal_min | hand_min |
|---|---|---|---|---|
| push_s1 | +489 | 59.4% | 0.08 | 0.27 |
| push_s2 | +430 | 62.5% | 0.06 | 0.26 |
| random | -163 | 6.3% | 0.28 | 0.40 |
| zero | -288 | 0.0% | 0.33 | 0.44 |

加载/normalizer/adapter 管线健康,push 源是真专家。package 上的结果可信。

## 主结果:package 上全军覆没,失败模式 = approach 阶段

| source | return | success | goal_fin | goal_min | hand_min |
|---|---|---|---|---|---|
| zero | -6099 | 3.1%* | 2.02 | 1.95 | **1.28** |
| push_s1 | -6187 | 0% | 2.05 | 2.01 | **1.32** |
| stand | -6457 | 0% | 2.46 | 2.15 | 1.66 |
| reach | -6946 | 0% | 2.35 | 1.89 | **0.72** |
| random | -7491 | 0% | 2.62 | 2.01 | 0.97 |
| push_s2 | -7910 | 3.1%* | 2.63 | 2.07 | 1.34 |
| walk | -14013 | 0% | 4.13 | 2.03 | 1.32 |
| run | -17808 | 0% | 2.75 | 1.94 | 1.56 |

\* zero/push_s2 的 3.1% = 1/32,是"box 初始位置恰好落在 destination 0.1m 内"
的彩票(episode 即刻成功终止),不是技能。

三个细分事实:

1. **box 全程纹丝不动**:所有源 goal_dist_min ≈ goal_dist_final ≈ 2.0
   (两均匀随机点的期望距离),没有任何策略推动过 box;
2. **机器人从未接近过 box**:push 源 hand_dist_min ≈ 1.32 ≈ zero 动作的被动值
   (1.28)——push 源在 package 里的行为等价于原地站桩;
3. **reach 是唯一表现出方向性行为的源**(hand_min 0.72,因为我们把它的目标输入
   填成 box 位置,它确实朝 box 伸手),但只伸手不挪步,远够不到 2m 外的 box。

## 机理解释

push 任务里机器人站在桌前,box 永远在手边(hand_min 0.27)——push 源的技能
前提是"box 已在手边",它**根本不含"走向远处物体"的子技能**。package 的 box
在平均 2m 外的地面(z=0.35,push 见过的是桌面 z≈1.0),destination 也在地面。
机器人初始姿态 ≈ push 的静止站姿 → push 源输出"站桩 + 手部小动作"→ 与 zero
无异。

**package 是组合任务**:approach(走向 box)→ manipulate(搬/推)→ 可能再
approach(随 box 走向 destination)。手头源池只覆盖 manipulate 段(push/reach)
或无目标导向的 locomotion 段(walk/run 直行,不会朝物体走),**没有源覆盖
goal-conditioned approach**,而 zero-shot 下永远到不了 manipulate 段。

## 决策含义

1. **push→package 的 PTF pilot 按预注册规则不启动**:行为克隆一个"不会走向
   box"的源,λ 调到天上也教不会机器人走过去。省下的训练算力即本探针的价值。
2. **ED-SF push transfer 失败获得新解释**:body-SF 从 reach/stand/walk 迁移到
   push 时同样面临"源技能不含目标任务关键子技能"的问题(L1),
   与 reward 回归不拟合(L2)叠加。
3. **对 skill-library HRL(创新方向)是强 motivating evidence**:单一源全部
   zero-shot 失败 + 失败模式分解(approach 缺位)= 论文 motivation 的 Table 1。
   技能组合若要成立,源池必须补一个 goal-conditioned approach 技能
   (HB 没有现成任务,需要定制训练)。

## 近身探针(同日补充):manipulate 段技能可迁移

`--env package_near`:reset 时把 box 摆到机器人正前方手边(push 训练分布,
x∈[0.65,0.85],y∈[-0.15,0.15]),destination 摆到 box 旁 0.25~0.58m——模拟
"approach 已完成"的状态,隔离测 manipulate 段。每源 32 eps:

| source | success | goal_min | goal_fin | 解读 |
|---|---|---|---|---|
| reach | **21.9%** | **0.19** | 0.64 | 手持续贴 box 温和推挤,box 留在附近 |
| push_s1 | **15.6%** | **0.26** | 0.58 | 方向性推动(初始距离 0.42→0.26),box 不丢 |
| random | 9.4% | 0.27 | **2.37** | 把 box 碰飞(布朗运动,非技能) |
| stand | 6.3% | 0.39 | 0.64 | 站桩吃运气 |
| push_s2 | 6.3% | 0.31 | **2.25** | 动作暴力,box 打飞 |
| zero | 0% | 0.41 | 0.41 | box 纹丝不动 |

要点:近距下"碰运气"基率 ≈6-9%(random/stand 水平);reach 21.9% 显著超出
(p≈0.02),push_s1 边缘显著;行为学指标(定向推近 + box 保持在附近 vs 打飞)
区分了技能与噪声。**意外发现**:reach 比 push 迁移更好(push 技能绑定桌面
z≈1.0 几何,reach 的伸手技能更几何通用);**push 源 seed 间方差巨大**
(s1 可用/s2 打飞)→ 探针可兼作 PTF 源池的 seed 级筛选工具。

## 证据链闭合 → coverage-aware PTF

- 远端(标准 package):全部源 0%,机器人到不了 box——源池缺 approach,PTF 无信号可蒸;
- 近端(近身 package):操作类源显著 >0——manipulate 技能可跨任务迁移,PTF 有信号可蒸。

PTF 蒸馏不要求源是专家,只要求比随机探索好的行为先验。结论:**源池补一个
goal-conditioned approach 源后,PTF(Q_o 调度:远→approach,近→push/reach)
第一次获得公平出场机会**。

## 下一步(已定,方案 A)

在 package 环境上用 auxiliary approach reward(走近 box + 站立项,**不含任务
reward**)训练 approach 源(FastTD3,~100k iters);论文表述为
auxiliary-reward skill pretraining in the target domain。之后:
PTF pilot v2,源池 = {approach, push_s1, reach, null},主指标 success rate,
对照 package scratch 同预算。

## 产物

- 明细:`logs/probe/zero_shot_push_full.jsonl`、`logs/probe/zero_shot_package_full_v2.jsonl`
- 日志:`logs/probe/probe_push_full.log`、`logs/probe/probe_package_full_v2.log`
- 复现:`python scripts/probe_zero_shot_package.py --env {push,package} --num-envs 32`
