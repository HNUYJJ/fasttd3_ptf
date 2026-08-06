# fasttd3_ptf 框架复盘:从迁移强化学习理论审视整体设计的合理性与正确性

> 目的:退一步,不再做战术补丁(v1→v2-c),而是把整个框架(FastTD3 + PTF + entity encoder
> + HumanoidBench/H1)从**迁移强化学习(transfer RL)理论**的角度逐层理清,判断每一层设计的
> 合理性/正确性,定位"为什么一直失败"的**根本原因层次**,并据此判断 thesis B 与 package 路线。
> 本文档基于实际代码(已通读)+ 实测数据(A=−386 / A2=−271 / v1≈−425 / scratch=+472 / C=+510)。

---

## 0. 一句话结论

**框架在"obs 维度统一"上工程优雅,但在概念上把"表征统一"误当成了"迁移可行"。** 三组实证
(A2、anchored v1、C)一致指向:transfer 失败的真正 bottleneck 不在编码器的 pooling/锚定机制,
而在两个更上游的层次 ——(1)**冻结的单任务表征无法为目标任务适配**(A2 证),(2)**reach→push
的技能重叠本就低**(C 仅比 scratch 高 +38 证)。这两者都**不是改 pooling(v1/v2-c)能解决的**,
这解释了 v1 为何失败,也预测 v2-c 大概率同样失败。

---

## 1. 整体数据流(框架地图)

```
HumanoidBench h1hand env
  ├─ 共享:61 actuators + proprioception qpos[0:76]+qvel[0:75]=151(始终是 obs 前 151 维)
  └─ 任务特定:追加的 object/goal 状态(reach +6 → 157;push +6 → 163)
        │ raw_obs
        ▼
  ┌─────────────────────────── obs 处理(两条路)──────────────────────────┐
  │ (a) 标准 FastTD3:EmpiricalNormalization(z-score)→ MLP 直接吃 norm_obs   │  ← scratch(B)/slice(C)
  │ (b) entity encoder(step-2):                                              │  ← learned-z(A/v1/v2-c)
  │     tokenizer(raw_obs) → 类型化 tokens[robot,hand,goal,(object)]          │
  │     → hypernet 按类型生成投影 + type-emb → set-Transformer(self-attn)     │
  │     → pool(mean / anchor / anchor_xattn)→ z ∈ R^128                       │
  └──────────────────────────────────────────────────────────────────────────┘
        │ z(或 normalized obs)
        ▼
  FastTD3 网络:Actor(z)→a; 双 distributional Critic(z,a)→C51 分布; OptionModule(z)→(Q_o, β)
        │
        ▼
  ┌──────────────────────────── PTF 迁移 ─────────────────────────────────────┐
  │ OptionSelector(call-and-return):每步按 β 决定是否换 option,ε-greedy 选源/null │
  │ source_bank.act_selected(raw_obs, option):选中的源给 source_action          │
  │   - z-native teacher:自己过 frozen E 得 z,再过 reach actor(无 slice adapter)│
  │   - slice teacher:raw_obs 经手写 slice adapter → 源原生 obs 布局 → 源 actor    │
  │ actor_loss = −Q(s,π) + λ(t)·(1−β)·masked_action_distillation(π, source, mask)  │
  │ option 学习:option-critic U=(1−β)Q+β·maxQ,compatibility(高斯动作相似)加权    │
  │ β 学习:termination_loss,advantage 对称 clamp(修 β saturation)               │
  └────────────────────────────────────────────────────────────────────────────┘
```

关键工程不变量:entity encoder 路用 **raw obs**(z-score 会破坏四元数单位范数/重标 proprio);
frozen E 时排除出 critic optimizer+grad-clip、obs 在 no_grad 下编码;z-native teacher 要求 E 冻结
(否则 E 漂移使 teacher 失效)。

---

## 2. 逐层理论分析(合理性 / 正确性)

### 2.1 FastTD3 backbone —— ✅ 合理
off-policy 分布式 TD3(C51 双 critic + CDQ + 大 batch + 并行环境)。迁移 RL 角度:off-policy +
大 batch 天然适合从 teacher 蒸馏/重放学习。无理论问题,是 strong base。坐实(文献已确证)。

### 2.2 PTF 迁移机制 —— ⚠️ 范式合理,但核心假设跨任务易失效
- 理论来源:PTF(Yang 2020)把源策略当 option,提供"何时迁移"(β termination)+ "迁移哪个源"
  (option Q + compatibility)的调度,actor 端用 **masked action distillation** 落地迁移。
- **隐患 A(最根本):masked action distillation 是行为克隆式迁移,隐含假设"源策略在目标 state
  上输出的动作对目标有益"。** 跨任务时这个假设极脆:reach actor 在 push 的状态上仍然输出"把手
  伸向 goal"的动作,而 push 需要的是"全身把箱子推向 goal"。源动作在目标 state 上**可用但未必有
  益,甚至有害**。λ(t)·(1−β) 只是 heuristic 门控,**不保证"只在有益时蒸馏"**。
- **隐患 B:compatibility 用的是"动作相似度"(高斯),不是"动作有益度"。** 它能筛掉与当前 actor
  动作差异大的源,但筛不出"对 task return 有益"的源——相似 ≠ 有益。
- 实测印证:A 全程为负(不是 ≤ scratch,而是被**主动带偏**到负)——与"蒸馏把 actor 拖向目标上
  无益/有害的源动作"一致(channel 2)。

### 2.3 entity encoder obs 统一(step-2)—— ⚠️ 工程优雅,但概念上混淆了两件事
- 它正确解决的问题:**跨任务 obs 维度/实体集合不一致**(tokens 数变、token 宽不变;encoder.* 权重
  schema-agnostic 可跨任务加载)。这是真问题,做得 elegant。
- **隐患 C(核心概念错位):"统一 obs 表征" ≠ "技能可迁移"。** 即使把 reach/push 的 obs 完美映到同一
  z 空间,reach actor 读 push 的 z 得到的仍是 reach 的动作——z 空间统一了,**策略的技能没有因此变得
  可迁移**。obs 统一是迁移的**必要不充分**条件;充分条件是源-目标**技能重叠**。整个 step-2 把大量精力
  投在"必要条件"上,而失败发生在"充分条件"。
- **隐患 D:frozen 单任务 E 当跨任务前端,理论上本就可疑。** E 在 reach 上为 reach RL 优化,object
  投影从未训练;冻结后既不能学 box、又主导 mean-pool。A2(frozen E + 无 teacher = −271 ≪ scratch)
  直接证明:**冻结的单任务表征本身就是坏的目标前端**,与 teacher 无关。

### 2.4 HumanoidBench + H1 特性 —— 迁移的有利与不利面
- 有利:所有任务**共享 61-act + 151 proprio**——body 不变,这是 morphology 层面理想的迁移底座
  (proprio 锚点天然存在)。
- 不利:任务差异在 **object/goal + 奖励结构 + 所需技能**。HumanoidBench 原文已指出 manipulation
  需"先会站走再操作",技能是**分层、长程**的。
- 结论:H1 给了**共享 body 这个迁移红利**,但红利只在"技能相近"时兑现。

### 2.5 reach→push 迁移对 —— ❌ 技能重叠低,是 MVP 的根本失误
- 共享:body、proprio。不共享:object(box)、技能(伸手到点 vs 全身推箱)、奖励。
- 迁移 RL 基本规律:**迁移收益 ∝ 源-目标的技能/任务相似度**。reach(upper-body 静态伸手)与 push
  (whole-body 动态推箱)重叠低 → 迁移收益本就低。
- **实测铁证:C(slice-adapter,一个忠实的 raw-obs reach teacher)只比 scratch 高 +38(单 seed,
  噪声内)。** 即:**连"正确实现的 reach→push 迁移"都几乎没有正迁移。** 这把锅从"编码器机制"摘下来,
  扣到"**源-目标配对本身没有多少可迁移信号**"上——这是比任何 pooling 都上游的问题。

---

## 3. 核心理论诊断:失败发生在三个层次,从下到上

| 层次 | 问题 | 实证 | 能否用 pooling(v1/v2-c)修? |
|---|---|---|---|
| **L3 表征机制**(最下游) | mean-pool 稀释 / readout 锚定 | dilution 诊断 | ⚠️ v1 已证否(≈mean A);v2-c 预测同 |
| **L2 frozen 表征** | 冻结单任务 E 无法为目标适配、object 投影未训练 | **A2 = −271 ≪ scratch** | ❌ pooling 改不动冻结的未训练投影 |
| **L1 技能可迁移性**(最上游) | reach→push 技能重叠低,源动作在目标 state 上无益 | **C 仅 +38;A 被带到负** | ❌ 与编码器完全无关 |

**关键洞察:我们一直在 L3(pooling)打补丁,但真正的 bottleneck 在 L2 和 L1。** 这就是 v1 失败、
且 v2-c 大概率也失败的根本原因——**改 L3 绕不开 L2/L1**。
- "dilution-by-domination(box 占 z 方差 75%)"这个 L3 指标还是**confounded** 的:push 本就是 box
  任务,任何训练好的 push 编码器 box 也会占高比例;真正有害的是 L2(box 投影冻结在未训练态),不是占比。

---

## 4. 对 thesis B 与 package 的含义

- **thesis B(proprio-anchored 跨任务表征)解决的是 L3/L1-obs 维度问题**,但 HumanoidBench 上的
  bottleneck 是 **L2(frozen)+ L1(技能重叠)**。即使 thesis B 把表征做到完美,reach→push 也迁不动
  (技能不匹配)。所以 thesis B 在当前 MVP 设置下**修的不是真 bottleneck**——这是它连续失败的深层原因。
- **新颖性 vs 价值要分开看**:thesis B 在文献缝隙上可能"新"(CARE 不碰维度不一致、morphology 线是
  cross-morphology),但"新"不等于"解决了 HumanoidBench transfer 的真问题"。顶会既要新、也要 result;
  没有 positive transfer 的 result,新颖性是空中楼阁(这也是给 Codex 审查问题里我已强调的)。
- **对 package**:headline 是 package(移箱到目标)。按 L1 规律,**package 的好源应是技能重叠高的任务
  ——push(同样 box manipulation)才是 package 的自然源,reach 不是。** 在投任何表征机制前,应先用
  **success rate**(而非 shaped return)确认"存在一个源,其技能真能迁到 package"。

---

## 5. 理论上更合理的框架方向(排序)

1. **先解 L1(可迁移性)再谈表征:** 选技能重叠高的源-目标对(push→package / locomotion→whole-body),
   用 success-rate 量化是否有 positive transfer。**这是最高杠杆,且便宜。**
2. **解 L2(frozen):** 若要走 learned-z,**E 必须可训练**(reach 初始化 + target 微调),放弃"frozen +
   z-native teacher"这个把自己锁死的不变量。代价是不能用 z-native teacher,改配 slice/value-based teacher。
3. **迁移机制超越行为克隆(攻 L2-A 隐患):** masked action distillation 假设太强;可考虑 **value/feature
   层面的迁移**(蒸馏 Q/表征而非动作)、或 **子目标/技能级迁移**(HRL),减弱"源动作必须在目标 state 上有益"
   的假设。
4. **L3(表征机制)作为支撑,不作 headline:** robot-anchored readout / 可学 object context 等是必要工程,
   但它们不是 HumanoidBench transfer 的 bottleneck,不该当论文主创新。

---

## 6. 决策建议(给 human_pi + Codex)

- **暂停"在 frozen 表征上换 pooling"这条线**(v1 已否,v2-c 预测同;让 v2-c pilot 跑完仅作完整性,不为它扩 seed)。
- **下一个 decisive 实验仍是 trainable-E**(隔离 L2:frozen 是不是元凶),但**真正该补的是 L1 验证**:
  一个技能重叠高的源-目标 pilot(如 push→package 或一个 locomotion 源),用 success rate 看有没有 positive
  transfer。**L1 验证比 L2/L3 更上游、信息增益更高。**
- **headline 重定位**:把研究问题从"如何统一跨任务 obs 表征"(L3,bottleneck 之外)抬到"**在共享 body、
  技能部分重叠的 whole-body 任务间,如何获得可证的 positive transfer**"(L1/L2,真 bottleneck)——
  表征统一是其中一个组件,不是全部。

> 一句话给 PI:**我们造了一座很漂亮的桥(obs 统一),但架在了没有水的地方(reach→push 技能重叠低、
> frozen 不可适配)。先确认哪里真有"河"(技能可迁移的源-目标对),再决定桥怎么修。**
