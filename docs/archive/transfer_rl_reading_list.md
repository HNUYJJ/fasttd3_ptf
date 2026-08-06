# 迁移强化学习阅读清单(2024–2026)+ 对 HumanoidBench whole-body 的启发

> 用途:供 PI 自己读论文、长 idea。不是给方案。按"主线 + 必读 + 核心 insight + 对我们的启发"组织。
> 链接里 2026 年的 arXiv ID(26xx.xxxxx)是搜索返回的,**部分可能 ID 有误,以标题/作者去 arXiv/Google Scholar 搜更稳**;经典论文 ID 已确认。

---

## 0. 一个被我们忽略的框架级观察(最重要)

**HumanoidBench 所有任务 = 同一个 H1 body = 同一个 transition dynamics,只是 reward/task 不同。**
这正是迁移 RL 两大理论主线的标准假设(*same dynamics, varying reward*):
- **Successor Features + GPI**(value-based transfer,有零样本性能保证);
- **Zero-shot RL / Behavior Foundation Models**(reward-free 预训练 + 对任意新 reward 零样本)。

我们之前只做 **PTF 的 masked action distillation(行为克隆式迁移)**,它没有理论保证、且假设"源动作在目标上有益"(我们已证伪)。**而上面两条有理论根基、且与 HB 设定天然契合的主线,我们一条都没碰。在高维 whole-body humanoid 上做 SF/GPI 或 BFM 几乎是空白区。** 这是读论文时最该带着的问题。

---

## 主线 1 —— Successor Features (SF) + Generalised Policy Improvement (GPI):value-based transfer 的基石

- **Barreto et al., "Successor Features for Transfer in RL", NeurIPS 2017**(arXiv:1606.05312)——**必读经典**。
- **Barreto et al., "Transfer in Deep RL using SF & GPI", ICML 2018**(arXiv:1901.10964)——深度版 + 放松线性 reward 假设。
- **核心 insight:** 把 Q 分解为 `Q^π(s,a)=ψ^π(s,a)·w`,其中 **ψ(successor features)只编码 dynamics、与 reward 无关**,reward 由 `r=φ·w` 的权重 `w` 决定。换任务=换 `w`;用 **GPI** 在一组已学策略上取 max,**对新任务在学习前就有性能下界保证**。天然支持"skill library 零样本组合"。
- **对我们的启发:** HB 同 body 不同 task = SF/GPI 的理想场景。reach/walk/push 各学一套 ψ,新任务(package)用 GPI 组合——这是比 action distillation **有理论保证**的迁移。值得想:高维 whole-body 上 ψ 怎么学、`w` 怎么从 manipulation reward 估。

## 主线 2 —— Zero-shot RL / Behavior Foundation Models (BFM):最前沿、最契合

- **Touati & Ollivier, Forward-Backward (FB) representations**(zero-shot RL 的核心方法,务必搜来读)——reward-free 预训练出 FB 表征,test 时给任意 reward 即可零样本算出近最优策略。
- **"A Unified Framework for Zero-Shot RL"**(arXiv:2510.20542, 2025)——综述+统一视角,入门首选。
- **TD-JEPA**(latent-predictive representations for zero-shot RL, 2025);**"Regularized Latent Dynamics Prediction is a Strong Baseline for BFMs"**(arXiv:2603.15857);**"Zero-Shot Adaptation of BFMs to Unseen Dynamics"**(arXiv:2505.13150)。
- **核心 insight:** 在**无奖励**的交互数据上预训练一个"行为基础模型",它对一大类 reward 都能零样本给近最优策略——RL 版的 foundation model。
- **对我们的启发:** 在 H1 上用海量 reward-free 探索(**FastTD3 高吞吐恰好能产**)预训练一个 whole-body BFM,然后零样本/少样本解 manipulation。**高维 whole-body BFM 几乎没人做**——这可能是真问题+新颖+契合资产三者兼备的方向。务必核实:有没有人已在 HB/高维 humanoid 上做过 FB/BFM。

## 主线 3 —— Unsupervised Skill Discovery(可组合、可迁移的技能)

- **DUSDi, "Disentangled Unsupervised Skill Discovery", NeurIPS 2024**(arXiv:2410.11251)——**核心 insight:把技能 disentangle 成多个 component,每个只影响一个 state factor,可并发组合、可链式;用 HRL 解 downstream。** disentanglement→composability。
- **FoG, "Guiding Skill Discovery with Foundation Models", ICLR 2025**(把人类意图/FM 引入无监督技能发现)。
- **对我们的启发:** body-hand 协调可能正需要"disentangled skills"(身体 component vs 手 component 各自可控、可组合)——比我们 naive 的"body-hand 解耦"有原则得多。

## 主线 4 —— Cross-domain / Cross-embodiment 表征对齐

- **"Cross-Domain Policy Transfer by Representation Alignment via Multi-Domain BC"**(arXiv:2407.16912)。
- **"Cross-Embodiment Skill Transfer using Latent Space Alignment"**(arXiv:2406.01968)。
- **awesome-cross-domain-policy-transfer**(IJCAI'24 index,GitHub t6-thu)——一个系统化的算法索引,适合快速建立全景。
- **核心 insight:** 显式对齐源/目标的 latent 空间(而非我们 naive 的"共享 encoder 自动对齐")——对齐是要**专门学的目标**,不是免费的。这正是我们 step-2 失败的理论解释之一。

## 主线 5 —— 最新 skill/task-shift transfer(2026,前沿但需核实链接)

- **"Predictive Representations for Skill Transfer in RL"**(arXiv:2604.07016)——predictive representation 做技能迁移。
- **"Optimistic Transfer under Task Shift via Bellman Alignment"**(arXiv:2601.21924)——task shift 下用 Bellman alignment 的乐观迁移。
- **"Mapping Representations in RL via Semantic Alignment for Zero-Shot Stitching"**(arXiv:2503.01881)——zero-shot 拼接不同模块的表征。

## Survey(建立全景)

- **"A Survey on Transfer Reinforcement Learning"**(IEEE 2025)。
- 主线 2 的统一框架综述(2510.20542)也兼具 survey 作用。

---

## 读这些时建议带着的几个问题(帮你长 idea)

1. **SF/GPI 和 BFM 都假设 same-dynamics-varying-reward —— HB 完全满足。为什么还没人在高维 whole-body humanoid manipulation 上认真做?是理论/工程障碍,还是单纯没人试?**(若是后者 = 机会;若是前者 = 那个障碍本身可能就是论文。)
2. 我们已证伪 action-distillation 迁移。**value-based(SF)或 representation-based(BFM/FB)迁移会不会绕开"源动作必须有益"这个致命假设?**
3. body-hand 协调:**disentangled skills(DUSDi)能不能给"身体 + 灵巧手"一个可组合的原则性分解?**
4. 我们的真资产是**高吞吐 reward-free 探索能力(FastTD3)**——哪条主线最吃这个红利?(直觉:BFM/zero-shot RL 最吃,因为它要海量 reward-free 数据。)
