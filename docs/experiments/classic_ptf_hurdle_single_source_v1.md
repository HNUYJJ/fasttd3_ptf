# Classic PTF + FastTD3 单教师可行性实验 v1

## 核心问题

在不使用 reward-bearing bootstrap、MCG、admission、source action execution
或 replay 重加权时，原始 PTF 的 option-value / termination / distillation
结构能否利用冻结的 walk 教师，提高 FastTD3 student 在 HumanoidBench Hurdle
上的样本效率或 95k 性能？

## 唯一主要假设

`walk` 对 Hurdle 的早期 locomotion 学习有帮助，但不包含完整的跨障碍行为；因此
PTF 应在部分状态/阶段选择 walk，并通过 learned termination 或 no-transfer option
降低后期约束。若 PTF 只持续选择 walk、beta 饱和、transfer loss 近零，或显著落后
scratch，则优先判为调度/实现问题，而不是直接宣判 PTF 理论无效。

## 实验臂

- `ptf`：冻结 walk + no-transfer option；student 始终执行；训练 Q_o 与 beta；
  `lambda: 1 -> 0 / 100k`。
- `scratch`：空 source bank；同一 FastTD3 代码、超参、seeds 与评估协议。

首轮结果显示 learned PTF 早期 3/3 seed 加速、但最终高度不稳定，因此追加一个
最小因果消融：

- `fixed`：walk 是唯一 option；保持相同的 Huber、compatibility 和
  `lambda: 1 -> 0 / 100k` 名义蒸馏日程，但关闭 beta 对 transfer loss 的调权。
  它不训练出有行为意义的 source-vs-null 选择，用于测量 walk imitation 本身。

`learned PTF - fixed` 才是当前实验对 option/termination 自适应价值的估计；
`fixed - scratch` 是固定教师动作蒸馏的价值。fixed 与 learned 的实际累计蒸馏强度
不强行事后配平，因为 no-transfer 与 beta 降权正是 learned 调度产生的 treatment。

两臂均为 128 env、100k outer steps、5k eval interval。先做 3k real-environment
smoke；机制信号健康后再执行 seeds 1/2/3 正式实验。

## 必看指标

- 主结果：5k--95k normalized AUC、10k/30k/60k/95k return。
- 调度：rollout/replay 中 walk 与 no-transfer option 占比。
- 终止：walk/no-transfer beta、当前 option age。
- 迁移强度：raw distillation loss、`lambda*(1-beta)`、active fraction。
- 健康性：Q_o loss/Q 均值、compatibility、FastTD3 actor/critic loss。

## 裁决边界

单教师 Hurdle 是正向可行性场地，不构成通用迁移性指标证据。正结果才进入多教师
自动切换；负结果先区分 transfer 未生效、beta/option 退化和已生效但性能负迁移。
