# 预注册：选源本身有价值吗——hurdle 上的 argmin 对照臂

> 2026-07-31。**本文必须在 stand 臂的任何评估结果产出之前提交。**
> 目的：补齐 hurdle 端到端链条的最后一环，回答
> "**自动选源**有价值，还是**随便用个源**就行"。

## 1. 缺的是哪一环

```
① racing 自动选出 run              已有：RACING_K，两批 6 个 learner 全部选中 run（a744adb）
② run 带来早期 3.5–4.4× 加速        已有：hurdle_speedup_v1，vs scratch，3 seeds（20f1e11）
③ run 是否优于「随便选一个源」        ← 本实验
```

没有 ③，①②合起来只能证明"**迁移**有用"（②已证），
**不能**证明"**自动选源**有用"——因为无法排除"任何源都能带来同样加速"。

这正是 `CLAUDE.md` §8.1 要求写出并排除的平凡解释。

## 2. 为什么 argmin 是 `stand`

hurdle 的 `EQD30K` 标签：`run(+379.66) > walk(+104.89) > stand(+51.28)`。
`RACING_K` 在 K=10000 的 6 个 learner 上一致给出同一排序（`run > walk > stand`）。
故 racing 的 `argmin` 为 **stand**，且 stand 的 `U` 仍为正——
这使对照**更严格**：不是"好源 vs 有害源"，而是"**最好的源 vs 最差但仍有益的源**"。

## 3. 关键设计：从 `t=0` 重新训练，而非从 racing 臂续训

`RACE-then-RUN` 被否的核心原因之一是 order-statistic 效应：
从同一批 checkpoint 取 best/worst 再各自续训，`B>C` 可能只反映"运气最好的 learner path"。

**本实验从 `t=0` 重新训练 stand 臂**，与 `hurdle_speedup_v1` 的 source 臂逐项同协议。
因此三条臂（scratch / run / stand）都是独立的完整训练，
不存在"从已知更好的 checkpoint 继续"这一混淆。

## 4. 协议（冻结）

```
target        h1hand-hurdle-v0
本轮新跑      stand 源臂，seeds 1,2,3，t=0 → 100k
已有对照      run 源臂  （hurdle_speedup_v1 的 source 臂，同 seeds）
              scratch  （hurdle_speedup_v1 的 scratch 臂，同 seeds）
bank          configs/source_banks/calibration/h1hand_hurdle_rbo_stand.yaml
其余参数      与 run 臂**逐项相同**：PTF_MCG=1, MCG_ABLATION=bootstrap_only,
              WARMUP_MODE=admission_bootstrap, ADMISSION_MODE=all,
              EXPECTED_SOURCE_MASS=0.5, WARMUP_MIN_STEPS=25,
              NUM_ENVS=128 BATCH=32768 BUFFER=51200 NUM_UPDATES=2 COMPILE=0 AMP=1
评估点        10k, 20k, 30k, 50k, 75k, 100k；source-free, deterministic, 128 ep
```

**剂量验收**：stand 臂 behavior share ∈ `[0.48,0.52]`（run 臂实测 0.4983–0.4995）。
**与 run 臂的 share 差须 ≤ 2 个百分点**（`M26`：剂量不得与源身份共变）。

## 5. 判据（冻结）

沿用 `hurdle_speedup_v1` 的达阈口径（`analyze_hurdle_speedup_v1.py`，`1fcf136`）：

```
steps_X(θ) = 臂 X 的 source-free 回报首次 ≥ θ 的步数（相邻评估点线性插值，右删失记 100k）
θ ∈ {200, 300}   ← 沿用已冻结阈值；θ=400 因 run 臂已含右删失而不用
```

**主判据（选源的价值）**：

```
per-seed 配对：steps_stand(θ) > steps_run(θ)
SELECTION_VALUABLE   ⟺  θ=200 与 θ=300 上均 3/3 seed 满足
SELECTION_PARTIAL    ⟺  仅其中一个 θ 满足 3/3
SELECTION_NULL       ⟺  两个 θ 均 ≤2/3 —— **有意义的负结果**：
                        用源即可，选哪个源无关紧要，racing 的实用价值被否定
```

**次判据（记录，不参与主裁决）**：`steps_stand(θ)` vs `steps_scratch(θ)`
——检验 stand 是否仍优于 scratch（预期是，因其 `U=+51.28` 为正）。

**偶然通过率**：零效应下每个 seed 二选一为 `1/2`，3/3 为 `1/8`；
两个 θ 均要求 3/3，若二者独立则 `1/64`，实际因强相关介于 `1/8`~`1/64`。
**保守报 `1/8 = 12.5%`。**

## 6. `CLAUDE.md` §8 设计层自查

- **8.1 辨别力**：平凡解释="任何源都带来同样加速"。**本实验就是为排除它而设**；
  `SELECTION_NULL` 真实可达。第二个平凡解释="stand 更差只因它剂量低"——
  由 §4 的 share 差 ≤2pp 检查排除。
- **8.2 混淆**：三臂剂量同为 0.5 且逐 checkpoint 验收；stand 臂与 run 臂逐项同参数。
- **8.3 独立重复**：**单批 3 seeds，这是本设计的主要弱点**。
  按 `M24`，若结果为正**只能报为 pilot**，须注明"待独立重复"。
  缓解：判据是 per-seed **配对**比较（同 seed 的 run vs stand），
  且 run/scratch 侧复用已发表数据，故比较本身是配对的。
- **8.4 前提蕴含**：racing 选中 run 是已知事实，但 **stand 臂的达阈步数完全未知**；
  `SELECTION_NULL` 可达。不蕴含。
- **8.5 site selection**：hurdle 是已知有好源的 target，只能声称该案例。
- **8.6 是否重演本轮教训**：M25（不涉跨 target）、M26（share 差已查）、
  M27（不做跨批数值比对，只比达阈步数的**序**）、M28（bank 配置将按实际输出核验）、
  M29（无事后归纳）、M30（判据先于新数据冻结）、M31（hurdle 符号已在 6 learner 上稳定）。
- **8.7 判据切换红线**：全新数据 + 沿用已冻结的达阈口径，不涉及换门。

## 7. 能与不能声称

**能**（若 `SELECTION_VALUABLE`）：在 hurdle 上，racing 选中的源（run）
比 racing 判定最差的源（stand）带来显著更快的达阈，
因此"自动选源"的价值不能被"任何源都行"解释。

**不得**：不得称跨任务成立（单 target）；不得称统计显著（判据是 per-seed 配对的序）；
不得省略 `M24` 的单批限制；不得把本结果与 door 的避损核算合并。

## 8. 不得做的事

- 裁决后不得调 θ、seed 数或剂量带。
- 若 `SELECTION_NULL`，**如实报告**——它直接削弱 racing 的实用主张，不得改判据抢救。
