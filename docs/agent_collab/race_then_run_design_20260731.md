# 【已作废，未执行】设计草案：RACE-then-RUN

> **2026-07-31 作废，未跑任何臂。** Codex 决策前 review 判 `do-not-proceed-yet`，
> 五条致命缺陷我逐条核实成立。
>
> ## 作废原因
>
> **F1（我完全没想到）· `argmax_i U_i ≡ argmax_i J_i` 是数学恒等。**
> 因 `U_i = J_i − J_student` 且 student 基线与 `i` 无关，
> 在**同一批数据内**选源时，"测 U"与"直接看谁当前回报最高"是**同一个操作**。
> 故本设计无法区分"racing 测到了延迟迁移效用"与"贪心续训当前最好的 checkpoint"。
> （此点**不影响** `RACING_K`：那里是 `U(小K)` 排序对比 `U(30k)` ground truth 的
> **跨 horizon** 验证，且有 12/12 排出 `walk>stand`、与 zero-shot 反向的辨别证据。）
>
> **F2 · best-of-two 的 order-statistic 效应。** B/C 从同一批 20k checkpoint 取
> best/worst 再各自续训，故 `B>C` 可能只反映"这批里运气最好的 learner path"，
> 而非"更好的 source"。修复需用独立批决定 source identity，再用新随机流从共同 anchor 续训。
>
> **F3 · speedup 口径失真 + 成本核算错误。** `hurdle_speedup_v1` 的公式成立是因为两臂
> 均从 `t=0` 出发；本设计让 A/B/C 从**性能不同**的 anchor 起步却直接套用该公式。
> 且"A = 30k + 80k"不成立——另外两个源的 20k 交互并未改善 A，
> 记到 A 账上是会计对齐而非学习预算对齐；20k 处的 128-episode 选择评估也是真实 target 交互，未计入。
>
> **F4 · 单批 n=3 违反 M24**（RACING_MULTI 已因同一条被判 FATAL）。
>
> **F5 · outcome-informed site selection 未被排除**：slide 是已知 `walk` 胜出后才选的，
> 一个**完全不跑 racing、永远选 walk** 的规则会得到同样的 B。
>
> **另一条我自己刚犯过的错**：草案让 B/C 以 50% 剂量续训到 100k，全程不终止教师——
> 而当天的源天花板探针（`469c1fb`）刚证明这正是 hurdle 长程崩溃（1.24×、s2 −84%）的原因。
> 一个会在长程崩溃的方法不能称"完整方法"。
>
> ## 唯一通过的一项
>
> 增设 argmin 对照臂 C 使 `B≈C` 真实可达，**未重犯 v1 的不可证伪错误**；
> 直接测 `U` 的路线本身合法，不属 CGT 式代理换皮。
>
> ## Codex 给出的下一步排序（已采纳）
>
> 1. **RACING_REJECT v3** —— 先裁决当前真正未决的 door 拒绝主终点；
> 2. 重写 RACE-then-RUN 并先做**廉价的 selector-stability gate**，不启动 80k 续训；
> 3. gate 通过后才跑完整版，且需两批新 learner seeds；
> 4. 最后才扩候选源/跨 target（且不得用已知弱的 stand 充当容易的 argmin）。
>
> ---
>
> 以下为原始草案，仅供追溯，**不得执行**。


> 2026-07-31。**尚未写代码、尚未跑任何臂。**
> 目标：把已有的两个结果串成一个**完整方法**，并回应"单任务"与"源是人工指定"两条边界。

## 1. 动机：两个已有结果之间缺一条链

| 已有 | 结论 | 边界 |
|---|---|---|
| `RACING_K`（`a744adb`） | K\*=10000，hurdle 上 6/6 选出 run | 只**选源**，不改进学习本身 |
| `hurdle_speedup_v1`（`20f1e11`） | run 源带来早期 3.5–4.4× 加速 | **run 源是人工指定的** |

两者从未端到端串联：racing 选出的臂没有被继续训练，speedup 的源是我手工选的。
串起来就是一个不需要人工介入的方法：

```
RACE-then-RUN：N 个源各跑 K 步（并行）→ 按 argmax_i U_i(K) 选源（或 REJECT）
              → 从选中臂继续训练 → 与 scratch 比较加速
```

## 2. `CLAUDE.md` §8 设计层自查（这一节是本草案的重点）

### 8.1 辨别力 —— **一个差点漏掉的致命缺陷**

平凡解释：**"用任何源都能加速，选哪个无所谓。"**

若只跑「选中源 vs scratch」两臂，结果无论多好都**无法区分**
"racing 选对了源"与"随便用个源都行"——那样这个实验证明的是"迁移有用"
（`hurdle_speedup_v1` 已证），而不是"**自动选源**有用"（本实验的主张）。

**修复：必须增加"racing 判定最差源"的对照臂。**

```
臂 A  scratch（student-only 续训）
臂 B  racing 选中的源（argmax_i U_i(K)）续训       ← 处理
臂 C  racing 判定最差的源（argmin_i U_i(K)）续训   ← 关键对照
```

主张成立需要 **B > C**，而不只是 B > A。
若 B ≈ C 且都 > A，结论是"用源有益但选源无价值"——这会**否定** racing 的实用主张，
是一个真实可达且有意义的负结果。

第二个平凡解释："选 |U| 最大者"——在本设计中 `argmax U` 就是判据本身，不构成额外捷径。

### 8.2 混淆变量

- **剂量**：三臂 B/C 用相同的 `behavior 0.5 / replay 0.5`，逐 checkpoint 验收在
  `[0.48,0.52]`；A 为纯 student。
- **续训起点**：B/C/A 均从**各自 racing 臂的 anchor** 续训，
  故 B/C 已各自消耗了 K 步。**A 也必须从 student 臂的 anchor 续训**，
  保证三臂的总交互步数相同。
- **racing 阶段的交互成本**必须计入 B/C 的总预算（见 §4）。

### 8.3 独立重复

3 seeds 为主，**如实标注为单批**。`M24` 要求独立重复——本设计因成本暂不做第二批，
故**不得**给出 `CONFIRMED` 级别的裁决，只能报为 pilot；
若结果为正，须在结论中写明"待独立重复"。

### 8.4 前提是否蕴含结论

racing 阶段的输出（选中/最差）是续训阶段的**输入**，不是其判据。
racing 若选错，实验照常进行并可能得到 B ≈ C 或 B < C。**不蕴含。**

### 8.5 site selection

slide 是**已知 walk 有用**（`U=+56.95`，3/3）之后才选的。
故只能声称"在这个已知有好源的任务上"，不得声称对任意任务成立。

## 3. 协议（草案）

```
target        h1hand-slide-v0
anchor        artifacts/slide_bac_gate_v1/anchors/s{1,2,3}（现成，10k 纯 student）
候选源        walk, stair源            （sibling gate 已测：U_walk=+56.9 > U_stair源=+36.1）
seeds         1, 2, 3
--- 阶段一 racing ---
臂            student / walk / stair源，各从 anchor 续训 K=10000（到 global_step 20000）
              **每臂在 20000 处存 anchor**（learner+replay+rng），供阶段二续训
选源规则      argmax_i U_i(K=10000)  → 臂 B 的源；argmin → 臂 C 的源
              若 max_i U_i ≤ 0 → REJECT，阶段二只跑 A（记为 RACE_REJECTED）
--- 阶段二 续训 ---
臂 A          student anchor@20k  → 100000
臂 B          argmax 源 anchor@20k → 100000
臂 C          argmin 源 anchor@20k → 100000
评估点        20k(=起点), 40k, 60k, 80k, 100k，source-free 128 ep
```

**总交互**：阶段一 3×10k=30k（并行，墙钟 10k）+ 阶段二 3×80k=240k。

## 4. 判据（草案，待冻结）

**主判据（选源是否有价值）**：

```
speedup(θ) = steps_A(θ) / steps_B(θ)          与 hurdle_speedup_v1 同口径
gain_of_selection(θ) = steps_C(θ) / steps_B(θ)     ← 本实验的核心量
```

- `gain_of_selection > 1` 且 3/3 seed 同向 → 选源有价值
- `gain_of_selection ≈ 1` → **选源无价值**（用源即可，不必选）——有意义的负结果
- 阈值 θ 须在跑前按 slide 的已公开 `r@end` 冻结，不得看结果后定

**成本核算**：B 的总交互 = 30k(racing) + 80k(续训)；A = 30k + 80k（同预算）。
racing 成本已内含，**不额外扣除**。

## 5. 请 review 的问题

1. §8.1 的"最差源对照臂 C"是否足以排除"用任何源都行"？还有没有别的平凡解释？
2. 三臂从各自 anchor 续训、总步数相同——这个对齐方式有没有隐藏的不公平？
   （B/C 的 anchor 是"带源训练过 10k"的，A 的 anchor 是"纯 student 训练过 10k"的，
   它们的起点性能本就不同，这是否使 speedup 的定义失真？）
3. 只有 2 个候选源（walk / stair源），`argmax` 与 `argmin` 只差一个二选一，
   对照臂 C 的信息量是否太弱？是否该扩到 3 个源（加 stand）？
4. 单批 3 seeds 的限制（§8.3）是否严重到不该开跑？
5. 成本约 4–5 机时。以当前证据状态，这是否是最该投入的下一步？
   还是应该先补 `RACING_REJECT v3`（door 主终点仍未裁决）？

## 6. 出处

```
RACING_K            docs/experiments/racing_min_horizon_v1_results_20260730.md   (a744adb)
speedup             docs/experiments/hurdle_speedup_v1_results_20260730.md       (20f1e11)
slide ground truth  docs/experiments/sibling_source_gate_v1_results_20260729.md
U 漂移 M27          docs/experiments/racing_reject_door_v2_results_20260731.md   (05c9002)
设计层规范          CLAUDE.md §8                                                  (5546ce2)
```
