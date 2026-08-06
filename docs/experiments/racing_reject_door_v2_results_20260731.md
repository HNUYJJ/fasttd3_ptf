# 结果：RACING_REJECT v2 —— `REPLICATION_DIVERGED`，主终点不予裁决

> 2026-07-31。预注册 `docs/experiments/racing_reject_door_v2_prereg_20260730.md`
> （v2 = `ac1bd26`，修订 v2.1 = `971b6af`，R8 = `67d139c`），
> 裁决脚本 `scripts/analysis/analyze_racing_reject_door_v2.py`（`06d37dc`）。
> 全部判据在揭盲前提交；揭盲前未查看任何 U / return 数值。

## 1. 裁决

```
VERDICT: REPLICATION_DIVERGED        （按 §5.4 优先级，主终点与层3 均不执行）

层1 工程硬检查   PASS   24 臂 × 3 K：剂量 / 臂身份 / 冻结面板 / sha256 / 协议 全通过
层2 复制检查     6/9 FAIL
```

脚本已验证**未输出任何 `K≤5000` 的 per-seed U、排序或命中数**。
**主终点（"racing 能否在 K≤5000 提前拒绝"）本轮无结论。**

## 2. 层2 明细：符号复现，数值不复现

```
              本次      gate      差       容差(评估噪声)
stand_s1    −50.25    −23.43   −26.82      ±11.04   FAIL
stand_s2    −47.76    −42.83    −4.93      ±10.84   PASS
stand_s3    −34.98    −31.66    −3.32      ±14.86   PASS
walk_s1     −31.29     −7.06   −24.23      ± 8.80   FAIL
walk_s2     −82.36    −38.58   −43.78      ±15.84   FAIL
walk_s3      −7.57    −20.96   +13.39      ±13.90   PASS
run_s1      −50.40    −17.78   −32.62      ±11.31   FAIL
run_s2      −52.02    −41.04   −10.98      ± 9.62   FAIL
run_s3       −8.67    −33.08   +24.41      ±14.27   FAIL

符号一致  9/9        |差| 中位 24.23，最大 43.78        容差中位 11.31
```

**door 的"三源全负"在方向上完整复现（9/9），但数值无法复现到评估噪声精度。**

## 3. 根因：容差量纲错配（我的设计错误）

容差取 `3 × paired_se`，即**评估噪声**（同一 checkpoint 的 128 episode 间方差，±3–5），
却用来卡 **run-to-run 训练漂移**（同 seed、同 anchor、同协议重跑之间的差异，实测 ±3.3–43.8）。
这与 `M16` 属同一类错误——**用 episode 尺度代替 learner 尺度**。

该失效模式**已在预注册 §5.2 预先声明**：

> paired SE 是**评估噪声**，不含 run-to-run 训练漂移（E15），
> 故本检查对 CUDA 漂移的误杀率无法预先标定。
> `REPLICATION_DIVERGED` **不自动等同于实现 bug**。

因此按预注册如实裁决，**不调容差抢救**（§9 明令禁止）。

**独立佐证漂移属正常量级**：`RACING_K` 在 hurdle 上同 seed 两批的 K=10000 漂移为
`−12.4 / −20.1 / −13.0`（约 15），与 door 此处的中位 24 同量级。
两者都不是实现错误，而是 E10/E15 描述的 CUDA 非确定性本底在**训练**层的累积。

## 4. 这个结果的真正价值：一个影响面很大的方法学发现

> **本项目此前全部 per-seed `U` 标签都只有单次运行，从未刻画 run-to-run 不确定性。**
> 本轮首次做了同协议重跑，测得 door 上 `|ΔU|` 中位 **24.23**、最大 **43.78**——
> 而这些标签的效应量本身只有 `−7` 到 `−43`。

直接后果（须写入标签清单的使用说明）：

1. **符号/排序可用，数值不可用**。door 的符号 9/9 复现，但把 per-seed 数值当作
   可复现到 ±10 以内的真值是错的。`EQD30K` / `sibling gate` / `door gate` 的
   per-seed 点值同样只有单次运行支撑。
2. **凡"与已发表值比对"的复制检查，容差必须基于 run-to-run 漂移**，
   而非评估噪声。用后者会系统性误杀。
3. 这为 `M17`（learner-path dependence）补上了**同协议重跑**这一层证据：
   此前 M17 依据的是通道归因跨 seed 反向，本轮显示即使固定
   `(source, target, stage, dose, anchor, noise seed)`，`U` 的数值仍漂移到与效应量同量级。

**对 racing 的含义**：racing 的决策（`argmax` 或 `REJECT`）只依赖**符号与排序**，
不依赖精确数值。door 符号 9/9 复现是有利证据；但这条不能替代主终点——
主终点问的是"**短 K** 能否复现该决策"，本轮未能进行。

## 5. 另一处已修正的错误（R8）

首轮裁决输出 `VOID_ENGINEERING`，54 条缺陷全部是
`source_names=['stand']（应为['stand','null']）`。**这是验收项写错，不是数据问题。**

- door 的 bank：`null_option: false` → `source_names = ['stand']`
- hurdle 的 bank：`null_option: true` → `source_names = ['run','null']`

R3 写死 `[arm,"null"]` 是把 hurdle 的模式套到了 door 上。修正为与配置无关的等价判据
`[n for n in names if n != "null"] == [arm]`，仍精确防臂对调，主终点未变。

**两点自我记录**：(a) 我在数小时前的剂量验收里**自己打印过** `names=['stand']`，
写 R3 时没有回看；(b) 该项源自 Codex 建议，它引用了 `source_bank.py` 的保存逻辑——
我核实了那段代码，却没核实**它在 door 场景下的实际输出**。
**核实必须到"本场景实际值"这一层。**

## 6. 下一步（不在本文裁决范围）

主终点要得到结论，需要 v3，且必须用**新数据**：

- 复制检查的容差从**独立来源**标定（run-to-run 漂移尺度），或改为
  **只要求符号与排序复现**——但该判据须在看到新数据之前冻结；
- 不得复用本轮数据重新裁决（容差已被本轮结果影响）；
- 仍须满足 `CLAUDE.md` §8 的设计层五问。

## 7. 数据与复现

```
预注册        docs/experiments/racing_reject_door_v2_prereg_20260730.md  (ac1bd26 / 971b6af / 67d139c)
裁决脚本      scripts/analysis/analyze_racing_reject_door_v2.py          (06d37dc)
批1 数据      docs/data/racing_reject_door_v1/source_free_eval/   36 点（seeds 1-3，v1 未揭盲留存）
批2 数据      docs/data/racing_reject_door_v2/source_free_eval/   36 点（seeds 4-6，新建 anchor）
裁决输出      docs/data/racing_reject_door_v2/results.json        run_id=2aada41d4b69
剂量          批1 0.4980–0.5006   批2 0.4990–0.5047   带 [0.48,0.52]
anchor        24/24 臂日志含 "Resumed core learner ... at step 10000"
ground truth  docs/experiments/door_at10k_gate_v1_results_20260727.md
```

批2 的 seeds 4–6 anchor 为本轮新建，协议与 s1–3 逐项相同
（`completed_vector_steps=10000`、`environment_transitions=1280000`、`num_envs=128`）。
