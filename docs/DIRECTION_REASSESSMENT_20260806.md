# 方向重估记录：从"估计 U"转向"改变 U 的可达上界"

> 2026-08-06。触发者：PI。外部 review：ChatGPT（两轮）。
> 本文件只记录**为什么转向**与**核实了什么**，不含任何新实验结论。

## 1. PI 的批评与其精确表述

PI 指出："从最开始到现在思路没有根本性转变，做来做去还是源怎么选、加多少源、什么时候退出。"

更精确的表述是：

```
族 1–12   : 预测 U   → 全败（不可能性刻画已成文）
racing    : 测量 U   → 可行，但正是 PI 最初就提出的"放进目标环境跑一跑排序"
共同点    : estimand 恒为 U 本身，十三次尝试从未试图改变 U 的可达上界
```

按 CLAUDE.md §1 的判据（"estimand 没变就是换皮"），这确实是同一族的第十三次。
该判据由本项目自己写下，却未对自身执行。

## 2. 三条被核实的关键事实

| # | 断言 | 核实方式 | 结果 |
|---|---|---|---|
| 1 | slide 上同源同剂量、仅差退出时机 → 293.3 vs 929.1（3.17×） | `docs/data/endtoend_v1/results.json` 原始字段 | **成立**。且全程注入（293.3）比 scratch（792.4）差 2.7× |
| 2 | 该现象**不是**"不可逆局部最优" | hard-exit 臂未加任何逃逸机制，仅从同一 30k anchor 恢复并置 `admission_mode=none`，普通 FastTD3 即达 929 | **我的原解释被推翻**。正确读法是**持续占用的机会成本**，非不可逆吸引盆 |
| 3 | slide/hurdle 均无展示"上限提升"的空间 | `tasks.py:12` `max_episode_steps=1000`；`Slide(ClimbingUpwards)`/`Hurdle(Walk)` reward 均为 `[0,1]` 项相乘；`Walk.success_bar = 700` | **成立且比预期更强**：slide 929.1、hurdle 840.4 **均已超过官方 `success_bar`**，按 benchmark 标准已判定解决 |

事实 2 是我自己的实质错误：用了一个比数据更强的词（"吸引盆"），
而排除平凡解释正是 CLAUDE.md §8.1 的要求。记入教训。

## 3. 外部 review 的三处事实错误（逐条核实原文）

不盲信 review 结论，逐条 `grep` 核实（CLAUDE.md §0）：

1. **`fixed_quota` 不是训练开关**。`admission_replay_mode` 只有 `shared` /
   `student_only` 两个合法值（`train_ptf.py` 显式 `raise ValueError`）；
   `physical_after_authority` 是 `--ptf-admission-replay-handoff` 的值；
   `fixed_quota` 只是 `analyze_admission_handoff.py` 里的分析分组名。
   **后果**：2×2 通道分解的 `B⁻R⁺` 臂**没有现成开关，必须改代码**，
   review 称"原则上不需要改核心算法"不成立。
2. **十一族 / 十二族口径**：review 称仓库口径是十一族 —— 部分正确。
   主文档已是十二族，但 `transfer_utility_is_not_a_property_20260731.md` 与
   `PIPELINE_FULL_WALKTHROUGH_20260802.md` 尚未同步。这是 P0 的遗漏，**已修**。
3. **review 只查了 slide 的 headroom，未查 hurdle**。核实后 hurdle 同样饱和。

review 对我的两条批评则成立：解释过度（§2 事实 2），以及
"普通 checkpoint 不含 replay"——已核实 `save_ptf_params` 的 `save_dict` 中确无
replay transition，故 30k→100k 期间的真实训练 replay 占用**无法**离线重建；
只有 30k anchor 含完整 `replay.pt`。我上一轮"anchor 和 checkpoint 都在"的说法不严谨，
且路径写错（slide anchor 在 `slide_hard_exit_v1/` 而非 `endtoend_v1/`）。

## 4. 新旧方向对比

| | 旧（族 1–12 + racing） | 新 |
|---|---|---|
| estimand | `U` 的值 | `max U` 的**可达上界** / 退出后的探索分布 |
| 干预对象 | 选哪个源、灌多少、何时退 | 源退出后 target 能否突破源诱导的起点平台 |
| 现状 | 已收敛，负面结果已成文 | 有一个 3.17× 的已测效应待解释，但**尚无合适判决场** |

**再提"选哪个源 / 灌多少 / 何时退"即第十四次换皮。**

## 5. 当前状态与下一步

- 论文双支柱（不可能性刻画 + 最小充分测量）已成文，`docs/PAPER_DRAFT_20260806.md`，
  作为保底路线保留，**不因本次转向而作废**；
- 新方向的第一个障碍不是机制设计，而是**没有场地**：两个正面 target 均已饱和；
- 因此下一步是场地普查，判据已冻结于
  `docs/experiments/post_transfer_autonomy_site_screen_prereg_20260806.md`；
- 普查若输出 `INSUFFICIENT_SITES`，则新方向在本 benchmark 内无法验证，
  应如实记录并回到保底路线，**不得降低门槛凑数**。

## 6. 已停止的事项（追加到 guardrails）

- 停止寻找第十三种零成本迁移性代理量；
- 停止 exit-threshold 的调参式实验（27k / 30k / 32k 之类）；
- 停止在 slide / hurdle 上追求"上限提升"——两者按官方 `success_bar` 均已解决。
