# 结果：hurdle 上 reward-bearing bootstrap 的样本效率加速倍数 —— SPEEDUP_CONFIRMED

> 2026-07-30。预注册 `docs/experiments/hurdle_speedup_v1_prereg_20260730.md`（commit `ba7a7de`），
> 裁决脚本 `scripts/analysis/analyze_hurdle_speedup_v1.py`（commit `1fcf136`），
> 二者**均在任何长程臂被评估之前提交**。本文只报告脚本输出，未做任何事后调整。

## 1. 裁决

```
VERDICT: SPEEDUP_CONFIRMED

θ= 200  speedup 中位数=4.38  均值=4.34  per-seed=[4.382, 4.592, 4.059]   PASS
θ= 300  speedup 中位数=3.59  均值=3.72  per-seed=[3.472, 4.099, 3.594]   PASS
θ= 400  speedup 中位数=3.20  均值=2.77  per-seed=[1.494, 3.610, 3.199]   ----  (含右删失)

100k 终点: scratch=387.40  source=479.07  反超=False
```

判据要求"≥2/3 阈值上 speedup ≥ 2.0 且这些阈值上 3/3 seed 的 per-seed speedup ≥ 1.5"。
θ=200 与 θ=300 满足，θ=400 不满足，2/3 → `SPEEDUP_CONFIRMED`。

**θ=400 是以 0.006 之差未过**（s1 实得 1.494，门槛 1.5）。不做任何补救，
但必须指出这个阈值上有两处方向相反的偏差，都在裁决脚本冻结时就存在：

- scratch 的 s1/s2 在 100k 仍未达到 400，按预注册记为右删失 `steps=100000`。
  真实达阈步数只会更大，因此**分子被低估** → speedup 被低估；
- source 的 s1 因 30k→50k 的回撤（见 §4），达阈被推迟到 66919 步，
  **分母被推高** → speedup 被低估。

两者都指向"θ=400 的真实 speedup 高于 1.494"，但预注册不允许我据此改判。
**记录为未通过。**

## 2. 剂量验收（预注册 §3 的作废条件）

```
behavior share:  0.4994 / 0.4983 / 0.4995     带 [0.48, 0.52]   PASS
critic   share:  0.5001 / 0.5000 / 0.5000
```

三 seed 全部达标，与 EQD30K 实测的 0.500–0.502 一致。
这排除了"加速来自源被用得更多"这一平凡解释：**两臂的行为预算严格对半**。

臂身份的运行时证据（读自 checkpoint，非命令行）：

```
source_s1/s2/s3 : source_names=['run','null']   有 admission_audit   exec share 0.4994/0.4983/0.4995
scratch_s1/s2/s3: source_names=['null']         无 admission_audit
```

scratch 臂只含 `null`，符合项目一贯的盲化硬判据。两臂的其余训练参数逐字节相同
（`--num-envs 128 --batch-size 32768 --buffer-size 51200 --learning-starts 10
--num-updates 2 --no-compile`，seed 配对），唯一差别是 `SOURCE_BANK` 与 admission 配置。

## 3. 两臂曲线（source-free student，deterministic，128 episodes）

```
[scratch]
   s1: 10k=3.4    20k=5.3    30k=7.2    50k=37.5   75k=138.3  100k=349.6
   s2: 10k=12.9   20k=15.5   30k=37.8   50k=128.1  75k=192.1  100k=352.4
   s3: 10k=4.0    20k=8.0    30k=19.5   50k=49.8   75k=226.4  100k=460.2

[source]
   s1: 10k=57.2   20k=219.8  30k=332.5  50k=114.5  75k=536.3  100k=569.9
   s2: 10k=93.8   20k=254.7  30k=443.3  50k=565.3  75k=698.5  100k=111.3
   s3: 10k=43.3   20k=250.7  30k=412.2  50k=626.7  75k=702.8  100k=756.0
```

| 步数 | scratch 均值 | source 均值 | 差 | 倍率 |
|---|---|---|---|---|
| 10k | 6.78 | 64.75 | +57.98 | 9.56× |
| 20k | 9.60 | 241.72 | +232.12 | **25.18×** |
| 30k | 21.49 | 396.01 | +374.51 | 18.42× |
| 50k | 71.81 | 435.51 | +363.69 | 6.06× |
| 75k | 185.60 | 645.88 | +460.29 | 3.48× |
| 100k | 387.40 | 479.07 | +91.67 | **1.24×** |

**倍率随训练推进单调衰减到 1.24×。** 这不是免责声明，而是定量结果：
它直接说明本实验只能支撑**早期样本效率**的主张，
到 100k 时 scratch 已基本追平。见 §5。

## 4. 必须同时报告的负面发现：训练不稳定性是 source 臂独有的

```
source  s1: 30k=332.5 → 50k=114.5    回撤 −218.0  (−66%)
source  s2: 75k=698.5 → 100k=111.3   回撤 −587.3  (−84%)
scratch s1/s2/s3:                    无任何回撤（三 seed 六个评估点全程单调上升）
```

2/3 的 source seed 出现大幅回撤，scratch 臂**一次都没有**。这是干净的配对对照，
不能作为噪声处理。

s2@100k 已排除评估侧成因：

- checkpoint 的 sha256 与 75k 不同，`identity_checked=True`，非串台；
- 回报分布是**单峰低位**（min=9.09 / 中位 101.43 / max=282.45），
  而非"少数 episode 拉低均值"的双峰；
- 早停计数从 75k 的 7/128 回升到 100k 的 69/128，与回报反向一致。

> **字段语义警告**：`aggregate.success_count` 读自 `terminated`，
> 在 hurdle 上是 `terminate_when_unhealthy` 触发的**摔倒早停计数**，
> **不是任务成功率**（`p0_evaluator.py:17` 的"成功"语义只对 package/truck 成立）。
> 该字段与 return 强反向（s2：10k 摔 103 次 return 93.8；75k 摔 7 次 return 698.5）。
> 引用时务必按"摔倒"解读，否则结论会被完全读反。

崩溃的时点只能定位到 **75k 与 100k 之间**：训练日志因 `EVAL_INTERVAL=0` 只有进度条，
wandb 本地记录因 SDK 版本不兼容无法解析，可用粒度就是这 6 个评估点。

**这一条削弱的是"长程可用性"，不是"早期加速"**：两次回撤都发生在全部三个阈值达成之后，
不影响 §1 的达阈步数。但它意味着不能声称该方法能稳定地训到更高水平。

## 5. 与已发表 EQD30K 的核对：一次独立复现，以及一处 40% 偏差

| 量 | 本实验 | EQD30K 已报 |
|---|---|---|
| 30k 的 U（source − scratch，配对） | **+374.51** | **+379.66**，CI90 [+271.5, +487.9] |
| scratch@30k 绝对水平 | **21.49** | **35.94**（48.54/26.92/32.35） |

**U 落在已报 CI90 内，与点估计只差 5.15** —— 这是 `EQD30K.hurdle.run` 的一次独立复现
（不同随机流、不同干预窗口长度）。

但 **scratch@30k 的绝对水平差了约 40%**（21.49 vs 35.94）。已核实两个实验的
训练配置逐项相同（见 §2），因此这个偏差归因于 **seed 与随机流**，而非配置差异：
hurdle 在 30k 时 scratch 仍处于极低回报的早期段，绝对水平的 seed 间方差本就很大
（本实验三 seed 为 7.2 / 37.8 / 19.5，跨度 5 倍）。

**方法学要点**：同一批数据里，**配对差值 U 的复现精度（差 5.15）远高于绝对水平
（差 14.45，占 40%）**。这支持项目一贯的做法——效应量一律按 learner seed 配对报告，
不比较跨实验的绝对回报。

## 6. 本结果能声称什么、不能声称什么

**能声称：**

- 在源已知有用时，reward-bearing bootstrap 在 hurdle 上提供
  **约 3.5–4.4× 的早期样本效率提升**（θ=200 中位数 4.38，θ=300 中位数 3.59，
  两个阈值各 3/3 seed per-seed ≥ 1.5），剂量严格对半，配对同 seed。
- hurdle 是主流 model-based 方法几乎解不了的任务（TD-MPC2 仅 64.68/700），
  因此该加速发生在一个有外部难度依据的任务上。

**不能声称：**

1. **不能声称最终性能。** 本实验只跑到 100k，而 FastTD3 原文在单 A100 三小时内解多任务，
   100k 只是其中一段。倍率到 100k 已衰减到 1.24×，
   且 source 臂在此处出现 84% 回撤。**主张严格限于早期样本效率。**
2. **不能声称解决了自动选源。** `run` 源是**人工指定**的——它之所以已知有用，
   正是因为先前用真实交互测出了 `U=+379.66`。本项目已有**十一个**迁移性预测信号族
   全部失败（`docs/impossibility_characterization_of_transfer_prediction_20260730.md`），
   自动选源仍未解决。本结果量化的是"选对源之后的收益上限"，
   它恰恰构成了那条不可能性刻画的**另一半**：**值得为选源付出交互代价**
   （hurdle 上 run 与 stand 的 U 差额为 +379.66 − 51.28 = +328）。
3. **不能声称跨任务成立。** 单 target（hurdle）、单源（run）、3 seeds。
   已知 door 上三个 loco 源一致有害（9/9 per-seed 负），
   crawl 上存在负迁移。**方向依赖是本项目反复确认的事实。**
4. **不能声称长程稳定。** 见 §4。

## 7. 数据与复现

```
预注册            docs/experiments/hurdle_speedup_v1_prereg_20260730.md   (ba7a7de)
裁决脚本          scripts/analysis/analyze_hurdle_speedup_v1.py           (1fcf136)
剂量脚本          scripts/analysis/audit_hurdle_speedup_dose_v1.py        (d66cee1)
训练              scripts/run_hurdle_speedup_v1.sh
评估              scripts/eval_hurdle_speedup_v1.sh   (128 ep, deterministic, source-free)
裁决输出          docs/data/hurdle_speedup_v1/hurdle_speedup_v1_results.json
剂量输出          docs/data/hurdle_speedup_v1/dose_audit.json
36 个评估点       docs/data/hurdle_speedup_v1/source_free_eval/*.json
                  （每个含 checkpoint sha256、identity_checked、128 条 episode 明细）
```

**评估执行中的一处操作失误（已修复，未污染数据）**：首轮 6 路并行评估时，
两组的 `STEPS_LIST` 在 `{50000, 75000, 100000}` 上重叠，而
`eval_hurdle_speedup_v1.sh` 的 `[[ -f "$OUT" ]] && skip` 只在启动瞬间检查、不是原子锁，
导致三个 50k 点被两个进程并发写同一文件。已停止全部进程，
对已产出的 33 个 json 逐个校验（JSON 可解析、episode 数=128、
`global_step`/seed/臂/ckpt 路径/`identity_checked` 全部自洽）确认无损坏，
再以单 seed 单步数的方式重跑那 3 个点。**最终 36 点全部通过校验。**
