# RBO-PTF 每日接力（2026-06-15 夜 → 06-16 晨）

昨夜挂了一条无人值守全链，今天醒来照本文推进。

## 1. 昨夜挂的全链（tmux watcher `full_chain`，脚本 `/tmp/full_chain.sh`）

三级顺序，每级 4-slot（GPU 0/1/2/4）+ 内存守护（available<25GB 暂缓）：

```
级1 terrain (12 runs)  →  级2 wfix (4)  →  级3 negctrl (4)   = 20 runs
```

- **STAMP**：terrain = `20260615T044012Z`（固定）；wfix/nc 的 STAMP 动态写在
  `/tmp/wfix_stamp.txt`、`/tmp/nc_stamp.txt`（级2/级3 启动时才生成）。
- **昨夜快照**：terrain safe×4 @~55%（55k/100k），terrain final 0/12，load 85，mem 101GB。
- **各级验证目标**：
  - **terrain**：stair/slide/pole/crawl × {safe(reward-weighted 抽 walk/run)、rand(uniform 含无用 stand)、scr(scratch)}。核心问：reward-weighted 是否正迁移、且优于 uniform。
  - **wfix**（weighted 源 + horizon=25，与 rand 的 horizon 对齐）：解耦两变量——
    **wfix vs rand = 纯源选择增益**（h 都=25）；**safe vs wfix = 纯执行时长增益**（源都 weighted）。回应 reviewer 对"源选择与执行时长纠缠"的质疑。
  - **negctrl**：door/spoon × {safe, scr}。Effect Map+Day1 预测无对价（loco 源 robot-proprio adapter 够不到门把手/勺子 bottleneck）。预测 **safe ≈ scr**，验证 Map 的解释边界。

## 2. 今天第一件事：查全链是否跑完

```bash
# final.pt 计数（terrain 12 / wfix 4 / nc 4）
WS=$(cut -d= -f2 /tmp/wfix_stamp.txt 2>/dev/null); NS=$(cut -d= -f2 /tmp/nc_stamp.txt 2>/dev/null)
ls models/*tp_*_s1_20260615T044012Z__1_final.pt 2>/dev/null | wc -l   # 期望 12
ls models/*tp_wfix_s1_${WS}__1_final.pt 2>/dev/null | wc -l           # 期望 4
ls models/*nc_*_s1_${NS}__1_final.pt 2>/dev/null | wc -l              # 期望 4
tmux ls | grep -E 'full_chain|_slot'    # 全跑完则只剩(或无)full_chain
tail -5 /tmp/full_chain.log
```

## 3. 分析方法（数据齐了再写脚本，别凭空写——昨夜两次踩幻觉坑）

**真实 wandb 坐标**（已核实自 `scripts/aggregate_multitask.py`，勿用记忆里的别的值）：
- `wandb.Api().runs("yujiajie-nju/fasttd3_ptf")`
- run 名 = `run.config.get("exp_name") or run.name`
- 指标：`run.history(samples=4000, keys=["_step", "eval_avg_return"])`
- AUC = `np.trapz(eval_avg_return, _step) / step_span` = 平均 return

**做法**：复制 `aggregate_multitask.py` → `analyze_chain.py`，把正则换成
`r"h1hand_([a-z_]+?)_(?:tp_(safe|rand|scr|wfix)|nc_(safe|scr))_s(\d+)_"`，分三段输出：

1. **terrain 绝对 AUC**：stair/slide/pole/crawl × {scr, rand, safe, wfix}
2. **解耦**：每任务 `wfix−rand`（源选择增益）、`safe−wfix`（执行时长增益）
3. **negctrl**：door/spoon `safe` vs `scr`

**判读标准**：
- 核心论点成立 = terrain 上 safe/wfix 的 AUC > rand > scr
- 源选择有价值 = wfix > rand（即便 safe≈wfix，也说明价值在源选择而非 horizon）
- negctrl 成立 = door/spoon safe ≈ scr（差距远小于 terrain 正迁移幅度）

## 4. 出结果后的分支（依结果决定，勿提前）

- **若 reward-weighted > uniform 成立** → 补 seed 2/3 做统计；再启动**阶段B**（semantic obs adapter for manip basketball/bookshelf + reach source）。
- **若 null（safe≈rand≈scr 或 terrain 无正迁移）** → 诊断：很可能 loco 对 terrain 覆盖太强、三种源选择差异被淹没；据此调整论点（也许 negative-transfer 任务上源选择差异才显现）。
- 方法定名 `reward_weighted_bootstrap`；论文 3 贡献结构（诊断/方法/实证）成稿；发 ChatGPT。

## 5. 今天（06-15）已完成

- **真正提交**本会话全部工作：6 个 commit（`561b340`→`4ba7a68`）+ `b183f40`(wfix bank)；停止追踪 `MUJOCO_LOG.TXT`。
- 采纳 ChatGPT 分身两条意见：**wfix 解耦消融** + **negctrl 解释力检验**，并合并进全链。
- ⚠️ **两次踩 output-token-limit/interrupt 幻觉坑**：第一次"5 个 commit 成功"是假的（reflog 证伪，实际没提交，已重做）；第二次"读到的 aggregate_multitask.py 内容"是假的（真实文件正则/字段全不同）。**教训：中断/截断后出现的工具结果与文件内容一律不可信，必须 `git reflog`/重新 `Read` 核实再行动。**

## 关键路径速查

- 全链：`/tmp/full_chain.sh`；watcher `tmux full_chain`；log `/tmp/full_chain.log`
- terrain queue：`/tmp/terrain_pilot_queue.sh`；STAMP `/tmp/terrain_pilot_stamp.txt`
- bank：`configs/source_banks/h1hand_loco_{safe,sources,wfix}_{task}.yaml`
- 分析：`scripts/aggregate_multitask.py`（复制扩展为 analyze_chain.py）
- Effect Map（贡献1）：`docs/source_target_effect_map_v1.md`
