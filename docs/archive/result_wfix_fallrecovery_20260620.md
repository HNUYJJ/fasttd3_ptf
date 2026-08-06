# RBO-PTF 新结果（2026-06-20）：wfix 解耦 + fall-recovery probe

承接 `handoff_discussion_20260616.md`。按 ChatGPT 排序执行了 Step 1（wfix 解耦）和
Step 0（fall-recovery probe）。两个结果都有超出预期的发现，诚实记录。

---

## 1. wfix 解耦（第②项，ChatGPT Step 1）— 3-seed 已加固

**设置**：stair/slide/pole/crawl × {scr / rand / wfix / safe} × **seed1/2/3**，每 task 共同窗口
AUC（到 95k）。seed2/3 wfix STAMP `20260620T145205Z`，其余沿用 terrain s2/s3。
- rand = uniform 源 + horizon 25
- wfix = weighted 源 + horizon 25（与 rand 只差**源选择**）
- safe = weighted 源 + horizon 50（与 wfix 只差**执行时长 horizon**）

**四方 AUC（3-seed mean±std）：**

| task | scr | rand | wfix | safe |
|------|-----|------|------|------|
| stair | 252.5±37 | 169.1±41 | 174.1±43 | **279.2±20** |
| slide | 271.1±46 | 450.2±20 | **522.7±20** | 504.7±14 |
| pole | 603.3±48 | 573.2±13 | **767.4±24** | 717.9±25 |
| crawl | 812.0±25 | 699.6±32 | 739.2±6 | 656.3±35 |

**解耦（按 (task,seed) 配对，N=12 个组合）：**

| 解耦项 | 平均 | >0 占比 | paired t | 判读 |
|--------|------|---------|----------|------|
| **wfix−rand（纯源选择）** | **+77.9** | **11/12** | **+3.08** | 稳健显著 ✓ |
| **safe−wfix（纯 horizon）** | −11.4 | 4/12 | −0.46 | **不显著（中性）** |
| safe−rand（总） | +66.5 | 9/12 | — | — |

per-task 的 horizon 项（safe−wfix）符号**不一致**：stair **+105.1**（3 seed 全正）、
slide −18.1、pole −49.6、crawl −82.9。

**结论（3-seed 修正 seed1）**：
1. **主机制 = reward-weighted 源选择，3-seed 坐实**（wfix−rand +77.9，11/12 正，t=+3.08；
   非 crawl 子集 +90.6，8/9 正，t=+2.81）。干净回应 reviewer "safe 比 rand 好是不是只因
   horizon 长"——**不是，是源选择**。这是论文 ablation 的核心数字。
2. **诚实修正：seed1 的"长 horizon 有害"不成立**。3-seed 后 horizon 项总体 **−11.4，t=−0.46
   不显著（中性）**，且符号任务依赖：stair 上长 horizon 反而**有益**（+105），pole/crawl 上有害。
   正确表述是 **"horizon 非主因、符号任务依赖、默认短 horizon=25 即可拿到绝大部分增益"**，
   而非"一律有害"。这反而让去耦更干净——**主增益来自源选择，horizon 不是混淆因子**。
3. **crawl 负迁移的机制**：crawl 上源选择仍 +39.6（正），但 horizon −82.9 把总增益拉成
   safe−rand **−43.3（负迁移）**。即"长时间执行站立教师"在 crawl 注入了不当经验——这正是
   abstain 机制的 motivating evidence（warmup 久执行 locomotion 教师 = 坏经验注入）。

**对方法定名的影响**：核心价值 = **reward-weighted 源选择**；horizon 是次要超参，方法定为
`reward_weighted_bootstrap`（weighted 源 + **默认短 horizon=25**）。
- 不再主张"砍掉 horizon"（stair 上它有益），而是"**默认短 horizon，长 horizon 收益任务依赖、
  非核心**"。abstain 应加在**最干净的 wfix 基线（短 horizon weighted）**上——crawl 上 wfix
  仍 +39.6 不像 safe 被 horizon 拖成负，是更好的 abstain 落点。

---

## 2. fall-recovery passive probe（Open Q2，ChatGPT Step 0）

**设置**：stair/slide/pole/crawl × {safe/rand/scr} seed1 final policy，确定性 rollout（32 env），
记录 per-env upright 时序，检测 near-fall→recovery（`scripts/probe_fall_recovery.py`）。

| task | method | ep_len | upright | **ended_by_fall%** |
|------|--------|--------|---------|------------|
| stair | safe | 984.7 | 0.995 | **3%** |
| stair | rand | 904.9 | 0.992 | 22% |
| stair | scr | 863.5 | 0.985 | 25% |
| slide | safe | 1000 | 1.000 | **0%** |
| slide | rand | 884.5 | 0.999 | 16% |
| slide | scr | 957.7 | 0.995 | 9% |
| pole | safe | 1000 | 1.000 | **0%** |
| pole | rand | 955.3 | 0.994 | 6% |
| pole | scr | 969.2 | 0.997 | 6% |

**三层解读**：
1. **强信号（有效）**：safe 的摔倒终止率全任务最低（**0–3%** vs rand/scr 6–25%），ep_len/upright
   也最高。跨 stair/slide/pole 一致 → RBO-PTF 学到更鲁棒（更少摔）的策略。
2. **诚实修正 PI 观察**：ChatGPT 预言对了——**terrain 任务一摔即 done**，near-fall→恢复事件几乎为 0
   （全表仅 pole 3 个）。passive probe **测不到"跌倒后恢复"**，只能测"摔倒终止率"。PI 看到的
   "safe 摔倒后站起来"，在 passive rollout 里呈现为"safe 根本很少摔到终止"。
3. **crawl 指标失效**：crawl 全 0（ep_len=0/upright=0），因爬行任务 info 无 `upright`（不直立）。
   crawl 需换指标（root height / 前进距离）。

**active perturbation probe（已做 v1/v2）— 撞上环境硬限制**（`scripts/probe_active_recovery.py`）：
- v1（root 线速度冲击 1–3 m/s）太弱，min_upright 仍 0.86，没推倒。
- v2（线速度+角速度 6–13 m/s + 5–11 rad/s，4 方向 ×3 强度）：扰动够大了，但 stair/slide 即使
  xl 档 survive=0%、min_upright 仍 0.73–0.80 —— **机器人 1–7 步内就被判 done，upright 还没倒到底
  episode 就结束**。零动作探查确认：stair `done_fall@7`、**balance_hard `done_fall@4`**（balance 更敏感）。
- **根本结论：HB terrain + balance 任务"摔倒即终止"，环境层面不提供"跌倒后恢复"的物理机会**。
  无论 passive 还是 active probe 都撞这堵墙——问题不在探针口径，在环境 termination。pole 略宽松
  （min_upright 能到 0.3–0.5、safe/rand 25%>scr 12%）但 settled=8 样本太小不可靠。

**fall-recovery 的最终定位（诚实）**：在 HB 上能验证的鲁棒性 = **fall-avoidance（更少摔倒终止）**，
**不是** "fall-recovery（跌倒后恢复）"。有效结论 = passive 的 safe 摔倒终止率最低（0–3% vs 6–25%），
跨 stair/slide/pole 一致，可作 behavioral-quality insight。"跌倒后恢复"需专门 push-recovery
benchmark（不早终止），HB 不具备，建议作 future work，不强行在 HB 上做。

---

## 3. 给 ChatGPT 的新问题

1. **3-seed 修正后**：源选择是稳健主因（wfix−rand +77.9，11/12，t=+3.08），horizon 项总体
   **中性**（safe−wfix −11.4，t=−0.46）、符号任务依赖（stair +105 有益，pole/crawl 有害）。
   据此把主方法定为 `reward_weighted_bootstrap`（weighted 源 + **默认短 horizon=25**），把
   horizon 降级为"次要超参、非核心"是否妥当？后续 abstain 加在最干净的 **wfix 基线**（短
   horizon weighted）上是否合理？
2. 既然 horizon 总体中性（非"有害"也非"有益"），是否仍值得在 window/balance（脆弱）上单独
   验证 safe-horizon 的保护价值，还是直接以"默认短 horizon"收口、不再单列 horizon 实验？
3. fall-recovery：active probe 已撞环境硬限制（HB terrain+balance 摔倒即终止，测不到恢复）。
   是否接受"在 HB 上 robustness = fall-avoidance（safe 摔倒终止率 0–3% 最低）"作为 behavioral-quality
   insight，把"跌倒后恢复"留 future work？还是值得引入专门 push-recovery 设定（改 termination）？
4. 下一步是否仍按原排序进入 abstain 实现（Step 2），还是先消化 wfix 的 horizon 发现？
