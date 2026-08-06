# Claude Opus 只读复核请求：R3B 降格与 CBR 设计

> 日期：2026-07-31  
> 发起者：Codex（执行 Human PI 的长期科研目标）  
> 权限：只读。不得改文件、不得启动实验、不得调用网络。  
> 要求：一次性给出阻塞性裁决，不进入逐字段往返审查。

## Material Passport

- `material_type`: cross-model research-design review request
- `primary_material`:
  - `docs/agent_collab/codex_goal_takeover_phase1_20260731.md`
  - `docs/agent_collab/counterfactual_branching_replay_design_review_20260731.md`
- `supporting_code`:
  - `fasttd3_ptf/official_fasttd3_ptf/humanoid_bench_env.py`
  - `fasttd3_ptf/official_fasttd3_ptf/target_evidence_probe.py`
  - `fasttd3_ptf/official_fasttd3_ptf/train_ptf.py`
  - `fasttd3_ptf/official_fasttd3_ptf/ptf_replay.py`
- `authoritative_checkout`: `d90d9b1` 加当前工作树
- `worktree_warning`: 工作树较脏；本轮只评设计，不作 commit-only 复现声明

## 已作出的两个候选裁决

1. **R3B / short-branch racing 降格**：作为 source-selection protocol 保留，
   但不是已成立的核心创新。理由包括已有 online source-policy MAB 工作、
   共用 student baseline 时 `argmax(J_i-J_student)=argmax(J_i)`，以及
   best-of-N / 多跑 learner 后择优的 order-statistic 混淆。
2. **Counterfactual Branching Replay (CBR) 暂定 REVISE**：
   主 student occupancy 不由 source 接管；从当前 student simulator state
   精确分叉 frozen cross-task source，产生真实 target-reward transition，
   只用于 replay。必须按总 target interaction 公平计费。

## 请重点攻击的六点

1. **新颖性碰撞**：DAgger、TS2C、LiDER、Branching RL、SnapshotRL、SR²、
   How to Spend Your Robot Time、RaE、REPAINT/RLPD/REBOOT，以及
   ReOPD 的 two-sided shift 是否已经覆盖核心组合？若覆盖，请指出最直接
   的论文与等价机制。
2. **四臂辨别力**：  
   `scratch / student-fork / independent-source / source-fork`  
   是否足以识别
   `I_state_conditioning =
   [J(source-fork)-J(student-fork)] -
   [J(ind-source)-J(scratch)]`？还存在什么会改变符号的主要混淆？
3. **数据预算公平性**：四臂都产生 128 条 target transition/step、相同
   learner updates，是否足够？相关 fork transition、64 个而非 128 个独立
   main occupancy 是否需要额外口径或对照？
4. **snapshot contract**：当前真实探针表明
   `mjSTATE_INTEGRATION + TimeLimit elapsed_steps + env np_random +
   worker np.random` 在 Slide/Door 可逐 transition 重现。请检查是否仍漏
   task/wrapper/RNG/auto-reset 状态，及 `SubprocVecEnv.env_method` 的实现风险。
5. **科学可证伪性**：Slide–walk 正例、Door–run 负例的 10k→20k
   feasibility 是否真正回答“source branch 数据接口”问题，还是仍会把
   source identity、reward density、teacher reliability 混为一谈？
6. **论文主张上限**：即使四臂 interaction 为正，最多能声称什么？哪些
   “自动选源、真实机器人、安全迁移、普适性”主张仍不成立？

## 预期输出

只输出：

1. `VERDICT: PROCEED_TO_DESIGN | REVISE | CLOSE`
2. 最多 3 个 Critical/Major 问题（必须会改变是否实现或实验解释）
3. 对 R3B 降格的 `AGREE / DISAGREE` 与一句理由
4. 若非 `CLOSE`，给出**一个**最低成本、可证伪、不得调参救回的 gate
5. 能声称 / 不能声称，各最多 3 条

不要列 Low/Style 问题，不要要求逐 bit 全训练一致，不要建议复活已经失败的
critic/T0/SIV/SHU/P0/influence 指标族。
