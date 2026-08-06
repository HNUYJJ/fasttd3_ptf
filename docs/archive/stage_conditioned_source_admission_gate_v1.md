# Stage-Conditioned Source Admission Gate v1

日期：2026-07-12  
状态：**Measurement/Scientific Stop；cabinet mandatory regression矛盾；closed-loop禁止实现**  
上位决策：[`paper_core_contribution_reconstruction_v2.md`](paper_core_contribution_reconstruction_v2.md)

## 1. Gate 目标

只检验 Stage-Conditioned Handoff Utility（SHU）的辨识力，不训练 learner、不优化 replay、不比较最终
return。核心问题：

> 在当前 student occupancy 上，source→student paired intervention 能否同时识别已有 positive、
> negative 与 late-stage-null source-target-stage cells？

若不能，source admission 不进入闭环，避免再次用大批100k runs验证一个不可测信号。

## 2. 固定 potential paths

每个 `(task, student checkpoint, source)` cell 从相同 simulator anchors 生成：

- `SS`：student prefix `h=25` + student follow-up `f=25`；
- `iS`：source full-action prefix `h=25` + 同一 student follow-up `f=25`。

控制项：

- 相同 reset/branch state、student exploration noise、episode termination语义；
- source/student prefix不同，follow-up student完全相同；
- 在全部 paths 收集完前不更新任何 learner；
- source/student trajectories只写 quarantine artifact；
- duplicate `SS` 必须逐元素一致；
- source action与student action必须有可检测 treatment distance。

新建state bank时，student exploration使用checkpoint冻结的 `std_min/std_max`，按
`(noise_scale_seed, anchor_id)`确定性采样每个anchor的noise scale；预计算formal bank则继续使用record中
已经冻结的scale。所有SS/iS分支都读取anchor record的同一scale并复用逐步noise tape；source prefix
使用冻结策略的确定性动作。不得把训练时 `num_envs` 维度的瞬时buffer任意映射到collector worker。

第一版默认使用 full-action source，避免同时检验 admission 与 EPS mask。唯一例外是 cabinet formal
regression cell：它严格复用既有 `run(legs_torso+arms)+student(hands)` candidate，只用于验证 SHU
实现与既有正式结果一致；不得把该结果外推成 full-action run 的标签。

## 3. 预注册 cells

不搜索 task/source。首轮使用已有证据最清楚的四类：

| cell | student stage | sources | 既有外部标签 | SHU预期 |
|---|---:|---|---|---|
| hurdle | scratch50k checkpoint | walk/run | run/walk probe与bootstrap均有正证据 | 至少一个 source eligible |
| crawl | scratch100k/final checkpoint | stand/walk/run | student已学会crawl，locomotion迁移全局为负 | 所有 source rejected |
| cabinet regression | formal scratch10k anchor | 既有run-composite | 已有同candidate的25+25 local intervention null/negative | reject at 10k |
| basketball diagnostic | 30k与100k scratch | stand/run/slide | 只有WFix混合源的负标签，无单源训练标签 | 只报告，不进入Go/Stop分类 |

checkpoint存在性与精确路径在实现前写入 machine-readable config；缺失 cell 不用相邻方法checkpoint替代，
而是标记 unavailable。不得为得到预期符号改 source、stage 或 horizon。

机器可读冻结配置：
[`stage_conditioned_source_admission_gate_v1.yaml`](../configs/experiments/stage_conditioned_source_admission_gate_v1.yaml)。

其中 hurdle 使用现存 `h1hand_hurdle_scratch50k` checkpoint，crawl 使用3个现存scratch final中的
预注册seed 1；正式路径与hash写入config。两者只有在正确seeded branch-state anchors可构造时才进入
正式gate。basketball已有密集scratch checkpoints但缺少单源下游ground truth，因此只检查SHU是否
暴露“即时姿态/return与handoff分离”，不拿未知答案凑分类准确率；cabinet复用已验证formal anchor。

## 4. Estimands

每个 anchor `j`：

- `d^D_{ij}=G_prefix(iS_j)−G_prefix(SS_j)`；
- `d^H_{ij}=G_follow(iS_j)−G_follow(SS_j)`；
- `d^K_{ij}=Risk(iS_j)−Risk(SS_j)`。

`G` 为未归一化 target reward sum，并同时报告 per-active-step reward与active exposure；`Risk` 是
`termination/truncation OR fall-proxy`，其中通用fall-proxy固定为任一active step
`head_height<1.2 OR torso_upright<0.5`。无效anchor进入state-bank rejection audit且不得凑入512个
valid anchors。cabinet的door fraction/subtask/reach/contact等hard progress只作不参与admission的审计列。

为跨task显示，effect使用同cell independent-noise `student→student` null的robust scale：
`scale=max(IQR/1.349, 1.4826*MAD, 1e-6)`；原始量纲值必须并列保留。confidence unit是anchor，
不把time step当独立样本。每个mandatory cell固定512个valid anchors；若候选池无法得到512个，
Measurement Stop，不通过减少样本数继续。

## 5. Admission rule

对 standardized paired effects做 one-sided 95% anchor-bootstrap bound。source eligible 当且仅当：

1. `LCB(H_i) >= -δ_H`；
2. `LCB(D_i) > δ_D` 或 `LCB(H_i) > δ_H+`；
3. `UCB(K_i) <= δ_K`。

practical thresholds由以下数据在看正式cell结果前冻结：

- duplicate `SS` noise floor；
- student-vs-student independent-noise paired null；
- target reward robust scale。

具体地，standardized null paired effects的双侧95%绝对分位定义 `δ_D/δ_H`，risk null的95%上分位
定义 `δ_K`；若不同mandatory task的标准化null阈值相差超过2倍，说明统一scale失败，Measurement Stop，
不得改成per-task手调阈值。

不允许以最大化四cell分类准确率选择阈值。若null噪声无法给出稳定阈值，Engineering/Measurement Stop。

## 6. Gate 裁决

### Engineering Go

- branch state与首观测一致；
- follow-up student/noise一致；
- duplicate exact；
- provenance完整；
- quarantine artifact未写入任何main replay；
- 每cell有效anchors不少于预注册下限。

### Scientific Go

同时满足：

- hurdle：run或walk至少一个 eligible；
- crawl：stand/walk/run全部 rejected；
- cabinet@10k：run rejected；
- basketball diagnostic不作为分类分母；若source通过，必须明确其prefix/handoff/risk依据，不能仅因
  posture或episode length正向就宣称selector正确；
- source排序/准入不能由fall或episode length单一指标解释；
- no threshold/task/source/horizon修改。

如果只满足部分：**Stop closed-loop**。不得把四cell删成只剩正例。

## 7. Go 后的最小闭环，而非本 gate 内容

只有 Scientific Go 后才实现：

1. student作为reference arm，无eligible source时 `p(student)=1`；
2. source probe quarantine；
3. admitted-source bootstrap；
4. revoked source transition从active replay采样中移除；
5. actor/critic共享相同source-stratified sampling distribution；
6. positive/negative/null各一个单 learner-seed 100k feasibility。

MCG/EPS、PER、TD-priority、更多source与多seed均在该闭环gate之后。

## 8. 明确的失败解释

- hurdle也拒绝：SHU漏掉真正的delayed data value，不能做selector；
- crawl被接受：student-follow-up仍无法捕捉长期技能干扰；
- basketball若出现posture正但handoff负：应由non-harm gate拒绝；若handoff也正，只能记为未知单源
  diagnostic，不能用WFix混合源负标签把它事后判错；
- cabinet@10k通过：与已验证formal intervention矛盾，优先查实现/metric；
- 分类依赖手工task metric：通用source admission主张失败。

任何一项都比继续调温度或跑大表更有信息量。

## 9. 实现与工程验证（2026-07-12）

已实现以下只读路径：

- `official_fasttd3_ptf/source_admission.py`：schema、duplicate/provenance校验、robust scale、anchor
  bootstrap、non-compensatory admission与审计统计；
- `probe_stage_conditioned_source_admission.py`：`preflight/smoke/collect/analyze/report`五阶段CLI，
  state bank、paired potential paths、quarantine artifact、hash与aggregate Go/Stop；
- `hb_branch_state.py`：所有HumanoidBench任务的通用姿态/控制诊断，同时保留cabinet专用机制诊断；
- `test_source_admission.py`与`test_hb_branch_state.py`：9项聚焦测试全部通过。

真实checkpoint smoke已覆盖cabinet anchor、hurdle scratch50k与crawl scratch100k；均满足branch restore、
duplicate exact、非零treatment distance和report schema。smoke固定只有1个anchor、不落盘、不得用于判断
source eligible或Scientific Go。正式命令在服务器负载允许时逐cell执行：

```bash
conda run -n FastTD3 python scripts/probe_stage_conditioned_source_admission.py collect \
  --cell <frozen-cell-id> --device <cuda-device>
conda run -n FastTD3 python scripts/probe_stage_conditioned_source_admission.py analyze \
  --cell <frozen-cell-id>
conda run -n FastTD3 python scripts/probe_stage_conditioned_source_admission.py report
```

当前没有修改learner、optimizer、main replay或closed-loop source selector；这些仍受第7节Scientific Go
约束。

## 10. 正式裁决：cabinet mandatory regression触发提前Stop（2026-07-12）

只运行了足以形成致命反例的cabinet mandatory cell；hurdle/crawl正式cell未运行，避免在Scientific Go
已不可能后继续消耗算力。最终artifact：

- `artifacts/source_admission_gate_v1/cabinet_s1_step10000/quarantine.pt`；
- `artifacts/source_admission_gate_v1/cabinet_s1_step10000/analysis.json`；
- `artifacts/source_admission_gate_v1/gate_report.json`。

工程约束全部成立：512/512 anchors、same-run duplicate bit exact、非零treatment、quarantine-only、
learner/main replay零写入；accepted IDs、horizon、source groups、path seed与旧formal trajectory bank一致，
初始observation exact，step-0 action误差小于固定`1e-5` float32容差。跨运行完整轨迹不要求bit exact，
因为约`1e-6`的GPU动作差会被MuJoCo混沌放大；该差异被保留为audit。

正式结果：run-composite被判为`eligible=true`，三项检查均通过：

- direct raw mean `+0.0253`；
- student follow-up raw mean `+0.1728`，其one-sided LCB为正；
- combined risk差 `−0.0156`，没有触发risk veto；
- source follow-up mean door fraction `0.05087`，student为`0.04839`，局部hard-progress audit也未给出
  足以否决source的方向。

但旧formal 2×2在相同source/stage/horizon/operator下给出完整downstream local intervention
`T=-0.04836 max-door`、`-0.02288 normalized AUC`，要求该cell被拒绝。因此
`cabinet_regression_all_rejected=false`，aggregate写入
`stop_reason=mandatory_label_contradiction`与`route=STOP_CLOSED_LOOP`。

此外independent student-null的direct/handoff/risk robust scale全部落到`1e-6` floor，统一标准化尺度
失效。结论不是“run有可迁移skill”，而是：当前SHU测得的短期behavior/handoff utility与source
transition经过off-policy update后的data utility不是同一对象。SHU v1不得作为自动source admission、
replay admission或论文核心贡献；不调阈值、不换task、不补跑hurdle/crawl。
