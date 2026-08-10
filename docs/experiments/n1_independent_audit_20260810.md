# N1 独立审计：从理论到实现的逐层核实

日期：2026-08-10 · 审计对象：Codex 提交的 `930021a` / `de37d57` / `aafec93` / `10e77ef`
性质：**审计报告**。不改动 N1 已冻结的 `NDT_NOT_SUPPORTED`，不启动新机制。

---

## 0. 结论摘要

**N1 的裁决站得住，工程实现是本项目迄今最干净的一次。** 审计发现的问题都不在
执行层，而在两处：一处是我自己此前的理论错误（已被外部 review 纠正），
另一处是一个**从未被记录的 source mixture 事实**，它限定了全部 truck 结论的范围。

| 层 | 结论 |
|---|---|
| 理论 | 我的 `A=0.545` 是混用时间基准，外部修正正确；已独立复核 |
| 流程 | 三段式成立；`aafec93` 的"放宽"发生在无性能数据时，是合法工程修复 |
| 实现 | 四臂共享 anchor、LP 的 arms 保护真实、剂量逐 seed 匹配，全部核实通过 |
| 设计 | `H_REC` 的失败在设计上可预期：LP 保护了弱通路，强通路仍归 source |
| **范围** | **唯一的静止源 `stand` 在 10k→20k 只被执行 25 次（0.004%）** |

---

## 1. 理论：`A=0.545` 的撤回（我的错误）

我此前把**累计** critic source share `q̄_S=0.1534` 与**终点** buffer 占比
`ρ_S(u)=0.25` 塞进同一个 odds ratio，得到 `A=0.545`，并据此声称
"physical 欠采样 source，需要造一个 A=1 补偿臂"。

**这是错的，两个量不是同一个时间口径。** 正确表述：

- physical 每一时刻在 allowed slots 上均匀采样，故瞬时 `q_t = ρ_t`，
  `A_inst ≡ 1` 严格成立；
- `q̄_S = m[1 − (H/u)ln(1+u/H)] = 0.1534` 是 late-entry cohort 的
  **寿命效应**（source 后来才出现，早期不存在自然没被采过），不是欠采样；
- 要求累计 `q̄_S = ρ_S(u)` 等于要求给后进 transition **补偿它过去不存在的时间**，
  那是 age-compensated prioritization，是新算法，不是缺失的对照组。

我据此提议造 A=1 臂，属于典型的机制扩张（CLAUDE.md §8.2b 反面）。
N1 预注册已明文禁止该臂，处置正确。

---

## 2. 流程：三段式与"放宽审计"的时机

`aafec93`（19:56）夹在冻结（18:27）与结果（21:53）之间，形式上是"冻结后改判据"，
须逐条核实：

- 改动内容：仅放宽 **S 臂**的 `sampling_phase` 命名检查
  （`authority_quota` → 允许 `authority_quota` 或 `physical_allowed`），
  实质判据（`replay_physical=false` 且 source execution/replay counts 严格为零）**未动**。
- 时机核实：19:56 时 s4/s5 训练已完成，但**评估结果最早 21:35 才存在**；
  且训练配置 `eval_interval=0`，训练日志不含任何 return。
  **改动时不可能看到任何性能数字。**
- 结论：合法的工程修复，不构成 outcome-contingent gate switching（§8.7）。

`10e77ef` 的 `git show --stat` 只有 `.json` 与 `.md`，符合 §4.1。

---

## 3. 实现：逐项核实通过

1. **四臂共享 anchor**（`run_n1_..._v1.sh:64-92`）：`branch_common` 含
   `PTF_ANCHOR_RESUME` 与 `PTF_RESUME_NOISE_SEED`，S/FF/FP/LP **全部**引用它。
   我在 PARE Gate A 犯的"只有一条臂重启"（§8.2）在此已被修正。
2. **LP 的 arms 保护是真的**，两条独立证据：
   - 代码（`mcg.py:719-726`）：`current[:, disabled] = -1`，而合成时
     `m = sel >= 0` 对该组恒 False，故 `actions` 保持 `a_student` 原值；
   - provenance：LP 的 `rho_arms` 五个 seed **严格为 0.0000**，FF/FP 则等于 `rho_legs`。
3. **剂量匹配**：同一 seed 内 FF/FP/LP 的 `execution_source_share` 逐位相同
   （如 s4 三臂均为 0.4971），且 `0.4971 × (10k/20k) = 0.2486 = rho_legs` 自洽。
4. **replay 语义**：FF 累计 critic source share ≈ 0.495（fixed quota），
   FP/LP ≈ 0.152–0.155（理论 `0.5(1−ln2)=0.1534`）。
5. **evaluator**：预注册只要求 source-free / panel128 / deterministic，
   未指定版本；`p0_evaluator.py` 结构上不构造 bank/option/admission，
   且 truck 的 `terminated` 语义确为"成功"（`truck.py:206-210`），不是摔倒。

---

## 4. 设计：`H_REC` 的失败在设计上可预期

### 4.1 已冻结的探针（判据先于运行）

`n1_discriminability_probe_prereg_20260810.md` → 结果 `ARMS_PATHWAY_PARTIAL`。
**我的主判据未通过**：我预期 arms 完全无关（δ<0.05），实测 `δ_rand_arms=0.0550`。
故**收回**"H_A 的否证纯属平凡结果"这一说法；arms 通路弱但非零。

### 4.2 独立于该判据的描述性事实

| 扰动（10 checkpoint × 16 ep） | δ return |
|---|---|
| 随机化 `legs_torso` | **0.4089** |
| 随机化 `arms` | 0.0550 |
| `arms` 置零 | 0.0447 |

**`legs_torso` 的因果影响是 `arms` 的 7.4 倍。**

配合直接测量的 reward 分量（`tasks.py:68` 把 `reward_info` 并入 info）：

| 臂 | return | **upright** | package | path_len |
|---|---|---|---|---|
| S | 1042.6 | **0.8776** | 0.189 | 2.39 |
| FP | 939.8 | 0.7771 | 0.207 | 4.46 |
| LP | 913.8 | 0.7691 | 0.189 | 4.22 |
| FF | 927.8 | **0.7525** | 0.227 | 7.73 |

`corr(upright, return) = **+0.938**`（t=11.47, n=20）。0→20k 期间
`reward = upright × (1 + reward_robot_package_truck)`，而 upright 由躯干姿态
决定——即 `legs_torso`。

**于是 `H_REC` 的失败是设计的推论**：LP 把弱通路（arms）还给 student，
却把强通路（`legs_torso`，7.4 倍）继续交给 source。这条推论只依赖上表的比值，
不依赖已未通过的主判据。

### 4.3 必须自我修正的一处

我先前用 128-ep 面板算出 `corr(位移, return) = −0.545` 并称"强烈反向"。
在 16-ep 测量面板上该相关只有 **−0.332（t=−1.49，不显著）**。
**位移本身不是稳健的解释变量，upright 才是。** 两个面板的 FF/LP return 序
也相反，故本测量只作机制描述，不得用于重裁 N1。

---

## 5. 范围：一个从未被记录的 source mixture 事实

从 `admission_audit.execution_counts`（N1 的 FP/s4，其余 seed 同量级）：

| 源 | bootstrap weight | candidate_mass | 10k→20k 执行次数 | 占 source 执行 |
|---|---|---|---|---|
| **stand** | 5.200 | 6.07e-05 | **25** | **0.004%** |
| walk | 12.816 | 0.1232 | 156,375 | 24.6% |
| run | 12.613 | 0.1006 | 126,500 | 19.9% |
| **hurdle** | 13.623 | 0.2761 | 353,325 | **55.5%** |

已验证 `candidate_masses = softmax(bootstrap_weight) × 0.5`（四项比值精确 1.0000）。
`stand` 与 `hurdle` 的 weight 只差 **8.42**，经 `exp` 放大成 **4550 倍**的执行概率差。

weight 是 per-target 计算的（maze bank 为 4.583/11.262/10.023/16.414，不同），
所以这**不是** bank 搬错，而是：**在 truck 上，probe 打分经 softmax 把唯一的
静止源压到了 0.004%**。

而 truck 0→20k 的高分策略恰恰是"站稳少动"——scratch 的 upright 最高（0.8776）、
path_length 最短（2.39）、return 最高。

### 这意味着什么

- **不推翻 N1 的裁决。** 四臂共用同一 mixture，内部对照有效。
- **但限定了全部 truck 结论的范围**：从 Gate A → T2 → T3 → T4-R → N1，
  所有"truck 负迁移"证据都是在"跨栏源主导 55.5% + 站立源近乎为零"的
  特定 mixture 下取得的。诚实表述应带上这个条件。
- 这同时是又一个**"行为代理 ≠ 学习效用"**的例证：probe 分数把 hurdle 排在最前，
  而 hurdle 在 truck 早期恰是移动最剧烈的源。与 door gate 的既有发现
  （run 行为 +58% 却 harmful）方向一致。

### 未做与为什么

一个自然的追问是"若把 mass 给 `stand`，负迁移是否消失"。**本次未做**，因为：
需要 source obs_adapter 链路的新代码（`hb_robot_qpos_qvel`，111→151），
而 adapter mismatch 是 silent corruption；且是否深入涉及"选源"方向的边界，
应由 PI 裁量，不由审计自行启动。

---

## 6. 审计未发现的问题

以下逐项查过且**未**发现问题，记录以免重复劳动：
配置冻结矩阵（18 项逐 cell 校验）、anchor 与 noise seed 一致性、
provenance 两组的写入完整性、evaluator 的 source-free 结构性保证、
`success_count` 的 truck 语义、结果 commit 的代码洁净度、
判决脚本的 fail-closed 行为（缺数据 → INCOMPLETE 且非零退出）。
