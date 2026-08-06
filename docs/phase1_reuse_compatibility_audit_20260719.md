# Phase-1 基线复用兼容性审计（零训练）

> 日期：2026-07-19
> 依据：规格 v0.3 §7（十七次复核放行标准：逐任务 REUSE_PASS/REUSE_FAIL）。
> 本审计未运行任何训练；全部证据=git 对象、历史 meta/artifacts、日志。

## 0. 结论摘要

| 任务 | retention 臂 | 判定 | 关键证据 |
|---|---|---|---|
| basketball | `adaptive_admission_v1/basketball_static_s1-3` | **REUSE_PASS（强）** | 实现被**逐位锚定**到 git 树（a5cec9d），到 HEAD 的全部差异逐段定性中性 |
| truck | `admission_handoff_v1/truck_admission_h4_fix_s1-3` | **REUSE_FAIL** | replay 通道逐位等价，但 train_ptf/mcg/admission_control 的 7-13 内容不可重构 → 0–30k 行为与学习路径可比性无法验证（十八次复核裁定，我接受；限定 REUSE 撤回，理由见 §4） |

**预算档位定案（十八次复核）：9 × 100k**——basketball 复用 retention ×3
+ 新跑 hard-exit ×3；truck 两臂全新跑（retention ×3 + hard-exit ×3）。
约 13–14 GPU-hours。

**hard-exit 臂实现决策（§7 第 6 项）**：历史实现**没有 `AdmissionSchedule`**
（b183f40 树无此类；它在 7-14 адaptive 开发中加入）——"用历史冻结实现启动
hard-exit 臂"不可行，**必须用当前 HEAD**；HEAD 与两个历史锚点的语义等价
即本审计的核心内容。

## 1. 实现锚定（git_head 记录失效的发现与替代锚定法）

两族 meta 的 `git_head` 均为 `b183f40`——**该 commit 是 2026-06-15 的**
（wfix banks ablation）。6-15 → 7-16 整月的开发（admission core/handoff/
adaptive/schedule）全部发生在未提交的工作树上，git_head 无锚定价值。
真实锚点是 meta 的 `implementation_sha256`：

- **计算定义**（launcher 脚本，a5cec9d 树内可考）：
  `sha256( sha256sum(admission_control.py, ptf_replay.py, train_ptf.py,
  mcg.py, 两个 shell 脚本) )`。
- **basketball 臂（7-14）**：用 a5cec9d 快照树六文件按同一公式复算 →
  `4318b8b7…` **逐位命中** meta 值 → basketball 臂实现 ≡ a5cec9d 内容
  （完整可考）。
- **truck 臂（7-13）**：`frozen_implementation.sha256` 逐文件清单在案
  （admission_control=ce57b40 / ptf_replay=**ee96b46** / train_ptf=16cfa74 /
  mcg=2e59cbe）。其中 **ptf_replay=ee96b46 与 a5cec9d 逐位相同**；其余三
  文件为 7-13 工作树状态，内容不可从 git 重构（只有 hash）。
- 臂内一致性：每臂 3 seeds 的 implementation_sha256 唯一 ✓。
- bank 完整性：两 bank yaml 当前 sha256 与历史 meta 逐位一致
  （a570bae… / 02ffb1ff…）✓。

## 2. a5cec9d → HEAD 的语义定性（basketball 锚点 → 当前实现，3 commits）

1. **04f4b0d（清理）**：admission_control/ptf_replay 变更**纯注释**；
   mcg=删除死线 `chain` warmup 模式（本实验 warmup_mode=
   admission_bootstrap，不触发）；train_ptf=删除 entity encoder 全套+
   chain+ZNativeSourcePolicy（历史 runs 均未启用，launcher yaml cells 无
   entity 配置）。**活跃路径零改动**（diff 逐段+排除法核实）。
2. **a49e423（P0 anchor-resume）**：新 CLI 全部默认不激活；resume 块有
   守卫，非 resume 不进入；`--ptf-eval-checkpoint-steps` 新保存路径本
   实验不使用。
3. **9570a5b（P0 修复）**：41 键配置断言/segment 续接=resume-only；
   ptf_replay +17 行=新增只读访问器（不调用即中性）。
4. **关键零触及项**：`save_interval` 保存逻辑、`evaluate()` 函数、
   eval_envs 构建、np/torch 播种——`a5cec9d..HEAD` diff 中**无一行**命中
   （曾担心的 checkpoint off-by-one 修复只存在于本实验不使用的
   eval_checkpoint_steps 新路径；save_interval 旧机制未动）。

结论：HEAD 对"非 resume、admission-all/schedule、100k 普通 run"与
a5cec9d 语义等价 → **与 basketball 臂实现语义等价**。

## 3. eval 协议与日志完整性

- 历史 [eval] 行由 train_ptf 内置 `evaluate()` 产生（eval_interval=5000，
  10k–95k 19 点网格）；该函数与播种自 a5cec9d 至 HEAD 零改动（§2.4），
  7-13→7-14 亦无 evaluate 改动记录需求（两臂日志同网格自证协议一致）。
- **35k–80k 完整性：6/6 条复用 log 全部 10/10 点** ✓（truck s1-3 与
  basketball s1-3 各 19 点全网格）。

## 4. truck REUSE_FAIL 定案（十八次复核裁定，我接受）

- **可直接验证的**：replay 采样通道 7-13 版=ee96b46 → a5cec9d 逐位相同
  → HEAD 差异=中性访问器+注释。replay 通道逐位等价 ✓。
- **决定性缺口（改判理由）**：本实验声称"唯一差异是 30k 后的 R(t)"，
  这要求两臂 0–30k 的行为分布与学习路径可比。但 **30k 时的 learner
  state 由 train_ptf/mcg/admission_control 共同决定**，而这三个文件的
  7-13 工作树内容不可从 git 重构（只有 hash）。只证明 replay 代码相同
  **不能**证明以下任一项与当前实现等价：0–30k source/student 行为分布、
  MCG 锁存与动作组装、candidate 概率与执行计数、RNG 消耗、evaluator/
  horizon/播种、learner 更新路径。
- **34 项认证的局限**：它们只证明 run 完成、配置字段正确、30k authority
  释放、physical handoff 生效、checkpoint/日志存在——**不证明**上述任一
  行为/学习等价项。故不足以支撑正式配对裁决。
- **结论**：truck 历史 retention 只作历史背景，**不进入正式配对裁决**；
  truck 两臂全部用当前 HEAD + schedule 路径新跑（§见规格 v0.4）。
  basketball 复用不受影响。

## 5. 参数逐项核对

warmup=30000、admission_mode=all、handoff=physical_after_authority、
recency=0.0、uniform_mix=1.0、priority_alpha=0.0、num_envs=128、
batch=32768、buffer=51200、num_updates=2、eval_interval=5000、seeds
{1,2,3}、100k、MCG groups=[legs_torso, arms]、student_logit（truck bank
族值/basketball=3.5892126423877646）——两族 launcher yaml 与 meta 逐项
一致；hard-exit 臂沿用全部值，唯一差异=AdmissionSchedule 的 (30000,
exact abstention) 决策（treatment 本身）。

## 6. 工程教训（E17 候选）

git_head 在长期脏树下无锚定价值；本次靠 launcher 预记的
implementation_sha256（含计算定义可考）才完成锚定。今后每个正式 run 的
meta 必记：per-file implementation SHA 清单 + `base_git_head + dirty
状态`（承接十三次复核的 provenance 三元组建议）。

## 8. scratch 对照兼容性审计（二十次复核批准，2026-07-19）

scratch（run card §7.4 的层-5 限定标签外部参照）为更早的 b2/br 族
（2026-07-04/05）：

- basketball：`b2_20260705T153732Z/b2_basketball_scr_s1` +
  `b2s_20260705T224905Z/bs..._s2,s3`；
- truck：`br_20260704T015912Z/br_truck_scr_s1` +
  `brseed_20260704T135105Z/bs_truck_scr_s2,s3`。

### 8.1 更正：我的"无任何 provenance"判断是错的

初版 §8 判 scratch "无任何 provenance（无 meta/SHA/认证）"——**错误**。
我只查了 `logs/train/` 目录，**没查 `wandb/`**（494 个 run 档案）。二十一
次复核指出后核实：六条 scratch run 的 W&B 本地档案**完整存在**，含
`config.yaml`、`wandb-metadata.json`（完整 CLI + git commit + 环境）、
**历史入口代码副本** `files/code/.../train_ptf.py`、`diff.patch`、
`output.log`、`requirements.txt`。

（这是我第二次"只查一处即断言不存在"——前一次是 `execution_counts_at_apply`
只查 `policy_events` 未查 `decision_history`。教训见 ISSUES E18。）

### 8.2 核实证据（脚本固化：`scripts/analysis/p1_freeze_delta.py`）

| 项 | 结果 |
|---|---|
| 六份历史入口 `train_ptf.py` 快照 SHA256 | **全部相同** `6e3d228d79908da5356aa2fdcb5ef7d811fb797778215971e45b8a1761190270` |
| git base commit | 全部 `b183f40bcfe6…` |
| CLI 纯净性 | 六条全部 `pure_scratch=True`（无 `--ptf*` / bank / admission / mcg） |
| W&B `output.log` vs `logs/train` 曲线 | 六条 35k–80k nAUC **逐条完全相同**（同一次运行的两份记录，provenance 绑定成立） |
| `diff.patch` | 五条相同（`623c4196…`）；**truck s1 不同**（`79a6ca2d…`）——差异经核实**仅在 `scripts/probe_transfer_map_v2.py`**（离线探针脚本，不参与训练），文件列表与入口代码均无差异 |
| eval 协议 | 5k 网格，35k–80k 10/10 完整 |

### 8.3 分层裁定（二十一次复核方案，采纳）

1. **`CAUSAL_COMPARATOR_REUSE_FAIL`（仍成立）**：scratch @ `b183f40` 与
   hard-exit @ HEAD 非同批实现（HEAD 另有 target-only RNG 恢复等变化），
   不能承担严格配对因果标签 → run card §7.4 的 scratch 限定标签
   **保持描述性**，不进正式判序；
2. **`METRIC_SCALE_REUSE_PASS`（成立）**：历史 evaluator、任务、return
   单位、5k 网格、运行参数与入口代码均有充分证据 → scratch 跨 seed SD
   **可作为预先固定的外部实用效应尺度**。

**δ 的定位（据此改写）**：δ_task 是从历史、同任务、同 return 口径的
scratch 曲线得到的 **externally anchored SESOI / practical margin**；
**不是**当前实现的数值噪声地板，**也不是**新实验的方差估计。核心
estimand Δ_exit = hard-exit − retention 为配对差，不依赖 scratch，主
判序（层 3/4）不受层-5 降级影响。

## 9. 下一步

1. 本审计（含 §8 scratch 结论）随 run card v0.3 交 ChatGPT 复核；
2. 通过后 δ 冻结脚本 → PI 批准；
3. 此前不启动训练。
