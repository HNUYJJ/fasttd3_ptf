# 仓库结构地图（REPO MAP）

> 2026-07-16 大整理后的权威地图。本次整理删除了全部已弃用路线代码
> （entity encoder / ED-SF / package 专项 / SIV / SHU 闭包），文档整合为三个主文档
> +归档。删除前全量快照 = git commit `a5cec9d`，需要历史代码时从该快照恢复。
> （更早的历史：2026-06-10 整理基线 `40b04cc`，含已删除的 my_fasttd3_ptf/。）

## 顶层结构

```
fasttd3_ptf/            # 核心 Python 包（见下）
configs/                # source bank yaml + 预注册实验配置
scripts/                # 训练/分析/审计脚本（全部属活跃链）
tests/                  # pytest 单测（91 个，全部对应活跃模块）
docs/                   # 三个主文档 + REPO_MAP + agent_collab/ + archive/ + reports/
papers/                 # 参考论文 PDF
requirements/           # 依赖
checkpoints/            # 源策略 checkpoint + manifest（*.pt 不进 git；勿删，bank 仍引用）
models/ logs/ wandb/    # 训练产物（不进 git）
artifacts/              # 实验审计产物（进 git：json/md 审计证据链）
reference_source_code/  # PTF 原论文代码等参考（humanoid-bench 副本已删，official_code 即备份）
```

## fasttd3_ptf/ 包（单一活跃主线）

```
official_fasttd3_ptf/   # 训练入口层
├── train_ptf.py        # 唯一训练入口（PTF/RBO/admission 全机制接线）
├── ptf_replay.py       # replay wrapper：provenance、admission 配额、authority handoff
├── admission_control.py# 准入快照/调度/adaptive 撤销状态机（纯 CPU、零 RNG）
├── rng_isolation.py    # RNG 隔离（exact abstention 因果证据的机制基础）
├── humanoid_bench_env.py # HB 向量环境 + 双 RNG 正确播种
├── anchor_io.py        # anchor bundle 快照（learner+replay+rng；paired probe 基础设施）
├── source_admission.py # quarantine bank 结构校验（审计链用）
└── paths.py            # official_code 的 sys.path 接线

ptf/                    # PTF 核心机制层
├── mcg.py              # 行为调度核心：ModularGating + McgBehaviorController
│                       #   warmup_mode ∈ {random, safe_bootstrap(=静态RBO),
│                       #   online_bootstrap(=student-as-arm), admission_bootstrap(=主路径)}
│                       #   + AdmissionSegmentTracker（adaptive 撤销的 segment 结算）
├── option_module.py    # PTF option-value + β termination 网络
├── option_selector.py  # call-and-return option 选择
├── option_update.py    # U-value / termination loss
├── distillation.py     # masked action distillation loss
├── compatibility.py    # 高斯动作兼容度（PTF ξ 项）
├── source_policy.py    # SourcePolicy（旧格式 ckpt 加载 + adapter + 归一化复原）
├── source_bank.py      # SourcePolicyBank（act_all/act_selected + null option）
├── adapters.py         # 显式 obs/action 适配器 + action mask（禁隐式截断）
├── action_schema.py    # h1hand 61 维动作分组（legs/torso/arms/hands）
└── legacy_actors.py    # 旧格式源 ckpt 兼容层（checkpoints/sources/ 仍被 bank 用，勿删）

source_bank/            # bank 工具链：exporter(ckpt→manifest) / builder(→yaml) / manifest
utils/                  # checkpoint / normalization / schedules（均被主链引用）
config.py               # YAML 配置加载
official_code/          # 上游 FastTD3 + humanoid-bench 快照（保持 source-compatible，勿放项目逻辑）
```

依赖方向单向：`official_fasttd3_ptf/ → ptf/ → utils/ + config.py`；
`source_bank/` 由 shell 工具链经 `python -m` 调用。没有反向依赖。

## scripts/（按职能）

| 职能 | 脚本 |
|---|---|
| 训练入口 | `official_fasttd3_train_target_ptf.sh`（枢纽，全部实验 launcher 包裹它）、`official_fasttd3_train_target_scratch.sh`、`official_fasttd3_train_h1hand_sources.sh` |
| 源导出 | `official_fasttd3_export_h1hand_sources.sh`、`export_source_policy.sh` |
| bank 生成 | `build_{expanded,safe_bootstrap,std9,bigsrc,stability_audit}_banks.py`（与 configs/source_banks/*.yaml 一一对应，可复现性来源） |
| RBO probe | `probe_transfer_map_v2.py`（T⁰ 选源权重来源）、`probe_hb_task_layouts.py`（其数据上游）、`probe_{active_recovery,fall_recovery,hurdle_to_stair}.py` |
| admission 审计链 | `run/complete/finalize/orchestrate_admission_*.sh`、`analyze/audit/adjudicate/verify/backfill_admission_*.py`（frozen-SHA256 整体，勿单删） |
| 多任务分析 | `analyze_{terrain,wfix,onlineb,breadth_batch2_local,warmup_source_dose}.py`、`aggregate_multitask.py`、`analyze_tcritic_offline.py`（T^critic 线） |
| 公共库 | `probe_lib.py`（从已删 probe 抽出的 make_env_fn/load_student 等）、`stability_deconfounded_audit.py`、`task_progress_audit.py`、`research_pair.py`（协作流程） |

## configs/

- `source_banks/*.yaml` — 活跃 bank（loco/big/hurdle4/std9 家族 + wfix 变体 + `audit/` 单源审计 bank + `official/`）；每个 yaml 头部注释记录生成脚本
- `experiments/*.yaml|json` — 预注册冻结配置（admission_core/handoff/adaptive、SIV/SHU gate 等历史预注册**保留**，审计链一部分）

## docs/

- [`RESEARCH_ROADMAP.md`](RESEARCH_ROADMAP.md) — 科研路线（时间线 + 六组件坐标 + 待决方向）
- [`EXPERIMENT_LOG.md`](EXPERIMENT_LOG.md) — 实验结果及分析（机制栈 + 代号字典 + 全实验总表）
- [`ISSUES_AND_LESSONS.md`](ISSUES_AND_LESSONS.md) — 问题记录（工程纪律 + 方法论教训 + checklist）
- `agent_collab/` — Claude/ChatGPT 协作记录（追加式轮次，活跃）
- `archive/` — 全部历史文档（[索引](archive/README.md)；旧引用 `docs/X.md` → `docs/archive/X.md`）
- `reports/` — PI 汇报文件（docx/pptx）

## 2026-07-16 删除清单（可从 a5cec9d 恢复）

- **代码**：`fasttd3_ptf/edsf/`（ED-SF）、`fasttd3_ptf/ptf/entity/`（entity encoder/z-native/anchored readout）、`fasttd3_ptf/envs_ext/`（package 辅助环境）、SIV 闭包（`factorial_data/update_kernel/learner_factory/hb_branch_state.py`）、`train_fasttd3.py`、`source_bank/validator.py`、`utils/device.py`；`train_ptf.py`/`mcg.py` 中的 entity/z-native/chain 分支
- **脚本**：package/SIV/SHU 全部 probe、`scripts/step2/`、`scripts/experiments/`、一次性工具（约 30 个）
- **测试**：8 个死线测试
- **configs**：package/znative bank 10 个、误置的 `configs/source_banks/wandb/`
- 各死线的**结论**均保留在 `docs/archive/` 与 `EXPERIMENT_LOG.md`
