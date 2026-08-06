# FastTD3-PTF

在 HumanoidBench (h1hand) 上研究跨任务策略迁移强化学习：官方 FastTD3 训练源任务
策略（stand/walk/run/hurdle 等），通过 PTF（Policy Transfer Framework）机制在复杂
目标任务上做 reward-bearing bootstrap 迁移 + source admission lifecycle。

基本流水线：

1. 官方 FastTD3 训练源策略；
2. 导出 source manifest 并构建 source bank；
3. `train_ptf.py` 加载冻结源库，在目标任务上训练 FastTD3 + PTF/RBO/admission。

## 文档导航

| 文档 | 内容 |
|---|---|
| [docs/RESEARCH_ROADMAP.md](docs/RESEARCH_ROADMAP.md) | 科研路线：总目标、时间线、当前坐标、待决方向 |
| [docs/EXPERIMENT_LOG.md](docs/EXPERIMENT_LOG.md) | 实验结果及分析：机制栈、方法代号、全实验总表与裁决 |
| [docs/ISSUES_AND_LESSONS.md](docs/ISSUES_AND_LESSONS.md) | 问题记录：工程纪律、方法论教训、实验 checklist |
| [docs/REPO_MAP.md](docs/REPO_MAP.md) | 仓库结构地图（每个文件属于哪条线、删改影响范围） |
| [docs/archive/](docs/archive/README.md) | 全部历史文档归档（预注册/结果/审计原件） |

## 安装

```bash
python -m pip install -e .
python -m pip install -r requirements/requirements.txt
```

HumanoidBench 环境依赖 MuJoCo；`h1hand-*` 环境注册使用项目内快照
`fasttd3_ptf/official_code/humanoid-bench`，无需额外安装 humanoid_bench 包。
训练统一使用 conda env `FastTD3`。

## 标准工作流

```bash
# 1) 训练源策略(官方 train.py)
conda run -n FastTD3 bash scripts/official_fasttd3_train_h1hand_sources.sh

# 2) 导出 manifest 并构建 source bank
conda run -n FastTD3 bash scripts/official_fasttd3_export_h1hand_sources.sh
conda run -n FastTD3 python -m fasttd3_ptf.source_bank.builder \
  --sources checkpoints/official_sources/*/manifest.json \
  --output configs/source_banks/my_bank.yaml

# 3a) 目标任务 scratch 基线
ENV_NAME=h1hand-push-v0 conda run -n FastTD3 bash scripts/official_fasttd3_train_target_scratch.sh

# 3b) 目标任务 PTF(静态 RBO 主方法示例: wfix bank + safe_bootstrap)
conda run -n FastTD3 python -m fasttd3_ptf.official_fasttd3_ptf.train_ptf \
  --env-name h1hand-truck-v0 \
  --exp-name wfix_truck \
  --project fasttd3_ptf \
  --ptf-source-bank configs/source_banks/h1hand_hurdle4_wfix_truck.yaml \
  --ptf-mcg --ptf-mcg-ablation bootstrap_only --ptf-mcg-warmup-mode safe_bootstrap
```

长训练务必 tmux + `PYTHONUNBUFFERED=1` + `tee` 日志（详见
[docs/ISSUES_AND_LESSONS.md](docs/ISSUES_AND_LESSONS.md) §4 checklist）。

## 测试

```bash
conda run -n FastTD3 python -m pytest tests/ -q
```

## 关键运行注记

- PTF 激活时自动关闭 `torch.compile`（动态 option 控制流不兼容）；
  scratch vs PTF 对比必须按 env steps 而非 wallclock。
- 源策略冻结，经 masked action distillation 影响目标 actor：
  `actor_loss = -Q(s,π) + λ(t)·(1-β)·D(π, source, mask)`；null option 提供安全出口。
- 跨任务 obs 适配必须用显式 adapter（`robot_only`/`reach`/`slice`/
  `humanoidbench_robot_qpos_qvel`），禁止隐式截断/补零——维度合法但语义错误的
  适配会**静默**腐蚀蒸馏。
- h1hand 动作分组（legs 0:10 / torso 10 / arms 11:21 / hands 21:61）见
  `fasttd3_ptf/ptf/action_schema.py`。
- `checkpoints/sources/` 旧格式源 checkpoint 仍被 bank 引用（加载依赖
  `ptf/legacy_actors.py`），勿删。
- 历史代码恢复点：2026-07-16 整理前全量快照 = git `a5cec9d`。

---

## 关于本仓库（2026-08-06 重建）

本仓库只收录**代码、配置与科研证据文档**（约 23 MB / 1800 文件），
不含任何模型权重、replay buffer、训练产物或 3D 网格资产。

原仓库的 `.git` 因早期误提交了 2.4 GB 的 `replay.pt` 而膨胀到 59 GB，
且超过 GitHub 单文件 100 MB 硬限制、永久无法推送，故重建为干净历史。
原始 215 个提交的完整时序记录见
[docs/GIT_HISTORY_ORIGINAL.md](docs/GIT_HISTORY_ORIGINAL.md)
——本项目方法论要求**判据必须先于数据冻结并提交**（CLAUDE.md §4），
该文件保留了可逐条核对的时间戳。

### 补回运行所需的 3D 资产

MJCF 环境定义（`*.xml`）已全部收录，但机器人网格（`.obj` / `.stl` / 纹理 `.png`，
共约 209 MB）未收录——它们是 HumanoidBench 上游的公开资产：

```bash
git clone https://github.com/carlosferrazza/humanoid-benchmark.git /tmp/hb
rsync -a --include='*/' --include='*.obj' --include='*.stl' --include='*.STL' \
      --include='*.mtl' --include='*.png' --exclude='*' \
      /tmp/hb/humanoid_bench/assets/ \
      fasttd3_ptf/official_code/humanoid-bench/humanoid_bench/assets/
```

补回后即可正常创建 `h1hand-*` 环境。仅做代码审阅或读实验结论时无需此步。

### 数据与结论的对应

全部实验的原始裁决输出在 `docs/data/`（JSON + 含 `VERDICT` 行的 `.log`），
结论文档在 `docs/experiments/`。每份结论文档均标注其预注册文件与冻结时点。
