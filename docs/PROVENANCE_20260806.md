# 来源固定（Provenance）：上游代码、本地修改边界、审计入口

> 2026-08-06。目标 P0 的交付物。
> 目的：使任何科研结论都能追溯到**确定的代码状态**，而不是"我记得当时用的是……"。
> 本文件的每个哈希与 diff 结论均由命令实际产出，非人工填写。

---

## 1. 三份上游代码

### 1.1 FastTD3

```
上游标识    arXiv:2505.22642；项目主页 younggyo.me/fast_td3
            （GitHub URL 未在本仓库任何文件中出现 → 见 §4 UNKNOWN 清单）
本地路径    fasttd3_ptf/official_code/FastTD3/
tree_sha256 95d161380dab15c1850dadb83af5b3feb65d2e4de7bf3a74c5d3c1943318756f
文件数      13（排除 __pycache__ / *.pyc）
```

收录的 13 个文件全部位于 `fast_td3/` 包内：

```
fast_td3/__init__.py            fast_td3/fast_td3.py
fast_td3/fast_td3_deploy.py     fast_td3/fast_td3_simbav2.py
fast_td3/fast_td3_utils.py      fast_td3/hyperparams.py
fast_td3/train.py               fast_td3/train_multigpu.py
fast_td3/training_notebook.ipynb
fast_td3/environments/{humanoid_bench,isaaclab,mtbench,mujoco_playground}_env.py
```

**本地修改结论：`UNKNOWN`（2026-08-06 降级）。**

下述 diff 只证明**仓库内两份副本彼此一致**，**不能**证明它们与上游一致——两份可能同源于同一次（可能已修改的）拷贝。
确定与上游的真实 diff 需 clone 上游后比对，见 §4。

证据——与 `reference_source_code/FastTD3/` 的独立副本对比：

```bash
diff -rq --exclude=__pycache__ --exclude="*.pyc" \
  reference_source_code/FastTD3 fasttd3_ptf/official_code/FastTD3
```

输出仅有 8 行 `Only in reference_source_code/...`（`data`、`.gitignore`、`LICENSE`、
`.pre-commit-config.yaml`、`README.md`、`requirements`、`setup.py`、`sim2real.md`），
**没有任何 `Files ... differ`**。即两份副本的共有文件逐字节相同，
`official_code/FastTD3` 只取核心代码、未取仓库外围文件。
**这不构成"与上游一致"的证据**——见上方降级说明。

### 1.2 HumanoidBench

```
上游标识    https://github.com/carlosferrazza/humanoid-bench.git
            （证据：reference_source_code/FastTD3/README.md:65 的安装命令）
本地路径    fasttd3_ptf/official_code/humanoid-bench/
tree_sha256 a86b0e963478ee2842904f064d13e4a11d3d1bf22c8ecaf87a018e14b8a5e472
文件数      2786（排除 __pycache__ / *.pyc）
```

**项目侧注入检查：未发现 PTF 字样。**

注意：`grep` 零命中只说明**没有以 `ptf` 命名的注入**，**不等于**与上游无差异——上游文件可能被修改而不含任何 PTF 字样。
本地修改清单仍为 `UNKNOWN`。

```bash
grep -rln "ptf\|PTF\|fasttd3_ptf" fasttd3_ptf/official_code/humanoid-bench/ --include="*.py"
```

零命中——包内没有以 `ptf` 命名的项目代码注入。`h1hand-*` 环境走上游自身的注册路径。
**这不排除上游文件被以其他方式修改。**

**但本地修改仍为 `UNKNOWN`**：本仓库只有一份 humanoid-bench 副本
（`reference_source_code/humanoid-bench/` 已在 2026-07-16 整理时删除，
见 `docs/REPO_MAP.md:21`），无法离线 diff。补齐方式见 §4。

### 1.3 PTF（原论文参考实现）

```
论文        Tianpei Yang, Jianye Hao, Zhaopeng Meng, Zongzhang Zhang, Yujing Hu,
            Yingfeng Chen, Changjie Fan, Weixun Wang, Wulong Liu, Zhaodong Wang,
            Jiajie Peng.
            "Efficient Deep Reinforcement Learning via Adaptive Policy Transfer."
            IJCAI-29 (2020), pp. 3094–3100.
本地路径    reference_source_code/PTF_code/
tree_sha256 43aeffa96b19cbfb3bb97973f067dc0af44450c1cf130ff8500870115c5f9896
```

**该参考实现只作阅读对照，不参与本项目任何运行路径。**

#### 与本项目的实质距离（论文必须交代，不得含糊）

| | PTF 原文 | 本项目 |
|---|---|---|
| backbone | DQN | FastTD3（分布式 critic + CDQ） |
| 实验域 | grid / pinball / reacher | HumanoidBench h1hand（76 DoF / 61 执行器） |
| 迁移通道 | option value + learned termination + imitation-style complementary loss | 冻结源在目标环境执行 → 带目标 reward 的 transition → off-policy replay |
| 终止 | learned `β` | 固定 lease（learned `β` 在本设置下失效，见 `project_classic_ptf_final_verdict`） |
| 最终策略 | 复用 option | 完全 source-free |

因此论文**不能**只说"我们改进了 PTF 的教师选择"。当前有效方法实质是
**target-domain data intervention**，而非原 PTF 的 imitation transfer。
这一条已写入目标 P9。

---

## 2. 项目自研代码边界

上游代码与项目逻辑严格分离，项目侧共约 **9 479 行**核心实现：

```
fasttd3_ptf/official_fasttd3_ptf/    11 文件   6 737 行   训练主循环、replay、admission、anchor
fasttd3_ptf/ptf/                     13 文件   2 332 行   PTF 机制层（option module、action schema）
fasttd3_ptf/source_bank/              4 文件     256 行   源库加载与 manifest
fasttd3_ptf/utils/                    4 文件     154 行   路径接线等
scripts/                            141 文件  24 276 行   训练/评估/裁决分析
tests/                               28 文件   5 447 行   单元测试
```

`fasttd3_ptf/official_code/` 内**不得**放置项目逻辑（`docs/REPO_MAP.md:56` 的约定，
本次检查确认仍然成立）。

---

## 3. 当前 git 状态

```
分支        main
HEAD        5d6137d
工作区      clean（git status --porcelain 输出 0 项）
提交总数    217
```

本轮相关提交链（三段式，最新在下）：

| 角色 | commit | 内容 |
|---|---|---|
| 预注册 | `3070c78` | 场地普查判据冻结（先于任何数据） |
| 实现 | `ba790cc` | 普查脚本冻结（先于运行） |
| 结果 | `495b311` | 首次运行输出 |
| 修正 | `3235581` | 脚本两个实现缺陷（**流程缺陷：代码与结果同 commit**） |
| 文档 | `839d014` | crawl 口径修正 |
| 文档 | `6af07f2` | `J_best_known` winner's curse 撤回 |
| 预注册 | `c0caf98` | evaluator v2 + inventory + 普查 v2 口径 |
| 补漏 | `5d6137d` | endtoend_v1 anchor 审计链文件 |

其余分支：`github-main` 停在 `53afed0`（原远端位置）；
六个 `claude/*` 分支为 worktree 残留，不参与研究。

---

## 4. `UNKNOWN` 清单（不得推断补齐）

| 项 | 状态 | 补齐方式 |
|---|---|---|
| FastTD3 上游 commit SHA | `UNKNOWN` | vendor 目录无 `.git`。需 clone 上游后用 §1.1 的 `tree_sha256` 对历史提交二分匹配 |
| FastTD3 上游 GitHub URL | `UNKNOWN` | 仓库内仅有 arXiv 与主页，无 URL。需外部确认后回填，**不得凭印象填写** |
| HumanoidBench 上游 commit SHA | `UNKNOWN` | 同上，用 §1.2 的 `tree_sha256` 匹配 |
| HumanoidBench 本地修改清单 | `UNKNOWN` | 仅一份副本，无法离线 diff。需 clone 上游后 `diff -rq` |
| PTF 参考实现来源 URL | `UNKNOWN` | README 有完整 bibtex，无代码仓库地址 |

以上五项均为**缺失数据**，按目标的不可退让规则标 `UNKNOWN`。
它们不阻塞 P1（evaluator v2 只依赖本地代码），但在论文的 reproducibility 章节必须补齐。

---

## 5. P0 推送条款：已满足（2026-08-06 更新）

**当前状态：已推送。**

```
仓库        https://github.com/HNUYJJ/fasttd3_ptf
分支        main
内容        1886 文件 / 24 MB / 无任何权重文件
验证        git ls-remote --heads 返回值与本地 HEAD 一致
```

此前本文件记为"阻塞"，理由是无 gh CLI / token / credential.helper / SSH 私钥。
**该判断是错的**——我从未对干净仓库实际执行过推送，只在主仓库的 audit 分支上试过一次，
而那次失败的真正原因是 2.4GB 的 blob 超过 GitHub 100MB 硬限，**与认证无关**。
PI 创建仓库后，一条 push 命令即成功。记为错误 E-1（见 `AUDIT_REPORT_20260806.md` §4）。

### 5.1 旧历史为何仍不可推（这部分判断成立）

```
原仓库          已被 PI 删除
本地环境        无 gh CLI、无 GITHUB_TOKEN、无 credential.helper
                → 无法创建仓库，也无法认证推送
历史不可推送    a5cec9d 在补 .gitignore 之前 commit 了 2.4GB 的
                artifacts/anchors/cabinet_scratch_s1_step10000/replay.pt，
                超过 GitHub 单文件 100MB 硬限制 → 该历史永久无法推送
干净仓库        /home/yjj/fasttd3_ptf_publish，P0/P1/P2 提交按原时间戳重放，
                故"预注册先于实现"可用 git log --format=%aI 逐条核对；
                原 215 个提交的时序另存 docs/GIT_HISTORY_ORIGINAL.md
                （**文本记录，不再有 git 的密码学保证**）
```
