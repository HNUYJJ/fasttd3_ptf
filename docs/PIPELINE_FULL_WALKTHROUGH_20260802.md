# 全流程梳理：从环境创建到裁决的完整生命周期

> 2026-08-02。本文按**执行顺序**梳理本课题的每一个环节，
> 每一节都给出「这一步在哪个文件、消费什么、产出什么」。
> 目的是让人只读这一份文档就能完整理解整条链路。
>
> 阅读约定：`路径:行号` 可直接跳转；「上游」=HumanoidBench/FastTD3 原始代码，
> 不含本项目逻辑；「本项目」=我们自己写的迁移机制。

---

## 0. 三层架构与依赖方向

```
┌─────────────────────────────────────────────────────────────┐
│  L3  实验编排层  scripts/*.sh + scripts/analysis/*.py        │  ← 预注册 / 裁决
├─────────────────────────────────────────────────────────────┤
│  L2  本项目迁移机制                                           │
│      official_fasttd3_ptf/  训练入口 + replay + admission     │  ← 科研创新在这里
│      ptf/                   PTF / MCG / QMP / source bank     │
├─────────────────────────────────────────────────────────────┤
│  L1  上游代码（保持 source-compatible，禁止放项目逻辑）        │
│      official_code/FastTD3/          算法骨架 + 网络           │
│      official_code/humanoid-bench/   环境 + 机器人 MJCF        │
└─────────────────────────────────────────────────────────────┘
```

依赖单向：`official_fasttd3_ptf/ → ptf/ → utils/ + config.py`，无反向依赖。
上游代码通过 `fasttd3_ptf/official_fasttd3_ptf/paths.py` 挂到 `sys.path`
（`ensure_fasttd3_import_path()` / `ensure_humanoidbench_import_path()`），
**不修改上游源码**，所有补丁写在旁边的 wrapper 里。

---

## 阶段 1｜环境创建：从一个字符串到 MuJoCo 物理世界

### 1.1 入口：`ENV_NAME=h1hand-hurdle-v0`

| 步骤 | 文件 | 做了什么 |
|---|---|---|
| ① 注册 | `official_code/humanoid-bench/humanoid_bench/__init__.py` | 双重循环 `ROBOTS × TASKS` 注册出所有 `{robot}-{task}-v0` gym id |
| ② 分派 | `humanoid_bench/env.py:66-100` | `ROBOTS` 字典（6 种机器人）+ `TASKS` 字典（32 个任务）|
| ③ 构造 | `humanoid_bench/env.py:103` `HumanoidEnv.__init__` | 拼出 `model_path = assets/envs/{robot}_{control}_{task}.xml` |

对 `h1hand-hurdle-v0`：`robot=h1hand`, `control=pos`, `task=hurdle`
→ 加载 `assets/envs/h1hand_pos_hurdle.xml`。

### 1.2 MJCF 装配链（场景是由 include 拼出来的）

`assets/envs/h1hand_pos_hurdle.xml` 只有 12 行，本身不含任何几何体，它 include 四块：

```xml
<option timestep="0.002" iterations="100" solver="Newton"/>   <!-- 物理步长 2ms -->
<include file="../common/visual.xml"/>      <!-- 灯光/材质/渲染设置 -->
<include file="../common/floor.xml"/>       <!-- 地面平面 -->
<include file="../robots/h1hand_pos.xml"/>  <!-- 机器人本体（545 行）-->
<include file="../tasks/hurdle.xml"/>       <!-- 任务场景物件 -->
<keyframe><key name="qpos0" qpos="0 0 0.98 1 0 0 0 ... "/></keyframe>  <!-- 76 个初始 qpos -->
```

**任务场景文件（`assets/tasks/*.xml`）就是「target 之间唯一的物理差异」**：

| target | 场景文件 | 内容 |
|---|---|---|
| walk / run / stand | 无（只有平地） | `h1hand_pos_walk.xml` 不 include tasks/ |
| hurdle | `tasks/hurdle.xml`（5 行） | 再 include `locomotion/generated_xml_hurdles.xml`，一排栏架 |
| slide | `tasks/slide.xml`（32 行） | 9 段 mesh 斜坡（顶点 `0.3 5 0 … 5.3 5 1.75 …`）+ 两侧围墙 |
| crawl | `tasks/crawl.xml`（13 行） | 一条 16m 长、顶高 1.35m 的隧道（三块 box）|
| stair | `tasks/stair.xml` | 楼梯 |
| door | `tasks/door.xml`（51 行） | 门 + 把手（额外 2 个 DoF）|

`assets/envs/` 共 **112 个** XML（6 robot × 若干 task 的组合）。

### 1.3 机器人本体：`assets/robots/h1hand_pos.xml`

**关节树**（`h1hand_pos.xml:248-415`）：

```
pelvis (freejoint: 3 平移 + 4 四元数 = qpos 7 / qvel 6)
├── left_hip_yaw → hip_roll → hip_pitch → knee → ankle      (5 关节)
├── right_hip_yaw → hip_roll → hip_pitch → knee → ankle     (5 关节)
└── torso_link (torso 关节, 1)
    ├── left_shoulder_pitch → roll → yaw → elbow → wrist_yaw (5)
    │   └── left_hand  ← include shadow_hand_menagerie/left_hand.xml   (24 关节)
    └── right_shoulder_pitch → roll → yaw → elbow → wrist_yaw (5)
        └── right_hand ← include shadow_hand_menagerie/right_hand.xml  (24 关节)
```

数字对账（这是理解 obs/action 维度的基础）：

```
qpos = 7 (free base) + 21 (本体关节) + 48 (两只 Shadow hand 各 24) = 76 = H1Hand.dof
qvel = 6 (free base) + 21 + 48                                     = 75 = dof - 1
actuator = 61（见下）
```

`H1Hand.dof = 76` 定义在 `humanoid_bench/robots.py:76`；
`H1.dof = 26`（无手版本）、`G1.dof = 44`。

**执行器 61 个**（`h1hand_pos.xml:419-535`，三个 `<actuator>` 块），顺序即动作向量顺序：

| 索引 | 数量 | 名称 |
|---|---|---|
| 0–9 | 10 | `left/right_{hip_yaw, hip_roll, hip_pitch, knee, ankle}` |
| 10 | 1 | `torso` |
| 11–15 | 5 | `left_{shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_yaw}` |
| 16–20 | 5 | `right_*` 同上 |
| 21–40 | 20 | 左 Shadow hand：`lh_A_{WRJ2,WRJ1,THJ5..THJ1,FFJ4,FFJ3,FFJ0,MFJ*,RFJ*,LFJ*}` |
| 41–60 | 20 | 右 Shadow hand：`rh_A_*` |

手部 24 关节只有 20 个执行器，因为 Shadow hand 的远端指节是**耦合欠驱动**
（`FFJ1+FFJ2` 由 `FFJ0` 一个执行器驱动）。

`control="pos"` → 全部是 `<position>` 执行器（位置控制，带 `kp/kv/forcerange`）。

### 1.4 并行化：128 个独立进程

`fasttd3_ptf/official_fasttd3_ptf/humanoid_bench_env.py`（本项目，158 行）：

```python
self.envs = SubprocVecEnv([make_env(env_name, rank) for rank in range(num_envs)])
```

- `num_envs = 128`（本课题全部实验固定，见 `scripts/run_*_v1.sh` 的 `NUM_ENVS=128`）
- **每个 env 是一个独立的操作系统进程**（stable-baselines3 `SubprocVecEnv`），
  MuJoCo 本身在 CPU 上跑；这就是为什么 RAM 是硬瓶颈、并行训练进程数上限为 3
- 另外还建了一个 `render_env`（`num_envs=1`，`render_mode="rgb_array"`），
  只在 `RENDER_INTERVAL > 0` 时用；本课题实验一律设 0

**episode 长度**：`max_episode_steps(env_name)` 返回 1000（push/cube/basketball/kitchen 为 500），
用 `gymnasium.wrappers.TimeLimit` 包住。
`frame_skip = 10`、`timestep = 0.002s` → 控制频率 **50 Hz**，1000 步 = 20 秒仿真。

**播种（本项目专门修的坑）**：上游只调 `env.unwrapped.seed()`，那个方法只播 NumPy
进程全局 RNG，而 reset 噪声来自 Gymnasium 的 per-env `np_random`；
反过来 basketball 等任务的 reset 又用全局 `np.random`。
本项目用 `GlobalNumpySeedOnReset` wrapper 同时播两个 RNG
（`humanoid_bench_env.py:37-46`）。SB3 把 seed 存下来交给下一次 `reset(seed=seed+rank)`。

### 1.5 reset 与初始状态随机化

`humanoid_bench/env.py:229-242`：

```python
mujoco.mj_resetDataKeyframe(model, data, keyframe)   # 载入 XML 里的 qpos0
init_qpos + np_random.uniform(-0.01, +0.01, size=nq) # randomness = 0.01
task.reset_model()                                    # 任务特定 reset
```

---

## 阶段 2｜观测与动作：维度到底是什么

### 2.1 观测的构成

基类 `humanoid_bench/tasks.py:32` 的默认 `get_obs()`：

```python
state = concatenate(data.qpos.flat, data.qvel.flat)
```

即 **全量 qpos + 全量 qvel**（含任务物体的自由度），无视觉、无特权观测。
`envs.asymmetric_obs = False`，所以 **critic 与 actor 看同一个 obs**。

各任务的 `observation_space` 各自覆写。本课题涉及的 target：

| target | 定义位置 | 公式 | h1hand 实际维度 |
|---|---|---|---|
| stand / walk / run | `basic_locomotion_envs.py:41` | `robot.dof*2 - 1` | **151** |
| hurdle | 同上（继承 `Walk`） | 同上 | **151** |
| slide / stair | 同上（继承 `ClimbingUpwards`←`Walk`）| 同上 | **151** |
| crawl | 同上（继承 `Walk`） | 同上 | **151** |
| door | `envs/door.py:57`，`dof=2` | `robot.dof*2-1 + dof*2` | 155 |
| reach | `envs/reach.py:44` | `robot.dof*2-1 + 6` | 157 |
| push | `envs/push.py:53`，`dof=7` | `robot.dof*2-1 + 12` | 163 |
| powerlift | `envs/powerlift.py:55`，`dof=7` | `+ dof*2-1` | 164 |
| window | `envs/window.py:54`，`dof=11` | `+ dof*2-2` | 171 |
| package | `envs/package.py:62`，`dof=7` | `+ dof*2-1 + 9` | 173 |
| cabinet | `envs/cabinet.py:63`，`dof=33` | `+ dof*2-1-3` | 213 |
| truck | `envs/truck.py:70`，`dof=35` | `(robot.dof+dof)*2 - 6` | 216 |

**这张表是迁移能否成立的物理前提**：源策略（stand/walk/run，obs=151）
要用在 target 上，必须先把 target 的 obs 切回 151 维。
locomotion 系 target（hurdle/slide/stair/crawl）本身就是 151 维 → `identity` adapter；
door/cabinet 等则必须显式切片，否则**静默维度错配**（见 §5.3）。

机器人本体状态的读取接口在 `humanoid_bench/robots.py`：
`head_height()` / `torso_upright()` / `center_of_mass_velocity()` /
`actuator_forces()` / `left_hand_position()` 等——这些是 **reward 函数**用的，
不是 obs 的一部分。

### 2.2 动作

- `action_space = Box(-1, 1, shape=(61,))`（`env.py:164`，归一化到 ±1）
- `Task.step()` 先 `unnormalize_action` 映射回执行器的真实 `ctrl` 范围
  （`tasks.py:55-58`），再 `do_simulation(action, frame_skip=10)`
- 本项目把这 61 维按身体部位切成组：`fasttd3_ptf/ptf/action_schema.py:44-69`

```python
legs        [0, 10)     torso      [10, 11)    legs_torso [0, 11)
left_arm    [11, 16)    right_arm  [16, 21)    arms       [11, 21)
left_hand   [21, 41)    right_hand [41, 61)    hands      [21, 61)
```

MCG 的默认三组 `DEFAULT_GROUPS = ("legs_torso", "arms", "hands")`
（`ptf/mcg.py:41`），互不重叠（构造时断言）。

### 2.3 归一化链

| 对象 | 类 | 位置 | 说明 |
|---|---|---|---|
| obs | `EmpiricalNormalization` | `FastTD3/fast_td3/fast_td3_utils.py:402` | 在线更新 running mean/std，`(x-μ)/(σ+1e-2)`；**默认开启** |
| critic obs | 同上（独立实例） | `train_ptf.py:954` | 本项目 `asymmetric_obs=False`，输入相同但统计量独立 |
| reward | `RewardNormalizer` | `fast_td3_utils.py:500` | `reward_normalization` **默认 False**，本课题未开 |

关键点：replay buffer 里存的是**未归一化的 raw obs**，
采样后才做 `normalize_obs`（`train_ptf.py:3233-3236`）。
`data["raw_observations"]` 保留原始值专供**源策略**使用——
源有自己冻结的 normalizer，不能吃 target 的归一化结果。

---

## 阶段 3｜网络结构

全部在上游 `official_code/FastTD3/fast_td3/fast_td3.py`（278 行），本项目未改。

### 3.1 Actor（`fast_td3.py:157`）

```
obs(151) → Linear(151→512) → ReLU
         → Linear(512→256) → ReLU
         → Linear(256→128) → ReLU
         → Linear(128→61)  → Tanh          # fc_mu，权重 N(0, init_scale=0.01)
```

- 确定性策略（TD3），输出直接是动作
- 探索噪声：`explore()` 里 **per-env 的 noise_scale**，
  从 `U(std_min=0.001, std_max=0.4)` 采样，**每个 env done 时重采样一次**
  （`fast_td3.py:202-225`）——这是 FastTD3 的关键设计，不同 env 探索强度不同
- `hidden_dim = actor_hidden_dim = 512`

### 3.2 Critic：**分布式 double Q**（`fast_td3.py:84`）

```
Critic
├── qnet1: DistributionalQNetwork
└── qnet2: DistributionalQNetwork

DistributionalQNetwork:
  cat(obs, action)(212) → Linear(212→1024) → ReLU
                        → Linear(1024→512) → ReLU
                        → Linear(512→256)  → ReLU
                        → Linear(256→101)              # num_atoms = 101
  q_support = linspace(v_min=-250, v_max=250, 101)
```

- 输出是 **101 个原子上的 logits**，不是标量 Q
- `get_value(probs) = Σ probs · q_support` 才得到标量
- `projection()`：C51 式的分布投影（`fast_td3.py:36-81`）
- `use_cdq=True`：取两个 head 中 value 较小者的**整条分布**做 target

### 3.3 OptionModule（本项目，经典 PTF 路径用）

`fasttd3_ptf/ptf/option_module.py:49`：

```
obs → MLP(hidden_dims 默认 (256,256)) → ┬→ q_head:    Linear(→num_options)   # Q_ω
                                        └→ beta_head: Linear(→num_options)   # 终止概率 β
```

- 默认路径：`β = beta_min + (beta_max-beta_min)·sigmoid(logit)`，即 rescale 到 `[0.05, 0.95]`
- `released_code_fidelity=True`：还原 PTF 原作者代码（单层 ReLU6、`tanh` 的 Q、裸 sigmoid 的 β、
  权重初始化 `N(0, 0.01)`）
- `beta_logit_clip` 用 **straight-through clamp**（`option_module.py:14-27`），
  普通 `clamp` 区间外梯度为零，等于把 sigmoid 饱和换成硬夹死区

> 已裁决结论：β 信号被 sigmoid 死区吞噬，logit 坠到 −15~−17，
> 任何 5k gate 都没有诊断力。经典 PTF 修复线**已停**。

---

## 阶段 4｜源策略的完整生命周期

这是本课题的**前置资产**，与 target 训练完全解耦。

```
① 训练源                          ② 导出 manifest              ③ 组 bank
scripts/official_fasttd3_train_   scripts/official_fasttd3_     configs/source_banks/
  h1hand_sources.sh          →      export_h1hand_sources.sh →    calibration/*.yaml
  ↓调用上游 train.py                ↓调用 source_bank/exporter    ↓被 train_ptf.py 读
models/h1hand-walk-v0__..._final.pt  checkpoints/official_sources/
                                       h1hand_walk/manifest.json
```

### 4.1 ① 训练源（`scripts/official_fasttd3_train_h1hand_sources.sh`）

直接调**上游** `official_code/FastTD3/fast_td3/train.py`，训练四个源：

```bash
run_source h1hand-stand-v0  h1hand_stand_source_official
run_source h1hand-walk-v0   h1hand_walk_source_official
run_source h1hand-run-v0    h1hand_run_source_official
run_source h1hand-reach-v0  h1hand_reach_source_official
```

（另有 `checkpoints/terrain_sources/` 下的 crawl/hurdle/pole/slide/stair 地形源。）

### 4.2 ② 导出 manifest（`fasttd3_ptf/source_bank/exporter.py`）

从 checkpoint 里抠出维度、hidden dims、normalizer 位置，写成 JSON：

```json
{ "name": "walk", "env_id": "h1hand-walk-v0",
  "checkpoint": "models/h1hand-walk-v0__h1hand_walk_source_official__1_final.pt",
  "obs_dim": 151, "action_dim": 61, "actor_hidden_dims": [512, 256, 128],
  "checkpoint_format": "OfficialFastTD3.fasttd3.state_dict.v1",
  "normalizer": {"obs": "checkpoint.obs_normalizer_state", ...},
  "obs_adapter": {"type": "identity", "output_dim": 151},
  "action_adapter": {"type": "passthrough", "output_dim": 61},
  "action_mask": {"type": "full"} }
```

### 4.3 ③ source bank YAML（`configs/source_banks/`，共 ~100 个）

一个实际在用的（`calibration/h1hand_hurdle_rbo_run.yaml`）：

```yaml
null_option: true
sources:
- name: run
  manifest: checkpoints/official_sources/h1hand_run/manifest.json
  obs_adapter:   {type: identity, output_dim: 151}
  action_adapter:{type: passthrough}
  action_mask:   {type: full}
  compatibility_sigma: 1.5
  bootstrap: {weight: 0.0, horizon: 25}
```

bank 家族（`configs/source_banks/`）：

| 目录/前缀 | 用途 |
|---|---|
| `empty.yaml` | **scratch 对照专用**（空 bank = 纯 FastTD3，PTF/MCG 全部短路）|
| `calibration/h1hand_{target}_rbo_{src}.yaml` | 等剂量单源校准 bank（当前主力）|
| `official/` | 早期官方源 bank |
| `audit/` | 单源审计 bank |
| `pure_ptf/` | 经典 PTF 保真度实验 |
| `h1hand_loco_*`, `h1hand_std9_*`, `*_wfix_*` | 历史多源 bank 家族 |

生成脚本：`scripts/build_{expanded,safe_bootstrap,std9,bigsrc,stability_audit}_banks.py`
（每个 yaml 头部注释记录了生成来源，这是可复现性的依据）。

### 4.4 运行时加载（`fasttd3_ptf/ptf/source_policy.py` + `source_bank.py`）

`SourcePolicy.__init__` 做四件事：

1. 按 `model_class` 重建 actor 结构（`OfficialFastTD3Actor` / `UpstreamFastTD3Actor` / legacy `Actor`）
2. `_load_matching_state()` 只加载 **shape 匹配**的张量，加载数为 0 直接报错
3. **冻结源自己的 obs normalizer**（`_FrozenOfficialEmpiricalNormalizer`，`update=False`）
4. 构造 obs adapter / action adapter / action mask

`act()` 的三步（`source_policy.py:178-183`）：

```python
source_obs = self.obs_adapter(target_obs_raw)          # 维度对齐
source_obs = self.obs_normalizer.normalize(source_obs) # 源自己的冻结统计量
return self.action_adapter(self.actor(source_obs))     # 动作对齐
```

`SourcePolicyBank`（`ptf/source_bank.py`）提供 `act_all()`（所有源同时给动作，
返回 `[B, S, A]`）和 `act_selected()`（按 option id 选），外加 **null option**
（`names()` 里的 `"null"`，表示「不迁移」）。

> `null_option` 的真假会改变 `source_names` 的内容：
> hurdle bank 是 `['run','null']`，slide bank 是 `['walk']`。
> 审计脚本判断臂身份必须用「非 null 部分 == [arm]」，不能硬编码。

### 4.5 adapter：**禁止隐式截断/补零**（`ptf/adapters.py`）

这是本项目刻意加的安全阀。`IdentityObsAdapter` 在维度不符时**直接抛异常**，
除非显式传 `allow_truncate` / `allow_pad`：

```python
raise ValueError(f"IdentityObsAdapter got obs_dim={x.shape[-1]} but expected {self.output_dim}; "
                 "use an explicit slice/robot_only adapter for cross-task transfer.")
```

可用类型：`identity` / `slice` / `robot_only` / `reach` /
`humanoidbench_robot_qpos_qvel`（后者专为 door/cabinet 这类
「qpos 里混了任务 DoF」的 target，显式取 `qpos[:76] + qvel[:75]`）。

---

## 阶段 5｜训练主循环（`train_ptf.py`，3759 行，唯一训练入口）

### 5.0 启动链

```
scripts/run_hurdle_speedup_v1.sh          # 实验 launcher，冻结全部超参
  └─ env VAR=... bash scripts/official_fasttd3_train_target_ptf.sh   # 枢纽脚本
       └─ python -m fasttd3_ptf.official_fasttd3_ptf.train_ptf --env-name ... --ptf-source-bank ...
```

`official_fasttd3_train_target_ptf.sh`（253 行）的唯一职责：
把 ~60 个环境变量翻译成 `--ptf-*` 命令行参数。**所有实验都经由它启动**。

### 5.1 初始化顺序（`train_ptf.py:884` `main()`）

```
_parse_ptf_cli()          → ptf_cfg（本项目参数，~60 个）
get_args()                → args（上游 FastTD3 参数，来自 hyperparams.py）
                            ⚠ torch.compile 被强制关闭（PTF 有动态分支）
seed 三件套               → random / np.random / torch.manual_seed
_make_envs()              → envs(128) / eval_envs(=envs!) / render_env(1)
n_obs / n_act 从 env 读    → 151 / 61
EmpiricalNormalization    → obs + critic_obs
actor / qnet / qnet_target
AdamW ×2 + CosineAnnealingLR ×2
SimpleReplayBuffer  →  PTFReplayWrapper(rb)
SourcePolicyBank.from_config()      ← 打印 "Loaded source bank options: [...]"
OptionModule / ModularGating / McgBehaviorController / AdmissionSnapshot
```

> **`eval_envs = envs`**（`train_ptf.py:782`）——HumanoidBench 分支下
> 评估环境**就是训练环境本身**。这是本课题一律设 `EVAL_INTERVAL=0`、
> 改用离线 `scripts/p0_evaluator.py` 的原因之一。

### 5.2 超参（`official_code/FastTD3/fast_td3/hyperparams.py` + launcher 覆盖）

| 参数 | 上游默认 | 本课题实验值 | 说明 |
|---|---|---|---|
| `num_envs` | 128 | 128 | 并行环境 |
| `total_timesteps` | 100000 (HumanoidBench) | 100000 | **vector step**，×128 = 12.8M 环境交互 |
| `batch_size` | 32768 | 32768 | 实际按 `batch_size // num_envs = 256` per-env 采样 |
| `buffer_size` | 51200 | 51200 | per-env 槽位数 → 总容量 128×51200 |
| `num_updates` | 2 | 2 | 每个 vector step 做 2 次 critic 更新 |
| `policy_frequency` | 2 | 2 | actor 延迟更新 |
| `gamma` / `tau` | 0.99 / 0.1 | 同 | `tau=0.1` 是 FastTD3 的激进软更新 |
| `num_atoms` / `v_min` / `v_max` | 101 / −250 / 250 | 同 | 分布式 critic 支撑 |
| `critic/actor_hidden_dim` | 1024 / 512 | 同 | |
| `learning_starts` | 10 | 10 | |
| `compile` | True | **0（强制关）** | PTF 路径不兼容 |
| `amp` / `amp_dtype` | True / bf16 | 同 | |
| `use_cdq` | True | 同 | Clipped Double Q |

> **LR 日程的陷阱（已核实）**：
> `CosineAnnealingLR(T_max=args.total_timesteps, eta_min=critic_learning_rate_end)`，
> 但上游默认 `critic_learning_rate == critic_learning_rate_end == 3e-4`，
> 即 `eta_min == base_lr`，**余弦退火恒为常数**。
> 用 `PTF_RUN_STOP_STEP` 提前停止分支训练不会压缩 LR 日程——因为根本没有日程。

### 5.3 主循环逐步（`train_ptf.py:2532` `while global_step < run_stop_step`）

#### Step A：选动作（`train_ptf.py:2765-2960`）

四条互斥分支：

| 分支 | 条件 | 行为 |
|---|---|---|
| `target_only_behavior` | 空 bank / exact abstain | 纯 student：`policy(norm_obs, dones)` |
| `qmp_enabled` | `PTF_QMP=1` | per-state 全策略 `argmax_h min Q_h`，选完**再**加一次噪声 |
| `mcg_enabled` | `PTF_MCG=1` | **当前主路径**，见下 |
| else | 经典 PTF | `OptionSelector` + `Q_ω` + β termination |

**MCG 主路径**（`train_ptf.py:2797-2850`）：

```python
actions = policy(obs=norm_obs, dones=dones)         # 先拿 student 动作
in_warmup = global_step < mcg_warmup_steps
do_exec = (in_warmup and mcg_warmup_bootstrap) or (not in_warmup and mcg_gate_active)
if do_exec:
    src_actions_all, _ = source_bank.act_all(obs)   # 注意：raw obs，不是 norm_obs
    if in_warmup:
        mcg_best = mcg_gate = None                  # warmup：无条件 bootstrap
    else:
        deltas = mcg_gating.deltas(qheads_value, norm_obs, actions, src_actions_all)
        best, sig, gate, conf = mcg_gating.select(deltas, margins=ema, ...)
    actions, info = mcg_behavior.step(actions, src_actions_all, best, gate, dones)
```

本课题实验一律设 `PTF_MCG_WARMUP_STEPS = 总步数`，即**全程都在 warmup 分支**，
配 `PTF_MCG_ABLATION=bootstrap_only`（关闭 gate 蒸馏）——
所以实际跑的是**纯行为 bootstrap + replay 配额**，不含 critic gating。

**`admission_bootstrap` 抽源逻辑**（`ptf/mcg.py:573-617`）：

```python
probs = self.admission_probabilities()      # softmax([source_logits, student_logit] / τ)
arm = multinomial(probs, n)                 # 单一 categorical，源与 student 平等竞争
new = where(arm == num_src, -1, arm)        # -1 = student
self.current[env_exp] = new                 # per-env per-group 锁存
self.steps_left[env_exp] = horizon (=25)    # 锁存 25 步（call-and-return）
```

`PTF_ADMISSION_STUDENT_LOGIT=0.0` + source logit 0 → 候选质量 `[0.5, 0.5]`，
即 **50% 剂量**。实测 behavior share 0.477–0.499，验收带 `[0.45, 0.55]`。

#### Step B：与环境交互

```python
next_obs, rewards, dones, infos = envs.step(actions)
true_next_obs = where(dones, infos["observations"]["raw"]["obs"], next_obs)  # 处理 truncation
```

`humanoid_bench_env.py:125-155` 负责把 `TimeLimit.truncated` 的
`terminal_observation` 还原成正确的 bootstrap 目标。

#### Step C：写 replay + provenance（`train_ptf.py:3007-3094`）

```python
transition = TensorDict({observations, actions, next:{observations, rewards, truncations, dones}})
replay_provenance = {behavior_source, source_by_group, executed_group_mask,
                     segment_id, segment_step, anchor_id, env_rank, learner_step}
rb.extend(transition, option_ids, provenance=replay_provenance)
```

写入前有一条**硬断言**：rejected 源的 transition 不得进主 replay
（`train_ptf.py:3084`）。

#### Step D：更新（`train_ptf.py:3230-3335`，`global_step > learning_starts` 后）

```python
for i in range(num_updates=2):
    data = rb.sample(batch_size // num_envs)          # PTFReplayWrapper 的加权采样
    data["raw_observations"] = data["observations"].clone()   # ← 源要用 raw
    data["observations"] = normalize_obs(...)                 # ← 学生要用 normalized
    update_main(data, logs)                            # critic
    if actor_should_update: update_pol(data, logs, step)  # actor + 蒸馏
    if 经典 PTF: update_option(data, logs, step)          # Q_ω + β
    soft_update(qnet, qnet_target, tau=0.1)
```

**`update_main`（critic 更新，`train_ptf.py:1694`）**：

```python
next_action = (actor(next_obs) + clipped_noise).clamp(-1, 1)          # target policy smoothing
bootstrap = (truncations | ~dones).float()                            # truncation 仍 bootstrap
q1_proj, q2_proj = qnet_target.projection(next_obs, next_action, rewards, bootstrap, γ)
if use_cdq: 取 value 较小的那条**整分布**
qf_loss = 交叉熵(-Σ target_dist · log_softmax(qnet(obs, actions)))     # 分布式 TD
```

**`update_pol`（actor 更新，`train_ptf.py:1899`）**：

```python
pi_action = actor(obs)
qf_value = min(get_value(qnet1), get_value(qnet2))
rl_actor_loss = -qf_value.mean()
transfer_loss = compute_mcg_transfer_loss(...) 或 compute_transfer_loss(...) 或 0
actor_loss = rl_actor_loss + transfer_loss
```

- 经典 PTF 蒸馏（`compute_transfer_loss:1769`）：
  `loss = λ(t) · (1-β_o) · masked_distill(π(s), a_source)`
- MCG 蒸馏（`compute_mcg_transfer_loss:1814`）：
  权重换成 `λ(t) · 1[Δ_{i,g}(s) > margin_g] · conf`，
  即「向谁学、学哪个部位、在哪些状态学」全由 target critic 决定
- `isolate_classic_ptf` 或 `bootstrap_only` ablation 下，`transfer_loss ≡ 0`

**λ 日程**（`fasttd3_ptf/utils/schedules.py`）：
`LinearScheduler(start, end, duration)` 或 `ReleasedPTFTanhScheduler`
（还原原作者的 `0.5 + tanh(3 − 6·progress)/2`）。

#### Step E：replay 采样的配额机制（`ptf_replay.py`）

这是**迁移的第二条通道**（第一条是行为）。`_admission_slot_weights:383`：

1. 按 stratum（每个 admitted 源 + student）分配 candidate mass，
   只在「buffer 里确实还有该源数据」的 stratum 间归一
2. stratum 内按 recency 半衰 + 可选 priority 加权，再混 uniform floor

**authority handoff**（`draw_indices:484`）：源的 behavior authority 结束后，
配额从 admission mass 切回「allowed 槽位的物理占比」——
不这么做会在源数据物理残留 1.2% 时仍给它 50% 配额，
造成 43× 过采样（这就是已修复的 **80k repetition divergence** 崩点）。

`PTF_ADMISSION_REPLAY_MODE={shared, student_only}` 可**只覆盖 replay 侧配额**
而保持 behavior 侧不变——这是 Door 通道分解实验的机制基础。

---

## 阶段 6｜迁移机制层全景（本项目的科研内容）

| 机制 | 开关 | 核心文件 | 状态 |
|---|---|---|---|
| **经典 PTF** | 默认（无 `PTF_MCG`）| `option_module.py` / `option_selector.py` / `option_update.py` / `compatibility.py` / `distillation.py` | 已裁决：β 死区，**停修复线**；`fixed-walk` 3/3 加速是唯一稳健正结果 |
| **MCG** | `PTF_MCG=1` | `ptf/mcg.py` | gate 部分未采纳；`bootstrap_only` ablation 是**当前主路径** |
| **QMP** | `PTF_QMP=1` | `ptf/qmp.py` | `QMP_FIDELITY_PARTIAL`，退化成 student（source share 0.3–5.5%），**不解禁** |
| **RBO / bootstrap** | `PTF_MCG_WARMUP_MODE` | `mcg.py:McgBehaviorController` | 四种：`random` / `safe_bootstrap`(静态 RBO) / `online_bootstrap`(student-as-arm) / `admission_bootstrap`(**主路径**) |
| **admission lifecycle** | `PTF_ADMISSION_MODE` | `admission_control.py` + `ptf_replay.py` | 七种模式：`legacy/all/none/static/manifest/schedule/target_evidence` |
| **adaptive 撤销** | `PTF_ADMISSION_ADAPTIVE=1` | `admission_control.py:AdaptiveAdmissionController` | 已归档（truck 上代价 −119.7/−204.9）|

当前主路径的完整配置（`scripts/run_hurdle_speedup_v1.sh:35-44`）：

```bash
SOURCE_BANK=configs/source_banks/calibration/h1hand_hurdle_rbo_run.yaml
PTF_MCG=1  PTF_MCG_GROUPS=legs_torso,arms,hands
PTF_MCG_WARMUP_STEPS=100000        # = 总步数，全程 warmup
PTF_MCG_WARMUP_MIN_STEPS=25        # call-and-return horizon
PTF_MCG_WARMUP_MODE=admission_bootstrap
PTF_MCG_ABLATION=bootstrap_only    # 关闭 critic gating 蒸馏
PTF_ADMISSION_MODE=all
PTF_ADMISSION_STUDENT_LOGIT=0.0
PTF_ADMISSION_EXPECTED_SOURCE_MASS=0.5     # 50% 剂量
PTF_ADMISSION_REPLAY_RECENCY_HALF_LIFE=0
PTF_ADMISSION_REPLAY_UNIFORM_MIX=1
PTF_ADMISSION_REPLAY_PRIORITY_ALPHA=0
PTF_ADMISSION_REPLAY_HANDOFF=physical_after_authority
```

对照臂（scratch）只改两项：`SOURCE_BANK=empty.yaml`、`PTF_ADMISSION_MODE=legacy`。
**这就是「单因素只改是否有源」的实现方式。**

---

## 阶段 7｜checkpoint / anchor / 分支训练

### 7.1 普通 checkpoint（`train_ptf.py:834` `save_ptf_params`）

存 actor / qnet / qnet_target / 两个 normalizer 的 state / `args` / `global_step` /
`ptf_cfg` / `source_names` / option 模块 + optimizer / `admission_audit` / `training_audit`。

触发点：
- `SAVE_INTERVAL`（本课题实验设 0，关闭）
- **`PTF_EVAL_CHECKPOINT_STEPS=10000,20000,30000,50000,75000,100000`** ← 实际用的
  （`train_ptf.py:3529`，在 `global_step` 精确命中时保存，文件名 `models/{run_name}_{step}.pt`）

### 7.2 anchor bundle（`official_fasttd3_ptf/anchor_io.py`，442 行）

比 checkpoint 强得多：**learner + replay buffer + 全部 RNG 状态**的原子快照。

```
artifacts/{exp}/anchors/s{seed}/
├── manifest.json    # schema、git state、checksums
├── learner.pt       # modules + optimizers + schedulers + scaler
├── replay.pt        # 完整 replay（含 provenance）
├── rng.pt           # torch / numpy / python / cuda / 各 Generator
└── checksums.json
```

写入时断言 `replay.ptr == completed_vector_steps`（边界一致）。

**用途**：paired branch probe——从同一个 anchor 分叉出多条臂
（每条臂换一个源），学习轨迹在分叉点之前**逐位相同**，
分叉后的差异就是该源的因果效应 `U_i`。

相关开关：
- `PTF_ANCHOR_STEP` / `PTF_ANCHOR_DIR`：写 anchor
- `PTF_ANCHOR_RESUME`：从 anchor 恢复（日志打印 `Resumed core learner ... at step 10000`）
- `PTF_RUN_STOP_STEP`：分支提前停止（**不压缩 LR 日程**，见 §5.2）
- `PTF_RESUME_NOISE_SEED`：控制分支的探索噪声流

### 7.3 RNG 隔离（`official_fasttd3_ptf/rng_isolation.py`）

`GlobalRngState.capture(device)` 在构造 source/option/MCG **之前**捕获，
保证 empty-bank scratch 与 exact abstention 走同一条 target-only 快路径，
**不会因为构造了用不到的迁移组件而多消耗 RNG**。
这是「exact abstention 因果证据」的机制基础。

---

## 阶段 8｜评估

### 8.1 in-loop 评估（`train_ptf.py:1636` `evaluate()`）——本课题不用

因为 `eval_envs = envs`，会干扰训练环境。一律 `EVAL_INTERVAL=0`。

### 8.2 离线冻结评估（`scripts/p0_evaluator.py`）——**唯一采信的评估**

```bash
python scripts/p0_evaluator.py \
  --checkpoint models/*hspd_source_s1__*_30000.pt \
  --env-name h1hand-hurdle-v0 \
  --out docs/data/hurdle_speedup_v1/source_free_eval/source_s1_step30000.json \
  --expect-global-step 30000 --expect-seed 1 --expect-admission-mode all \
  --eval-seeds panel128
```

协议（**冻结，不得改动**）：

| 项 | 值 |
|---|---|
| **结构性 source-free** | 只加载 actor + 冻结 obs normalizer，**从不构造 bank/option/MCG/admission**——评估路径在结构上不可能碰到源 |
| 面板 | 16 eval seeds × 8 ranks = **128 deterministic episodes** |
| reset seed | `eval_seed * 1000 + rank`，分支间**逐位相同** |
| seed 列表 | `(11,23,37,53, 71,89,103,113, 131,149,163,179, 193,211,227,241)`，前 4 个必须保持不变（向后兼容 32-episode 子面板）|
| 播种 | 双播种：`np.random.seed(seed)` + `env.reset(seed=seed)` |
| 动作 | deterministic（无探索噪声）|
| episode | 1000 步 |
| 身份校验 | `--expect-*` 三项不符即拒绝，防止喂错 checkpoint |
| 输出 | JSON：per-episode 明细 + aggregate + checkpoint sha256 + git HEAD |

输出的 `aggregate` 字段：

```
return_mean / return_std / progress_max_dx_mean / posture_mean /
package_reward_mean / success_count / episode_count
```

> **`success_count` 的语义陷阱**：它读自 `terminated`。
> 在 hurdle/slide/stair 等 locomotion 任务上，`get_terminated()` 是
> **摔倒早停**（`qpos[2] < 0.2` 或 `torso_upright < 0.1`），
> 所以 `success_count` 是「摔倒次数」，**与 return 强反向**。
> 只有 package/truck 上它才真的是「成功」。

`load_student` 在 `scripts/probe_lib.py:58`，同样只加载 actor + 冻结 normalizer。

### 8.3 评估驱动脚本

`scripts/eval_{hurdle_speedup,slide_speedup,slide_bac_gate,racing_min_horizon,...}_v1.sh`
—— 遍历 `SEEDS × STEPS_LIST`，逐点调 `p0_evaluator.py`。

> 并行时必须先验证各链的**输出文件集合两两不相交**：
> `[[ -f "$OUT" ]] && skip` 只在启动瞬间检查，**不是原子锁**。

---

## 阶段 9｜裁决

`scripts/analysis/analyze_*.py`（21 个），每个对应一个预注册实验。
共同结构：

```
层1 工程检查   剂量 / 臂间 share 差 / 臂身份 / 冻结面板 sha256 / 协议一致性
层2 数据完整性  缺失 → 输出 INCOMPLETE 并非零退出（绝不落进 REFUTED/PASS 分支）
层3 主判据      预注册冻结的阈值与统计量
```

以 `analyze_hurdle_speedup_v1.py` 为例，核心是达阈步数的线性插值：

```python
def steps_to(pts, theta):
    prev = None
    for s, r in pts:
        if r >= theta:
            if prev is None: return float(s), False
            (s0, r0) = prev
            return s0 + (theta - r0) / (r - r0) * (s - s0), False
        prev = (s, r)
    return CENSOR, True          # 全程未达到 → 右删失
speedup(θ) = steps_scratch(θ) / steps_source(θ)
```

裁决输出写 `docs/data/{experiment}/results.json`（含 `run_id`），
结论文档写 `docs/experiments/{experiment}_results_{date}.md`。

**统计尺度的硬规矩**：跨 learner 的结论必须用 **learner 间方差**，
不是 episode 面板 SE。`RACING_K` 批1 用 episode-SE 看到 8.4–14.8 个 SE 的领先，
独立重复只有 1/3；正确尺度下 `t=1.57`，不显著。

---

## 阶段 10｜端到端示例（hurdle 加速实验，完整一遍）

```bash
# ── 前置（一次性）：训练并导出源 ─────────────────────────────
bash scripts/official_fasttd3_train_h1hand_sources.sh        # → models/*_final.pt
bash scripts/official_fasttd3_export_h1hand_sources.sh       # → checkpoints/official_sources/*/manifest.json
# bank yaml 已在 configs/source_banks/calibration/ 冻结

# ── ① 预注册（必须先于任何数据）──────────────────────────────
# docs/experiments/hurdle_speedup_v1_prereg_20260730.md   + git commit
# scripts/analysis/analyze_hurdle_speedup_v1.py           + git commit

# ── ② 训练两臂（tmux + PYTHONUNBUFFERED=1，并行度 ≤3）────────
tmux new -s hspd
GPU=0 SEEDS='1 2 3' ARM=scratch bash scripts/run_hurdle_speedup_v1.sh
GPU=1 SEEDS='1 2 3' ARM=source  bash scripts/run_hurdle_speedup_v1.sh
#   → models/h1hand-hurdle-v0__hspd_{arm}_s{seed}__{seed}_{10000..100000}.pt

# ── ③ source-free 评估（128 episodes/点）────────────────────
GPU=0 SEEDS='1 2 3' ARM=scratch bash scripts/eval_hurdle_speedup_v1.sh
GPU=0 SEEDS='1 2 3' ARM=source  bash scripts/eval_hurdle_speedup_v1.sh
#   → docs/data/hurdle_speedup_v1/source_free_eval/{arm}_s{seed}_step{N}.json

# ── ④ 剂量验收（M26：排除"某源更好只因用得更多"）────────────
python scripts/analysis/audit_hurdle_speedup_dose_v1.py

# ── ⑤ 裁决（判据在看到结果前已冻结，只准改路径参数）──────────
python scripts/analysis/analyze_hurdle_speedup_v1.py
#   → docs/data/hurdle_speedup_v1/results.json  → SPEEDUP_CONFIRMED

# ── ⑥ 结论文档 + 边界 ───────────────────────────────────────
# docs/experiments/hurdle_speedup_v1_results_20260730.md
```

---

## 附录 A｜文件总清单（按职能，一个不漏）

### A.1 上游 HumanoidBench（`fasttd3_ptf/official_code/humanoid-bench/`）

| 类别 | 文件 |
|---|---|
| 注册 / 分派 | `humanoid_bench/__init__.py`、`humanoid_bench/env.py` |
| 机器人本体定义（Python） | `humanoid_bench/robots.py`（H1/H1Hand/H1SimpleHand/H1Touch/H1Strong/G1 + 状态读取接口）|
| 任务基类 | `humanoid_bench/tasks.py`（`get_obs` / `get_reward` / `get_terminated` / 动作归一化）|
| 任务实现（我们用到的） | `envs/basic_locomotion_envs.py`（**stand/walk/run/hurdle/crawl/stair/slide/sit**）、`envs/door.py`、`envs/cabinet.py`、`envs/push.py`、`envs/package.py`、`envs/truck.py`、`envs/window.py`、`envs/powerlift.py`、`envs/reach.py`、`envs/pole.py`、`envs/maze.py`、`envs/balance.py`、`envs/basketball.py`、`envs/spoon.py`、`envs/bookshelf.py`、`envs/highbar.py`、`envs/insert.py`、`envs/room.py`、`envs/cube.py`、`envs/kitchen.py` |
| wrapper | `humanoid_bench/wrappers.py`（`BlockedHandsLocoWrapper` / `ObservationWrapper` / reach 层级 wrapper）|
| dm_control 依赖 | `dmc_deps/{dmc_index,dmc_sizes,dmc_util,dmc_wrapper}.py`（named indexing）|
| **机器人 MJCF** | `assets/robots/h1hand_pos.xml`（**本课题唯一用的本体**，545 行）；同目录还有 `h1_pos.xml` / `h1simplehand_pos.xml` / `h1touch_pos.xml` / `h1strong_pos.xml` / `h1gripper_pos.xml` / `g1_torque.xml` / `digit_torque.xml` |
| **手部 MJCF** | `assets/shadow_hand_menagerie/{left_hand,right_hand,keyframes}.xml` + `assets/` 网格 |
| **场景 MJCF** | `assets/envs/*.xml`（112 个，`{robot}_{control}_{task}.xml`）|
| **任务物件 MJCF** | `assets/tasks/*.xml`（29 个：hurdle/slide/stair/crawl/door/cabinet/…）+ `assets/tasks/assets/` |
| 公共 MJCF | `assets/common/{visual,floor}.xml` |
| 地形生成 | `assets/locomotion/generated_xml_hurdles.xml` 等 |
| 机器人网格 | `assets/h1/assets/`（STL）|

### A.2 上游 FastTD3（`fasttd3_ptf/official_code/FastTD3/fast_td3/`）

| 文件 | 用途 |
|---|---|
| `fast_td3.py` | **Actor / Critic / DistributionalQNetwork**（+ MultiTask 变体）|
| `fast_td3_utils.py` | `SimpleReplayBuffer` / `EmpiricalNormalization` / `RewardNormalizer` / `cpu_state` / `mark_step` |
| `hyperparams.py` | **全部默认超参**（`BaseArgs` + 各任务特化类）|
| `train.py` | 上游训练脚本 —— **只用于训练源策略** |
| `environments/humanoid_bench_env.py` | 上游 HB wrapper（本项目**未用**，被自己的替换）|
| `fast_td3_simbav2.py` / `fast_td3_deploy.py` / `train_multigpu.py` / `environments/{isaaclab,mtbench,mujoco_playground}_env.py` | 未在本课题使用 |

### A.3 本项目训练入口层（`fasttd3_ptf/official_fasttd3_ptf/`）

| 文件 | 行数 | 用途 |
|---|---|---|
| `train_ptf.py` | 3759 | **唯一训练入口**，全机制接线 |
| `ptf_replay.py` | 947 | replay wrapper：provenance / admission 配额 / authority handoff |
| `admission_control.py` | 570 | 准入快照 / 调度 / adaptive 撤销状态机（纯 CPU、零 RNG）|
| `anchor_io.py` | 442 | anchor bundle 快照（learner + replay + rng）|
| `target_evidence_probe.py` | 365 | MuJoCo `mjSTATE_FULLPHYSICS` 快照 + matched branch rollout 探针 |
| `target_evidence.py` | 223 | target-evidence 契约 |
| `humanoid_bench_env.py` | 158 | **HB 向量环境 + 双 RNG 正确播种**（替换上游 wrapper）|
| `source_admission.py` | 100 | quarantine bank 结构校验 |
| `paths.py` | 85 | `official_code` 的 `sys.path` 接线 |
| `rng_isolation.py` | 61 | RNG 隔离 |

### A.4 本项目 PTF 机制层（`fasttd3_ptf/ptf/`）

| 文件 | 行数 | 用途 |
|---|---|---|
| `mcg.py` | 751 | `ModularGating` + `McgBehaviorController` + `AdmissionSegmentTracker`（**行为调度核心**）|
| `adapters.py` | 304 | obs/action 适配器 + action mask（**禁隐式截断**）|
| `option_update.py` | 211 | `Q_ω` TD target / U-value / termination loss |
| `source_policy.py` | 207 | 单个源的加载 + adapter + 冻结归一化 |
| `legacy_actors.py` | 177 | 旧格式源 ckpt 兼容层（`checkpoints/sources/` 仍在用，**勿删**）|
| `option_selector.py` | 148 | call-and-return option 选择 |
| `qmp.py` | 148 | QMP per-state Q-switch |
| `option_module.py` | 145 | PTF option-value + β termination 网络 |
| `source_bank.py` | 78 | `SourcePolicyBank`（`act_all` / `act_selected` / null option）|
| `action_schema.py` | 69 | h1hand 61 维动作分组 |
| `compatibility.py` | 52 | 高斯动作兼容度（PTF ξ 项）|
| `distillation.py` | 26 | masked action distillation loss |

### A.5 本项目工具层

| 文件 | 用途 |
|---|---|
| `fasttd3_ptf/config.py` | YAML 配置加载 |
| `fasttd3_ptf/utils/schedules.py` | `LinearScheduler` / `ReleasedPTFTanhScheduler` |
| `fasttd3_ptf/utils/normalization.py` | `TensorNormalizer`（legacy 源用）|
| `fasttd3_ptf/utils/checkpoint.py` | `load_torch` / `load_json` |
| `fasttd3_ptf/source_bank/{exporter,builder,manifest}.py` | bank 工具链 |

### A.6 配置（`configs/`）

| 路径 | 内容 |
|---|---|
| `source_banks/empty.yaml` | **scratch 对照专用** |
| `source_banks/calibration/*.yaml` | 等剂量单源校准 bank（当前主力，23 个）|
| `source_banks/{official,audit,pure_ptf}/` | 官方 / 单源审计 / 经典 PTF 保真 bank |
| `source_banks/h1hand_{loco,std9,big,hurdle4}_*` | 历史多源 bank 家族 |
| `experiments/*.yaml` | **预注册冻结配置**（含 research_gate / arms / metrics / decision）|
| `admission_schedules/*.yaml` | 准入调度表（hard_exit / retention）|
| `target_evidence/*.yaml` | target-evidence 契约配置 |
| `reward_structure/humanoidbench_v1.py` | **17 个 target 的 reward 组合结构规格**（ADDITIVE / MULTIPLICATIVE / GATED / UNBOUNDED + `min_groups`）——BAC 主线的基础 |

### A.7 脚本（`scripts/`）

| 职能 | 文件 |
|---|---|
| **训练枢纽** | `official_fasttd3_train_target_ptf.sh`（全部实验经它）、`official_fasttd3_train_target_scratch.sh` |
| 源训练 / 导出 | `official_fasttd3_train_h1hand_sources.sh`、`official_fasttd3_export_h1hand_sources.sh`、`export_source_policy.sh` |
| 实验 launcher | `run_{hurdle_speedup,slide_speedup,slide_bac_gate,stair_bac_gate,door_at10k_gate,cabinet_at10k_gate,racing_min_horizon,racing_reject_door,qmp_fidelity,critic_first_bridge,slide_hard_exit,door_channel_decomposition,door_prefix_handoff,classic_ptf_*,admission_*,phase1_bounded_bank_lease,stage_conditioned_*}_v1.sh` |
| **评估驱动** | `p0_evaluator.py`（**核心**）、`eval_*_v1.sh`、`probe_lib.py` |
| 裁决分析 | `analysis/analyze_*.py`（19 个）、`analysis/adjudicate_critic_first_bridge_v1.py`、`p0_adjudicate.py`、`adjudicate_admission_core_v1.py` |
| 审计 | `analysis/audit_hurdle_speedup_dose_v1.py`、`audit_admission_checkpoint.py`、`verify_admission_*.py`、`task_progress_audit.py`、`stability_deconfounded_audit.py` |
| bank 生成 | `build_{expanded,safe_bootstrap,std9,bigsrc,stability_audit}_banks.py` |
| 探针 | `probe_{transfer_map_v2,hb_task_layouts,active_recovery,fall_recovery,hurdle_to_stair,hurdle_source_ceiling,stage_conditioned_components}_*.py`、`analysis/{per_state_qswitch_probe,influence_gate_v1_probe,label_identifiability_audit,bottleneck_aligned_coverage,extract_task_taxonomy}_v1.py` |
| 编排 | `p0_orchestrator.py`、`orchestrate_*.sh`、`drive_racing_min_horizon_v1.sh` |

### A.8 测试（`tests/`，28 文件 / 223 个测试函数）

按覆盖对象：`test_{option_module,option_selector,mcg,qmp,adapters,source_policy,core}.py`（机制层）、
`test_{ptf_replay_snapshot,replay_channel_decoupling,admission_control,source_admission,anchor_io,anchor_core_resume,branch_anchor_controls,rng_isolation}.py`（训练入口层）、
`test_{p0_adjudicate,p0_orchestrator,p1_freeze,p1_gate_a,analyze_*,adjudicate_*}.py`（裁决层）、
`test_humanoid_bench_seed.py`（播种正确性）。

### A.9 产物目录

| 目录 | 内容 | 进 git？ |
|---|---|---|
| `models/` (22G) | 训练 checkpoint `{env}__{exp}__{seed}_{step}.pt` | 否 |
| `checkpoints/` (14G) | 源策略 + manifest（`official_sources/` / `terrain_sources/` / `sources/`）| manifest 进，`.pt` 不进 |
| `artifacts/` (68G) | anchor bundle + 审计证据链 | json/md 进 |
| `logs/` (2.6G) | 训练日志 | 否 |
| `wandb/` (34G) | wandb 运行记录 | 否 |
| `docs/data/{experiment}/` | **评估 JSON + 裁决 results.json** | **是**（证据链）|

### A.10 文档

| 文件 | 内容 |
|---|---|
| `CLAUDE.md` | **强制执行规范**（唯一强制加载点，§0–§8）|
| `docs/REPO_MAP.md` | 仓库结构地图 |
| `docs/RESEARCH_ROADMAP.md` | 科研路线（时间线 + 六组件坐标）|
| `docs/EXPERIMENT_LOG.md` | 全实验总表 + 机制栈 + 代号字典 |
| `docs/ISSUES_AND_LESSONS.md` | 问题记录 + 方法论教训（M1–M32）|
| `docs/EVIDENCE_STATE_20260731.md` | **证据状态总表**（已裁决 / 边界 / 缺口）|
| `docs/experiments/*_prereg_*.md` / `*_results_*.md` | 预注册与结果（成对）|
| `docs/agent_collab/` | Claude/ChatGPT 协作记录 |
| `docs/archive/` | 历史文档（旧引用 `docs/X.md` → `docs/archive/X.md`）|

---

## 附录 B｜数据流一图流

```
                      configs/source_banks/*.yaml
                               │ SourcePolicyBank.from_config
                               ▼
checkpoints/official_sources/*/manifest.json ──► SourcePolicy(冻结 actor + 冻结 normalizer)
                                                        │ act(raw_obs)
                                                        ▼
assets/envs/*.xml ─► HumanoidEnv ─► SubprocVecEnv(128) ─► obs[128,151]
   (robots/ + tasks/ + common/)         │                    │
                                        │                    ├─► normalize_obs ─► Actor ─► a_student[128,61]
                                        │                    │                              │
                                        │                    │      McgBehaviorController ◄─┤
                                        │                    │      (admission_bootstrap    │
                                        │                    │       抽 arm，锁存 25 步)     │
                                        │                    │            │                 │
                                        │◄─── actions[128,61] ◄───────────┘                 │
                                        │                                                    │
                                        ▼                                                    │
                          reward / done / next_obs                                           │
                                        │                                                    │
                                        ▼                                                    │
                    PTFReplayWrapper.extend(transition, option_ids, provenance)               │
                                        │                                                     │
                                        │ sample(按 admission 配额加权)                        │
                                        ▼                                                     │
                    ┌───────────────────┴────────────────────┐                                │
                    ▼                                        ▼                                │
            update_main (Critic)                      update_pol (Actor) ─────────────────────┘
      分布式 TD + CDQ + target smoothing        −min Q(s,π(s)) + λ(t)·distill_loss
                    │                                        │
                    └──────────► soft_update(τ=0.1) ◄────────┘
                                        │
                        PTF_EVAL_CHECKPOINT_STEPS 命中
                                        ▼
                            models/{run}_{step}.pt
                                        │
                                        ▼
                    scripts/p0_evaluator.py（128 ep，结构性 source-free）
                                        ▼
                    docs/data/{exp}/source_free_eval/*.json
                                        ▼
                    scripts/analysis/analyze_*_v1.py（判据已冻结）
                                        ▼
                              results.json → VERDICT
```

---

## 附录 C｜易踩的六个坑（都已实际发生过）

1. **`success_count` 在 locomotion 上是「摔倒」不是「成功」**——它读 `terminated`，
   而 locomotion 的 `get_terminated()` 是摔倒早停判据。引用前先确认当前 target 的语义。
2. **LR 日程实际是常数**——`eta_min == base_lr`，余弦退火恒定。
   不要凭 `T_max=total_timesteps` 就断言分支训练压缩了日程。
3. **`eval_envs = envs`**——HumanoidBench 分支下 in-loop 评估会动训练环境，
   所以一律 `EVAL_INTERVAL=0` + 离线 `p0_evaluator.py`。
4. **`source_names` 的格式随 `null_option` 变**——hurdle `['run','null']` vs
   slide `['walk']`。审计判据要写「非 null 部分 == [arm]」。
5. **replay 存 raw obs，源用 raw、学生用 normalized**——
   split replay 首版漏了 `normalize_obs`，让 actor 在 raw obs 上更新，整轮报废。
6. **`[[ -f "$OUT" ]] && skip` 不是原子锁**——并行评估前必须先验证
   各链输出文件集合两两不相交。

---

## 附录 D｜当前科研状态一句话定位

- **已成立**：hurdle 早期加速 3.5–4.4×（`SPEEDUP_CONFIRMED`）；
  30k 步交互可自动选对源（`RACING_VIABLE`，K\*=10000）；
  slide 选源决策在 6 个 learner 上可推广（`GEN_OK`）；
  hurdle 与 slide 在**同一候选集合**上 argmax 反转（真正的 crossover）。
- **已否定**：零成本迁移性预测不可行（十二个信号族、七个信号空间全败）；
  learned 自适应调度劣于固定 schedule。
- **主要缺口**：跨任务加速只有 hurdle 一例（slide 已预注册未跑）；
  「选对源 > 选错源」从未验证；door 判决场已关闭（ground truth 本身不跨 learner 稳定）。

详见 `docs/EVIDENCE_STATE_20260731.md`。
