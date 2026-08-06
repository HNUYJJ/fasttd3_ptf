# 结果：Evaluator schema v2 —— P1 完成（36/36 通过）

> 2026-08-06。预注册 `docs/experiments/evaluator_schema_v2_prereg_20260806.md`。
> 本文件是 P1 的**结果** commit，不含任何代码改动（三段式提交）。

## 1. 裁决

```
P1_PASSED   36 passed, 0 failed
  T1–T3, T5–T10  35 项纯逻辑测试（无 GPU / 无 MuJoCo，0.1 秒）
  T4             1 项集成测试（真实 checkpoint + MuJoCo，42 秒）
```

按目标 P1 的停止条件「任何测试失败，不进入 P2」——**全部通过，可进入 P2**。

## 2. T4 的实证结果：v1 缺陷不再是推断

此前 v1 的 `if terminated: success = True` 缺陷只有源码证据。
T4 用真实 checkpoint 跑出了数据证据：

```
checkpoint  models/h1hand-slide-v0__slide_bac_walk_s1__1_20000.pt
env         h1hand-slide-v0    episodes 8    device cpu

数值逐位一致    return / progress_max_dx / reset seed 全部相同
语义变化        5 / 8 例，全部 True → False
  seed=11000  11002  11005  11006  11007
```

**slide 上 8 个 episode 有 5 个（62.5%）因 `torso_upright < 0.1` 终止（摔倒），
v1 把它们全部记为 success。**

T4 同时证明数值路径未被破坏——这条测试防的正是改口径导致数字变化被误当成修好了 bug。

## 3. 完整测试输出

```
tests/test_evaluator_schema_v2.py::test_T1_locomotion_fall_is_not_success[h1hand-walk-v0] PASSED [  2%]
tests/test_evaluator_schema_v2.py::test_T1_locomotion_fall_is_not_success[h1hand-run-v0] PASSED [  5%]
tests/test_evaluator_schema_v2.py::test_T1_locomotion_fall_is_not_success[h1hand-hurdle-v0] PASSED [  8%]
tests/test_evaluator_schema_v2.py::test_T1_locomotion_fall_is_not_success[h1hand-slide-v0] PASSED [ 11%]
tests/test_evaluator_schema_v2.py::test_T1_locomotion_fall_is_not_success[h1hand-stair-v0] PASSED [ 13%]
tests/test_evaluator_schema_v2.py::test_T1_locomotion_fall_is_not_success[h1hand-powerlift-v0] PASSED [ 16%]
tests/test_evaluator_schema_v2.py::test_T1b_truncation_is_not_success PASSED [ 19%]
tests/test_evaluator_schema_v2.py::test_T1c_crawl_never_terminates PASSED [ 22%]
tests/test_evaluator_schema_v2.py::test_T1d_bookshelf_termination_is_conditional PASSED [ 25%]
tests/test_evaluator_schema_v2.py::test_T1f_conditional_task_not_terminated_is_neutral_not_unknown PASSED [ 27%]
tests/test_evaluator_schema_v2.py::test_T1g_unterminated_locomotion_is_neutral_not_failure PASSED [ 30%]
tests/test_evaluator_schema_v2.py::test_T1h_milestones_extracted_even_when_not_terminated PASSED [ 33%]
tests/test_evaluator_schema_v2.py::test_T1e_basketball_needs_state_else_null PASSED [ 36%]
tests/test_evaluator_schema_v2.py::test_T2_manipulation_success_detected PASSED [ 38%]
tests/test_evaluator_schema_v2.py::test_T2b_manipulation_non_success_termination PASSED [ 41%]
tests/test_evaluator_schema_v2.py::test_T3_unregistered_task_yields_null PASSED [ 44%]
tests/test_evaluator_schema_v2.py::test_T3b_null_is_distinct_from_false PASSED [ 47%]
tests/test_evaluator_schema_v2.py::test_T5_missing_milestone_never_saturated PASSED [ 50%]
tests/test_evaluator_schema_v2.py::test_T5b_saturated_requires_both PASSED [ 52%]
tests/test_evaluator_schema_v2.py::test_T6_no_ceiling_no_percentage PASSED [ 55%]
tests/test_evaluator_schema_v2.py::test_T6b_ceiling_present_computes PASSED [ 58%]
tests/test_evaluator_schema_v2.py::test_T7_cross_budget_comparison_rejected PASSED [ 61%]
tests/test_evaluator_schema_v2.py::test_T7b_same_step_allowed PASSED     [ 63%]
tests/test_evaluator_schema_v2.py::test_T7c_missing_step_is_incomparable PASSED [ 66%]
tests/test_evaluator_schema_v2.py::test_T8_single_seed_not_robustly_solved PASSED [ 69%]
tests/test_evaluator_schema_v2.py::test_T8b_two_seeds_still_insufficient PASSED [ 72%]
tests/test_evaluator_schema_v2.py::test_T8c_three_seeds_all_above_bar PASSED [ 75%]
tests/test_evaluator_schema_v2.py::test_T8d_three_seeds_one_below_bar PASSED [ 77%]
tests/test_evaluator_schema_v2.py::test_T9_no_hard_exit_arm_no_deficit_verdict PASSED [ 80%]
tests/test_evaluator_schema_v2.py::test_T9b_hard_exit_present_yields_verdict PASSED [ 83%]
tests/test_evaluator_schema_v2.py::test_T10_required_field_unparseable_fails_closed PASSED [ 86%]
tests/test_evaluator_schema_v2.py::test_T10b_unregistered_nonscalar_recorded_not_dropped PASSED [ 88%]
tests/test_evaluator_schema_v2.py::test_T10c_required_nonscalar_parsed_correctly PASSED [ 91%]
tests/test_evaluator_schema_v2.py::test_T10d_nan_inf_do_not_crash PASSED [ 94%]
tests/test_evaluator_schema_v2.py::test_T10e_time_varying_keys_handled PASSED [ 97%]
tests/test_evaluator_schema_v2.py::test_T4_v1_v2_bitwise_identical_on_shared_fields PASSED [100%]
============================= 36 passed in 43.36s ==============================
```

## 4. Review packet

### 4.1 提交链（三段式）

| 角色 | commit | 内容 |
|---|---|---|
| 预注册 | `751bc89` | 契约修订 + T1–T10 冻结（**先于实现**） |
| 规格修订 | `3b22f5a` | 逐任务核实终止语义，契约放宽为 `str \| Callable` |
| 实现 | `d97ea9a` | 三个纯函数模块 |
| 实现 | `c31cc8d` | evaluator 主流程 + T4 通过 |
| 结果 | 本 commit | 仅文档，无代码改动 |

### 4.2 复现命令

```bash
PY=/home/yjj/miniconda3/envs/FastTD3/bin/python
export EVAL_V2_INTEGRATION_CKPT=models/h1hand-slide-v0__slide_bac_walk_s1__1_20000.pt
export EVAL_V2_INTEGRATION_ENV=h1hand-slide-v0
export EVAL_V2_INTEGRATION_N=8 EVAL_V2_INTEGRATION_DEVICE=cpu
PYTHONPATH=. $PY -m pytest tests/test_evaluator_schema_v2.py -v
```

不设 `EVAL_V2_INTEGRATION_CKPT` 时 T4 自动 skip，其余 35 项仍在 0.1 秒内跑完。

### 4.3 artifact SHA256

```
5012ef9ac471af945b52a8bbfea6fbddc6647500eb4ebbcf4d70ac7e036a1b1f  fasttd3_ptf/evaluation/task_metrics.py
96e55f560d5aefbfc0a81dafcdc57deae9a1733c358501d488d06bd425fb20af  fasttd3_ptf/evaluation/schema_v2.py
0b84f12bd43fa397196af33e819b5abe99c8113383a3a07ca724ebe14c0e4b11  fasttd3_ptf/evaluation/site_rules.py
27d481c5db710325689bc33568113306ffa73cc4a84c0761e5d5b10eab857543  scripts/p0_evaluator_v2.py
bc1983a9a0b3f7da9635a911847856d1acd533c2a6fd4ab430a29971a9f51630  tests/test_evaluator_schema_v2.py
cf0a52a233ab875bbd485340f811523da62aeadcaa72ec8f6456e217b8681ca9  models/h1hand-slide-v0__slide_bac_walk_s1__1_20000.pt
```

### 4.4 git 状态

```
branch   main
HEAD     c31cc8d13f4e47c3bd93e7c47b15dfb850a8caa7
status   1 项未提交
remote   未推送（见 §5 已知限制）
```

## 5. 已知限制

1. **未推送**。原仓库已删、本地无 gh/token/credential，且旧历史含 2.4GB blob
   超 GitHub 100MB 硬限。干净仓库已备于 `/home/yjj/fasttd3_ptf_publish`（23MB）。
   目标 P0 要求外部 reviewer 无法访问提交时不得继续正式实验——
   P1 是写代码与测试，不产生实验数据，故判定可继续；**P2 的重评产生正式结果，
   届时推送必须已完成**。
2. **T4 只覆盖 1 个 checkpoint、1 个任务、8 episodes**。逐位一致在 slide 上成立，
   未在 manipulation 任务上验证（那里 `terminated` 语义相反，数值路径相同但
   值得单独确认）。P2 重评时对每个 task family 各跑一次 smoke 可覆盖。
3. **registry 只注册了 15 个任务**，均已逐条读过 `get_terminated` 源码。
   未注册任务返回 `task_success=null` + `UNREGISTERED`——这是 fail-closed 的
   正确行为，但意味着它们暂时无法参与任何 milestone 判据。
4. **basketball 的 `ball_to_hoop_dist` 提取路径未经真实运行验证**
   （`_basketball_state` 用 try/except 兜底返回 None → INSUFFICIENT_STATE）。
   fail-closed 方向正确，但 P2 首次评估 basketball 时须确认它不是恒为 None。
5. **v1 evaluator 未删除**。两份并存期间，任何新结果必须显式标注用的是哪个
   schema_version；`terminated_success` 与 `task_success` 不得混用。
