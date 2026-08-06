# QMP-fidelity v1 运行 provenance

代码提交: be7cedfc3c7e74b40a6d1f61dcd3e0c7954aa056
生成时间: 2026-07-29T14:25:30Z

## 运行时工作树中仍未提交的文件(E17 要求的脏树记录)

这些改动**先于**本轮工作已存在于工作树,且 sibling gate 等既有实验即在其上运行。
为与历史对照臂保持可比,本轮不 revert 它们。逐个论证其与 QMP 路径无关:

FILE                                                       SHA256
fasttd3_ptf/official_fasttd3_ptf/admission_control.py      3a387a1bd3079779
fasttd3_ptf/official_fasttd3_ptf/ptf_replay.py             7b1b00abb9726d0b
fasttd3_ptf/ptf/compatibility.py                           5e1369322cc2e615
fasttd3_ptf/ptf/mcg.py                                     b5f518a35665e5cf
fasttd3_ptf/ptf/option_module.py                           e7c7f876105d8b7d
fasttd3_ptf/ptf/option_selector.py                         3b9b3e5cf66672e9
fasttd3_ptf/ptf/option_update.py                           30aa5eff6fba8920
fasttd3_ptf/utils/schedules.py                             d1c3c765f01f29e7
tests/test_admission_control.py                            f803c8f48763a589
tests/test_option_module.py                                e5bdfb51e9bb4b90
tests/test_option_selector.py                              2969efac22f7cb29
tests/test_ptf_replay_snapshot.py                          6a25b0e4fc4762c2

### 与 QMP 路径的无关性论证

| 文件 | 组件 | QMP 是否走该路径 |
|---|---|---|
| `option_module.py` / `option_selector.py` / `option_update.py` / `schedules.py` | classic PTF 的 Q_ω/β/OptionSelector/λ 调度 | **否**——`isolate_classic_ptf` 跳过；option/beta optimizer state 实测为空 |
| `compatibility.py` | classic PTF 的 option 兼容性权重 | **否**——同上 |
| `mcg.py` | MCG 身体组 gating | **否**——启动断言 `mcg_enabled=False` |
| `admission_control.py` | admission 调度 | **否**——启动断言 `admission_enabled=False` |
| `ptf_replay.py` | replay wrapper | **部分**——三处改动中两处为注释/错误消息；唯一功能改动 `_admission_uniform_mix * one / stratum_count` 位于 **admission 分层采样**路径，QMP 断言 admission 关闭，走 uniform sampling，不经过该行 |
| `tests/*` | 测试 | 不参与运行 |

论证的实证支撑见提交 `be7cedf`：200-step forced-student smoke 的
option/beta optimizer state 完全为空，而同 bank/anchor/步数的 classic PTF
阳性对照达到 `option: n=6 steps=[400]`。
