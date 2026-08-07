# 本目录的 `smoke.json`：`DIAGNOSTIC_PASS / REVERIFY_REQUIRED`

2026-08-07。

`smoke.json` 是 2026-08-06 P1.1 首轮 smoke 的**原始运行输出**，原样保留、未作任何修改
（改它会破坏"原始输出"的意义）。状态词含义：`DIAGNOSTIC_PASS` = 本轮确实发现并修复了真实缺陷（basketball
提取路径），该诊断价值保留；`REVERIFY_REQUIRED` = 但验证效力不成立，
结论须由一次干净的重验证重新建立。**不得再声称原三段式链有效。**

它不构成验证证据，理由有二：

1. **产出它的 commit 违反三段式。** `19948c4`（publish `691dcff`）名义是"结果提交"，
   实际同时修改了 `smoke_evaluator_v21.py` 与 `p0_evaluator_v2.py`。
   basketball 修复、S2 换 checkpoint、`VACUOUS` 判定都是**看到首轮结果之后**
   在结果 commit 里加进去的，实现与结果之间没有边界。
2. **文件里的 `"verdict": "ALL_PASS"` 作废。** 它与同一份输出中 S4 的 `VACUOUS` 自相矛盾。
   正确表述是"核心路径基本可用，bookshelf runtime termination path 未验证"。

另外，产出本文件的实现有五处缺陷（D1–D5），见
`docs/experiments/evaluator_v21_hardening_results_20260806.md` §9。
其中 D3（milestone 只读最后一步）会**真实丢失数据**，因此本目录里
任何与 milestone 相关的数字都不可信。

**请引用 P1.1b 的产物**：`docs/data/evaluator_v21b_smoke/`，
预注册 `docs/experiments/evaluator_v21b_prereg_20260807.md`。
