"""端到端系统的决策环：把 P1 的 racing 测量转成每个 (target, seed) 的执行决策。

这一步是**自动的**：输入只有 `racing_admission_v1/results.json`（P1 的冻结裁决输出），
输出是主训练该怎么跑。它必须在任何主训练启动之前运行并提交 git，
使决策可审计、不可事后调整。

决策规则（与 P1 预注册 §3 同一套，不引入任何新参数）：

    admit(T,s) = ∃i: U_i > 2·SE_i
    若 admit  → source = argmax_i U_i，主训练带该源，30k 硬退出
    若 reject → 纯 student 训练（不带源）

用法: PYTHONPATH=. python scripts/analysis/decide_racing_admission_v1.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

SRC = Path("docs/data/racing_admission_v1/results.json")
OUT = Path("docs/data/endtoend_v1/decisions.json")

# racing 的交互成本：4 臂(3 源 + student) × K，全部计入端到端的总预算
K = 10000
N_ARMS = 4
EXIT_STEP = 30000          # 剂量退出点，沿用 slide_hard_exit_v1 已验证的协议


def main() -> int:
    if not SRC.exists():
        print(f"INCOMPLETE: missing {SRC}")
        return 2
    blob = json.loads(SRC.read_text())
    if blob.get("verdict") != "ADMISSION_VIABLE":
        print(f"REFUSE: upstream verdict={blob.get('verdict')}, 端到端决策环只在 ADMISSION_VIABLE 下有意义")
        return 3

    decisions = {}
    for target, rows in blob["per_target"].items():
        for r in rows:
            seed = r["seed"]
            if r["admit"]:
                # argmax 只在 admit 时才有意义（P1 §5.1：全负时 argmax 是噪声）
                src = max(r["U"], key=r["U"].get)
                d = {"admit": True, "source": src, "exit_step": EXIT_STEP,
                     "U": r["U"][src], "SE": r["SE"][src]}
            else:
                d = {"admit": False, "source": None, "exit_step": None,
                     "U": max(r["U"].values()), "SE": None}
            d["racing_cost_steps"] = N_ARMS * K
            decisions[f"{target}|s{seed}"] = d

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "source": str(SRC),
        "upstream_verdict": blob["verdict"],
        "rule": "admit = ∃i U_i > 2·SE_i ; source = argmax_i U_i ; exit at 30k",
        "racing_cost_steps_per_run": N_ARMS * K,
        "decisions": decisions,
    }, indent=2, ensure_ascii=False))

    print("=" * 72)
    print("端到端决策（自动导出，先于任何主训练）")
    print("=" * 72)
    for k, d in decisions.items():
        if d["admit"]:
            print(f"  {k:12s}  ADMIT   source={d['source']:6s} "
                  f"U={d['U']:+8.2f}±{d['SE']:.2f}  退出@{d['exit_step']}")
        else:
            print(f"  {k:12s}  REJECT  纯 student（最强源 U={d['U']:+8.2f} 未过阈）")
    print(f"\n每次运行的 racing 成本 = {N_ARMS} 臂 × {K} = {N_ARMS*K} 步")
    print(f"\nwritten {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
