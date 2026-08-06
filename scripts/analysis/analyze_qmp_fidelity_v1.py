"""QMP-fidelity v1 裁决脚本。

判据冻结于 run card docs/run_card_qmp_fidelity_v1.md §6(提交 1648b9c),
**先于任何 QMP 臂被评估**。本脚本只实现,不得事后调整。

外部审查(2026-07-29)否决了 v1 的 "90% CI 上界 > 0 = 非劣效" 判据:
那只能说明"无法排除有益",一个均值严重为负但方差很大的结果也会通过。
改为不引入新 δ 的方向 gate。**CI 照常报告,但不用于声称非劣效。**

    Slide(正例):  A  = mean[J_qmp − J_student] > 0  且 3/3 seed 为正
    Door (负例):  B1 = mean[J_qmp − J_walk]    > 0  且 3/3 seed 为正
                  B2 = J_qmp − J_student 在 3/3 seed 上非负  ← 才配称负迁移免疫

    A ∧ B1 ∧ B2                      → QMP_FIDELITY_SUPPORTED  (才解禁身体组)
    Door 过 B1 不过 B2,或仅一侧通过  → QMP_FIDELITY_PARTIAL    (不解禁)
    A 与 B1 均不成立                 → QMP_FIDELITY_REFUTED    (整线停止)
"""
from __future__ import annotations

import json
import statistics as st
from pathlib import Path

SEEDS = (1, 2, 3)
HIST = {
    "door": "docs/data/door_at10k_gate_v1/source_free_eval",
    "slide": "docs/data/slide_bac_gate_v1/source_free_eval",
}
QMP_ROOT = Path("docs/data/qmp_fidelity_v1/source_free_eval")
OUT = Path("docs/data/qmp_fidelity_v1/qmp_fidelity_v1_results.json")
# t 分布 90% 双侧, df=2
T90_DF2 = 2.920


def _read(path: Path) -> float:
    return float(json.loads(path.read_text())["aggregate"]["return_mean"])


def load_arm(task: str, arm: str) -> list[float]:
    root = Path(HIST[task])
    return [_read(root / f"{arm}_s{s}_step20000.json") for s in SEEDS]


def load_qmp(task: str) -> list[float]:
    return [_read(QMP_ROOT / f"qmp_{task}_s{s}_step20000.json") for s in SEEDS]


def paired(a: list[float], b: list[float]) -> dict:
    d = [x - y for x, y in zip(a, b)]
    mean = st.mean(d)
    if len(d) > 1:
        se = st.stdev(d) / len(d) ** 0.5
        half = T90_DF2 * se
    else:
        se = half = float("nan")
    return {
        "per_seed": [round(x, 4) for x in d],
        "mean": round(mean, 4),
        "se": round(se, 4),
        "ci90": [round(mean - half, 4), round(mean + half, 4)],
        "n_pos": sum(1 for x in d if x > 0),
        "n_nonneg": sum(1 for x in d if x >= 0),
    }


def main() -> None:
    res: dict = {
        "run_card": "docs/run_card_qmp_fidelity_v1.md",
        "run_card_commit": "1648b9c",
        "note": "CI 报告但不用于声称非劣效(外部审查 2026-07-29 否决该用法)",
        "arms": {},
        "criteria": {},
    }

    # ---- Slide: 判据 A ----
    slide_qmp = load_qmp("slide")
    slide_stu = load_arm("slide", "student")
    slide_walk = load_arm("slide", "walk")
    A = paired(slide_qmp, slide_stu)
    res["arms"]["slide"] = {
        "qmp": [round(x, 2) for x in slide_qmp],
        "student": [round(x, 2) for x in slide_stu],
        "walk": [round(x, 2) for x in slide_walk],
    }
    res["criteria"]["A_slide_qmp_vs_student"] = A
    res["criteria"]["slide_qmp_vs_walk_reference"] = paired(slide_qmp, slide_walk)
    A_pass = A["mean"] > 0 and A["n_pos"] == 3

    # ---- Door: 判据 B1 / B2 ----
    door_qmp = load_qmp("door")
    door_stu = load_arm("door", "student")
    door_walk = load_arm("door", "walk")
    B1 = paired(door_qmp, door_walk)
    B2 = paired(door_qmp, door_stu)
    res["arms"]["door"] = {
        "qmp": [round(x, 2) for x in door_qmp],
        "student": [round(x, 2) for x in door_stu],
        "walk": [round(x, 2) for x in door_walk],
        "stand": [round(x, 2) for x in load_arm("door", "stand")],
        "run": [round(x, 2) for x in load_arm("door", "run")],
    }
    res["criteria"]["B1_door_qmp_vs_walk"] = B1
    res["criteria"]["B2_door_qmp_vs_student"] = B2
    B1_pass = B1["mean"] > 0 and B1["n_pos"] == 3
    B2_pass = B2["n_nonneg"] == 3

    if A_pass and B1_pass and B2_pass:
        verdict = "QMP_FIDELITY_SUPPORTED"
    elif not A_pass and not B1_pass:
        verdict = "QMP_FIDELITY_REFUTED"
    else:
        verdict = "QMP_FIDELITY_PARTIAL"

    res["pass"] = {"A": A_pass, "B1": B1_pass, "B2": B2_pass}
    res["verdict"] = verdict
    res["unlocks_body_groups"] = verdict == "QMP_FIDELITY_SUPPORTED"

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2, ensure_ascii=False))

    print("=" * 84)
    print(f"VERDICT: {verdict}   (解禁身体组: {res['unlocks_body_groups']})")
    print("=" * 84)
    for task in ("slide", "door"):
        print(f"\n[{task}] J_sf@20k per seed")
        for arm, vals in res["arms"][task].items():
            print(f"   {arm:9s} {vals}")
    print(f"\nA  slide qmp−student  mean={A['mean']:+8.2f} ci90={A['ci90']} "
          f"pos={A['n_pos']}/3  → {'PASS' if A_pass else 'FAIL'}")
    print(f"B1 door  qmp−walk     mean={B1['mean']:+8.2f} ci90={B1['ci90']} "
          f"pos={B1['n_pos']}/3  → {'PASS' if B1_pass else 'FAIL'}")
    print(f"B2 door  qmp−student  mean={B2['mean']:+8.2f} ci90={B2['ci90']} "
          f"nonneg={B2['n_nonneg']}/3  → {'PASS' if B2_pass else 'FAIL'}")
    print(f"\nwritten {OUT}")


if __name__ == "__main__":
    main()
