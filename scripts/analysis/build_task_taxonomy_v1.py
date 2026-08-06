"""任务分类学 v1：合成 F1–F6 → 任务族划分 / source-bank coverage / held-out splits。

阶段一产物（预注册 task_taxonomy_v1_prereg_20260729.md §4）。
**不读取任何 U 标签**——U 的外部投影是阶段二，须在本文件提交后单独进行。
"""
from __future__ import annotations

import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
F1F5 = json.loads((REPO / "docs/data/task_taxonomy_v1_f1f5.json").read_text())
F2F3F6 = json.loads((REPO / "docs/data/task_taxonomy_v1_f2f3f6.json").read_text())
UNKNOWN = "unknown"

SOURCE_TASKS = ("stand", "walk", "run")   # **本轮评估的 loco bank**，非项目全部 source 资产
# 注：项目另有 checkpoints/terrain_sources/{slide,stair,crawl,hurdle,pole} 等非 loco 冻结源，
#     覆盖结论仅针对上面这三个，不得外推为"项目没有非 loco 源"。

# F4：物理量 → 目标变量类型。规则机械且冻结（见预注册 F4）。
F4_RULES = [
    (r"actuator_forces",                              "control_effort"),
    (r"center_of_mass_velocity|com_velocity|horizontal_velocity", "velocity"),
    (r"torso_upright|head_height|xquat|qpos\[2\]|site_xpos\['imu'", "posture"),
    (r"distance|_dist|proximity|linalg\.norm",        "goal_or_object_distance"),
    (r"site_xpos|xpos|qpos",                          "object_state"),
]


def f4_types(task: str) -> dict:
    fps = F1F5[task]["F1"].get("component_fingerprints", {})
    hits: dict[str, int] = {}
    for fp in fps.values():
        for term in fp.get("terms", []):
            q = term.get("quantity", "")
            for pat, label in F4_RULES:
                if re.search(pat, q):
                    hits[label] = hits.get(label, 0) + 1
                    break
    kind = F1F5[task]["F1"]["composition"]["kind"]
    if kind in ("event_count", "event_dominated"):
        hits["sparse_completion_event"] = hits.get("sparse_completion_event", 0) + 1
    task_types = {k: v for k, v in hits.items() if k != "control_effort"}
    return {"all": hits, "task_specific": task_types or {UNKNOWN: 1}}


def build() -> dict:
    out = {}
    for t in F1F5:
        f1 = F1F5[t]["F1"]
        sc = F2F3F6[t]
        out[t] = {
            "reward_owner": f1["reward_owner"],
            "composition_kind": f1["composition"]["kind"],
            "uses_min": f1["composition"].get("uses_min", False),
            "class_constants": f1.get("class_constants", {}),
            "terrain_class": sc["F2_terrain_class"],
            "n_freejoint": sc["F2_F3_scene"]["n_freejoint"],
            "articulated": bool(sc["F2_F3_scene"]["joint_types"]),
            "joint_types": sc["F2_F3_scene"]["joint_types"],
            "F4": f4_types(t),
            "termination": F1F5[t]["F5"].get("text", UNKNOWN)[:120],
            "evidence": {
                "reward": f"{f1['reward_owner']}:{f1.get('reward_owner_line')}",
                "scene": sc["F2_F3_scene"].get("scene_file"),
                "env": sc["F6_robot"].get("env_file"),
            },
        }
    return out


def families(tax: dict) -> dict:
    def group(keyfn):
        g: dict[str, list] = {}
        for t, v in tax.items():
            g.setdefault(keyfn(v), []).append(t)
        return dict(sorted(g.items(), key=lambda kv: -len(kv[1])))

    return {
        # 注意：这是"**完全相同的 reward 实现**"族，不是语义任务相似度。
        "by_exact_reward_implementation": group(lambda v: v["reward_owner"]),
        "by_terrain": group(lambda v: v["terrain_class"]),
        "by_composition": group(lambda v: v["composition_kind"]),
        "by_object_or_articulation_present": group(
            lambda v: "has_object_or_articulation" if (v["n_freejoint"] > 0 or v["articulated"])
            else "no_free_object"),
    }


def source_bank_coverage(tax: dict, fam: dict) -> dict:
    """现有 stand/walk/run bank 在各族上的覆盖情况（纯结构，不涉及效用）。"""
    cov = {}
    for dim, groups in fam.items():
        src_groups = {g for g, ts in groups.items() if any(s in ts for s in SOURCE_TASKS)}
        covered, uncovered = [], []
        for g, ts in groups.items():
            targets = [t for t in ts if t not in SOURCE_TASKS]
            if not targets:
                continue
            (covered if g in src_groups else uncovered).append({g: targets})
        cov[dim] = {"source_groups": sorted(src_groups),
                    "targets_in_source_groups": covered,
                    "targets_in_uncovered_groups": uncovered}
    return cov


def held_out_splits(fam: dict) -> dict:
    """三种评估划分：同族任务不得同时出现在 train 与 test 两侧。"""
    out = {}
    for name, dim in (("exact_reward_implementation_held_out", "by_exact_reward_implementation"),
                      ("terrain_family_held_out", "by_terrain"),
                      ("object_articulation_held_out", "by_object_or_articulation_present")):
        groups = {g: [t for t in ts if t not in SOURCE_TASKS]
                  for g, ts in fam[dim].items()}
        groups = {g: ts for g, ts in groups.items() if ts}
        out[name] = {"groups": groups,
                     "rule": "留出整个族作为 test；同族任务不得跨 train/test 分割"}
    return out


def main() -> None:
    tax = build()
    fam = families(tax)
    cov = source_bank_coverage(tax, fam)
    spl = held_out_splits(fam)

    dst = REPO / "docs/data/task_taxonomy_v1.json"
    dst.write_text(json.dumps({"taxonomy": tax, "families": fam,
                               "source_bank_coverage": cov,
                               "held_out_splits": spl}, indent=1, ensure_ascii=False))

    print("=== 任务特征表（32 任务）===")
    print(f"{'task':17s} {'reward owner':32s} {'kind':18s} {'terrain':24s} {'free':>4s} 目标变量")
    print("-" * 130)
    for t, v in tax.items():
        tt = ",".join(sorted(v["F4"]["task_specific"]))
        mark = " *" if t in SOURCE_TASKS else "  "
        print(f"{t:15s}{mark} {v['reward_owner']:32s} {v['composition_kind']:18s} "
              f"{v['terrain_class']:24s} {v['n_freejoint']:4d} {tt[:36]}")

    print("\n\n=== 任务族划分（不使用 U 标签）===")
    for dim, groups in fam.items():
        print(f"\n[{dim}]")
        for g, ts in groups.items():
            print(f"  {g:34s} ({len(ts):2d}) {', '.join(ts)}")

    print("\n\n=== source bank (stand/walk/run) 结构性覆盖 ===")
    for dim, c in cov.items():
        print(f"\n[{dim}]  源所在族: {c['source_groups']}")
        n_in = sum(len(list(d.values())[0]) for d in c["targets_in_source_groups"])
        n_out = sum(len(list(d.values())[0]) for d in c["targets_in_uncovered_groups"])
        print(f"  同族 target {n_in} 个 / 异族 target {n_out} 个")
        for d in c["targets_in_uncovered_groups"]:
            for g, ts in d.items():
                print(f"    未覆盖族 {g:30s} {', '.join(ts)}")
    print(f"\nsaved: {dst}")


if __name__ == "__main__":
    main()
