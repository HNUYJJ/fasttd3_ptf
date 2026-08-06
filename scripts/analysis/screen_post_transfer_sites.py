#!/usr/bin/env python3
"""迁移后自主性（post-transfer autonomy）判决场普查。

判据冻结于 docs/experiments/post_transfer_autonomy_site_screen_prereg_20260806.md
（提交 3070c78，先于本脚本）。本脚本只执行该文件的规则，不引入任何新判据。

**本普查不预测 U**，只筛选"哪个场地能检验机制"，见预注册 §0。
其输出不得被引用为任何迁移性证据。

数据不全一律输出 UNKNOWN 并非零退出（CLAUDE.md §4）。
"""

from __future__ import annotations

import ast
import glob
import json
import os
import sys
from collections import defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ENVS_DIR = os.path.join(
    REPO, "fasttd3_ptf/official_code/humanoid-bench/humanoid_bench/envs"
)
OUT_DIR = os.path.join(REPO, "docs/data/post_transfer_site_screen_v1")

# 预注册 §1 冻结的候选池。排除项与理由同预注册。
EXCLUDED = {
    "stand": "本项目的源，非 target",
    "walk": "本项目的源，非 target",
    "run": "本项目的源，非 target",
    "reach": "本项目的源，非 target",
    "door": "判决场已关闭（M31：U 符号跨 learner 反转）",
    "kitchen": "success_bar=4，语义与其余任务不可比",
}

# 预注册 §5：hurdle/slide 必须参与普查以验证规则本身，但不得入选 dev/holdout。
ROLE_LOCKED = {
    "hurdle": "早期加速与稳定性对照",
    "slide": "生命周期机制对照",
}

# env_name（h1hand-<task>-v0）→ humanoid_bench 任务类名。
TASK_TO_CLASS = {
    "hurdle": "Hurdle", "slide": "Slide", "stair": "Stair", "crawl": "Crawl",
    "sit": "Sit", "sit_hard": "SitHard", "truck": "Truck", "cabinet": "Cabinet",
    "bookshelf_simple": "BookshelfSimple", "bookshelf_hard": "BookshelfHard",
    "package": "Package", "maze": "Maze", "basketball": "Basketball",
    "balance_simple": "BalanceSimple", "balance_hard": "BalanceHard",
    "highbar_simple": "HighBarSimple", "highbar_hard": "HighBarHard",
    "pole": "Pole", "powerlift": "Powerlift", "push": "Push",
    "window": "Window", "room": "Room", "insert": "Insert",
    "spoon": "Spoon", "cube": "Cube",
}

# 预注册 §2.1 的 H_raw 取法需要判定 reward 结构，这必须逐个读源码核实。
# 只登记**已人工核实过 get_reward 全文**的任务；未核实的一律 NOT_AUDITED，
# 不得凭"看起来像 locomotion"推断（M33：采了数据却用推理代替查询）。
REWARD_STRUCTURE_AUDITED = {
    "Walk": {
        "bounded": True,
        "note": "stand_reward * small_control * move，全为 [0,1] tolerance 相乘，无稀疏加项",
        "source": "basic_locomotion_envs.py:173-214",
    },
    "ClimbingUpwards": {
        "bounded": True,
        "note": "同 Walk 结构；Slide/Stair 均继承此实现（原理性反例 A 的基础）",
        "source": "basic_locomotion_envs.py:172-218",
    },
    "Hurdle": {
        "bounded": True,
        "note": "stand_reward * small_control * move * wall_collision_discount，折扣项 ≤ 1",
        "source": "basic_locomotion_envs.py:232-275",
    },
}

# G1/G3 依赖的既有裁决。每条必须给出**已冻结**的出处文件；本脚本不重新裁决。
# 未列出的 target 一律 UNKNOWN——不得凭推理补齐（M33）。
PRIOR_VERDICTS = {
    "hurdle": {
        "g1_scaffold": True,
        "g1_source": "docs/experiments/hurdle_speedup_v1_results_20260730.md",
        "g1_note": "早期 3.5-4.4x，θ=200/300 各 3/3 seed，CONFIRMED",
        "g3_source_solves": None,
        "g3_source": None,
    },
    "slide": {
        "g1_scaffold": True,
        "g1_source": "docs/experiments/slide_speedup_v1_results_20260804.md",
        "g1_note": "10k-30k 优于 scratch；但 100k SPEEDUP_REFUTED（被反超 2.7x）",
        "g3_source_solves": None,
        "g3_source": None,
    },
    "crawl": {
        "g1_scaffold": False,
        "g1_source": "docs/experiments/racing_admission_v1_results_20260804.md",
        "g1_note": "三源 9/9 显著负，racing 判全拒；无源提供前期收益",
        "g3_source_solves": None,
        "g3_source": None,
    },
}


def parse_success_bars() -> tuple[dict, dict, dict]:
    """从 humanoid_bench 源码解析 success_bar 与 max_episode_steps，沿类继承链回溯。

    返回 (类名 → (success_bar, 出处), 类名 → 基类名, 类名 → (steps, 出处))。
    """
    own: dict[str, tuple[int, str]] = {}
    bases: dict[str, str] = {}
    steps: dict[str, tuple[int, str]] = {}
    for path in sorted(glob.glob(os.path.join(ENVS_DIR, "*.py"))):
        try:
            tree = ast.parse(open(path, encoding="utf-8").read())
        except SyntaxError:
            continue
        rel = os.path.relpath(path, REPO)
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            if node.bases and isinstance(node.bases[0], ast.Name):
                bases[node.name] = node.bases[0].id
            for stmt in node.body:
                if (
                    isinstance(stmt, ast.Assign)
                    and len(stmt.targets) == 1
                    and isinstance(stmt.targets[0], ast.Name)
                    and isinstance(stmt.value, ast.Constant)
                ):
                    if stmt.targets[0].id == "success_bar":
                        own[node.name] = (stmt.value.value, f"{rel}:{stmt.lineno}")
                    elif stmt.targets[0].id == "max_episode_steps":
                        steps[node.name] = (stmt.value.value, f"{rel}:{stmt.lineno}")
    return own, bases, steps


def resolve_bar(cls: str, own: dict, bases: dict) -> tuple[int | None, str | None, str]:
    """沿继承链解析 success_bar；返回 (值, 出处, 定义所在类)。"""
    seen = set()
    cur = cls
    while cur and cur not in seen:
        seen.add(cur)
        if cur in own:
            val, src = own[cur]
            return val, src, cur
        cur = bases.get(cur)
    return None, None, ""


def resolve_reward_bound(cls: str, bases: dict) -> dict | None:
    """沿继承链找已人工核实的 reward 结构；未核实返回 None（不推断）。"""
    seen = set()
    cur = cls
    while cur and cur not in seen:
        seen.add(cur)
        if cur in REWARD_STRUCTURE_AUDITED:
            return {**REWARD_STRUCTURE_AUDITED[cur], "audited_at": cur}
        cur = bases.get(cur)
    return None


def collect_best_known() -> dict:
    """扫描全部 source-free 评估，按 target 取 return_mean 最大值。

    预注册 §2.1：J_best_known = 该 target 上任何 source-free 评估的最大值。
    """
    best: dict[str, dict] = {}
    counts: dict[str, int] = defaultdict(int)
    milestone_seen: dict[str, bool] = defaultdict(bool)
    for path in glob.glob(
        os.path.join(REPO, "docs/data/**/source_free_eval/*.json"), recursive=True
    ):
        try:
            d = json.load(open(path, encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        env = d.get("env_name")
        if not env:
            continue
        task = env.replace("h1hand-", "").replace("-v0", "")
        counts[task] += 1
        # milestone 信号：任务自定义字段，非 evaluator 的 success_count（CLAUDE.md §6）
        eps = d.get("episodes") or []
        if eps and any(
            k in eps[0] for k in ("success_subtasks", "task_success")
        ):
            milestone_seen[task] = True
        agg = d.get("aggregate") or {}
        rm = agg.get("return_mean")
        if rm is None:
            continue
        if task not in best or rm > best[task]["return_mean"]:
            best[task] = {
                "return_mean": float(rm),
                "source": os.path.relpath(path, REPO),
                "global_step": (d.get("checkpoint") or {}).get("global_step"),
            }
    for t in counts:
        if t in best:
            best[t]["n_eval_files"] = counts[t]
            best[t]["has_milestone_field"] = milestone_seen[t]
    return best


def screen() -> dict:
    own, bases, steps_map = parse_success_bars()
    best = collect_best_known()
    rows = []

    for task, cls in sorted(TASK_TO_CLASS.items()):
        if task in EXCLUDED:
            continue
        bar, bar_src, bar_cls = resolve_bar(cls, own, bases)
        row = {
            "target": task,
            "class": cls,
            "success_bar": bar,
            "success_bar_source": bar_src,
            "success_bar_defined_in": bar_cls,
            "gates": {},
            "notes": [],
        }
        b = best.get(task)
        row["J_best_known"] = b["return_mean"] if b else None
        row["J_best_known_source"] = b["source"] if b else None
        row["n_eval_files"] = b.get("n_eval_files") if b else 0

        # ---- H_raw：理论上限（预注册 §2.1，仅用于快速排除）----
        rb = resolve_reward_bound(cls, bases)
        ep_steps, ep_src, _ = resolve_bar(cls, steps_map, bases)
        if ep_steps is None:
            ep_steps, ep_src = 1000, "tasks.py:12 (Task 基类默认)"
        row["max_episode_steps"] = ep_steps
        row["max_episode_steps_source"] = ep_src
        if rb is None:
            row["J_theory_max"] = "NOT_AUDITED"
            row["H_raw"] = None
            row["notes"].append("reward 结构未人工核实，按预注册 §2.1 不估算理论上限")
        elif not rb["bounded"]:
            row["J_theory_max"] = "UNBOUNDED_ANALYTIC"
            row["H_raw"] = None
        else:
            row["J_theory_max"] = ep_steps * 1.0
            row["reward_structure"] = rb["note"]
            row["reward_structure_source"] = rb["source"]
            row["H_raw"] = (
                round(ep_steps - b["return_mean"], 3) if b else None
            )
            if b:
                row["pct_of_theory_max"] = round(
                    100.0 * b["return_mean"] / ep_steps, 1
                )

        # ---- milestone headroom（预注册 §2.3）：必须先于 G2 求值 ----
        # 只用任务自定义字段，禁用 evaluator 的 success_count（CLAUDE.md §6）。
        if b and b.get("has_milestone_field"):
            row["H_ms"] = "PRESENT_BUT_UNPARSED"
            row["notes"].append("检出 milestone 字段，需专门解析")
        else:
            row["H_ms"] = None
            row["notes"].append(
                "现有 source-free 评估未采集 info['success'] / ['success_subtasks']"
            )
        h_ms_known = isinstance(row["H_ms"], (int, float))

        # ---- G2：退出后仍有缺口 = (H_op > 0) 或 (H_ms > 0)（预注册 §3）----
        # 关键：H_op ≤ 0 但 H_ms 未测时**不得**判 FAIL——H_ms > 0 仍可能使 G2 通过。
        # 数据不全必须 UNKNOWN，不得落进实质裁决分支（CLAUDE.md §4）。
        if bar is None:
            row["H_op"] = None
            row["gates"]["G2"] = "UNKNOWN_G2"
            row["notes"].append("success_bar 未在继承链中定义")
        elif b is None:
            row["H_op"] = None
            row["gates"]["G2"] = "UNKNOWN_G2"
            row["notes"].append("无任何 source-free 评估数据")
        else:
            row["H_op"] = round(bar - b["return_mean"], 3)
            if row["H_op"] > 0:
                row["gates"]["G2"] = "PASS"
            elif h_ms_known and row["H_ms"] <= 0:
                row["gates"]["G2"] = "FAIL"
            else:
                row["gates"]["G2"] = "UNKNOWN_G2"
                row["notes"].append(
                    "H_op ≤ 0 但 milestone 未测；预注册 §4 的 SATURATED 需"
                    "H_op ≤ 0 **且** H_ms ≤ 0，故不得判定为已饱和"
                )

        # ---- G1：前期 scaffold 有效（只读已冻结裁决）----
        pv = PRIOR_VERDICTS.get(task)
        if pv is None or pv.get("g1_scaffold") is None:
            row["gates"]["G1"] = "UNKNOWN_G1"
        else:
            row["gates"]["G1"] = "PASS" if pv["g1_scaffold"] else "FAIL"
            row["g1_source"] = pv["g1_source"]
            row["notes"].append(pv["g1_note"])

        # ---- G3：源本身解不了终点瓶颈 ----
        if pv is None or pv.get("g3_source_solves") is None:
            row["gates"]["G3"] = "UNKNOWN_G3"
        else:
            row["gates"]["G3"] = "FAIL" if pv["g3_source_solves"] else "PASS"

        # ---- G4：可测量（需 milestone 事件率或效应尺度；现无数据）----
        row["gates"]["G4"] = "UNKNOWN_G4"

        # ---- 分类（预注册 §4，互斥穷尽）----
        g = row["gates"]
        if g["G2"] == "FAIL":
            row["verdict"] = "SATURATED"
        elif g["G1"] == "FAIL":
            row["verdict"] = "NO_SCAFFOLD"
        elif g["G3"] == "FAIL":
            row["verdict"] = "SOURCE_SOLVES_IT"
        elif g["G4"] == "FAIL":
            row["verdict"] = "UNMEASURABLE"
        elif all(v == "PASS" for v in g.values()):
            row["verdict"] = "CANDIDATE"
        else:
            row["verdict"] = "UNKNOWN"
            row["unknown_gates"] = [k for k, v in g.items() if v.startswith("UNKNOWN")]

        if task in ROLE_LOCKED:
            row["role_locked"] = ROLE_LOCKED[task]
        rows.append(row)

    # ---- dev/holdout 划分（预注册 §5，确定性规则）----
    cands = [
        r for r in rows
        if r["verdict"] == "CANDIDATE" and r["target"] not in ROLE_LOCKED
    ]
    cands.sort(key=lambda r: (-(r["success_bar"] or 0), r["target"]))
    if len(cands) >= 2:
        split = {
            "development": cands[0]["target"],
            "holdout": cands[1]["target"],
            "rule": "CANDIDATE 按 success_bar 降序，同分按任务名字典序",
        }
    else:
        split = {
            "development": None,
            "holdout": None,
            "status": "INSUFFICIENT_SITES",
            "n_candidates": len(cands),
            "rule": "CANDIDATE < 2，按预注册 §5 停止，不得降低门槛凑数",
        }

    unknown = [r["target"] for r in rows if r["verdict"] == "UNKNOWN"]
    return {
        "prereg": "docs/experiments/post_transfer_autonomy_site_screen_prereg_20260806.md",
        "prereg_commit": "3070c78",
        "excluded": EXCLUDED,
        "rows": rows,
        "split": split,
        "n_unknown": len(unknown),
        "unknown_targets": unknown,
        "counts": {
            v: sum(1 for r in rows if r["verdict"] == v)
            for v in [
                "SATURATED", "NO_SCAFFOLD", "SOURCE_SOLVES_IT",
                "UNMEASURABLE", "CANDIDATE", "UNKNOWN",
            ]
        },
    }


def main() -> int:
    res = screen()
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, "screen.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)

    print(f"{'target':20s} {'bar':>6s} {'J_best':>9s} {'H_op':>9s} {'%theory':>8s}  verdict")
    print("-" * 82)
    for r in res["rows"]:
        bar = r["success_bar"] if r["success_bar"] is not None else "-"
        jb = f"{r['J_best_known']:.1f}" if r["J_best_known"] is not None else "-"
        h = f"{r['H_op']:.1f}" if r["H_op"] is not None else "-"
        pct = r.get("pct_of_theory_max")
        ps = f"{pct:.1f}%" if pct is not None else "-"
        print(f"{r['target']:20s} {str(bar):>6s} {jb:>9s} {h:>9s} {ps:>8s}  {r['verdict']}")

    print("\ncounts:", json.dumps(res["counts"], ensure_ascii=False))
    print("split :", json.dumps(res["split"], ensure_ascii=False))
    print(f"\nwrote {os.path.relpath(out, REPO)}")

    if res["n_unknown"] > 0 or res["split"].get("status") == "INSUFFICIENT_SITES":
        print(
            f"\nINCOMPLETE: {res['n_unknown']} target(s) UNKNOWN"
            f"{', INSUFFICIENT_SITES' if res['split'].get('status') else ''}",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
