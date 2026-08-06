"""任务分类学 F2/F3/F6：从 MJCF 场景递归提取几何、自由物体与机器人信息。

只解析 `assets/tasks/<task>.xml` 及其递归 include（即**场景**部分），
不含机器人本体（`assets/robots/*.xml`）——后者对所有任务相同，
计入场景会淹没任务间差异。

无法机械确定的一律记 "unknown"，不按直觉补值。
"""
from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

HB = Path(__file__).resolve().parents[2] / \
    "fasttd3_ptf/official_code/humanoid-bench/humanoid_bench"
ASSETS = HB / "assets"
UNKNOWN = "unknown"


def load_recursive(path: Path, seen: set | None = None) -> list[tuple[Path, ET.Element]]:
    """递归展开 <include file="..."/>，返回 (来源文件, 根元素) 列表。"""
    seen = seen if seen is not None else set()
    if not path.exists() or path in seen:
        return []
    seen.add(path)
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError:
        return []
    out = [(path, root)]
    for inc in root.iter("include"):
        f = inc.get("file")
        if f:
            out += load_recursive((path.parent / f).resolve(), seen)
    return out


def scene_features(task: str) -> dict:
    task_xml = ASSETS / "tasks" / f"{task}.xml"
    if not task_xml.exists():
        return {"scene_file": None, "note": "无独立场景文件（任务可能只用 floor）",
                "geom_types": {}, "n_freejoint": 0, "joint_types": {},
                "n_body": 0, "mesh_names": []}

    trees = load_recursive(task_xml)
    geom_types: dict[str, int] = {}
    joint_types: dict[str, int] = {}
    box_roles: dict[str, int] = {}
    n_free = n_body = 0
    meshes, sizes_box, hfields = [], [], []

    for src, root in trees:
        for g in root.iter("geom"):
            # class="visual" 的几何不参与物理；只统计参与碰撞的与未标注的
            if g.get("class") == "visual":
                continue
            t = g.get("type") or ("mesh" if g.get("mesh") else "box_or_default")
            geom_types[t] = geom_types.get(t, 0) + 1
            if t in ("box", "box_or_default") and g.get("size"):
                sizes_box.append(g.get("size"))
                r = classify_box(g.get("size"))
                box_roles[r] = box_roles.get(r, 0) + 1
        for m in root.iter("mesh"):
            if m.get("name"):
                meshes.append({"name": m.get("name"), "vertex": m.get("vertex"),
                               "file": m.get("file")})
        for h in root.iter("hfield"):
            hfields.append(h.get("name"))
        for j in root.iter("joint"):
            jt = j.get("type") or "hinge(default)"
            joint_types[jt] = joint_types.get(jt, 0) + 1
        n_free += len(list(root.iter("freejoint")))
        n_body += len(list(root.iter("body")))

    return {
        "scene_file": str(task_xml.relative_to(ASSETS)),
        "included_files": [str(p.relative_to(ASSETS)) for p, _ in trees],
        "geom_types": geom_types, "joint_types": joint_types,
        "n_freejoint": n_free, "n_body": n_body,
        "mesh_defs": meshes[:6], "box_sizes_sample": sizes_box[:6],
        "box_roles": box_roles, "hfields": hfields,
    }


def robot_and_obs(task: str) -> dict:
    """F6：机器人变体与场景挂载方式（以 h1hand 为准，本项目全部使用该变体）。"""
    env_xml = ASSETS / "envs" / f"h1hand_pos_{task}.xml"
    if not env_xml.exists():
        return {"env_file": None, "robot": UNKNOWN, "includes": []}
    root = ET.parse(env_xml).getroot()
    incs = [i.get("file") for i in root.iter("include")]
    robot = next((i for i in incs if "robots/" in (i or "")), UNKNOWN)
    key = root.find(".//key")
    qpos = key.get("qpos") if key is not None else None
    return {"env_file": f"envs/h1hand_pos_{task}.xml", "robot": robot,
            "includes": incs,
            "qpos0_dim": len(qpos.split()) if qpos else UNKNOWN,
            "qpos0_sha": None if not qpos else __import__("hashlib").sha256(
                qpos.encode()).hexdigest()[:12]}


def classify_box(size_str: str) -> str:
    """按半尺寸的几何语义区分 box 的角色。纯几何判据，与任何实验结果无关。

    MuJoCo 的 box size 是三个半长 (sx, sy, sz)：
      · 最小维在 z 且明显扁 → 水平板（可行走表面 / 台阶踏面）
      · 最小维在 x 或 y 且 z 明显更大 → 竖直薄板（边界墙 / 挡板）
      · 其余 → 物体或结构件
    """
    try:
        s = [abs(float(x)) for x in size_str.split()]
    except Exception:
        return UNKNOWN
    if len(s) != 3:
        return UNKNOWN
    sx, sy, sz = s
    mn = min(s)
    if sz == mn and sz < 0.5 * max(sx, sy):
        return "horizontal_slab"
    if mn in (sx, sy) and sz > 2 * mn:
        return "vertical_wall"
    return "object_or_structure"


def terrain_class(feat: dict) -> str:
    """F2 地形分类：只统计**构成可行走表面**的几何，排除边界墙与器械。

    证据不足以判定时返回 unknown，不按直觉补值。
    """
    if feat.get("scene_file") is None or not feat.get("included_files"):
        return "flat_floor_only"
    if feat["hfields"]:
        return "heightfield"
    roles = feat.get("box_roles", {})
    n_slab = roles.get("horizontal_slab", 0)
    n_mesh = feat["geom_types"].get("mesh", 0)
    total_geom = sum(feat["geom_types"].values())

    if total_geom == 0:
        return "flat_floor_only"
    if n_mesh > 0 and n_slab == 0:
        return "continuous_mesh"
    if n_mesh > 0 and n_slab > 0:
        return "mixed_mesh_and_slabs"
    if n_slab >= 5:
        return "discrete_slabs"
    if n_slab > 0:
        return "few_slabs"
    # 只有墙/器械/自由物体，地面本身是平的
    return "flat_floor_with_objects"


def main() -> None:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    f1 = json.loads((Path(__file__).resolve().parents[2] /
                     "docs/data/task_taxonomy_v1_f1f5.json").read_text())

    out = {}
    for task in f1:
        sc = scene_features(task)
        out[task] = {"F2_F3_scene": sc, "F2_terrain_class": terrain_class(sc),
                     "F6_robot": robot_and_obs(task)}

    dst = Path(__file__).resolve().parents[2] / "docs/data/task_taxonomy_v1_f2f3f6.json"
    dst.write_text(json.dumps(out, indent=1, ensure_ascii=False))

    print(f"{'task':18s} {'terrain_class':24s} {'slab':>5s} {'wall':>5s} {'mesh':>5s} "
          f"{'obj':>4s} {'free':>5s} {'articulated joints':22s}")
    print("-" * 100)
    for t, v in out.items():
        s = v["F2_F3_scene"]; r = s.get("box_roles", {})
        j = ",".join(f"{k}:{n}" for k, n in sorted(s["joint_types"].items()))
        print(f"{t:18s} {v['F2_terrain_class']:24s} "
              f"{r.get('horizontal_slab',0):5d} {r.get('vertical_wall',0):5d} "
              f"{s['geom_types'].get('mesh',0):5d} {r.get('object_or_structure',0):4d} "
              f"{s['n_freejoint']:5d} {j[:22]:22s}")
    print(f"\nsaved: {dst}")


if __name__ == "__main__":
    main()
