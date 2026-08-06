"""标准 9 源 bank 生成器——第二批目标任务（2026-07-05，PI 定向源库扩大）。

源库 = 4 官方(stand/walk/run/reach) + 5 自训 terrain(hurdle/stair/slide/crawl/pole)。
哲学: 库尽量大, 选择交给 T⁰(静态 softmax 加权下低分源自动边缘化, 无在线探索税;
hurdle 增量实验已验证该机制: truck 上 stair probe 0.3 被自动边缘化)。

adapter 规则(按 hb_task_layouts 的 obs_dim/nq, metadata 驱动无任务名分支):
- 151 源(loco/terrain): 目标 151 → identity; 否则 hb_robot_qpos_qvel(qpos_dim=nq)
- reach 源(157=151 proprio+6 目标位): 目标 151 → identity+pad6;
  否则 slice(indices=[0..75, nq..nq+74]) + pad 到 157(task 6 维置零)

输出: h1hand_std9_wfix_{task}.yaml(weight=probe, h25) + h1hand_std9_sources_{task}.yaml(uniform)

用法: python scripts/build_std9_banks.py <task> <probe_jsonl> [<task> <probe_jsonl> ...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

HORIZON = 25
SRC151 = {
    "stand": "checkpoints/official_sources/h1hand_stand/manifest.json",
    "walk": "checkpoints/official_sources/h1hand_walk/manifest.json",
    "run": "checkpoints/official_sources/h1hand_run/manifest.json",
    "hurdle": "checkpoints/terrain_sources/h1hand_hurdle/manifest.json",
    "stair": "checkpoints/terrain_sources/h1hand_stair/manifest.json",
    "slide": "checkpoints/terrain_sources/h1hand_slide/manifest.json",
    "crawl": "checkpoints/terrain_sources/h1hand_crawl/manifest.json",
    "pole": "checkpoints/terrain_sources/h1hand_pole/manifest.json",
}
REACH = "checkpoints/official_sources/h1hand_reach/manifest.json"

layouts = json.loads(Path("logs/probe/hb_task_layouts.json").read_text())
LAY = {r["task"]: r for r in layouts} if isinstance(layouts, list) else layouts


def adapters_for(task: str) -> tuple[dict, dict]:
    rec = LAY[f"h1hand-{task}-v0"]
    obs_dim, nq = int(rec["obs_dim"]), int(rec["nq"])
    if obs_dim == 151:
        a151 = {"type": "identity", "output_dim": 151}
        a_reach = {"type": "identity", "output_dim": 157, "allow_pad": True}
    else:
        a151 = {"type": "hb_robot_qpos_qvel", "qpos_dim": nq, "output_dim": 151}
        a_reach = {"type": "slice", "indices": list(range(76)) + list(range(nq, nq + 75)),
                   "output_dim": 157, "allow_pad": True}
    return a151, a_reach


def main() -> None:
    args = sys.argv[1:]
    pairs = list(zip(args[0::2], args[1::2]))
    for task, probe_path in pairs:
        scores: dict[str, float] = {}
        for line in Path(probe_path).read_text().strip().split("\n"):
            r = json.loads(line)
            if r["target"] == f"h1hand-{task}-v0":
                scores[r["source"]] = max(0.0, float(r["best_score"]))
        a151, a_reach = adapters_for(task)
        srcs = []
        for name in list(SRC151) + ["reach"]:
            manifest = SRC151.get(name, REACH)
            adapter = a_reach if name == "reach" else a151
            srcs.append({
                "name": name, "manifest": manifest,
                "obs_adapter": dict(adapter),
                "action_adapter": {"type": "passthrough"},
                "action_mask": {"type": "full"},
                "compatibility_sigma": 1.5,
            })
        uni = {"null_option": True, "sources": [dict(s) for s in srcs]}
        p_uni = Path(f"configs/source_banks/h1hand_std9_sources_{task}.yaml")
        p_uni.write_text(f"# 标准 9 源 uniform bank for {task}(rand 用; build_std9_banks.py 生成)。\n"
                         + yaml.safe_dump(uni, sort_keys=False, allow_unicode=True))
        for s in srcs:
            s["bootstrap"] = {"weight": round(scores.get(s["name"], 0.0), 3), "horizon": HORIZON}
        big = {"null_option": True, "sources": srcs}
        p_big = Path(f"configs/source_banks/h1hand_std9_wfix_{task}.yaml")
        p_big.write_text(f"# 标准 9 源 weighted bank for {task}(weight=T⁰ probe, h{HORIZON};\n"
                         f"# build_std9_banks.py 生成, probe={Path(probe_path).name})。\n"
                         + yaml.safe_dump(big, sort_keys=False, allow_unicode=True))
        w = {s["name"]: s["bootstrap"]["weight"] for s in srcs}
        print(f"{task:18s} -> {p_big.name}  weights={w}")


if __name__ == "__main__":
    main()
