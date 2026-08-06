"""从 7 源 transfer map probe 生成扩源(big)bank——扩源扩任务第一批(2026-07-04)。

输入: logs/probe/transfer_map_v2_bigsrc.jsonl(7 源 × maze/truck/cabinet,
      同批 zero baseline,不与旧 probe 混算)。
输出(每任务两个 bank):
  h1hand_big_wfix_{task}.yaml    7 源 + bootstrap{weight=clamp(best_score,0), horizon=25}
                                  → obrw-big 用(与主表 wfix h25 口径一致)
  h1hand_big_sources_{task}.yaml 7 源无 bootstrap → rand-big(uniform warmup)用

源构成: stand/walk/run(官方 loco) + crawl/pole/slide/stair(terrain scr s1 final,
obs 151 同布局 identity 直连)。obs_adapter 按目标任务布局:
  maze(151/76)   → identity/151
  truck(216/111) → hb_robot_qpos_qvel(qpos_dim=111)
  cabinet(213/109)→ hb_robot_qpos_qvel(qpos_dim=109)

用法: python scripts/build_bigsrc_banks.py
"""
from __future__ import annotations

import json
from pathlib import Path

import yaml

PROBE = Path("logs/probe/transfer_map_v2_bigsrc.jsonl")
OUT_DIR = Path("configs/source_banks")
HORIZON = 25  # 主表 wfix/obrw 口径

SOURCES = {
    "stand": "checkpoints/official_sources/h1hand_stand/manifest.json",
    "walk": "checkpoints/official_sources/h1hand_walk/manifest.json",
    "run": "checkpoints/official_sources/h1hand_run/manifest.json",
    "crawl": "checkpoints/terrain_sources/h1hand_crawl/manifest.json",
    "pole": "checkpoints/terrain_sources/h1hand_pole/manifest.json",
    "slide": "checkpoints/terrain_sources/h1hand_slide/manifest.json",
    "stair": "checkpoints/terrain_sources/h1hand_stair/manifest.json",
}
# 目标任务 obs 布局(logs/probe/hb_task_layouts.json 核实)
TASK_ADAPTER = {
    "maze": {"type": "identity", "output_dim": 151},
    "truck": {"type": "hb_robot_qpos_qvel", "qpos_dim": 111, "output_dim": 151},
    "cabinet": {"type": "hb_robot_qpos_qvel", "qpos_dim": 109, "output_dim": 151},
}


def main() -> None:
    scores: dict[str, dict[str, float]] = {}
    for line in PROBE.read_text().strip().split("\n"):
        r = json.loads(line)
        task = r["target"].replace("h1hand-", "").replace("-v0", "")
        if r["source"] in SOURCES:
            scores.setdefault(task, {})[r["source"]] = max(0.0, float(r["best_score"]))

    for task, adapter in TASK_ADAPTER.items():
        if task not in scores:
            print(f"[skip] {task}: probe 数据缺失")
            continue
        srcs = []
        for name, manifest in SOURCES.items():
            s = {
                "name": name,
                "manifest": manifest,
                "obs_adapter": dict(adapter),
                "action_adapter": {"type": "passthrough"},
                "action_mask": {"type": "full"},
                "compatibility_sigma": 1.5,
            }
            srcs.append(s)
        # uniform 版(rand-big)
        uni = {"null_option": True, "sources": [dict(s) for s in srcs]}
        p_uni = OUT_DIR / f"h1hand_big_sources_{task}.yaml"
        p_uni.write_text(
            f"# 扩源 uniform bank for {task}(7 源无 bootstrap, rand-big 用;\n"
            f"# 由 build_bigsrc_banks.py 自动生成,勿手改)。\n"
            + yaml.safe_dump(uni, sort_keys=False, allow_unicode=True))
        # weighted 版(obrw-big)
        for s in srcs:
            s["bootstrap"] = {"weight": scores[task].get(s["name"], 0.0), "horizon": HORIZON}
        big = {"null_option": True, "sources": srcs}
        p_big = OUT_DIR / f"h1hand_big_wfix_{task}.yaml"
        p_big.write_text(
            f"# 扩源 weighted bank for {task}(7 源+vs-zero weight, horizon={HORIZON};\n"
            f"# weight 来自 transfer_map_v2_bigsrc.jsonl 同批 probe;\n"
            f"# 由 build_bigsrc_banks.py 自动生成,勿手改)。\n"
            + yaml.safe_dump(big, sort_keys=False, allow_unicode=True))
        w = {s["name"]: round(s["bootstrap"]["weight"], 3) for s in srcs}
        print(f"{task:8s} -> {p_big.name} + {p_uni.name}  weights={w}")


if __name__ == "__main__":
    main()
