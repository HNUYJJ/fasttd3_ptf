"""从 expanded Source-Target-Effect Map 生成新 target 的 source bank(RBO-PTF 扩展)。

terrain target(stair/slide/pole/crawl,均 151 布局)的 loco source(stand/walk/run)用
identity adapter(零适配)。base bank(均匀,供 uniform/rand 对比)+ safe bank(per-source
bootstrap.{weight=vs-zero reward-bearing score, horizon=safe_horizon},供 reward-weighted)。

用法: python scripts/build_expanded_banks.py
"""
from __future__ import annotations

import json
from pathlib import Path

import yaml

ALPHA = 1.0
FIX_HORIZON = 25   # = warmup_min_steps;wfix 消融用,与 rand horizon 对齐
TERRAIN = ["stair", "slide", "pole", "crawl"]   # loco 强覆盖(expanded map 验证)
SOURCES = [
    ("stand", "checkpoints/official_sources/h1hand_stand/manifest.json"),
    ("walk", "checkpoints/official_sources/h1hand_walk/manifest.json"),
    ("run", "checkpoints/official_sources/h1hand_run/manifest.json"),
]
SRC_DIR = Path("configs/source_banks")


def src_spec(name, manifest, bootstrap=None):
    s = {"name": name, "manifest": manifest,
         "obs_adapter": {"type": "identity", "output_dim": 151},
         "action_adapter": {"type": "passthrough"}, "action_mask": {"type": "full"},
         "compatibility_sigma": 1.5}
    if bootstrap is not None:
        s["bootstrap"] = bootstrap
    return s


def main():
    rows = [json.loads(l) for l in open("logs/probe/transfer_map_v2_expanded.jsonl")]
    cfg = {}
    for r in rows:
        t = r["target"].replace("h1hand-", "").replace("-v0", "")
        best = -1e9
        for h, d in r["per_horizon"].items():
            best = max(best, d["reward_gain_vs_zero"] - ALPHA * d["fall_prob"])
        cfg.setdefault(t, {})[r["source"]] = {
            "weight": round(max(best, 0.0), 3), "horizon": int(r["safe_horizon"])}

    for task in TERRAIN:
        tc = cfg.get(task, {})
        base = {"null_option": True, "sources": [src_spec(n, m) for n, m in SOURCES]}
        (SRC_DIR / f"h1hand_loco_sources_{task}.yaml").write_text(
            f"# expanded terrain loco bank for {task}(uniform, 151 布局 identity)\n"
            + yaml.safe_dump(base, sort_keys=False, allow_unicode=True))
        safe = {"null_option": True,
                "sources": [src_spec(n, m, tc.get(n)) for n, m in SOURCES]}
        (SRC_DIR / f"h1hand_loco_safe_{task}.yaml").write_text(
            f"# expanded terrain safe bank for {task}(auto; weight=vs-zero reward-bearing,\n"
            f"# horizon=safe_horizon from transfer_map_v2_expanded)\n"
            + yaml.safe_dump(safe, sort_keys=False, allow_unicode=True))
        # wfix(消融解耦): weighted 抽源同 safe,但 horizon 固定 = warmup_min_steps(25),
        # 与 uniform(rand) 的 horizon 对齐 → wfix vs rand 隔离"源选择"、safe vs wfix
        # 隔离"执行时长"。回应 reviewer 对两变量纠缠的质疑。
        def _fixh(n):
            b = tc.get(n)
            return None if b is None else {"weight": b["weight"], "horizon": FIX_HORIZON}
        wfix = {"null_option": True,
                "sources": [src_spec(n, m, _fixh(n)) for n, m in SOURCES]}
        (SRC_DIR / f"h1hand_loco_wfix_{task}.yaml").write_text(
            f"# expanded terrain wfix bank for {task}(weighted 源同 safe, horizon 固定\n"
            f"# {FIX_HORIZON}=warmup_min_steps; 消融解耦源选择 vs 执行时长)\n"
            + yaml.safe_dump(wfix, sort_keys=False, allow_unicode=True))
        print(f"{task:8s} -> base + safe + wfix bank  {{{', '.join(f'{n}:{tc.get(n)}' for n,_ in SOURCES)}}}")


if __name__ == "__main__":
    main()
