"""从 Transfer Map v2 的 bootstrap_config 生成 safe-horizon bootstrap 源库。

不破坏原 loco bank(random warmup 仍用原 bank);生成带 per-source `bootstrap:
{weight, horizon}` 字段的新 bank,供 warmup_mode=safe_bootstrap 用。Day5-6 的
random vs safe 对比 = 切 SOURCE_BANK 指向不同 bank,其余超参一致。

weight = per-task 各源的 vs-zero reward-bearing score(Transfer Map v2,
max_h[reward_gain_vs_zero − fall],clamp≥0)——决定 warmup 抽哪个源;
horizon = safe_horizon(time-to-fall < 0.5 的最长 prefix)——决定执行多久,
脆弱任务(window 25 / balance 10-15)被限制到 safe prefix,不注入摔倒片段。

注:跨任务"哪些任务值得 bootstrap"无干净预测信号(见 transfer_map_v2_analysis.md),
但 per-task"抽哪个源、执行多久"由 Transfer Map 探针给出且有效——这正是本 config。

用法: python scripts/build_safe_bootstrap_banks.py
"""
from __future__ import annotations

import json
from pathlib import Path

import yaml

CFG = json.loads(Path("logs/probe/bootstrap_config.json").read_text())
SRC_DIR = Path("configs/source_banks")
TASKS = ["hurdle", "cabinet", "powerlift", "maze", "window", "balance_hard",
         "truck", "spoon", "door"]


def main() -> None:
    for task in TASKS:
        src_path = SRC_DIR / f"h1hand_loco_sources_{task}.yaml"
        if not src_path.exists():
            print(f"[skip] {task}: no base bank {src_path}")
            continue
        bank = yaml.safe_load(src_path.read_text())
        task_cfg = CFG.get(task, {})
        for s in bank.get("sources", []):
            bc = task_cfg.get(s["name"])
            if bc is not None:
                s["bootstrap"] = {"weight": float(bc["weight"]), "horizon": int(bc["horizon"])}
        out_path = SRC_DIR / f"h1hand_loco_safe_{task}.yaml"
        header = (f"# safe-horizon bootstrap 源库 for {task}(自动生成,勿手改;\n"
                  f"# 改 logs/probe/bootstrap_config.json 后重跑 build_safe_bootstrap_banks.py)。\n"
                  f"# per-source bootstrap.weight=Transfer Map v2 vs-zero reward-bearing score,\n"
                  f"# bootstrap.horizon=safe_horizon(time-to-fall)。warmup_mode=safe_bootstrap 消费。\n")
        out_path.write_text(header + yaml.safe_dump(bank, sort_keys=False, allow_unicode=True))
        ws = {s["name"]: s.get("bootstrap") for s in bank.get("sources", [])}
        print(f"{task:12s} -> {out_path.name}  {ws}")


if __name__ == "__main__":
    main()
