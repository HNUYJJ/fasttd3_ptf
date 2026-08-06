"""P0 δ 阈值冻结脚本(run card §5;五次复核阻塞问题 2)。

δ_return_task = 0.5 × 历史 scratch 跨 seed SD(10k-15k 窗口的 [eval] return)。
数据源=既有 scratch 训练 log 的 `[eval] step=... return=...` 行(非 P0 数据);
必须在任何 P0 分支启动前运行并冻结,输出文件拒绝覆盖(预注册纪律)。

δ_progress_crawl = 0.5 m(绝对物理阈,不从数据估计,直接写死);
truck progress 为描述性指标,无 δ(v2.1.1 裁决)。

用法:
  python scripts/p0_freeze_delta.py \
      --task crawl --logs logs/train/tp_20260615T044012Z/crawl_scr_s1.log \
                          logs/train/seed_20260616T000532Z/crawl_scr_s2.log \
                          logs/train/seed_20260616T000532Z/crawl_scr_s3.log \
      --out configs/experiments/p0_frozen_delta_crawl.json
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

WINDOW_STEPS = (10000, 15000)  # 冻结:窗口恰好两个 eval 点(eval_interval=5000)
EVAL_LINE = re.compile(r"\[eval\] step=(\d+) return=(-?\d+(?:\.\d+)?)")
DELTA_PROGRESS_CRAWL_M = 0.5  # 绝对物理阈(米),写死
REQUIRED_LOG_COUNT = 3        # 冻结:恰好 3 个 scratch seed(与 P0 seed 面板同基数)


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _window_mean(log_path: Path) -> float:
    """窗口均值;要求窗口内恰好覆盖 10000 与 15000 两个 eval 点各一次
    (缺点/重复点都说明 log 与预期训练协议不符,拒绝)。"""
    by_step: dict[int, list[float]] = {}
    for line in log_path.read_text(errors="replace").splitlines():
        match = EVAL_LINE.search(line)
        if match:
            step, value = int(match.group(1)), float(match.group(2))
            if step in WINDOW_STEPS:
                by_step.setdefault(step, []).append(value)
    for step in WINDOW_STEPS:
        if len(by_step.get(step, [])) != 1:
            raise ValueError(
                f"{log_path}: window step {step} must appear exactly once, "
                f"got {len(by_step.get(step, []))}"
            )
    return statistics.mean(by_step[step][0] for step in WINDOW_STEPS)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=["crawl", "truck"])
    parser.add_argument("--logs", nargs="+", required=True,
                        help=f"scratch seed logs(恰好 {REQUIRED_LOG_COUNT} 个,不同 seed)")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    if len(args.logs) != REQUIRED_LOG_COUNT:
        raise ValueError(f"exactly {REQUIRED_LOG_COUNT} scratch logs required, got {len(args.logs)}")
    if len(set(args.logs)) != REQUIRED_LOG_COUNT:
        raise ValueError("scratch logs must be distinct files")
    means = {}
    log_hashes = {}
    for log in args.logs:
        path = (REPO_ROOT / log).resolve() if not Path(log).is_absolute() else Path(log)
        means[str(path)] = _window_mean(path)
        log_hashes[str(path)] = _sha256(path)
    cross_seed_sd = statistics.stdev(means.values())
    delta_return = 0.5 * cross_seed_sd

    git_head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False,
    ).stdout.strip()
    frozen = {
        "task": args.task,
        "delta_return": delta_return,
        "delta_progress_m": DELTA_PROGRESS_CRAWL_M if args.task == "crawl" else None,
        "definition": "0.5 * cross-seed SD of scratch [eval] return, window steps 10000+15000",
        "window_steps": list(WINDOW_STEPS),
        "per_seed_window_means": means,
        "input_log_sha256": log_hashes,
        "cross_seed_sd": cross_seed_sd,
        "git_head": git_head,
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
    }
    out_path = Path(args.out)
    if out_path.exists():
        raise FileExistsError(
            f"δ already frozen at {out_path}; refusing to overwrite (preregistration)"
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(frozen, indent=2) + "\n")
    print(f"[p0_freeze_delta] {args.task}: delta_return={delta_return:.3f} "
          f"(SD={cross_seed_sd:.3f} over {len(means)} seeds) -> {out_path}")


if __name__ == "__main__":
    sys.exit(main())
