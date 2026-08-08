#!/usr/bin/env python3
"""PARE 2k smoke 的判据裁定（spec §12 第 3 项）。

四项判据逐条对齐 spec 原文。任一项数据缺失一律输出 ``INCOMPLETE`` 并**非零退出**
（CLAUDE.md §4）——"没跑完"绝不能被读成"通过"，也不得落进 PASS 分支。

    S1  provenance 两类都存在   release anchor 的 replay 里 z=1 与 z=0 都非空
    S2  D 输出有限              PARE 臂全程 d_loss / source_affinity 有限
    S3  actor 梯度有限          PARE 臂 actor_grad_norm 全程有限
    S4  PARE 生效                PARE-on 与 PARE-off 的 final actor 不相同

用法：python scripts/analysis/adjudicate_pare_smoke_v1.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

LOGDIR = REPO / "logs/train/pare_smoke_v1"
ANCHOR = REPO / "artifacts/pare_smoke_v1/anchor_k1000"
MODELS = REPO / "models"

VERDICTS: dict[str, str] = {}
DETAIL: dict[str, object] = {}


def _fail(key: str, verdict: str, **detail) -> None:
    VERDICTS[key] = verdict
    DETAIL[key] = detail


def find_final(exp: str) -> Path | None:
    hits = sorted(MODELS.glob(f"*__{exp}__*_2000.pt")) or sorted(
        MODELS.glob(f"*__{exp}__*_final.pt")
    )
    return hits[-1] if hits else None


# ── S1  provenance 两类都存在 ────────────────────────────────────────
def check_s1() -> None:
    bundle = ANCHOR / "replay.pt"
    if not bundle.exists():
        _fail("S1", "INCOMPLETE", reason=f"release anchor replay 缺失: {bundle}")
        return
    blob = torch.load(bundle, map_location="cpu", weights_only=False)
    prov = blob.get("provenance") if isinstance(blob, dict) else None
    if not prov or "executed_group_mask" not in prov:
        _fail("S1", "INCOMPLETE", reason="anchor replay 内无 executed_group_mask")
        return
    mask = torch.as_tensor(prov["executed_group_mask"])
    written = torch.as_tensor(prov.get("provenance_written", torch.ones_like(mask[..., 0])))
    is_src = mask.any(dim=-1) & written.bool()
    n_src = int(is_src.sum())
    n_stu = int((~is_src & written.bool()).sum())
    ok = n_src > 0 and n_stu > 0
    VERDICTS["S1"] = "PASS" if ok else "FAIL"
    DETAIL["S1"] = {"n_source": n_src, "n_student": n_stu,
                    "source_share": round(n_src / max(1, n_src + n_stu), 4)}


# ── S2 / S3  日志里的 PARE 指标有限性 ────────────────────────────────
#: 训练循环每 100 步打一行 ``[pare] step=N {json}``（不依赖 wandb）。
#: **不能锚定行首**：tqdm 用 ``\r`` 刷新进度条，print 的内容会被拼在同一物理行里，
#: 首版用 ``^\[pare\]`` 匹配到 0 条，险些把"跑通了"读成"PARE 没生效"。
PARE_LINE = re.compile(r"\[pare\] step=(\d+) (\{[^{}]*\})")


def scan_metrics(log: Path) -> dict[str, list[float]]:
    """解析结构化的 ``[pare]`` 行。

    刻意不做宽松 grep：宽松匹配会把别处出现的同名子串当成指标，
    于是"指标缺失"被读成"指标存在且有限"——判据就假通过了。
    """
    out: dict[str, list[float]] = {}
    for _step, blob in PARE_LINE.findall(log.read_text(errors="replace")):
        try:
            payload = json.loads(blob)
        except json.JSONDecodeError:
            continue
        for k, v in payload.items():
            if isinstance(v, (int, float)):
                out.setdefault(k, []).append(float(v))
    return out


def _finite_check(name: str, values: list[float], key: str) -> None:
    if not values:
        _fail(name, "INCOMPLETE", reason=f"日志中未出现 {key}")
        return
    bad = [v for v in values if v != v or v in (float("inf"), float("-inf"))]
    VERDICTS[name] = "PASS" if not bad else "FAIL"
    DETAIL[name] = {"key": key, "n_samples": len(values),
                    "n_nonfinite": len(bad),
                    "min": min(values), "max": max(values)}


S2_KEYS = ("pare/d_loss", "pare/source_affinity")
S3_KEYS = ("pare/base_grad_norm_scaled", "pare/expansion_norm_ratio")


def check_s2_s3() -> None:
    log = LOGDIR / "psmoke_on.log"
    if not log.exists():
        _fail("S2", "INCOMPLETE", reason=f"缺日志 {log}")
        _fail("S3", "INCOMPLETE", reason=f"缺日志 {log}")
        return
    got = scan_metrics(log)

    # 每组要求**组内每个键**都存在且有限——不接受"有一个就算过"。
    for name, keys in (("S2", S2_KEYS), ("S3", S3_KEYS)):
        missing = [k for k in keys if not got.get(k)]
        if missing:
            _fail(name, "INCOMPLETE", reason=f"缺指标 {missing}")
            continue
        vals: list[float] = []
        for k in keys:
            vals += got[k]
        _finite_check(name, vals, " + ".join(keys))

    DETAIL["mechanism_sample"] = {
        k: round(v[-1], 6) for k, v in sorted(got.items()) if v
    }
    DETAIL["n_pare_log_lines"] = max((len(v) for v in got.values()), default=0)


# ── S4  PARE 确实改变了 actor ────────────────────────────────────────
def check_s4() -> None:
    a, b = find_final("psmoke_off"), find_final("psmoke_on")
    if a is None or b is None:
        _fail("S4", "INCOMPLETE",
              reason=f"缺 final checkpoint: off={a}, on={b}")
        return
    sa = torch.load(a, map_location="cpu", weights_only=False)["actor_state_dict"]
    sb = torch.load(b, map_location="cpu", weights_only=False)["actor_state_dict"]
    if set(sa) != set(sb):
        _fail("S4", "FAIL", reason="两臂 actor 结构不同")
        return
    max_abs = max(float((sa[k].float() - sb[k].float()).abs().max()) for k in sa)
    VERDICTS["S4"] = "PASS" if max_abs > 0 else "FAIL"
    DETAIL["S4"] = {"max_abs_param_diff": max_abs,
                    "note": "差异为 0 说明 PARE 是空操作"}


def main() -> int:
    check_s1()
    check_s2_s3()
    check_s4()

    order = ("S1", "S2", "S3", "S4")
    for k in order:
        VERDICTS.setdefault(k, "INCOMPLETE")
    # 全称词只在**每一项都真实走过判定路径**时才允许（CLAUDE.md §4）。
    if any(VERDICTS[k] == "INCOMPLETE" for k in order):
        overall = "INCOMPLETE"
    elif all(VERDICTS[k] == "PASS" for k in order):
        overall = "SMOKE_PASS"
    else:
        overall = "SMOKE_FAIL"

    report = {"verdict": overall, "checks": {k: VERDICTS[k] for k in order},
              "detail": DETAIL,
              "scope": "工程冒烟，不构成任何科学结论"}
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if overall == "SMOKE_PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
