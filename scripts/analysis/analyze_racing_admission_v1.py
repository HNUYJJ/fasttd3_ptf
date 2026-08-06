"""racing 准入能力裁决（预注册 docs/experiments/racing_admission_v1_prereg_20260804.md §6）。

判据在见到 crawl/slide 任何评估结果之前冻结：

    U_i,s = 逐 episode 配对差值(源 i 臂 − student 臂) 的均值
    SE_i,s = 该 128 条差值序列的标准误
    admit(T,s) = ∃i: U_i,s > 2·SE_i,s

    false_admit  = crawl 上 admit 的 seed 数                 期望 0
    false_reject = hurdle/slide 上 not admit 的 seed 数       期望 0

    false_admit > 0                        → ADMISSION_FALSE_ADMIT
    false_admit==0 且 false_reject > 0     → ADMISSION_FALSE_REJECT
    两者皆 0                                → ADMISSION_VIABLE
    组合缺失                                → INCOMPLETE（非零退出）
    工程验收失败                            → ENGINEERING_INVALID

三层结构：层1 工程检查（剂量/臂身份/面板/identity）→ 层2 完整性 → 层3 主判据。
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np

K = 10000
SRCS = ("stand", "walk", "run")
ARMS = ("student",) + SRCS
TARGETS = ("crawl", "hurdle", "slide")
NEGATIVE = ("crawl",)                      # 期望全拒
POSITIVE = ("hurdle", "slide")             # 期望全纳
DOSE_BAND = (0.45, 0.55)
SHARE_GAP_MAX = 0.05                       # M26：同 target 内臂间 share 差上限
OUT = Path("docs/data/racing_admission_v1/results.json")

EVAL_DIR = {
    "crawl": Path("docs/data/racing_admission_v1/crawl/source_free_eval"),
    "slide": Path("docs/data/racing_admission_v1/slide/source_free_eval"),
    "hurdle": Path("docs/data/racing_min_horizon_v1/correct_lr/source_free_eval"),
}
CKPT_PAT = {
    "crawl": "models/*rad_crawl_{arm}_s{seed}__*_{k}.pt",
    "slide": "models/*rad_slide_{arm}_s{seed}__*_{k}.pt",
    "hurdle": "models/*rck_{arm}_s{seed}__*_{k}.pt",
}


def eval_path(t: str, arm: str, seed: int) -> Path:
    return EVAL_DIR[t] / f"{arm}_s{seed}_step{K}.json"


def load_panel(p: Path) -> tuple[np.ndarray, list[int]]:
    d = json.loads(p.read_text())
    eps = d["episodes"]
    return (np.array([e["return"] for e in eps], dtype=np.float64),
            [int(e["seed"]) for e in eps])


def engineering_checks(seeds: list[int]) -> tuple[list[str], dict]:
    """层1：剂量 / 臂身份 / 面板一致性 / checkpoint identity。"""
    import torch

    defects: list[str] = []
    audit: dict = {}
    ref_panel: list[int] | None = None

    for t in TARGETS:
        shares: dict[str, float] = {}
        for arm in ARMS:
            for s in seeds:
                # --- 面板一致性（全部 target/arm/seed 共用同一 128-episode 面板）---
                p = eval_path(t, arm, s)
                if p.exists():
                    _, panel = load_panel(p)
                    if ref_panel is None:
                        ref_panel = panel
                    elif panel != ref_panel:
                        defects.append(f"{t}/{arm}/s{s}: panel mismatch")
                    d = json.loads(p.read_text())
                    if not d["checkpoint"].get("identity_checked"):
                        defects.append(f"{t}/{arm}/s{s}: identity_checked=false")

                # --- 剂量与臂身份（源臂才有 admission_audit）---
                hits = sorted(glob.glob(CKPT_PAT[t].format(arm=arm, seed=s, k=K)))
                if len(hits) != 1:
                    defects.append(f"{t}/{arm}/s{s}: checkpoint matches={len(hits)}")
                    continue
                ck = torch.load(hits[0], map_location="cpu", weights_only=False)
                names = ck.get("source_names") or []
                non_null = [n for n in names if n != "null"]      # M28：不硬编码列表
                if arm == "student":
                    if non_null:
                        defects.append(f"{t}/student/s{s}: expected no source, got {names}")
                    continue
                if non_null != [arm]:
                    defects.append(f"{t}/{arm}/s{s}: source_names non-null={non_null} != [{arm}]")
                a = ck.get("admission_audit")
                if not a or "execution_counts" not in a:
                    defects.append(f"{t}/{arm}/s{s}: no admission_audit")
                    continue
                c = [float(x) for x in a["execution_counts"]]
                share = c[0] / max(sum(c), 1.0)
                shares[arm] = share
                if not (DOSE_BAND[0] <= share <= DOSE_BAND[1]):
                    defects.append(f"{t}/{arm}/s{s}: behavior share {share:.4f} outside {DOSE_BAND}")
                audit[f"{t}|{arm}|s{s}"] = {"behavior_share": share, "source_names": names}
        if len(shares) >= 2:                                       # M26：臂间共变剂量
            gap = max(shares.values()) - min(shares.values())
            audit[f"{t}|arm_share_gap"] = gap
            if gap > SHARE_GAP_MAX:
                defects.append(f"{t}: inter-arm share gap {gap:.4f} > {SHARE_GAP_MAX}")
    return defects, audit


def main() -> int:
    seeds = [1, 2, 3]
    OUT.parent.mkdir(parents=True, exist_ok=True)

    # ---- 层2：完整性，独立扫描全部组合（§4：不得在前置缺失时提前 continue）----
    required = [(t, a, s) for t in TARGETS for a in ARMS for s in seeds]
    missing = [f"{t}/{a}/s{s}" for t, a, s in required if not eval_path(t, a, s).exists()]
    if missing:
        print(f"VERDICT: INCOMPLETE  ({len(missing)}/{len(required)} combos missing)")
        for m in missing[:20]:
            print(f"  missing: {m}")
        if len(missing) > 20:
            print(f"  ... 共 {len(missing)} 项")
        OUT.write_text(json.dumps({"verdict": "INCOMPLETE", "missing": missing},
                                  indent=2, ensure_ascii=False))
        return 2

    # ---- 层1：工程检查 ----
    defects, audit = engineering_checks(seeds)
    if defects:
        print("VERDICT: ENGINEERING_INVALID")
        for d in defects:
            print(f"  defect: {d}")
        OUT.write_text(json.dumps({"verdict": "ENGINEERING_INVALID", "defects": defects,
                                   "audit": audit}, indent=2, ensure_ascii=False))
        return 3

    # ---- 层3：主判据 ----
    per_target: dict = {}
    false_admit = 0
    false_reject = 0
    for t in TARGETS:
        rows = []
        for s in seeds:
            base, _ = load_panel(eval_path(t, "student", s))
            entry = {"seed": s, "U": {}, "SE": {}, "significant_positive": []}
            for i in SRCS:
                arm_ret, _ = load_panel(eval_path(t, i, s))
                diff = arm_ret - base                       # 逐 episode 配对
                u = float(diff.mean())
                se = float(diff.std(ddof=1) / np.sqrt(diff.size))
                entry["U"][i] = round(u, 4)
                entry["SE"][i] = round(se, 4)
                if u > 2.0 * se:
                    entry["significant_positive"].append(i)
            entry["admit"] = bool(entry["significant_positive"])
            rows.append(entry)
            if t in NEGATIVE and entry["admit"]:
                false_admit += 1
            if t in POSITIVE and not entry["admit"]:
                false_reject += 1
        per_target[t] = rows

    if false_admit > 0:
        verdict = "ADMISSION_FALSE_ADMIT"
    elif false_reject > 0:
        verdict = "ADMISSION_FALSE_REJECT"
    else:
        verdict = "ADMISSION_VIABLE"

    report = {
        "prereg": "docs/experiments/racing_admission_v1_prereg_20260804.md",
        "K": K, "delta_rule": "U_i > 2 * paired_SE_i",
        "negative_targets": list(NEGATIVE), "positive_targets": list(POSITIVE),
        "per_target": per_target,
        "false_admit": false_admit, "false_reject": false_reject,
        "engineering_audit": audit,
        "verdict": verdict,
    }
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    print("=" * 84)
    print(f"VERDICT: {verdict}     false_admit={false_admit}  false_reject={false_reject}")
    print("=" * 84)
    for t in TARGETS:
        role = "负例(期望全拒)" if t in NEGATIVE else "正例(期望全纳)"
        print(f"\n{t}  {role}")
        for r in per_target[t]:
            cells = "  ".join(
                f"{i}={r['U'][i]:+8.2f}±{r['SE'][i]:5.2f}"
                + ("*" if i in r["significant_positive"] else " ")
                for i in SRCS)
            print(f"  s{r['seed']}  {cells}   admit={r['admit']}")
    print("\n  (* = U > 2·SE)")
    print(f"\nwritten {OUT}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:                 # §4：异常绝不落进实质裁决分支
        print(f"VERDICT: INCOMPLETE  (unhandled error: {type(exc).__name__}: {exc})")
        raise
