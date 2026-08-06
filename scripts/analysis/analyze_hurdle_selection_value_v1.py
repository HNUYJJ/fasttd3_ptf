"""hurdle 选源价值裁决：racing 选中的源 vs racing 判定最差的源。

判据冻结于 docs/experiments/hurdle_selection_value_v1_prereg_20260731.md (06f1adf)。
本脚本只实现，不得事后调整。

达阈口径沿用 analyze_hurdle_speedup_v1.py (1fcf136)：
    steps_X(θ) = 臂 X 的 source-free 回报首次 ≥ θ 的步数（相邻评估点线性插值；
                 全程未达到记 100000 并标右删失）
    θ ∈ {200, 300}   ← 沿用已冻结阈值；θ=400 因 run 臂已含右删失而不用

主判据（预注册 §5）：per-seed 配对 steps_stand(θ) > steps_run(θ)
    SELECTION_VALUABLE ⟺ θ=200 与 θ=300 上均 3/3
    SELECTION_PARTIAL  ⟺ 仅其中一个 θ 满足 3/3
    SELECTION_NULL     ⟺ 两个 θ 均 ≤2/3 —— 有意义的负结果：选哪个源无关紧要

次判据（不参与主裁决）：steps_stand(θ) vs steps_scratch(θ)。

异常分类：缺产物 → INCOMPLETE(exit 2)；产物存在但无效 / 未预期异常 → VOID_ENGINEERING(exit 2)。
"""
from __future__ import annotations

import glob
import hashlib
import json
import math
import os
import statistics as st
from pathlib import Path

import torch

ROOT = Path("docs/data/hurdle_selection_value_v1")
EVAL_STAND = ROOT / "source_free_eval"                                  # 本轮新跑
EVAL_REF = Path("docs/data/hurdle_speedup_v1/source_free_eval")         # run / scratch 已有
TRAIN_LOG = Path("logs/train/hurdle_speedup_v1")
SEEDS = (1, 2, 3)
STEPS = (10000, 20000, 30000, 50000, 75000, 100000)
THRESHOLDS = (200.0, 300.0)
CENSOR = float(STEPS[-1])
DOSE_BAND = (0.48, 0.52)
MAX_SHARE_SPREAD_VS_RUN = 0.02      # 预注册 §4：stand 与 run 臂的 share 差 ≤2pp（M26）
ENV_NAME = "h1hand-hurdle-v0"
PANEL_SEEDS = [s * 1000 + r for s in
               (11, 23, 37, 53, 71, 89, 103, 113, 131, 149, 163, 179, 193, 211, 227, 241)
               for r in range(8)]

ARMS = {"stand": ("hspd_stand", EVAL_STAND, "stand"),
        "run": ("hspd_source", EVAL_REF, "source"),
        "scratch": ("hspd_scratch", EVAL_REF, "scratch")}


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def layer1() -> tuple[list[str], list[str], dict]:
    """只校验本轮新产出的 stand 臂 + 确认参照臂数据齐全（run/scratch 已发表）。"""
    absent: list[str] = []
    defects: list[str] = []
    dose: dict[str, float] = {}

    for arm, (prefix, evaldir, fname) in ARMS.items():
        for sd in SEEDS:
            for stp in STEPS:
                f = evaldir / f"{fname}_s{sd}_step{stp}.json"
                if not f.exists():
                    absent.append(f"{arm}_s{sd}_{stp}: 缺 eval json")

    # stand 臂的 checkpoint / 剂量 / 臂身份（本轮新跑，需完整校验）
    for sd in SEEDS:
        lg = TRAIN_LOG / f"hspd_stand_s{sd}.log"
        if not lg.exists():
            absent.append(f"stand_s{sd}: 缺训练日志")
        for stp in STEPS:
            hits = glob.glob(f"models/*hspd_stand_s{sd}__*_{stp}.pt")
            if not hits:
                absent.append(f"stand_s{sd}_{stp}: 缺 checkpoint")
                continue
            if len(hits) > 1:
                defects.append(f"stand_s{sd}_{stp}: checkpoint 命中 {len(hits)} 个")
                continue
            try:
                c = torch.load(hits[0], map_location="cpu", weights_only=False)
            except Exception as e:
                defects.append(f"stand_s{sd}_{stp}: ckpt 无法加载 {type(e).__name__}"); continue
            if int(c.get("global_step", -1)) != stp:
                defects.append(f"stand_s{sd}_{stp}: global_step={c.get('global_step')}")
            names = c.get("source_names")
            non_null = [n for n in (names or []) if n != "null"]   # M28
            if non_null != ["stand"]:
                defects.append(f"stand_s{sd}_{stp}: source_names={names} 非 null 部分应为 ['stand']")
            aud = c.get("admission_audit")
            if aud is None:
                defects.append(f"stand_s{sd}_{stp}: 缺 admission_audit")
            else:
                for fld in ("execution_counts", "critic_sample_counts"):
                    v = aud.get(fld)
                    if not isinstance(v, (list, tuple)) or len(v) != 2:
                        defects.append(f"stand_s{sd}_{stp}: {fld} 形状非法"); continue
                    ec = [int(x) for x in v]
                    if min(ec) < 0 or sum(ec) <= 0:
                        defects.append(f"stand_s{sd}_{stp}: {fld} 非负/总和失败"); continue
                    sh = ec[0] / sum(ec)
                    if fld == "execution_counts" and stp == STEPS[-1]:
                        dose[f"stand_s{sd}"] = round(sh, 4)
                    if not (DOSE_BAND[0] <= sh <= DOSE_BAND[1]):
                        defects.append(f"stand_s{sd}_{stp}: {fld} share={sh:.4f} 越界{DOSE_BAND}")

    # stand 与 run 的 share 差 ≤2pp（M26；run 臂 100k 的 share 从其 checkpoint 读）
    for sd in SEEDS:
        hits = glob.glob(f"models/*hspd_source_s{sd}__*_{STEPS[-1]}.pt")
        if not hits or f"stand_s{sd}" not in dose:
            continue
        try:
            a = torch.load(hits[0], map_location="cpu", weights_only=False)["admission_audit"]
            ec = [int(x) for x in a["execution_counts"]]
            run_sh = ec[0] / sum(ec)
        except Exception:
            defects.append(f"s{sd}: 无法读取 run 臂 share"); continue
        dose[f"run_s{sd}"] = round(run_sh, 4)
        if abs(dose[f"stand_s{sd}"] - run_sh) > MAX_SHARE_SPREAD_VS_RUN:
            defects.append(f"s{sd}: stand/run 的 share 差 "
                           f"{abs(dose[f'stand_s{sd}']-run_sh):.4f} > {MAX_SHARE_SPREAD_VS_RUN}")

    # 全部 eval json 的结构/面板/协议（含参照臂：确认其未被改动）
    for arm, (prefix, evaldir, fname) in ARMS.items():
        for sd in SEEDS:
            for stp in STEPS:
                f = evaldir / f"{fname}_s{sd}_step{stp}.json"
                if not f.exists():
                    continue
                try:
                    d = json.loads(f.read_text())
                    cj, ag, proto, eps = d["checkpoint"], d["aggregate"], d["protocol"], d["episodes"]
                except Exception as e:
                    defects.append(f"{arm}_s{sd}_{stp}: json 无效 {type(e).__name__}"); continue
                t = f"{arm}_s{sd}_{stp}"
                if ag.get("episode_count") != 128 or len(eps) != 128:
                    defects.append(f"{t}: episode 数不符")
                if cj.get("identity_checked") is not True:
                    defects.append(f"{t}: identity_checked={cj.get('identity_checked')!r}")
                if cj.get("global_step") != stp:
                    defects.append(f"{t}: json global_step 不符")
                if f"{prefix}_s{sd}__" not in str(cj.get("path", "")):
                    defects.append(f"{t}: ckpt 路径与臂/seed 不符")
                if d.get("env_name") != ENV_NAME:
                    defects.append(f"{t}: env_name={d.get('env_name')}")
                if proto.get("deterministic") is not True:
                    defects.append(f"{t}: deterministic 非 True")
                if [e.get("seed") for e in eps] != PANEL_SEEDS:
                    defects.append(f"{t}: episode seed 序列与冻结面板不符")
                if not all(isinstance(e.get("return"), (int, float)) and math.isfinite(e["return"])
                           for e in eps):
                    defects.append(f"{t}: 存在非有限 return")
    return absent, defects, dose


def curve(arm: str, sd: int) -> list[tuple[int, float]]:
    _, evaldir, fname = ARMS[arm]
    pts = []
    for stp in STEPS:
        f = evaldir / f"{fname}_s{sd}_step{stp}.json"
        if f.exists():
            pts.append((stp, float(json.loads(f.read_text())["aggregate"]["return_mean"])))
    return pts


def steps_to(pts: list[tuple[int, float]], theta: float) -> tuple[float, bool]:
    """首次 ≥ θ 的步数；线性插值。返回 (steps, censored)。与 1fcf136 逐行一致。"""
    prev = None
    for s, r in pts:
        if r >= theta:
            if prev is None:
                return float(s), False
            (s0, r0) = prev
            if r == r0:
                return float(s), False
            return s0 + (theta - r0) / (r - r0) * (s - s0), False
        prev = (s, r)
    return CENSOR, True


def main() -> None:
    try:
        _run()
    except SystemExit:
        raise
    except Exception as e:
        ROOT.mkdir(parents=True, exist_ok=True)
        out = ROOT / "results.json"
        tmp = out.with_suffix(".json.tmp")
        tmp.write_text(json.dumps({"verdict": "VOID_ENGINEERING",
                                   "unexpected_exception": f"{type(e).__name__}: {e}"},
                                  indent=2, ensure_ascii=False))
        tmp.replace(out)
        print("=" * 84); print("VERDICT: VOID_ENGINEERING"); print("=" * 84)
        print(f"  未预期异常: {type(e).__name__}: {e}")
        raise SystemExit(2)


def _run() -> None:
    run_id = os.environ.get("HSV_RUN_ID") or _sha256(__file__)[:12]
    report: dict = {"prereg": "docs/experiments/hurdle_selection_value_v1_prereg_20260731.md",
                    "prereg_commit": "06f1adf", "run_id": run_id,
                    "threshold_source": "沿用 analyze_hurdle_speedup_v1.py (1fcf136) 的达阈口径"}
    ROOT.mkdir(parents=True, exist_ok=True)
    out = ROOT / "results.json"
    if out.exists():
        out.unlink()

    def dump(verdict: str, lines: list[str], code: int = 0):
        report["verdict"] = verdict
        tmp = out.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(report, indent=2, ensure_ascii=False))
        tmp.replace(out)
        print("=" * 84); print(f"VERDICT: {verdict}"); print("=" * 84)
        for ln in lines:
            print(ln)
        print(f"\nrun_id={run_id}  written {out}")
        raise SystemExit(code)

    absent, defects, dose = layer1()
    report["dose"] = dose
    if absent:
        dump("INCOMPLETE", [f"  尚缺 {len(absent)} 项产物"] + [f"    {a}" for a in absent[:8]], 2)
    if defects:
        report["defects"] = defects
        dump("VOID_ENGINEERING", [f"  {d}" for d in defects[:20]], 2)

    curves = {a: {sd: curve(a, sd) for sd in SEEDS} for a in ARMS}
    report["curves"] = {a: {str(sd): curves[a][sd] for sd in SEEDS} for a in ARMS}
    lines = ["  层1 工程检查 PASS（stand 臂剂量/臂身份/面板/协议；参照臂数据齐全）"]
    lines.append("\n  source-free 回报曲线：")
    for a in ("scratch", "run", "stand"):
        for sd in SEEDS:
            lines.append(f"    {a:8s} s{sd}: " + "  ".join(
                f"{s//1000}k={r:.1f}" for s, r in curves[a][sd]))

    res, pass_flags = {}, []
    for th in THRESHOLDS:
        rows, hits, hits_vs_scratch = [], 0, 0
        for sd in SEEDS:
            st_stand, c_stand = steps_to(curves["stand"][sd], th)
            st_run, c_run = steps_to(curves["run"][sd], th)
            st_scr, c_scr = steps_to(curves["scratch"][sd], th)
            hit = st_stand > st_run
            hits += int(hit)
            hits_vs_scratch += int(st_stand < st_scr)
            rows.append({"seed": sd,
                         "steps_stand": round(st_stand), "steps_run": round(st_run),
                         "steps_scratch": round(st_scr),
                         "stand_slower_than_run": hit,
                         "stand_faster_than_scratch": st_scr > st_stand,
                         "censored": {"stand": c_stand, "run": c_run, "scratch": c_scr}})
        res[str(int(th))] = {"per_seed": rows, "hits": f"{hits}/3", "pass": hits == 3,
                             "secondary_stand_beats_scratch": f"{hits_vs_scratch}/3"}
        pass_flags.append(hits == 3)

    report["thresholds"] = res
    n = sum(pass_flags)
    verdict = "SELECTION_VALUABLE" if n == 2 else ("SELECTION_PARTIAL" if n == 1 else "SELECTION_NULL")
    report["chance_pass"] = "保守 1/8（零效应下每 seed 1/2，3/3 为 1/8；两个 θ 强相关）"
    report["caveat_M24"] = "单批 3 seeds；若为正只能报 pilot，须注明待独立重复"

    for th in THRESHOLDS:
        d = res[str(int(th))]
        lines.append(f"\n  --- θ={int(th)} ---  stand 慢于 run: {d['hits']}  "
                     f"{'PASS' if d['pass'] else '----'}   "
                     f"(次判据 stand 快于 scratch: {d['secondary_stand_beats_scratch']})")
        for r in d["per_seed"]:
            lines.append(f"    s{r['seed']}: stand={r['steps_stand']:6d}  run={r['steps_run']:6d}  "
                         f"scratch={r['steps_scratch']:6d}   "
                         + ("stand 慢于 run ✓" if r["stand_slower_than_run"] else "**stand 不慢于 run**"))
    dump(verdict, lines)


if __name__ == "__main__":
    main()
