"""slide 标签可推广性审计裁决（`M31` 的首次实操）。

判据冻结于 docs/experiments/slide_generalizability_v1_prereg_20260731.md (2c1804f)。
本脚本只实现，不得事后调整。

    U_i = J_sf(源臂 i at 20000) − J_sf(student 臂 at 20000)      per-seed 配对
    GEN_OK      ⟺ seeds 4–6 中 3/3 满足 argmax_i U_i = walk
    GEN_PARTIAL ⟺ 2/3
    GEN_FAILED  ⟺ ≤1/3

参照（BAC gate，seeds 1–3，已发表）：3/3 argmax = walk；
U_walk = +65.66/+47.61/+57.57，U_run = +26.29/+7.27/+17.13，
U_stand = −1.84/−4.48/+2.69（本就 `uncertain`，不作要求）。

异常分类：缺产物 → INCOMPLETE(exit 2)；产物存在但无效 / 未预期异常 → VOID_ENGINEERING(exit 2)。
盲态：层1 的输出与控制流不含/不依赖任何 return 派生量（只做类型与有限性的有效性判定）。
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

ROOT = Path("docs/data/slide_generalizability_v1")
EVAL = ROOT / "source_free_eval"
TRAIN_LOG = Path("logs/train/slide_bac_gate_v1")
SEEDS = (4, 5, 6)
GSTEP, ANCHOR = 20000, 10000
SOURCES = ("stand", "walk", "run")
ARMS = ("student",) + SOURCES
TRUE_ARGMAX = "walk"
DOSE_BAND = (0.45, 0.55)          # 预注册 §4：沿用 BAC gate 的带
MAX_SHARE_SPREAD = 0.05           # 同 target 内各源臂 share 差 ≤5pp（M26）
ENV_NAME = "h1hand-slide-v0"
T_095_DF2 = 2.919986
PANEL_SEEDS = [s * 1000 + r for s in
               (11, 23, 37, 53, 71, 89, 103, 113, 131, 149, 163, 179, 193, 211, 227, 241)
               for r in range(8)]

# BAC gate 的 seeds 1–3 参照（仅报告，不参与裁决）
GATE_REF = {"stand": (-1.84, -4.48, 2.69), "walk": (65.66, 47.61, 57.57),
            "run": (26.29, 7.27, 17.13)}


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def layer1() -> tuple[list[str], list[str], dict]:
    absent: list[str] = []
    defects: list[str] = []
    dose: dict[str, float] = {}

    for sd in SEEDS:
        for arm in ARMS:
            tag = f"{arm}_s{sd}"
            lg = TRAIN_LOG / f"slide_bac_{arm}_s{sd}.log"
            if not lg.exists():
                absent.append(f"{tag}: 缺训练日志")
            else:
                txt = lg.read_text(errors="ignore")
                if "Resumed core learner" not in txt or f"at step {ANCHOR}" not in txt:
                    defects.append(f"{tag}: 日志缺 'Resumed core learner ... at step {ANCHOR}'")

            hits = glob.glob(f"models/*slide_bac_{arm}_s{sd}__*_{GSTEP}.pt")
            f = EVAL / f"{arm}_s{sd}_step{GSTEP}.json"
            if not hits:
                absent.append(f"{tag}: 缺 checkpoint")
            elif len(hits) > 1:
                defects.append(f"{tag}: checkpoint 命中 {len(hits)} 个")
            if not f.exists():
                absent.append(f"{tag}: 缺 eval json")
            if not hits or not f.exists():
                continue

            try:
                c = torch.load(hits[0], map_location="cpu", weights_only=False)
            except Exception as e:
                defects.append(f"{tag}: checkpoint 无法加载 {type(e).__name__}"); continue
            if int(c.get("global_step", -1)) != GSTEP:
                defects.append(f"{tag}: ckpt global_step={c.get('global_step')}")
            names, aud = c.get("source_names"), c.get("admission_audit")
            if arm == "student":
                if names != ["null"]:
                    defects.append(f"{tag}: source_names={names}(应为['null'])")
                if aud is not None:
                    defects.append(f"{tag}: student 不应有 admission_audit")
            else:
                # M28：按 slide bank 的实际输出（null_option:false → ['walk']）判非 null 部分
                non_null = [n for n in (names or []) if n != "null"]
                if non_null != [arm]:
                    defects.append(f"{tag}: source_names={names} 非 null 部分应为 ['{arm}']")
                if aud is None:
                    defects.append(f"{tag}: 缺 admission_audit")
                else:
                    for fld in ("execution_counts", "critic_sample_counts"):
                        v = aud.get(fld)
                        if not isinstance(v, (list, tuple)) or len(v) != 2:
                            defects.append(f"{tag}: {fld} 形状非法"); continue
                        ec = [int(x) for x in v]
                        if min(ec) < 0 or sum(ec) <= 0:
                            defects.append(f"{tag}: {fld} 非负/总和校验失败"); continue
                        sh = ec[0] / sum(ec)
                        if fld == "execution_counts":
                            dose[tag] = round(sh, 4)
                        if not (DOSE_BAND[0] <= sh <= DOSE_BAND[1]):
                            defects.append(f"{tag}: {fld} share={sh:.4f} 越界{DOSE_BAND}")

            try:
                d = json.loads(f.read_text())
                cj, ag, proto, eps = d["checkpoint"], d["aggregate"], d["protocol"], d["episodes"]
            except Exception as e:
                defects.append(f"{tag}: eval json 无效 {type(e).__name__}"); continue
            if ag.get("episode_count") != 128 or len(eps) != 128:
                defects.append(f"{tag}: episode 数不符")
            if cj.get("identity_checked") is not True:
                defects.append(f"{tag}: identity_checked={cj.get('identity_checked')!r}")
            if cj.get("global_step") != GSTEP:
                defects.append(f"{tag}: json global_step 不符")
            if f"slide_bac_{arm}_s{sd}__" not in str(cj.get("path", "")):
                defects.append(f"{tag}: ckpt 路径与臂/seed 不符")
            if cj.get("sha256") != _sha256(hits[0]):
                defects.append(f"{tag}: json sha256 与当前 checkpoint 不一致")
            if d.get("env_name") != ENV_NAME:
                defects.append(f"{tag}: env_name={d.get('env_name')}")
            if proto.get("deterministic") is not True:
                defects.append(f"{tag}: protocol.deterministic 非 True")
            if not proto.get("source_free"):
                defects.append(f"{tag}: protocol.source_free 为空")
            if [e.get("seed") for e in eps] != PANEL_SEEDS:
                defects.append(f"{tag}: episode seed 序列与冻结面板不符")
            if not all(isinstance(e.get("return"), (int, float)) and math.isfinite(e["return"])
                       for e in eps):
                defects.append(f"{tag}: 存在非有限 return")

    # M26：同 seed 内各源臂的 share 差 ≤5pp
    for sd in SEEDS:
        vals = [dose[f"{a}_s{sd}"] for a in SOURCES if f"{a}_s{sd}" in dose]
        if len(vals) == len(SOURCES) and (max(vals) - min(vals)) > MAX_SHARE_SPREAD:
            defects.append(f"s{sd}: 源臂间 share 差 {max(vals)-min(vals):.4f} > {MAX_SHARE_SPREAD}")

    # sha256 两两不同
    seen: dict[str, list[str]] = {}
    for p in sorted(EVAL.glob("*.json")):
        try:
            seen.setdefault(json.loads(p.read_text())["checkpoint"]["sha256"], []).append(p.name)
        except Exception:
            continue
    for k, v in seen.items():
        if len(v) > 1:
            defects.append(f"sha256 重复: {v}")
    return absent, defects, dose


def paired(arm: str, sd: int) -> tuple[float, float]:
    ld = lambda a: json.loads((EVAL / f"{a}_s{sd}_step{GSTEP}.json").read_text())["episodes"]
    es, eb = ld(arm), ld("student")
    d = [x["return"] - y["return"] for x, y in zip(es, eb)]
    return st.mean(d), st.stdev(d) / math.sqrt(len(d))


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
        print("=" * 80); print("VERDICT: VOID_ENGINEERING"); print("=" * 80)
        print(f"  未预期异常: {type(e).__name__}: {e}")
        raise SystemExit(2)


def _run() -> None:
    run_id = os.environ.get("SLGEN_RUN_ID") or _sha256(__file__)[:12]
    report: dict = {"prereg": "docs/experiments/slide_generalizability_v1_prereg_20260731.md",
                    "prereg_commit": "2c1804f", "run_id": run_id,
                    "gate_reference_seeds123": GATE_REF}
    ROOT.mkdir(parents=True, exist_ok=True)
    out = ROOT / "results.json"
    if out.exists():
        out.unlink()

    def dump(verdict: str, lines: list[str], code: int = 0):
        report["verdict"] = verdict
        tmp = out.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(report, indent=2, ensure_ascii=False))
        tmp.replace(out)
        print("=" * 80); print(f"VERDICT: {verdict}"); print("=" * 80)
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

    lines = [f"  层1 工程检查 PASS（seeds 4–6 共 12 臂：剂量/臂间 share 差/臂身份/面板/sha256/协议）"]
    U, SE = {}, {}
    for sd in SEEDS:
        for a in SOURCES:
            U[(a, sd)], SE[(a, sd)] = paired(a, sd)

    rows, hits = [], 0
    for sd in SEEDS:
        per = {a: U[(a, sd)] for a in SOURCES}
        mx = max(per, key=per.get)
        hit = mx == TRUE_ARGMAX
        hits += int(hit)
        rows.append({"seed": sd, "U": {a: round(per[a], 2) for a in SOURCES},
                     "paired_se": {a: round(SE[(a, sd)], 2) for a in SOURCES},
                     "argmax": mx, "hit": hit})
    report["per_seed"] = rows
    report["argmax_hits"] = f"{hits}/3"
    report["mean_U"] = {a: round(st.mean([U[(a, s)] for s in SEEDS]), 2) for a in SOURCES}
    report["learner_ci90"] = {
        a: {"mean": round(st.mean(v := [U[(a, s)] for s in SEEDS]), 2),
            "half_width": round(T_095_DF2 * st.stdev(v) / math.sqrt(3), 2)} for a in SOURCES}
    # 次判据：walk / run 是否仍 3/3 为正（stand 本就 uncertain，不作要求）
    report["secondary_sign"] = {a: f"{sum(1 for s in SEEDS if U[(a, s)] > 0)}/3" for a in SOURCES}

    verdict = "GEN_OK" if hits == 3 else ("GEN_PARTIAL" if hits == 2 else "GEN_FAILED")
    report["chance_pass"] = "(1/3)^3 = 3.7%（三选一，单 K 单判据，无 look-elsewhere）"

    lines.append(f"\n  参照 seeds 1–3（BAC gate）：3/3 argmax = walk")
    lines.append(f"  本轮 seeds 4–6：argmax = walk 命中 {hits}/3")
    lines.append("\n  per-seed U ± paired SE:")
    for r in rows:
        lines.append(f"    s{r['seed']}: " + "  ".join(
            f"{a}={r['U'][a]:+8.2f}±{r['paired_se'][a]:5.2f}" for a in SOURCES)
            + f"   argmax={r['argmax']:5s} {'HIT' if r['hit'] else 'MISS'}")
    lines.append("\n  mean U: " + "  ".join(f"{a}={report['mean_U'][a]:+.2f}" for a in SOURCES))
    lines.append("  次判据 U>0 的 seed 数: " + "  ".join(
        f"{a}={report['secondary_sign'][a]}" for a in SOURCES))
    dump(verdict, lines)


if __name__ == "__main__":
    main()
