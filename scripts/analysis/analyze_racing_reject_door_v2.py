"""RACING_REJECT v2 裁决：racing 能否**提前**做出正确的拒绝决策。

判据冻结于 docs/experiments/racing_reject_door_v2_prereg_20260730.md
（v2 = ac1bd26，修订 v2.1 = 971b6af）。本脚本只实现，不得事后调整。

    decide(K) = REJECT  ⟺  max_i U_i(K) ≤ 0            （R1 零点消歧，以 §2 主定义为准）
    H：存在 K ∈ {2000,5000} 使 decide(K)=REJECT 在 批1 3/3 且 批2 3/3 上成立

优先级（§5.4）：VOID_ENGINEERING > REPLICATION_DIVERGED > PARTICIPANT_DIVERGED > 主终点
异常分类（R4）：缺产物 → INCOMPLETE；产物存在但无效 → VOID_ENGINEERING；混合 → INCOMPLETE 优先

盲态封闭（R2）：层1 的**输出与控制流**不得含/依赖任何 return 派生量。
U 只在层1 全部通过后才计算。
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

ROOT = Path("docs/data/racing_reject_door_v2")
BATCHES = {
    "batch1": {"seeds": (1, 2, 3), "prefix": "rjd",
               "eval": Path("docs/data/racing_reject_door_v1/source_free_eval")},
    "batch2": {"seeds": (4, 5, 6), "prefix": "rjd2",
               "eval": Path("docs/data/racing_reject_door_v2/source_free_eval")},
}
TRAIN_LOG = Path("logs/train/racing_reject_door_v1")
ANCHOR, ENV_NAME = 10000, "h1hand-door-v0"
KS = (2000, 5000, 10000)
SOURCES = ("stand", "walk", "run")
ARMS = ("student",) + SOURCES
DOSE_BAND = (0.48, 0.52)
REPL_SE_MULT = 3.0
T_095_DF2 = 2.919986
PANEL_SEEDS = [s * 1000 + r for s in
               (11, 23, 37, 53, 71, 89, 103, 113, 131, 149, 163, 179, 193, 211, 227, 241)
               for r in range(8)]          # 冻结面板：16 eval seeds × 8 ranks = 128

GT_PER_SEED = {
    "stand": {1: -23.43, 2: -42.83, 3: -31.66},
    "walk": {1: -7.06, 2: -38.58, 3: -20.96},
    "run": {1: -17.78, 2: -41.04, 3: -33.08},
}


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


# ---------- 层 1 · 工程硬检查（输出与控制流均不含 return 派生量，R2） ----------

def layer1() -> tuple[list[str], list[str], dict]:
    absent: list[str] = []
    defects: list[str] = []
    dose: dict[str, float] = {}

    def ck_defect(tag: str, msg: str):
        defects.append(f"{tag}: {msg}")

    for bname, b in BATCHES.items():
        for sd in b["seeds"]:
            for arm in ARMS:
                tag0 = f"{bname}/{arm}_s{sd}"
                # 训练日志的 anchor 恢复行（R3）
                lg = TRAIN_LOG / f"{b['prefix']}_{arm}_s{sd}.log"
                if not lg.exists():
                    absent.append(f"{tag0}: 缺训练日志")
                else:
                    txt = lg.read_text(errors="ignore")
                    if f"Resumed core learner" not in txt or f"at step {ANCHOR}" not in txt:
                        ck_defect(tag0, f"训练日志缺 'Resumed core learner ... at step {ANCHOR}'")

                for k in KS:
                    g = ANCHOR + k
                    tag = f"{tag0}_{g}"
                    hits = glob.glob(f"models/*{b['prefix']}_{arm}_s{sd}__*_{g}.pt")
                    if not hits:
                        absent.append(f"{tag}: 缺 checkpoint")
                    elif len(hits) > 1:
                        ck_defect(tag, f"checkpoint 命中 {len(hits)} 个(应为1)")
                    f = b["eval"] / f"{arm}_s{sd}_step{g}.json"
                    if not f.exists():
                        absent.append(f"{tag}: 缺 eval json")
                    if not hits or not f.exists():
                        continue
                    # ---- checkpoint 校验 ----
                    try:
                        c = torch.load(hits[0], map_location="cpu", weights_only=False)
                    except Exception as e:
                        ck_defect(tag, f"checkpoint 无法加载: {type(e).__name__}"); continue
                    if int(c.get("global_step", -1)) != g:
                        ck_defect(tag, f"ckpt global_step={c.get('global_step')}")
                    names, aud = c.get("source_names"), c.get("admission_audit")
                    if arm == "student":
                        if names != ["null"]:
                            ck_defect(tag, f"source_names={names}(应为['null'])")
                        if aud is not None:
                            ck_defect(tag, "student 臂不应有 admission_audit")
                    else:
                        # R8：判"非 null 部分恰为 [arm]"，与 bank 的 null_option 配置无关。
                        # door 的 bank 是 null_option:false → ['stand']；
                        # hurdle 是 true → ['run','null']。原写死 [arm,"null"] 是把
                        # hurdle 的模式套到了 door 上。本式仍精确防臂对调。
                        non_null = [n for n in (names or []) if n != "null"]
                        if non_null != [arm]:
                            ck_defect(tag, f"source_names={names} 的非 null 部分应恰为 ['{arm}']")
                        if aud is None:
                            ck_defect(tag, "缺 admission_audit")
                        else:
                            for fld, band_check in (("execution_counts", True),
                                                    ("critic_sample_counts", True)):
                                v = aud.get(fld)
                                if not isinstance(v, (list, tuple)) or len(v) != 2:
                                    ck_defect(tag, f"{fld} 形状非法: {v}"); continue
                                try:
                                    ec = [int(x) for x in v]
                                except Exception:
                                    ck_defect(tag, f"{fld} 非整数"); continue
                                if min(ec) < 0 or sum(ec) <= 0:
                                    ck_defect(tag, f"{fld} 非负/总和校验失败: {ec}"); continue
                                sh = ec[0] / sum(ec)
                                if fld == "execution_counts":
                                    dose[tag] = round(sh, 4)
                                if band_check and not (DOSE_BAND[0] <= sh <= DOSE_BAND[1]):
                                    ck_defect(tag, f"{fld} share={sh:.4f} 越界{DOSE_BAND}")
                    # ---- eval json 校验（只碰身份/协议字段） ----
                    try:
                        d = json.loads(f.read_text())
                    except Exception as e:
                        ck_defect(tag, f"eval json 损坏: {type(e).__name__}"); continue
                    try:
                        cj, ag, proto, eps = d["checkpoint"], d["aggregate"], d["protocol"], d["episodes"]
                    except KeyError as e:
                        ck_defect(tag, f"eval json 缺字段 {e}"); continue
                    if ag.get("episode_count") != 128 or len(eps) != 128:
                        ck_defect(tag, f"episode_count={ag.get('episode_count')} len={len(eps)}")
                    if cj.get("identity_checked") is not True:
                        ck_defect(tag, f"identity_checked={cj.get('identity_checked')!r}(须为 True)")
                    if cj.get("global_step") != g:
                        ck_defect(tag, "json global_step 不符")
                    if f"{b['prefix']}_{arm}_s{sd}__" not in str(cj.get("path", "")):
                        ck_defect(tag, "json ckpt 路径与臂/seed 不符")
                    if cj.get("sha256") != _sha256(hits[0]):     # R3：与实际 ckpt 绑定
                        ck_defect(tag, "json sha256 与当前 checkpoint 不一致(旧评估?)")
                    if d.get("env_name") != ENV_NAME:
                        ck_defect(tag, f"env_name={d.get('env_name')}")
                    if proto.get("deterministic") is not True:
                        ck_defect(tag, "protocol.deterministic 非 True")
                    if not proto.get("source_free"):
                        ck_defect(tag, "protocol.source_free 为空")
                    if [e.get("seed") for e in eps] != PANEL_SEEDS:
                        ck_defect(tag, "episode seed 序列与冻结面板不符")
                    # 有效性判定（R2'）：只提取 1 bit（有限/非有限），
                    # 不依赖也不输出 return 的数值大小、排序或聚合。
                    # 不做此检查则 NaN 会污染 max/比较（NaN 的比较恒为 False），静默改判。
                    if not all(isinstance(e.get("return"), (int, float))
                               and math.isfinite(e["return"]) for e in eps):
                        ck_defect(tag, "存在非有限 return")

    # sha256 两两不同
    seen: dict[str, list[str]] = {}
    for bname, b in BATCHES.items():
        for p in sorted(b["eval"].glob("*.json")):
            try:
                s = json.loads(p.read_text())["checkpoint"]["sha256"]
            except Exception:
                continue
            seen.setdefault(s, []).append(f"{bname}/{p.name}")
    for s, v in seen.items():
        if len(v) > 1:
            defects.append(f"sha256 重复: {v}")
    return absent, defects, dose


# ---------- 结果读取（层1 全部通过后才允许） ----------

def paired(bname: str, arm: str, sd: int, g: int) -> tuple[float, float]:
    """逐 episode 配对差的 (mean, SE)；面板已在层1 校验为逐位一致。"""
    ld = lambda a: json.loads((BATCHES[bname]["eval"] / f"{a}_s{sd}_step{g}.json").read_text())["episodes"]
    es, eb = ld(arm), ld("student")
    d = [a["return"] - b["return"] for a, b in zip(es, eb)]
    return st.mean(d), st.stdev(d) / math.sqrt(len(d))


def main() -> None:
    """兜底：任何未预期异常都归入 VOID_ENGINEERING，绝不裸崩（静态自查发现 12 处潜在裸抛点）。"""
    try:
        _run()
    except SystemExit:
        raise                                    # dump() 的正常退出路径
    except Exception as e:
        ROOT.mkdir(parents=True, exist_ok=True)
        out = ROOT / "results.json"
        rep = {"verdict": "VOID_ENGINEERING",
               "unexpected_exception": f"{type(e).__name__}: {e}",
               "note": "未预期异常，按预注册 R4 归入 VOID_ENGINEERING；不输出任何主结果"}
        tmp = out.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(rep, indent=2, ensure_ascii=False)); tmp.replace(out)
        print("=" * 84); print("VERDICT: VOID_ENGINEERING"); print("=" * 84)
        print(f"  未预期异常: {type(e).__name__}: {e}")
        raise SystemExit(2)


def _run() -> None:
    run_id = os.environ.get("RJD2_RUN_ID") or _sha256(__file__)[:12]
    report: dict = {"prereg": "docs/experiments/racing_reject_door_v2_prereg_20260730.md",
                    "prereg_commits": ["ac1bd26", "971b6af"], "run_id": run_id,
                    "reject_rule": "REJECT ⟺ max_i U_i(K) <= 0  (R1)"}
    ROOT.mkdir(parents=True, exist_ok=True)
    out = ROOT / "results.json"
    if out.exists():
        out.unlink()                                   # R5：先删旧文件，杜绝陈旧泄漏

    def dump(verdict: str, lines: list[str], code: int = 0):
        report["verdict"] = verdict
        tmp = out.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(report, indent=2, ensure_ascii=False))
        tmp.replace(out)                               # R5：原子替换
        print("=" * 84); print(f"VERDICT: {verdict}"); print("=" * 84)
        for ln in lines:
            print(ln)
        print(f"\nrun_id={run_id}  written {out}")
        raise SystemExit(code)

    # ---- 层 1 ----
    absent, defects, dose = layer1()
    report["dose"] = dose
    if absent:                                          # R4：缺产物优先判 INCOMPLETE
        report["absent_count"] = len(absent)
        dump("INCOMPLETE", [f"  尚缺 {len(absent)} 项产物（工程未完成）"]
             + [f"    {a}" for a in absent[:8]]
             + ([f"    ...(另有 {len(absent)-8} 项)"] if len(absent) > 8 else [])
             + ([f"  另检出 {len(defects)} 项缺陷，待产物齐备后复判"] if defects else []), 2)
    if defects:
        report["defects"] = defects
        dump("VOID_ENGINEERING", [f"  {d}" for d in defects[:20]]
             + ([f"  ...(共 {len(defects)} 条)"] if len(defects) > 20 else []), 2)
    lines = [f"  层1 工程检查 PASS（24 臂 × 3 K，剂量/臂身份/面板/sha256/协议 全通过）"]

    # ---- U ----
    U, PSE = {}, {}
    for bname, b in BATCHES.items():
        for k in KS:
            for sd in b["seeds"]:
                for a in SOURCES:
                    U[(bname, k, a, sd)], PSE[(bname, k, a, sd)] = paired(bname, a, sd, ANCHOR + k)

    # ---- 层 2 · 批1 复制检查（仅 K=10000）----
    repl, ok2 = {}, True
    for a in SOURCES:
        for sd in BATCHES["batch1"]["seeds"]:
            un, se, ug = U[("batch1", 10000, a, sd)], PSE[("batch1", 10000, a, sd)], GT_PER_SEED[a][sd]
            p = (un < 0) == (ug < 0) and abs(un - ug) <= REPL_SE_MULT * se
            ok2 = ok2 and p
            repl[f"{a}_s{sd}"] = {"U_new": round(un, 2), "U_gate": ug, "paired_se": round(se, 2),
                                  "tol": round(REPL_SE_MULT * se, 2), "pass": p}
    report["layer2_replication"] = {"per_cell": repl, "pass": ok2}
    lines.append("\n  层2 批1 复制检查（K=10000 vs gate 已发表）:")
    lines += [f"    {k:10s} 本次={v['U_new']:+8.2f} gate={v['U_gate']:+8.2f} "
              f"容差=±{v['tol']:.2f}  {'PASS' if v['pass'] else 'FAIL'}" for k, v in repl.items()]
    if not ok2:
        dump("REPLICATION_DIVERGED", lines, 3)

    # ---- 层 3 · 批2 批内参照 ----
    b2, ok3 = {}, True
    for sd in BATCHES["batch2"]["seeds"]:
        per = {a: U[("batch2", 10000, a, sd)] for a in SOURCES}
        mx = max(per, key=per.get)
        rej = per[mx] <= 0
        ok3 = ok3 and rej
        b2[f"s{sd}"] = {"U": {a: round(per[a], 2) for a in SOURCES}, "argmax": mx,
                        "max_U": round(per[mx], 2), "reject": rej}
    report["layer3_batch2_reference"] = {"per_seed": b2, "pass": ok3}
    lines.append("\n  层3 批2 批内参照（K=10000 决策须为 REJECT）:")
    lines += [f"    {k}: " + "  ".join(f"{a}={v['U'][a]:+7.2f}" for a in SOURCES)
              + f"   max={v['argmax']}({v['max_U']:+.2f})  " + ("REJECT" if v["reject"] else "**未拒绝**")
              for k, v in b2.items()]
    if not ok3:
        dump("PARTICIPANT_DIVERGED", lines, 4)

    # ---- 主终点 ----
    per_K = {}
    for k in KS:
        e = {}
        for bname, b in BATCHES.items():
            rows, hits = [], 0
            for sd in b["seeds"]:
                per = {a: U[(bname, k, a, sd)] for a in SOURCES}
                mx = max(per, key=per.get)
                rej = per[mx] <= 0                      # R1
                hits += int(rej)
                rows.append({"seed": sd, "U": {a: round(per[a], 2) for a in SOURCES},
                             "argmax": mx, "max_U": round(per[mx], 2), "reject": rej})
            e[bname] = {"per_seed": rows, "reject_hits": f"{hits}/3", "all_reject": hits == 3}
        e["both_batches_3of3"] = e["batch1"]["all_reject"] and e["batch2"]["all_reject"]
        e["learner_ci90"] = {bn: {a: {"mean": round(st.mean(v := [U[(bn, k, a, s)] for s in bb["seeds"]]), 2),
                                      "half_width": round(T_095_DF2 * st.stdev(v) / math.sqrt(3), 2)}
                                  for a in SOURCES} for bn, bb in BATCHES.items()}
        per_K[str(k)] = e
    report["per_K"] = per_K

    if per_K["5000"]["both_batches_3of3"]:
        v = "EARLY_REJECT_CONFIRMED"
    elif per_K["2000"]["both_batches_3of3"]:
        v = "EARLY_REJECT_NONMONOTONIC"
    else:
        v = "EARLY_REJECT_REFUTED"
    report["cost"] = {"reject_cost_steps": 3 * (5000 if v == "EARLY_REJECT_CONFIRMED"
                                                else 2000 if v == "EARLY_REJECT_NONMONOTONIC" else 10000),
                      "note": "避损口径，不得与 hurdle 的加速收益合并比较（§8.3）"}
    report["chance_pass_upper_bound"] = "≈1/32 (3.1%)：复合零假设 (1/2)^6 × look-elsewhere(2 个 K)"
    for k in KS:
        e = per_K[str(k)]
        lines.append(f"\n  --- K={k} ---  批1 {e['batch1']['reject_hits']}  批2 {e['batch2']['reject_hits']}"
                     + ("   两批各 3/3" if e["both_batches_3of3"] else ""))
        for bn in ("batch1", "batch2"):
            lines += [f"    {bn} s{r['seed']}: " + "  ".join(f"{a}={r['U'][a]:+7.2f}" for a in SOURCES)
                      + f"   max={r['argmax']:5s}  " + ("REJECT" if r["reject"] else "**未拒绝**")
                      for r in e[bn]["per_seed"]]
    dump(v, lines)


if __name__ == "__main__":
    main()
