"""Phase-1 门 B 容差与 ε 预冻结（bounded bank lease run card §4 门 B）。

从历史 retention 同配置臂（basketball_static、truck admission-all handoff；
均 admission-all + warmup 30k + physical_after_authority）实测门 B 各量，
并冻结判据容差。

**ε 的来源（二十二次复核阻塞 1 更正）**：ε **不是**"观测越界×1.5+硬编码
0.01"（旧版如此，且被误称"采样地板"——错误）。现改为**统计式**：

    ε(N) = sqrt( ln(2M/α) / (2N) )     [Hoeffding 两侧界]

N=各区间 critic 采样总数的**最小值**（保守），M=预注册比较次数，
α=总错误率；结果向上取整到 0.001。剂量带 = 历史 min/max ± 0.02，
**诚实标注为"预注册工程风险预算"**（不是机制仿真推导）。

**输入身份验证（阻塞 2）**：每个 checkpoint 逐项核对 env_name/seed/
global_step/source_bank/admission_mode/mcg_warmup_steps/
admission_replay_handoff/mcg_groups，并将其 SHA256 写入输出。

**80k 口径（小修正）**：历史 basketball retention 只有 30k/60k/90k/final，
**无 80k**——历史侧 80k 记为 `NA`，不构造不存在的数据点；新跑臂按
30/60/80/90/100k 断言。

用法：
  生成候选：  python p1_gate_b_tolerance.py [--out candidate.json]
  正式冻结：  python p1_gate_b_tolerance.py --finalize --candidate <json>
              --expected-candidate-sha256 <sha> --out <frozen.json>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
GENERATOR = Path(__file__).resolve()

# ε 统计式参数（预注册）
M_COMPARISONS = 24      # 新跑臂：2 task × 3 seed × 4 区间
ALPHA = 0.001           # 总错误率
EPS_ROUND_UP = 0.001    # 保守向上取整粒度
DOSE_RISK_BUDGET = 0.02  # 预注册工程风险预算（非机制推导）
# N 必须取**新跑臂**的最小区间采样数（ε 用于判定新跑臂，非历史臂）：
# 新跑 checkpoint 30/60/80/90/100k → 最小区间 10k 步；每步 num_updates×batch。
NUM_UPDATES, BATCH_SIZE = 2, 32768

HIST_CKPTS = ["30000", "60000", "90000", "final"]   # 历史臂实有（无 80k）
NEW_ARM_CKPTS = ["30000", "60000", "80000", "90000", "100000"]  # 新跑臂断言点
SEEDS = [1, 2, 3]

ARMS = {
    "basketball": {
        "num_sources": 9, "env_name": "h1hand-basketball-v0",
        "bank": "configs/source_banks/h1hand_std9_wfix_basketball.yaml",
        "student_logit": 3.5892126423877646,
        "tmpl": "models/h1hand-basketball-v0__h1hand_basketball_adaptive_admission_v1_static_s{s}_20260714T110054Z__{s}_{k}.pt",
    },
    "truck": {
        "num_sources": 4, "env_name": "h1hand-truck-v0",
        "bank": "configs/source_banks/h1hand_hurdle4_wfix_truck.yaml",
        "student_logit": 14.216676716804526,
        "tmpl": "models/h1hand-truck-v0__h1hand_truck_admission_handoff_v1_all_s{s}_20260713THANDOFFV1Z__{s}_{k}.pt",
    },
}
# 两臂共有且必须相等的 ptf_cfg 项（二十三次复核阻塞 2：验证范围须覆盖
# "同配置"声称的全部维度）。
REQUIRED_CFG = {
    "admission_mode": "all",
    "mcg": True,
    "mcg_warmup_mode": "admission_bootstrap",
    "mcg_ablation": "bootstrap_only",
    "mcg_warmup_steps": 30000,
    "mcg_warmup_min_steps": 25,
    "mcg_groups": ["legs_torso", "arms"],
    "admission_replay_handoff": "physical_after_authority",
    "admission_replay_recency_half_life": 0.0,
    "admission_replay_uniform_mix": 1.0,
    "admission_replay_priority_alpha": 0.0,
}
# 仅部分实现版本存在的项：存在则必须相等，缺失则如实记录（truck 的
# 7-13 版早于 adaptive 开发，根本没有这些字段——不得假装验证过）。
CONDITIONAL_CFG = {"admission_adaptive": False}
REQUIRED_ARGS = {
    "num_envs": 128, "batch_size": 32768, "buffer_size": 51200,
    "num_updates": 2, "learning_starts": 10, "total_timesteps": 100000,
    "eval_interval": 5000,
}


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_state() -> tuple[str, bool]:
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                          check=False).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=REPO, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                check=False).stdout.strip())
    return head, dirty


def load_verified(path: Path, task: str, seed: int, ckpt: str) -> tuple[dict, dict]:
    """加载并逐项验证 checkpoint 身份（阻塞 2：不再只信文件名）。"""
    spec = ARMS[task]
    st = torch.load(path, map_location="cpu", weights_only=False)
    a, p = st.get("args") or {}, st.get("ptf_cfg") or {}
    if not isinstance(a, dict) or not isinstance(p, dict):
        raise ValueError(f"{path.name}: args/ptf_cfg must be mappings")
    if a.get("env_name") != spec["env_name"]:
        raise ValueError(f"{path.name}: env_name={a.get('env_name')} != {spec['env_name']}")
    if int(a.get("seed", -1)) != seed:
        raise ValueError(f"{path.name}: seed={a.get('seed')} != {seed}")
    gs = int(st.get("global_step", -1))
    if ckpt != "final" and gs != int(ckpt):
        raise ValueError(f"{path.name}: global_step={gs} != {ckpt}")
    if p.get("source_bank") != spec["bank"]:
        raise ValueError(f"{path.name}: source_bank={p.get('source_bank')} != {spec['bank']}")
    if p.get("admission_student_logit") != spec["student_logit"]:
        raise ValueError(
            f"{path.name}: admission_student_logit={p.get('admission_student_logit')!r} "
            f"!= {spec['student_logit']!r}")
    for k, v in REQUIRED_CFG.items():
        if p.get(k) != v:
            raise ValueError(f"{path.name}: ptf_cfg.{k}={p.get(k)!r} != {v!r}")
    conditional = {}
    for k, v in CONDITIONAL_CFG.items():
        if k in p:
            if p[k] != v:
                raise ValueError(f"{path.name}: ptf_cfg.{k}={p[k]!r} != {v!r}")
            conditional[k] = p[k]
        else:
            conditional[k] = "absent (field did not exist in this implementation vintage)"
    for k, v in REQUIRED_ARGS.items():
        if a.get(k) != v:
            raise ValueError(f"{path.name}: args.{k}={a.get(k)!r} != {v!r}")
    aud = st.get("admission_audit") or {}
    masses = aud.get("candidate_masses") or []
    if len(masses) != spec["num_sources"] + 1:
        raise ValueError(f"{path.name}: candidate_masses len={len(masses)} != {spec['num_sources']+1}")
    ident = {
        "path": str(path.relative_to(REPO)), "sha256": sha(path),
        "env_name": a.get("env_name"), "seed": int(a.get("seed")), "global_step": gs,
        "source_bank": p.get("source_bank"), "admission_mode": p.get("admission_mode"),
        "admission_student_logit": p.get("admission_student_logit"),
        "mcg_warmup_steps": p.get("mcg_warmup_steps"),
        "admission_replay_handoff": p.get("admission_replay_handoff"),
        "conditional_cfg": conditional,
        "candidate_masses": [float(x) for x in masses],
    }
    return aud, ident


def src_share(counts, nsrc: int) -> float:
    tot = sum(counts)
    return sum(counts[:nsrc]) / tot if tot else float("nan")


def compute() -> dict:
    result, identities, min_interval_n = {}, [], math.inf

    for task, spec in ARMS.items():
        nsrc = spec["num_sources"]
        beh30, crit30 = [], []
        q = {k: [] for k in HIST_CKPTS}
        rint: dict[str, list[float]] = {"30_60": [], "60_90": []}
        active_eq_main = True

        masses_seen: dict[str, list] = {}
        for s in SEEDS:
            aud = {}
            for k in HIST_CKPTS:
                aud[k], ident = load_verified(REPO / spec["tmpl"].format(s=s, k=k), task, s, k)
                identities.append(ident)
                # 完整 masses 向量须在同任务全部 checkpoint/seed 间一致
                # （阻塞 2：只比长度与 source 总和不足以证明"同配置"）。
                masses_seen[f"s{s}_{k}"] = ident["candidate_masses"]
            a30 = aud["30000"]
            beh30.append(src_share(a30["execution_counts"], nsrc))
            crit30.append(src_share(a30["critic_sample_counts"], nsrc))
            for k in HIST_CKPTS:
                q[k].append(src_share(aud[k]["main_buffer_counts"], nsrc))
                if list(aud[k]["active_buffer_counts"]) != list(aud[k]["main_buffer_counts"]):
                    active_eq_main = False
            for lbl, (a, b) in (("30_60", ("30000", "60000")), ("60_90", ("60000", "90000"))):
                d = [x - y for x, y in zip(aud[b]["critic_sample_counts"], aud[a]["critic_sample_counts"])]
                rint[lbl].append(src_share(d, nsrc))
                min_interval_n = min(min_interval_n, sum(d))

        distinct_masses = {tuple(v) for v in masses_seen.values()}
        if len(distinct_masses) != 1:
            raise ValueError(
                f"{task}: candidate_masses vector differs across checkpoints/seeds: "
                + json.dumps({k: v[:3] for k, v in masses_seen.items()}, indent=2))

        def rng(v):
            fin = [x for x in v if x == x]
            return {"min": min(fin), "max": max(fin), "mean": sum(fin) / len(fin)} if fin else None

        envelope_exceed = []
        for lbl, (a, b) in (("30_60", ("30000", "60000")), ("60_90", ("60000", "90000"))):
            for i in range(len(SEEDS)):
                r, qa, qb = rint[lbl][i], q[a][i], q[b][i]
                lo, hi = min(qa, qb), max(qa, qb)
                envelope_exceed.append(max(0.0, r - hi, lo - r))

        result[task] = {
            "num_sources": nsrc,
            "behavior_source_share_30k": rng(beh30),
            "critic_source_share_30k": rng(crit30),
            "q_physical_share": {**{k: rng(q[k]) for k in HIST_CKPTS},
                                 "80000": "NA (historical arms have no 80k checkpoint)"},
            "r_interval_critic_share": {k: rng(v) for k, v in rint.items()},
            "active_equals_main_all_ckpts": active_eq_main,
            "envelope_exceedance_observed_max": max(envelope_exceed) if envelope_exceed else 0.0,
            "candidate_masses_identical_across_all_ckpts_and_seeds": True,
            "candidate_masses": list(next(iter(distinct_masses))),
            "dose_band_engineering_risk_budget": {
                "note": "historical min/max ± 0.02; a pre-registered ENGINEERING RISK BUDGET, not a mechanism-simulation derivation",
                "behavior_source_share_30k": [round(min(beh30) - DOSE_RISK_BUDGET, 4),
                                              round(max(beh30) + DOSE_RISK_BUDGET, 4)],
                "critic_source_share_30k": [round(min(crit30) - DOSE_RISK_BUDGET, 4),
                                            round(max(crit30) + DOSE_RISK_BUDGET, 4)],
            },
        }

    # N 取新跑臂最小区间（10k 步），非历史臂实测区间——ε 用于判定新跑臂。
    new_steps = [int(b) - int(a) for a, b in zip(NEW_ARM_CKPTS[:-1], NEW_ARM_CKPTS[1:])]
    n_new_min = min(new_steps) * NUM_UPDATES * BATCH_SIZE
    eps_raw = math.sqrt(math.log(2 * M_COMPARISONS / ALPHA) / (2 * n_new_min))
    eps_frozen = math.ceil(eps_raw / EPS_ROUND_UP) * EPS_ROUND_UP
    head, dirty = git_state()
    return {
        "epsilon": {
            "formula": "eps(N) = sqrt(ln(2M/alpha) / (2N))  [Hoeffding two-sided]",
            "M_comparisons": M_COMPARISONS, "alpha": ALPHA,
            "N_min_interval_critic_samples": int(n_new_min),
            "N_derivation": (
                f"new-arm min interval {min(new_steps)} steps x num_updates {NUM_UPDATES} "
                f"x batch {BATCH_SIZE}; historical-arm min interval was {int(min_interval_n)} "
                "(recorded for reference only, NOT used for eps)"
            ),
            "N_historical_min_reference": int(min_interval_n),
            "eps_raw": eps_raw, "eps_frozen": round(eps_frozen, 6),
            "note": "statistical sampling tolerance only; NOT an engineering margin",
        },
        "ckpt_schedule": {"historical_arms": HIST_CKPTS, "new_arms": NEW_ARM_CKPTS,
                          "note": "historical basketball/truck retention have no 80k checkpoint; 80k asserted only on new arms"},
        "hard_exit_asserts": {
            "post_30k_source_execution_increment": 0,
            "post_30k_source_critic_increment": 0,
            "60k_80k_main_source_gt0_active_source_eq0": True,
            "90k_100k_no_main_source_requirement": "physical data already covered to zero by 81.2k",
        },
        "per_task": result,
        "input_identities": identities,
        "generator": str(GENERATOR.relative_to(REPO)),
        "generator_sha256": sha(GENERATOR),
        "git_head": head, "git_dirty": dirty,
    }


DYNAMIC_FIELDS = {"status", "git_head", "git_dirty", "finalized_from_candidate"}


def _payload_diff(cand: dict, fresh: dict) -> list[str]:
    """返回科学字段中不一致的键（排除运行时动态字段）。"""
    keys = (set(cand) | set(fresh)) - DYNAMIC_FIELDS
    return sorted(k for k in keys if cand.get(k) != fresh.get(k))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(REPO / "docs/data/p1_bounded_bank_lease/gate_b_tolerance_candidate.json"))
    ap.add_argument("--finalize", action="store_true")
    ap.add_argument("--candidate")
    ap.add_argument("--expected-candidate-sha256")
    args = ap.parse_args()

    out_path = Path(args.out)
    if out_path.exists():
        raise FileExistsError(f"{out_path} exists; refusing to overwrite")
    payload = compute()

    if args.finalize:
        if not args.candidate or not args.expected_candidate_sha256:
            raise SystemExit("--finalize requires --candidate and --expected-candidate-sha256")
        cand_path = Path(args.candidate)
        cand_sha = sha(cand_path)
        if cand_sha != args.expected_candidate_sha256:
            raise ValueError(f"candidate sha mismatch: {cand_sha} != {args.expected_candidate_sha256}")
        if payload["git_dirty"]:
            raise RuntimeError("working tree is dirty; finalize requires a clean tree")
        cand = json.loads(cand_path.read_text())
        # 完整科学 payload 逐位比较（二十三次复核阻塞 1：只比汇总数时，
        # 输入 checkpoint / generator / 身份被替换但汇总恰好不变仍会通过）。
        diff = _payload_diff(cand, payload)
        if diff:
            raise ValueError(
                "recomputed scientific payload differs from candidate; refusing to "
                f"finalize. differing keys: {diff}")
        payload["status"] = "frozen"
        payload["finalized_from_candidate"] = {"path": str(cand_path), "sha256": cand_sha}
    else:
        payload["status"] = "candidate"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    e = payload["epsilon"]
    print(f"[{payload['status']}] ε_raw={e['eps_raw']:.3e} ε_frozen={e['eps_frozen']} "
          f"(N_min={e['N_min_interval_critic_samples']:,}) -> {out_path}")
    for t, r in payload["per_task"].items():
        print(f"  {t}: exceed_max={r['envelope_exceedance_observed_max']:.2e} "
              f"beh30={r['behavior_source_share_30k']['mean']:.3f} "
              f"crit30={r['critic_source_share_30k']['mean']:.3f}")


if __name__ == "__main__":
    main()
