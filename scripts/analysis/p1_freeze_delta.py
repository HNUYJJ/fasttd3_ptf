"""Phase-1 δ 冻结脚本（bounded bank lease run card §7.3）。

δ_task = 0.5 × (历史 scratch 的 35k–80k normalized trapezoidal AUC 跨
3-seed SD)。**δ 定位**：externally anchored SESOI / practical margin
（最小关切效应量）——来自历史、同任务、同 return 口径的 scratch 曲线；
**不是**当前实现的数值噪声地板，也**不是**新实验的方差估计。

自动验证（二十二次复核阻塞 3——不接受硬编码断言文本）：
  1. 每个 exp_name 在 wandb 中**恰好匹配一个** run；
  2. W&B output.log 的 nAUC 与 logs/train 日志逐条相同（同一次运行）；
  3. 六份历史入口 train_ptf.py 快照 SHA256 全同、git base 全同；
  4. CLI 纯 scratch **且**历史 PTF 配置实际为
     `mcg=False / execute_sources=False / source_bank 空`（只查 CLI 无
     `--ptf` 不足以证明 PTF 未启用）；
  5. **六份 diff.patch 逐段比较**：剔除 `scripts/probe_transfer_map_v2.py`
     段后必须完全一致（truck s1 的差异断言不再是固定文本）。

用法：
  生成候选：  python p1_freeze_delta.py [--out candidate.json]
  正式冻结：  python p1_freeze_delta.py --finalize --candidate <json>
              --expected-candidate-sha256 <sha> --out <frozen.json>
两者均拒绝覆盖；finalize 额外要求工作树干净并重新验证全部输入。
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import re
import subprocess
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[2]
GENERATOR = Path(__file__).resolve()
WINDOW = (35000, 80000)
EVAL_RE = re.compile(r"\[eval\] step=(\d+) return=(-?\d+(?:\.\d+)?)")
PROBE_PATH = "scripts/probe_transfer_map_v2.py"

SCRATCH = {
    ("basketball", 1): ("logs/train/b2_20260705T153732Z/b2_basketball_scr_s1.log",
                        "h1hand_basketball_b2_scr_s1_20260705T153732Z"),
    ("basketball", 2): ("logs/train/b2s_20260705T224905Z/b2s_basketball_scr_s2.log",
                        "h1hand_basketball_b2_scr_s2_20260705T153732Z"),
    ("basketball", 3): ("logs/train/b2s_20260705T224905Z/b2s_basketball_scr_s3.log",
                        "h1hand_basketball_b2_scr_s3_20260705T153732Z"),
    ("truck", 1): ("logs/train/br_20260704T015912Z/br_truck_scr_s1.log",
                   "h1hand_truck_br_scr_s1_20260704T015912Z"),
    ("truck", 2): ("logs/train/brseed_20260704T135105Z/bs_truck_scr_s2.log",
                   "h1hand_truck_br_scr_s2_20260704T135105Z"),
    ("truck", 3): ("logs/train/brseed_20260704T135105Z/bs_truck_scr_s3.log",
                   "h1hand_truck_br_scr_s3_20260704T135105Z"),
}
HIST_ENTRY = "files/code/fasttd3_ptf/official_fasttd3_ptf/train_ptf.py"
PTF_SCRATCH_REQUIRED = {"mcg": False, "execute_sources": False}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def git_state() -> tuple[str, bool]:
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                          check=False).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=REPO, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                check=False).stdout.strip())
    return head, dirty


def nauc(path: Path) -> float:
    lo, hi = WINDOW
    pts = {}
    for line in path.read_text(errors="replace").splitlines():
        m = EVAL_RE.search(line)
        if m and lo <= int(m.group(1)) <= hi:
            pts[int(m.group(1))] = float(m.group(2))
    xs = sorted(pts)
    if len(xs) != 10:
        raise ValueError(f"{path}: expected 10 points in [35k,80k], got {len(xs)}")
    return float(np.trapz([pts[x] for x in xs], xs) / (hi - lo))


def find_wandb_run(exp_name: str) -> Path:
    """必须恰好匹配一个 run（阻塞 3：不接受 glob 首个命中）。"""
    hits = []
    for md in glob.glob(str(REPO / "wandb/run-*/files/wandb-metadata.json")):
        try:
            d = json.loads(Path(md).read_text())
        except Exception:  # noqa: BLE001
            continue
        a = d.get("args", [])
        if "--exp_name" in a and a[a.index("--exp_name") + 1] == exp_name:
            hits.append(Path(md).parent.parent)
    if len(hits) != 1:
        raise ValueError(f"exp_name={exp_name} matched {len(hits)} wandb runs (need exactly 1)")
    return hits[0]


def patch_sections_excluding_probe(patch_path: Path) -> str:
    """按 `diff --git` 切段，剔除离线 probe 段后拼回（用于跨 run 比较）。"""
    text = patch_path.read_text(errors="replace")
    parts = re.split(r"(?m)^(?=diff --git )", text)
    kept = [p for p in parts if p.strip() and PROBE_PATH not in p.split("\n", 1)[0]]
    return "".join(kept)


def assert_pure_scratch_config(run_dir: Path, exp_name: str) -> dict:
    """CLI 纯净 + 历史 PTF 实际配置为 scratch（阻塞 3 第 4 点）。"""
    meta = json.loads((run_dir / "files/wandb-metadata.json").read_text())
    cli = meta.get("args", [])
    bad = [x for x in cli if x.startswith("--ptf") or "bank" in x or "admission" in x or "mcg" in x]
    if bad:
        raise ValueError(f"{exp_name}: CLI is not pure scratch: {bad}")
    cfg = yaml.safe_load((run_dir / "files/config.yaml").read_text())
    ptf = cfg.get("ptf", {})
    ptf = ptf.get("value", ptf) if isinstance(ptf, dict) else {}
    for key, expected in PTF_SCRATCH_REQUIRED.items():
        if ptf.get(key) is not expected:
            raise ValueError(f"{exp_name}: historical ptf.{key}={ptf.get(key)!r}, expected {expected!r}")
    bank = cfg.get("source_bank", cfg.get("ptf_source_bank"))
    bank = bank.get("value", bank) if isinstance(bank, dict) else bank
    if bank:
        raise ValueError(f"{exp_name}: historical source_bank is not empty: {bank!r}")
    return {"cli": cli, "ptf_mcg": ptf.get("mcg"), "ptf_execute_sources": ptf.get("execute_sources"),
            "source_bank": bank, "git_base": meta.get("git", {}).get("commit", "")}


def compute() -> dict:
    entries, train_ptf_shas, git_bases, patch_bodies = {}, set(), set(), {}
    per_task: dict[str, list[float]] = {"basketball": [], "truck": []}

    for (task, seed), (log_rel, exp_name) in SCRATCH.items():
        log_path = REPO / log_rel
        run_dir = find_wandb_run(exp_name)
        ident = assert_pure_scratch_config(run_dir, exp_name)
        n_log, n_wandb = nauc(log_path), nauc(run_dir / "files/output.log")
        if abs(n_log - n_wandb) > 1e-6:
            raise ValueError(f"{exp_name}: wandb/logs nAUC mismatch {n_wandb} vs {n_log}")
        train_ptf_shas.add(sha(run_dir / HIST_ENTRY))
        git_bases.add(ident["git_base"])
        patch_bodies[exp_name] = hashlib.sha256(
            patch_sections_excluding_probe(run_dir / "files/diff.patch").encode()).hexdigest()
        per_task[task].append(n_log)
        entries[f"{task}_s{seed}"] = {
            "nauc_35k_80k": n_log, "exp_name": exp_name,
            "train_log": log_rel, "train_log_sha256": sha(log_path),
            "wandb_run": run_dir.name,
            "wandb_config_sha256": sha(run_dir / "files/config.yaml"),
            "wandb_metadata_sha256": sha(run_dir / "files/wandb-metadata.json"),
            "wandb_hist_train_ptf_sha256": sha(run_dir / HIST_ENTRY),
            "wandb_diff_patch_sha256": sha(run_dir / "files/diff.patch"),
            "wandb_diff_excl_probe_sha256": patch_bodies[exp_name],
            "wandb_output_sha256": sha(run_dir / "files/output.log"),
            "ptf_mcg": ident["ptf_mcg"], "ptf_execute_sources": ident["ptf_execute_sources"],
        }

    if len(train_ptf_shas) != 1:
        raise ValueError(f"historical train_ptf.py SHAs differ: {train_ptf_shas}")
    if len(git_bases) != 1:
        raise ValueError(f"git base commits differ: {git_bases}")
    distinct_patches = set(patch_bodies.values())
    if len(distinct_patches) != 1:
        raise ValueError(
            "diff.patch bodies differ beyond the offline probe: "
            + json.dumps({k: v[:12] for k, v in patch_bodies.items()}, indent=2)
        )

    deltas = {}
    for task, vals in per_task.items():
        sd = float(np.std(vals, ddof=1))
        deltas[task] = {"cross_seed_sd_nauc": sd, "delta_sesoi": 0.5 * sd}

    head, dirty = git_state()
    return {
        "definition": (
            "delta_task = 0.5 * cross-seed SD of historical scratch 35k-80k normalized "
            "trapezoidal AUC; externally anchored SESOI / practical margin (NOT a "
            "numerical noise floor, NOT a new-experiment variance estimate)"
        ),
        "window": list(WINDOW),
        "deltas": deltas,
        "verified_asserts": {
            "each_exp_name_matches_exactly_one_wandb_run": True,
            "wandb_logs_nauc_match": True,
            "historical_train_ptf_sha256": next(iter(train_ptf_shas)),
            "git_base": next(iter(git_bases)),
            "cli_pure_scratch": True,
            "historical_ptf_mcg_false_execute_sources_false_bank_empty": True,
            "diff_patch_identical_excluding_offline_probe": True,
            "diff_excl_probe_sha256": next(iter(distinct_patches)),
        },
        "scratch_dual_verdict": {
            "CAUSAL_COMPARATOR": "REUSE_FAIL (scratch @ b183f40 != hard-exit @ HEAD; scratch labels stay descriptive)",
            "METRIC_SCALE": "REUSE_PASS (evaluator/task/return-unit/5k-grid/entry-code provable; SD usable as external margin)",
        },
        "sources": entries,
        "generator": str(GENERATOR.relative_to(REPO)),
        "generator_sha256": sha(GENERATOR),
        "git_head": head,
        "git_dirty": dirty,
    }


DYNAMIC_FIELDS = {"status", "git_head", "git_dirty", "finalized_from_candidate"}


def _payload_diff(cand: dict, fresh: dict) -> list[str]:
    """返回科学字段中不一致的键（排除运行时动态字段）。"""
    keys = (set(cand) | set(fresh)) - DYNAMIC_FIELDS
    return sorted(k for k in keys if cand.get(k) != fresh.get(k))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(REPO / "docs/data/p1_bounded_bank_lease/delta_candidate.json"))
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
        # 完整科学 payload 逐位比较（阻塞 1）：sources/verified_asserts/
        # generator SHA 等任一变化都必须拒绝，即使 δ 汇总值恰好不变。
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
    d = payload["deltas"]
    print(f"[{payload['status']}] basketball δ={d['basketball']['delta_sesoi']:.6f}  "
          f"truck δ={d['truck']['delta_sesoi']:.6f}  -> {out_path}")


if __name__ == "__main__":
    main()
