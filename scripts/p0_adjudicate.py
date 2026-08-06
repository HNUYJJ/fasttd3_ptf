"""P0 裁决器(run card §5/§7/§11;五次复核阻塞问题 2)。

字面执行冻结协议,无自由参数:
- 统计:配对 U_s = J_lease,s − J_abstain,s(s=1..3);one-sided t,df=2,
  t₀.₉₀,₂ = 1.8856(写死);bound = mean ± 1.8856·SD/√3;
- per-task 五分类(v2.1.2 判序,d_dup 前置):
  1. d_dup ≥ δ           → UNCERTAIN_NUMERIC(UNCERTAIN 子类,数值地板);
  2. HETEROGENEOUS        → 存在 |U_s|>δ 且符号相反的 seed 对;
  3. POSITIVE / NEGATIVE  → LCB>+δ / UCB<−δ,crawl 须 progress 同向;
  4. NULL                 → CI ⊂ (−δ,+δ);
  5. UNCERTAIN            → 其余(CI 跨界/crawl progress 反向);
  边界(恰好等于阈值)一律归 UNCERTAIN;
- joint 六步判序(§7):工程无效 → HET(F-d) → UNCERTAIN(F-a) →
  双 NULL(F-b/LOCAL_NULL) → crawl NEG ∧ truck POS(PASS) → SURROGATE_FAIL;
- treatment 审计(§11,第 0 层核对项):lease source 行为占比≈0.10±0.02
  (admission_execution_counts)、critic source 配额≈0.10、abstain 双零。

输入:eval JSON(p0_evaluator 产物)按 分支×seed×checkpoint 组织的清单文件
+ 冻结 δ JSON + duplicate 分支 eval JSON。输出:裁决 JSON(拒绝覆盖)。
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

T_090_DF2 = 1.8856  # one-sided Student-t, df=2(写死,run card §5)
SOURCE_MASS_TARGET = 0.10
SOURCE_MASS_TOL = 0.02


def _load(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def _paired_effects(manifest: dict, metric: str) -> list[float]:
    """按 seed 配对的 lease−abstain 效应(primary endpoint checkpoint)。"""
    effects = []
    for seed_entry in manifest["seeds"]:
        lease = _load(seed_entry["lease_eval"])["aggregate"][metric]
        abstain = _load(seed_entry["abstain_eval"])["aggregate"][metric]
        effects.append(float(lease) - float(abstain))
    return effects


def _classify_task(task: str, manifest: dict, delta: dict, d_dup: float) -> dict:
    delta_return = float(delta["delta_return"])
    effects = _paired_effects(manifest, "return_mean")
    n = len(effects)
    if n != len(REQUIRED_SEEDS):
        # t₀.₉₀ 阈值按 df=2 写死;seed 数不为 3 时该阈值不适用,防御性拒绝
        # (入口验证已保证此处不可达,双保险)。
        raise ValueError(f"paired effects require exactly {len(REQUIRED_SEEDS)} seeds, got {n}")
    mean_u = statistics.mean(effects)
    sd_u = statistics.stdev(effects) if n > 1 else float("nan")
    half_width = T_090_DF2 * sd_u / math.sqrt(n)
    lcb, ucb = mean_u - half_width, mean_u + half_width

    progress_delta = None
    progress_concordant = None
    if task == "crawl":
        progress_effects = _paired_effects(manifest, "progress_max_dx_mean")
        progress_delta = statistics.mean(progress_effects)
        delta_progress = float(delta["delta_progress_m"])
        # 同向性 = progress 效应与 return 效应同号且 |delta| > δ_progress。
        progress_concordant = (
            abs(progress_delta) > delta_progress
            and math.copysign(1.0, progress_delta) == math.copysign(1.0, mean_u)
        )

    stats = {
        "effects": effects,
        "mean": mean_u,
        "sd": sd_u,
        "lcb": lcb,
        "ucb": ucb,
        "delta_return": delta_return,
        "d_dup": d_dup,
        "progress_delta": progress_delta,
        "progress_concordant": progress_concordant,
    }

    # 判序 1:数值地板前置(五次复核阻塞问题 1 于判序——d_dup≥δ 时任何
    # 后续分类不可信,不允许 POSITIVE/NEGATIVE/NULL 短路)。
    if d_dup >= delta_return:
        return {**stats, "classification": "UNCERTAIN_NUMERIC"}
    # 判序 2:HETEROGENEOUS(存在超阈且反号的 seed 对)。
    big = [u for u in effects if abs(u) > delta_return]
    if any(a * b < 0 for i, a in enumerate(big) for b in big[i + 1:]):
        return {**stats, "classification": "HETEROGENEOUS"}
    # 判序 3:POSITIVE/NEGATIVE(严格不等号;边界归 UNCERTAIN)。
    if lcb > delta_return and (progress_concordant is None or progress_concordant):
        return {**stats, "classification": "POSITIVE"}
    if ucb < -delta_return and (progress_concordant is None or progress_concordant):
        return {**stats, "classification": "NEGATIVE"}
    # 判序 4:NULL(CI 严格含于 (−δ,+δ))。
    if -delta_return < lcb and ucb < delta_return:
        return {**stats, "classification": "NULL"}
    # 判序 5:其余。
    return {**stats, "classification": "UNCERTAIN"}


def _joint(crawl_class: str, truck_class: str) -> dict:
    """§7 六步判序(工程层由 treatment 审计单独给出,此处为统计层组合)。"""
    combo = f"crawl_{crawl_class}__truck_{truck_class}"
    classes = (crawl_class, truck_class)
    if "HETEROGENEOUS" in classes:
        return {"verdict": "F-d", "statement": "不稳定:无可部署统一判据;封存", "combo": combo}
    if any(c.startswith("UNCERTAIN") for c in classes):
        reason = "数值地板" if "UNCERTAIN_NUMERIC" in classes else "CI 过宽/progress 冲突"
        return {"verdict": "F-a", "statement": f"统计不可测(本预算分辨率不足,原因:{reason});判据封存", "combo": combo}
    if classes == ("NULL", "NULL"):
        return {"verdict": "F-b/LOCAL_NULL",
                "statement": "实验可测(measurement 通过),(η=0.1, L=3000) 局部效应≈0;封存,提示长时程累积效应", "combo": combo}
    if classes == ("NEGATIVE", "POSITIVE"):
        return {"verdict": "P0_PASS", "statement": "measurement+concordance 双过 → 进入 P1' 讨论", "combo": combo}
    return {"verdict": "SURROGATE_FAIL",
            "statement": "局部 lease utility 可测,但不是完整训练收益的可靠 surrogate;lease 判据封存,estimand 分离入双通道证据链", "combo": combo}


REQUIRED_SEEDS = (1, 2, 3)          # 冻结:恰好三个 seed,不得重复/缺席
PRIMARY_STEP = 13000                # 冻结:primary endpoint checkpoint
EVAL_PANEL = {                      # 冻结:evaluator 面板(与 p0_evaluator 一致)
    "eval_seeds": [11, 23, 37, 53],
    "ranks": list(range(8)),
    "episode_steps": 1000,
    "episode_count": 32,
}
TASK_ENV = {"crawl": "h1hand-crawl-v0", "truck": "h1hand-truck-v0"}
ARM_ADMISSION_MODE = {"lease": "all", "abstain": "none"}
NOISE_SEED_BASE = 77000  # 与 orchestrator 冻结面板一致
# 冻结 treatment 配置(七次复核:checkpoint 身份不止 admission_mode——bank/
# groups/student_logit/noise seed/warmup/replay 冻结参数任何一项被换都必须拦下)。
TASK_TREATMENT = {
    "crawl": {
        "source_bank": "configs/source_banks/h1hand_loco_wfix_crawl.yaml",
        "mcg_groups": ["legs_torso", "arms", "hands"],
        "admission_student_logit": 16.6823567039,
        # 冻结源集合(八次复核 5:truck 4 源→审计计数必须 5 列;checkpoint 的
        # source_names 含 null option 记录)。
        "source_names": ["stand", "walk", "run", "null"],
        "num_sources": 3,
    },
    "truck": {
        "source_bank": "configs/source_banks/h1hand_hurdle4_wfix_truck.yaml",
        "mcg_groups": ["legs_torso", "arms"],
        "admission_student_logit": 16.4139012941,
        "source_names": ["stand", "walk", "run", "hurdle", "null"],
        "num_sources": 4,
    },
}
# δ 冻结文件的验证参照(与 p0_freeze_delta.py 输出一致)。
DELTA_DEFINITION = "0.5 * cross-seed SD of scratch [eval] return, window steps 10000+15000"
DELTA_WINDOW_STEPS = [10000, 15000]
SOURCE_FREE_DECLARATION = "structural (no bank/option/admission components constructed)"
COMMON_TREATMENT = {
    "mcg_warmup_mode": "admission_bootstrap",
    "mcg_ablation": "bootstrap_only",
    "mcg_warmup_steps": 30000,
    "admission_replay_handoff": "physical_after_authority",
    "admission_replay_recency_half_life": 0.0,
    "admission_replay_uniform_mix": 1.0,
    "admission_replay_priority_alpha": 0.0,
    "run_stop_step": 13000,
}


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _delta_window_mean(path: Path) -> float:
    """Recompute the two-point historical window used by the freeze script."""
    import re

    pattern = re.compile(r"\[eval\] step=(\d+) return=(-?\d+(?:\.\d+)?)")
    by_step: dict[int, list[float]] = {}
    for line in path.read_text(errors="replace").splitlines():
        match = pattern.search(line)
        if match:
            step, value = int(match.group(1)), float(match.group(2))
            if step in DELTA_WINDOW_STEPS:
                by_step.setdefault(step, []).append(value)
    values = []
    for step in DELTA_WINDOW_STEPS:
        if len(by_step.get(step, [])) != 1:
            raise ValueError(
                f"{path}: delta window step {step} must appear exactly once"
            )
        values.append(by_step[step][0])
    return statistics.mean(values)


def _validate_delta(task: str, delta: dict, errors: list[str]) -> None:
    """δ 冻结文件验证(八次复核 5:crawl 的 δ 文件声明属于 truck 也曾通过——
    task/definition/window/输入哈希/有限性全部必须核对)。"""
    if delta.get("task") != task:
        errors.append(f"{task}: delta file declares task={delta.get('task')!r}")
    if delta.get("definition") != DELTA_DEFINITION:
        errors.append(f"{task}: delta definition mismatch (frozen text required)")
    if list(delta.get("window_steps") or []) != DELTA_WINDOW_STEPS:
        errors.append(f"{task}: delta window_steps={delta.get('window_steps')} != {DELTA_WINDOW_STEPS}")
    value = delta.get("delta_return")
    if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        errors.append(f"{task}: delta_return={value!r} is not a finite positive number")
    import re

    hashes = delta.get("input_log_sha256") or {}
    means = delta.get("per_seed_window_means") or {}
    if not isinstance(hashes, dict) or not isinstance(means, dict):
        errors.append(f"{task}: delta hashes and per-seed means must be JSON objects")
        hashes, means = {}, {}
    if len(hashes) != 3 or set(hashes) != set(means):
        errors.append(
            f"{task}: input hashes and per-seed means must cover the same 3 logs"
        )
    recomputed_means: list[float] = []
    for raw_path, expected_hash in hashes.items():
        if not isinstance(expected_hash, str) or re.fullmatch(r"[0-9a-f]{64}", expected_hash) is None:
            errors.append(f"{task}: invalid sha256 syntax for delta source {raw_path}")
            continue
        path = Path(raw_path)
        if not path.is_file():
            errors.append(f"{task}: delta source log does not exist: {path}")
            continue
        if _sha256(path) != expected_hash:
            errors.append(f"{task}: delta source log sha mismatch: {path}")
            continue
        recorded_mean = means.get(raw_path)
        if not isinstance(recorded_mean, (int, float)) or not math.isfinite(recorded_mean):
            errors.append(f"{task}: invalid per-seed window mean for {path}")
            continue
        try:
            actual_mean = _delta_window_mean(path)
        except (OSError, ValueError) as exc:
            errors.append(f"{task}: cannot recompute delta source {path} ({exc})")
            continue
        if not math.isclose(float(recorded_mean), actual_mean, rel_tol=0.0, abs_tol=1e-9):
            errors.append(
                f"{task}: recorded window mean {recorded_mean} != recomputed {actual_mean} for {path}"
            )
        recomputed_means.append(actual_mean)
    if len(recomputed_means) == 3:
        actual_sd = statistics.stdev(recomputed_means)
        recorded_sd = delta.get("cross_seed_sd")
        if not isinstance(recorded_sd, (int, float)) or not math.isclose(
            float(recorded_sd), actual_sd, rel_tol=0.0, abs_tol=1e-9
        ):
            errors.append(f"{task}: cross_seed_sd does not match recomputed historical SD")
        if isinstance(value, (int, float)) and math.isfinite(value) and not math.isclose(
            float(value), 0.5 * actual_sd, rel_tol=0.0, abs_tol=1e-9
        ):
            errors.append(f"{task}: delta_return does not equal 0.5 * recomputed SD")
    if task == "crawl":
        progress = delta.get("delta_progress_m")
        if not isinstance(progress, (int, float)) or abs(float(progress) - 0.5) > 1e-12:
            errors.append(f"crawl: delta_progress_m={progress!r} != frozen 0.5")


def _validate_manifest(task: str, manifest: dict, errors: list[str]) -> None:
    """入口结构验证(六次复核阻塞 1):任何缺失/伪造路径都必须在此被拦下,
    绝不允许静默跳过后仍产生方向性裁决。"""
    seeds_entries = manifest.get("seeds") or []
    if not isinstance(seeds_entries, list) or any(
        not isinstance(entry, dict) for entry in seeds_entries
    ):
        errors.append(f"{task}: manifest.seeds must be a list of objects")
        return
    seeds = [entry.get("seed") for entry in seeds_entries]
    if sorted(seeds) != list(REQUIRED_SEEDS):
        errors.append(f"{task}: seeds must be exactly {REQUIRED_SEEDS}, got {seeds}")
    for entry in seeds_entries:
        for key in ("lease_eval", "abstain_eval", "lease_checkpoint", "abstain_checkpoint"):
            if key not in entry:
                errors.append(f"{task} seed {entry.get('seed')}: missing required field '{key}'")
            elif not Path(entry[key]).exists():
                errors.append(f"{task} seed {entry.get('seed')}: {key} path does not exist")
    duplicate = manifest.get("duplicate") or {}
    if not isinstance(duplicate, dict):
        errors.append(f"{task}: manifest.duplicate must be an object")
        return
    # 八次复核 5:duplicate 的 checkpoint 与 execution record 均为必填——
    # 缺失即无法验证身份与独立性,fail-closed。
    for key in ("eval_a", "eval_b", "checkpoint_a", "checkpoint_b",
                "execution_record_a", "execution_record_b"):
        if key not in duplicate:
            errors.append(f"{task}: duplicate.{key} is missing (mandatory)")
        elif not Path(duplicate[key]).exists():
            errors.append(f"{task}: duplicate.{key} path does not exist")
    if duplicate.get("eval_a") and duplicate.get("eval_a") == duplicate.get("eval_b"):
        errors.append(f"{task}: duplicate eval_a and eval_b point to the same artifact")


def _resolve_evidence_path(raw_path: str) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else REPO_ROOT / path


def _validate_execution_record(task: str, letter: str, record_path: str,
                               checkpoint_path: str, errors: list[str]) -> dict | None:
    """Validate that duplicate evidence is bound to an actual successful process."""
    label = f"{task} duplicate {letter}"
    try:
        record = _load(record_path)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"{label}: execution record unreadable ({exc})")
        return None
    if not isinstance(record, dict):
        errors.append(f"{label}: execution record must be a JSON object")
        return None
    if record.get("schema_version") != 1:
        errors.append(f"{label}: execution record schema_version must be 1")
    if record.get("task") != task or record.get("arm") != "abstain" or record.get("seed") != 1:
        errors.append(f"{label}: execution record task/arm/seed identity mismatch")
    if record.get("exit_code") != 0:
        errors.append(f"{label}: execution record exit_code is not zero")
    if not isinstance(record.get("pid"), int) or record["pid"] <= 0:
        errors.append(f"{label}: execution record pid is invalid")
    cli = record.get("cli")
    if not isinstance(cli, list) or not cli:
        errors.append(f"{label}: execution record CLI is missing")
    else:
        cli_json = json.dumps(cli, ensure_ascii=False, separators=(",", ":"))
        expected_cli_sha = __import__("hashlib").sha256(cli_json.encode()).hexdigest()
        if record.get("cli_sha256") != expected_cli_sha:
            errors.append(f"{label}: execution record CLI hash mismatch")
    if not record.get("start_utc") or not record.get("end_utc"):
        errors.append(f"{label}: execution record timestamps are missing")
    log_ref = record.get("log")
    if not isinstance(log_ref, str) or not log_ref:
        errors.append(f"{label}: execution record log path is missing")
    else:
        log_path = _resolve_evidence_path(log_ref)
        if not log_path.is_file():
            errors.append(f"{label}: execution log does not exist: {log_path}")
        elif record.get("log_sha256") != _sha256(log_path):
            errors.append(f"{label}: execution log sha mismatch")
    artifacts = record.get("artifact_sha256") or {}
    if not isinstance(artifacts, dict):
        errors.append(f"{label}: execution artifact hashes must be a JSON object")
        artifacts = {}
    checkpoint = Path(checkpoint_path)
    matches = [value for key, value in artifacts.items() if Path(key).name == checkpoint.name]
    if len(matches) != 1 or matches[0] != _sha256(checkpoint):
        errors.append(f"{label}: primary checkpoint is not bound to execution artifacts")
    git_head = record.get("git_head")
    import re

    if not isinstance(git_head, str) or re.fullmatch(r"[0-9a-f]{40}", git_head) is None:
        errors.append(f"{label}: execution record git_head is invalid")
    return record


def _validate_eval(task: str, seed: int, arm: str, eval_path: str,
                   checkpoint_path: str | None, errors: list[str]) -> dict | None:
    """eval JSON 身份验证(七次复核收紧):面板冻结值、episode 数、任务、
    deterministic/source_free/identity_checked 协议标志、episodes 明细的
    真实 reset seed 集合、checkpoint 对应(SHA256 交叉验证、step=13000)。"""
    label = f"{task} seed {seed} {arm}"
    try:
        data = _load(eval_path)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"{label}: eval JSON unreadable ({exc})")
        return None
    if not isinstance(data, dict):
        errors.append(f"{label}: eval root must be a JSON object")
        return None
    protocol = data.get("protocol") or {}
    if not isinstance(protocol, dict):
        errors.append(f"{label}: eval protocol must be a JSON object")
        protocol = {}
    for key, expected in (
        ("eval_seeds", EVAL_PANEL["eval_seeds"]),
        ("ranks", EVAL_PANEL["ranks"]),
        ("episode_steps", EVAL_PANEL["episode_steps"]),
    ):
        if protocol.get(key) != expected:
            errors.append(f"{label}: protocol.{key}={protocol.get(key)} != frozen {expected}")
    # 协议标志(八次复核收紧:source_free 用精确白名单——truthiness 判断会
    # 把字符串 "false" 当通过)。
    if protocol.get("deterministic") is not True:
        errors.append(f"{label}: protocol.deterministic must be exactly true")
    source_free = protocol.get("source_free")
    if source_free != SOURCE_FREE_DECLARATION:
        errors.append(
            f"{label}: protocol.source_free={source_free!r} is not the frozen "
            "structural declaration"
        )
    aggregate = data.get("aggregate") or {}
    if not isinstance(aggregate, dict):
        errors.append(f"{label}: eval aggregate must be a JSON object")
        aggregate = {}
    if aggregate.get("episode_count") != EVAL_PANEL["episode_count"]:
        errors.append(f"{label}: episode_count={aggregate.get('episode_count')} != 32")
    # episodes 明细的真实面板(八次复核:只比集合会放过 32 行复制成 64 行)——
    # 必须恰好 32 行且每个冻结 reset seed 恰好出现一次。
    from collections import Counter

    episodes = data.get("episodes") or []
    if not isinstance(episodes, list):
        errors.append(f"{label}: episodes must be a JSON list")
        episodes = []
    if any(not isinstance(episode, dict) for episode in episodes):
        errors.append(f"{label}: every episode row must be a JSON object")
    expected_reset_seeds = {
        eval_seed * 1000 + rank
        for eval_seed in EVAL_PANEL["eval_seeds"]
        for rank in EVAL_PANEL["ranks"]
    }
    seed_counts = Counter(e.get("seed") for e in episodes if isinstance(e, dict))
    if len(episodes) != EVAL_PANEL["episode_count"]:
        errors.append(f"{label}: episodes has {len(episodes)} rows, expected exactly 32")
    elif set(seed_counts) != expected_reset_seeds or any(v != 1 for v in seed_counts.values()):
        errors.append(f"{label}: episode reset-seed panel mismatch (each frozen seed must appear exactly once)")
    if data.get("env_name") != TASK_ENV[task]:
        errors.append(f"{label}: env_name={data.get('env_name')} != {TASK_ENV[task]}")
    value = aggregate.get("return_mean")
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        errors.append(f"{label}: return_mean is missing or non-finite")
    # Aggregate 必须能由 episode 明细重算。只验证 seed 面板、却信任一个可单独
    # 修改的 aggregate，会允许 episode 全为 0 而方向性结论仍保持不变。
    episode_returns = [e.get("return") for e in episodes if isinstance(e, dict)]
    if len(episode_returns) != EVAL_PANEL["episode_count"] or any(
        not isinstance(v, (int, float)) or not math.isfinite(v) for v in episode_returns
    ):
        errors.append(f"{label}: every episode must contain a finite return")
    elif isinstance(value, (int, float)) and math.isfinite(value):
        recomputed_return = statistics.mean(float(v) for v in episode_returns)
        if not math.isclose(float(value), recomputed_return, rel_tol=0.0, abs_tol=1e-9):
            errors.append(
                f"{label}: aggregate return_mean={value} != episode mean {recomputed_return}"
            )
    progress_value = aggregate.get("progress_max_dx_mean")
    episode_progress = [e.get("progress_max_dx") for e in episodes if isinstance(e, dict)]
    if len(episode_progress) != EVAL_PANEL["episode_count"] or any(
        not isinstance(v, (int, float)) or not math.isfinite(v) for v in episode_progress
    ):
        errors.append(f"{label}: every episode must contain finite progress_max_dx")
    elif not isinstance(progress_value, (int, float)) or not math.isfinite(progress_value):
        errors.append(f"{label}: progress_max_dx_mean is missing or non-finite")
    else:
        recomputed_progress = statistics.mean(float(v) for v in episode_progress)
        if not math.isclose(
            float(progress_value), recomputed_progress, rel_tol=0.0, abs_tol=1e-9
        ):
            errors.append(
                f"{label}: aggregate progress={progress_value} != episode mean {recomputed_progress}"
            )
    ckpt_meta = data.get("checkpoint") or {}
    if not isinstance(ckpt_meta, dict):
        errors.append(f"{label}: eval checkpoint metadata must be a JSON object")
        ckpt_meta = {}
    if ckpt_meta.get("identity_checked") is not True:
        errors.append(f"{label}: eval was produced without --expect-* identity checks")
    if ckpt_meta.get("global_step") != PRIMARY_STEP:
        errors.append(
            f"{label}: eval checkpoint global_step={ckpt_meta.get('global_step')} != {PRIMARY_STEP}"
        )
    if checkpoint_path is not None and Path(checkpoint_path).exists():
        actual_sha = _sha256(Path(checkpoint_path))
        if ckpt_meta.get("sha256") != actual_sha:
            errors.append(f"{label}: eval checkpoint sha256 does not match manifest checkpoint file")
    return data


def _validate_checkpoint(task: str, seed: int, arm: str, ckpt_path: str,
                         errors: list[str]) -> dict | None:
    """checkpoint 身份验证:任务/seed/arm(admission_mode)/global_step。"""
    import torch

    label = f"{task} seed {seed} {arm}"
    try:
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"{label}: checkpoint unreadable ({exc})")
        return None
    if not isinstance(state, dict):
        errors.append(f"{label}: checkpoint root must be a mapping")
        return None
    ckpt_args = state.get("args") or {}
    ptf_cfg = state.get("ptf_cfg") or {}
    if not isinstance(ckpt_args, dict) or not isinstance(ptf_cfg, dict):
        errors.append(f"{label}: checkpoint args/ptf_cfg must be mappings")
        return state
    if ckpt_args.get("env_name") != TASK_ENV[task]:
        errors.append(f"{label}: checkpoint env_name={ckpt_args.get('env_name')} != {TASK_ENV[task]}")
    if ckpt_args.get("seed") != seed:
        errors.append(f"{label}: checkpoint seed={ckpt_args.get('seed')} != manifest seed {seed}")
    if state.get("global_step") != PRIMARY_STEP:
        errors.append(f"{label}: checkpoint global_step={state.get('global_step')} != {PRIMARY_STEP}")
    admission_mode = ptf_cfg.get("admission_mode")
    if admission_mode != ARM_ADMISSION_MODE[arm]:
        errors.append(
            f"{label}: checkpoint admission_mode={admission_mode} != {ARM_ADMISSION_MODE[arm]} (arm identity)"
        )
    # 完整 treatment 配置校验(七次复核阻塞 2):任务级冻结项。
    task_treatment = TASK_TREATMENT[task]
    if ptf_cfg.get("source_bank") != task_treatment["source_bank"]:
        errors.append(f"{label}: source_bank={ptf_cfg.get('source_bank')} != frozen")
    if list(ptf_cfg.get("mcg_groups") or []) != task_treatment["mcg_groups"]:
        errors.append(f"{label}: mcg_groups={ptf_cfg.get('mcg_groups')} != frozen")
    logit = ptf_cfg.get("admission_student_logit")
    if not isinstance(logit, (int, float)) or abs(float(logit) - task_treatment["admission_student_logit"]) > 1e-9:
        errors.append(f"{label}: admission_student_logit={logit} != frozen")
    # 通用冻结项。
    for key, expected in COMMON_TREATMENT.items():
        actual = ptf_cfg.get(key)
        if isinstance(expected, float):
            matches = isinstance(actual, (int, float)) and abs(float(actual) - expected) < 1e-12
        else:
            matches = actual == expected
        if not matches:
            errors.append(f"{label}: ptf_cfg.{key}={actual} != frozen {expected}")
    # noise seed 配对面板。
    expected_noise_seed = NOISE_SEED_BASE + seed
    if ptf_cfg.get("resume_noise_seed") != expected_noise_seed:
        errors.append(
            f"{label}: resume_noise_seed={ptf_cfg.get('resume_noise_seed')} != {expected_noise_seed}"
        )
    # anchor 溯源:必须存在 anchor_resume_manifest 且 bundle 指向该任务/seed。
    anchor_manifest = ptf_cfg.get("anchor_resume_manifest") or {}
    if not isinstance(anchor_manifest, dict):
        errors.append(f"{label}: anchor_resume_manifest must be a mapping")
        anchor_manifest = {}
    bundle = str(anchor_manifest.get("bundle") or "")
    if f"{task}_s{seed}" not in bundle:
        errors.append(f"{label}: anchor_resume_manifest.bundle={bundle!r} does not match {task}_s{seed}")
    # source schema(八次复核 5):source_names 必须逐位等于冻结列表(truck
    # 4 源+null=5 项);审计计数列数在 _treatment_audit 按 num_sources+1 验证。
    expected_names = TASK_TREATMENT[task]["source_names"]
    if list(state.get("source_names") or []) != expected_names:
        errors.append(
            f"{label}: source_names={state.get('source_names')} != frozen {expected_names}"
        )
    return state


def _frac(numerator, denominator) -> float | None:
    """占比;分母缺失/为零 → None(调用方判 INVALID,绝不当作通过)。"""
    if numerator is None or not denominator:
        return None
    value = numerator / denominator
    return value if math.isfinite(value) else None


def _treatment_audit(task: str, manifest: dict, checkpoints: dict, errors: list[str]) -> dict:
    """§11 treatment 保真(六次复核收紧):
    - lease:行为 source 占比与 critic source 占比 **都**必须在
      [target−tol, target+tol]=[0.08, 0.12];
    - abstain:两种 source 计数都必须**存在**且严格为零(缺失≠通过);
    - 任何字段缺失/分母为零/非有限 → 记入 errors → ENGINEERING_INVALID。"""
    rows = []
    for entry in manifest.get("seeds") or []:
        seed = entry.get("seed")
        for arm in ("lease", "abstain"):
            label = f"{task} seed {seed} {arm}"
            state = checkpoints.get((seed, arm))
            if state is None:
                errors.append(f"{label}: checkpoint unavailable for treatment audit")
                continue
            audit = state.get("admission_audit") or {}
            if not isinstance(audit, dict):
                errors.append(f"{label}: admission_audit must be a mapping")
                continue
            exec_counts = audit.get("execution_counts")
            critic_counts = audit.get("critic_sample_counts")
            if not exec_counts:
                errors.append(f"{label}: execution_counts missing from admission_audit")
                continue
            if not critic_counts:
                errors.append(f"{label}: critic_sample_counts missing from admission_audit")
                continue
            # 计数 schema(八次复核 5):列数必须= num_sources+1(truck=5),
            # 且全部为非负整数——列数错即源集合与冻结协议不符。
            expected_columns = TASK_TREATMENT[task]["num_sources"] + 1
            schema_bad = False
            for name, counts in (("execution_counts", exec_counts),
                                 ("critic_sample_counts", critic_counts)):
                if len(counts) != expected_columns:
                    errors.append(
                        f"{label}: {name} has {len(counts)} columns, expected {expected_columns}"
                    )
                    schema_bad = True
                elif any((not isinstance(x, int)) or x < 0 for x in counts):
                    errors.append(f"{label}: {name} contains non-integer or negative entries")
                    schema_bad = True
            if schema_bad:
                continue
            source_exec = sum(int(x) for x in exec_counts[:-1])
            total_exec = sum(int(x) for x in exec_counts)
            source_critic = sum(int(x) for x in critic_counts[:-1])
            total_critic = sum(int(x) for x in critic_counts)
            exec_frac = _frac(source_exec, total_exec)
            critic_frac = _frac(source_critic, total_critic)
            if arm == "lease":
                if exec_frac is None or critic_frac is None:
                    errors.append(f"{label}: lease source fractions undefined (zero denominator)")
                    row_ok = False
                else:
                    row_ok = (
                        abs(exec_frac - SOURCE_MASS_TARGET) <= SOURCE_MASS_TOL
                        and abs(critic_frac - SOURCE_MASS_TARGET) <= SOURCE_MASS_TOL
                    )
                    if not row_ok:
                        errors.append(
                            f"{label}: source fractions exec={exec_frac:.4f} critic={critic_frac:.4f} "
                            f"outside [{SOURCE_MASS_TARGET - SOURCE_MASS_TOL}, {SOURCE_MASS_TARGET + SOURCE_MASS_TOL}]"
                        )
            else:
                row_ok = source_exec == 0 and source_critic == 0
                if not row_ok:
                    errors.append(
                        f"{label}: abstain must have strictly zero source counts "
                        f"(exec={source_exec}, critic={source_critic})"
                    )
            rows.append({
                "seed": seed, "arm": arm,
                "source_exec_frac": exec_frac, "source_critic_frac": critic_frac,
                "ok": row_ok,
            })
    expected_rows = len(REQUIRED_SEEDS) * 2
    if len(rows) != expected_rows:
        errors.append(f"{task}: treatment audit covered {len(rows)}/{expected_rows} arms")
    all_ok = len(rows) == expected_rows and all(row["ok"] for row in rows)
    return {"rows": rows, "all_ok": all_ok}


def _validate_duplicate_abstain_audit(task: str, state: dict | None, label: str,
                                      errors: list[str]) -> None:
    """Both duplicate arms must actually realize exact abstention, not just request it."""
    if state is None:
        return
    audit = state.get("admission_audit") or {}
    if not isinstance(audit, dict):
        errors.append(f"{task} duplicate {label}: admission_audit must be a mapping")
        return
    expected_columns = TASK_TREATMENT[task]["num_sources"] + 1
    for name in ("execution_counts", "critic_sample_counts"):
        counts = audit.get(name)
        if not isinstance(counts, list) or len(counts) != expected_columns:
            errors.append(f"{task} duplicate {label}: {name} schema mismatch")
            continue
        if any(type(x) is not int or x < 0 for x in counts):
            errors.append(f"{task} duplicate {label}: {name} has invalid entries")
            continue
        if sum(counts[:-1]) != 0 or sum(counts) <= 0:
            errors.append(
                f"{task} duplicate {label}: {name} does not prove exact abstention"
            )


def adjudicate(crawl_manifest: str | Path, truck_manifest: str | Path,
               crawl_delta: str | Path, truck_delta: str | Path) -> dict:
    """完整裁决(入口验证→身份验证→treatment 审计→统计→joint)。
    纯函数(不落盘),供 main 与入口级反例测试共用。"""
    result: dict = {"tasks": {}}
    validation_errors: list[str] = []
    for task, manifest_path, delta_path in (
        ("crawl", crawl_manifest, crawl_delta),
        ("truck", truck_manifest, truck_delta),
    ):
        task_errors: list[str] = []
        try:
            manifest = _load(manifest_path)
            delta = _load(delta_path)
        except Exception as exc:  # noqa: BLE001
            task_errors.append(f"{task}: manifest/delta JSON unreadable ({exc})")
            validation_errors.extend(task_errors)
            result["tasks"][task] = {
                "classification": "ENGINEERING_INVALID", "errors": task_errors,
            }
            continue
        if not isinstance(manifest, dict) or not isinstance(delta, dict):
            if not isinstance(manifest, dict):
                task_errors.append(f"{task}: manifest root must be a JSON object")
            if not isinstance(delta, dict):
                task_errors.append(f"{task}: delta root must be a JSON object")
            validation_errors.extend(task_errors)
            result["tasks"][task] = {
                "classification": "ENGINEERING_INVALID", "errors": task_errors,
            }
            continue
        _validate_delta(task, delta, task_errors)
        _validate_manifest(task, manifest, task_errors)
        # 结构验证失败即止:字段缺失时后续身份验证无从谈起。
        if task_errors:
            validation_errors.extend(task_errors)
            result["tasks"][task] = {"classification": "ENGINEERING_INVALID",
                                     "errors": task_errors}
            continue
        checkpoints: dict = {}
        for entry in manifest["seeds"]:
            seed = entry["seed"]
            for arm in ("lease", "abstain"):
                checkpoints[(seed, arm)] = _validate_checkpoint(
                    task, seed, arm, entry[f"{arm}_checkpoint"], task_errors
                )
                _validate_eval(
                    task, seed, arm, entry[f"{arm}_eval"],
                    entry[f"{arm}_checkpoint"], task_errors,
                )
        # duplicate 走与正式臂完全相同的 eval+checkpoint 身份验证(八次复核:
        # 错误 seed/arm/任务/step/面板的 duplicate checkpoint 必须被拦下)。
        # duplicate=abstain s1 的进程重启,身份=(task, seed=1, arm=abstain)。
        duplicate = manifest["duplicate"]
        dup_state_a = _validate_checkpoint(
            task, 1, "abstain", duplicate["checkpoint_a"], task_errors
        )
        dup_state_b = _validate_checkpoint(
            task, 1, "abstain", duplicate["checkpoint_b"], task_errors
        )
        _validate_duplicate_abstain_audit(task, dup_state_a, "A", task_errors)
        _validate_duplicate_abstain_audit(task, dup_state_b, "B", task_errors)
        dup_a = _validate_eval(task, 1, "abstain(dupA)", duplicate["eval_a"],
                               duplicate.get("checkpoint_a"), task_errors)
        dup_b = _validate_eval(task, 1, "abstain(dupB)", duplicate["eval_b"],
                               duplicate.get("checkpoint_b"), task_errors)
        # 两次独立重启允许产出逐位相同 checkpoint(d_dup=0)。独立性由进程
        # 启动时生成、并绑定 CLI/log/artifact 的 execution record 证明。
        record_a = _validate_execution_record(
            task, "A", duplicate["execution_record_a"],
            duplicate["checkpoint_a"], task_errors,
        ) or {}
        record_b = _validate_execution_record(
            task, "B", duplicate["execution_record_b"],
            duplicate["checkpoint_b"], task_errors,
        ) or {}
        if record_a and record_b:
            if not record_a.get("execution_id") or record_a.get("execution_id") == record_b.get("execution_id"):
                task_errors.append(f"{task}: duplicate execution_ids missing or identical")
            if record_a.get("log") == record_b.get("log"):
                task_errors.append(f"{task}: duplicate run logs missing or identical")
            if record_a.get("cli") != record_b.get("cli"):
                task_errors.append(f"{task}: duplicate CLI differs between A and B")
            if record_a.get("git_head") != record_b.get("git_head"):
                task_errors.append(f"{task}: duplicate git_head differs between A and B")
        dup_values = (
            ((dup_a or {}).get("aggregate") or {}).get("return_mean"),
            ((dup_b or {}).get("aggregate") or {}).get("return_mean"),
        )
        if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in dup_values):
            task_errors.append(f"{task}: duplicate return_mean missing or non-finite")
            d_dup = float("nan")
        else:
            d_dup = abs(dup_values[0] - dup_values[1])
        audit = _treatment_audit(task, manifest, checkpoints, task_errors)
        if task_errors:
            validation_errors.extend(task_errors)
            result["tasks"][task] = {"classification": "ENGINEERING_INVALID",
                                     "errors": task_errors,
                                     "treatment_audit": audit}
            continue
        result["tasks"][task] = _classify_task(task, manifest, delta, d_dup)
        result["tasks"][task]["treatment_audit"] = audit

    audits_ok = not validation_errors and all(
        result["tasks"][t].get("treatment_audit", {}).get("all_ok") for t in result["tasks"]
    )
    if not audits_ok:
        result["joint"] = {
            "verdict": "ENGINEERING_INVALID",
            "statement": "入口验证/treatment 审计未过:修复后重跑同配置(不算改参)",
            "validation_errors": validation_errors,
        }
    else:
        result["joint"] = _joint(
            result["tasks"]["crawl"]["classification"],
            result["tasks"]["truck"]["classification"],
        )
    result["frozen_constants"] = {
        "t_090_df2": T_090_DF2,
        "source_mass_target": SOURCE_MASS_TARGET,
        "source_mass_tol": SOURCE_MASS_TOL,
        "required_seeds": list(REQUIRED_SEEDS),
        "primary_step": PRIMARY_STEP,
    }
    result["git_head"] = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False,
    ).stdout.strip()
    result["utc"] = datetime.now(timezone.utc).isoformat()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crawl-manifest", required=True,
                        help="JSON:{seeds:[{seed,lease_eval,abstain_eval,lease_checkpoint,abstain_checkpoint}],duplicate:{eval_a,eval_b}}")
    parser.add_argument("--truck-manifest", required=True)
    parser.add_argument("--crawl-delta", required=True)
    parser.add_argument("--truck-delta", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    result = adjudicate(args.crawl_manifest, args.truck_manifest,
                        args.crawl_delta, args.truck_delta)
    out_path = Path(args.out)
    if out_path.exists():
        raise FileExistsError(f"adjudication already exists at {out_path}; refusing to overwrite")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"[p0_adjudicate] joint={result['joint']['verdict']} -> {out_path}")


if __name__ == "__main__":
    sys.exit(main())
