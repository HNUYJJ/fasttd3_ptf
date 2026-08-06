"""P0 裁决器测试:判序角例 + 入口级反例(六次复核阻塞 1——每个已演示的
绕过路径都必须有反例测试,伪造/缺失输入绝不允许到达方向性裁决)。"""
from __future__ import annotations

import hashlib
import json
import statistics
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from p0_adjudicate import (  # noqa: E402
    EVAL_PANEL,
    PRIMARY_STEP,
    SOURCE_FREE_DECLARATION,
    TASK_ENV,
    _classify_task,
    _joint,
    adjudicate,
)


def _manifest(tmp_path: Path, task: str, lease_returns, abstain_returns,
              lease_progress=None, abstain_progress=None) -> dict:
    seeds = []
    for idx, (lease, abstain) in enumerate(zip(lease_returns, abstain_returns), start=1):
        entries = {}
        for arm, value in (("lease", lease), ("abstain", abstain)):
            aggregate = {"return_mean": value, "progress_max_dx_mean": 0.0}
            if arm == "lease" and lease_progress is not None:
                aggregate["progress_max_dx_mean"] = lease_progress[idx - 1]
            if arm == "abstain" and abstain_progress is not None:
                aggregate["progress_max_dx_mean"] = abstain_progress[idx - 1]
            path = tmp_path / f"{task}_{arm}_s{idx}.json"
            path.write_text(json.dumps({"aggregate": aggregate}))
            entries[f"{arm}_eval"] = str(path)
        seeds.append({"seed": idx, **entries})
    return {"seeds": seeds}


DELTA = {"delta_return": 10.0, "delta_progress_m": 0.5}


def test_dup_floor_preempts_positive(tmp_path):
    """d_dup ≥ δ 必须前置短路——即使效应本可判 POSITIVE(五次复核阻塞 1)。"""
    manifest = _manifest(tmp_path, "truck", [200.0, 201.0, 202.0], [100.0, 100.0, 100.0])
    result = _classify_task("truck", manifest, DELTA, d_dup=10.0)
    assert result["classification"] == "UNCERTAIN_NUMERIC"
    # 同一数据在 d_dup 低于 δ 时判 POSITIVE。
    result2 = _classify_task("truck", manifest, DELTA, d_dup=1.0)
    assert result2["classification"] == "POSITIVE"


def test_heterogeneous_requires_opposite_signs_beyond_delta(tmp_path):
    manifest = _manifest(tmp_path, "truck", [150.0, 60.0, 101.0], [100.0, 100.0, 100.0])
    result = _classify_task("truck", manifest, DELTA, d_dup=1.0)
    assert result["classification"] == "HETEROGENEOUS"


def test_null_requires_ci_strictly_inside(tmp_path):
    manifest = _manifest(tmp_path, "truck", [101.0, 100.0, 99.0], [100.0, 100.0, 100.0])
    result = _classify_task("truck", manifest, DELTA, d_dup=1.0)
    assert result["classification"] == "NULL"


def test_boundary_falls_to_uncertain(tmp_path):
    """CI 跨阈值(不满足严格不等号)归 UNCERTAIN,不判方向。
    效应 [15, 8, -5] 同号超阈对不存在(不 HET),CI 跨 +δ → UNCERTAIN。"""
    manifest = _manifest(tmp_path, "truck", [115.0, 108.0, 95.0], [100.0, 100.0, 100.0])
    result = _classify_task("truck", manifest, DELTA, d_dup=1.0)
    assert result["classification"] == "UNCERTAIN"


def test_crawl_progress_discordance_blocks_direction(tmp_path):
    """crawl 的方向判定需要 progress 同向:return 显著负但 progress 正向
    (超过 0.5 m)时不判 NEGATIVE。"""
    manifest = _manifest(
        tmp_path, "crawl", [50.0, 49.0, 51.0], [100.0, 100.0, 100.0],
        lease_progress=[2.0, 2.0, 2.0], abstain_progress=[0.0, 0.0, 0.0],
    )
    result = _classify_task("crawl", manifest, DELTA, d_dup=1.0)
    assert result["classification"] == "UNCERTAIN"
    # progress 同向(lease 更小,负向一致)→ NEGATIVE。
    sub_dir = tmp_path / "sub"
    sub_dir.mkdir(exist_ok=True)
    manifest2 = _manifest(
        sub_dir, "crawl", [50.0, 49.0, 51.0], [100.0, 100.0, 100.0],
        lease_progress=[0.0, 0.0, 0.0], abstain_progress=[2.0, 2.0, 2.0],
    )
    result2 = _classify_task("crawl", manifest2, DELTA, d_dup=1.0)
    assert result2["classification"] == "NEGATIVE"


@pytest.mark.parametrize(
    "crawl_class, truck_class, verdict",
    [
        ("NEGATIVE", "POSITIVE", "P0_PASS"),
        ("NULL", "NULL", "F-b/LOCAL_NULL"),
        ("HETEROGENEOUS", "POSITIVE", "F-d"),
        ("UNCERTAIN_NUMERIC", "POSITIVE", "F-a"),
        ("UNCERTAIN", "NULL", "F-a"),
        ("NEGATIVE", "NULL", "SURROGATE_FAIL"),
        ("POSITIVE", "NEGATIVE", "SURROGATE_FAIL"),
        ("NULL", "POSITIVE", "SURROGATE_FAIL"),
    ],
)
def test_joint_six_step_order(crawl_class, truck_class, verdict):
    assert _joint(crawl_class, truck_class)["verdict"] == verdict


def test_joint_priority_het_over_uncertain():
    """判序 2(HET) 先于判序 3(UNCERTAIN):同时出现时判 F-d。"""
    assert _joint("HETEROGENEOUS", "UNCERTAIN")["verdict"] == "F-d"


# ---------- 入口级反例测试(每个已演示的绕过路径一个反例) ----------

def _task_counts(task: str, arm: str, *, source_total: int = 100, student_total: int = 900):
    """按任务冻结源数生成合法计数列(crawl=3 源→4 列,truck=4 源→5 列)。"""
    from p0_adjudicate import TASK_TREATMENT

    num_sources = TASK_TREATMENT[task]["num_sources"]
    if arm == "lease":
        base = source_total // num_sources
        sources = [base] * num_sources
        sources[0] += source_total - base * num_sources
        return [*sources, student_total]
    return [0] * num_sources + [student_total + source_total]


def _fake_checkpoint(path: Path, task: str, seed: int, arm: str, *,
                     step: int = PRIMARY_STEP,
                     exec_counts=None, critic_counts="auto",
                     admission_mode: str | None = "auto",
                     ptf_overrides: dict | None = None,
                     source_names=None) -> Path:
    from p0_adjudicate import COMMON_TREATMENT, NOISE_SEED_BASE, TASK_TREATMENT

    if admission_mode == "auto":
        admission_mode = "all" if arm == "lease" else "none"
    if exec_counts is None:
        exec_counts = _task_counts(task, arm)
    audit = {"execution_counts": exec_counts}
    if critic_counts == "auto":
        audit["critic_sample_counts"] = _task_counts(task, arm)
    elif critic_counts is not None:
        audit["critic_sample_counts"] = critic_counts
    ptf_cfg = {
        "admission_mode": admission_mode,
        "source_bank": TASK_TREATMENT[task]["source_bank"],
        "mcg_groups": list(TASK_TREATMENT[task]["mcg_groups"]),
        "admission_student_logit": TASK_TREATMENT[task]["admission_student_logit"],
        **COMMON_TREATMENT,
        "resume_noise_seed": NOISE_SEED_BASE + seed,
        "anchor_resume_manifest": {"bundle": f"checkpoints/p0_anchors/{task}_s{seed}"},
    }
    ptf_cfg.update(ptf_overrides or {})
    torch.save(
        {
            "args": {"env_name": TASK_ENV[task], "seed": seed},
            "global_step": step,
            "ptf_cfg": ptf_cfg,
            "source_names": (
                list(TASK_TREATMENT[task]["source_names"]) if source_names is None else source_names
            ),
            "admission_audit": audit,
        },
        path,
    )
    return path


def _delta_payload(task: str, source_dir: Path) -> dict:
    from p0_adjudicate import DELTA_DEFINITION, DELTA_WINDOW_STEPS

    source_dir.mkdir(parents=True, exist_ok=True)
    means = [80.0, 100.0, 120.0]
    hashes = {}
    recorded_means = {}
    for seed, mean in enumerate(means, start=1):
        path = (source_dir / f"{task}_scratch_s{seed}.log").resolve()
        path.write_text(
            f"[eval] step=10000 return={mean - 1.0}\n"
            f"[eval] step=15000 return={mean + 1.0}\n"
        )
        hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        recorded_means[str(path)] = mean
    sd = statistics.stdev(means)
    return {
        "task": task,
        "delta_return": 0.5 * sd,
        "delta_progress_m": 0.5 if task == "crawl" else None,
        "definition": DELTA_DEFINITION,
        "window_steps": list(DELTA_WINDOW_STEPS),
        "input_log_sha256": hashes,
        "per_seed_window_means": recorded_means,
        "cross_seed_sd": sd,
    }


def _panel_episodes(return_mean: float = 0.0, progress: float = 0.0) -> list[dict]:
    return [
        {
            "seed": eval_seed * 1000 + rank,
            "return": return_mean,
            "progress_max_dx": progress,
        }
        for eval_seed in EVAL_PANEL["eval_seeds"]
        for rank in EVAL_PANEL["ranks"]
    ]


def _fake_eval(path: Path, task: str, ckpt_path: Path, return_mean: float,
               *, progress: float = 0.0, episode_count: int = 32,
               ckpt_step: int = PRIMARY_STEP, sha_override: str | None = None,
               deterministic: bool = True, source_free: bool = True,
               identity_checked: bool = True, episodes: list | None = None) -> Path:
    sha = sha_override or hashlib.sha256(ckpt_path.read_bytes()).hexdigest()
    protocol = {
        "eval_seeds": EVAL_PANEL["eval_seeds"],
        "ranks": EVAL_PANEL["ranks"],
        "episode_steps": EVAL_PANEL["episode_steps"],
        "deterministic": deterministic,
    }
    if source_free:
        protocol["source_free"] = SOURCE_FREE_DECLARATION
    path.write_text(json.dumps({
        "protocol": protocol,
        "env_name": TASK_ENV[task],
        "checkpoint": {"sha256": sha, "global_step": ckpt_step,
                       "identity_checked": identity_checked},
        "episodes": _panel_episodes(return_mean, progress) if episodes is None else episodes,
        "aggregate": {
            "return_mean": return_mean,
            "progress_max_dx_mean": progress,
            "episode_count": episode_count,
        },
    }))
    return path


def _fake_execution_record(path: Path, task: str, checkpoint: Path,
                           execution_id: str, log_name: str, *, pid: int) -> Path:
    log_path = (path.parent / log_name).resolve()
    log_path.write_text(f"successful duplicate execution {execution_id}\n")
    cli = ["python", "train_ptf.py", "--task", task, "--arm", "abstain"]
    cli_json = json.dumps(cli, ensure_ascii=False, separators=(",", ":"))
    path.write_text(json.dumps({
        "schema_version": 1,
        "execution_id": execution_id,
        "job_id": f"{task}_abstain_s1",
        "task": task,
        "arm": "abstain",
        "seed": 1,
        "pid": pid,
        "gpu": "0",
        "cli": cli,
        "cli_sha256": hashlib.sha256(cli_json.encode()).hexdigest(),
        "start_utc": "2026-07-17T00:00:00+00:00",
        "end_utc": "2026-07-17T00:01:00+00:00",
        "exit_code": 0,
        "git_head": "a" * 40,
        "log": str(log_path),
        "log_sha256": hashlib.sha256(log_path.read_bytes()).hexdigest(),
        "artifact_sha256": {
            str(checkpoint): hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        },
    }))
    return path


def _full_setup(tmp_path: Path, *, mutate=None) -> tuple[Path, Path, Path, Path]:
    """合法双任务 setup(crawl NEGATIVE + truck POSITIVE);mutate(task_dir, task)
    钩子可在写 manifest 前篡改单项以构造反例。"""
    manifests = {}
    for task, lease_returns, abstain_returns in (
        ("crawl", [50.0, 49.0, 51.0], [100.0, 100.0, 100.0]),
        ("truck", [200.0, 201.0, 199.0], [100.0, 100.0, 100.0]),
    ):
        task_dir = tmp_path / task
        task_dir.mkdir()
        seeds = []
        for idx in (1, 2, 3):
            entry = {"seed": idx}
            for arm, value in (("lease", lease_returns[idx - 1]),
                               ("abstain", abstain_returns[idx - 1])):
                lease = arm == "lease"
                ckpt = _fake_checkpoint(
                    task_dir / f"{arm}_s{idx}.pt", task, idx, arm,
                )
                progress = 0.0
                if task == "crawl":
                    progress = 0.5 if lease else 2.0   # lease 更小,与 return 负向一致
                evaluation = _fake_eval(
                    task_dir / f"{arm}_s{idx}.json", task, ckpt,
                    lease_returns[idx - 1] if lease else abstain_returns[idx - 1],
                    progress=progress,
                )
                entry[f"{arm}_checkpoint"] = str(ckpt)
                entry[f"{arm}_eval"] = str(evaluation)
            seeds.append(entry)
        dup_ckpt_a = _fake_checkpoint(task_dir / "dupA.pt", task, 1, "abstain")
        dup_ckpt_b = _fake_checkpoint(task_dir / "dupB.pt", task, 1, "abstain",
                                      exec_counts=_task_counts(task, "abstain", source_total=101))
        dup_a = _fake_eval(task_dir / "dupA.json", task, dup_ckpt_a, 100.0)
        dup_b = _fake_eval(task_dir / "dupB.json", task, dup_ckpt_b, 100.5)
        rec_a = task_dir / "execution_record_A.json"
        _fake_execution_record(
            rec_a, task, dup_ckpt_a, f"{task}aaa", f"{task}_abstain_s1.log", pid=101,
        )
        rec_b = task_dir / "execution_record_B.json"
        _fake_execution_record(
            rec_b, task, dup_ckpt_b, f"{task}bbb", f"{task}_abstain_dup_s1.log", pid=102,
        )
        manifest = {"seeds": seeds,
                    "duplicate": {"eval_a": str(dup_a), "eval_b": str(dup_b),
                                  "checkpoint_a": str(dup_ckpt_a),
                                  "checkpoint_b": str(dup_ckpt_b),
                                  "execution_record_a": str(rec_a),
                                  "execution_record_b": str(rec_b)}}
        if mutate is not None:
            mutate(task_dir, task, manifest)
        manifest_path = tmp_path / f"{task}_manifest.json"
        manifest_path.write_text(json.dumps(manifest))
        manifests[task] = manifest_path
    crawl_delta = tmp_path / "crawl_delta.json"
    crawl_delta.write_text(json.dumps(_delta_payload("crawl", tmp_path / "crawl_delta_sources")))
    truck_delta = tmp_path / "truck_delta.json"
    truck_delta.write_text(json.dumps(_delta_payload("truck", tmp_path / "truck_delta_sources")))
    return manifests["crawl"], manifests["truck"], crawl_delta, truck_delta


def test_entry_valid_setup_reaches_p0_pass(tmp_path):
    """阳性对照:全部合法输入时合法裁决路径仍可达(crawl NEG+truck POS)。"""
    result = adjudicate(*_full_setup(tmp_path))
    assert result["joint"]["verdict"] == "P0_PASS"


def test_entry_two_seeds_is_engineering_invalid(tmp_path):
    """ChatGPT 实锤场景:只有两个 seed 绝不允许产生 P0_PASS。"""
    def mutate(task_dir, task, manifest):
        manifest["seeds"] = manifest["seeds"][:2]
    result = adjudicate(*_full_setup(tmp_path, mutate=mutate))
    assert result["joint"]["verdict"] == "ENGINEERING_INVALID"


def test_entry_missing_checkpoint_field_is_invalid(tmp_path):
    def mutate(task_dir, task, manifest):
        manifest["seeds"][0].pop("lease_checkpoint")
    result = adjudicate(*_full_setup(tmp_path, mutate=mutate))
    assert result["joint"]["verdict"] == "ENGINEERING_INVALID"


def test_entry_lease_critic_source_100pct_is_invalid(tmp_path):
    """行为占比 10% 但 critic source 占比 100%——必须拦下。"""
    def mutate(task_dir, task, manifest):
        if task != "crawl":
            return
        ckpt = Path(manifest["seeds"][0]["lease_checkpoint"])
        _fake_checkpoint(ckpt, task, 1, "lease",
                         critic_counts=_task_counts(task, "lease", source_total=1000, student_total=0))
        _fake_eval(Path(manifest["seeds"][0]["lease_eval"]), task, ckpt, 50.0,
                   progress=0.5)
    result = adjudicate(*_full_setup(tmp_path, mutate=mutate))
    assert result["joint"]["verdict"] == "ENGINEERING_INVALID"


def test_entry_abstain_missing_critic_counts_is_invalid(tmp_path):
    """abstain critic 统计缺失——缺失不等于零,必须拦下。"""
    def mutate(task_dir, task, manifest):
        if task != "crawl":
            return
        ckpt = Path(manifest["seeds"][0]["abstain_checkpoint"])
        _fake_checkpoint(ckpt, task, 1, "abstain", critic_counts=None)
        _fake_eval(Path(manifest["seeds"][0]["abstain_eval"]), task, ckpt, 100.0,
                   progress=2.0)
    result = adjudicate(*_full_setup(tmp_path, mutate=mutate))
    assert result["joint"]["verdict"] == "ENGINEERING_INVALID"


def test_entry_duplicate_same_artifact_is_invalid(tmp_path):
    def mutate(task_dir, task, manifest):
        manifest["duplicate"]["eval_b"] = manifest["duplicate"]["eval_a"]
    result = adjudicate(*_full_setup(tmp_path, mutate=mutate))
    assert result["joint"]["verdict"] == "ENGINEERING_INVALID"


def test_entry_wrong_global_step_is_invalid(tmp_path):
    def mutate(task_dir, task, manifest):
        if task != "truck":
            return
        ckpt = Path(manifest["seeds"][1]["lease_checkpoint"])
        _fake_checkpoint(ckpt, task, 2, "lease", step=11500)
    result = adjudicate(*_full_setup(tmp_path, mutate=mutate))
    assert result["joint"]["verdict"] == "ENGINEERING_INVALID"


def test_entry_arm_identity_mismatch_is_invalid(tmp_path):
    """abstain 槽位塞进 admission_mode=all 的 checkpoint——arm 身份必须验证。"""
    def mutate(task_dir, task, manifest):
        if task != "crawl":
            return
        ckpt = Path(manifest["seeds"][2]["abstain_checkpoint"])
        _fake_checkpoint(ckpt, task, 3, "abstain", admission_mode="all")
        _fake_eval(Path(manifest["seeds"][2]["abstain_eval"]), task, ckpt, 100.0,
                   progress=2.0)
    result = adjudicate(*_full_setup(tmp_path, mutate=mutate))
    assert result["joint"]["verdict"] == "ENGINEERING_INVALID"


def test_entry_wrong_panel_is_invalid(tmp_path):
    """episode_count != 32(面板不符)必须拦下。"""
    def mutate(task_dir, task, manifest):
        if task != "truck":
            return
        ckpt = Path(manifest["seeds"][0]["lease_checkpoint"])
        _fake_eval(Path(manifest["seeds"][0]["lease_eval"]), task, ckpt, 200.0,
                   episode_count=8)
    result = adjudicate(*_full_setup(tmp_path, mutate=mutate))
    assert result["joint"]["verdict"] == "ENGINEERING_INVALID"


def test_entry_checkpoint_sha_mismatch_is_invalid(tmp_path):
    """eval 声称的 checkpoint SHA 与 manifest checkpoint 文件不符——拦下。"""
    def mutate(task_dir, task, manifest):
        if task != "crawl":
            return
        ckpt = Path(manifest["seeds"][0]["lease_checkpoint"])
        _fake_eval(Path(manifest["seeds"][0]["lease_eval"]), task, ckpt, 50.0,
                   progress=0.5, sha_override="deadbeef" * 8)
    result = adjudicate(*_full_setup(tmp_path, mutate=mutate))
    assert result["joint"]["verdict"] == "ENGINEERING_INVALID"


# ---------- 七次复核新增反例(三类证据绕过) ----------

def test_entry_duplicate_wrong_step_or_panel_is_invalid(tmp_path):
    """duplicate 用错误 step / 错误面板——必须走全套 eval 验证拦下。"""
    def mutate(task_dir, task, manifest):
        if task != "crawl":
            return
        ckpt = task_dir / "dupA.pt"
        _fake_eval(Path(manifest["duplicate"]["eval_a"]), task, ckpt, 100.0,
                   ckpt_step=11500, episode_count=8)
    result = adjudicate(*_full_setup(tmp_path, mutate=mutate))
    assert result["joint"]["verdict"] == "ENGINEERING_INVALID"


def test_entry_nondeterministic_eval_is_invalid(tmp_path):
    """primary eval 声称非 deterministic / 非 source-free / 未做身份验证——拦下。"""
    def mutate(task_dir, task, manifest):
        if task != "truck":
            return
        ckpt = Path(manifest["seeds"][0]["lease_checkpoint"])
        _fake_eval(Path(manifest["seeds"][0]["lease_eval"]), task, ckpt, 200.0,
                   deterministic=False, source_free=False, identity_checked=False)
    result = adjudicate(*_full_setup(tmp_path, mutate=mutate))
    assert result["joint"]["verdict"] == "ENGINEERING_INVALID"


def test_entry_wrong_treatment_config_is_invalid(tmp_path):
    """checkpoint 用错误 source bank / MCG groups / noise seed——拦下。"""
    def mutate(task_dir, task, manifest):
        if task != "crawl":
            return
        ckpt = Path(manifest["seeds"][0]["lease_checkpoint"])
        _fake_checkpoint(ckpt, task, 1, "lease",
                         ptf_overrides={
                             "source_bank": "configs/source_banks/h1hand_loco_safe_crawl.yaml",
                             "mcg_groups": ["legs_torso"],
                             "resume_noise_seed": 12345,
                         })
        _fake_eval(Path(manifest["seeds"][0]["lease_eval"]), task, ckpt, 50.0,
                   progress=0.5)
    result = adjudicate(*_full_setup(tmp_path, mutate=mutate))
    assert result["joint"]["verdict"] == "ENGINEERING_INVALID"


def test_entry_wrong_episode_panel_is_invalid(tmp_path):
    """episodes 明细的 reset seed 集合与冻结面板不符——拦下。"""
    def mutate(task_dir, task, manifest):
        if task != "truck":
            return
        ckpt = Path(manifest["seeds"][2]["abstain_checkpoint"])
        _fake_eval(Path(manifest["seeds"][2]["abstain_eval"]), task, ckpt, 100.0,
                   episodes=[{"seed": i} for i in range(32)])
    result = adjudicate(*_full_setup(tmp_path, mutate=mutate))
    assert result["joint"]["verdict"] == "ENGINEERING_INVALID"


# ---------- 八次复核新增反例(五类证据绕过) ----------

def test_entry_delta_wrong_task_is_invalid(tmp_path):
    """crawl δ 文件声明属于 truck——必须拦下。"""
    def mutate(task_dir, task, manifest):
        pass
    manifests = _full_setup(tmp_path, mutate=mutate)
    wrong = tmp_path / "crawl_delta_wrong.json"
    payload = _delta_payload("truck", tmp_path / "wrong_delta_sources")
    payload["delta_progress_m"] = 0.5
    wrong.write_text(json.dumps(payload))
    result = adjudicate(manifests[0], manifests[1], wrong, manifests[3])
    assert result["joint"]["verdict"] == "ENGINEERING_INVALID"


def test_entry_missing_duplicate_checkpoint_is_invalid(tmp_path):
    """manifest 删除 duplicate 的 checkpoint_a——必填,缺失即拦。"""
    def mutate(task_dir, task, manifest):
        manifest["duplicate"].pop("checkpoint_a")
    result = adjudicate(*_full_setup(tmp_path, mutate=mutate))
    assert result["joint"]["verdict"] == "ENGINEERING_INVALID"


def test_entry_duplicate_wrong_identity_is_invalid(tmp_path):
    """duplicate-B checkpoint 错 seed/错 arm——走全套 checkpoint 验证拦下。"""
    def mutate(task_dir, task, manifest):
        if task != "crawl":
            return
        ckpt = Path(manifest["duplicate"]["checkpoint_b"])
        _fake_checkpoint(ckpt, task, 2, "lease")
        _fake_eval(Path(manifest["duplicate"]["eval_b"]), task, ckpt, 100.5)
    result = adjudicate(*_full_setup(tmp_path, mutate=mutate))
    assert result["joint"]["verdict"] == "ENGINEERING_INVALID"


def test_entry_doubled_episode_rows_is_invalid(tmp_path):
    """32 行复制成 64 行但 seed 集合相同——行数与每 seed 恰好一次都必须验证。"""
    def mutate(task_dir, task, manifest):
        if task != "truck":
            return
        ckpt = Path(manifest["seeds"][0]["lease_checkpoint"])
        doubled = _panel_episodes(200.0) + _panel_episodes(200.0)
        _fake_eval(Path(manifest["seeds"][0]["lease_eval"]), task, ckpt, 200.0,
                   episodes=doubled)
    result = adjudicate(*_full_setup(tmp_path, mutate=mutate))
    assert result["joint"]["verdict"] == "ENGINEERING_INVALID"


def test_entry_source_free_string_false_is_invalid(tmp_path):
    """source_free 写成字符串 'false'——truthiness 判断会放过,白名单拦下。"""
    def mutate(task_dir, task, manifest):
        if task != "crawl":
            return
        eval_path = Path(manifest["seeds"][0]["lease_eval"])
        data = json.loads(eval_path.read_text())
        data["protocol"]["source_free"] = "false"
        eval_path.write_text(json.dumps(data))
    result = adjudicate(*_full_setup(tmp_path, mutate=mutate))
    assert result["joint"]["verdict"] == "ENGINEERING_INVALID"


def test_entry_wrong_source_schema_is_invalid(tmp_path):
    """truck checkpoint 带 3 源 source_names/4 列计数——列数与源集合必须
    等于冻结值(truck=4 源+null,5 列)。"""
    def mutate(task_dir, task, manifest):
        if task != "truck":
            return
        ckpt = Path(manifest["seeds"][0]["lease_checkpoint"])
        _fake_checkpoint(ckpt, task, 1, "lease",
                         exec_counts=[40, 30, 30, 900],
                         critic_counts=[40, 30, 30, 900],
                         source_names=["stand", "walk", "run", "null"])
        _fake_eval(Path(manifest["seeds"][0]["lease_eval"]), task, ckpt, 200.0)
    result = adjudicate(*_full_setup(tmp_path, mutate=mutate))
    assert result["joint"]["verdict"] == "ENGINEERING_INVALID"


def test_entry_identical_execution_ids_is_invalid(tmp_path):
    """execution_id 相同——独立性证明失败即拦(SHA 相同本身不再是无效条件)。"""
    def mutate(task_dir, task, manifest):
        if task != "crawl":
            return
        path_a = Path(manifest["duplicate"]["execution_record_a"])
        path_b = Path(manifest["duplicate"]["execution_record_b"])
        record_a = json.loads(path_a.read_text())
        record_b = json.loads(path_b.read_text())
        record_b["execution_id"] = record_a["execution_id"]
        path_b.write_text(json.dumps(record_b))
    result = adjudicate(*_full_setup(tmp_path, mutate=mutate))
    assert result["joint"]["verdict"] == "ENGINEERING_INVALID"


def test_entry_identical_checkpoint_sha_is_still_valid(tmp_path):
    """两次独立重启产出逐位相同 checkpoint(理想 d_dup=0)——不得因 SHA 相同
    判无效;独立性由 execution record 证明(八次复核 6)。"""
    def mutate(task_dir, task, manifest):
        ckpt_a = Path(manifest["duplicate"]["checkpoint_a"])
        ckpt_b = Path(manifest["duplicate"]["checkpoint_b"])
        _fake_checkpoint(ckpt_a, task, 1, "abstain")
        import shutil
        shutil.copy2(ckpt_a, ckpt_b)
        _fake_eval(Path(manifest["duplicate"]["eval_a"]), task, ckpt_a, 100.0)
        _fake_eval(Path(manifest["duplicate"]["eval_b"]), task, ckpt_b, 100.0)
        _fake_execution_record(
            Path(manifest["duplicate"]["execution_record_a"]), task, ckpt_a,
            f"{task}aaa", f"{task}_abstain_s1.log", pid=101,
        )
        _fake_execution_record(
            Path(manifest["duplicate"]["execution_record_b"]), task, ckpt_b,
            f"{task}bbb", f"{task}_abstain_dup_s1.log", pid=102,
        )
    result = adjudicate(*_full_setup(tmp_path, mutate=mutate))
    assert result["joint"]["verdict"] == "P0_PASS"


# ---------- 九次复核新增反例(证据内容与物理输入绑定) ----------

def test_entry_delta_nonexistent_sources_are_invalid(tmp_path):
    """64 位字符串不是证据:δ 必须能回到真实日志并重算。"""
    manifests = _full_setup(tmp_path)
    payload = json.loads(Path(manifests[2]).read_text())
    old_paths = list(payload["input_log_sha256"])
    payload["input_log_sha256"] = {
        f"/does/not/exist/{idx}.log": payload["input_log_sha256"][old]
        for idx, old in enumerate(old_paths)
    }
    payload["per_seed_window_means"] = {
        path: mean
        for path, mean in zip(
            payload["input_log_sha256"], payload["per_seed_window_means"].values()
        )
    }
    Path(manifests[2]).write_text(json.dumps(payload))
    result = adjudicate(*manifests)
    assert result["joint"]["verdict"] == "ENGINEERING_INVALID"


def test_entry_eval_aggregate_episode_mismatch_is_invalid(tmp_path):
    """aggregate 不得成为可独立篡改的第二套真值。"""
    def mutate(task_dir, task, manifest):
        if task != "truck":
            return
        eval_path = Path(manifest["seeds"][0]["lease_eval"])
        data = json.loads(eval_path.read_text())
        data["aggregate"]["return_mean"] += 123.0
        eval_path.write_text(json.dumps(data))

    result = adjudicate(*_full_setup(tmp_path, mutate=mutate))
    assert result["joint"]["verdict"] == "ENGINEERING_INVALID"


def test_entry_source_free_prefix_is_invalid(tmp_path):
    """语义相近或带前缀/后缀的声明都不能冒充冻结 source-free 协议。"""
    def mutate(task_dir, task, manifest):
        if task != "crawl":
            return
        eval_path = Path(manifest["seeds"][0]["lease_eval"])
        data = json.loads(eval_path.read_text())
        data["protocol"]["source_free"] = SOURCE_FREE_DECLARATION + " maybe"
        eval_path.write_text(json.dumps(data))

    result = adjudicate(*_full_setup(tmp_path, mutate=mutate))
    assert result["joint"]["verdict"] == "ENGINEERING_INVALID"


def test_entry_execution_record_missing_log_is_invalid(tmp_path):
    """execution record 必须绑定真实日志，而非仅提供 execution_id。"""
    def mutate(task_dir, task, manifest):
        if task != "truck":
            return
        record_path = Path(manifest["duplicate"]["execution_record_b"])
        record = json.loads(record_path.read_text())
        record["log"] = str(task_dir / "missing.log")
        record_path.write_text(json.dumps(record))

    result = adjudicate(*_full_setup(tmp_path, mutate=mutate))
    assert result["joint"]["verdict"] == "ENGINEERING_INVALID"


def test_entry_duplicate_nonzero_source_counts_are_invalid(tmp_path):
    """duplicate 也必须实际实现 exact abstention，不能只在配置中声明 none。"""
    def mutate(task_dir, task, manifest):
        if task != "crawl":
            return
        ckpt = Path(manifest["duplicate"]["checkpoint_b"])
        _fake_checkpoint(
            ckpt, task, 1, "abstain",
            exec_counts=_task_counts(task, "lease", source_total=1, student_total=999),
            critic_counts=_task_counts(task, "lease", source_total=1, student_total=999),
        )
        _fake_eval(Path(manifest["duplicate"]["eval_b"]), task, ckpt, 100.5)
        _fake_execution_record(
            Path(manifest["duplicate"]["execution_record_b"]), task, ckpt,
            f"{task}bbb", f"{task}_abstain_dup_s1.log", pid=102,
        )

    result = adjudicate(*_full_setup(tmp_path, mutate=mutate))
    assert result["joint"]["verdict"] == "ENGINEERING_INVALID"


def test_entry_malformed_delta_container_is_invalid_not_exception(tmp_path):
    manifests = _full_setup(tmp_path)
    payload = json.loads(Path(manifests[2]).read_text())
    payload["input_log_sha256"] = []
    payload["per_seed_window_means"] = "not-an-object"
    Path(manifests[2]).write_text(json.dumps(payload))
    result = adjudicate(*manifests)
    assert result["joint"]["verdict"] == "ENGINEERING_INVALID"


def test_entry_malformed_episode_rows_are_invalid_not_exception(tmp_path):
    def mutate(task_dir, task, manifest):
        if task != "truck":
            return
        eval_path = Path(manifest["seeds"][0]["lease_eval"])
        data = json.loads(eval_path.read_text())
        data["episodes"] = [17] * 32
        eval_path.write_text(json.dumps(data))

    result = adjudicate(*_full_setup(tmp_path, mutate=mutate))
    assert result["joint"]["verdict"] == "ENGINEERING_INVALID"
