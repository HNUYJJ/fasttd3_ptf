"""P0 orchestrator 测试(七次复核阻塞 1 与最小修复 2):
GPU 池无碰撞、失败清理、并发上限、产物强制、计划展开计数、
duplicate 归档语义、smoke 计划隔离。"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import p0_orchestrator as orch  # noqa: E402


def _job(job_id: str, cli: list[str], log_dir: Path, artifacts: list[str] | None = None) -> dict:
    return {
        "id": job_id,
        "cli": cli,
        "log": str(log_dir / f"{job_id}.log"),
        "expected_artifacts": artifacts or [],
    }


def test_plan_counts_and_manifest_consistency():
    plan = orch.build_plan()
    assert len(plan["anchors"]) == 6
    assert len(plan["branches"]) == 12
    assert len(plan["duplicates"]) == 2
    assert len(plan["evals"]) == 52  # 48(12 分支×4 ckpt) + 4(2 duplicate×primary A/B)
    for task, manifest in plan["adjudication_manifests"].items():
        assert [entry["seed"] for entry in manifest["seeds"]] == [1, 2, 3]
        eval_outs = {job["out"] for job in plan["evals"]}
        for entry in manifest["seeds"]:
            assert entry["lease_eval"] in eval_outs
            assert entry["abstain_eval"] in eval_outs
        assert manifest["duplicate"]["eval_a"] in eval_outs
        assert manifest["duplicate"]["eval_b"] in eval_outs
        assert manifest["duplicate"]["checkpoint_a"] != manifest["duplicate"]["checkpoint_b"]
    # duplicate 使用独立日志(不得覆盖正式 abstain 的日志)。
    branch_logs = {job["log"] for job in plan["branches"]}
    for dup in plan["duplicates"]:
        assert dup["log"] not in branch_logs
    # 每个 eval job 都带 expected_artifacts(执行器自动确认 JSON 产出)。
    assert all(job.get("expected_artifacts") for job in plan["evals"])
    # 冻结对象必须包括 manifest 实际指向的 source 权重，而不只是 YAML/JSON。
    checkpoint_hashes = plan["frozen_inputs"]["source_checkpoint_sha256"]
    assert checkpoint_hashes
    for checkpoint, expected in checkpoint_hashes.items():
        assert checkpoint.endswith(".pt")
        assert orch._file_sha256(orch.REPO_ROOT / checkpoint) == expected


def test_gpu_pool_returns_freed_gpu(tmp_path):
    """job1(快,GPU a)完成后,job3 必须拿回 job1 释放的 GPU——
    len(active) 索引法会把 job3 派到 job2 正占用的卡上(七次复核实锤)。"""
    marker = tmp_path / "gpu_{}.txt"
    jobs = [
        _job("fast", ["bash", "-c", f"echo $CUDA_VISIBLE_DEVICES > {marker.with_name('gpu_fast.txt')}"], tmp_path),
        _job("slow", ["bash", "-c", f"echo $CUDA_VISIBLE_DEVICES > {marker.with_name('gpu_slow.txt')}; sleep 4"], tmp_path),
        _job("third", ["bash", "-c", f"echo $CUDA_VISIBLE_DEVICES > {marker.with_name('gpu_third.txt')}"], tmp_path),
    ]
    orch._run_queue(jobs, ["7", "8"], {})
    gpu_fast = (tmp_path / "gpu_fast.txt").read_text().strip()
    gpu_slow = (tmp_path / "gpu_slow.txt").read_text().strip()
    gpu_third = (tmp_path / "gpu_third.txt").read_text().strip()
    assert {gpu_fast, gpu_slow} == {"7", "8"}
    # third 必须复用 fast 释放的卡,绝不能与 slow 同卡。
    assert gpu_third == gpu_fast
    assert gpu_third != gpu_slow


def test_queue_failure_terminates_active_jobs(tmp_path):
    """任一作业失败:其余活动子进程必须被终止,不留训练孤儿。"""
    pid_file = tmp_path / "sleeper_pid.txt"
    jobs = [
        _job("sleeper", ["bash", "-c", f"echo $$ > {pid_file}; sleep 60"], tmp_path),
        _job("failer", ["bash", "-c", "sleep 0.5; exit 3"], tmp_path),
    ]
    with pytest.raises(RuntimeError, match="failer"):
        orch._run_queue(jobs, ["0", "1"], {})
    sleeper_pid = int(pid_file.read_text().strip())
    # terminate 后进程应已不存在(给关闭一点时间)。
    time.sleep(1)
    assert not Path(f"/proc/{sleeper_pid}").exists()


def test_queue_rejects_more_than_two_gpus(tmp_path):
    with pytest.raises(ValueError, match="at most 2"):
        orch._run_queue([], ["0", "1", "2"], {})


def test_queue_enforces_expected_artifacts(tmp_path):
    jobs = [_job("noartifact", ["true"], tmp_path, artifacts=[str(tmp_path / "missing.json")])]
    with pytest.raises(RuntimeError, match="missing expected artifact"):
        orch._run_queue(jobs, ["0"], {})


def test_queue_rejects_stale_expected_artifact_before_launch(tmp_path):
    """旧产物不能被新进程的 exit=0 重新认证。"""
    stale = tmp_path / "stale.pt"
    stale.write_text("OLD")
    marker = tmp_path / "should_not_launch"
    jobs = [_job(
        "stale",
        ["bash", "-c", f"touch {marker}"],
        tmp_path,
        artifacts=[str(stale)],
    )]
    with pytest.raises(FileExistsError, match="pre-existing expected artifacts"):
        orch._run_queue(jobs, ["0"], {})
    assert not marker.exists()


def test_duplicate_archive_flow(tmp_path):
    """归档语义:A=第一次产物(不可变),B=第二次,正式路径恢复为 A 内容。"""
    task = "crawl"
    run_name = orch._run_name(f"p0_{task}_abstain", task, 1)
    models = tmp_path / "models"
    models.mkdir()
    products = [f"{run_name}_{step}.pt" for step in orch.CKPT_STEPS] + [f"{run_name}_final.pt"]
    for name in products:
        (models / name).write_text("FIRST")
    first_record = tmp_path / "logs" / "first.json"
    second_record = tmp_path / "logs" / "second.json"
    first_record.parent.mkdir(parents=True)
    first_record.write_text(json.dumps({"execution_id": "A"}))

    def fake_queue(jobs, gpus, env_extra):
        for name in products:
            (models / name).write_text("SECOND")
        second_record.write_text(json.dumps({"execution_id": "B"}))

    dup = {
        "task": task,
        "archive_a": f"models/p0_dup_archive/{task}_A",
        "archive_b": f"models/p0_dup_archive/{task}_B",
        "first_run_execution_record": str(first_record.relative_to(tmp_path)),
        "execution_record": str(second_record.relative_to(tmp_path)),
        "cli": [], "id": "dup", "log": str(tmp_path / "dup.log"),
    }
    orch._run_duplicates([dup], ["0"], run_queue=fake_queue, repo_root=tmp_path)
    for name in products:
        assert (tmp_path / dup["archive_a"] / name).read_text() == "FIRST"
        assert (tmp_path / dup["archive_b"] / name).read_text() == "SECOND"
        # 正式路径 = A(第一次运行)。
        assert (models / name).read_text() == "FIRST"
    # 归档不可变:重复执行必须拒绝。
    for name in products:
        (models / name).write_text("FIRST")
    with pytest.raises(RuntimeError, match="already exists"):
        orch._run_duplicates([dup], ["0"], run_queue=fake_queue, repo_root=tmp_path)


def test_smoke_plan_is_isolated_from_formal_paths():
    plan = orch.build_smoke_plan()
    assert len(plan["jobs"]) == 3
    formal_prefixes = ("checkpoints/p0_anchors", orch.OUT_ROOT, "models/p0_dup_archive")
    for job in plan["jobs"]:
        for token in [
            *job["cli"],
            *(job.get("expected_artifacts") or []),
            job.get("log", ""),
            job.get("execution_record", ""),
        ]:
            for prefix in formal_prefixes:
                assert prefix not in str(token), f"smoke job {job['id']} touches formal path: {token}"
    assert plan["jobs"][1]["execution_record"].startswith(orch.SMOKE_ROOT)
    # smoke 分支 exp-name 使用 p0smoke 前缀(与正式 p0_ 运行名不冲突)。
    branch = plan["jobs"][1]
    exp = branch["cli"][branch["cli"].index("--exp-name") + 1]
    assert exp.startswith("p0smoke")


# ---------- 八次复核新增 ----------

def test_queue_kills_grandchildren_via_process_group(tmp_path):
    """父进程再启动子进程(模拟 SubprocVecEnv worker):失败清理必须干掉整个
    进程组,孙进程不得存活(八次复核阻塞 2 实锤场景)。"""
    grandchild_pid_file = tmp_path / "grandchild_pid.txt"
    jobs = [
        _job("parent_with_child",
             ["bash", "-c",
              f"bash -c 'echo $$ > {grandchild_pid_file}; sleep 60' & sleep 60"],
             tmp_path),
        _job("failer", ["bash", "-c", "sleep 1; exit 3"], tmp_path),
    ]
    with pytest.raises(RuntimeError, match="failer"):
        orch._run_queue(jobs, ["0", "1"], {})
    time.sleep(1)
    grandchild_pid = int(grandchild_pid_file.read_text().strip())
    assert not Path(f"/proc/{grandchild_pid}").exists(), "grandchild survived group kill"


def test_queue_cleans_descendants_of_the_failed_job_itself(tmp_path):
    """失败 leader 自己先退出时，其遗留 worker 不在 active 列表里；完成路径
    必须主动 kill 该 leader 的进程组，而不能只清理其他并行作业。"""
    child_pid_file = tmp_path / "failed_job_child_pid.txt"
    child_code = "import time; time.sleep(60)"
    parent_code = (
        "import subprocess,sys; "
        f"p=subprocess.Popen([sys.executable,'-c',{child_code!r}]); "
        f"open({str(child_pid_file)!r},'w').write(str(p.pid)); "
        "sys.exit(3)"
    )
    jobs = [_job("failed_parent", [sys.executable, "-c", parent_code], tmp_path)]
    with pytest.raises(RuntimeError, match="failed_parent"):
        orch._run_queue(jobs, ["0"], {})
    child_pid = int(child_pid_file.read_text())
    time.sleep(0.5)
    assert not Path(f"/proc/{child_pid}").exists(), "failed job's child survived cleanup"


def test_queue_rejects_duplicate_gpu_ids():
    with pytest.raises(ValueError, match="duplicate GPU ids"):
        orch._run_queue([], ["0", "0"], {})


def test_queue_metrics_collection(tmp_path):
    metrics: dict = {}
    jobs = [_job("metered", ["bash", "-c", "echo '12.34 sps, foo'; sleep 3"], tmp_path)]
    orch._run_queue(jobs, ["0"], {}, metrics=metrics)
    entry = metrics["metered"]
    assert entry["exit_code"] == 0
    assert entry["wall_seconds"] >= 3
    assert entry["sps"] == 12.34
    assert entry["peak_group_rss_gib"] >= 0


def test_duplicate_transactional_rollback(tmp_path):
    """第二次运行失败:A 暂存产物必须恢复回正式路径,且不留 staging/半成品
    归档,重试不被 'archive exists' 卡死(八次复核 4 实锤场景)。"""
    task = "crawl"
    run_name = orch._run_name(f"p0_{task}_abstain", task, 1)
    models = tmp_path / "models"
    models.mkdir()
    products = [f"{run_name}_{step}.pt" for step in orch.CKPT_STEPS] + [f"{run_name}_final.pt"]
    for name in products:
        (models / name).write_text("FIRST")
    first_record = tmp_path / "logs" / "first.json"
    second_record = tmp_path / "logs" / "second.json"
    first_record.parent.mkdir(parents=True)
    first_record.write_text(json.dumps({"execution_id": "A"}))

    def failing_queue(jobs, gpus, env_extra):
        raise RuntimeError("simulated second-run failure")

    dup = {
        "task": task,
        "archive_a": f"models/p0_dup_archive/{task}_A",
        "archive_b": f"models/p0_dup_archive/{task}_B",
        "first_run_execution_record": str(first_record.relative_to(tmp_path)),
        "execution_record": str(second_record.relative_to(tmp_path)),
        "cli": [], "id": "dup", "log": str(tmp_path / "dup.log"),
    }
    with pytest.raises(RuntimeError, match="simulated"):
        orch._run_duplicates([dup], ["0"], run_queue=failing_queue, repo_root=tmp_path)
    # 回滚完成:正式产物齐全,无归档目录残留。
    for name in products:
        assert (models / name).read_text() == "FIRST"
    assert not (tmp_path / dup["archive_a"]).exists()
    assert not (tmp_path / dup["archive_b"]).exists()
    assert not list((models / "p0_dup_archive").glob("*.staging")) if (models / "p0_dup_archive").exists() else True

    # 缺产物的 preflight 失败也不得留下目录。
    (models / products[0]).unlink()
    with pytest.raises(RuntimeError, match="precondition missing"):
        orch._run_duplicates([dup], ["0"], run_queue=failing_queue, repo_root=tmp_path)
    assert not (tmp_path / dup["archive_a"]).exists()

    # 修复前置后重试可以正常走完(不被 'archive exists' 卡死)。
    (models / products[0]).write_text("FIRST")

    def ok_queue(jobs, gpus, env_extra):
        for name in products:
            (models / name).write_text("SECOND")
        second_record.write_text(json.dumps({"execution_id": "B"}))

    orch._run_duplicates([dup], ["0"], run_queue=ok_queue, repo_root=tmp_path)
    for name in products:
        assert (tmp_path / dup["archive_a"] / name).read_text() == "FIRST"
        assert (tmp_path / dup["archive_b"] / name).read_text() == "SECOND"
        assert (models / name).read_text() == "FIRST"
    # execution_record 已写入两侧归档。
    assert (tmp_path / dup["archive_a"] / "execution_record.json").exists()
    assert (tmp_path / dup["archive_b"] / "execution_record.json").exists()


def test_duplicate_partial_second_run_restores_every_a_product(tmp_path):
    """第二次运行写出部分 B 后失败时，正式路径不能残留 B/B/A/A/A 混合物。"""
    task = "crawl"
    run_name = orch._run_name(f"p0_{task}_abstain", task, 1)
    models = tmp_path / "models"
    models.mkdir()
    products = [f"{run_name}_{step}.pt" for step in orch.CKPT_STEPS] + [f"{run_name}_final.pt"]
    for name in products:
        (models / name).write_text("FIRST")
    first_record = tmp_path / "logs" / "first.json"
    first_record.parent.mkdir(parents=True)
    first_record.write_text(json.dumps({"execution_id": "A"}))
    dup = {
        "task": task,
        "archive_a": f"models/p0_dup_archive/{task}_A",
        "archive_b": f"models/p0_dup_archive/{task}_B",
        "first_run_execution_record": str(first_record.relative_to(tmp_path)),
        "execution_record": "logs/second.json",
        "cli": [], "id": "dup", "log": str(tmp_path / "dup.log"),
    }

    def partial_queue(jobs, gpus, env_extra):
        for name in products[:2]:
            (models / name).write_text("SECOND")
        (tmp_path / dup["execution_record"]).write_text(json.dumps({"execution_id": "B"}))
        raise RuntimeError("partial B failure")

    with pytest.raises(RuntimeError, match="partial B"):
        orch._run_duplicates([dup], ["0"], run_queue=partial_queue, repo_root=tmp_path)
    assert [(models / name).read_text() for name in products] == ["FIRST"] * len(products)
    assert not (tmp_path / dup["archive_a"]).exists()
    assert not (tmp_path / dup["archive_b"]).exists()
    assert not (tmp_path / dup["execution_record"]).exists()


def test_smoke_execution_is_strictly_sequential(monkeypatch, tmp_path):
    """smoke 三作业存在硬依赖:执行必须逐作业顺序调用队列(每次一个 job),
    绝不将 anchor/branch/eval 同时入队(八次复核阻塞 1)。"""
    calls: list[list[str]] = []

    def recording_queue(jobs, gpus, env_extra, metrics=None):
        calls.append([job["id"] for job in jobs])
        if metrics is not None:
            for job in jobs:
                metrics[job["id"]] = {"sps": 100.0, "wall_seconds": 1.0}

    monkeypatch.setattr(orch, "_run_queue", recording_queue)
    monkeypatch.setattr(sys, "argv",
                        ["p0_orchestrator.py", "--smoke", "--execute",
                         "--acknowledge-authorization", "--gpus", "0,1"])
    orch.main()
    assert len(calls) == 3
    assert all(len(batch) == 1 for batch in calls), f"smoke jobs were batched: {calls}"
    assert calls[0][0].startswith("smoke_anchor")
    assert calls[1][0].startswith("smoke_branch")
    assert calls[2][0].startswith("eval_smoke")


def test_smoke_rejects_duplicate_gpu_ids_before_single_gpu_slice(monkeypatch, tmp_path):
    """--gpus 0,0 不能先切成 [0] 后绕过重复 ID 审计。"""
    plan = {"jobs": [], "git_head": "a" * 40}
    monkeypatch.setattr(orch, "build_smoke_plan", lambda: plan)
    monkeypatch.setattr(orch, "SMOKE_ROOT", str(tmp_path / "smoke"))
    monkeypatch.setattr(
        sys, "argv",
        ["p0_orchestrator.py", "--smoke", "--execute",
         "--acknowledge-authorization", "--gpus", "0,0"],
    )
    with pytest.raises(ValueError, match="duplicate GPU ids"):
        orch.main()


def test_frozen_plan_verification(tmp_path, monkeypatch):
    """--execute-plan 的指纹核对:SHA 不符/HEAD 不符必须拒绝。"""
    plan = orch.build_plan()
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(__import__("json").dumps(plan))
    good_sha = orch._file_sha256(plan_path)
    with pytest.raises(RuntimeError, match="sha256 mismatch"):
        orch.verify_frozen_plan(plan_path, "0" * 64)
    # HEAD 相同+树干净时通过(测试环境树可能脏,仅验证 SHA 分支已覆盖;
    # HEAD 不符分支:篡改 plan 的 git_head)。
    plan["git_head"] = "not-a-real-head"
    plan_path.write_text(__import__("json").dumps(plan))
    with pytest.raises(RuntimeError, match="git_head"):
        orch.verify_frozen_plan(plan_path, orch._file_sha256(plan_path))
