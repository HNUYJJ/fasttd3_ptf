"""P0 orchestrator(六次复核阻塞问题 2):把预注册矩阵展开为可执行计划。

职责:
1. --dry-run(默认):确定性展开并打印全部作业——6 anchor 命令、14 branch
   命令(anchor-resume/noise seed/GPU/输出路径)、52 个 checkpoint 离线评估
   任务(含 evaluator 身份验证参数)、2 个 adjudication manifest 骨架、
   产物完整性检查清单——并写入 plan.json 供 SHA256 冻结;
2. --execute:按计划双并发执行(E2 RAM 上限)。执行仍受 PI 授权流程约束:
   本脚本不自行判断授权,--execute 必须与 --acknowledge-authorization 同时
   给出,后者仅是操作者对"已获 PI 批准"的显式确认记录。

duplicate 语义:与对应 abstain 分支 CLI **逐位一致**(含 exp-name),第一次
运行的产物先归档改名,再重启进程跑第二次——保证被测对象是"完全同参数的
进程重启重复性"(d_dup),而非任何配置差异。

评估任务口径:52 = 12 正式分支 × 4 checkpoint(48,全部离线评估)
+ 2 duplicate × 2 份 primary checkpoint(4)。duplicate 的中间 checkpoint
不评——d_dup 按预注册定义只在 primary endpoint(13000)度量。

用法:
  python scripts/p0_orchestrator.py --dry-run          # 打印+落盘 plan.json
  python scripts/p0_orchestrator.py --execute \
      --acknowledge-authorization --gpus 0,1           # 授权后执行
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

TRAIN = "fasttd3_ptf/official_fasttd3_ptf/train_ptf.py"
SEEDS = (1, 2, 3)
NOISE_SEED_BASE = 77000
CKPT_STEPS = (10750, 11500, 12250, 13000)
PRIMARY_STEP = 13000
OUT_ROOT = "logs/p0_lease_oracle"
DELTA_PATHS = {
    "crawl": "configs/experiments/p0_frozen_delta_crawl.json",
    "truck": "configs/experiments/p0_frozen_delta_truck.json",
}

TASKS = {
    "crawl": {
        "env": "h1hand-crawl-v0",
        "bank": "configs/source_banks/h1hand_loco_wfix_crawl.yaml",
        "student_logit": "16.6823567039",
        "mcg_groups": "legs_torso,arms,hands",
        "provenance_groups": "3",
    },
    "truck": {
        "env": "h1hand-truck-v0",
        "bank": "configs/source_banks/h1hand_hurdle4_wfix_truck.yaml",
        "student_logit": "16.4139012941",
        "mcg_groups": "legs_torso,arms",
        "provenance_groups": "2",
    },
}

COMMON_CLI = [
    "--num-envs", "128",
    "--batch-size", "32768",
    "--buffer-size", "51200",
    "--learning-starts", "10",
    "--eval-interval", "0",
    "--render-interval", "0",
    "--save-interval", "0",
]


def _anchor_dir(task: str, seed: int) -> str:
    return f"checkpoints/p0_anchors/{task}_s{seed}"


def _run_name(exp_name: str, task: str, seed: int) -> str:
    return f"{TASKS[task]['env']}__{exp_name}__{seed}"


def _anchor_job(task: str, seed: int) -> dict:
    exp = f"p0_anchor_{task}"
    cli = [
        sys.executable, TRAIN,
        "--env-name", TASKS[task]["env"],
        "--exp-name", exp,
        "--seed", str(seed),
        *COMMON_CLI,
        "--ptf-anchor-step", "10000",
        "--ptf-anchor-dir", _anchor_dir(task, seed),
        "--ptf-run-stop-step", "10000",
        "--ptf-anchor-provenance-groups", TASKS[task]["provenance_groups"],
    ]
    return {
        "id": f"anchor_{task}_s{seed}",
        "kind": "anchor",
        "cli": cli,
        "log": f"{OUT_ROOT}/anchor_{task}_s{seed}.log",
        "expected_artifacts": [f"{_anchor_dir(task, seed)}/manifest.json"],
    }


def _branch_job(task: str, arm: str, seed: int, *, exp_override: str | None = None) -> dict:
    spec = TASKS[task]
    exp = exp_override or f"p0_{task}_{arm}"
    lease = arm == "lease"
    cli = [
        sys.executable, TRAIN,
        "--env-name", spec["env"],
        "--exp-name", exp,
        "--seed", str(seed),
        *COMMON_CLI,
        "--ptf-anchor-resume", _anchor_dir(task, seed),
        "--ptf-run-stop-step", "13000",
        "--ptf-eval-checkpoint-steps", ",".join(str(s) for s in CKPT_STEPS),
        "--ptf-resume-noise-seed", str(NOISE_SEED_BASE + seed),
        "--ptf-source-bank", spec["bank"],
        "--ptf-mcg",
        "--ptf-mcg-groups", spec["mcg_groups"],
        "--ptf-mcg-warmup-mode", "admission_bootstrap",
        "--ptf-mcg-ablation", "bootstrap_only",
        "--ptf-mcg-warmup-steps", "30000",
        "--ptf-admission-mode", "all" if lease else "none",
        "--ptf-admission-student-logit", spec["student_logit"],
        "--ptf-admission-expected-source-mass", "0.10" if lease else "0.0",
        "--ptf-admission-replay-handoff", "physical_after_authority",
        "--ptf-admission-replay-recency-half-life", "0",
        "--ptf-admission-replay-uniform-mix", "1.0",
        "--ptf-admission-replay-priority-alpha", "0",
    ]
    run_name = _run_name(exp, task, seed)
    return {
        "id": f"{task}_{arm}_s{seed}" if exp_override is None else exp_override + f"_s{seed}",
        "kind": "branch",
        "task": task,
        "arm": arm,
        "seed": seed,
        "run_name": run_name,
        "cli": cli,
        "log": f"{OUT_ROOT}/{exp}_s{seed}.log",
        "execution_record": f"{OUT_ROOT}/execution/{exp}_s{seed}.json",
        "expected_artifacts": [f"models/{run_name}_{step}.pt" for step in CKPT_STEPS]
        + [f"models/{run_name}_final.pt"],
    }


def _eval_job(task: str, arm: str, seed: int, step: int, checkpoint: str, *, tag: str | None = None) -> dict:
    identity = [
        "--expect-global-step", str(step),
        "--expect-seed", str(seed),
        "--expect-admission-mode", "all" if arm == "lease" else "none",
    ]
    out = f"{OUT_ROOT}/eval/{tag or f'{task}_{arm}_s{seed}'}_{step}.json"
    return {
        "id": f"eval_{tag or f'{task}_{arm}_s{seed}'}_{step}",
        "kind": "eval",
        "cli": [
            sys.executable, "scripts/p0_evaluator.py",
            "--checkpoint", checkpoint,
            "--env-name", TASKS[task]["env"],
            "--out", out,
            *identity,
        ],
        "out": out,
        # 执行器按此自动确认 JSON 已生成(七次复核:完整性检查必须真正执行)。
        "expected_artifacts": [out],
    }


def build_plan() -> dict:
    anchors = [_anchor_job(task, seed) for task in TASKS for seed in SEEDS]
    branches = []
    for task in TASKS:
        for arm in ("lease", "abstain"):
            for seed in SEEDS:
                branches.append(_branch_job(task, arm, seed))
    # duplicate:与 abstain s1 的 CLI 逐位一致(含 exp-name;log 是执行器的
    # stdout 重定向,不属于 CLI,故用独立日志文件——七次复核:不得覆盖正式
    # abstain 的日志)。归档语义(定稿):
    #   正式 abstain_s1 = 第一次运行(A)。执行顺序:第一次跑完 → 全部产物
    #   mv 到不可变归档 A → 重启进程跑第二次 → 第二次产物 mv 到归档 B →
    #   归档 A copy 回正式路径。manifest 的 abstain_s1 证据链=A;
    #   duplicate 度量=A vs B 的 primary eval。
    duplicates = [
        {**_branch_job(task, "abstain", 1), "id": f"{task}_duplicate_s1",
         "kind": "duplicate",
         "log": f"{OUT_ROOT}/p0_{task}_abstain_dup_s1.log",
         "first_run_log": f"{OUT_ROOT}/p0_{task}_abstain_s1.log",
         "execution_record": f"{OUT_ROOT}/execution/p0_{task}_abstain_dup_s1.json",
         "first_run_execution_record": f"{OUT_ROOT}/execution/p0_{task}_abstain_s1.json",
         "archive_a": f"models/p0_dup_archive/{task}_A",
         "archive_b": f"models/p0_dup_archive/{task}_B",
         "note": "CLI identical to abstain_s1 (independent log); formal s1 = first run (A)"}
        for task in TASKS
    ]
    evals = []
    for job in branches:
        run_name = job["run_name"]
        for step in CKPT_STEPS:
            evals.append(_eval_job(job["task"], job["arm"], job["seed"], step,
                                   f"models/{run_name}_{step}.pt"))
    for task in TASKS:
        # duplicate 的两份 primary checkpoint(归档 A/B)各评一次。
        run_name = _run_name(f"p0_{task}_abstain", task, 1)
        for letter in ("A", "B"):
            evals.append(_eval_job(
                task, "abstain", 1, PRIMARY_STEP,
                f"models/p0_dup_archive/{task}_{letter}/{run_name}_{PRIMARY_STEP}.pt",
                tag=f"{task}_dup{letter}_s1",
            ))

    manifests = {}
    for task in TASKS:
        manifests[task] = {
            "seeds": [
                {
                    "seed": seed,
                    "lease_eval": f"{OUT_ROOT}/eval/{task}_lease_s{seed}_{PRIMARY_STEP}.json",
                    "abstain_eval": f"{OUT_ROOT}/eval/{task}_abstain_s{seed}_{PRIMARY_STEP}.json",
                    "lease_checkpoint": f"models/{_run_name(f'p0_{task}_lease', task, seed)}_{PRIMARY_STEP}.pt",
                    "abstain_checkpoint": f"models/{_run_name(f'p0_{task}_abstain', task, seed)}_{PRIMARY_STEP}.pt",
                }
                for seed in SEEDS
            ],
            "duplicate": {
                "eval_a": f"{OUT_ROOT}/eval/{task}_dupA_s1_{PRIMARY_STEP}.json",
                "eval_b": f"{OUT_ROOT}/eval/{task}_dupB_s1_{PRIMARY_STEP}.json",
                "checkpoint_a": f"models/p0_dup_archive/{task}_A/{_run_name(f'p0_{task}_abstain', task, 1)}_{PRIMARY_STEP}.pt",
                "checkpoint_b": f"models/p0_dup_archive/{task}_B/{_run_name(f'p0_{task}_abstain', task, 1)}_{PRIMARY_STEP}.pt",
                # 独立性证明(八次复核 6:两次独立重启完全可能产出逐位相同的
                # checkpoint——理想 d_dup=0——故不以 SHA 不同为条件,改由
                # 执行记录证明:execution_id/日志路径不同)。
                "execution_record_a": f"models/p0_dup_archive/{task}_A/execution_record.json",
                "execution_record_b": f"models/p0_dup_archive/{task}_B/execution_record.json",
            },
        }
    integrity_checklist = [
        "每个 anchor/branch/duplicate 作业 exit code == 0(执行器强制)",
        "anchor manifest.json 存在(执行器 expected_artifacts 强制)",
        "每分支 4 显式 checkpoint + final 存在(执行器 expected_artifacts 强制)",
        "duplicate 第一次产物归档成功后才允许第二次启动(执行器顺序强制)",
        "全部 52 个 eval JSON 存在(执行器 expected_artifacts 强制)",
        "adjudication manifest 引用的全部路径存在(裁决器入口验证强制)",
        "treatment/身份审计(裁决器内)通过后才有统计裁决(裁决器强制)",
    ]
    # 冻结输入指纹:bank yaml、source manifest、manifest 实际指向的 source
    # checkpoint 都必须进入 plan。只冻结 manifest 文本无法阻止权重文件被替换。
    # δ 在 smoke 后、正式 plan 前冻结；缺失时 plan 可供审阅，但 verify_frozen_plan
    # 会拒绝执行，必须冻结 δ 后重新生成正式 plan。
    frozen_inputs: dict = {
        "bank_yaml_sha256": {},
        "source_manifest_sha256": {},
        "source_checkpoint_sha256": {},
        "delta_sha256": {},
    }
    for task, spec in TASKS.items():
        bank_path = REPO_ROOT / spec["bank"]
        if not bank_path.is_file():
            raise FileNotFoundError(f"source bank does not exist: {bank_path}")
        frozen_inputs["bank_yaml_sha256"][task] = _file_sha256(bank_path)
        import re as _re

        manifest_refs = _re.findall(r"manifest:\s*(\S+)", bank_path.read_text())
        if not manifest_refs:
            raise ValueError(f"source bank contains no manifest references: {bank_path}")
        for match in manifest_refs:
            manifest_path = REPO_ROOT / match
            if not manifest_path.is_file():
                raise FileNotFoundError(f"source manifest does not exist: {manifest_path}")
            frozen_inputs["source_manifest_sha256"][match] = _file_sha256(manifest_path)
            manifest_data = json.loads(manifest_path.read_text())
            checkpoint_ref = manifest_data.get("checkpoint")
            if not isinstance(checkpoint_ref, str) or not checkpoint_ref:
                raise ValueError(f"source manifest has no checkpoint path: {manifest_path}")
            checkpoint_path = REPO_ROOT / checkpoint_ref
            if not checkpoint_path.is_file():
                raise FileNotFoundError(f"source checkpoint does not exist: {checkpoint_path}")
            frozen_inputs["source_checkpoint_sha256"][checkpoint_ref] = _file_sha256(
                checkpoint_path
            )
    for task, rel_path in DELTA_PATHS.items():
        delta_path = REPO_ROOT / rel_path
        frozen_inputs["delta_sha256"][task] = (
            _file_sha256(delta_path) if delta_path.is_file() else None
        )
    return {
        "utc": datetime.now(timezone.utc).isoformat(),
        "git_head": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False,
        ).stdout.strip(),
        "noise_seed_panel": {f"s{seed}": NOISE_SEED_BASE + seed for seed in SEEDS},
        "frozen_inputs": frozen_inputs,
        "anchors": anchors,
        "branches": branches,
        "duplicates": duplicates,
        "evals": evals,
        "adjudication_manifests": manifests,
        "integrity_checklist": integrity_checklist,
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_frozen_plan(plan_path: Path, expected_sha256: str) -> dict:
    """加载并验证冻结 plan(八次复核 7):文件 SHA、当前 HEAD、工作树干净、
    bank/source manifest 输入指纹逐一复核。通过才返回 plan。"""
    actual_sha = _file_sha256(plan_path)
    if actual_sha != expected_sha256:
        raise RuntimeError(f"plan sha256 mismatch: file={actual_sha}, expected={expected_sha256}")
    plan = json.loads(plan_path.read_text())
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False,
    ).stdout.strip()
    if plan.get("git_head") != head:
        raise RuntimeError(f"plan git_head={plan.get('git_head')} != current HEAD {head}")
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False,
    ).stdout.strip()
    if dirty:
        raise RuntimeError("working tree is dirty; frozen-plan execution requires a clean tree")
    frozen = plan.get("frozen_inputs") or {}
    required_fingerprint_sections = {
        "bank_yaml_sha256",
        "source_manifest_sha256",
        "source_checkpoint_sha256",
        "delta_sha256",
    }
    if not required_fingerprint_sections.issubset(frozen):
        missing = sorted(required_fingerprint_sections - set(frozen))
        raise RuntimeError(f"frozen plan is missing input fingerprint sections: {missing}")
    if set((frozen.get("bank_yaml_sha256") or {})) != set(TASKS):
        raise RuntimeError("frozen plan must fingerprint every task bank")
    for task, expected in (frozen.get("bank_yaml_sha256") or {}).items():
        actual = _file_sha256(REPO_ROOT / TASKS[task]["bank"])
        if actual != expected:
            raise RuntimeError(f"bank yaml sha mismatch for {task}")
    for rel_path, expected in (frozen.get("source_manifest_sha256") or {}).items():
        actual = _file_sha256(REPO_ROOT / rel_path)
        if actual != expected:
            raise RuntimeError(f"source manifest sha mismatch: {rel_path}")
    checkpoint_fingerprints = frozen.get("source_checkpoint_sha256") or {}
    if not checkpoint_fingerprints:
        raise RuntimeError("frozen plan contains no source checkpoint fingerprints")
    for rel_path, expected in checkpoint_fingerprints.items():
        actual = _file_sha256(REPO_ROOT / rel_path)
        if actual != expected:
            raise RuntimeError(f"source checkpoint sha mismatch: {rel_path}")
    delta_fingerprints = frozen.get("delta_sha256") or {}
    if set(delta_fingerprints) != set(DELTA_PATHS):
        raise RuntimeError("frozen plan must fingerprint both task delta files")
    for task, expected in delta_fingerprints.items():
        if not isinstance(expected, str) or len(expected) != 64:
            raise RuntimeError(
                f"delta for {task} was not frozen when this plan was built; "
                "freeze delta and regenerate the formal plan"
            )
        actual = _file_sha256(REPO_ROOT / DELTA_PATHS[task])
        if actual != expected:
            raise RuntimeError(f"delta sha mismatch for {task}")
    return plan


MAX_CONCURRENCY = 2  # E2:节点 RAM 上限,硬性双并发


SMOKE_ROOT = "logs/p0_smoke"
SMOKE_ANCHOR_STEP = 500
SMOKE_BRANCH_STOP = 700


def build_smoke_plan() -> dict:
    """throughput smoke 隔离计划(run card §9;七次复核最小修复 6):
    1 anchor 段(500 步,正式 num_envs=128 参数,足以测 sps/VRAM/RAM)
    + 1 lease 分支 200 步段 + 1 次完整 eval 面板。
    全部产物走 p0smoke_* exp-name 与 checkpoints/p0_smoke/ 路径,
    与正式 P0 的 anchor/branch/eval/manifest 路径零交集。
    无结果窥视:smoke 只读吞吐指标,不读 return/曲线。"""
    task = "crawl"
    smoke_anchor_dir = "checkpoints/p0_smoke/anchor_crawl_s1"
    anchor = _anchor_job(task, 1)
    anchor = {
        **anchor,
        "id": "smoke_anchor_crawl_s1",
        "kind": "smoke_anchor",
        "log": f"{SMOKE_ROOT}/smoke_anchor.log",
        "cli": [
            arg
            for arg in anchor["cli"]
        ],
        "expected_artifacts": [f"{smoke_anchor_dir}/manifest.json"],
    }
    # 改写 exp-name/anchor 参数为 smoke 隔离值。
    cli = anchor["cli"]
    cli[cli.index("--exp-name") + 1] = "p0smoke_anchor_crawl"
    cli[cli.index("--ptf-anchor-step") + 1] = str(SMOKE_ANCHOR_STEP)
    cli[cli.index("--ptf-anchor-dir") + 1] = smoke_anchor_dir
    cli[cli.index("--ptf-run-stop-step") + 1] = str(SMOKE_ANCHOR_STEP)

    branch = _branch_job(task, "lease", 1, exp_override="p0smoke_crawl_lease")
    bcli = branch["cli"]
    bcli[bcli.index("--ptf-anchor-resume") + 1] = smoke_anchor_dir
    bcli[bcli.index("--ptf-run-stop-step") + 1] = str(SMOKE_BRANCH_STOP)
    bcli[bcli.index("--ptf-eval-checkpoint-steps") + 1] = str(SMOKE_BRANCH_STOP)
    smoke_run_name = f"{TASKS[task]['env']}__p0smoke_crawl_lease__1"
    branch = {
        **branch,
        "id": "smoke_branch_crawl_lease_s1",
        "kind": "smoke_branch",
        "log": f"{SMOKE_ROOT}/smoke_branch.log",
        "execution_record": f"{SMOKE_ROOT}/execution/smoke_branch.json",
        "expected_artifacts": [f"models/{smoke_run_name}_{SMOKE_BRANCH_STOP}.pt"],
    }
    eval_job = _eval_job(task, "lease", 1, SMOKE_BRANCH_STOP,
                         f"models/{smoke_run_name}_{SMOKE_BRANCH_STOP}.pt",
                         tag="smoke_crawl_lease_s1")
    eval_job["out"] = f"{SMOKE_ROOT}/eval_smoke.json"
    eval_job["expected_artifacts"] = [eval_job["out"]]
    eval_job["cli"][eval_job["cli"].index("--out") + 1] = eval_job["out"]
    eval_job["log"] = f"{SMOKE_ROOT}/smoke_eval.log"
    return {
        "utc": datetime.now(timezone.utc).isoformat(),
        "git_head": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False,
        ).stdout.strip(),
        "mode": "throughput_smoke_isolated",
        "isolation": "p0smoke_* exp-names + checkpoints/p0_smoke/ + logs/p0_smoke/; 与正式 P0 路径零交集",
        "jobs": [anchor, branch, eval_job],
        "measurement": "只读 sps/VRAM/RAM;不读 return/曲线(无结果窥视)",
    }


def _group_rss_bytes(pgid: int) -> int:
    """进程组当前 RSS 总和(主进程+SubprocVecEnv worker 等全部成员)。"""
    import os

    total = 0
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            if os.getpgid(int(entry)) != pgid:
                continue
            with open(f"/proc/{entry}/statm") as handle:
                total += int(handle.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
        except (OSError, ValueError, ProcessLookupError):
            continue
    return total


def _gpu_used_mib(gpu_id: str) -> int:
    result = subprocess.run(
        ["nvidia-smi", f"--id={gpu_id}", "--query-gpu=memory.used",
         "--format=csv,noheader,nounits"],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False,
    )
    try:
        return int(result.stdout.strip().splitlines()[0])
    except (ValueError, IndexError):
        return -1


def _last_sps(log_path: Path) -> float | None:
    import re

    match = None
    try:
        for line in log_path.read_text(errors="replace").splitlines():
            found = re.findall(r"(\d+(?:\.\d+)?) sps", line)
            if found:
                match = float(found[-1])
    except OSError:
        return None
    return match


def _validate_gpu_ids(gpus: list[str]) -> None:
    if not gpus:
        raise ValueError("at least one GPU id is required")
    if len(set(gpus)) != len(gpus):
        raise ValueError(f"duplicate GPU ids are not allowed: {gpus}")
    if len(gpus) > MAX_CONCURRENCY:
        raise ValueError(
            f"at most {MAX_CONCURRENCY} GPUs allowed (E2 RAM constraint), got {len(gpus)}"
        )


def _live_group_pids(pgid: int) -> list[int]:
    """Return non-zombie members of a process group.

    The group leader can become a zombie until ``Popen.wait`` reaps it. Counting
    that zombie as live makes every cleanup wait the full grace period even when
    all executable descendants have already exited.
    """
    import os

    members: list[int] = []
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        pid = int(entry)
        try:
            if os.getpgid(pid) != pgid:
                continue
            stat = Path(f"/proc/{pid}/stat").read_text()
            state = stat[stat.rfind(")") + 2 :].split()[0]
            if state != "Z":
                members.append(pid)
        except (OSError, ValueError, ProcessLookupError, IndexError):
            continue
    return members


def _kill_group(pgid: int) -> None:
    """SIGTERM 整个进程组;宽限后 SIGKILL(八次复核阻塞 2:只 terminate 顶层
    Popen 会留下 SubprocVecEnv 的环境孙进程)。"""
    import os
    import signal
    import time as _time

    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = _time.time() + 30
    while _time.time() < deadline:
        if not _live_group_pids(pgid):
            return
        _time.sleep(0.5)
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    kill_deadline = _time.time() + 5
    while _time.time() < kill_deadline and _live_group_pids(pgid):
        _time.sleep(0.1)


def _write_execution_record(entry: dict) -> None:
    """Persist evidence from the process that actually produced the artifacts."""
    job = entry["job"]
    record_ref = job.get("execution_record")
    if not record_ref:
        return
    record_path = Path(record_ref)
    if not record_path.is_absolute():
        record_path = REPO_ROOT / record_path
    if record_path.exists():
        raise FileExistsError(f"execution record already exists: {record_path}")
    artifacts: dict[str, str] = {}
    for artifact in job.get("expected_artifacts", []):
        artifact_path = Path(artifact)
        if not artifact_path.is_absolute():
            artifact_path = REPO_ROOT / artifact_path
        artifacts[str(artifact)] = _file_sha256(artifact_path)
    cli_json = json.dumps(job["cli"], ensure_ascii=False, separators=(",", ":"))
    git_head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False,
    ).stdout.strip()
    record = {
        "schema_version": 1,
        "execution_id": entry["execution_id"],
        "job_id": job["id"],
        "task": job.get("task"),
        "arm": job.get("arm"),
        "seed": job.get("seed"),
        "pid": entry["proc"].pid,
        "gpu": entry["gpu"],
        "cli": job["cli"],
        "cli_sha256": hashlib.sha256(cli_json.encode()).hexdigest(),
        "start_utc": entry["start_utc"],
        "end_utc": datetime.now(timezone.utc).isoformat(),
        "exit_code": entry["proc"].returncode,
        "git_head": git_head,
        "log": str(job.get("log", entry["log_path"])),
        "log_sha256": _file_sha256(entry["log_path"]),
        "artifact_sha256": artifacts,
    }
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text(json.dumps(record, indent=2) + "\n")


def _run_queue(jobs: list[dict], gpus: list[str], env_extra: dict,
               metrics: dict | None = None) -> None:
    """双并发队列:
    - 显式空闲 GPU 池(先完成先归还);--gpus 拒绝重复 ID;
    - 每作业独立进程组(start_new_session);失败/中断时 killpg 清理整棵
      进程树(含 SubprocVecEnv 环境孙进程)后才抛出;
    - metrics 非 None 时按 job id 记录 wall-clock/SPS/峰值进程组 RSS/
      峰值 GPU 显存(该卡总 used,共卡任务会计入,报告中注明)。"""
    import os
    import time as _time

    _validate_gpu_ids(gpus)
    idle_gpus = list(gpus)
    active: list[dict] = []
    queue = list(jobs)
    try:
        while queue or active:
            while queue and idle_gpus:
                job = queue.pop(0)
                gpu = idle_gpus.pop(0)
                log_path = REPO_ROOT / job.get("log", f"{OUT_ROOT}/{job['id']}.log")
                record_ref = job.get("execution_record")
                if record_ref:
                    record_path = Path(record_ref)
                    if not record_path.is_absolute():
                        record_path = REPO_ROOT / record_path
                    if record_path.exists():
                        raise FileExistsError(f"execution record already exists: {record_path}")
                # Every planned output is one-shot evidence. Reject stale files before
                # launch so an exit-0 process cannot accidentally authenticate artifacts
                # left by an earlier failed/partial attempt.
                stale_artifacts = []
                for artifact in job.get("expected_artifacts", []):
                    artifact_path = Path(artifact)
                    if not artifact_path.is_absolute():
                        artifact_path = REPO_ROOT / artifact_path
                    if artifact_path.exists():
                        stale_artifacts.append(str(artifact_path))
                if stale_artifacts:
                    raise FileExistsError(
                        f"job {job['id']} has pre-existing expected artifacts: {stale_artifacts}"
                    )
                log_path.parent.mkdir(parents=True, exist_ok=True)
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, PYTHONUNBUFFERED="1",
                           PYTHONPATH=str(REPO_ROOT), **env_extra)
                handle = log_path.open("w")
                try:
                    proc = subprocess.Popen(job["cli"], cwd=REPO_ROOT, env=env,
                                            stdout=handle, stderr=subprocess.STDOUT,
                                            start_new_session=True)
                except BaseException:
                    handle.close()
                    idle_gpus.append(gpu)
                    raise
                import uuid as _uuid

                active.append({
                    "proc": proc, "job": job, "gpu": gpu, "handle": handle,
                    "log_path": log_path, "start": _time.time(),
                    "start_utc": datetime.now(timezone.utc).isoformat(),
                    "execution_id": _uuid.uuid4().hex,
                    "peak_rss": 0, "peak_gpu_mib": 0,
                })
                print(f"[launch] {job['id']} on GPU {gpu} (pgid {proc.pid}) -> {log_path}")
            # 轮询任意已完成作业(先完成者先归还 GPU);顺带采样资源峰值。
            finished_index = None
            while finished_index is None:
                for index, entry in enumerate(active):
                    if entry["proc"].poll() is not None:
                        finished_index = index
                        break
                if finished_index is None:
                    if metrics is not None:
                        for entry in active:
                            entry["peak_rss"] = max(
                                entry["peak_rss"], _group_rss_bytes(entry["proc"].pid)
                            )
                            entry["peak_gpu_mib"] = max(
                                entry["peak_gpu_mib"], _gpu_used_mib(entry["gpu"])
                            )
                    _time.sleep(2)
            entry = active.pop(finished_index)
            proc, job, gpu = entry["proc"], entry["job"], entry["gpu"]
            entry["handle"].close()
            # ``poll`` has reaped the leader, but a failed/successful leader may
            # still have left descendants in its process group. Clean this
            # finished job's own group before inspecting its result; the outer
            # exception handler only sees jobs that remain in ``active``.
            _kill_group(proc.pid)
            idle_gpus.append(gpu)
            if metrics is not None:
                metrics[job["id"]] = {
                    "wall_seconds": round(_time.time() - entry["start"], 1),
                    "sps": _last_sps(entry["log_path"]),
                    "peak_group_rss_gib": round(entry["peak_rss"] / 2**30, 2),
                    "peak_gpu_mib_total_on_card": entry["peak_gpu_mib"],
                    "gpu": gpu,
                    "exit_code": proc.returncode,
                }
            if proc.returncode != 0:
                raise RuntimeError(f"job {job['id']} failed with exit code {proc.returncode}")
            for artifact in job.get("expected_artifacts", []):
                if not (REPO_ROOT / artifact).exists():
                    raise RuntimeError(f"job {job['id']} missing expected artifact: {artifact}")
            _write_execution_record(entry)
            print(f"[done] {job['id']}")
    except BaseException:
        # 失败/KeyboardInterrupt:killpg 清理全部活动进程组,不留训练孤儿。
        for entry in active:
            if entry["proc"].poll() is None:
                print(f"[terminate-group] {entry['job']['id']} (pgid {entry['proc'].pid})")
            _kill_group(entry["proc"].pid)
            entry["proc"].wait()
            entry["handle"].close()
        raise


def _run_duplicates(duplicates: list[dict], gpus: list[str],
                    run_queue=None, repo_root: Path | None = None) -> None:
    """duplicate 归档流程(八次复核事务化;正式 s1 = 第一次运行 A):
    - **preflight 先行**:先验证全部前置(归档不存在+正式产物齐全),任何
      检查失败都不留任何目录/移动痕迹;
    - staging 目录暂存,成功后原子 rename 提交为 A/B(半成品不会占用
      不可变归档名,失败重试不被 'archive exists' 卡死);
    - 第二次运行失败时 finally 把 A 暂存产物**恢复回正式路径**并清理
      staging,正式证据链不缺失。
    run_queue/repo_root 可注入(单测用)。"""
    import shutil

    run_queue = run_queue or _run_queue
    root = repo_root or REPO_ROOT
    for dup in duplicates:
        task = dup["task"]
        run_name = _run_name(f"p0_{task}_abstain", task, 1)
        products = [f"models/{run_name}_{step}.pt" for step in CKPT_STEPS] + [
            f"models/{run_name}_final.pt"
        ]
        archive_a = root / dup["archive_a"]
        archive_b = root / dup["archive_b"]
        first_record = root / dup["first_run_execution_record"]
        second_record = root / dup["execution_record"]
        # ---- preflight(不产生任何副作用) ----
        for archive in (archive_a, archive_b):
            if archive.exists():
                raise RuntimeError(f"immutable duplicate archive already exists: {archive}")
        for product in products:
            if not (root / product).exists():
                raise RuntimeError(f"duplicate precondition missing: {root / product}")
        if not first_record.is_file():
            raise RuntimeError(f"duplicate first-run execution record missing: {first_record}")
        if second_record.exists():
            raise RuntimeError(f"duplicate second-run execution record already exists: {second_record}")
        # ---- 事务体 ----
        staging_a = archive_a.with_name(archive_a.name + ".staging")
        staging_b = archive_b.with_name(archive_b.name + ".staging")
        for staging in (staging_a, staging_b):
            if staging.exists():
                shutil.rmtree(staging)
        staging_a.mkdir(parents=True)
        moved_to_a: list[str] = []
        committed = False
        renamed_a = False
        renamed_b = False
        try:
            for product in products:
                source = root / product
                shutil.move(str(source), staging_a / source.name)
                moved_to_a.append(product)
            shutil.copy2(first_record, staging_a / "execution_record.json")
            run_queue([dup], gpus, {})
            if not second_record.is_file():
                raise RuntimeError(
                    f"duplicate second-run execution record missing: {second_record}"
                )
            staging_b.mkdir(parents=True)
            for product in products:
                source = root / product
                if not source.exists():
                    raise RuntimeError(f"duplicate second run missing product: {source}")
                shutil.move(str(source), staging_b / source.name)
            shutil.copy2(second_record, staging_b / "execution_record.json")
            # A 内容 copy 回正式路径(正式 s1=A),然后原子提交两个归档。
            for product in products:
                name = Path(product).name
                shutil.copy2(staging_a / name, root / product)
            staging_a.rename(archive_a)
            renamed_a = True
            staging_b.rename(archive_b)
            renamed_b = True
            committed = True
        finally:
            if not committed:
                # 回滚:第二次运行可能已在正式路径写出部分 B。对所有曾移入 A
                # 的产品先删除任意 formal B，再无条件恢复 A，杜绝 B/B/A/A/A。
                a_source = archive_a if renamed_a else staging_a
                for product in moved_to_a:
                    name = Path(product).name
                    formal = root / product
                    if formal.exists():
                        formal.unlink()
                    if (a_source / name).exists():
                        shutil.copy2(a_source / name, formal)
                # 两个 rename 不是联合原子操作；若 A 已提交而 B 提交失败，移除
                # 本事务的半提交归档，确保修复后可重试。
                if renamed_a and archive_a.exists():
                    shutil.rmtree(archive_a)
                if renamed_b and archive_b.exists():
                    shutil.rmtree(archive_b)
                # Preflight proved this path did not exist before the transaction, so
                # any record now present belongs to the failed B attempt and must not
                # block a clean retry or masquerade as evidence for it.
                if second_record.exists():
                    second_record.unlink()
                for staging in (staging_a, staging_b):
                    if staging.exists():
                        shutil.rmtree(staging)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", default=True)
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--acknowledge-authorization", action="store_true",
                        help="操作者确认已获得 PI 执行批准(与 --execute 联用)")
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--stage", choices=["anchors", "branches", "evals", "all"], default="all")
    parser.add_argument("--smoke", action="store_true",
                        help="throughput smoke 隔离计划(1 anchor 500 步+1 分支 200 步+1 eval 面板;与正式 P0 路径零交集)")
    # 冻结 plan 执行(八次复核 7):正式执行必须加载先前冻结的 plan 文件并
    # 通过全部指纹核对,而不是执行时重新 build(重新 build 的 SHA 因 UTC
    # 字段必然变化,冻结失去意义)。
    parser.add_argument("--execute-plan", default=None,
                        help="冻结 plan JSON 路径(与 --expected-plan-sha256 联用)")
    parser.add_argument("--expected-plan-sha256", default=None)
    args = parser.parse_args()

    if args.execute_plan:
        if not args.expected_plan_sha256:
            raise SystemExit("--execute-plan requires --expected-plan-sha256")
        if not args.acknowledge_authorization:
            raise SystemExit("frozen-plan execution requires --acknowledge-authorization")
        plan = verify_frozen_plan(Path(args.execute_plan), args.expected_plan_sha256)
        print(f"[frozen-plan] verified: sha={args.expected_plan_sha256[:12]}… "
              f"head={plan['git_head'][:8]} clean-tree ok, input fingerprints ok")
        gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
        if args.stage in ("anchors", "all"):
            _run_queue(plan["anchors"], gpus, {})
        if args.stage in ("branches", "all"):
            _run_queue(plan["branches"], gpus, {})
            _run_duplicates(plan["duplicates"], gpus)
        if args.stage in ("evals", "all"):
            _run_queue(plan["evals"], gpus, {})
            for task, manifest in plan["adjudication_manifests"].items():
                manifest_path = REPO_ROOT / OUT_ROOT / f"adjudication_manifest_{task}.json"
                manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
                print(f"[manifest] {manifest_path}")
        return

    if args.smoke:
        plan = build_smoke_plan()
        plan_dir = REPO_ROOT / SMOKE_ROOT
        plan_dir.mkdir(parents=True, exist_ok=True)
        plan_path = plan_dir / f"smoke_plan_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
        plan_path.write_text(json.dumps(plan, indent=2) + "\n")
        plan_sha = hashlib.sha256(plan_path.read_bytes()).hexdigest()
        print(f"[smoke-plan] {len(plan['jobs'])} jobs -> {plan_path}\n[smoke-plan] sha256={plan_sha}")
        if not args.execute:
            for job in plan["jobs"]:
                print(f"\n# {job['id']}\n" + " ".join(job["cli"]))
            print("\n[dry-run] no jobs were executed.")
            return
        if not args.acknowledge_authorization:
            raise SystemExit("--execute requires --acknowledge-authorization")
        gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
        # 先验证完整用户输入，再取单卡执行依赖链；否则 --gpus 0,0 会在
        # gpus[:1] 后绕过重复 ID 检查。
        _validate_gpu_ids(gpus)
        # 三阶段严格顺序(八次复核阻塞 1):anchor→branch→eval 存在硬依赖
        # (branch 需要 anchor bundle,eval 需要 branch checkpoint),绝不并行;
        # 每阶段产物由 expected_artifacts 强制验证后才进入下一阶段。
        metrics: dict = {}
        for stage_job in plan["jobs"]:
            _run_queue([stage_job], gpus[:1], {}, metrics=metrics)
        # smoke report(run card §9):SPS/峰值 VRAM/峰值进程组 RSS/wall-clock
        # + 依实测外推正式预算。共卡说明:peak_gpu_mib 是该卡总 used,
        # 其他任务共卡时数值偏大,应在空闲卡上运行。
        anchor_metrics = metrics.get("smoke_anchor_crawl_s1") or {}
        branch_metrics = metrics.get("smoke_branch_crawl_lease_s1") or {}
        eval_metrics = metrics.get("eval_smoke_crawl_lease_s1_700") or {}
        anchor_sps = anchor_metrics.get("sps")
        branch_sps = branch_metrics.get("sps")
        extrapolation = None
        if anchor_sps and branch_sps:
            # 将一次性启动/加载/保存开销与稳态吞吐拆开。只用 steps/SPS 会在
            # 6+14 个短作业上系统性漏掉重复启动成本。
            anchor_wall = float(anchor_metrics.get("wall_seconds") or 0.0)
            branch_wall = float(branch_metrics.get("wall_seconds") or 0.0)
            anchor_overhead = max(0.0, anchor_wall - SMOKE_ANCHOR_STEP / anchor_sps)
            branch_steps = SMOKE_BRANCH_STOP - SMOKE_ANCHOR_STEP
            branch_overhead = max(0.0, branch_wall - branch_steps / branch_sps)
            anchor_hours = (anchor_overhead + 10000 / anchor_sps) / 3600 * 6
            branch_hours = (branch_overhead + 3000 / branch_sps) / 3600 * 14
            eval_hours = eval_metrics.get("wall_seconds", 0) / 3600 * 52
            extrapolation = {
                "anchor_gpu_hours_6x10000": round(anchor_hours, 2),
                "branch_gpu_hours_14x3000": round(branch_hours, 2),
                "eval_gpu_hours_52x": round(eval_hours, 2),
                "total_gpu_hours": round(anchor_hours + branch_hours + eval_hours, 2),
                "measured_anchor_startup_seconds": round(anchor_overhead, 1),
                "measured_branch_startup_seconds": round(branch_overhead, 1),
                "method": "per_job_startup_overhead + target_steps / steady_sps",
                "budget_target_hours": 24,
                "rebudget_threshold_hours": 48,
            }
        smoke_report = {
            "plan_sha256": plan_sha,
            "git_head": plan["git_head"],
            "utc": datetime.now(timezone.utc).isoformat(),
            "stage_metrics": metrics,
            "extrapolation": extrapolation,
            "caveat": "peak_gpu_mib_total_on_card 为该卡总占用;共卡任务会计入",
        }
        report_path = plan_dir / f"smoke_report_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
        report_path.write_text(json.dumps(smoke_report, indent=2) + "\n")
        print(f"[smoke-report] {report_path}")
        if extrapolation:
            print(f"[smoke-report] extrapolated total: {extrapolation['total_gpu_hours']} GPU-hours "
                  f"(target <=24, rebudget >48)")
        return

    plan = build_plan()
    plan_dir = REPO_ROOT / OUT_ROOT
    plan_dir.mkdir(parents=True, exist_ok=True)
    plan_path = plan_dir / f"plan_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    plan_path.write_text(json.dumps(plan, indent=2) + "\n")
    plan_sha = hashlib.sha256(plan_path.read_bytes()).hexdigest()

    counts = (f"{len(plan['anchors'])} anchors + {len(plan['branches'])} branches "
              f"+ {len(plan['duplicates'])} duplicates + {len(plan['evals'])} evals")
    print(f"[plan] {counts}\n[plan] written to {plan_path}\n[plan] sha256={plan_sha}")

    if not args.execute:
        for section in ("anchors", "branches", "duplicates"):
            for job in plan[section]:
                print(f"\n# {job['id']}\n" + " ".join(job["cli"]))
        print(f"\n# evals ({len(plan['evals'])} jobs, 详见 plan.json)")
        print("\n# integrity checklist:")
        for item in plan["integrity_checklist"]:
            print(f"  - {item}")
        print("\n[dry-run] no jobs were executed.")
        return

    # 正式矩阵不允许即时 build 执行(八次复核 7):执行时重新 build 的 plan
    # 从未被冻结复核过。正确路径=先 --dry-run 冻结 plan,复核后用
    # --execute-plan <plan.json> --expected-plan-sha256 <sha> 执行。
    raise SystemExit(
        "direct --execute of a freshly built formal plan is not allowed; "
        "freeze a plan with --dry-run first, then run with "
        "--execute-plan <plan.json> --expected-plan-sha256 <sha>"
    )


if __name__ == "__main__":
    main()
