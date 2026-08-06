"""P0 anchor-resume 等价性测试(run card v2.1.2 附录 A.3,GPU 实机部分)。

覆盖 A.3 清单中需要真实 train_ptf 路径的三项(其余四项见
tests/test_anchor_core_resume.py 与 resume 块内断言):

- 测试 4(control-arm 等价,双层判定——2026-07-17 实测诊断后修订,
  详见 run card A.3 的诊断记录):
  * **语义层(CPU,逐位,决定性)**:CPU 上同进程顺序跑分支 A(挂 crawl bank
    + admission_mode=none)与分支 B(空 bank 纯 student),全部核心组件
    checksum 逐位一致 + 硬断言(RNG 终态/option optimizer 无状态/
    execution_counts source 位严格 0)。CPU 无 cuBLAS 执行历史噪声,
    逐位一致 ⇒ 两代码路径语义等价;
  * **执行层(GPU,幅度)**:同进程顺序跑 A、B、B'(B'=B 完全重复),
    每组件 max-abs 须满足 |A-B| ≤ 3×|B-B'|(纯执行噪声地板)。
    诊断记录:GPU 逐位对照对本 stack 不可达——B-B 对照复现了 A-B 的
    分叉模式(qnet 大 GEMM 的 cublasLt 执行历史依赖,fp32 与
    CUBLAS_WORKSPACE_CONFIG 均不免疫)。
- 测试 5(duplicate 语义):完全相同参数与 seed,重启进程 ×2 跑分支 B,
  报告核心参数分叉幅度(d_dup 生成机制预演;报告值,不作零断言——E10)。
- 测试 6(option 族不参与):分支 A 的 option/beta optimizer 必须无任何
  Adam step 状态(bootstrap_only+admission none 下 option 族不参与行为与 loss)。

用法(FastTD3 conda env;GPU 选择用 CUDA_VISIBLE_DEVICES 环境变量):
  CUDA_VISIBLE_DEVICES=1 python scripts/p0_equivalence_tests.py --phase all
产物:每次运行创建不可变目录 logs/p0_equivalence/run_<UTC>/,内含 report.json
(含 git SHA/CLI/device/stop/anchor checksum manifest)。旧运行目录不被覆盖。
不看任何训练 outcome(return/曲线)——本脚本只核对执行等价性。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

ENV_NAME = "h1hand-crawl-v0"
SEED = 1
NUM_ENVS = 16
BUFFER_SIZE = 2048
ANCHOR_STEP = 120
BRANCH_STOP = 320  # 可被 --stop 覆盖(分叉步二分诊断用)
NOISE_SEED = 4242
CRAWL_BANK = "configs/source_banks/h1hand_loco_wfix_crawl.yaml"
WORK_DIR = REPO_ROOT / "logs" / "p0_equivalence"
ANCHOR_DIR = WORK_DIR / "toy_anchor"

# 核心 learner 组件在 final.pt 中的键(与 run card A.1 白名单对应;
# final.pt 不含 optimizer/scheduler,它们的等价性由单测+resume 断言覆盖)。
CORE_KEYS = (
    "actor_state_dict",
    "qnet_state_dict",
    "qnet_target_state_dict",
    "obs_normalizer_state",
    "critic_obs_normalizer_state",
)


def _common_cli(exp_name: str) -> list[str]:
    return [
        sys.executable,
        "fasttd3_ptf/official_fasttd3_ptf/train_ptf.py",
        "--env-name", ENV_NAME,
        "--exp-name", exp_name,
        "--seed", str(SEED),
        "--num-envs", str(NUM_ENVS),
        "--num-eval-envs", str(NUM_ENVS),
        "--buffer-size", str(BUFFER_SIZE),
        "--learning-starts", "10",
        "--eval-interval", "0",
        "--render-interval", "0",
        "--save-interval", "0",
        "--no-use-wandb",
        # 等价性测试固定 fp32:噪声地板比 bf16 更小,幅度判据更严格;
        # 语义层(CPU 逐位)与执行层(幅度比值)的结论均不依赖精度模式。
        # 正式 P0 分支按冻结参数表继承 anchor 的 AMP 配置,与本测试无关。
        "--no-amp",
    ]


def _branch_cli(exp_name: str, *, with_bank: bool) -> list[str]:
    cli = _common_cli(exp_name) + [
        "--ptf-anchor-resume", str(ANCHOR_DIR),
        "--ptf-run-stop-step", str(BRANCH_STOP),
        "--ptf-resume-noise-seed", str(NOISE_SEED),
    ]
    if with_bank:
        cli += [
            "--ptf-source-bank", CRAWL_BANK,
            "--ptf-mcg",
            "--ptf-mcg-warmup-mode", "admission_bootstrap",
            "--ptf-mcg-ablation", "bootstrap_only",
            "--ptf-mcg-warmup-steps", "100000",
            "--ptf-admission-mode", "none",
            "--ptf-admission-student-logit", "16.6823567039",
            "--ptf-admission-replay-handoff", "physical_after_authority",
            "--ptf-admission-replay-recency-half-life", "0",
            "--ptf-admission-replay-uniform-mix", "1.0",
            "--ptf-admission-replay-priority-alpha", "0",
            "--ptf-admission-expected-source-mass", "0.0",
        ]
    return cli


def _final_path(exp_name: str) -> Path:
    return REPO_ROOT / "models" / f"{ENV_NAME}__{exp_name}__{SEED}_final.pt"


def _tensor_digest(state: dict) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        value = state[key]
        digest.update(key.encode())
        if isinstance(value, torch.Tensor):
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        else:
            digest.update(repr(value).encode())
    return digest.hexdigest()


def _load_final(exp_name: str) -> dict:
    path = _final_path(exp_name)
    if not path.exists():
        raise FileNotFoundError(f"missing final checkpoint: {path}")
    return torch.load(path, map_location="cpu", weights_only=False)


def _run_inprocess(cli: list[str]) -> str:
    """同进程调 train_ptf.main()(测试 4 的硬要求,E10:跨进程无逐位可比性),
    返回运行结束时刻的全局 RNG 终态 digest。"""
    from fasttd3_ptf.official_fasttd3_ptf import train_ptf

    old_argv = sys.argv
    sys.argv = [cli[1], *cli[2:]]
    try:
        train_ptf.main()
    finally:
        sys.argv = old_argv
    digest = hashlib.sha256()
    digest.update(torch.get_rng_state().numpy().tobytes())
    if torch.cuda.is_available():
        digest.update(torch.cuda.get_rng_state().cpu().numpy().tobytes())
    return digest.hexdigest()


def _run_subprocess(cli: list[str]) -> None:
    existing = os.environ.get("PYTHONPATH", "")
    pythonpath = f"{REPO_ROOT}{os.pathsep}{existing}" if existing else str(REPO_ROOT)
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONPATH=pythonpath)
    result = subprocess.run(cli, cwd=REPO_ROOT, env=env)
    if result.returncode != 0:
        raise RuntimeError(f"subprocess failed ({result.returncode}): {' '.join(cli[2:6])}")


def phase_anchor(report: dict) -> None:
    """toy anchor 段:空 bank scratch 训练 120 步并保存 bundle(独立进程,
    与真实 P0 的 anchor 生产方式一致)。"""
    if ANCHOR_DIR.exists():
        shutil.rmtree(ANCHOR_DIR)
    cli = _common_cli("p0eq_anchor") + [
        "--ptf-anchor-step", str(ANCHOR_STEP),
        "--ptf-anchor-dir", str(ANCHOR_DIR),
        "--ptf-run-stop-step", str(ANCHOR_STEP),
        # crawl 目标组数=3(DEFAULT_GROUPS);显式声明以固定优先方案语义。
        "--ptf-anchor-provenance-groups", "3",
    ]
    _run_subprocess(cli)
    manifest = json.loads((ANCHOR_DIR / "manifest.json").read_text())
    assert manifest["completed_vector_steps"] == ANCHOR_STEP
    report["anchor"] = {"bundle": str(ANCHOR_DIR), "step": ANCHOR_STEP, "status": "PASS"}
    print(f"[anchor] saved toy anchor at step {ANCHOR_STEP}")


def _hard_assertions(final_a: dict) -> tuple[dict, bool, bool]:
    """测试 6 + treatment 双零断言(两层判定共用)。"""
    option_states = {
        name: len(final_a[name].get("state", {}))
        for name in ("option_optimizer_state_dict", "beta_optimizer_state_dict")
        if name in final_a
    }
    option_untouched = all(count == 0 for count in option_states.values())
    audit = final_a.get("admission_audit") or {}
    execution_counts = audit.get("execution_counts") or []
    source_exec = sum(int(x) for x in execution_counts[:-1]) if execution_counts else None
    source_exec_zero = source_exec == 0 if source_exec is not None else False
    return option_states, option_untouched, source_exec_zero


def _max_abs(state_a: dict, state_b: dict) -> float:
    out = 0.0
    for name, value in state_a.items():
        other = state_b[name]
        if isinstance(value, torch.Tensor) and value.is_floating_point():
            out = max(out, float((value.float() - other.float()).abs().max()))
    return out


def phase_semantic_cpu(report: dict) -> None:
    """测试 4 语义层:CPU 同进程 A-B 逐位对照(决定性;CPU 无执行历史噪声)。
    小 batch 加速——语义证明只依赖代码路径分支走向,不依赖 batch 尺寸
    (admission/replay/option 逻辑与 batch 无关;32768 batch 在 CPU 上
    一条链要 ~2h,1024 缩到分钟级)。"""
    cpu_batch = ["--batch-size", "1024"]
    cpu_anchor = WORK_DIR / "toy_anchor_cpu"
    if not cpu_anchor.exists():
        cli = _common_cli("p0eq_cpu_anchor") + cpu_batch + [
            "--ptf-anchor-step", str(ANCHOR_STEP),
            "--ptf-anchor-dir", str(cpu_anchor),
            "--ptf-run-stop-step", str(ANCHOR_STEP),
            "--ptf-anchor-provenance-groups", "3",
            "--no-cuda",
        ]
        _run_subprocess(cli)
    rngs, finals = [], []
    for exp, with_bank in (("p0eq_cpu_a", True), ("p0eq_cpu_b", False)):
        _final_path(exp).unlink(missing_ok=True)
        cli = [x if x != str(ANCHOR_DIR) else str(cpu_anchor) for x in _branch_cli(exp, with_bank=with_bank)]
        cli += cpu_batch
        cli.append("--no-cuda")
        rngs.append(_run_inprocess(cli))
        finals.append(_load_final(exp))
    mismatches = [
        key for key in CORE_KEYS
        if _tensor_digest(finals[0][key]) != _tensor_digest(finals[1][key])
    ]
    rng_match = rngs[0] == rngs[1]
    option_states, option_untouched, source_exec_zero = _hard_assertions(finals[0])
    status = "PASS" if (not mismatches and rng_match and option_untouched and source_exec_zero) else "FAIL"
    report["control_arm_semantic_cpu"] = {
        "core_param_mismatches": mismatches,
        "rng_terminal_match": rng_match,
        "option_optimizer_state_sizes": option_states,
        "status": status,
    }
    print(f"[semantic-cpu] {status} (mismatches={mismatches or 'none'}, rng={rng_match}, "
          f"option_untouched={option_untouched}, source_exec_zero={source_exec_zero})")
    if status == "FAIL":
        raise SystemExit("CPU semantic equivalence FAILED — genuine semantic bug, diagnose")


def phase_branches(report: dict) -> None:
    """测试 4 执行层:GPU 同进程顺序跑 A、B、B'(B'=B 完全重复),
    幅度判据 |A-B| ≤ 3×|B-B'| + 硬断言(RNG 终态/option/双零)。
    同时验证显式 checkpoint 的 completed-step 语义(off-by-one 修复):
    列表含中间步与 run_stop_step 本身,全部文件必须生成。"""
    ckpt_steps = [ANCHOR_STEP + 30, (ANCHOR_STEP + BRANCH_STOP) // 2, BRANCH_STOP]
    ckpt_cli = ["--ptf-eval-checkpoint-steps", ",".join(str(x) for x in ckpt_steps)]
    runs = [("p0eq_branch_bank_none", True), ("p0eq_branch_empty", False), ("p0eq_branch_empty2", False)]
    rngs, finals = {}, {}
    for exp, with_bank in runs:
        _final_path(exp).unlink(missing_ok=True)
        for step in ckpt_steps:
            _final_path(exp).with_name(f"{ENV_NAME}__{exp}__{SEED}_{step}.pt").unlink(missing_ok=True)
        rngs[exp] = _run_inprocess(_branch_cli(exp, with_bank=with_bank) + ckpt_cli)
        finals[exp] = _load_final(exp)
        assert finals[exp]["global_step"] == BRANCH_STOP

    missing_ckpts = []
    for exp, _ in runs:
        for step in ckpt_steps:
            path = _final_path(exp).with_name(f"{ENV_NAME}__{exp}__{SEED}_{step}.pt")
            if not path.exists():
                missing_ckpts.append(str(path.name))
            else:
                saved = torch.load(path, map_location="cpu", weights_only=False)
                assert saved["global_step"] == step, f"{path.name}: step={saved['global_step']}"
    if missing_ckpts:
        raise SystemExit(f"explicit eval checkpoints missing: {missing_ckpts}")

    magnitude = {}
    magnitude_ok = True
    for key in CORE_KEYS:
        ab = _max_abs(finals["p0eq_branch_bank_none"][key], finals["p0eq_branch_empty"][key])
        bb = _max_abs(finals["p0eq_branch_empty"][key], finals["p0eq_branch_empty2"][key])
        ratio = ab / bb if bb > 0 else (float("inf") if ab > 0 else 1.0)
        magnitude[key] = {"ab": ab, "bb_floor": bb, "ratio": ratio}
        if ratio > 3.0:
            magnitude_ok = False

    rng_match = rngs["p0eq_branch_bank_none"] == rngs["p0eq_branch_empty"]
    option_states, option_untouched, source_exec_zero = _hard_assertions(
        finals["p0eq_branch_bank_none"]
    )
    status = "PASS" if (magnitude_ok and rng_match and option_untouched and source_exec_zero) else "FAIL"
    report["control_arm_execution_gpu"] = {
        "magnitude": magnitude,
        "rng_terminal_match": rng_match,
        "option_optimizer_state_sizes": option_states,
        "eval_checkpoints_verified": ckpt_steps,
        "status": status,
    }
    print(f"[branches-gpu] {status} (magnitude_ok={magnitude_ok}, rng={rng_match}, "
          f"option_untouched={option_untouched}, source_exec_zero={source_exec_zero})")
    for key, entry in magnitude.items():
        print(f"  {key}: |A-B|={entry['ab']:.3e} floor={entry['bb_floor']:.3e} ratio={entry['ratio']:.2f}")
    if status == "FAIL":
        raise SystemExit("GPU execution-layer equivalence FAILED — diagnose before any smoke run")


def phase_duplicate(report: dict) -> None:
    """测试 5:完全同参数同 seed,重启进程 ×2 跑分支 B,报告分叉幅度(不断言零)。"""
    finals = []
    for run_idx in (1, 2):
        exp = "p0eq_duplicate"
        _final_path(exp).unlink(missing_ok=True)
        _run_subprocess(_branch_cli(exp, with_bank=False))
        state = _load_final(exp)
        finals.append(state)
        kept = _final_path(exp).with_name(_final_path(exp).stem + f"_run{run_idx}.pt")
        shutil.move(_final_path(exp), kept)
    divergence = {}
    for key in CORE_KEYS:
        max_abs = 0.0
        for name, value in finals[0][key].items():
            other = finals[1][key][name]
            if isinstance(value, torch.Tensor) and value.is_floating_point():
                max_abs = max(max_abs, float((value - other).abs().max()))
        divergence[key] = max_abs
    report["duplicate"] = {
        "max_abs_divergence_per_component": divergence,
        "note": "repeatability discrepancy floor (d_dup mechanism preview); reported, not asserted",
        "status": "REPORTED",
    }
    print(f"[duplicate] max-abs divergence per component: {divergence}")


def _run_manifest(argv: list[str]) -> dict:
    """不可变审计 manifest:git SHA/CLI/device/anchor checksum/时间。"""
    def _git(*cmd: str) -> str:
        result = subprocess.run(
            ["git", *cmd], cwd=REPO_ROOT, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"

    anchor_checksums = {}
    for name, path in (("gpu", ANCHOR_DIR), ("cpu", WORK_DIR / "toy_anchor_cpu")):
        checksum_file = path / "checksums.json"
        if checksum_file.exists():
            anchor_checksums[name] = json.loads(checksum_file.read_text())
    return {
        "utc": datetime.now(timezone.utc).isoformat(),
        "git_head": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "cli": argv,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch_version": torch.__version__,
        "branch_stop": BRANCH_STOP,
        "anchor_step": ANCHOR_STEP,
        "env_name": ENV_NAME,
        "num_envs": NUM_ENVS,
        "anchor_checksums": anchor_checksums,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["anchor", "semantic", "branches", "duplicate", "all"], default="all")
    parser.add_argument("--stop", type=int, default=None, help="覆盖分支 run_stop_step(二分诊断)")
    args = parser.parse_args()
    if args.stop is not None:
        global BRANCH_STOP
        BRANCH_STOP = int(args.stop)

    # 不可变审计目录:每次运行独立,禁止覆盖合并(五次复核审计缺口 2——
    # 旧 report.json 跨 phase 合并会把不同代码/stop/设备的结果混在一起)。
    run_dir = WORK_DIR / f"run_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    run_dir.mkdir(parents=True, exist_ok=False)
    report: dict = {}

    if args.phase in ("anchor", "all"):
        phase_anchor(report)
    if args.phase in ("semantic", "all"):
        phase_semantic_cpu(report)
    if args.phase in ("branches", "all"):
        phase_branches(report)
    if args.phase in ("duplicate", "all"):
        phase_duplicate(report)

    # manifest 在全部 phase 执行后采集(六次复核高优 3):否则 --phase all
    # 会先记录旧 anchor 的 checksum,随后 phase_anchor 删除重建 anchor,
    # 报告里的 checksum 就是过时值。
    report["manifest"] = _run_manifest(sys.argv)
    out_path = run_dir / "report.json"
    out_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"report written to {out_path}")


if __name__ == "__main__":
    main()
