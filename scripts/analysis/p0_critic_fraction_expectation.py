"""P0 critic 采样占比冷启动仿真 v3（期望法，冻结配置输入）。

post-hoc engineering sensitivity analysis 的机制验证脚本（十二次复核修复版）。

方法：每个评估步复用真实 `PTFReplayWrapper._admission_slot_weights`
（被审计机制的核心权重函数）计算 critic 采样中源 slot 的条件期望占比：
    E[frac_t] = mean_env( sum(w_row[source slots]) / sum(w_row) )
multinomial(replacement=True) 的样本占比是该期望的无偏估计——期望法给出
同一数值的零方差版本。行为流（segment 边界的 option 选择）为随机，
用 3 个仿真 seed 覆盖，输出为"3 个仿真实现的范围"（非统计预测区间）。

输入契约（十二次复核修复 1）：candidate_masses **从冻结配置重算**——
bank YAML 各源 `bootstrap.weight` 作 source_logits + 冻结 CLI 的
student_logit，`softmax([source_logits, student_logit])`（τ=1，与
admission_control.AdmissionSnapshot.masses 相同公式）；lease checkpoint 的
`policy_events[0]` 仅用于一致性断言，不作为输入。其余冻结机制参数：
n_env=128、h=25、anchor=10000 步纯 student、branch=3000 步、buffer 51200、
uniform_mix=1.0 / recency=0 / priority=0 / authority active。
不读取任何 return、U 值、eval 结果或实测 critic 计数。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import torch
import yaml

torch.set_num_threads(16)

REPO = Path(__file__).resolve().parents[2]
import sys  # noqa: E402

sys.path.insert(0, str(REPO / "fasttd3_ptf" / "official_fasttd3_ptf"))
from ptf_replay import PTFReplayWrapper  # noqa: E402

N_ENV = 128
BUFFER_SIZE = 51200
ANCHOR_STEPS = 10000
BRANCH_STEPS = 3000
H = 25
EVAL_EVERY = 5
SEGMENT_BOUNDS = (750, 1500, 2250, 3000)
SIM_SEEDS = (101, 102, 103)

# 冻结配置（与正式 plan/run card §11 一致）。
TASKS = {
    "crawl": {
        "bank": "configs/source_banks/h1hand_loco_wfix_crawl.yaml",
        "student_logit": 16.6823567039,
        "env_name": "h1hand-crawl-v0",
    },
    "truck": {
        "bank": "configs/source_banks/h1hand_hurdle4_wfix_truck.yaml",
        "student_logit": 16.4139012941,
        "env_name": "h1hand-truck-v0",
    },
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def masses_from_frozen_config(task: str) -> tuple[list[float], list[str]]:
    """softmax([bank bootstrap weights, student_logit])，与训练侧同一公式。"""
    spec = TASKS[task]
    bank = yaml.safe_load((REPO / spec["bank"]).read_text())
    names = [s["name"] for s in bank["sources"]]
    logits = [float(s["bootstrap"]["weight"]) for s in bank["sources"]]
    full = torch.tensor([*logits, spec["student_logit"]], dtype=torch.float32)
    return torch.softmax(full, dim=0).tolist(), names


def assert_matches_checkpoint(task: str, masses: list[float]) -> None:
    """一致性断言（非输入）：冻结配置重算值 == 分支起点安装的先验。"""
    spec = TASKS[task]
    ckpt = REPO / "models" / f"{spec['env_name']}__p0_{task}_lease__1_13000.pt"
    if not ckpt.is_file():
        print(f"[warn] {ckpt} 不存在，跳过一致性断言")
        return
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    event = state["admission_audit"]["policy_events"][0]
    assert event["replay_ptr"] == 0
    recorded = event["candidate_masses"]
    for a, b in zip(masses, recorded):
        assert abs(a - b) < 1e-6, f"{task}: frozen-config mass {a} != checkpoint {b}"
    print(f"[assert] {task}: 冻结配置重算 masses 与 checkpoint 先验一致(atol 1e-6)")


class FakeBase:
    n_steps = 1

    def __init__(self) -> None:
        self.n_env = N_ENV
        self.buffer_size = BUFFER_SIZE
        self.device = torch.device("cpu")
        self.ptr = 0


def simulate(masses: list[float], num_sources: int, seed: int) -> dict:
    torch.manual_seed(seed)
    wrapper = PTFReplayWrapper(FakeBase())
    wrapper.set_admission_policy(
        admitted_sources=torch.ones(num_sources, dtype=torch.bool),
        candidate_masses=torch.tensor(masses, dtype=torch.float32),
        recency_half_life=0.0,
        uniform_mix=1.0,
        priority_alpha=0.0,
    )
    # anchor 段快进：options 初始化即全 -1（student），ptr 置 10000。
    wrapper.base.ptr = ANCHOR_STEPS
    probs = torch.tensor(masses, dtype=torch.float64)
    current_option = torch.full((N_ENV,), -1, dtype=torch.long)
    fracs: list[tuple[int, float]] = []
    for t in range(BRANCH_STEPS):
        if t % H == 0:
            draw = torch.multinomial(probs.expand(N_ENV, -1), 1).squeeze(1)
            current_option = torch.where(
                draw == num_sources, torch.full_like(draw, -1), draw
            )
        ptr = wrapper.base.ptr % BUFFER_SIZE
        wrapper.options[:, ptr] = current_option
        wrapper.base.ptr += 1
        if t % EVAL_EVERY == 0:
            valid_n = wrapper.valid_size
            w = wrapper._admission_slot_weights(valid_n)
            src = (wrapper.options[:, :valid_n] >= 0).float()
            frac = float(((w * src).sum(dim=1) / w.sum(dim=1)).mean())
            fracs.append((t + 1, frac))
    result = {}
    prev_bound = 0
    cum_num = 0.0
    cum_den = 0
    for bound in SEGMENT_BOUNDS:
        seg_vals = [f for s, f in fracs if prev_bound < s <= bound]
        cum_num += sum(seg_vals)
        cum_den += len(seg_vals)
        result[bound] = {
            "segment": sum(seg_vals) / len(seg_vals),
            "cumulative": cum_num / cum_den,
        }
        prev_bound = bound
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default=str(REPO / "docs" / "data" / "p0_posthoc" / "critic_fraction_expectation.json"),
    )
    args = parser.parse_args()
    git_head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False,
    ).stdout.strip()
    out = {
        "analysis_type": "posthoc_engineering_sensitivity_simulation",
        "method": "expectation over real _admission_slot_weights (v3, frozen-config masses)",
        "git_head": git_head,
        "inputs": {},
        "tasks": {},
    }
    for task, spec in TASKS.items():
        masses, names = masses_from_frozen_config(task)
        assert_matches_checkpoint(task, masses)
        num_sources = len(names)
        runs = {s: simulate(masses, num_sources, s) for s in SIM_SEEDS}
        cum_final = [runs[s][3000]["cumulative"] for s in SIM_SEEDS]
        out["inputs"][task] = {
            "bank_yaml": spec["bank"],
            "bank_yaml_sha256": _sha256(REPO / spec["bank"]),
            "student_logit": spec["student_logit"],
            "source_names": names,
            "recomputed_candidate_masses": masses,
        }
        out["tasks"][task] = {
            "per_seed": {str(s): runs[s] for s in SIM_SEEDS},
            "cumulative_at_3000_realization_range": [min(cum_final), max(cum_final)],
        }
        print(f"== {task} ==")
        for s in SIM_SEEDS:
            segs = ", ".join(f"{b}:{runs[s][b]['segment']:.4f}" for b in SEGMENT_BOUNDS)
            print(f"  sim{s}: cumulative@3000={runs[s][3000]['cumulative']:.4f}  分段[{segs}]")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    print(f"-> {out_path}")


if __name__ == "__main__":
    main()
