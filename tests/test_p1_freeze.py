"""Phase-1 δ/门 B 冻结脚本的反例单测（二十二次复核要求）。

覆盖：checkpoint 身份验证、patch 逐段比较、wandb run 唯一匹配、历史 PTF
配置验证、ε 统计式、finalize 安全流程。
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "analysis"))

import p1_freeze_delta as fd  # noqa: E402
import p1_gate_b_tolerance as gb  # noqa: E402


# ---------- 门 B：checkpoint 身份验证 ----------

def _ckpt(path: Path, *, env="h1hand-basketball-v0", seed=1, step=30000,
          bank="configs/source_banks/h1hand_std9_wfix_basketball.yaml",
          logit=3.5892126423877646, n_masses=10,
          cfg_over: dict | None = None, args_over: dict | None = None) -> Path:
    """构造一个通过全部身份验证的 checkpoint；反例通过 cfg_over/args_over 改单项。"""
    cfg = {"source_bank": bank, "admission_student_logit": logit, **gb.REQUIRED_CFG}
    cfg.update(cfg_over or {})
    args = {"env_name": env, "seed": seed, **gb.REQUIRED_ARGS}
    args.update(args_over or {})
    torch.save({
        "args": args, "global_step": step, "ptf_cfg": cfg,
        "admission_audit": {"candidate_masses": [0.1] * n_masses,
                            "execution_counts": [1] * n_masses,
                            "critic_sample_counts": [1] * n_masses,
                            "main_buffer_counts": [1] * n_masses,
                            "active_buffer_counts": [1] * n_masses},
    }, path)
    return path


def test_gate_b_accepts_valid_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(gb, "REPO", tmp_path)
    p = _ckpt(tmp_path / "ok.pt")
    aud, ident = gb.load_verified(p, "basketball", 1, "30000")
    assert ident["seed"] == 1 and ident["global_step"] == 30000
    assert len(ident["sha256"]) == 64


@pytest.mark.parametrize("kw,msg", [
    ({"env": "h1hand-truck-v0"}, "env_name"),
    ({"seed": 2}, "seed"),
    ({"step": 60000}, "global_step"),
    ({"bank": "configs/source_banks/other.yaml"}, "source_bank"),
    ({"cfg_over": {"admission_mode": "none"}}, "admission_mode"),
    ({"cfg_over": {"mcg_warmup_steps": 15000}}, "mcg_warmup_steps"),
    ({"cfg_over": {"admission_replay_handoff": "fixed_quota"}}, "admission_replay_handoff"),
    ({"cfg_over": {"mcg_groups": ["legs_torso"]}}, "mcg_groups"),
    ({"n_masses": 5}, "candidate_masses"),
])
def test_gate_b_rejects_identity_mismatch(tmp_path, monkeypatch, kw, msg):
    """身份任一项不符必须拒绝——不再只信文件名（阻塞 2）。"""
    monkeypatch.setattr(gb, "REPO", tmp_path)
    p = _ckpt(tmp_path / "bad.pt", **kw)
    with pytest.raises(ValueError, match=msg):
        gb.load_verified(p, "basketball", 1, "30000")


def test_gate_b_epsilon_is_statistical_not_hardcoded():
    """ε 必须由 Hoeffding 公式给出，且按新跑臂最小区间推导（阻塞 1）。"""
    n = min(int(b) - int(a) for a, b in zip(gb.NEW_ARM_CKPTS[:-1], gb.NEW_ARM_CKPTS[1:]))
    n_samples = n * gb.NUM_UPDATES * gb.BATCH_SIZE
    expected = math.sqrt(math.log(2 * gb.M_COMPARISONS / gb.ALPHA) / (2 * n_samples))
    assert n_samples == 655_360_000
    assert expected == pytest.approx(9.068e-05, rel=1e-3)
    frozen = math.ceil(expected / gb.EPS_ROUND_UP) * gb.EPS_ROUND_UP
    assert frozen == pytest.approx(0.001)
    # 旧硬编码值 0.01 必须不再出现为 ε
    assert frozen < 0.01


def test_gate_b_historical_has_no_80k_point():
    """历史臂无 80k checkpoint，不得构造该数据点（口径修正）。"""
    assert "80000" not in gb.HIST_CKPTS
    assert "80000" in gb.NEW_ARM_CKPTS


# ---------- δ：patch 比较 / wandb 唯一匹配 / PTF 配置 ----------

def test_patch_excludes_only_offline_probe(tmp_path):
    """剔除 probe 段后相同的两个 patch 必须判为一致（阻塞 3）。"""
    common = "diff --git a/foo.py b/foo.py\n@@ -1 +1 @@\n-a\n+b\n"
    probe_a = f"diff --git a/{fd.PROBE_PATH} b/{fd.PROBE_PATH}\n@@ -1 +1 @@\n-x\n+y\n"
    probe_b = f"diff --git a/{fd.PROBE_PATH} b/{fd.PROBE_PATH}\n@@ -9 +9 @@\n-p\n+q\n"
    pa, pb = tmp_path / "a.patch", tmp_path / "b.patch"
    pa.write_text(common + probe_a)
    pb.write_text(common + probe_b)
    assert fd.patch_sections_excluding_probe(pa) == fd.patch_sections_excluding_probe(pb)


def test_patch_difference_outside_probe_is_detected(tmp_path):
    """probe 之外的差异必须被检出（不得被剔除逻辑掩盖）。"""
    pa, pb = tmp_path / "a.patch", tmp_path / "b.patch"
    pa.write_text("diff --git a/train.py b/train.py\n@@ -1 +1 @@\n-a\n+b\n")
    pb.write_text("diff --git a/train.py b/train.py\n@@ -1 +1 @@\n-a\n+C\n")
    assert fd.patch_sections_excluding_probe(pa) != fd.patch_sections_excluding_probe(pb)


def _wandb_run(root: Path, name: str, exp: str, *, mcg=False, execute=False, bank=None):
    d = root / "wandb" / name / "files"
    d.mkdir(parents=True)
    (d / "wandb-metadata.json").write_text(json.dumps(
        {"args": ["--env_name", "h1hand-basketball-v0", "--exp_name", exp, "--seed", "1"],
         "git": {"commit": "b" * 40}}))
    (d / "config.yaml").write_text(
        f"ptf:\n  value:\n    mcg: {str(mcg).lower()}\n    execute_sources: {str(execute).lower()}\n"
        + (f"source_bank:\n  value: {bank}\n" if bank else ""))
    return d.parent


def test_find_wandb_run_requires_exactly_one(tmp_path, monkeypatch):
    monkeypatch.setattr(fd, "REPO", tmp_path)
    with pytest.raises(ValueError, match="matched 0 wandb runs"):
        fd.find_wandb_run("nope")
    _wandb_run(tmp_path, "run-1", "dup_exp")
    _wandb_run(tmp_path, "run-2", "dup_exp")
    with pytest.raises(ValueError, match="matched 2 wandb runs"):
        fd.find_wandb_run("dup_exp")


@pytest.mark.parametrize("kw,msg", [
    ({"mcg": True}, "ptf.mcg"),
    ({"execute": True}, "ptf.execute_sources"),
    ({"bank": "configs/source_banks/x.yaml"}, "source_bank is not empty"),
])
def test_historical_ptf_config_must_be_scratch(tmp_path, monkeypatch, kw, msg):
    """只查 CLI 无 --ptf 不够；历史实际配置必须是 scratch（阻塞 3 第 4 点）。"""
    monkeypatch.setattr(fd, "REPO", tmp_path)
    run = _wandb_run(tmp_path, "run-x", "exp_x", **kw)
    with pytest.raises(ValueError, match=msg):
        fd.assert_pure_scratch_config(run, "exp_x")


def test_historical_pure_scratch_config_passes(tmp_path, monkeypatch):
    monkeypatch.setattr(fd, "REPO", tmp_path)
    run = _wandb_run(tmp_path, "run-ok", "exp_ok")
    ident = fd.assert_pure_scratch_config(run, "exp_ok")
    assert ident["ptf_mcg"] is False and ident["ptf_execute_sources"] is False


# ---------- finalize 安全流程 ----------

@pytest.mark.parametrize("mod", [fd, gb])
def test_finalize_rejects_candidate_sha_mismatch(tmp_path, monkeypatch, mod):
    cand = tmp_path / "cand.json"
    cand.write_text(json.dumps({"deltas": {}, "epsilon": {}, "per_task": {}}))
    monkeypatch.setattr(sys, "argv", [
        "x", "--finalize", "--candidate", str(cand),
        "--expected-candidate-sha256", "0" * 64, "--out", str(tmp_path / "frozen.json")])
    monkeypatch.setattr(mod, "compute", lambda: {"git_dirty": False, "deltas": {}, "epsilon": {}, "per_task": {}})
    with pytest.raises(ValueError, match="candidate sha mismatch"):
        mod.main()


@pytest.mark.parametrize("mod", [fd, gb])
def test_finalize_rejects_dirty_tree(tmp_path, monkeypatch, mod):
    cand = tmp_path / "cand.json"
    cand.write_text(json.dumps({"deltas": {}, "epsilon": {}, "per_task": {}}))
    real_sha = hashlib.sha256(cand.read_bytes()).hexdigest()
    monkeypatch.setattr(sys, "argv", [
        "x", "--finalize", "--candidate", str(cand),
        "--expected-candidate-sha256", real_sha, "--out", str(tmp_path / "frozen.json")])
    monkeypatch.setattr(mod, "compute", lambda: {"git_dirty": True, "deltas": {}, "epsilon": {}, "per_task": {}})
    with pytest.raises(RuntimeError, match="dirty"):
        mod.main()


@pytest.mark.parametrize("mod", [fd, gb])
def test_output_refuses_overwrite(tmp_path, monkeypatch, mod):
    out = tmp_path / "exists.json"
    out.write_text("{}")
    monkeypatch.setattr(sys, "argv", ["x", "--out", str(out)])
    with pytest.raises(FileExistsError):
        mod.main()


def test_finalize_rejects_recomputed_mismatch(tmp_path, monkeypatch):
    """重算结果与 candidate 不一致必须拒绝冻结。"""
    cand = tmp_path / "cand.json"
    cand.write_text(json.dumps({"deltas": {"basketball": {"delta_sesoi": 1.0}}}))
    real_sha = hashlib.sha256(cand.read_bytes()).hexdigest()
    monkeypatch.setattr(sys, "argv", [
        "x", "--finalize", "--candidate", str(cand),
        "--expected-candidate-sha256", real_sha, "--out", str(tmp_path / "f.json")])
    monkeypatch.setattr(fd, "compute", lambda: {
        "git_dirty": False, "deltas": {"basketball": {"delta_sesoi": 999.0}}})
    with pytest.raises(ValueError, match="differ"):
        fd.main()


# ---------- 二十三次复核：finalize 必须比较完整科学 payload ----------

def _finalize_with(mod, tmp_path, monkeypatch, cand_payload, fresh_payload):
    cand = tmp_path / "cand.json"
    cand.write_text(json.dumps(cand_payload))
    real_sha = hashlib.sha256(cand.read_bytes()).hexdigest()
    monkeypatch.setattr(sys, "argv", [
        "x", "--finalize", "--candidate", str(cand),
        "--expected-candidate-sha256", real_sha, "--out", str(tmp_path / "frozen.json")])
    monkeypatch.setattr(mod, "compute", lambda: fresh_payload)
    mod.main()


@pytest.mark.parametrize("mod,mutate_key", [
    (fd, "sources"), (fd, "verified_asserts"), (fd, "generator_sha256"),
    (gb, "input_identities"), (gb, "generator_sha256"), (gb, "ckpt_schedule"),
])
def test_finalize_rejects_input_change_even_if_summary_unchanged(
        tmp_path, monkeypatch, mod, mutate_key):
    """汇总数不变但输入身份/generator 变化时，finalize 必须拒绝（阻塞 1）。"""
    base = {
        "git_dirty": False,
        "deltas": {"basketball": {"delta_sesoi": 1.0}},
        "epsilon": {"eps_frozen": 0.001}, "per_task": {"basketball": {}},
        "sources": {"basketball_s1": {"train_log_sha256": "a" * 64}},
        "verified_asserts": {"git_base": "b" * 40},
        "input_identities": [{"sha256": "c" * 64}],
        "ckpt_schedule": {"new_arms": ["30000"]},
        "generator_sha256": "d" * 64,
    }
    fresh = json.loads(json.dumps(base))
    # 只改被审输入身份，汇总值（deltas/epsilon/per_task）保持不变
    fresh[mutate_key] = {"tampered": True} if isinstance(base[mutate_key], dict) else [{"sha256": "e" * 64}]
    if mutate_key == "generator_sha256":
        fresh[mutate_key] = "e" * 64
    with pytest.raises(ValueError, match="differ"):
        _finalize_with(mod, tmp_path, monkeypatch, base, fresh)


def _printable_payload(mod) -> dict:
    """构造能走完 main() 打印路径的最小完整 payload。"""
    base = {"git_dirty": False, "git_head": "old", "generator_sha256": "d" * 64}
    if mod is fd:
        base["deltas"] = {"basketball": {"delta_sesoi": 1.0}, "truck": {"delta_sesoi": 2.0}}
        base["sources"] = {"basketball_s1": {"train_log_sha256": "a" * 64}}
    else:
        base["epsilon"] = {"eps_raw": 9.068e-05, "eps_frozen": 0.001,
                           "N_min_interval_critic_samples": 655_360_000}
        base["per_task"] = {"basketball": {
            "envelope_exceedance_observed_max": 0.0,
            "behavior_source_share_30k": {"mean": 0.49},
            "critic_source_share_30k": {"mean": 0.497}}}
        base["input_identities"] = [{"sha256": "c" * 64}]
    return base


@pytest.mark.parametrize("mod", [fd, gb])
def test_finalize_accepts_identical_scientific_payload(tmp_path, monkeypatch, mod):
    """科学 payload 完全一致时应成功冻结（动态字段差异不阻断）。"""
    base = _printable_payload(mod)
    fresh = json.loads(json.dumps(base))
    fresh["git_head"] = "new"  # 动态字段：不应阻断
    _finalize_with(mod, tmp_path, monkeypatch, base, fresh)
    frozen = json.loads((tmp_path / "frozen.json").read_text())
    assert frozen["status"] == "frozen"
    assert frozen["finalized_from_candidate"]["sha256"]


# ---------- 二十三次复核：门 B 完整同配置验证 ----------

@pytest.mark.parametrize("cfg_key,bad_value", [
    ("mcg", False),
    ("mcg_warmup_mode", "random"),
    ("mcg_ablation", "full"),
    ("mcg_warmup_min_steps", 10),
    ("admission_replay_recency_half_life", 5.0),
    ("admission_replay_uniform_mix", 0.05),
    ("admission_replay_priority_alpha", 0.5),
])
def test_gate_b_rejects_wrong_mechanism_cfg(tmp_path, monkeypatch, cfg_key, bad_value):
    """机制配置任一项不符必须拒绝（阻塞 2）。"""
    monkeypatch.setattr(gb, "REPO", tmp_path)
    p = tmp_path / "bad_cfg.pt"
    cfg = {"source_bank": "configs/source_banks/h1hand_std9_wfix_basketball.yaml",
           "admission_student_logit": 3.5892126423877646, **gb.REQUIRED_CFG}
    cfg[cfg_key] = bad_value
    torch.save({"args": {"env_name": "h1hand-basketball-v0", "seed": 1, **gb.REQUIRED_ARGS},
                "global_step": 30000, "ptf_cfg": cfg,
                "admission_audit": {"candidate_masses": [0.1] * 10}}, p)
    with pytest.raises(ValueError, match=cfg_key):
        gb.load_verified(p, "basketball", 1, "30000")


@pytest.mark.parametrize("arg_key,bad_value", [
    ("num_envs", 64), ("batch_size", 16384), ("buffer_size", 25600),
    ("num_updates", 1), ("learning_starts", 1000), ("total_timesteps", 50000),
    ("eval_interval", 10000),
])
def test_gate_b_rejects_wrong_training_scale(tmp_path, monkeypatch, arg_key, bad_value):
    """训练规模参数不符必须拒绝（阻塞 2）。"""
    monkeypatch.setattr(gb, "REPO", tmp_path)
    p = tmp_path / "bad_args.pt"
    args = {"env_name": "h1hand-basketball-v0", "seed": 1, **gb.REQUIRED_ARGS}
    args[arg_key] = bad_value
    torch.save({"args": args, "global_step": 30000,
                "ptf_cfg": {"source_bank": "configs/source_banks/h1hand_std9_wfix_basketball.yaml",
                            "admission_student_logit": 3.5892126423877646, **gb.REQUIRED_CFG},
                "admission_audit": {"candidate_masses": [0.1] * 10}}, p)
    with pytest.raises(ValueError, match=arg_key):
        gb.load_verified(p, "basketball", 1, "30000")


def test_gate_b_rejects_wrong_student_logit(tmp_path, monkeypatch):
    monkeypatch.setattr(gb, "REPO", tmp_path)
    p = tmp_path / "bad_logit.pt"
    torch.save({"args": {"env_name": "h1hand-basketball-v0", "seed": 1, **gb.REQUIRED_ARGS},
                "global_step": 30000,
                "ptf_cfg": {"source_bank": "configs/source_banks/h1hand_std9_wfix_basketball.yaml",
                            "admission_student_logit": 99.9, **gb.REQUIRED_CFG},
                "admission_audit": {"candidate_masses": [0.1] * 10}}, p)
    with pytest.raises(ValueError, match="admission_student_logit"):
        gb.load_verified(p, "basketball", 1, "30000")


def test_gate_b_conditional_field_absence_is_recorded_not_faked(tmp_path, monkeypatch):
    """truck 版本无 admission_adaptive 字段——须如实记录缺失，不得假装验证。"""
    monkeypatch.setattr(gb, "REPO", tmp_path)
    p = tmp_path / "no_adaptive.pt"
    torch.save({"args": {"env_name": "h1hand-truck-v0", "seed": 1, **gb.REQUIRED_ARGS},
                "global_step": 30000,
                "ptf_cfg": {"source_bank": "configs/source_banks/h1hand_hurdle4_wfix_truck.yaml",
                            "admission_student_logit": 14.216676716804526, **gb.REQUIRED_CFG},
                "admission_audit": {"candidate_masses": [0.2] * 5}}, p)
    _, ident = gb.load_verified(p, "truck", 1, "30000")
    assert "absent" in ident["conditional_cfg"]["admission_adaptive"]


def test_gate_b_rejects_conditional_field_wrong_value(tmp_path, monkeypatch):
    """字段存在但值错（adaptive=True）必须拒绝。"""
    monkeypatch.setattr(gb, "REPO", tmp_path)
    p = tmp_path / "adaptive_on.pt"
    torch.save({"args": {"env_name": "h1hand-basketball-v0", "seed": 1, **gb.REQUIRED_ARGS},
                "global_step": 30000,
                "ptf_cfg": {"source_bank": "configs/source_banks/h1hand_std9_wfix_basketball.yaml",
                            "admission_student_logit": 3.5892126423877646,
                            "admission_adaptive": True, **gb.REQUIRED_CFG},
                "admission_audit": {"candidate_masses": [0.1] * 10}}, p)
    with pytest.raises(ValueError, match="admission_adaptive"):
        gb.load_verified(p, "basketball", 1, "30000")


def test_gate_b_full_masses_vector_is_captured(tmp_path, monkeypatch):
    """身份记录须含完整 masses 向量（供跨 ckpt/seed 一致性比较）。"""
    monkeypatch.setattr(gb, "REPO", tmp_path)
    p = tmp_path / "ok.pt"
    masses = [0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.19]
    torch.save({"args": {"env_name": "h1hand-basketball-v0", "seed": 1, **gb.REQUIRED_ARGS},
                "global_step": 30000,
                "ptf_cfg": {"source_bank": "configs/source_banks/h1hand_std9_wfix_basketball.yaml",
                            "admission_student_logit": 3.5892126423877646, **gb.REQUIRED_CFG},
                "admission_audit": {"candidate_masses": masses}}, p)
    _, ident = gb.load_verified(p, "basketball", 1, "30000")
    assert ident["candidate_masses"] == masses
