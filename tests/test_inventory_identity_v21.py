#!/usr/bin/env python3
"""Inventory v2.1 身份模型的单元测试（P2.1 预注册 §2–§6）。

每组对应 v2 的一个实测缺陷；构造用例覆盖真实数据中不存在的场景
（预注册 §9 允许，须在结果中标明是构造用例）。

运行：python tests/test_inventory_identity_v21.py
"""

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from fasttd3_ptf.evaluation import inventory_identity as ident  # noqa: E402

FAILS = []


def check(cond, msg):
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    if not cond:
        FAILS.append(msg)


print("T1  文件名解析必须支持含下划线的 env（v2 的真实 bug）")
for nm, env, seed, step in [
    ("h1hand-balance_hard-v0__h1hand_balance_hard_b2_scr_s1__1_100000.pt",
     "h1hand-balance_hard-v0", 1, 100000),
    ("h1hand-bookshelf_simple-v0__x_b2_scr_s1__2_final.pt",
     "h1hand-bookshelf_simple-v0", 2, None),
    ("h1hand-slide-v0__shev1_exit_s3__3_20000.pt", "h1hand-slide-v0", 3, 20000),
    ("h1hand-crawl-v0__p0_crawl_abstain__1_13000.pt", "h1hand-crawl-v0", 1, 13000),
]:
    r = ident.parse_filename(nm)
    check(r["fname_parsed"] and r["fname_env"] == env
          and r["fname_seed"] == seed and r["fname_step"] == step,
          f"{env} 解析正确（v2 在此失配并 fail-open）")

r = ident.parse_filename("not_a_valid_name.pt")
check(r["fname_parsed"] is False, "不可解析的名字必须 fname_parsed=False")

print("\nT2  effective_endpoint（禁止 min()）")
e = ident.effective_endpoint({"total_timesteps": 100000}, {"run_stop_step": 30000})
check(e["endpoint"] == 30000 and e["source"] == "run_stop_step",
      "显式 run_stop_step 优先")
e = ident.effective_endpoint({"total_timesteps": 100000}, {})
check(e["endpoint"] == 100000 and e["source"] == "total_timesteps",
      "无 run_stop_step 时用 total_timesteps")
e = ident.effective_endpoint({"total_timesteps": 100000}, {"run_stop_step": 200000})
check(e["source"] == ident.INVALID_ENDPOINT_CONFIG,
      "run_stop > total → INVALID_ENDPOINT_CONFIG（**不是**静默 min()）")
e = ident.effective_endpoint({"total_timesteps": 100000}, {"run_stop_step": 0})
check(e["source"] == ident.INVALID_ENDPOINT_CONFIG, "run_stop=0 → INVALID")
e = ident.effective_endpoint({}, {})
check(e["endpoint"] is None, "两者都无 → endpoint 不可知")

print("\nT3  canonical 只保留 <= effective_endpoint（v2 让 total=13 列出 100k）")
c = ident.canonical_steps({"total_timesteps": 13}, {})
check(c["steps"] == [13], f"total=13 → canonical 只有 [13]，实得 {c['steps']}")
check(set(c["out_of_scope"]) >= {10000, 20000, 50000, 100000},
      "四个固定点全部进 out_of_scope")
c = ident.canonical_steps({"total_timesteps": 100000}, {"mcg_warmup_steps": 30000})
check(30000 in c["steps"] and 100000 in c["steps"], "正常 run 保留 bootstrap 与终点")
# 预注册 §9 第 8 项：endpoint < bootstrap_end
c = ident.canonical_steps({"total_timesteps": 100000},
                          {"run_stop_step": 20000, "mcg_warmup_steps": 30000})
check(30000 in c["out_of_scope"] and 30000 not in c["steps"],
      "endpoint(20k) < bootstrap_end(30k) → bootstrap_end 进 out_of_scope")
check(c["steps"] == [10000, 20000], f"只剩 10k/20k，实得 {c['steps']}")

print("\nT4  completion 在 run 层（预注册 §9 第 6/7 项）")
check(ident.completion_status(30000, 30000) == "COMPLETED",
      "run_stop=30k,total=100k,observed=30k → COMPLETED（v2 会误判 TRUNCATED）")
check(ident.completion_status(20000, 30000) == "TRUNCATED_RUN",
      "observed < endpoint → TRUNCATED_RUN")
check(ident.completion_status(None, 30000) == ident.UNKNOWN_COMPLETION,
      "observed 未知 → UNKNOWN_COMPLETION")
check(ident.completion_status(30000, None) == ident.UNKNOWN_COMPLETION,
      "endpoint 未知 → UNKNOWN_COMPLETION")

print("\nT5  digest 三分（v2 要求配对时 ptf_cfg 相同会拒绝所有真实对照）")
base_args = {"total_timesteps": 100000, "batch_size": 256, "gamma": 0.99}
cont = {"admission_mode": "all", "mcg_warmup_steps": 30000, "anchor_dir": "a1"}
exit_ = {"admission_mode": "none", "mcg_warmup_steps": 30000, "anchor_dir": "a1"}
d1 = ident.compute_digests(base_args, cont, ["walk"])
d2 = ident.compute_digests(base_args, exit_, ["walk"])
check(d1["ptf_cfg_digest"] != d2["ptf_cfg_digest"], "两臂的 ptf_cfg_digest 不同（正常）")
check(d1["treatment_digest"] != d2["treatment_digest"], "treatment_digest 不同（正常）")
check(d1["pairing_invariant_digest"] == d2["pairing_invariant_digest"],
      "**pairing_invariant_digest 相同 → 两臂可配对**（v2 在此会拒绝）")
d3 = ident.compute_digests({**base_args, "batch_size": 512}, cont, ["walk"])
check(d1["pairing_invariant_digest"] != d3["pairing_invariant_digest"],
      "batch_size 不同 → pairing_invariant 不同，正确拒绝")
d4 = ident.compute_digests(base_args, {**cont, "anchor_dir": "a2"}, ["walk"])
check(d1["pairing_invariant_digest"] != d4["pairing_invariant_digest"],
      "anchor_dir 不同 → pairing_invariant 不同")
dm = ident.compute_digests({}, {}, None)
check(dm["ptf_cfg_digest"] == ident.NO_PROTOCOL, "无 ptf_cfg → NO_PROTOCOL")

print("\nT6  run card 匹配（禁止词义猜测；path_prefix 优先）")
entries = json.loads(
    (REPO / "docs/data/run_cards/run_card_registry_v1.json").read_text(encoding="utf-8")
)["entries"]

m = ident.match_run_card(entries, "p0_crawl_abstain",
                         "models/h1hand-crawl-v0__p0_crawl_abstain__1_13000.pt")
check(m and m["execution_role"] == ident.EXEC_FORMAL, "正式路径 → FORMAL")
m_a = ident.match_run_card(
    entries, "p0_crawl_abstain",
    "models/p0_dup_archive/crawl_A/h1hand-crawl-v0__p0_crawl_abstain__1_13000.pt")
check(m_a and m_a["execution_role"] == ident.EXEC_FORMAL and m_a["alias_of_formal_path"],
      "archive_A → FORMAL 且 alias_of_formal_path（与正式路径同一 execution）")
m_b = ident.match_run_card(
    entries, "p0_crawl_abstain",
    "models/p0_dup_archive/crawl_B/h1hand-crawl-v0__p0_crawl_abstain__1_13000.pt")
check(m_b and m_b["execution_role"] == ident.EXEC_DUPLICATE,
      "archive_B → REPEATABILITY_DUPLICATE")
check(m_b["counts_as_new_learner_replication"] is False,
      "B 不计作新的 learner replication")
check(m["match_group"] == m_a["match_group"] == m_b["match_group"] == "p0_lease_oracle_crawl",
      "三者同一 match_group（模板 {task} 已填充）")

for exp, role in [("shev1_prefix_s1", "PREFIX"), ("shev1_cont_s2", "CONTINUOUS"),
                  ("shev1_exit_s3", "HARD_EXIT")]:
    m = ident.match_run_card(entries, exp, f"models/x__{exp}__1_20000.pt")
    check(m and m["experiment_role"] == role, f"{exp} → {role}")
m1 = ident.match_run_card(entries, "shev1_cont_s2", "models/a.pt")
m2 = ident.match_run_card(entries, "shev1_exit_s2", "models/b.pt")
check(m1["match_group"] == m2["match_group"],
      "同 seed 的 cont / exit 同一 match_group（共享 prefix bundle）")

m = ident.match_run_card(entries, "rck_walk_s1", "models/x__rck_walk_s1__1_10000.pt")
check(m and m["experiment_role"] == "RACING_ARM_walk", "rck_walk → RACING_ARM_walk")
m = ident.match_run_card(entries, "rad_crawl_run_s1", "models/x__rad_crawl_run_s1__1_10000.pt")
check(m and m["experiment_role"] == "RACING_ARM_run"
      and m["match_group"] == "racing_admission_v1_crawl_s1",
      "rad_crawl_run_s1 → RACING_ARM_run，match_group 含 target")

check(ident.match_run_card(entries, "h1hand_truck_scratch50k_s1_2026", "models/x.pt") is None,
      "名字里有 scratch 但无冻结证据 → 不匹配（由调用方记 UNKNOWN_ROLE）")
check(ident.match_run_card(entries, None, "models/x.pt") is None, "exp_name 为 None → 不匹配")

print("\nT7  三层身份：duplicate 不产生新的 learner replication")
i_a = ident.build_identity("h1hand-crawl-v0", "p0_crawl_abstain", 1, ident.EXEC_FORMAL)
i_b = ident.build_identity("h1hand-crawl-v0", "p0_crawl_abstain", 1, ident.EXEC_DUPLICATE)
check(i_a["run_family_id"] == i_b["run_family_id"], "A/B 同一 run_family")
check(i_a["execution_instance_id"] != i_b["execution_instance_id"],
      "A/B 不同 execution_instance —— 这正是 v2 表达不了的")
check(i_a["learner_replication_id"] == i_b["learner_replication_id"],
      "A/B 同一 learner_replication（不计作两个独立 seed）")
i_none = ident.build_identity(None, "x", 1, ident.EXEC_FORMAL)
check(i_none["run_family_id"] is None, "缺 env → 身份为 None，不猜")

print()
print("全部通过" if not FAILS else f"{len(FAILS)} 项失败")
sys.exit(0 if not FAILS else 1)
