"""inventory v2 的单元测试（不读真实 checkpoint）。"""
import sys, json
from pathlib import Path
REPO = Path("/home/yjj/fasttd3_ptf")
sys.path.insert(0, str(REPO / "scripts/analysis"))
import build_checkpoint_inventory_v2 as inv

fails = []
def check(cond, msg):
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    if not cond: fails.append(msg)

print("T1 文件名解析")
r = inv.parse_filename("models/h1hand-slide-v0__slide_bac_walk_s1__1_20000.pt")
check(r["fname_env"]=="h1hand-slide-v0" and r["fname_seed"]==1 and r["fname_step"]==20000,
      f"数字步解析 {r['fname_env']}/{r['fname_seed']}/{r['fname_step']}")
r = inv.parse_filename("models/h1hand-door-v0__rjd_student_s1__1_final.pt")
check(r["fname_is_final"] and r["fname_step"] is None, "final 解析")
r = inv.parse_filename("garbage.pt")
check(r["fname_parsed"] is False, "不可解析返回 fname_parsed=False")

print("T2 mechanism 真值表（冻结）")
check(inv.classify_mechanism({}, None, None)==inv.MECH_NO_PTF, "无 ptf_cfg → NO_PTF")
check(inv.classify_mechanism({"a":1}, ["null"], None)==inv.MECH_PTF_NULL_BANK, "['null'] → NULL_BANK")
check(inv.classify_mechanism({"a":1}, [], None)==inv.MECH_PTF_NULL_BANK, "空 source → NULL_BANK")
check(inv.classify_mechanism({"a":1}, ["walk"], None)==inv.MECH_PTF_WITH_SOURCES, "有源 → WITH_SOURCES")

print("T3 协议感知 canonical")
c = inv.canonical_steps_for_run({"total_timesteps":100000},{"mcg_warmup_steps":30000})
check(30000 in c["steps"], "bootstrap_end 进入步集")
check(set(inv.FIXED_CANONICAL_STEPS) <= set(c["steps"]), "四个固定点都在")
check(c["hard_exit_step"]=="UNKNOWN_NO_DEDICATED_KEY", "hard_exit 记 UNKNOWN 不猜")
c2 = inv.canonical_steps_for_run({}, {})
check(30000 not in c2["steps"], "缺失项不填默认值")

print("T4 身份冲突三项逐一比对")
inner = {"inner_env_name":"h1hand-slide-v0","inner_seed":1,"inner_global_step":20000}
base = {"fname_parsed":True,"fname_env":"h1hand-slide-v0","fname_seed":1,
        "fname_step":20000,"fname_is_final":False}
check(inv.check_identity_conflict(base, inner)==[], "一致 → 无冲突")
for field, bad in (("fname_seed",9),("fname_step",99999),("fname_env","h1hand-crawl-v0")):
    c = inv.check_identity_conflict({**base, field:bad}, inner)
    check(len(c)==1, f"{field} 不符被单独检出")

print("T5 FINAL 去重与 AMBIGUOUS")
rows = [
 {"path":"a_100000.pt","eligibility":"ELIGIBLE","run_instance_id":"r#1",
  "inner_global_step":100000,"checkpoint_id":"SHA1","fname_is_final":False},
 {"path":"a_final.pt","eligibility":"ELIGIBLE","run_instance_id":"r#1",
  "inner_global_step":100000,"checkpoint_id":"SHA1","fname_is_final":True},
]
notes, amb = inv.resolve_final_and_duplicates(rows)
check(rows[1].get("final_resolution")=="FINAL_DUPLICATE_OF_100000", "同 sha → FINAL 去重")
check(rows[1].get("is_canonical") is False, "去重后不计入 canonical")
check(amb==[], "同 sha 不算 ambiguous")

rows2 = [dict(r) for r in rows]; rows2[1]["checkpoint_id"]="SHA2"
notes2, amb2 = inv.resolve_final_and_duplicates(rows2)
check(len(amb2)==1, "同 step 不同 sha → AMBIGUOUS_RUN_INSTANCE")
check(all(r["eligibility"]==inv.AMBIGUOUS_RUN_INSTANCE for r in rows2), "整组被标记")

print("T6 completion 在 run 层，中间点不判 interrupted")
rows3 = [
 {"eligibility":"ELIGIBLE","run_instance_id":"r#1","inner_global_step":20000,
  "configured_total_timesteps":100000},
 {"eligibility":"ELIGIBLE","run_instance_id":"r#1","inner_global_step":100000,
  "configured_total_timesteps":100000},
]
comp = inv.compute_run_completion(rows3)
check(comp["r#1"]["completion_status"]=="COMPLETED", "跑到终点 → COMPLETED")
check(rows3[0]["is_run_endpoint"] is False and rows3[1]["is_run_endpoint"] is True,
      "中间点 is_run_endpoint=False，但**不**被判 interrupted")
rows4 = [dict(rows3[0])]
check(inv.compute_run_completion(rows4)["r#1"]["completion_status"]=="TRUNCATED_RUN",
      "只有中间点的 run → TRUNCATED_RUN（run 层判断）")
rows5 = [{"eligibility":"ELIGIBLE","run_instance_id":"r#1","inner_global_step":5,
          "configured_total_timesteps":None}]
check(inv.compute_run_completion(rows5)["r#1"]["completion_status"]==inv.UNKNOWN_COMPLETION,
      "total 未知 → UNKNOWN_COMPLETION")

print("T7 experiment_role 恒 UNKNOWN（当前无 run card）")
check(inv.UNKNOWN_ROLE=="UNKNOWN_ROLE", "常量存在")

print()
print("全部通过" if not fails else f"{len(fails)} 项失败")
sys.exit(0 if not fails else 1)
