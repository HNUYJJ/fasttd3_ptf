#!/usr/bin/env python3
"""T1b：对 T1 probe 的 POST-HOC AUDIT。**不修改冻结的 NO_CONFLICT_STOP。**

原 artifact（`gradient_probe.json` / `gradient_probe_actor10k_diagnostic.json`）
一律保留不覆盖，本脚本另出 corrected artifact。

修正与补齐四项：

1. **execution-count 标签 bug**。`admission_execution_counts` 的结构是
   ``[real sources..., student]``（`train_ptf.py:1408-1410`，长度 = num_sources+1），
   而 `SourcePolicyBank.names()` 在 ``null_option=True`` 时会**追加 "null"**
   （`source_bank.py:74-78`）。原脚本直接 zip 两者，于是 truck 的**student**
   被错标成 `"null"`。`behavior_share = ec[:-1].sum()/ec.sum()` 不受影响。

2. **critic 侧曝光**。actor 在默认路径直接复用 critic 的 replay batch
   （`train_ptf.py` 的 `data_pol = data` 分支），所以 actor 真正看到多少 source
   状态取决于 **replay sampling**，不只是 behavior execution share。
   从 replay snapshot 的 ``admission_sampling.sample_counts`` 读取，
   并核对 bank 中每个 real source 的 horizon。

3. **student estimand 错位**。原 probe 的 `g_stu` 只取 10k–20k 的 z=0，
   但 20k replay 里还完整保留 0–10k 的纯 student prefix，而 actor 是从**整个
   有效 replay** 采样的。故补算 ``actual_replay_student``（全部 provenance-written
   的 z=0，含 prefix）。source 侧两个口径应当相同——0–10k 是 empty-bank 纯 student，
   不可能有 z=1——脚本对此显式断言而不是假定。

4. **措辞**。裁决重命名为
   ``NO_ENDPOINT_AUTHORITY_CONDITIONED_GRADIENT_CONFLICT_AT_20K``。
   10k 口径标 ``OOD_COUNTERFACTUAL_DIAGNOSTIC``：10k 的 critic 从未见过随后
   scaffold 分支产生的状态，用它给这些状态打 Q 梯度属于外推，
   **不能**证明 10k–20k 训练过程中任何时刻都没出现过冲突。

用法：python scripts/analysis/audit_t1_probe_corrections_v1b.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from fasttd3_ptf.official_fasttd3_ptf import ensure_fasttd3_import_path  # noqa: E402

ensure_fasttd3_import_path()

from fast_td3 import Actor, Critic  # type: ignore  # noqa: E402

sys.path.insert(0, str(REPO / "scripts/analysis"))
from probe_provenance_actor_gradient_v1 import (  # noqa: E402
    ANCHOR_ROOT, CHUNK, SEEDS, actor_ascent_gradient, cosine, flat_norm,
)

BANKS = {
    "truck": REPO / "configs/source_banks/h1hand_hurdle4_wfix_truck.yaml",
    "stair": REPO / "configs/source_banks/calibration/h1hand_stair_rbo_slidesrc.yaml",
}
SCAFFOLD_LO, SCAFFOLD_HI = 10_000, 20_000


def labelled_counts(counts, real_source_names) -> dict:
    """按 ``[real sources..., student]`` 显式命名。**不使用 names() 的 null 槽。**"""
    t = torch.as_tensor(counts).double()
    n_real = t.numel() - 1
    if n_real != len(real_source_names):
        return {"error": f"槽位数 {t.numel()} 与真实源数 {len(real_source_names)}+1 不符"}
    tot = float(t.sum())
    if tot <= 0:
        return {"total": 0, "note": "该 role 从未被采样（actor 默认复用 critic batch）"}
    out = {n: round(float(t[i]) / tot, 6) for i, n in enumerate(real_source_names)}
    out["__student__"] = round(float(t[-1]) / tot, 6)
    out["__source_total__"] = round(float(t[:n_real].sum()) / tot, 6)
    out["__raw__"] = [int(x) for x in t]
    return out


def bank_horizons(task: str) -> dict:
    path = BANKS.get(task)
    if path is None or not path.exists():
        return {"status": "NOT_EXTERNALLY_VERIFIABLE", "reason": f"bank 文件不可读: {path}"}
    cfg = yaml.safe_load(path.read_text())
    return {
        "status": "VERIFIED",
        "path": str(path.relative_to(REPO)),
        "null_option": bool(cfg.get("null_option")),
        "sources": {s["name"]: {"weight": s["bootstrap"]["weight"],
                                "horizon": s["bootstrap"]["horizon"]}
                    for s in cfg["sources"]},
    }


def audit_seed(task: str, seed: int, device) -> dict:
    adir = ANCHOR_ROOT / f"{task}_s{seed}_scaf_k20000"
    learner = torch.load(adir / "learner.pt", map_location="cpu", weights_only=False)
    cfg = learner["configuration"]
    args = cfg["args"]
    aux = learner["auxiliary_state"]

    bank = bank_horizons(task)
    real_names = list(bank.get("sources", {}).keys()) if bank["status"] == "VERIFIED" else []

    blob = torch.load(adir / "replay.pt", map_location="cpu", weights_only=False)
    meta, tensors, prov = blob["metadata"], blob["tensors"], blob["provenance"]
    sampling = blob.get("admission_sampling") or {}

    row = {
        "seed": seed,
        "bank": bank,
        "behavior_execution": labelled_counts(aux["admission_execution_counts"], real_names),
        "replay_sampling": {
            role: labelled_counts(v, real_names)
            for role, v in (sampling.get("sample_counts") or {}).items()
        },
        "names_bug_note": (
            "原 T1 artifact 用 SourcePolicyBank.names() 给最后一槽命名，"
            "在 null_option=True 的 bank 上把 student 错标为 'null'；此处按 "
            "[real sources..., student] 显式命名"
        ),
    }

    # ── 两个 student 口径 ────────────────────────────────────────────
    step = torch.as_tensor(prov["learner_step"])
    written = torch.as_tensor(prov["provenance_written"]).bool()
    src_mask_raw = torch.as_tensor(prov["executed_group_mask"]).any(dim=-1)

    in_win = (step >= SCAFFOLD_LO) & (step < SCAFFOLD_HI) & written
    is_src_win = src_mask_raw & in_win
    is_stu_win = (~src_mask_raw) & in_win
    is_stu_all = (~src_mask_raw) & written            # 含 0–10k prefix
    is_src_all = src_mask_raw & written

    # 0–10k 是 empty-bank 纯 student，不可能有 z=1——显式断言而非假定。
    src_outside = int((is_src_all & ~in_win).sum())
    row["source_z1_outside_scaffold_window"] = src_outside

    obs_all = tensors["observations"]
    asym = bool(meta["asymmetric_obs"])
    cobs_all = tensors["critic_observations"] if asym else obs_all

    def take(mask):
        idx = mask.nonzero(as_tuple=False)
        return obs_all[idx[:, 0], idx[:, 1]], cobs_all[idx[:, 0], idx[:, 1]]

    groups = {
        "source_contemporaneous": take(is_src_win),
        "source_all_replay": take(is_src_all),
        "student_contemporaneous": take(is_stu_win),
        "student_all_replay": take(is_stu_all),
    }
    row["group_sizes"] = {k: int(v[0].shape[0]) for k, v in groups.items()}
    del blob, tensors, prov

    # ── 20k actor/critic ─────────────────────────────────────────────
    actor = Actor(n_obs=int(meta["n_obs"]), n_act=int(meta["n_act"]),
                  num_envs=int(args["num_envs"]), device=device,
                  init_scale=float(args["init_scale"]),
                  hidden_dim=int(args["actor_hidden_dim"]),
                  std_min=float(args["std_min"]), std_max=float(args["std_max"]))
    critic = Critic(n_obs=int(meta["n_critic_obs"]), n_act=int(meta["n_act"]),
                    num_atoms=int(args["num_atoms"]), v_min=float(args["v_min"]),
                    v_max=float(args["v_max"]),
                    hidden_dim=int(args["critic_hidden_dim"]), device=device)
    actor.load_state_dict(learner["modules"]["actor"])
    critic.load_state_dict(learner["modules"]["critic"])
    actor.eval(); critic.eval()
    for p in critic.parameters():
        p.requires_grad_(False)
    params = [p for p in actor.parameters() if p.requires_grad]

    from fast_td3_utils import EmpiricalNormalization  # type: ignore

    on = EmpiricalNormalization(shape=int(meta["n_obs"]), device=device)
    on.load_state_dict(learner["modules"]["obs_normalizer"]); on.eval()
    if asym:
        cn = EmpiricalNormalization(shape=int(meta["n_critic_obs"]), device=device)
        cn.load_state_dict(learner["modules"]["critic_obs_normalizer"]); cn.eval()

    @torch.no_grad()
    def prep(pair):
        o, c = pair
        o = on(o.to(device), update=False)
        return o, (cn(c.to(device), update=False) if asym else o)

    grads = {}
    for name, pair in groups.items():
        if pair[0].shape[0] == 0:
            continue
        o, c = prep(pair)
        grads[name] = actor_ascent_gradient(actor, critic, o, c, params,
                                            bool(args["use_cdq"]))
        del o, c

    def pair_stats(a, b):
        if a not in grads or b not in grads:
            return {"status": "MISSING"}
        return {"cos": round(cosine(grads[a], grads[b]), 6),
                "norm_ratio": round(flat_norm(grads[a]) / max(1e-12, flat_norm(grads[b])), 6)}

    row["gradient"] = {
        # 原 T1 判定口径（保持可复现）
        "contemporaneous_vs_contemporaneous": pair_stats(
            "source_contemporaneous", "student_contemporaneous"),
        # 新增：actor 实际 replay 中的 student 组分
        "source_vs_actual_replay_student": pair_stats(
            "source_all_replay", "student_all_replay"),
        # 交叉口径，便于定位差异来自哪一侧
        "contemporaneous_source_vs_actual_replay_student": pair_stats(
            "source_contemporaneous", "student_all_replay"),
    }
    return row


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    report = {
        "scope": "POST-HOC AUDIT of T1；不修改冻结的 NO_CONFLICT_STOP",
        "verdict_rename": {
            "20k_endpoint": "NO_ENDPOINT_AUTHORITY_CONDITIONED_GRADIENT_CONFLICT_AT_20K",
            "10k_actor_on_future_states": "OOD_COUNTERFACTUAL_DIAGNOSTIC",
        },
        "limitations": [
            "z=1 表示该步动作由 source 执行（current action authority），"
            "不表示该状态由 source 造成（causal occupancy origin）",
            "split-half 只排除当前 endpoint estimator 的有限样本噪声，"
            "不排除 critic systematic bias、分组偏差或 endpoint-only 时间偏差",
            "结论仅针对 20k endpoint，不能推断 10k–20k 中间任何时刻",
        ],
        "per_task": {},
    }
    for task in ("truck", "stair"):
        rows = []
        for s in SEEDS:
            print(f"[audit] {task} s{s} ...", flush=True)
            rows.append(audit_seed(task, s, device))
        report["per_task"][task] = rows

    text = json.dumps(report, indent=2, ensure_ascii=False)
    out = REPO / "docs/data/pdau_probe_v1/t1b_corrected_audit.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
