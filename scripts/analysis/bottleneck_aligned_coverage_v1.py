"""Bottleneck-Aligned Coverage (BAC)：零交互的冻结源迁移效用预测器（v1）。

动机
----
本项目已封存八个迁移性信号族（zero-shot return、T⁰、T^critic、SIV、SHU、
P0 lease oracle、update-space influence、zero-shot 行为探针），全部失败。
它们的共同点是**都把 target 的 reward 聚合成一个标量**。

而 HumanoidBench 的 reward 不是标量，是带组合算子的分量结构。标量 return
把三样东西混在一起：

    return  =  每步分量质量  ×  分量组合算子  ×  生存时长

于是出现 return 系统性反向的情形（本文的四个独立机制）：
  · min 算子     crawl: stand 抬高 crawling 却压垮 crawling_head，min 取小者反而更差
  · 乘性归零     sit_hard: stand 把乘性因子 sit_reward 打到 0.005，总 reward 必崩
  · 生存时长     slide/stair: stand 每步 reward 最低，却因不摔而 return 最高
  · 权重错配     door: run 推进 passage(w=.35)，瓶颈 door_openness(w=.45) 零覆盖

BAC 的做法是不聚合：在**分量层**上问"源推进的是不是 student 的瓶颈"。

定义
----
对 target 的 reward 结构（见 configs/reward_structure/humanoidbench_v1.py），
以 zero-action 基线 x[zero] 作为 student 的起点代理（t=0 时 student ≈ 随机）：

  边际敏感度   m_c = ∂R/∂x_c 在 x[zero] 处
      加性项    m_c = w_c            （若被门控，再乘 gate[zero]）
      乘性因子  m_c = Π_{c'≠c} x_{c'}[zero]

  瓶颈集合     B = 按 m_c·(1 − x_c[zero]) 降序累计到 ≥ BOTTLENECK_MASS 的分量
               —— 即"占据可改进空间一半以上"的那些分量

  Coverage_i = Σ_{c∈B}   m_c · max(0, x_c[i] − x_c[zero])     只在瓶颈分量上计正向
  Damage_i   = Σ_{c∈all} m_c · min(0, x_c[i] − x_c[zero])     在所有分量上计负向
  NET_i      = Coverage_i + Damage_i                          ← 主量

这个正负不对称是有语义的，不是拟合手段：
  · 正向只算瓶颈分量——非瓶颈分量已接近饱和，把它推得更高不产生学习价值，
    只是让 return 好看（crawl 上 stand 把 crawling 从 0.544 抬到 0.845 即属此类）；
  · 负向算全部分量——在乘性/门控结构下任一因子被压垮都是结构性破坏，
    注入的 transition 会携带错误的 credit assignment，不可由其他分量补偿。

**记录一次事后调整**：v1 初稿只用 Coverage 作主量，它在 crawl 上预测 walk/run
有正效用（Coverage 0.104/0.113），与实测（−217/−208，全负）矛盾。加入 Damage
后 NET 全负且 stand 最负，与实测一致。此调整发生在已看到 crawl 结果之后，
因此 crawl 不能计为对 NET 的独立验证，只有前瞻预测才算。

通用姿态分量（stand_reward / small_control / upright / dont_move）在几乎所有
任务中都存在，是 loco 源最容易推高的部分，也正是 return 被"看起来在进步"污染
的来源。它们参与 m_c 计算（乘性任务里它们确实有否决权），但在加性任务里若落入
B，会被标注出来供审查。

无界任务（package / push / truck / cabinet）的 return 被稀疏事件主导，
BAC 对它们只输出结构性判定 UNMEASURABLE，不给排序——这与已观测到的
CABINET_UNCERTAIN 和 package「return 不可分辨」一致。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from configs.reward_structure.humanoidbench_v1 import (  # noqa: E402
    ADDITIVE, GATED, MULTIPLICATIVE, UNBOUNDED, GENERIC_TERMS, SPEC,
)

PROBE = REPO / "logs/probe/transfer_map_v1.jsonl"
BOTTLENECK_MASS = 0.50      # 瓶颈集合的累计可改进空间占比阈值（冻结）
SIGN_EPS = 0.005            # |NET| 小于此值视为"无实质效应"（reward 单位/步）
SEPARATION_MIN = 0.02       # NET 极差小于此值则不给排序，只给符号预测
SOURCES = ("stand", "walk", "run")


def load_probe() -> dict:
    by: dict = {}
    for line in PROBE.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        by.setdefault(r["target"], {})[r["source"]] = r
    return by


def _resolve(info: dict, key: str) -> float:
    """支持 'a*b' 形式的复合项（源码里 stand_reward*small_control 常作为一项）。"""
    if "*" in key:
        v = 1.0
        for part in key.split("*"):
            v *= float(info.get(part, 0.0))
        return v
    return float(info.get(key, 0.0))


def analyze(target: str, arms: dict) -> dict:
    spec = SPEC[target]
    kind = spec["kind"]
    zero = arms["zero"]["info_means"]

    if kind == UNBOUNDED:
        return {"target": target, "kind": kind, "verdict": "UNMEASURABLE",
                "reason": "return 被稀疏事件/无界距离项主导，分量层不足以定序"}

    # ---- 1. 建立分量清单与在 zero 点的边际敏感度 m_c ----
    m: dict[str, float] = {}
    x0: dict[str, float] = {}

    if kind == MULTIPLICATIVE:
        facs = spec["factors"]
        vals = {c: _resolve(zero, c) for c in facs}
        for c in facs:
            prod = 1.0
            for c2 in facs:
                if c2 != c:
                    prod *= vals[c2]
            m[c] = prod
            x0[c] = vals[c]
    else:                                    # ADDITIVE / GATED
        gate = 1.0
        for g in spec.get("gates", []):
            gate *= _resolve(zero, g)
        for c, w in spec.get("terms", {}).items():
            m[c] = w * gate
            x0[c] = _resolve(zero, c)
        for w, group in spec.get("min_groups", []):
            key = "min(" + ",".join(group) + ")"
            m[key] = w * gate
            x0[key] = min(_resolve(zero, g) for g in group)
        # 门控因子本身的敏感度 = 内和在 zero 点的值
        inner = sum(m[c] * x0[c] for c in list(m)) / max(gate, 1e-9)
        for g in spec.get("gates", []):
            m[g] = inner
            x0[g] = _resolve(zero, g)

    # ---- 2. 瓶颈集合 B：按 m_c·(1 − x_c) 降序累计到 ≥ BOTTLENECK_MASS ----
    headroom = {c: m[c] * max(0.0, 1.0 - x0[c]) for c in m}
    total = sum(headroom.values())
    B, acc = [], 0.0
    for c, h in sorted(headroom.items(), key=lambda kv: -kv[1]):
        if total <= 0:
            break
        B.append(c)
        acc += h
        if acc / total >= BOTTLENECK_MASS:
            break

    # ---- 3. 每个源的 BAC 与 Damage ----
    out_src = {}
    for s in SOURCES:
        if s not in arms:
            continue
        info = arms[s]["info_means"]
        xs = {}
        for c in m:
            if c.startswith("min("):
                grp = c[4:-1].split(",")
                xs[c] = min(_resolve(info, g) for g in grp)
            else:
                xs[c] = _resolve(info, c)
        cov = sum(m[c] * max(0.0, xs[c] - x0[c]) for c in B)
        dmg = sum(m[c] * min(0.0, xs[c] - x0[c]) for c in m)
        out_src[s] = {
            "Coverage": cov, "Damage": dmg, "NET": cov + dmg,
            "return": arms[s]["return_mean"],
            "per_step": arms[s]["info_means"].get("per_timestep_reward"),
            "delta_bottleneck": {c: xs[c] - x0[c] for c in B},
        }

    nets = {s: v["NET"] for s, v in out_src.items()}
    rank_net = [s for s, _ in sorted(nets.items(), key=lambda kv: -kv[1])]
    rank_ret = [s for s, _ in sorted(out_src.items(), key=lambda kv: -kv[1]["return"])]

    # 符号裁决（比排序更强、也更该被优先检验的预测）
    if all(v > SIGN_EPS for v in nets.values()):
        sign = "ALL_POSITIVE"
    elif all(v < -SIGN_EPS for v in nets.values()):
        sign = "ALL_NEGATIVE"
    elif all(abs(v) <= SIGN_EPS for v in nets.values()):
        sign = "ALL_NEGLIGIBLE"
    else:
        sign = "MIXED"

    spread = max(nets.values()) - min(nets.values())
    separable = spread >= SEPARATION_MIN

    return {
        "target": target, "kind": kind,
        "bottleneck_set": B,
        "bottleneck_is_generic": [c for c in B if c in GENERIC_TERMS],
        "headroom_share": {c: headroom[c] / total for c in B} if total > 0 else {},
        "sources": out_src,
        "sign_prediction": sign,
        "spread": spread,
        "rank_separable": separable,
        "rank_NET": rank_net if separable else None,
        "rank_return": rank_ret,
        "ranks_disagree": separable and rank_net != rank_ret,
        "ranks_inverted": separable and rank_net == rank_ret[::-1],
    }


def main() -> None:
    by = load_probe()
    results = {}
    for tg in sorted(SPEC):
        if tg not in by or "zero" not in by[tg]:
            continue
        results[tg] = analyze(tg, by[tg])

    print("Bottleneck-Aligned Coverage v1  （零交互，仅用 zero-shot probe + reward 源码结构）\n")
    hdr = (f"{'target':22s} {'结构':13s} {'符号预测':15s} {'NET 排序':18s} "
           f"{'return 排序':18s} {'':6s} 瓶颈分量")
    print(hdr); print("-" * (len(hdr) + 6))
    for tg, r in results.items():
        if r.get("verdict") == "UNMEASURABLE":
            print(f"{tg:22s} {r['kind']:13s} {'UNMEASURABLE':15s} "
                  f"{'（事件主导，不定序）':36s}")
            continue
        flag = "★反向" if r["ranks_inverted"] else ("不一致" if r["ranks_disagree"] else "一致")
        rk = ">".join(r["rank_NET"]) if r["rank_NET"] else "（差异过小，不定序）"
        print(f"{tg:22s} {r['kind']:13s} {r['sign_prediction']:15s} {rk:18s} "
              f"{'>'.join(r['rank_return']):18s} {flag:6s} "
              f"{','.join(r['bottleneck_set'])}")

    out = REPO / "docs/data/bottleneck_aligned_coverage_v1.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=1, ensure_ascii=False))
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
