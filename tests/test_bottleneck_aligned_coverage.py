"""BAC 指标的聚焦单元测试。

被测的是三件容易悄悄错、且一旦错就会让论文结论反向的事：
    1. 边际敏感度 m_c 的形式随组合算子改变（加性=权重，乘性=其他因子之积）
    2. min 组必须按最小分量参与，否则"抬高一个、压垮另一个"会被算成改善
    3. 无界任务必须判 UNMEASURABLE，不得给出排序
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts/analysis"))

from bottleneck_aligned_coverage_v1 import analyze, _resolve  # noqa: E402
from configs.reward_structure.humanoidbench_v1 import SPEC, UNBOUNDED  # noqa: E402


def _arms(target: str, per_source: dict) -> dict:
    return {s: {"info_means": info, "return_mean": ret}
            for s, (info, ret) in per_source.items()}


def test_multiplicative_sensitivity_is_product_of_others() -> None:
    """乘性结构下 m_c 必须是其余因子之积——否则低值因子的否决权会被低估。"""
    base = {"small_control": 1.0, "stand_reward": 0.5, "move": 0.2}
    arms = _arms("h1hand-stair-v0", {
        "zero": (base, 10.0),
        "stand": ({**base, "move": 0.2}, 10.0),
        # 只抬 move 0.2→0.4，收益应为 m_move*0.2 = (1.0*0.5)*0.2 = 0.1
        "walk": ({**base, "move": 0.4}, 10.0),
        "run": (base, 10.0),
    })
    r = analyze("h1hand-stair-v0", arms)
    assert abs(r["sources"]["walk"]["Coverage"] - 0.1) < 1e-9, r["sources"]["walk"]


def test_min_group_uses_the_smaller_component() -> None:
    """crawl 的真实反例：抬高 crawling、压垮 crawling_head，min 必须变差。"""
    zero = {"small_control": 0.987, "move": 0.168, "in_tunnel": 1.0,
            "crawling": 0.544, "crawling_head": 0.526}
    stand = {**zero, "crawling": 0.845, "crawling_head": 0.343}
    r = analyze("h1hand-crawl-v0", _arms("h1hand-crawl-v0", {
        "zero": (zero, 149.3), "stand": (stand, 151.5),
        "walk": (zero, 142.4), "run": (zero, 132.1)}))
    # min 从 0.526 掉到 0.343，必须记为 Damage，而不是因 crawling 上升而记为改善
    assert r["sources"]["stand"]["Damage"] < 0, r["sources"]["stand"]
    assert r["sources"]["stand"]["NET"] < r["sources"]["walk"]["NET"]


def test_unbounded_targets_refuse_to_rank() -> None:
    """事件主导的任务必须拒绝定序，而不是给一个由罕见事件驱动的假排序。"""
    for tg, spec in SPEC.items():
        if spec["kind"] != UNBOUNDED:
            continue
        r = analyze(tg, _arms(tg, {"zero": ({}, 0.0)}))
        assert r["verdict"] == "UNMEASURABLE", tg
        assert "rank_NET" not in r, tg


def test_composite_term_resolves_as_product() -> None:
    """'a*b' 形式的复合项（源码里 stand_reward*small_control 常作一项）。"""
    assert abs(_resolve({"a": 0.5, "b": 0.4}, "a*b") - 0.2) < 1e-12
    assert _resolve({"a": 0.5}, "missing") == 0.0


def test_bounded_specs_declare_no_direction_flip() -> None:
    """有界任务里不应出现"越小越好"的分量。

    代码在计算 Δ 时一律按"越大越好"处理；若某个有界任务引入了距离类分量而
    未在此处暴露，符号会静默反向。此测试是那道防线。
    """
    for tg, spec in SPEC.items():
        if spec["kind"] == UNBOUNDED:
            continue
        assert not spec.get("direction"), f"{tg} 声明了 direction 但代码未按方向翻转"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
