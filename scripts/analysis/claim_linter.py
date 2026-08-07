#!/usr/bin/env python3
"""Claim linter：阻止已撤回的过强措辞重新进入**活跃投稿文件**（R0.1 B2）。

背景：`docs/PAPER_CLAIMS_20260804.md` §5 列了 11 条禁写，R0（`a2fc173`）
把相应 claim 降了级——但 `PAPER_DRAFT` / `PAPER_OUTLINE` 的标题、摘要、
贡献列表当时**没有同步整改**，禁写措辞在正文里又活了三天。
文档里写"不许写"不构成执行；本脚本是执行点。

用法::

    python scripts/analysis/claim_linter.py            # 检查，违规则非零退出
    python scripts/analysis/claim_linter.py --list     # 列出规则

**只检查活跃投稿文件**（`ACTIVE_SUBMISSION_FILES`）。历史文档保留原貌，
但必须在文件头部带 HISTORICAL / PROVISIONAL 标记——这一条也由本脚本检查。
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

#: 活跃投稿文件——这些是要投出去的东西，禁写措辞一律不得出现。
ACTIVE_SUBMISSION_FILES = (
    "docs/PAPER_DRAFT_20260806.md",
    "docs/PAPER_OUTLINE_20260804.md",
)

#: 历史文档：保留原貌不改，但必须自带 HISTORICAL / PROVISIONAL 标记，
#: 否则它们会被当成当前有效结论引用。
HISTORICAL_FILES = (
    "docs/impossibility_characterization_of_transfer_prediction_20260730.md",
)

HISTORICAL_MARKERS = ("HISTORICAL", "PROVISIONAL", "已撤回", "已降级")

#: 每条规则 = (标识, 正则, 为什么禁, 正确写法)。
#: 正则在**去掉豁免行**之后的正文上匹配（见 `_strip_exempt`）。
RULES = [
    (
        "impossibility",
        re.compile(r"\bimpossibility\b", re.IGNORECASE),
        "PAPER_CLAIMS §5 第 8 条：无形式证明不得用一般意义的 impossibility 措辞",
        "empirical systematic failure of tested proxy families",
    ),
    (
        "minimum-sufficient-measurement",
        re.compile(r"minimum\s+(sufficient\s+)?measurement|minimum\s+measurement\s+that\s+suffices",
                   re.IGNORECASE),
        "PAPER_CLAIMS §5 第 9 条：不得称 K*=10000 是 minimum sufficient measurement",
        "smallest robust horizon among the budgets we tested ({2k,5k,10k})",
    ),
    (
        "full-static-spec-unpredictable",
        re.compile(r"(any|no)\s+function\s+of\s+the\s+static\s+task\s+specification"
                   r"|static\s+task\s+specification\s+receives\s+identical",
                   re.IGNORECASE),
        "PAPER_CLAIMS §5 I-2：反例只反驳'只读 reward 规格'的预测器；"
        "slide/stair 的地形几何、dynamics、初始分布、MJCF 均不同",
        "any predictor that reads only the reward specification",
    ),
    # ── 中文规则 ────────────────────────────────────────────────────
    # PAPER_OUTLINE 是中文文档。首版 linter 只写了英文模式，
    # 结果"不可能性刻画""最小充分测量""静态规格"三处全部漏网——
    # 规则语言必须覆盖被检查文件的实际语言。
    (
        "impossibility-zh",
        re.compile(r"不可能性(刻画|定理|结果)|证明了.{0,6}不可能"),
        "同 impossibility：无形式证明不得作一般意义的不可能性主张",
        "十二族已测代理信号的经验性系统失败",
    ),
    (
        "minimum-measurement-zh",
        re.compile(r"最小充分测量|最小(必要)?测量代价|最小的?充分"),
        "同 minimum-sufficient-measurement：K*=10000 只是已测预算中最小的稳健 horizon",
        "已测试预算 {2k,5k,10k} 中最小的稳健 horizon",
    ),
    (
        "full-static-spec-zh",
        re.compile(r"(任何)?只读.{0,24}静态规格.{0,40}(完全一样|相同|无法区分)"
                   r"|静态规格.{0,20}(无法预测|不可预测)"),
        "同 full-static-spec-unpredictable：反例只约束'只读 reward 规格'的预测器",
        "任何只读 reward 规格的量",
    ),
    (
        "unrevalidated-saturation-pct",
        re.compile(r"\b(98\.5|95\.1|85\.1)\s*%|\bof\s+theory\s+max\b", re.IGNORECASE),
        "PAPER_CLAIMS §5 第 10 条：evaluator v2 重评完成前不得引用任何饱和度百分比",
        "（暂不引用；重评后再议）",
    ),
]

#: 豁免标记：写在同一行或紧邻上一行时，该行不参与匹配。
#: 用于"我们**不**声称 impossibility"这类自我限制性表述，
#: 以及整改说明本身——否则 linter 会禁止解释为什么禁止。
EXEMPT_MARK = "claim-linter: allow"

#: 自我限制性表述的白名单模式（无需手工加豁免标记）。
SELF_LIMITING = (
    re.compile(r"not\s+an?\s+impossibility", re.IGNORECASE),
    re.compile(r"do\s+\*\*not\*\*\s+claim\s+an?\s+impossibility", re.IGNORECASE),
    re.compile(r"empirical,?\s+not\s+a\s+formal\s+impossibility", re.IGNORECASE),
    re.compile(r"不得使用.*impossibility", re.IGNORECASE),
    re.compile(r"不得称.*minimum\s+sufficient", re.IGNORECASE),
    re.compile(r"原标题|整改|已撤回|已降级|违反", re.IGNORECASE),
)


def _is_exempt(line: str, prev: str) -> bool:
    if EXEMPT_MARK in line or EXEMPT_MARK in prev:
        return True
    return any(p.search(line) for p in SELF_LIMITING)


def _in_comment_block(lines: list[str], idx: int) -> bool:
    """HTML 注释块（<!-- ... -->）内的行不参与匹配——整改说明写在那里。"""
    depth = 0
    for i in range(idx + 1):
        depth += lines[i].count("<!--") - lines[i].count("-->")
    return depth > 0


def lint_file(rel: str) -> list[tuple]:
    path = REPO / rel
    if not path.exists():
        return [(rel, 0, "MISSING-FILE", f"活跃投稿文件不存在：{rel}", "")]
    lines = path.read_text(encoding="utf-8").splitlines()
    hits = []
    for i, line in enumerate(lines):
        prev = lines[i - 1] if i else ""
        if _is_exempt(line, prev) or _in_comment_block(lines, i):
            continue
        for name, pattern, why, correct in RULES:
            if pattern.search(line):
                hits.append((rel, i + 1, name, why, line.strip()[:110]))
    return hits


def lint_historical() -> list[tuple]:
    """历史文档必须自带 HISTORICAL / PROVISIONAL 标记（检查前 40 行）。"""
    hits = []
    for rel in HISTORICAL_FILES:
        path = REPO / rel
        if not path.exists():
            continue
        head = "\n".join(path.read_text(encoding="utf-8").splitlines()[:40])
        if not any(m in head for m in HISTORICAL_MARKERS):
            hits.append((rel, 1, "missing-historical-marker",
                         "历史文档必须在头部标 HISTORICAL / PROVISIONAL，"
                         "否则会被当成当前有效结论引用",
                         "（文件头部前 40 行内无标记）"))
    return hits


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--list", action="store_true", help="列出规则后退出")
    ap.add_argument("--files", nargs="*", default=None, help="覆盖被检查的文件列表")
    args = ap.parse_args()

    if args.list:
        for name, pattern, why, correct in RULES:
            print(f"[{name}]\n  禁: {pattern.pattern}\n  因: {why}\n  改: {correct}\n")
        return 0

    targets = args.files if args.files is not None else list(ACTIVE_SUBMISSION_FILES)
    hits = []
    for rel in targets:
        hits.extend(lint_file(rel))
    hits.extend(lint_historical())

    if not hits:
        print(f"claim linter: OK（{len(targets)} 个活跃文件 + "
              f"{len(HISTORICAL_FILES)} 个历史文件）")
        return 0

    print(f"claim linter: {len(hits)} 处违规\n")
    for rel, line_no, name, why, text in hits:
        print(f"{rel}:{line_no}  [{name}]")
        print(f"    {text}")
        print(f"    因: {why}\n")
    print("若确属自我限制性表述或整改说明，在该行或上一行加注释标记："
          f" {EXEMPT_MARK}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
