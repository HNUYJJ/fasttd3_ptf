"""Helpers for the official FastTD3 source copy.

The official_code FastTD3 code under ``fasttd3_ptf/official_code/FastTD3/fast_td3`` is intentionally
kept byte-for-byte compatible with the upstream source snapshot. PTF integration
code should live outside that tree and import or execute it through helpers in
this package.
"""

from fasttd3_ptf.official_fasttd3_ptf.paths import (
    OFFICIAL_CODE_FASTTD3_ROOT,
    OFFICIAL_CODE_HUMANOID_BENCH_ROOT,
    OFFICIAL_CODE_ROOT,
    ensure_fasttd3_import_path,
    ensure_humanoidbench_import_path,
    get_fasttd3_source_root,
    get_humanoidbench_source_roots,
)

__all__ = [
    "OFFICIAL_CODE_FASTTD3_ROOT",
    "OFFICIAL_CODE_HUMANOID_BENCH_ROOT",
    "OFFICIAL_CODE_ROOT",
    "ensure_fasttd3_import_path",
    "ensure_humanoidbench_import_path",
    "get_fasttd3_source_root",
    "get_humanoidbench_source_roots",
]
