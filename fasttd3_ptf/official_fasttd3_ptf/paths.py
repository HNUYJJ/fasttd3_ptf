from __future__ import annotations

import sys
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_CODE_ROOT = PACKAGE_ROOT / "official_code"

# Project-local official_code copy. Keep this tree source-compatible with upstream;
# put PTF wrappers and patches beside it, not inside it.
OFFICIAL_CODE_FASTTD3_ROOT = OFFICIAL_CODE_ROOT / "FastTD3"

# Minimal official_code HumanoidBench runtime copy. It contains the importable
# ``humanoid_bench`` package plus the small top-level ``data`` folder used by
# optional reaching-policy wrappers.
OFFICIAL_CODE_HUMANOID_BENCH_ROOT = OFFICIAL_CODE_ROOT / "humanoid-bench"


def get_fasttd3_source_root() -> Path:
    """Return the FastTD3 source root used by official-based entry points."""

    if (OFFICIAL_CODE_FASTTD3_ROOT / "fast_td3" / "train.py").is_file():
        return OFFICIAL_CODE_FASTTD3_ROOT
    raise FileNotFoundError(
        "Could not find official FastTD3 source. Expected "
        f"{OFFICIAL_CODE_FASTTD3_ROOT}."
    )


def ensure_fasttd3_import_path() -> Path:
    """Put the official FastTD3 flat-module directory on ``sys.path``.

    Upstream ``train.py`` imports modules such as ``fast_td3_utils`` and
    ``hyperparams`` as flat files from the ``fast_td3`` directory. Keeping that
    import style intact avoids rewriting the upstream source tree.
    """

    root = get_fasttd3_source_root()
    module_dir = root / "fast_td3"
    module_dir_s = str(module_dir)
    if module_dir_s not in sys.path:
        sys.path.insert(0, module_dir_s)
    return root


def get_humanoidbench_source_roots() -> list[Path]:
    """Candidate roots that contain an importable ``humanoid_bench`` package."""

    candidates: list[Path] = [OFFICIAL_CODE_HUMANOID_BENCH_ROOT]
    out: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        root = candidate.resolve()
        if root in seen:
            continue
        seen.add(root)
        if (root / "humanoid_bench" / "__init__.py").is_file():
            out.append(root)
    return out


def ensure_humanoidbench_import_path() -> Path:
    roots = get_humanoidbench_source_roots()
    if not roots:
        raise FileNotFoundError(
            "Could not find an importable humanoid_bench package. Expected "
            f"{OFFICIAL_CODE_HUMANOID_BENCH_ROOT}."
        )
    root = roots[0]
    root_s = str(root)
    if root_s not in sys.path:
        # Keep the official flat FastTD3 module directory ahead of auxiliary
        # environment roots. A direct upstream run has ``fast_td3/`` as
        # sys.path[0]; preserving that order removes one avoidable difference
        # between wrapper-based and source-tree execution.
        insert_at = 0
        for idx, entry in enumerate(sys.path):
            if Path(entry).resolve() in {
                (OFFICIAL_CODE_FASTTD3_ROOT / "fast_td3").resolve(),
            }:
                insert_at = idx + 1
                break
        sys.path.insert(insert_at, root_s)
    return root
