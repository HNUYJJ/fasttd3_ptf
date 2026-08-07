#!/usr/bin/env python3
"""由 checkpoint 生成 formal evaluator 所需的 identity manifest。

补的是 `evaluator_v21c_results_20260807.md` §8.5 记的缺口：
manifest 此前没有生成器，须手工写——那会让 formal 模式在实践中
因为"太麻烦"被绕开，护栏形同虚设。

字段与 `p0_evaluator_v2.verify_checkpoint_identity` 的 formal 要求一一对应::

    checkpoint_sha256 / env_name / learner_seed / global_step   必需
    training_protocol_digest                                     checkpoint 有 ptf_cfg 时必需

**digest 口径与 evaluator 一致**（两者都是
``sha256(json.dumps(ptf_cfg, sort_keys=True, default=str, ensure_ascii=False))``），
否则 manifest 会对不上——已由 `--verify` 实际比对确认。

用法::

    python scripts/analysis/emit_identity_manifest.py --checkpoint X --out M.json
    python scripts/analysis/emit_identity_manifest.py --checkpoint X --verify   # 只校验不写
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))


def build_manifest(checkpoint: Path) -> dict:
    import torch

    import p0_evaluator_v2 as ev

    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    args = state.get("args") or {}
    ptf_cfg = state.get("ptf_cfg") or {}

    missing = [k for k, v in (("env_name", args.get("env_name")),
                              ("seed", args.get("seed")),
                              ("global_step", state.get("global_step")))
               if v is None]
    if missing:
        raise SystemExit(
            f"checkpoint 内缺 {missing}，无法生成 formal manifest。"
            f"这类文件只能走 --identity-mode debug（产物不得用于科学裁决）")

    manifest = {
        "_generated_by": "scripts/analysis/emit_identity_manifest.py",
        "_checkpoint_path": str(checkpoint.relative_to(REPO))
                            if checkpoint.is_absolute() and str(checkpoint).startswith(str(REPO))
                            else str(checkpoint),
        "checkpoint_sha256": ev._sha256(checkpoint),
        "env_name": args["env_name"],
        "learner_seed": args["seed"],
        "global_step": state["global_step"],
    }
    if ptf_cfg:
        manifest["training_protocol_digest"] = ev._digest_obj(ptf_cfg)
    del state
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--verify", action="store_true",
                    help="生成后用 evaluator 的 formal 校验实跑一遍，确认能通过")
    args = ap.parse_args()

    ckpt = Path(args.checkpoint)
    if not ckpt.is_absolute():
        ckpt = REPO / ckpt
    if not ckpt.exists():
        raise SystemExit(f"checkpoint 不存在：{ckpt}")

    manifest = build_manifest(ckpt)
    blob = json.dumps(manifest, ensure_ascii=False, indent=2)

    if args.verify:
        import p0_evaluator_v2 as ev

        info = ev.verify_checkpoint_identity(
            str(ckpt), manifest["env_name"], identity_mode="formal", manifest=manifest)
        checked = info["manifest_checked_fields"]
        need = {"checkpoint_sha256", "env_name", "learner_seed", "global_step"}
        if not need <= set(checked):
            raise SystemExit(f"formal 校验未覆盖全部必需字段：实得 {checked}")
        print(f"formal 校验通过；已核对字段：{checked}")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".tmp")
        tmp.write_text(blob, encoding="utf-8")
        os.replace(tmp, out)
        print(f"wrote {out}")
    else:
        print(blob)
    return 0


if __name__ == "__main__":
    sys.exit(main())
