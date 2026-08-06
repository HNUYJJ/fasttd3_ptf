from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data


def save_yaml(data: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def recursive_update(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    for k, v in update.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            recursive_update(base[k], v)
        else:
            base[k] = v
    return base


def set_by_dotted_key(cfg: dict[str, Any], dotted_key: str, value: Any) -> None:
    cur = cfg
    keys = dotted_key.split(".")
    for key in keys[:-1]:
        if key not in cur or not isinstance(cur[key], dict):
            cur[key] = {}
        cur = cur[key]
    cur[keys[-1]] = value


def parse_overrides(overrides: list[str] | None) -> dict[str, Any]:
    cfg: dict[str, Any] = {}
    if not overrides:
        return cfg
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must be key=value, got: {item}")
        key, raw = item.split("=", 1)
        try:
            value = ast.literal_eval(raw)
        except Exception:
            lowered = raw.lower()
            if lowered == "true":
                value = True
            elif lowered == "false":
                value = False
            elif lowered == "none" or lowered == "null":
                value = None
            else:
                value = raw
        set_by_dotted_key(cfg, key, value)
    return cfg


def _load_with_defaults(path: str | Path, seen: set[Path] | None = None) -> dict[str, Any]:
    path = Path(path).resolve()
    seen = set() if seen is None else seen
    if path in seen:
        raise ValueError(f"Recursive config defaults detected at {path}")
    seen.add(path)
    cfg = load_yaml(path)
    if "defaults" not in cfg:
        return cfg
    defaults = cfg.pop("defaults") or []
    merged: dict[str, Any] = {}
    for rel in defaults:
        default_path = (path.parent / rel).resolve()
        recursive_update(merged, _load_with_defaults(default_path, seen=seen))
    recursive_update(merged, cfg)
    return merged


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict[str, Any]:
    cfg = _load_with_defaults(path)
    recursive_update(cfg, parse_overrides(overrides))
    return cfg


def get_cfg(cfg: dict[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = cfg
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur
