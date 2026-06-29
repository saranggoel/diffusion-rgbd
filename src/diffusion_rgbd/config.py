import copy
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}
    if overrides:
        cfg = deep_update(cfg, overrides)
    return cfg


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_nested(cfg: dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    value: Any = cfg
    for part in dotted_key.split("."):
        if not isinstance(value, dict) or part not in value:
            return default
        value = value[part]
    return value
