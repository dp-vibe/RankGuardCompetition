"""Configuration loading.

A config is a plain dict loaded from `config.yaml` (by default) and
optionally overridden by environment variables of the form
``RANKGUARD_SECTION__KEY`` (double-underscore nesting). Values are also
coerced to ints/floats/bools where possible so YAML defaults stay readable.
"""

from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


def _coerce(value: str) -> Any:
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def _apply_env_overrides(cfg: dict[str, Any]) -> dict[str, Any]:
    prefix = "RANKGUARD_"
    for key, raw in os.environ.items():
        if not key.startswith(prefix):
            continue
        path = key[len(prefix):].split("__")
        if len(path) < 2:
            continue
        node = cfg
        for part in path[:-1]:
            part_lower = part.lower()
            if part_lower not in node or not isinstance(node[part_lower], dict):
                node[part_lower] = {}
            node = node[part_lower]
        node[path[-1].lower()] = _coerce(raw)
    return cfg


def load_config(path: str | os.PathLike | None = None) -> dict[str, Any]:
    """Load YAML config (default: repo-root ``config.yaml``) + env overrides."""
    cfg_path = Path(path) if path else _DEFAULT_CONFIG_PATH
    if cfg_path.exists():
        with cfg_path.open("r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh) or {}
    else:
        cfg = {}
    return _apply_env_overrides(cfg)


def get(key_path: str, default: Any = None, cfg: dict[str, Any] | None = None) -> Any:
    """Dotted lookup: get('judge.model', cfg=...)."""
    node = cfg if cfg is not None else load_config()
    for part in key_path.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node
