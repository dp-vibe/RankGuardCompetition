"""Shared plugin-loading utilities.

Attacks and defenses are intentionally kept in *separate* packages
(``rankguard.attacks`` / ``rankguard.defenses``) with their own registries, so
the two sides are fully decoupled: an attack never imports a defense and vice
versa, and the arbiter talks to each through its own registry. New algorithms
are added by dropping a module into the relevant package (auto-discovered) or
by pointing the loader at an external ``attack.py`` / ``defense.py`` file
(competitor submission format, spec section 九).
"""

from __future__ import annotations

import importlib
import importlib.util
import pkgutil
from pathlib import Path
from types import ModuleType


def load_module_from_file(path: str | Path, module_name: str | None = None) -> ModuleType:
    """Import an arbitrary .py file as a module (for user submissions)."""
    path = Path(path)
    name = module_name or f"_rg_user_{path.stem}_{abs(hash(str(path)))}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def discover_package(pkg: ModuleType) -> list[ModuleType]:
    """Import every submodule of ``pkg`` so decorators can register them.

    Skips names starting with ``_`` (private/dunder).
    """
    found: list[ModuleType] = []
    for info in pkgutil.iter_modules(pkg.__path__):
        if info.name.startswith("_"):
            continue
        found.append(importlib.import_module(f"{pkg.__name__}.{info.name}"))
    return found
