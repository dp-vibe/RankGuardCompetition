"""Defense plugin base + registry.

A *defense* rewrites the final prompt sent to the judge given a
:class:`~rankguard.env.DefenseEnv` (spec: 防御方任务). Defenses must preserve
the untrusted text verbatim (one occurrence), not call external services, and
may only use the standard library in competition submissions.
"""

from __future__ import annotations

import importlib
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable

from ..env import DefenseEnv
from ..plugins import discover_package, load_module_from_file


class BaseDefense(ABC):
    """Subclass and set ``name``; implement :meth:`defend`."""

    name: str = ""
    description: str = ""

    @abstractmethod
    def defend(self, env: DefenseEnv) -> str:
        """Return the protected prompt sent to the judge."""
        raise NotImplementedError

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Defense {self.name}>"


class FunctionDefense(BaseDefense):
    """Adapts a plain ``defend(env) -> str`` function into BaseDefense."""

    def __init__(self, name: str, fn: Callable[[DefenseEnv], str], description: str = ""):
        if not callable(fn):
            raise TypeError("defense function must be callable")
        self.name = name
        self.fn = fn
        self.description = description or (fn.__doc__ or "").strip()

    def defend(self, env: DefenseEnv) -> str:
        return self.fn(env)


class DefenseRegistry:
    _registry: dict[str, type[BaseDefense]] = {}

    @classmethod
    def register(cls, defense_cls: type[BaseDefense]) -> type[BaseDefense]:
        if not defense_cls.name:
            raise ValueError(f"{defense_cls.__name__} must define a non-empty `name`")
        cls._registry[defense_cls.name] = defense_cls
        return defense_cls

    @classmethod
    def all(cls) -> dict[str, BaseDefense]:
        return {name: klass() for name, klass in cls._registry.items()}

    @classmethod
    def names(cls) -> list[str]:
        return sorted(cls._registry)

    @classmethod
    def get(cls, name: str) -> BaseDefense:
        return cls._registry[name]()

    @classmethod
    def register_function(cls, name: str, fn: Callable[[DefenseEnv], str]) -> FunctionDefense:
        fd = FunctionDefense(name, fn)
        cls._registry[name] = fd.__class__
        return fd

    @classmethod
    def load_from_file(cls, path: str | Path, name: str | None = None) -> FunctionDefense:
        mod = load_module_from_file(path)
        if not hasattr(mod, "defend"):
            raise AttributeError(f"{path} must define `defend(env) -> str`")
        nm = name or getattr(mod, "DEFENSE_NAME", None) or Path(path).stem
        return cls.register_function(nm, mod.defend)


def discover_defenses() -> dict[str, BaseDefense]:
    """Import all defense modules so the registry is populated, then return it."""
    pkg = importlib.import_module("rankguard.defenses")
    discover_package(pkg)
    return DefenseRegistry.all()
