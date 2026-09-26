"""Attack plugin base + registry.

An *attack* turns an :class:`~rankguard.env.AttackEnv` into a payload string
written to the ``[INJECT]`` position (spec: 攻击方任务). Attacks are
self-contained, stateless, and may only use the standard library in
competition submissions.
"""

from __future__ import annotations

import importlib
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable

from ..env import AttackEnv
from ..plugins import discover_package, load_module_from_file


class BaseAttack(ABC):
    """Subclass and set ``name``; implement :meth:`attack`."""

    name: str = ""
    description: str = ""

    @abstractmethod
    def attack(self, env: AttackEnv) -> str:
        """Return the payload string for ``env`` (<= payload_max_bytes)."""
        raise NotImplementedError

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Attack {self.name}>"


class FunctionAttack(BaseAttack):
    """Adapts a plain ``attack(env) -> str`` function into BaseAttack."""

    def __init__(self, name: str, fn: Callable[[AttackEnv], str], description: str = ""):
        if not callable(fn):
            raise TypeError("attack function must be callable")
        self.name = name
        self.fn = fn
        self.description = description or (fn.__doc__ or "").strip()

    def attack(self, env: AttackEnv) -> str:
        return self.fn(env)


class AttackRegistry:
    _registry: dict[str, type[BaseAttack]] = {}

    @classmethod
    def register(cls, attack_cls: type[BaseAttack]) -> type[BaseAttack]:
        if not attack_cls.name:
            raise ValueError(f"{attack_cls.__name__} must define a non-empty `name`")
        cls._registry[attack_cls.name] = attack_cls
        return attack_cls

    @classmethod
    def all(cls) -> dict[str, BaseAttack]:
        return {name: klass() for name, klass in cls._registry.items()}

    @classmethod
    def names(cls) -> list[str]:
        return sorted(cls._registry)

    @classmethod
    def get(cls, name: str) -> BaseAttack:
        return cls._registry[name]()

    @classmethod
    def register_function(cls, name: str, fn: Callable[[AttackEnv], str]) -> FunctionAttack:
        fa = FunctionAttack(name, fn)
        cls._registry[name] = fa.__class__
        return fa

    @classmethod
    def load_from_file(cls, path: str | Path, name: str | None = None) -> FunctionAttack:
        mod = load_module_from_file(path)
        if not hasattr(mod, "attack"):
            raise AttributeError(f"{path} must define `attack(env) -> str`")
        nm = name or getattr(mod, "ATTACK_NAME", None) or Path(path).stem
        return cls.register_function(nm, mod.attack)


def discover_attacks() -> dict[str, BaseAttack]:
    """Import all attack modules so the registry is populated, then return it."""
    pkg = importlib.import_module("rankguard.attacks")
    discover_package(pkg)
    return AttackRegistry.all()
