"""Environment objects handed to attack / defense algorithms.

These are intentionally simple attribute containers backed only by the standard
library, because competitor submissions are restricted to the standard library
(spec section 九.三). They must never expose reference scores, reference
rankings, or opponent internals.
"""

from __future__ import annotations

from types import SimpleNamespace


class AttackEnv(SimpleNamespace):
    """Argument bundle passed to ``attack(env)`` (spec: 攻击接口).

    Attributes:
        sample_id:     id of the current sample.
        clean_text:    the original legitimate text (no attack content).
        domain:        text domain, e.g. ``"computer_science"``.
        language:      text language, e.g. ``"en"`` / ``"zh"``.
        rubric_instruction: the trusted scoring rubric in force.
        rubric_id:     id of the rubric (informational).
        inject_position: where the payload is placed (informational).
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __repr__(self) -> str:  # pragma: no cover - debugging only
        keys = ("sample_id", "domain", "language")
        kv = ", ".join(f"{k}={getattr(self, k, '?')!r}" for k in keys)
        return f"AttackEnv({kv})"


class DefenseEnv(SimpleNamespace):
    """Argument bundle passed to ``defend(env)`` (spec: 防御接口).

    Attributes:
        rubric_instruction: the trusted scoring rubric.
        untrusted_text:     clean_text + attack payload (the full untrusted text).
        base_prompt:        the prompt that would be sent to the judge with no defense.
        output_schema:      the fixed JSON output structure description.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __repr__(self) -> str:  # pragma: no cover - debugging only
        return f"DefenseEnv(base_prompt_len={len(getattr(self, 'base_prompt', ''))})"
