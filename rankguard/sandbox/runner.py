"""Sandboxed invocation of attack/defense plugins.

Provides:
  - ``safe_call_attack``  : timed call + payload byte-cap (spec: 512 bytes).
  - ``safe_call_defense`` : timed call + untrusted-text preservation check
                            (exactly once) + added-text byte-cap (spec: 2048).
  - ``validate_prompt_preserves_untrusted_text``.

Timeouts use a worker thread (cross-platform). For the spec's container-sandbox
hardening the same interfaces would run in a subprocess; the validation
contract here is identical.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from typing import Callable

from ..env import AttackEnv, DefenseEnv


def _call_with_timeout(fn: Callable, *args, timeout: float | None = None):
    if not timeout or timeout <= 0:
        return fn(*args)
    with ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(fn, *args)
        try:
            return fut.result(timeout=timeout)
        except FuturesTimeout:
            return None
        except Exception:
            return None


def _truncate_utf8(s: str, max_bytes: int) -> str:
    b = s.encode("utf-8")
    if len(b) <= max_bytes:
        return s
    b = b[:max_bytes]
    # walk back to a valid UTF-8 boundary
    while b and (b[-1] & 0xC0) == 0x80:
        b = b[:-1]
    if b and (b[-1] & 0xE0) == 0xC0 and len(b) < 2:
        b = b[:-1]
    return b.decode("utf-8", errors="ignore")


def safe_call_attack(attack, env: AttackEnv, timeout: float = 10.0, max_bytes: int = 512) -> str:
    """Run ``attack.attack(env)`` with timeout; clamp payload to ``max_bytes``.

    On any failure returns ``""`` (no payload) so the match continues.
    """
    try:
        payload = _call_with_timeout(attack.attack, env, timeout=timeout)
    except Exception:  # noqa: BLE001
        return ""
    if payload is None:
        return ""
    payload = str(payload)
    return _truncate_utf8(payload, max_bytes)


def validate_prompt_preserves_untrusted_text(prompt: str, untrusted_text: str) -> tuple[bool, str]:
    """Return (ok, reason). Must contain untrusted_text exactly once."""
    if not untrusted_text:
        return True, "empty"
    count = prompt.count(untrusted_text)
    if count == 0:
        return False, "untrusted_text_missing"
    if count > 1:
        return False, f"untrusted_text_duplicated:{count}"
    return True, "ok"


def safe_call_defense(
    defense,
    env: DefenseEnv,
    timeout: float = 10.0,
    max_added_bytes: int = 2048,
) -> tuple[str, list[str]]:
    """Run ``defense.defend(env)`` with validation.

    Returns (final_prompt, warnings). On any violation, falls back to
    ``env.base_prompt`` (NoDefense) and records a warning.
    """
    base = env.base_prompt
    try:
        prompt = _call_with_timeout(defense.defend, env, timeout=timeout)
    except Exception:  # noqa: BLE001
        return base, ["defense_call_failed"]

    if not isinstance(prompt, str) or not prompt:
        return base, ["defense_non_string"]

    ok, reason = validate_prompt_preserves_untrusted_text(prompt, env.untrusted_text)
    if not ok:
        return base, [reason]

    added = len(prompt.encode("utf-8")) - len(base.encode("utf-8"))
    if added > max_added_bytes:
        return base, [f"added_bytes_exceeded:{added}>{max_added_bytes}"]

    return prompt, []
