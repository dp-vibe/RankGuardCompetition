"""Strict JSON parsing & deterministic, non-LLM repair (spec section 五.6).

Repair steps (each is a fixed, non-LLM transform):
  1. Strip markdown code fences (```json ... ```).
  2. Locate the first *complete* JSON object via brace matching (so trailing
     judge chatter is ignored).
  3. Convert numeric strings to ints.
  4. Coerce missing/malformed fields to defaults and clamp to the scale.

If repair still cannot yield a valid object, ``parse_or_repair`` returns
``None`` so the arbiter can substitute the reference score (spec 五.6.1).
"""

from __future__ import annotations

import json
import re
from typing import Any

from ..schema import DIMENSIONS, Score

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def _strip_fences(text: str) -> str:
    m = _FENCE_RE.search(text)
    if m:
        return m.group(1).strip()
    return text.strip()


def _first_json_object(text: str) -> str | None:
    """Return the substring of the first balanced {...} object, else None."""
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None  # unbalanced


def _coerce_int(v: Any, default: int = 0) -> int:
    if isinstance(v, bool):  # bool is an int subclass; reject it explicitly
        return default
    if isinstance(v, (int, float)):
        return int(v)
    if isinstance(v, str):
        s = v.strip()
        # tolerate "16/20" style or trailing units
        s = re.split(r"[/\s]", s, 1)[0]
        try:
            return int(float(s))
        except ValueError:
            return default
    return default


def _normalize(obj: dict[str, Any]) -> Score:
    score = Score(
        writing_quality=_coerce_int(obj.get("writing_quality")),
        coherence=_coerce_int(obj.get("coherence")),
        informativeness=_coerce_int(obj.get("informativeness")),
        educational_value=_coerce_int(obj.get("educational_value")),
        trustworthiness=_coerce_int(obj.get("trustworthiness")),
        overall_score=_coerce_int(obj.get("overall_score"), 0),
        reason=str(obj.get("reason", ""))[:80],
    )
    return score.clamp()


def parse_or_repair(raw: str) -> Score | None:
    """Parse a judge response into a Score, or return None if unrepairable."""
    if not raw or not raw.strip():
        return None

    text = _strip_fences(raw)

    # Fast path: whole thing is JSON.
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return _normalize(obj)
    except json.JSONDecodeError:
        pass

    # Slow path: extract first balanced object.
    snippet = _first_json_object(text)
    if snippet is None:
        return None
    try:
        obj = json.loads(snippet)
    except json.JSONDecodeError:
        # Last resort: try to relax quotes / trailing commas.
        relaxed = re.sub(r",\s*}", "}", snippet)
        relaxed = re.sub(r",\s*]", "]", relaxed)
        try:
            obj = json.loads(relaxed)
        except json.JSONDecodeError:
            return None
    if isinstance(obj, dict):
        return _normalize(obj)
    return None


def is_valid(score: Score | None) -> bool:
    return score is not None and score.valid
