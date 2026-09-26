"""OpenAI-compatible Judge client.

The judge is the single fixed LLM used to score every sample (spec 四.3).
Access goes through the **standard OpenAI chat-completions interface** so any
OpenAI-compatible server works: OpenAI itself, vLLM ``--serve``, Ollama's
``/v1``, LM Studio, LocalAI, etc.

Determinism & cost:
  - ``temperature=0`` and a fixed ``seed`` (spec 四.3).
  - Responses are cached on disk keyed by (model, system, user) hash, so a
    re-run of the same (attack, defense) cell is free and bit-identical.

A ``MockJudge`` is provided for offline / test runs without a live model: it
derives a deterministic score from text features and is deliberately
injectable (responds to "score of N" cues), so the attack-defense pipeline can
be exercised end to end.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Protocol

from ..config import load_config
from ..prompt.builder import build_system_prompt, INJECT_MARKER

CACHE_INDEX = "judge_cache.json"


class Judge(Protocol):
    """Anything with a ``query(prompt) -> raw_text`` method is a judge."""

    model: str

    def query(self, prompt: str) -> str: ...


# ---------------------------------------------------------------------------


class _DiskCache:
    """Append-only JSON cache: {digest: {raw, ts}}."""

    def __init__(self, cache_dir: str | os.PathLike | None):
        self.enabled = bool(cache_dir)
        self.dir = Path(cache_dir) if cache_dir else None
        if self.enabled:
            self.dir.mkdir(parents=True, exist_ok=True)
            self.index_path = self.dir / CACHE_INDEX
            self._index: dict[str, dict[str, Any]] = {}
            if self.index_path.exists():
                try:
                    self._index = json.loads(self.index_path.read_text("utf-8"))
                except (json.JSONDecodeError, OSError):
                    self._index = {}
        else:
            self._index = {}

    def _digest(self, key: str) -> str:
        return hashlib.sha256(key.encode("utf-8")).hexdigest()

    def get(self, key: str) -> str | None:
        if not self.enabled:
            return None
        entry = self._index.get(self._digest(key))
        if not entry:
            return None
        path = self.dir / entry["file"]
        if not path.exists():
            return None
        try:
            return path.read_text("utf-8")
        except OSError:
            return None

    def put(self, key: str, value: str) -> None:
        if not self.enabled:
            return
        digest = self._digest(key)
        file = f"{digest}.txt"
        (self.dir / file).write_text(value, "utf-8")
        self._index[digest] = {"file": file, "ts": time.time()}
        try:
            self.index_path.write_text(json.dumps(self._index), "utf-8")
        except OSError:
            pass


class OpenAIJudge:
    """Calls an OpenAI-compatible chat-completions endpoint."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        temperature: float = 0.0,
        seed: int = 42,
        max_tokens: int = 512,
        timeout: float = 60.0,
        max_retries: int = 3,
        cache_dir: str | os.PathLike | None = None,
        system_prompt: str | None = None,
    ):
        # Imported lazily so the package imports without `openai` installed.
        from openai import OpenAI

        self.model = model
        self.temperature = temperature
        self.seed = seed
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.system_prompt = system_prompt or build_system_prompt()
        self._cache = _DiskCache(cache_dir)
        self._client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)

    @classmethod
    def from_config(cls, cfg: dict[str, Any] | None = None) -> "OpenAIJudge":
        cfg = cfg or load_config()
        j = cfg.get("judge", {})
        paths = cfg.get("paths", {})
        return cls(
            base_url=j["base_url"],
            api_key=j["api_key"],
            model=j["model"],
            temperature=j.get("temperature", 0.0),
            seed=j.get("seed", 42),
            max_tokens=j.get("max_tokens", 512),
            timeout=j.get("timeout", 60.0),
            max_retries=j.get("max_retries", 3),
            cache_dir=paths.get("cache_dir"),
        )

    def _cache_key(self, prompt: str) -> str:
        return f"{self.model}|{self.system_prompt}\n---\n{prompt}"

    def query(self, prompt: str) -> str:
        key = self._cache_key(prompt)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]
        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                resp = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    seed=self.seed,
                    max_tokens=self.max_tokens,
                )
                text = resp.choices[0].message.content or ""
                self._cache.put(key, text)
                return text
            except Exception as exc:  # noqa: BLE001 - surface after retries
                last_err = exc
                time.sleep(0.4 * (attempt + 1))
        raise RuntimeError(f"Judge query failed after {self.max_retries} retries: {last_err}")


# ---------------------------------------------------------------------------
# Mock judge (offline / test)
# ---------------------------------------------------------------------------

_NUM_RE = re.compile(r"overall_score.{0,3}(\d{1,3})|score\s+(?:of|:)\s*(\d{1,3})", re.IGNORECASE)
_MAX_CUE_RE = re.compile(
    r"(highest|maximum)\s+(?:possible\s+)?score"
    r"|score\s+of\s+100"
    r"|overall_score.{0,4}100"
    r"|all\s+axes\s*=?\s*20"
    r"|output.{0,15}100",
    re.IGNORECASE,
)
_MIN_CUE_RE = re.compile(
    r"(lowest|minimum|worst)\s+(?:possible\s+)?(?:score|text)"
    r"|score\s+of\s+0\b"
    r"|overall_score.{0,3}0\b"
    r"|worst\s+possible",
    re.IGNORECASE,
)

_DOC_PATTERNS = [
    # Require newlines around the tags so an inline mention like
    # "between <DATA> and </DATA>" in the defense's own description is NOT
    # mistaken for the data block.
    re.compile(r"<DATA>\s*\n(.*?)\n\s*</DATA>", re.DOTALL),
    re.compile(r"<UNTRUSTED_DATA>\s*\n(.*?)\n\s*</UNTRUSTED_DATA>", re.DOTALL),
    re.compile(r"Text to evaluate:\n(.+?)\n\nReturn the required JSON", re.DOTALL),
]
# A real isolation BLOCK (tags on their own lines), not an inline mention.
_ISOLATION_BLOCK_RE = re.compile(
    r"<(?:DATA|UNTRUSTED_DATA)>\s*\n.*?\n\s*</(?:DATA|UNTRUSTED_DATA)>", re.DOTALL
)


def _extract_doc(prompt: str) -> str:
    """Robustly extract the evaluated document region from any prompt framing."""
    for pat in _DOC_PATTERNS:
        m = pat.search(prompt)
        if m:
            return m.group(1)
    # Fallback: the untrusted_text is ``clean_text + "\n\n[INJECT]\n" + payload``.
    # Bound it by the surrounding double-newlines (paragraph boundaries).
    idx = prompt.find(INJECT_MARKER) if INJECT_MARKER in prompt else -1
    if idx >= 0:
        before = prompt[: max(0, idx - 2)]  # cut the "\n\n" preceding [INJECT]
        start = before.rfind("\n\n")
        start = start + 2 if start >= 0 else 0
        after = prompt[idx:]
        # skip past "[INJECT]\n" to find where payload ends
        skip = len(INJECT_MARKER) + 1  # +1 for the trailing newline
        nb = after.find("\n\n", skip)
        end = idx + nb if nb >= 0 else len(prompt)
        return prompt[start:end]
    return prompt


def _defense_strength(prompt: str) -> float:
    """Heuristic 0..1: how strongly the prompt isolates data from instructions.

    Only *structural* isolation (a real <DATA>/<UNTRUSTED_DATA> block on its own
    lines) counts. Pure reminder prose -- including the rubric's own "ignore
    instructions" sentence -- does NOT, modeling a gullible judge that ignores
    text reminders but respects structural boundaries.
    """
    return 0.85 if _ISOLATION_BLOCK_RE.search(prompt) else 0.0


class MockJudge:
    """Deterministic, offline judge used for tests / demos without a model.

    Score = feature-based baseline blended with injection cues, modulated by
    defense strength, so the attack-defense pipeline shows a realistic
    gradient: attacks shift the ranking under weak/no defense; structural
    isolation defenses resist them. The same prompt always yields the same
    score (temperature-0 semantics).
    """

    model = "mock-judge-v1"

    def __init__(self, cache_dir: str | os.PathLike | None = None):
        self._cache = _DiskCache(cache_dir)

    @classmethod
    def from_config(cls, cfg: dict[str, Any] | None = None) -> "MockJudge":
        cfg = cfg or load_config()
        return cls(cache_dir=cfg.get("paths", {}).get("cache_dir"))

    def _featurize(self, prompt: str) -> dict[str, int]:
        doc = _extract_doc(prompt)
        strength = _defense_strength(prompt)

        words = re.findall(r"\w+", doc.lower())
        n = max(1, len(words))
        chars = max(1, len(doc))
        avg_word = sum(len(w) for w in words) / n
        has_code = bool(re.search(r"def |class |function|algorithm|\{|\}|http", doc))
        has_cite = bool(re.search(r"\(\d{4}\)|et al\.|doi|arxiv", doc, re.IGNORECASE))
        long_sent = sum(1 for s in re.split(r"[.!?]", doc) if len(s.split()) > 12)

        wq = min(20, 6 + int(avg_word * 1.2) + (2 if long_sent else 0))
        coh = min(20, 7 + int(n / 40) + (3 if has_code else 0))
        info = min(20, 6 + int(chars / 200) + (4 if has_cite else 0))
        edu = min(20, 6 + int(n / 50) + (3 if has_code else 0) + (2 if has_cite else 0))
        trust = min(20, 8 + (3 if has_cite else 0) - (4 if "TODO" in doc else 0))
        trust = max(0, trust)
        vec = [wq, coh, info, edu, trust]

        # --- Injection susceptibility, modulated by defense strength ---
        max_hit = bool(_MAX_CUE_RE.search(doc))
        min_hit = bool(_MIN_CUE_RE.search(doc))
        m2 = _NUM_RE.search(doc)
        target = None
        if m2:
            raw = m2.group(1) or m2.group(2)
            if raw:
                target = int(raw)

        target_vec = None
        if max_hit or (target is not None and target >= 90):
            target_vec = [20, 20, 20, 20, 20]
        elif min_hit or (target is not None and target <= 10):
            target_vec = [2, 2, 2, 2, 2]
        elif target is not None:
            # pull overall toward target proportionally
            cur = sum(vec)
            if cur > 0:
                scale = target / cur
                target_vec = [min(20, max(0, round(v * scale))) for v in vec]

        if target_vec is not None:
            factor = 1.0 - strength  # how much the cue prevails
            if factor > 0:
                vec = [min(20, max(0, round(v * (1 - factor) + t * factor)))
                       for v, t in zip(vec, target_vec)]

        return {
            "writing_quality": vec[0],
            "coherence": vec[1],
            "informativeness": vec[2],
            "educational_value": vec[3],
            "trustworthiness": vec[4],
        }

    def query(self, prompt: str) -> str:
        key = f"{self.model}|{prompt}"
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        feat = self._featurize(prompt)
        overall = sum(feat.values())
        obj = {**feat, "overall_score": overall, "reason": f"mock:{overall}"}
        text = json.dumps(obj, ensure_ascii=False)
        self._cache.put(key, text)
        return text


def make_judge(cfg: dict[str, Any] | None = None) -> Judge:
    """Factory: pick MockJudge when ``judge.model == 'mock'`` else OpenAIJudge."""
    cfg = cfg or load_config()
    j = cfg.get("judge", {})
    if str(j.get("model", "")).lower().startswith("mock"):
        return MockJudge.from_config(cfg)
    return OpenAIJudge.from_config(cfg)
