"""Dataset download from Hugging Face.

Primary source: ``HuggingFaceFW/fineweb`` (public, pretraining-quality web
text with url/language/token_count metadata) -- streamed so only the needed
rows are fetched and persisted locally.

If ``HF_TOKEN`` is set, an attempt is also made on ``princeton-nlp/QuRating``
(the spec's recommended source; it is gated). On any failure the caller falls
back to :mod:`rankguard.dataset.synthetic`.
"""

from __future__ import annotations

import os
from typing import Iterator

from ..schema import Sample
from .classify import classify_domain


def _from_fineweb(
    n: int, split: str, domains: list[str], min_chars: int, max_chars: int,
    start_index: int = 0, skip: int = 0,
) -> Iterator[Sample]:
    from datasets import load_dataset

    ds = load_dataset(
        "HuggingFaceFW/fineweb", "sample-10BT", split="train", streaming=True
    )
    emitted = 0
    skipped = 0
    for ex in ds:
        if skipped < skip:
            skipped += 1
            continue
        text = (ex.get("text") or "").strip()
        if not (min_chars <= len(text) <= max_chars):
            continue
        tokens = ex.get("token_count") or 0
        if tokens < 30:
            continue
        lang = ex.get("language") or "en"
        url = ex.get("url") or ""
        domain = classify_domain(url, text, domains, emitted + start_index)
        sid = f"{split}_{domain[:4]}_{emitted:04d}"
        yield Sample(
            sample_id=sid,
            domain=domain,
            language=lang,
            clean_text=text,
            inject_position="append",
        )
        emitted += 1
        if emitted >= n:
            return


def _from_qurating(
    n: int, split: str, domains: list[str], min_chars: int, max_chars: int,
    start_index: int = 0, skip: int = 0,
) -> Iterator[Sample]:
    from datasets import load_dataset

    ds = load_dataset("princeton-nlp/QuRating", "sampled", split="train", streaming=True)
    emitted = 0
    skipped = 0
    for ex in ds:
        if skipped < skip:
            skipped += 1
            continue
        text = (ex.get("text") or "").strip()
        if not (min_chars <= len(text) <= max_chars):
            continue
        url = ex.get("url") or ""
        domain = classify_domain(url, text, domains, emitted + start_index)
        sid = f"{split}_{domain[:4]}_{emitted:04d}"
        yield Sample(
            sample_id=sid,
            domain=domain,
            language="en",
            clean_text=text,
            inject_position="append",
        )
        emitted += 1
        if emitted >= n:
            return


def download_samples(
    n: int,
    split: str,
    domains: list[str],
    min_chars: int,
    max_chars: int,
    source: str = "fineweb",
    start_index: int = 0,
    skip: int = 0,
) -> tuple[list[Sample], str]:
    """Download ``n`` samples for ``split``.

    Returns ``(samples, actual_source)``. Raises only if both QuRating and
    FineWeb fail (caller then falls back to synthetic).
    """
    source = (source or "fineweb").lower()
    if source == "qurating" or os.environ.get("HF_TOKEN"):
        try:
            return list(_from_qurating(n, split, domains, min_chars, max_chars,
                                       start_index, skip)), "qurating"
        except Exception as exc:  # noqa: BLE001 - fall through to fineweb
            print(f"[dataset] QuRating unavailable ({type(exc).__name__}); trying FineWeb")
    return list(_from_fineweb(n, split, domains, min_chars, max_chars,
                              start_index, skip)), "fineweb"
