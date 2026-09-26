"""Lightweight, deterministic domain classifier.

Maps a (url, text) pair to one of the configured domains using keyword
signals. Used only to label samples for stratification; it is not part of the
scoring path and has no effect on the judge.
"""

from __future__ import annotations

import re
from urllib.parse import urlparse

_KEYWORDS: dict[str, tuple[str, ...]] = {
    "computer_science": (
        "algorithm", "function", "programming", "software", "compiler", "database",
        "python", "javascript", "api", "server", "code", "kernel", "memory", "gpu",
        "neural", "model", "training", "dataset",
    ),
    "medicine": (
        "patient", "clinical", "diagnosis", "treatment", "disease", "symptom",
        "therapy", "drug", "dose", "medical", "health", "physician", "trial",
    ),
    "law": (
        "court", "statute", "contract", "liability", "plaintiff", "defendant",
        "jurisdiction", "amendment", "regulation", "legal", "tort", "clause",
    ),
    "finance": (
        "revenue", "equity", "asset", "portfolio", "market", "stock", "bond",
        "interest", "inflation", "fiscal", "audit", "earnings", "capital",
    ),
    "history": (
        "century", "empire", "war", "revolution", "dynasty", "ancient", "medieval",
        "treaty", "colonial", "civilization", "kingdom", "archaeolog",
    ),
    "literature": (
        "novel", "poem", "poetry", "protagonist", "metaphor", "narrative",
        "author", "verse", "stanza", "fiction", "literary", "prose",
    ),
    "mathematics": (
        "theorem", "proof", "integral", "derivative", "matrix", "vector",
        "polynomial", "equation", "lemma", "topology", "manifold", "algebra",
    ),
    "physics": (
        "quantum", "relativity", "particle", "energy", "momentum", "wavelength",
        "electromagnetic", "thermodynamic", "velocity", "photon", "fermion",
    ),
}


def classify_domain(url: str, text: str, fallback_domains: list[str], idx: int) -> str:
    blob = f"{url} {text[:2000]}".lower()
    best, best_hits = "", 0
    for domain, kws in _KEYWORDS.items():
        hits = sum(1 for kw in kws if kw in blob)
        if hits > best_hits:
            best, best_hits = domain, hits
    if best:
        return best
    # deterministic round-robin over the configured list
    return fallback_domains[idx % len(fallback_domains)]


def host_of(url: str) -> str:
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return ""
