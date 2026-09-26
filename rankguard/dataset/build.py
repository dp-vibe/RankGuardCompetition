"""Dataset build orchestrator.

Ensures ``data/public.jsonl`` and ``data/hidden.jsonl`` exist locally. Tries
the configured HF source first; on any failure (no network, gated repo, etc.)
falls back to the deterministic synthetic generator so the rest of the
pipeline always has data to run on.
"""

from __future__ import annotations

from pathlib import Path

from ..config import load_config
from .download import download_samples
from .loader import group_path, load_group, save_samples
from .synthetic import generate_samples


def _build_split(split: str, cfg: dict, skip: int = 0) -> tuple[int, str]:
    ds_cfg = cfg.get("dataset", {})
    paths = cfg.get("paths", {})
    data_dir = paths.get("data_dir", "data")
    size = int(ds_cfg.get(f"{split}_size", 1000))
    domains = list(ds_cfg.get("domains", ["computer_science"]))
    languages = list(ds_cfg.get("languages", ["en"]))
    min_chars = int(ds_cfg.get("min_chars", 200))
    max_chars = int(ds_cfg.get("max_chars", 4000))
    source = ds_cfg.get("source", "fineweb")

    target = group_path(split, data_dir)
    if target.exists() and load_group(split, data_dir).size >= size:
        return size, "cached"

    samples = None
    actual_source = None
    err = None
    if source != "synthetic":
        try:
            samples, actual_source = download_samples(
                n=size, split=split, domains=domains,
                min_chars=min_chars, max_chars=max_chars,
                source=source, start_index=skip, skip=skip,
            )
            if len(samples) < size:
                err = f"short read ({len(samples)}/{size})"
                samples = None
        except Exception as exc:  # noqa: BLE001
            err = f"{type(exc).__name__}: {exc}"

    if samples is None:
        if err:
            print(f"[dataset] {split}: download failed ({err}); using synthetic fallback")
        else:
            print(f"[dataset] {split}: using synthetic generator")
        samples = generate_samples(
            n=size, split=split, domains=domains, languages=languages,
            seed=42, start_index=skip,
        )
        actual_source = "synthetic"

    save_samples(samples, target)
    return len(samples), actual_source or source


def build_dataset(cfg: dict | None = None) -> dict:
    """Ensure both splits exist on disk; return a small status report.

    The hidden split is streamed starting past the public split so the two are
    disjoint samples.
    """
    cfg = cfg or load_config()
    paths = cfg.get("paths", {})
    data_dir = Path(paths.get("data_dir", "data"))
    data_dir.mkdir(parents=True, exist_ok=True)
    public_size = int(cfg.get("dataset", {}).get("public_size", 1000))
    report = {}
    pn, phow = _build_split("public", cfg, skip=0)
    report["public"] = {"count": pn, "source": phow,
                        "path": str(group_path("public", data_dir))}
    hn, hhow = _build_split("hidden", cfg, skip=public_size)
    report["hidden"] = {"count": hn, "source": hhow,
                        "path": str(group_path("hidden", data_dir))}
    return report
