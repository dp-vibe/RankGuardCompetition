"""Persistence + loading of sample sets (JSONL)."""

from __future__ import annotations

import json
from pathlib import Path

from ..schema import Group, Sample


def save_samples(samples: list[Sample], path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        for s in samples:
            fh.write(json.dumps(s.to_dict(), ensure_ascii=False) + "\n")
    return p


def load_samples(path: str | Path) -> list[Sample]:
    p = Path(path)
    if not p.exists():
        return []
    out: list[Sample] = []
    with p.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            out.append(Sample.from_dict(json.loads(line)))
    return out


def group_path(split: str, data_dir: str | Path) -> Path:
    return Path(data_dir) / f"{split}.jsonl"


def load_group(split: str, data_dir: str | Path) -> Group:
    samples = load_samples(group_path(split, data_dir))
    return Group(group_id=f"group_{split}", split=split, samples=samples)


def load_groups(data_dir: str | Path) -> list[Group]:
    return [
        load_group("public", data_dir),
        load_group("hidden", data_dir),
    ]
