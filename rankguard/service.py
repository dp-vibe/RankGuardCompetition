"""High-level orchestration service.

Ties together dataset, reference, judge, plugins, and arbiter into the
end-to-end pipeline used by both the CLI and the FastAPI server:

    build_dataset  ->  generate_references  ->  run_match / qualify / clean_gate

All plugin discovery (attacks / defenses) is decoupled here: the service
discovers each side independently and only combines them inside the arbiter.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import load_config
from .dataset.build import build_dataset
from .dataset.loader import load_group, load_groups
from .judge.client import make_judge, Judge
from .attacks.base import AttackRegistry, discover_attacks
from .defenses.base import DefenseRegistry, discover_defenses
from .reference.generator import generate_reference, load_reference
from .arbiter.round_robin import full_round_robin, RoundRobinResult
from .arbiter.qualify import qualify_attacks, qualify_defenses
from .arbiter.clean_gate import clean_gate


# --------------------------------------------------------------------------- #
# Plugin discovery (decoupled)                                                #
# --------------------------------------------------------------------------- #

def list_attacks() -> dict[str, str]:
    """Return {attack_name: description} for all discovered attacks."""
    return {name: a.description for name, a in discover_attacks().items()}


def list_defenses() -> dict[str, str]:
    """Return {defense_name: description} for all discovered defenses."""
    return {name: d.description for name, d in discover_defenses().items()}


def get_attacks(names: list[str] | None = None) -> dict[str, Any]:
    all_a = discover_attacks()
    if names is None:
        return all_a
    missing = [n for n in names if n not in all_a]
    if missing:
        raise KeyError(f"unknown attacks: {missing}; available={sorted(all_a)}")
    return {n: all_a[n] for n in names}


def get_defenses(names: list[str] | None = None) -> dict[str, Any]:
    all_d = discover_defenses()
    if names is None:
        return all_d
    missing = [n for n in names if n not in all_d]
    if missing:
        raise KeyError(f"unknown defenses: {missing}; available={sorted(all_d)}")
    return {n: all_d[n] for n in names}


# --------------------------------------------------------------------------- #
# Data + reference                                                            #
# --------------------------------------------------------------------------- #

def ensure_dataset(cfg: dict | None = None) -> dict:
    return build_dataset(cfg or load_config())


def ensure_references(
    cfg: dict | None = None,
    force: bool = False,
    judge: Judge | None = None,
    progress=None,
) -> dict:
    cfg = cfg or load_config()
    data_dir = cfg.get("paths", {}).get("data_dir", "data")
    groups = load_groups(data_dir)
    j = judge or make_judge(cfg)
    report = {}
    for g in groups:
        ref = generate_reference(g, j, data_dir=data_dir, force=force, progress=progress)
        report[g.group_id] = {"n": len(ref.ranking), "group_id": g.group_id}
    return report


def load_groups_with_references(cfg: dict | None = None) -> list[tuple[Any, Any]]:
    cfg = cfg or load_config()
    data_dir = cfg.get("paths", {}).get("data_dir", "data")
    out = []
    for g in load_groups(data_dir):
        ref = load_reference(g.group_id, data_dir)
        if ref is None:
            raise RuntimeError(f"reference for {g.group_id} not found; run generate-references first")
        out.append((g, ref))
    return out


# --------------------------------------------------------------------------- #
# Matches                                                                     #
# --------------------------------------------------------------------------- #

def run_match(
    cfg: dict | None = None,
    attack_names: list[str] | None = None,
    defense_names: list[str] | None = None,
    splits: list[str] | None = None,
    judge: Judge | None = None,
    progress=None,
) -> RoundRobinResult:
    """Run a full round-robin over the selected attacks x defenses x splits."""
    cfg = cfg or load_config()
    attacks = get_attacks(attack_names)
    defenses = get_defenses(defense_names)
    pairs = load_groups_with_references(cfg)
    if splits:
        pairs = [(g, r) for g, r in pairs if g.split in splits]
    j = judge or make_judge(cfg)
    return full_round_robin(attacks, defenses, pairs, j, cfg, progress=progress)


def run_qualification(
    cfg: dict | None = None,
    judge: Judge | None = None,
    top_fraction: float = 0.5,
) -> dict[str, Any]:
    cfg = cfg or load_config()
    attacks = get_attacks()
    defenses = get_defenses()
    no_defense = defenses.get("NoDefense") or defenses[min(defenses)]
    pairs = load_groups_with_references(cfg)
    j = judge or make_judge(cfg)

    # Official attack set = the two spec baselines.
    official = {n: attacks[n] for n in ("ScoreMaxAttack", "ScoreMinAttack")
                if n in attacks}
    if not official:
        official = dict(list(attacks.items())[:2])

    atk = qualify_attacks(attacks, no_defense, pairs, j, cfg, top_fraction)
    dfn = qualify_defenses(defenses, official, pairs, j, cfg, top_fraction)
    return {"attacks": atk, "defenses": dfn,
            "official_attacks": list(official)}


def run_clean_gate(cfg: dict | None = None, judge: Judge | None = None) -> dict[str, Any]:
    cfg = cfg or load_config()
    defenses = get_defenses()
    attacks = get_attacks()
    noop = attacks.get("NoOpAttack") or attacks[min(attacks)]
    pairs = load_groups_with_references(cfg)
    j = judge or make_judge(cfg)
    return clean_gate(defenses, noop, pairs, j, cfg)


# --------------------------------------------------------------------------- #
# Persistence                                                                 #
# --------------------------------------------------------------------------- #

def save_result(result: Any, name: str, cfg: dict | None = None) -> Path:
    cfg = cfg or load_config()
    out_dir = Path(cfg.get("paths", {}).get("results_dir", "results"))
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"{name}.json"
    if hasattr(result, "to_dict"):
        data = result.to_dict()
    else:
        data = result
    import json
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), "utf-8")
    return p
