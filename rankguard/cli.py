"""RankGuard command-line interface.

Subcommands:
    algorithms   list discovered attacks/defenses
    build        download/build the local dataset
    refs         generate reference scores & rankings (S0/R0)
    match        run the full round-robin (attacks x defenses x splits)
    qualify      run the qualification stage
    clean-gate   run the defense clean-utility gate
    debug        score one sample with a chosen attack+defense
    serve        start the FastAPI server

Examples:
    python -m rankguard build --source fineweb --public 200 --hidden 200
    RANKGUARD_JUDGE__MODEL=mock python -m rankguard refs
    python -m rankguard match --attacks ScoreMaxAttack --defenses NoDefense
    python -m rankguard serve --port 8000
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from .config import load_config


def _print(obj: Any) -> None:
    if isinstance(obj, (dict, list)):
        print(json.dumps(obj, ensure_ascii=False, indent=2))
    else:
        print(obj)


def _progress_factory(desc: str):
    try:
        from tqdm import tqdm
        pbar = tqdm(desc=desc, unit="smp", leave=False)
        def cb(i: int, n: int):
            pbar.total = n
            pbar.n = i
            pbar.refresh()
        def close():
            pbar.close()
        return cb, close
    except Exception:
        def cb(i, n):
            print(f"  {desc}: {i}/{n}")
        return cb, lambda: None


# --------------------------- commands ------------------------------------- #

def cmd_algorithms(args) -> None:
    from . import service
    _print({"attacks": service.list_attacks(), "defenses": service.list_defenses()})


def cmd_build(args) -> None:
    from .dataset.build import build_dataset
    cfg = load_config()
    if args.source:
        cfg.setdefault("dataset", {})["source"] = args.source
    if args.public:
        cfg["dataset"]["public_size"] = args.public
    if args.hidden:
        cfg["dataset"]["hidden_size"] = args.hidden
    if args.data_dir:
        cfg.setdefault("paths", {})["data_dir"] = args.data_dir
    _print(build_dataset(cfg))


def cmd_refs(args) -> None:
    from . import service
    from .judge.client import make_judge
    cfg = load_config()
    if args.mock:
        cfg.setdefault("judge", {})["model"] = "mock"
    j = make_judge(cfg)
    cb, close = _progress_factory("refs")
    try:
        rep = service.ensure_references(cfg, force=args.force, judge=j,
                                        progress=cb)
    finally:
        close()
    _print(rep)


def cmd_match(args) -> None:
    from . import service
    from .judge.client import make_judge
    cfg = load_config()
    if args.mock:
        cfg.setdefault("judge", {})["model"] = "mock"
    j = make_judge(cfg)

    cells = (len(args.attacks) if args.attacks else 0) * (len(args.defenses) if args.defenses else 0)
    try:
        from tqdm import tqdm
        pbar = tqdm(desc="cells", unit="cell", total=None, leave=False)
        def prog(done, total, aname, dname):
            pbar.total = total
            pbar.n = done
            pbar.set_postfix_str(f"{aname} x {dname}")
            pbar.refresh()
    except Exception:
        def prog(done, total, aname, dname):
            print(f"  cell {done}/{total}: {aname} x {dname}")

    try:
        res = service.run_match(cfg, args.attacks, args.defenses, args.splits,
                                judge=j, progress=prog)
    finally:
        try:
            pbar.close()
        except Exception:
            pass

    from .arbiter.round_robin import rank_attack_board, rank_defense_board
    print("=== Attack board (larger RankShift = better) ===")
    for name, val in rank_attack_board(res):
        print(f"  {name:28s} {val:8.4f}")
    print("=== Defense board (smaller RankShift = better) ===")
    for name, val in rank_defense_board(res):
        print(f"  {name:28s} {val:8.4f}")

    if args.save:
        p = service.save_result(res, args.save, cfg)
        print(f"saved -> {p}")
    if args.full:
        _print(res.to_dict())


def cmd_qualify(args) -> None:
    from . import service
    from .judge.client import make_judge
    cfg = load_config()
    if args.mock:
        cfg.setdefault("judge", {})["model"] = "mock"
    j = make_judge(cfg)
    _print(service.run_qualification(cfg, judge=j, top_fraction=args.top_fraction))


def cmd_clean_gate(args) -> None:
    from . import service
    from .judge.client import make_judge
    cfg = load_config()
    if args.mock:
        cfg.setdefault("judge", {})["model"] = "mock"
    j = make_judge(cfg)
    _print(service.run_clean_gate(cfg, judge=j))


def cmd_debug(args) -> None:
    from . import service
    from .judge.client import make_judge
    from .env import AttackEnv, DefenseEnv
    from .prompt.builder import build_base_prompt, build_untrusted_text
    from .schema import OUTPUT_SCHEMA, RUBRIC_INSTRUCTION
    from .sandbox.runner import safe_call_attack, safe_call_defense
    from .judge.parser import parse_or_repair
    from .dataset.loader import load_groups
    cfg = load_config()
    if args.mock:
        cfg.setdefault("judge", {})["model"] = "mock"
    sample = None
    for g in load_groups(cfg["paths"]["data_dir"]):
        for s in g.samples:
            if s.sample_id == args.sample_id:
                sample = s
    if sample is None:
        print(f"sample {args.sample_id} not found"); sys.exit(1)
    attacks = service.get_attacks(); defenses = service.get_defenses()
    if args.attack not in attacks:
        print(f"unknown attack {args.attack}"); sys.exit(1)
    if args.defense not in defenses:
        print(f"unknown defense {args.defense}"); sys.exit(1)
    limits = cfg.get("limits", {})
    rubric = RUBRIC_INSTRUCTION
    aenv = AttackEnv(sample_id=sample.sample_id, clean_text=sample.clean_text,
                     domain=sample.domain, language=sample.language,
                     rubric_instruction=rubric, rubric_id=sample.rubric_id,
                     inject_position=sample.inject_position)
    payload = safe_call_attack(attacks[args.attack], aenv,
                               timeout=float(limits.get("attack_timeout", 10)),
                               max_bytes=int(limits.get("payload_max_bytes", 512)))
    untrusted = build_untrusted_text(sample.clean_text, payload)
    base = build_base_prompt(rubric, untrusted)
    denv = DefenseEnv(rubric_instruction=rubric, untrusted_text=untrusted,
                      base_prompt=base, output_schema=OUTPUT_SCHEMA)
    final, warns = safe_call_defense(defenses[args.defense], denv,
                                     timeout=float(limits.get("defense_timeout", 10)),
                                     max_added_bytes=int(limits.get("defense_added_max_bytes", 2048)))
    judge = make_judge(cfg)
    raw = judge.query(final)
    score = parse_or_repair(raw)
    _print({
        "sample_id": sample.sample_id, "attack": args.attack, "defense": args.defense,
        "payload": payload, "payload_bytes": len(payload.encode("utf-8")),
        "warnings": warns, "final_prompt": final,
        "raw_response": raw, "score": score.to_dict() if score else None,
    })


def cmd_serve(args) -> None:
    import uvicorn
    uvicorn.run("rankguard.api.server:app", host=args.host, port=args.port,
                reload=args.reload)


# --------------------------- parser --------------------------------------- #

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="rankguard", description="RankGuard CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("algorithms", help="list attacks/defenses").set_defaults(func=cmd_algorithms)

    b = sub.add_parser("build", help="download/build local dataset")
    b.add_argument("--source", choices=["qurating", "fineweb", "synthetic"])
    b.add_argument("--public", type=int)
    b.add_argument("--hidden", type=int)
    b.add_argument("--data-dir")
    b.set_defaults(func=cmd_build)

    r = sub.add_parser("refs", help="generate reference scores & rankings")
    r.add_argument("--force", action="store_true")
    r.add_argument("--mock", action="store_true", help="use the offline mock judge")
    r.set_defaults(func=cmd_refs)

    m = sub.add_parser("match", help="run round-robin")
    m.add_argument("--attacks", nargs="*")
    m.add_argument("--defenses", nargs="*")
    m.add_argument("--splits", nargs="*", choices=["public", "hidden"])
    m.add_argument("--save")
    m.add_argument("--full", action="store_true", help="print full matrix JSON")
    m.add_argument("--mock", action="store_true")
    m.set_defaults(func=cmd_match)

    q = sub.add_parser("qualify", help="run qualification stage")
    q.add_argument("--top-fraction", type=float, default=0.5)
    q.add_argument("--mock", action="store_true")
    q.set_defaults(func=cmd_qualify)

    cg = sub.add_parser("clean-gate", help="run defense clean-utility gate")
    cg.add_argument("--mock", action="store_true")
    cg.set_defaults(func=cmd_clean_gate)

    d = sub.add_parser("debug", help="score one sample")
    d.add_argument("sample_id")
    d.add_argument("--attack", default="NoOpAttack")
    d.add_argument("--defense", default="NoDefense")
    d.add_argument("--mock", action="store_true")
    d.set_defaults(func=cmd_debug)

    s = sub.add_parser("serve", help="start FastAPI server")
    s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--reload", action="store_true")
    s.set_defaults(func=cmd_serve)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
