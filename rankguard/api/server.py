"""FastAPI service exposing the RankGuard pipeline.

Run with:
    uvicorn rankguard.api.server:app --reload --port 8000

Endpoints
---------
GET  /health                       -> liveness + config summary
GET  /algorithms                   -> {attacks: {name: desc}, defenses: {name: desc}}
GET  /config                       -> effective config
POST /dataset/build                -> download/build local dataset (idempotent)
GET  /dataset                      -> dataset status (counts)
POST /references/generate          -> (re)generate S0/R0 references
GET  /references                   -> list reference groups
POST /match                        -> run round-robin (attacks x defenses x splits)
POST /qualify                      -> run qualification stage
POST /clean-gate                   -> run defense clean-utility gate
POST /sample/debug                 -> score one sample with a chosen attack+defense
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .. import service
from ..config import load_config
from ..judge.client import make_judge
from ..env import AttackEnv, DefenseEnv
from ..prompt.builder import build_base_prompt, build_untrusted_text
from ..schema import OUTPUT_SCHEMA, RUBRIC_INSTRUCTION
from ..sandbox.runner import safe_call_attack, safe_call_defense
from ..judge.parser import parse_or_repair

app = FastAPI(title="RankGuard", version="0.1.0",
              description="LLM scoring-system ranking-shift attack-defense backend.")


# --------------------------- models --------------------------------------- #

class MatchRequest(BaseModel):
    attacks: list[str] | None = Field(None, description="attack names; None=all")
    defenses: list[str] | None = Field(None, description="defense names; None=all")
    splits: list[str] | None = Field(None, description="subset of [public, hidden]")
    save: str | None = Field(None, description="if set, save result to results/<name>.json")


class RefsRequest(BaseModel):
    force: bool = False


class SampleDebugRequest(BaseModel):
    sample_id: str
    attack: str = "NoOpAttack"
    defense: str = "NoDefense"


# --------------------------- endpoints ------------------------------------ #

@app.get("/health")
def health() -> dict[str, Any]:
    cfg = load_config()
    return {"status": "ok", "judge_model": cfg["judge"]["model"],
            "attacks": len(service.list_attacks()),
            "defenses": len(service.list_defenses())}


@app.get("/algorithms")
def algorithms() -> dict[str, dict[str, str]]:
    return {"attacks": service.list_attacks(), "defenses": service.list_defenses()}


@app.get("/config")
def config() -> dict[str, Any]:
    return load_config()


@app.post("/dataset/build")
def dataset_build() -> dict[str, Any]:
    return service.ensure_dataset(load_config())


@app.get("/dataset")
def dataset_status() -> dict[str, Any]:
    from ..dataset.loader import load_groups
    cfg = load_config()
    groups = load_groups(cfg["paths"]["data_dir"])
    return {g.split: {"group_id": g.group_id, "count": g.size} for g in groups}


@app.post("/references/generate")
def references_generate(req: RefsRequest) -> dict[str, Any]:
    return service.ensure_references(load_config(), force=req.force)


@app.get("/references")
def references_list() -> dict[str, Any]:
    from ..reference.generator import load_reference
    from ..dataset.loader import load_groups
    cfg = load_config()
    data_dir = cfg["paths"]["data_dir"]
    out = {}
    for g in load_groups(data_dir):
        ref = load_reference(g.group_id, data_dir)
        out[g.split] = {"group_id": g.group_id, "ready": ref is not None,
                        "n": len(ref.ranking) if ref else 0}
    return out


@app.post("/match")
def match(req: MatchRequest) -> dict[str, Any]:
    try:
        res = service.run_match(load_config(), req.attacks, req.defenses, req.splits)
    except KeyError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))
    out = res.to_dict()
    if req.save:
        p = service.save_result(res, req.save, load_config())
        out["saved"] = str(p)
    return out


@app.post("/qualify")
def qualify() -> dict[str, Any]:
    try:
        return service.run_qualification(load_config())
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))


@app.post("/clean-gate")
def clean_gate() -> dict[str, Any]:
    try:
        return service.run_clean_gate(load_config())
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))


@app.post("/sample/debug")
def sample_debug(req: SampleDebugRequest) -> dict[str, Any]:
    """Score a single sample with a chosen attack+defense (for debugging)."""
    cfg = load_config()
    from ..dataset.loader import load_groups
    groups = load_groups(cfg["paths"]["data_dir"])
    sample = None
    for g in groups:
        for s in g.samples:
            if s.sample_id == req.sample_id:
                sample = s
                break
    if sample is None:
        raise HTTPException(status_code=404, detail=f"sample {req.sample_id} not found")

    attacks = service.get_attacks()
    defenses = service.get_defenses()
    if req.attack not in attacks:
        raise HTTPException(status_code=400, detail=f"unknown attack {req.attack}")
    if req.defense not in defenses:
        raise HTTPException(status_code=400, detail=f"unknown defense {req.defense}")

    limits = cfg.get("limits", {})
    rubric = RUBRIC_INSTRUCTION
    aenv = AttackEnv(sample_id=sample.sample_id, clean_text=sample.clean_text,
                     domain=sample.domain, language=sample.language,
                     rubric_instruction=rubric, rubric_id=sample.rubric_id,
                     inject_position=sample.inject_position)
    payload = safe_call_attack(attacks[req.attack], aenv,
                               timeout=float(limits.get("attack_timeout", 10)),
                               max_bytes=int(limits.get("payload_max_bytes", 512)))
    untrusted = build_untrusted_text(sample.clean_text, payload)
    base = build_base_prompt(rubric, untrusted)
    denv = DefenseEnv(rubric_instruction=rubric, untrusted_text=untrusted,
                      base_prompt=base, output_schema=OUTPUT_SCHEMA)
    final, warns = safe_call_defense(defenses[req.defense], denv,
                                     timeout=float(limits.get("defense_timeout", 10)),
                                     max_added_bytes=int(limits.get("defense_added_max_bytes", 2048)))
    judge = make_judge(cfg)
    raw = judge.query(final)
    score = parse_or_repair(raw)
    return {
        "sample_id": sample.sample_id,
        "attack": req.attack, "defense": req.defense,
        "payload": payload,
        "payload_bytes": len(payload.encode("utf-8")),
        "final_prompt": final,
        "warnings": warns,
        "raw_response": raw,
        "score": score.to_dict() if score else None,
    }
