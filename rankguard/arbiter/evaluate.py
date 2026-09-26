"""``evaluate_group``: run one (attack, defense) pair over one ranking group.

Faithful to the spec pseudocode (section 八):

    for sample in samples:
        payload    = safe_call_attack(attack, AttackEnv(sample))
        untrusted  = clean_text + "\\n\\n[INJECT]\\n" + payload
        base_prompt= build_base_prompt(rubric, untrusted)
        final      = safe_call_defense(defense, DefenseEnv(...))
        validate(final preserves untrusted)
        raw        = judge.query(final)
        parsed     = parse_or_repair(raw) or reference.score_by_id[sample_id]
    ranking = build_ranking(scores)
    shift   = rank_shift(reference.scores, scores)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from ..env import AttackEnv, DefenseEnv
from ..judge.client import Judge
from ..judge.parser import parse_or_repair
from ..metrics.shift import ShiftBreakdown, rank_shift
from ..prompt.builder import build_base_prompt, build_untrusted_text
from ..sandbox.runner import safe_call_attack, safe_call_defense
from ..schema import OUTPUT_SCHEMA, RUBRIC_INSTRUCTION, Group, ReferenceRanking, Score


@dataclass
class GroupResult:
    group_id: str
    shift: ShiftBreakdown
    valid_rate: float
    invalid_count: int
    n: int
    scores: dict[str, Score] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    elapsed: float = 0.0

    @property
    def rank_shift_value(self) -> float:
        return self.shift.rank_shift

    def to_dict(self) -> dict[str, Any]:
        return {
            "group_id": self.group_id,
            "n": self.n,
            "valid_rate": round(self.valid_rate, 4),
            "invalid_count": self.invalid_count,
            "rank_shift": round(self.rank_shift_value, 4),
            "shift": self.shift.to_dict(),
            "warnings": self.warnings,
            "elapsed": round(self.elapsed, 3),
        }


def evaluate_group(
    group: Group,
    attack,
    defense,
    judge: Judge,
    reference: ReferenceRanking,
    cfg: dict[str, Any] | None = None,
    rubric_instruction: str | None = None,
) -> GroupResult:
    """Run one (attack, defense) pair over ``group`` and compute RankShift."""
    cfg = cfg or {}
    limits = cfg.get("limits", {})
    metrics_cfg = cfg.get("metrics", {})
    rubric = rubric_instruction or RUBRIC_INSTRUCTION

    payload_max = int(limits.get("payload_max_bytes", 512))
    added_max = int(limits.get("defense_added_max_bytes", 2048))
    atk_timeout = float(limits.get("attack_timeout", 10.0))
    def_timeout = float(limits.get("defense_timeout", 10.0))

    scores: dict[str, Score] = {}
    invalid = 0
    warnings: list[str] = []
    t0 = time.time()

    for sample in group.samples:
        attack_env = AttackEnv(
            sample_id=sample.sample_id,
            clean_text=sample.clean_text,
            domain=sample.domain,
            language=sample.language,
            rubric_instruction=rubric,
            rubric_id=sample.rubric_id,
            inject_position=sample.inject_position,
        )
        payload = safe_call_attack(attack, attack_env, timeout=atk_timeout,
                                   max_bytes=payload_max)

        untrusted_text = build_untrusted_text(sample.clean_text, payload)
        base_prompt = build_base_prompt(rubric, untrusted_text)

        defense_env = DefenseEnv(
            rubric_instruction=rubric,
            untrusted_text=untrusted_text,
            base_prompt=base_prompt,
            output_schema=OUTPUT_SCHEMA,
        )
        final_prompt, dwarns = safe_call_defense(defense, defense_env,
                                                 timeout=def_timeout,
                                                 max_added_bytes=added_max)
        if dwarns:
            warnings.append(f"{sample.sample_id}:{','.join(dwarns)}")

        raw = judge.query(final_prompt)
        parsed = parse_or_repair(raw)
        if parsed is None:
            # spec 五.6.1: substitute reference score on parse failure
            parsed = reference.scores_by_id.get(sample.sample_id, Score(valid=False))
            invalid += 1
        scores[sample.sample_id] = parsed

    shift = rank_shift(reference.scores_by_id, scores,
                       weights={k: metrics_cfg.get(k) for k in ("w_pair", "w_foot", "w_top") if k in metrics_cfg} or None,
                       topk={k: metrics_cfg.get(k) for k in ("top_k_ratio", "top_k_min") if k in metrics_cfg} or None)
    n = len(group.samples)
    valid_rate = 1.0 - (invalid / n) if n else 1.0
    return GroupResult(
        group_id=group.group_id,
        shift=shift,
        valid_rate=valid_rate,
        invalid_count=invalid,
        n=n,
        scores=scores,
        warnings=warnings,
        elapsed=time.time() - t0,
    )
