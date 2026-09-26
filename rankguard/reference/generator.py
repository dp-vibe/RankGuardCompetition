"""Reference score & ranking generation (spec section 七).

S0 = Judge(BasePrompt, clean_text)   -- no attack, no defense
R0 = Rank(S0)

Generated once and locked on disk (``data/reference/<group_id>.json``). The
reference is produced by running the *same* pipeline the match uses, but with
an empty payload and NoDefense, so S0/R0 are directly comparable to every
(attack, defense) cell.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from ..dataset.loader import group_path
from ..judge.client import Judge
from ..judge.parser import parse_or_repair
from ..metrics.ranking import build_ranking
from ..prompt.builder import build_base_prompt, build_untrusted_text
from ..schema import Group, ReferenceRanking, Sample, Score, RUBRIC_INSTRUCTION


def _reference_path(group_id: str, data_dir: str | Path) -> Path:
    d = Path(data_dir) / "reference"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{group_id}.json"


def score_sample_clean(sample: Sample, judge: Judge, rubric_instruction: str) -> Score:
    """Score one sample with no attack and no defense (the S0 path)."""
    untrusted = build_untrusted_text(sample.clean_text, "")  # empty payload
    prompt = build_base_prompt(rubric_instruction, untrusted)
    raw = judge.query(prompt)
    score = parse_or_repair(raw)
    if score is None:
        # Reference must exist; fall back to a zero vector and mark invalid.
        score = Score(valid=False)
    return score


def generate_reference(
    group: Group,
    judge: Judge,
    data_dir: str | Path = "data",
    rubric_instruction: str | None = None,
    force: bool = False,
    progress: Callable[[int, int], None] | None = None,
) -> ReferenceRanking:
    """Build (or load cached) S0/R0 for a group."""
    ref_path = _reference_path(group.group_id, data_dir)
    if ref_path.exists() and not force:
        return ReferenceRanking.from_dict(__import__("json").loads(ref_path.read_text("utf-8")))

    rubric = rubric_instruction or RUBRIC_INSTRUCTION
    scores: dict[str, Score] = {}
    n = len(group.samples)
    for i, sample in enumerate(group.samples):
        scores[sample.sample_id] = score_sample_clean(sample, judge, rubric)
        if progress:
            progress(i + 1, n)

    ranking = build_ranking(scores)
    ref = ReferenceRanking(group_id=group.group_id, scores_by_id=scores, ranking=ranking)
    ref_path.write_text(__import__("json").dumps(ref.to_dict(), ensure_ascii=False, indent=2), "utf-8")
    return ref


def load_reference(group_id: str, data_dir: str | Path = "data") -> ReferenceRanking | None:
    p = _reference_path(group_id, data_dir)
    if not p.exists():
        return None
    return ReferenceRanking.from_dict(__import__("json").loads(p.read_text("utf-8")))
