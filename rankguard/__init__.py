"""RankGuard: attack-defense backend for LLM scoring-system ranking shift.

Public subpackages:
  - judge      : OpenAI-compatible LLM judge client + JSON repair.
  - prompt     : base-prompt builder shared by attack/defense pipeline.
  - attacks    : decoupled attack plugins (auto-discovered).
  - defenses   : decoupled defense plugins (auto-discovered).
  - metrics    : ranking + RankShift (D_pair, D_foot, D_top).
  - dataset    : QuRating download / synthetic fallback, sample loader.
  - reference  : reference score & ranking generator (S0, R0).
  - arbiter    : evaluate_group, round-robin, qualification, clean gate.
  - sandbox    : safe, timed, validated invocation of user submissions.
  - api        : FastAPI service exposing the whole pipeline.
"""

__version__ = "0.1.0"
