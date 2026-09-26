"""Built-in attack plugins.

Each module in this package registers one or more attacks via
``@AttackRegistry.register``. ``discover_attacks()`` imports them all.

References (实现参考.txt):
  - JudgeDeceiver : optimized prompt-injection against LLM-as-a-Judge.
  - spec baselines: ScoreMaxAttack, ScoreMinAttack.
"""
