"""Built-in defense plugins.

Each module registers one or more defenses via ``@DefenseRegistry.register``.
``discover_defenses()`` imports them all.

References (实现参考.txt):
  - BIPIA : trusted/untrusted boundary, explicit reminder, instruction hierarchy.
  - spec baselines: NoDefense, ExplicitReminderDefense.

All defenses embed ``env.untrusted_text`` exactly once; the sandbox verifies
this before the judge is called (spec 防御约束).
"""
