"""No-op attack: emits an empty payload.

Used by the clean-utility gate (defense under no attack) and as the reference
baseline path. Not a real competitor attack.
"""

from .base import AttackRegistry, BaseAttack
from ..env import AttackEnv


@AttackRegistry.register
class NoOpAttack(BaseAttack):
    name = "NoOpAttack"
    description = "Empty payload (no attack). Used for reference and clean gate."

    def attack(self, env: AttackEnv) -> str:
        return ""
