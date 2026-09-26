"""Official baseline defense: no transformation (spec 防御基线 #1)."""

from .base import DefenseRegistry, BaseDefense
from ..env import DefenseEnv


@DefenseRegistry.register
class NoDefense(BaseDefense):
    name = "NoDefense"
    description = "Baseline: return the base prompt unchanged."

    def defend(self, env: DefenseEnv) -> str:
        return env.base_prompt
