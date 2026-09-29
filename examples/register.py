"""Register external competitor submission files into the plugin registries.

Usage:
    python examples/register.py
    python -m rankguard match --attacks my_team_attack --defenses my_team_defense
"""

import sys
from pathlib import Path

# Make the repo root importable when run directly as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rankguard.attacks.base import AttackRegistry, discover_attacks
from rankguard.defenses.base import DefenseRegistry, discover_defenses

HERE = Path(__file__).resolve().parent


def register_all() -> None:
    discover_attacks()   # populate built-ins first
    discover_defenses()
    AttackRegistry.load_from_file(HERE / "attack.py")
    DefenseRegistry.load_from_file(HERE / "defense.py")


if __name__ == "__main__":
    register_all()
    print("attacks :", AttackRegistry.names())
    print("defenses:", DefenseRegistry.names())
