"""
Auto-generated from assets/gakumasu-diff and localization JSON.
Do not edit manually; regenerate via devtools/generate_game_database_schemas.py.
"""

from __future__ import annotations

from dataclasses import dataclass

@dataclass(slots=True)
class ProduceItemEffect:
    id: str = None
    effectType: str = None
    effectTurn: int = None
    effectCount: int = None
    produceEffectId: str = None
    produceExamStatusEnchantId: str = None
