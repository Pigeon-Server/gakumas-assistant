"""
Auto-generated from assets/gakumasu-diff and localization JSON.
Do not edit manually; regenerate via devtools/generate_game_database_schemas.py.
"""

from __future__ import annotations

from dataclasses import dataclass

@dataclass(slots=True)
class CharacterTrueEndBonus:
    id: str = None
    produceType: str = None
    produceVocal: int = None
    produceDance: int = None
    produceVisual: int = None
    produceVocalGrowthRatePermil: int = None
    produceDanceGrowthRatePermil: int = None
    produceVisualGrowthRatePermil: int = None
    produceStamina: int = None
