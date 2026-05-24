"""
Auto-generated from assets/gakumasu-diff and localization JSON.
Do not edit manually; regenerate via devtools/generate_game_database_schemas.py.
"""

from __future__ import annotations

from dataclasses import dataclass

@dataclass(slots=True)
class ProduceEventCharacterGrowth:
    characterId: str = None
    number: int = None
    title: str = None
    description: str = None
    vocal: int = None
    dance: int = None
    visual: int = None
    produceStepEventDetailId: str = None
    localization: ProduceEventCharacterGrowthLocalization = None

@dataclass(slots=True)
class ProduceEventCharacterGrowthLocalization:
    characterId: str = None
    number: int = None
    title: str = None
    description: str = None
