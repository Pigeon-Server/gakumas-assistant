"""
Auto-generated from assets/gakumasu-diff and localization JSON.
Do not edit manually; regenerate via devtools/generate_game_database_schemas.py.
"""

from __future__ import annotations

from dataclasses import dataclass

@dataclass(slots=True)
class ProduceDescriptionProduceCardGrowEffect:
    type: str = None
    name: str = None
    noIcon: bool = None
    noReference: bool = None
    produceDescriptionLabelId: str = None
    produceCardCustomizeDescription: str = None
    localization: ProduceDescriptionProduceCardGrowEffectLocalization = None

@dataclass(slots=True)
class ProduceDescriptionProduceCardGrowEffectLocalization:
    type: str = None
    name: str = None
