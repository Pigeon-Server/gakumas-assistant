"""
Auto-generated from assets/gakumasu-diff and localization JSON.
Do not edit manually; regenerate via devtools/generate_game_database_schemas.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

@dataclass(slots=True)
class ProduceDescriptionExamEffect:
    type: str = None
    name: str = None
    produceDescriptionSwapId: str = None
    produceDescriptionLabelId: str = None
    examProduceDescriptionLabelId: str = None
    mainBuffMinThresholds: List[int] = field(default_factory=list)
    noIcon: bool = None
    noReference: bool = None
    localization: ProduceDescriptionExamEffectLocalization = None

@dataclass(slots=True)
class ProduceDescriptionExamEffectLocalization:
    type: str = None
    name: str = None
