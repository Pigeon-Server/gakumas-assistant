"""
Auto-generated from assets/gakumasu-diff and localization JSON.
Do not edit manually; regenerate via devtools/generate_game_database_schemas.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

@dataclass(slots=True)
class ProduceCardPoolProduceCardRatiosItem:
    id: str = None
    upgradeCount: int = None
    ratio: int = None

@dataclass(slots=True)
class ProduceCardPool:
    id: str = None
    produceCardRatios: List[ProduceCardPoolProduceCardRatiosItem] = field(default_factory=list)
