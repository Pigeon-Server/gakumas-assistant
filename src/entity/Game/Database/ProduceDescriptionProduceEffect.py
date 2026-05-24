"""
Auto-generated from assets/gakumasu-diff and localization JSON.
Do not edit manually; regenerate via devtools/generate_game_database_schemas.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

@dataclass(slots=True)
class ProduceDescriptionProduceEffect:
    type: Any = None
    name: str = None
    produceDescriptionLabelId: str = None
    localization: ProduceDescriptionProduceEffectLocalization = None

@dataclass(slots=True)
class ProduceDescriptionProduceEffectLocalization:
    type: str = None
    name: str = None
