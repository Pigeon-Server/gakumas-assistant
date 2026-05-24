"""
Auto-generated from assets/gakumasu-diff and localization JSON.
Do not edit manually; regenerate via devtools/generate_game_database_schemas.py.
"""

from __future__ import annotations

from dataclasses import dataclass

@dataclass(slots=True)
class ProduceDescriptionProduceStep:
    type: str = None
    name: str = None
    produceDescriptionLabelId: str = None
    localization: ProduceDescriptionProduceStepLocalization = None

@dataclass(slots=True)
class ProduceDescriptionProduceStepLocalization:
    type: str = None
    name: str = None
