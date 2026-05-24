"""
Auto-generated from assets/gakumasu-diff and localization JSON.
Do not edit manually; regenerate via devtools/generate_game_database_schemas.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

@dataclass(slots=True)
class ProduceStory:
    id: str = None
    type: str = None
    title: str = None
    advAssetId: str = None
    produceEventHintProduceConditionDescriptions: List[str] = field(default_factory=list)
    viewConditionSetId: str = None
    unlockConditionSetId: str = None
    isBusinessExcellent: bool = None
    order: int = None
    localization: ProduceStoryLocalization = None

@dataclass(slots=True)
class ProduceStoryLocalization:
    id: str = None
    title: str = None
    produceEventHintProduceConditionDescriptions: List[str] = field(default_factory=list)
