"""
Auto-generated from assets/gakumasu-diff and localization JSON.
Do not edit manually; regenerate via devtools/generate_game_database_schemas.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

@dataclass(slots=True)
class ProduceGroup:
    id: str = None
    name: str = None
    type: str = None
    produceIds: List[str] = field(default_factory=list)
    assetId: str = None
    viewConditionSetId: str = None
    unlockConditionSetId: str = None
    failedProduceMemoryAssetId: str = None
    description: str = None
    isForceLiveCommon: bool = None
    disableForceLiveCommonEndingLiveType: str = None
    limitGrade: str = None
    order: int = None
    localization: ProduceGroupLocalization = None

@dataclass(slots=True)
class ProduceGroupLocalization:
    id: str = None
    name: str = None
    description: str = None
