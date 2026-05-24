"""
Auto-generated from assets/gakumasu-diff and localization JSON.
Do not edit manually; regenerate via devtools/generate_game_database_schemas.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

@dataclass(slots=True)
class Mission:
    id: str = None
    missionGroupId: str = None
    name: str = None
    category: str = None
    type: str = None
    targetIds1: List[str] = field(default_factory=list)
    targetIds2: List[str] = field(default_factory=list)
    targetIds3: List[str] = field(default_factory=list)
    targetValue: int = None
    isLessThanTargetValue: bool = None
    isEventMission: bool = None
    missionDailyReleaseGroupId: str = None
    missionDailyReleaseDay: int = None
    viewConditionSetId: str = None
    unlockConditionSetId: str = None
    order: int = None
    localization: MissionLocalization = None

@dataclass(slots=True)
class MissionLocalization:
    id: str = None
    name: str = None
