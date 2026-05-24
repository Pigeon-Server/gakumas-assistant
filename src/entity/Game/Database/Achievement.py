"""
Auto-generated from assets/gakumasu-diff and localization JSON.
Do not edit manually; regenerate via devtools/generate_game_database_schemas.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

@dataclass(slots=True)
class Achievement:
    id: str = None
    category: str = None
    name: str = None
    description: str = None
    missionType: str = None
    targetIds1: List[str] = field(default_factory=list)
    targetIds2: List[str] = field(default_factory=list)
    targetIds3: List[str] = field(default_factory=list)
    targetValue: int = None
    viewConditionSetId: str = None
    unlockConditionSetId: str = None
    masterAchievementInitialRank: int = None
    isTrueEndAchievement: bool = None
    isMasterAchievement: bool = None
    characterId: str = None
    viewProduceResult: bool = None
    order: int = None
    localization: AchievementLocalization = None

@dataclass(slots=True)
class AchievementLocalization:
    id: str = None
    name: str = None
    description: str = None
