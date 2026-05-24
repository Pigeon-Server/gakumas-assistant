"""
Auto-generated from assets/gakumasu-diff and localization JSON.
Do not edit manually; regenerate via devtools/generate_game_database_schemas.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

@dataclass(slots=True)
class MainTaskRewardsItem:
    resourceType: str = None
    resourceId: str = None
    quantity: int = None

@dataclass(slots=True)
class MainTaskAdditionalRewardsItem:
    resourceType: str = None
    resourceId: str = None
    quantity: int = None

@dataclass(slots=True)
class MainTask:
    mainTaskGroupId: str = None
    number: int = None
    title: str = None
    description: str = None
    homeDescription: str = None
    missionType: str = None
    targetIds1: List[str] = field(default_factory=list)
    targetIds2: List[str] = field(default_factory=list)
    targetIds3: List[str] = field(default_factory=list)
    targetValue: int = None
    missionId: str = None
    threshold: int = None
    viewConditionSetId: str = None
    unlockConditionSetId: str = None
    rewards: List[MainTaskRewardsItem] = field(default_factory=list)
    additionalRewards: List[MainTaskAdditionalRewardsItem] = field(default_factory=list)
    unlockFeatureTutorialType: str = None
    localization: MainTaskLocalization = None

@dataclass(slots=True)
class MainTaskLocalization:
    mainTaskGroupId: str = None
    number: int = None
    title: str = None
    description: str = None
    homeDescription: str = None
