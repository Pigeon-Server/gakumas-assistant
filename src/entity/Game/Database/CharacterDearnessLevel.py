"""
Auto-generated from assets/gakumasu-diff and localization JSON.
Do not edit manually; regenerate via devtools/generate_game_database_schemas.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

@dataclass(slots=True)
class CharacterDearnessLevelProduceSkillsItem:
    id: str = None
    level: int = None

@dataclass(slots=True)
class CharacterDearnessLevelRewardsItem:
    resourceType: str = None
    resourceId: str = None
    quantity: int = None

@dataclass(slots=True)
class CharacterDearnessLevel:
    characterId: str = None
    dearnessLevel: int = None
    advAssetId: str = None
    storyId: str = None
    produceConditionDescription: str = None
    produceConditionAchievementId: str = None
    produceConditionAchievementThreshold: int = None
    produceSkills: List[CharacterDearnessLevelProduceSkillsItem] = field(default_factory=list)
    rewards: List[CharacterDearnessLevelRewardsItem] = field(default_factory=list)
    ignoreReport: bool = None
    itemUnlockConditionSetId: str = None
    isStepThresholdLevel: bool = None
    isTargetLevel: bool = None
    targetDescription: str = None
    trueEndAchievementProduceType: str = None
    dearnessPointThreshold: int = None
    localization: CharacterDearnessLevelLocalization = None

@dataclass(slots=True)
class CharacterDearnessLevelLocalization:
    characterId: str = None
    dearnessLevel: int = None
    produceConditionDescription: str = None
