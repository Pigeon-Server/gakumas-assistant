"""
Auto-generated from assets/gakumasu-diff and localization JSON.
Do not edit manually; regenerate via devtools/generate_game_database_schemas.py.
"""

from __future__ import annotations

from dataclasses import dataclass

@dataclass(slots=True)
class StoryReward:
    resourceType: str = None
    resourceId: str = None
    quantity: int = None

@dataclass(slots=True)
class Story:
    id: str = None
    type: str = None
    characterId: str = None
    title: str = None
    thumbnailAssetId: str = None
    advAssetId: str = None
    viewConditionSetId: str = None
    unlockConditionSetId: str = None
    reward: StoryReward = None
    previousStoryId: str = None
    order: int = None
    localization: StoryLocalization = None

@dataclass(slots=True)
class StoryLocalization:
    id: str = None
    title: str = None
