"""
Auto-generated from assets/gakumasu-diff and localization JSON.
Do not edit manually; regenerate via devtools/generate_game_database_schemas.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

@dataclass(slots=True)
class StoryGroup:
    id: str = None
    storyType: str = None
    storyEventType: str = None
    isCampaign: bool = None
    eventStoryFilterType: str = None
    title: str = None
    headerAssetId: str = None
    storyThumbnailAssetId: str = None
    viewConditionSetId: str = None
    characterId: str = None
    dearnessLevelMin: int = None
    dearnessLevelMax: int = None
    storyIds: List[str] = field(default_factory=list)
    storyEventId: str = None
    order: int = None
    localization: StoryGroupLocalization = None

@dataclass(slots=True)
class StoryGroupLocalization:
    id: str = None
    title: str = None
