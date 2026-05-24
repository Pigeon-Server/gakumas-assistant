"""
Auto-generated from assets/gakumasu-diff and localization JSON.
Do not edit manually; regenerate via devtools/generate_game_database_schemas.py.
"""

from __future__ import annotations

from dataclasses import dataclass

@dataclass(slots=True)
class StoryEvent:
    id: str = None
    title: str = None
    storyEventType: str = None
    titleAssetId: str = None
    bannerAssetId: str = None
    storyGroupId: str = None
    idolCardSkinBeforeAssetId: str = None
    idolCardSkinAfterAssetId: str = None
    order: int = None
    localization: StoryEventLocalization = None

@dataclass(slots=True)
class StoryEventLocalization:
    id: str = None
    title: str = None
