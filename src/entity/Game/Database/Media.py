"""
Auto-generated from assets/gakumasu-diff and localization JSON.
Do not edit manually; regenerate via devtools/generate_game_database_schemas.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

@dataclass(slots=True)
class Media:
    id: str = None
    name: str = None
    mediaType: str = None
    assetId: str = None
    thumbnailAssetId: str = None
    viewConditionSetId: str = None
    characterIds: List[str] = field(default_factory=list)
    externalUrl: str = None
    fourPanelComicEpisode: int = None
    caption: str = None
    fourPanelComicSeries: str = None
    mediaMovieType: str = None
    startTime: str = None
    endTime: str = None
    order: int = None
    localization: MediaLocalization = None

@dataclass(slots=True)
class MediaLocalization:
    id: str = None
    name: str = None
