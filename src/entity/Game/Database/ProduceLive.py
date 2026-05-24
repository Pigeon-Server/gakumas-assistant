"""
Auto-generated from assets/gakumasu-diff and localization JSON.
Do not edit manually; regenerate via devtools/generate_game_database_schemas.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

@dataclass(slots=True)
class ProduceLive:
    musicId: str = None
    type: str = None
    forceUnlockConditionSetId: str = None
    unlockConditionSetId: str = None
    thumbnailAssetId: str = None
    environmentAssetId: str = None
    timelineAssetId: str = None
    beforeAdvAssetId: str = None
    afterAdvAssetId: str = None
    liveMusicAssetId: str = None
    motionAssetIds: List[str] = field(default_factory=list)
    unitLiveThumbnailAssetCharacterIds: List[str] = field(default_factory=list)
    unitLiveThumbnailAssetIds: List[str] = field(default_factory=list)
    liveOverrideAssetId: str = None
    additionalActorAssetIds: List[str] = field(default_factory=list)
