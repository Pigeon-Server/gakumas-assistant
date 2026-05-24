"""
Auto-generated from assets/gakumasu-diff and localization JSON.
Do not edit manually; regenerate via devtools/generate_game_database_schemas.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

@dataclass(slots=True)
class Music:
    id: str = None
    title: str = None
    displayTitle: str = None
    lyrics: str = None
    composer: str = None
    arranger: str = None
    type: str = None
    jacketAssetId: str = None
    gameVersionAssetId: str = None
    idolCardSkinUnitId: str = None
    shortVersionStartMilliseconds: int = None
    shortVersionEndMilliseconds: int = None
    viewConditionSetId: str = None
    unlockConditionSetId: str = None
    produceLiveUnlockItemConditionSetId: str = None
    externalUrl: str = None
    unlockMusicIds: List[str] = field(default_factory=list)
    viewStartTime: str = None
    order: int = None
    localization: MusicLocalization = None

@dataclass(slots=True)
class MusicLocalization:
    id: str = None
    title: str = None
    displayTitle: str = None
    lyrics: str = None
    composer: str = None
    arranger: str = None
