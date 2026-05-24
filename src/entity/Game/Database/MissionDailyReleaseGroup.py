"""
Auto-generated from assets/gakumasu-diff and localization JSON.
Do not edit manually; regenerate via devtools/generate_game_database_schemas.py.
"""

from __future__ import annotations

from dataclasses import dataclass

@dataclass(slots=True)
class MissionDailyReleaseGroup:
    id: str = None
    name: str = None
    logoAssetId: str = None
    bannerAssetId: str = None
    gradientColor1: str = None
    gradientColor2: str = None
    gradientColor3: str = None
    missionPointId: str = None
    conditionSetId: str = None
    fromStartTimeUnlock: bool = None
    startTime: str = None
    endTime: str = None
