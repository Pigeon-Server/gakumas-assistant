"""
Auto-generated from assets/gakumasu-diff and localization JSON.
Do not edit manually; regenerate via devtools/generate_game_database_schemas.py.
"""

from __future__ import annotations

from dataclasses import dataclass

@dataclass(slots=True)
class TourStageTimeline:
    id: str = None
    stageNumber: int = None
    startTimelineAssetId: str = None
    examTimelineAssetId: str = None
    resultTimelineAssetId: str = None
    examBgmAssetId: str = None
    timelineBackgroundAssetId: str = None
    liveOverrideAssetId: str = None
