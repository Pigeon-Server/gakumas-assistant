"""
Auto-generated from assets/gakumasu-diff and localization JSON.
Do not edit manually; regenerate via devtools/generate_game_database_schemas.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

@dataclass(slots=True)
class TutorialProduce:
    tutorialType: str = None
    idolCardId: str = None
    produceCardIds: List[str] = field(default_factory=list)
    examSettingId: str = None
    produceSettingId: str = None
    idolCardParameterGrowthLimit: int = None
    produceNavigationNormalId: str = None
    produceNavigationAuditionId: str = None
    musicId: str = None
    environmentAssetId: str = None
    timelineAssetId: str = None
    memoryGiftId: str = None
