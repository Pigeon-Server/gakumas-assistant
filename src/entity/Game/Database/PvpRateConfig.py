"""
Auto-generated from assets/gakumasu-diff and localization JSON.
Do not edit manually; regenerate via devtools/generate_game_database_schemas.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

@dataclass(slots=True)
class PvpRateConfigStagesItem:
    stageType: str = None
    planType: str = None
    turn: int = None
    produceItemId: str = None
    produceItemIds: List[str] = field(default_factory=list)
    produceExamGimmickEffectGroupId: str = None
    bgmAssetId: str = None
    startTimelineAssetId: str = None
    examTimelineAssetId: str = None

@dataclass(slots=True)
class PvpRateConfig:
    id: str = None
    description: str = None
    vocal: int = None
    dance: int = None
    visual: int = None
    examSettingId: str = None
    produceExamBattleScoreConfigId: str = None
    examBattleFirstRankBonusPermil: int = None
    pvpRateCommonProduceCardId: str = None
    winTimelineAssetId: str = None
    loseTimelineAssetId: str = None
    startTimelineInitialTimePermil: int = None
    topAssetId: str = None
    stages: List[PvpRateConfigStagesItem] = field(default_factory=list)
    localization: PvpRateConfigLocalization = None

@dataclass(slots=True)
class PvpRateConfigLocalization:
    id: str = None
    description: str = None
