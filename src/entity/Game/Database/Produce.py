"""
Auto-generated from assets/gakumasu-diff and localization JSON.
Do not edit manually; regenerate via devtools/generate_game_database_schemas.py.
"""

from __future__ import annotations

from dataclasses import dataclass

@dataclass(slots=True)
class Produce:
    id: str = None
    name: str = None
    baseStepLevel: int = None
    maxRefreshCount: int = None
    produceSelectScreenOrderType: str = None
    challengeViewConditionSetId: str = None
    viewConditionSetId: str = None
    unlockConditionSetId: str = None
    examSettingId: str = None
    produceSettingId: str = None
    idolCardParameterGrowthLimit: int = None
    maxProduceEventCharacterGrowthNumber: int = None
    steps: int = None
    actionPointQuantity: int = None
    assetId: str = None
    produceNavigationNormalId: str = None
    produceNavigationAuditionId: str = None
    produceNavigationLoseId: str = None
    gradientColor1: str = None
    gradientColor2: str = None
    order: int = None
    localization: ProduceLocalization = None

@dataclass(slots=True)
class ProduceLocalization:
    id: str = None
    name: str = None
