"""
Auto-generated from assets/gakumasu-diff and localization JSON.
Do not edit manually; regenerate via devtools/generate_game_database_schemas.py.
"""

from __future__ import annotations

from dataclasses import dataclass

@dataclass(slots=True)
class GashaButton:
    id: str = None
    name: str = None
    description: str = None
    type: str = None
    rewardCount: int = None
    fixRewardCount: int = None
    resourceType: str = None
    resourceId: str = None
    limitType: str = None
    limitCount: int = None
    resourceQuantity: int = None
    maxDrawCount: int = None
    discountLimitType: str = None
    discountLimitCount: int = None
    discountResourceQuantity: int = None
    viewConditionSetId: str = None
    unlockConditionSetId: str = None
    order: int = None
    appealType: str = None
    appealText: str = None
    highAppealType: str = None
    highAppealText: str = None
    bottomAppealType: str = None
    bottomAppealText: str = None
    isOverride: bool = None
    localization: GashaButtonLocalization = None

@dataclass(slots=True)
class GashaButtonLocalization:
    id: str = None
    order: int = None
    name: str = None
    description: str = None
