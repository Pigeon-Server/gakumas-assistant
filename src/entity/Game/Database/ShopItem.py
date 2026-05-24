"""
Auto-generated from assets/gakumasu-diff and localization JSON.
Do not edit manually; regenerate via devtools/generate_game_database_schemas.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

@dataclass(slots=True)
class ShopItemRewardsItem:
    resourceType: str = None
    resourceId: str = None
    quantity: int = None

@dataclass(slots=True)
class ShopItem:
    id: str = None
    shopId: str = None
    name: str = None
    labelTypes: List[str] = field(default_factory=list)
    assetId: str = None
    shopProductId: str = None
    totalJewelQuantity: int = None
    paidOnlyJewelQuantity: int = None
    rewards: List[ShopItemRewardsItem] = field(default_factory=list)
    viewConditionSetId: str = None
    unlockConditionSetId: str = None
    resetTimingType: str = None
    resetHour: int = None
    resetMinute: int = None
    resetWeekday: str = None
    resetDay: int = None
    purchaseLimit: int = None
    consumptionResourceType: str = None
    consumptionResourceId: str = None
    consumptionResourceQuantity: int = None
    startTime: str = None
    endTime: str = None
    order: int = None
    localization: ShopItemLocalization = None

@dataclass(slots=True)
class ShopItemLocalization:
    id: str = None
    name: str = None
