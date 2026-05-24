"""
Auto-generated from assets/gakumasu-diff and localization JSON.
Do not edit manually; regenerate via devtools/generate_game_database_schemas.py.
"""

from __future__ import annotations

from dataclasses import dataclass

@dataclass(slots=True)
class MissionPass:
    id: str = None
    name: str = None
    description: str = None
    assetId: str = None
    missionPassPointId: str = None
    premiumPassShopItemId: str = None
    viewConditionSetId: str = None
    unlockConditionSetId: str = None
    startTime: str = None
    endTime: str = None
    order: int = None
    localization: MissionPassLocalization = None

@dataclass(slots=True)
class MissionPassLocalization:
    id: str = None
    name: str = None
    description: str = None
