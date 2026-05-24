"""
Auto-generated from assets/gakumasu-diff and localization JSON.
Do not edit manually; regenerate via devtools/generate_game_database_schemas.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

@dataclass(slots=True)
class MissionPointRewardSetRewardsItem:
    resourceType: str = None
    resourceId: str = None
    quantity: int = None

@dataclass(slots=True)
class MissionPointRewardSet:
    missionPointId: str = None
    point: int = None
    isFeature: bool = None
    rewards: List[MissionPointRewardSetRewardsItem] = field(default_factory=list)
