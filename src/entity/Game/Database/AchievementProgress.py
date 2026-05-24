"""
Auto-generated from assets/gakumasu-diff and localization JSON.
Do not edit manually; regenerate via devtools/generate_game_database_schemas.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

@dataclass(slots=True)
class AchievementProgressRewardsItem:
    resourceType: str = None
    resourceId: str = None
    quantity: int = None

@dataclass(slots=True)
class AchievementProgress:
    achievementId: str = None
    threshold: int = None
    assetId: str = None
    rewards: List[AchievementProgressRewardsItem] = field(default_factory=list)
    index: int = None
