"""
Auto-generated from assets/gakumasu-diff and localization JSON.
Do not edit manually; regenerate via devtools/generate_game_database_schemas.py.
"""

from __future__ import annotations

from dataclasses import dataclass

@dataclass(slots=True)
class TowerTotalClearRankRewardReward:
    resourceType: str = None
    resourceId: str = None
    quantity: int = None

@dataclass(slots=True)
class TowerTotalClearRankReward:
    rank: int = None
    reward: TowerTotalClearRankRewardReward = None
    isFeature: bool = None
