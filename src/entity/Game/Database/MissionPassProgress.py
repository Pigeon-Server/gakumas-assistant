"""
Auto-generated from assets/gakumasu-diff and localization JSON.
Do not edit manually; regenerate via devtools/generate_game_database_schemas.py.
"""

from __future__ import annotations

from dataclasses import dataclass

@dataclass(slots=True)
class MissionPassProgressNormalReward:
    resourceType: str = None
    resourceId: str = None
    quantity: int = None

@dataclass(slots=True)
class MissionPassProgressPremiumReward:
    resourceType: str = None
    resourceId: str = None
    quantity: int = None

@dataclass(slots=True)
class MissionPassProgress:
    missionPassId: str = None
    threshold: int = None
    normalReward: MissionPassProgressNormalReward = None
    premiumReward: MissionPassProgressPremiumReward = None
    feature: bool = None
    repeat: bool = None
    repeatPoint: int = None
