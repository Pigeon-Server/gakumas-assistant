"""
Auto-generated from assets/gakumasu-diff and localization JSON.
Do not edit manually; regenerate via devtools/generate_game_database_schemas.py.
"""

from __future__ import annotations

from dataclasses import dataclass

@dataclass(slots=True)
class WorkLevelReward:
    type: str = None
    durationMinutes: int = None
    level: int = None
    dearnessLevel: int = None
    rewardQuantity: int = None
    moneyRewardQuantity: int = None
    fanRewardQuantity: int = None
