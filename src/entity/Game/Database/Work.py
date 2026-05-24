"""
Auto-generated from assets/gakumasu-diff and localization JSON.
Do not edit manually; regenerate via devtools/generate_game_database_schemas.py.
"""

from __future__ import annotations

from dataclasses import dataclass

@dataclass(slots=True)
class Work:
    type: str = None
    name: str = None
    unlockConditionSetId: str = None
    rewardResourceType: str = None
    rewardResourceId: str = None
    localization: WorkLocalization = None

@dataclass(slots=True)
class WorkLocalization:
    type: str = None
    name: str = None
