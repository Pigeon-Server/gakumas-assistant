"""
Auto-generated from assets/gakumasu-diff and localization JSON.
Do not edit manually; regenerate via devtools/generate_game_database_schemas.py.
"""

from __future__ import annotations

from dataclasses import dataclass

@dataclass(slots=True)
class FeatureLock:
    tutorialType: str = None
    name: str = None
    description: str = None
    routeDescription: str = None
    unlockConditionSetId: str = None
    isForce: bool = None
    viewConditionSetId: str = None
    localization: FeatureLockLocalization = None

@dataclass(slots=True)
class FeatureLockLocalization:
    tutorialType: str = None
    name: str = None
    description: str = None
    routeDescription: str = None
