"""
Auto-generated from assets/gakumasu-diff and localization JSON.
Do not edit manually; regenerate via devtools/generate_game_database_schemas.py.
"""

from __future__ import annotations

from dataclasses import dataclass

@dataclass(slots=True)
class Tower:
    id: str = None
    characterId: str = None
    title: str = None
    viewConditionSetId: str = None
    unlockConditionSetId: str = None
    achievementId: str = None
    order: int = None
    localization: TowerLocalization = None

@dataclass(slots=True)
class TowerLocalization:
    id: str = None
    title: str = None
