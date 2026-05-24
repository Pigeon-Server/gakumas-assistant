"""
Auto-generated from assets/gakumasu-diff and localization JSON.
Do not edit manually; regenerate via devtools/generate_game_database_schemas.py.
"""

from __future__ import annotations

from dataclasses import dataclass

@dataclass(slots=True)
class MainTaskGroup:
    id: str = None
    title: str = None
    mainTaskType: str = None
    viewConditionSetId: str = None
    backgroundAssetId: str = None
    order: int = None
    localization: MainTaskGroupLocalization = None

@dataclass(slots=True)
class MainTaskGroupLocalization:
    id: str = None
    title: str = None
