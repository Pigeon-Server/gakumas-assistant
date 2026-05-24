"""
Auto-generated from assets/gakumasu-diff and localization JSON.
Do not edit manually; regenerate via devtools/generate_game_database_schemas.py.
"""

from __future__ import annotations

from dataclasses import dataclass

@dataclass(slots=True)
class Tips:
    id: str = None
    type: str = None
    title: str = None
    description: str = None
    characterId: str = None
    mediaId: str = None
    viewAreaType: str = None
    viewConditionSetId: str = None
    startTime: str = None
    endTime: str = None
    localization: TipsLocalization = None

@dataclass(slots=True)
class TipsLocalization:
    id: str = None
    title: str = None
    description: str = None
