"""
Auto-generated from assets/gakumasu-diff and localization JSON.
Do not edit manually; regenerate via devtools/generate_game_database_schemas.py.
"""

from __future__ import annotations

from dataclasses import dataclass

@dataclass(slots=True)
class ConditionSet:
    id: str = None
    number: int = None
    conditionOperatorType: str = None
    conditionType: str = None
    resourceId1: str = None
    resourceId2: str = None
    minMaxType: str = None
    min: str = None
    max: str = None
    beforeTime: str = None
    afterTime: str = None
    description: str = None
