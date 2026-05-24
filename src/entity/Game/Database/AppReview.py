"""
Auto-generated from assets/gakumasu-diff and localization JSON.
Do not edit manually; regenerate via devtools/generate_game_database_schemas.py.
"""

from __future__ import annotations

from dataclasses import dataclass

@dataclass(slots=True)
class AppReview:
    type: str = None
    conditionSetId: str = None
    gashaId: str = None
    mainTaskGroupId: str = None
    mainTaskNumber: int = None
    achievementId: str = None
    achievementProgressThreshold: int = None
    produceId: str = None
