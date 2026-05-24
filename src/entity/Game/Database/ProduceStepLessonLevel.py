"""
Auto-generated from assets/gakumasu-diff and localization JSON.
Do not edit manually; regenerate via devtools/generate_game_database_schemas.py.
"""

from __future__ import annotations

from dataclasses import dataclass

@dataclass(slots=True)
class ProduceStepLessonLevel:
    id: str = None
    progressLevel: int = None
    limitTurn: int = None
    successThreshold: int = None
    resultTargetValueLimit: int = None
