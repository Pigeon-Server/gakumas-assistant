"""
Auto-generated from assets/gakumasu-diff and localization JSON.
Do not edit manually; regenerate via devtools/generate_game_database_schemas.py.
"""

from __future__ import annotations

from dataclasses import dataclass

@dataclass(slots=True)
class ProduceStepLesson:
    id: str = None
    name: str = None
    produceStepLessonLevelId: str = None
    localization: ProduceStepLessonLocalization = None

@dataclass(slots=True)
class ProduceStepLessonLocalization:
    id: str = None
    name: str = None
