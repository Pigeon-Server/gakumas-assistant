"""
Auto-generated from assets/gakumasu-diff and localization JSON.
Do not edit manually; regenerate via devtools/generate_game_database_schemas.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

@dataclass(slots=True)
class SeminarExamTransitionRewardsItem:
    resourceType: str = None
    resourceId: str = None
    quantity: int = None

@dataclass(slots=True)
class SeminarExamTransition:
    examEffectType: str = None
    isLessonInt: int = None
    description: str = None
    seminarExamGroupId: str = None
    seminarExamId: str = None
    seminarExamGroupName: str = None
    seminarExamName: str = None
    produceIds: List[str] = field(default_factory=list)
    rewards: List[SeminarExamTransitionRewardsItem] = field(default_factory=list)
    localization: SeminarExamTransitionLocalization = None

@dataclass(slots=True)
class SeminarExamTransitionLocalization:
    examEffectType: str = None
    isLessonInt: int = None
    seminarExamId: str = None
    description: str = None
    seminarExamGroupName: str = None
    seminarExamName: str = None
