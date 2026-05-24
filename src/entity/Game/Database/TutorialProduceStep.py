"""
Auto-generated from assets/gakumasu-diff and localization JSON.
Do not edit manually; regenerate via devtools/generate_game_database_schemas.py.
"""

from __future__ import annotations

from dataclasses import dataclass

@dataclass(slots=True)
class TutorialProduceStep:
    tutorialType: str = None
    idolCardId: str = None
    stepNumber: int = None
    tutorialStep: int = None
    stepType: str = None
    name: str = None
    produceStepRefresh: bool = None
    produceStepLessonId: str = None
    progressLevel: int = None
    produceNavigationNumber: int = None
    rankThreshold: int = None
    parameterBaseLine: int = None
    baseScore: int = None
    forceEndScore: int = None
    produceExamBattleNpcGroupId: str = None
    produceExamBattleConfigId: str = None
    produceExamGimmickEffectGroupId: str = None
    localization: TutorialProduceStepLocalization = None

@dataclass(slots=True)
class TutorialProduceStepLocalization:
    tutorialType: str = None
    stepNumber: int = None
    tutorialStep: int = None
    stepType: str = None
    name: str = None
