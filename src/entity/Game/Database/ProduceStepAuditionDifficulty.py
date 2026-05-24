"""
Auto-generated from assets/gakumasu-diff and localization JSON.
Do not edit manually; regenerate via devtools/generate_game_database_schemas.py.
"""

from __future__ import annotations

from dataclasses import dataclass

@dataclass(slots=True)
class ProduceStepAuditionDifficulty:
    id: str = None
    produceId: str = None
    stepType: str = None
    number: int = None
    rankThreshold: int = None
    parameterBaseLine: int = None
    baseScore: int = None
    forceEndScore: int = None
    produceExamBattleNpcGroupId: str = None
    produceExamBattleConfigId: str = None
    produceExamGimmickEffectGroupId: str = None
    auditionType: str = None
    isUnlockAnimation: bool = None
    voteCountBaseLine: int = None
    dearnessLevel: int = None
    voteCount: int = None
