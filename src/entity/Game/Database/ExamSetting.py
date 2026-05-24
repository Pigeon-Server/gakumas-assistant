"""
Auto-generated from assets/gakumasu-diff and localization JSON.
Do not edit manually; regenerate via devtools/generate_game_database_schemas.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

@dataclass(slots=True)
class ExamSetting:
    id: str = None
    examStaminaConsumptionDownPermil: int = None
    examStaminaConsumptionAddPermil: int = None
    examBlockAddDownPermil: int = None
    examStaminaConsumptionAddDownPermil: int = None
    examStaminaReduceChange: int = None
    examStaminaConsumptionDownAddPermil: int = None
    examConcentrationLessonValueMultiplePermil: int = None
    fullPowerPlayableValueAdd: int = None
    examFullPowerLessonValueMultiplePermil: int = None
    holdLimit: int = None
    handLimit: int = None
    turnStartDistribute: int = None
    examGimmickParameterDebuffPermil: int = None
    examParameterBuffPermil: int = None
    examTurnEndRecoveryStamina: int = None
    produceExamPanicStaminaCandidates: List[int] = field(default_factory=list)
    examParameterBuffMultiplePerTurnPermil: int = None
    preservationReleasePlayableValueAdd1: int = None
    preservationReleasePlayableValueAdd2: int = None
    preservationReleaseBlockAdd1: int = None
    preservationReleaseBlockAdd2: int = None
    preservationReleaseEnthusiastic1: int = None
    preservationReleaseEnthusiastic2: int = None
    examConcentrationLessonValueMultiplePermil1: int = None
    examConcentrationLessonValueMultiplePermil2: int = None
    examPreservationLessonValueMultiplePermil1: int = None
    examPreservationLessonValueMultiplePermil2: int = None
    examConcentrationStaminaMultiplePermil1: int = None
    examConcentrationStaminaMultiplePermil2: int = None
    examPreservationStaminaMultiplePermil1: int = None
    examPreservationStaminaMultiplePermil2: int = None
    examConcentrationStaminaPenetrateReduce1: int = None
    examConcentrationStaminaPenetrateReduce2: int = None
    examAutoPlayEnableVersion: int = None
    examAutoPlaySearchCommandLimit: int = None
    overPreservationReleasePlayableValueAdd: int = None
    overPreservationReleaseBlockAdd: int = None
    overPreservationReleaseEnthusiastic: int = None
    examOverPreservationLessonValueMultiplePermil: int = None
    examOverPreservationStaminaMultiplePermil: int = None
    overPreservationReleaseToFullPowerGrowEffectLessonAdd: int = None
    examAutoPlaySearchCommandPlanLimits: List[int] = field(default_factory=list)
    examLessonValueMultipleDependReviewOrAggressiveMultiplePermil: int = None
    examLessonValueMultipleDependReviewOrAggressiveMaxPermil: int = None
    fixMoveCardShuffleDeckEnable: bool = None
