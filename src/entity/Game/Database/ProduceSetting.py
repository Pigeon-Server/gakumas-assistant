"""
Auto-generated from assets/gakumasu-diff and localization JSON.
Do not edit manually; regenerate via devtools/generate_game_database_schemas.py.
"""

from __future__ import annotations

from dataclasses import dataclass

@dataclass(slots=True)
class ProduceSetting:
    id: str = None
    initialProducePoint: int = None
    produceDrinkPossessLimit: int = None
    refreshStaminaRecoveryPermil: int = None
    customizeProduceCardCount: int = None
    stepSkipStaminaRecoveryPermil: int = None
    beforeAuditionRefreshStaminaRecoveryPermil: int = None
    stepCustomizeStartAlertProducePointThreshold: int = None
    examStartAlertStaminaThreshold: int = None
    continueCount: int = None
    produceAuditionTrendAssessmentPermilUpper: int = None
    produceAuditionTrendAssessmentPermilLower: int = None
    maxLegendProduceCardCount: int = None
